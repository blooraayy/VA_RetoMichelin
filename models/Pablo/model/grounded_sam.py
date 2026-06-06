# =============================================================================
# model/grounded_sam.py
# Two-stage Grounded-SAM pipeline.
#
# Stage 0 — Detect QR markers (for calibration).
# Stage 1 — Detect rubber strips with DINO+SAM (large, easy).
# Stage 2 — For each strip ROI, detect the thin cut line with DINO+SAM,
#           trying several prompts. Sample N points along the cut contour.
# =============================================================================

from __future__ import annotations

import numpy as np
import torch
import cv2
from dataclasses import dataclass, field
from typing import Optional

from config.config import (
    SAM_CHECKPOINT, SAM_MODEL_TYPE,
    PROMPT_QR, PROMPT_STRIP, PROMPT_CUT_CANDIDATES,
    GDINO_BOX_THRESHOLD,     GDINO_TEXT_THRESHOLD,
    GDINO_BOX_THRESHOLD_CUT, GDINO_TEXT_THRESHOLD_CUT,
    GDINO_BOX_THRESHOLD_CUT_FALLBACK, GDINO_TEXT_THRESHOLD_CUT_FALLBACK,
    NUM_SAMPLE_POINTS, DEVICE,
    QR_MAX_AREA_FRAC, QR_MIN_AREA_FRAC,
    QR_ASPECT_RATIO_MIN, QR_ASPECT_RATIO_MAX,
    STRIP_MAX_AREA_FRAC, STRIP_MIN_AREA_FRAC,
    CUT_MIN_AREA_FRAC_OF_STRIP, CUT_MAX_AREA_FRAC_OF_STRIP,
    CUT_MIN_ASPECT_RATIO, CUT_GAP_BRIGHTNESS_MIN,
    NMS_IOU_THRESHOLD,
    STRIP_MAX_DETECTIONS,
)
from utils.utils import bgr_to_rgb


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """One detected object (QR code, rubber strip, or cut edge)."""
    label:    str
    score:    float
    box_xyxy: np.ndarray        # [x1, y1, x2, y2] in pixels (full-image coords)
    mask:     Optional[np.ndarray] = None  # full-image (H, W) bool mask


@dataclass
class PipelineResult:
    """Full output of the Grounded-SAM pipeline for one image."""
    image_path:          str
    image_shape:         tuple
    qr_detections:       list[DetectionResult]
    strip_detections:    list[DetectionResult]    # NEW (stage 1)
    edge_detections:     list[DetectionResult]    # cut lines (stage 2)
    edge_sample_points:  list[np.ndarray]         # pixel coords (N, 2) per edge
    edge_sample_points_mm:   list[np.ndarray] = field(default_factory=list)
    distances_to_bottom_mm:  list[np.ndarray] = field(default_factory=list)


# ── Grounded-SAM pipeline ────────────────────────────────────────────────────

class GroundedSAMModel:
    """Grounding DINO (HuggingFace) + SAM, two-stage pipeline."""

    GDINO_MODEL_ID = "IDEA-Research/grounding-dino-base"

    def __init__(
        self,
        device:         str = DEVICE,
        sam_checkpoint: str = SAM_CHECKPOINT,
        sam_model_type: str = SAM_MODEL_TYPE,
        gdino_config:   str = "",   # legacy
        gdino_weights:  str = "",   # legacy
    ):
        self.device         = device
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = sam_model_type

        self._gdino_processor = None
        self._gdino_model     = None
        self._sam_predictor   = None

    # ── Lazy loading ─────────────────────────────────────────────────────────

    def _load_models(self) -> None:
        if self._gdino_model is not None:
            return
        print("[GroundedSAMModel] Loading Grounding DINO from HuggingFace...")
        self._load_grounding_dino_hf()
        print("[GroundedSAMModel] Loading SAM...")
        self._load_sam()
        print("[GroundedSAMModel] Models ready.")

    def _load_grounding_dino_hf(self) -> None:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self._gdino_processor = AutoProcessor.from_pretrained(self.GDINO_MODEL_ID)
        self._gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.GDINO_MODEL_ID
        ).to(self.device)
        self._gdino_model.eval()

    def _load_sam(self) -> None:
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry[self.sam_model_type](
            checkpoint=self.sam_checkpoint
        ).to(self.device)
        self._sam_predictor = SamPredictor(sam)

    # ── Public API ───────────────────────────────────────────────────────────

    def run(self, image_bgr: np.ndarray, image_path: str = "") -> PipelineResult:
        """Run the full two-stage Grounded-SAM pipeline on one image."""
        self._load_models()

        image_rgb = bgr_to_rgb(image_bgr)
        H, W = image_bgr.shape[:2]
        img_area = float(H * W)

        self._sam_predictor.set_image(image_rgb)

        # ── Stage 0 : QR codes ────────────────────────────────────────────
        qr_boxes_raw, qr_scores_raw = self._detect_hf(
            image_rgb, PROMPT_QR,
            box_thr=GDINO_BOX_THRESHOLD, text_thr=GDINO_TEXT_THRESHOLD,
        )
        qr_boxes, qr_scores = self._filter_and_nms(
            qr_boxes_raw, qr_scores_raw, img_area,
            min_area_frac=QR_MIN_AREA_FRAC,
            max_area_frac=QR_MAX_AREA_FRAC,
            aspect_min=QR_ASPECT_RATIO_MIN,
            aspect_max=QR_ASPECT_RATIO_MAX,
        )
        qr_detections = self._segment_boxes(qr_boxes, qr_scores, "qr_code", H, W)
        qr_detections = qr_detections[:3]   # at most 3 calibration markers
        print(f"[Stage 0] QR  raw={len(qr_boxes_raw)}  kept={len(qr_detections)}")

        # ── Stage 1 : rubber strips ──────────────────────────────────────
        strip_boxes_raw, strip_scores_raw = self._detect_hf(
            image_rgb, PROMPT_STRIP,
            box_thr=GDINO_BOX_THRESHOLD, text_thr=GDINO_TEXT_THRESHOLD,
        )
        strip_boxes, strip_scores = self._filter_and_nms(
            strip_boxes_raw, strip_scores_raw, img_area,
            min_area_frac=STRIP_MIN_AREA_FRAC,
            max_area_frac=STRIP_MAX_AREA_FRAC,
            aspect_min=None,
            aspect_max=None,
        )
        strip_detections = self._segment_boxes(strip_boxes, strip_scores, "strip", H, W)
        strip_detections = strip_detections[:STRIP_MAX_DETECTIONS]
        print(f"[Stage 1] STRIP raw={len(strip_boxes_raw)}  kept={len(strip_detections)}")

        # Intensity fallback: when DINO finds no strips use row/col projection to
        # locate dark rubber blobs directly.  Covers horizontal-strip images where
        # DINO confidence is low.
        if len(strip_detections) == 0:
            strip_detections = self._intensity_strip_fallback(image_bgr, img_area, H, W)
            if strip_detections:
                print(f"[Stage 1] Intensity fallback → {len(strip_detections)} strip(s)")

        # Decide whether Stage 2 should be run on a 90°-CW rotated ROI.
        # Strips placed side-by-side (centres separated horizontally, e.g. M35,
        # M44) systematically fail; rotating the ROI puts them in the same
        # configuration as the well-performing M31/M32/M33.
        rotate_stage2 = self._should_rotate_for_stage2(strip_detections)
        if rotate_stage2:
            print("[Stage 2] Strips arranged side-by-side → rotating ROIs 90° CW for cut detection")

        # ── Stage 2 : cut edge inside each strip ─────────────────────────
        edge_detections = []
        edge_sample_points = []

        # First pass — normal DINO threshold on all strips.
        strip_had_cut = [False] * len(strip_detections)
        for strip_idx, strip_det in enumerate(strip_detections):
            cuts = self._detect_cut_in_strip(
                image_rgb, strip_det, H, W, strip_idx,
                rotate_roi=rotate_stage2,
            )
            if cuts:
                strip_had_cut[strip_idx] = True
            for cut in cuts:
                edge_detections.append(cut)
                edge_sample_points.append(self._extract_cut_gap_borders(cut.mask, n_points=NUM_SAMPLE_POINTS))

        # Second pass — lower DINO threshold only on strips that yielded nothing,
        # and only when at least one other strip already found a cut.
        # This recovers second cuts without generating FP in no-cut images.
        if any(strip_had_cut):
            for strip_idx, strip_det in enumerate(strip_detections):
                if strip_had_cut[strip_idx]:
                    continue
                cuts = self._detect_cut_in_strip(
                    image_rgb, strip_det, H, W, strip_idx,
                    box_thr=GDINO_BOX_THRESHOLD_CUT_FALLBACK,
                    text_thr=GDINO_TEXT_THRESHOLD_CUT_FALLBACK,
                    rotate_roi=rotate_stage2,
                )
                for cut in cuts:
                    edge_detections.append(cut)
                    edge_sample_points.append(self._extract_cut_gap_borders(cut.mask, n_points=NUM_SAMPLE_POINTS))

        # Third pass — classical gradient fallback for any strip that still has no cut.
        # Previously gated on DINO finding at least one anchor cut; removed because
        # new competition images have barely-visible cuts that DINO misses entirely.
        if len(strip_detections) > 0 and len(edge_detections) < min(len(strip_detections), 2):
            classical_dets, classical_pts = self._conditioned_classical_fallback(
                image_bgr=image_bgr,
                strip_detections=strip_detections,
                H=H,
                W=W,
                max_new=2 - len(edge_detections),
                existing_edges=edge_detections,
            )
            edge_detections.extend(classical_dets)
            edge_sample_points.extend(classical_pts)

        # Fourth pass — intensity-valley fallback for pressed-rubber cuts where
        # the separation is a dark crevice rather than a bright white gap.
        # Runs only when strips are still missing a cut after the three DINO/gradient passes.
        if len(strip_detections) > 0 and len(edge_detections) < min(len(strip_detections), 2):
            valley_dets, valley_pts = self._intensity_valley_fallback(
                image_bgr=image_bgr,
                strip_detections=strip_detections,
                H=H,
                W=W,
                max_new=2 - len(edge_detections),
                existing_edges=edge_detections,
            )
            edge_detections.extend(valley_dets)
            edge_sample_points.extend(valley_pts)

        if len(edge_detections) > 1:
            edge_detections, edge_sample_points = self._deduplicate_edges(
                edge_detections, edge_sample_points
            )

        print(f"[Stage 2] CUT detected={len(edge_detections)}")

        return PipelineResult(
            image_path=image_path,
            image_shape=(H, W, 3),
            qr_detections=qr_detections,
            strip_detections=strip_detections,
            edge_detections=edge_detections,
            edge_sample_points=edge_sample_points,
        )

    # ── Stage 2 helper ───────────────────────────────────────────────────────

    def _detect_cut_in_strip(
        self,
        image_rgb:  np.ndarray,
        strip_det:  DetectionResult,
        H:          int,
        W:          int,
        strip_idx:  int,
        box_thr:    float = GDINO_BOX_THRESHOLD_CUT,
        text_thr:   float = GDINO_TEXT_THRESHOLD_CUT,
        rotate_roi: bool  = False,
    ) -> list[DetectionResult]:
        """Try several prompts on the strip's ROI; keep best valid cut.

        If ``rotate_roi`` is True, the ROI is rotated 90° CW before being fed
        to DINO; the resulting boxes are rotated back to ROI coordinates
        before being mapped to the full image. SAM is still called with
        boxes in original full-image coords (it operates on the un-rotated
        image already loaded in the predictor).
        """
        x1, y1, x2, y2 = strip_det.box_xyxy.astype(int)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W, x2); y2 = min(H, y2)

        if x2 <= x1 + 5 or y2 <= y1 + 5:
            return []

        roi_rgb = image_rgb[y1:y2, x1:x2]
        roi_h, roi_w = roi_rgb.shape[:2]
        roi_area = float(roi_h * roi_w)

        if rotate_roi:
            # 90° CW rotation: (roi_h, roi_w) → (roi_w, roi_h)
            dino_input = cv2.rotate(roi_rgb, cv2.ROTATE_90_CLOCKWISE)
        else:
            dino_input = roi_rgb

        din_h, din_w = dino_input.shape[:2]

        # The strip's longer side (in DINO's view) dictates which orientation
        # a cut should have.
        strip_is_horizontal = din_w >= din_h

        for prompt in PROMPT_CUT_CANDIDATES:
            boxes, scores = self._detect_hf(
                dino_input, prompt,
                box_thr=box_thr,
                text_thr=text_thr,
            )
            if len(boxes) == 0:
                continue

            # Keep elongated boxes whose orientation is perpendicular to the strip.
            keep_idx = []
            for i, box in enumerate(boxes):
                bx1, by1, bx2, by2 = box
                bw = max(bx2 - bx1, 1.0)
                bh = max(by2 - by1, 1.0)
                area_frac = (bw * bh) / roi_area
                aspect = max(bw, bh) / min(bw, bh)

                if not (CUT_MIN_AREA_FRAC_OF_STRIP <= area_frac <= CUT_MAX_AREA_FRAC_OF_STRIP):
                    continue
                if aspect < CUT_MIN_ASPECT_RATIO:
                    continue
                # Cut must be perpendicular to the strip's long axis:
                #  - horizontal strip → cut is vertical (bh > bw)
                #  - vertical   strip → cut is horizontal (bw > bh)
                cut_is_vertical = bh > bw
                if strip_is_horizontal and not cut_is_vertical:
                    continue
                if (not strip_is_horizontal) and cut_is_vertical:
                    continue
                keep_idx.append(i)

            if not keep_idx:
                continue

            boxes  = boxes[keep_idx]
            scores = scores[keep_idx]

            # NMS on the survivors; keep only the top-scoring detection per strip.
            # One rubber strip can have at most one cut, so a single detection is correct.
            from torchvision.ops import nms
            t_b = torch.from_numpy(boxes).float()
            t_s = torch.from_numpy(scores).float()
            keep = nms(t_b, t_s, iou_threshold=NMS_IOU_THRESHOLD).cpu().numpy()
            keep = keep[:1]
            boxes  = boxes[keep]
            scores = scores[keep]

            # If we rotated the ROI, map boxes back to original ROI coords.
            if rotate_roi:
                roi_boxes = np.stack([
                    self._box_back_from_cw(b, roi_h_orig=roi_h) for b in boxes
                ])
            else:
                roi_boxes = boxes

            # Map ROI boxes back to full-image coords
            full_boxes = roi_boxes.copy()
            full_boxes[:, [0, 2]] += x1
            full_boxes[:, [1, 3]] += y1

            print(f"  [Stage 2 / strip {strip_idx}] prompt='{prompt}' "
                  f"→ {len(full_boxes)} cut candidate(s) kept"
                  f"{' (rotated ROI)' if rotate_roi else ''}")

            results = self._segment_boxes(full_boxes, scores, "cut_edge", H, W)

            # Brightness sanity check: a real cut gap shows the white table at
            # its core. Detections whose central slice is dark are rubber, not
            # gaps — discard them. Catches FP introduced by rotated ROIs in
            # cut-less images (M48, M49, M51).
            kept = []
            for det in results:
                p90 = self._gap_centre_brightness(image_rgb, det.box_xyxy)
                if p90 < CUT_GAP_BRIGHTNESS_MIN:
                    print(f"  [Stage 2 / strip {strip_idx}] dropped FP: "
                          f"gap centre p90={p90:.1f} < {CUT_GAP_BRIGHTNESS_MIN}")
                    continue
                kept.append(det)

            if not kept:
                continue
            return kept

        print(f"  [Stage 2 / strip {strip_idx}] no cut detected with any prompt"
              f"{' (rotated ROI)' if rotate_roi else ''}")
        return []

    # ── Detection / segmentation helpers ─────────────────────────────────────

    def _detect_hf(
        self,
        image_rgb: np.ndarray,
        prompt:    str,
        box_thr:   float,
        text_thr:  float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run Grounding DINO on a (sub-)image. Returns (boxes_xyxy, scores)."""
        from PIL import Image

        pil_image = Image.fromarray(image_rgb)
        H, W = image_rgb.shape[:2]

        inputs = self._gdino_processor(
            images=pil_image, text=prompt, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._gdino_model(**inputs)

        # transformers >= 4.55: param renamed `box_threshold` -> `threshold`
        # and `text_threshold` removed. Try new API first, fall back to old.
        try:
            results = self._gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_thr,
                target_sizes=[(H, W)],
            )[0]
        except TypeError:
            results = self._gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_thr,
                text_threshold=text_thr,
                target_sizes=[(H, W)],
            )[0]

        boxes  = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        return boxes, scores

    @staticmethod
    def _filter_and_nms(
        boxes:          np.ndarray,
        scores:         np.ndarray,
        img_area:       float,
        min_area_frac:  float,
        max_area_frac:  float,
        aspect_min:     Optional[float],
        aspect_max:     Optional[float],
        nms_iou:        float = NMS_IOU_THRESHOLD,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Discard out-of-range boxes, then apply NMS."""
        if len(boxes) == 0:
            return boxes, scores

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        w = np.maximum(x2 - x1, 1e-6)
        h = np.maximum(y2 - y1, 1e-6)
        area_frac = (w * h) / img_area
        ratio = h / w

        keep = (area_frac >= min_area_frac) & (area_frac <= max_area_frac)
        if aspect_min is not None and aspect_max is not None:
            keep &= (ratio >= aspect_min) & (ratio <= aspect_max)

        boxes  = boxes[keep]
        scores = scores[keep]

        if len(boxes) == 0:
            return boxes, scores

        from torchvision.ops import nms
        b = torch.from_numpy(boxes).float()
        s = torch.from_numpy(scores).float()
        idx = nms(b, s, iou_threshold=nms_iou).cpu().numpy()
        return boxes[idx], scores[idx]

    def _segment_boxes(
        self,
        boxes:  np.ndarray,
        scores: np.ndarray,
        label:  str,
        H:      int,
        W:      int,
    ) -> list[DetectionResult]:
        """Run SAM on each box (assumed in full-image coords) and return results."""
        results = []
        if boxes is None or len(boxes) == 0:
            return results

        input_boxes = torch.tensor(boxes, device=self.device)
        transformed = self._sam_predictor.transform.apply_boxes_torch(
            input_boxes, (H, W)
        )

        masks_batch, _, _ = self._sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed,
            multimask_output=False,
        )
        # masks_batch: (N, 1, H, W)

        for i in range(len(boxes)):
            mask = masks_batch[i, 0].cpu().numpy()
            results.append(DetectionResult(
                label=label,
                score=float(scores[i]),
                box_xyxy=boxes[i],
                mask=mask,
            ))
        return results


    # ── Stage-2 rotation helpers ─────────────────────────────────────────────

    @staticmethod
    def _should_rotate_for_stage2(
        strip_detections: list[DetectionResult],
        spread_ratio_threshold: float = 1.2,
        single_strip_aspect_threshold: float = 1.2,
    ) -> bool:
        """Decide if Stage 2 should rotate ROIs 90° CW.

        With ≥2 strips: compare the spread of strip centres along X vs Y.
        If centres are spread out more horizontally than vertically the
        strips lie side-by-side (e.g. M35, M44) and a 90° CW rotation puts
        them in the well-performing M31/M32/M33 configuration.

        With 1 strip: fall back to the aspect ratio of its SAM mask
        (or bounding box if mask missing) — a tall mask means a vertical
        strip that should be rotated.
        """
        if len(strip_detections) >= 2:
            cxs = []
            cys = []
            for s in strip_detections:
                box = s.box_xyxy
                cxs.append(0.5 * (float(box[0]) + float(box[2])))
                cys.append(0.5 * (float(box[1]) + float(box[3])))
            spread_x = max(cxs) - min(cxs)
            spread_y = max(cys) - min(cys)
            return spread_x > spread_y * spread_ratio_threshold

        if len(strip_detections) == 1:
            s = strip_detections[0]
            extent_x = extent_y = 0.0
            if s.mask is not None and s.mask.any():
                ys, xs = np.where(s.mask)
                extent_x = float(xs.max() - xs.min())
                extent_y = float(ys.max() - ys.min())
            else:
                box = s.box_xyxy
                extent_x = float(box[2] - box[0])
                extent_y = float(box[3] - box[1])
            return extent_y > extent_x * single_strip_aspect_threshold

        return False

    @staticmethod
    def _gap_centre_brightness(
        image_rgb: np.ndarray,
        box: np.ndarray,
        core_frac: float = 0.4,
    ) -> float:
        """Return the 90th-percentile grayscale intensity of the bbox core.

        A real cut gap shows the white table at its centre; a false positive
        inside continuous rubber is uniformly dark. We sample a thin slice
        perpendicular to the gap's long axis to avoid mixing in the rubber
        edges that flank the gap.
        """
        H, W = image_rgb.shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W, x2); y2 = min(H, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0

        bw, bh = x2 - x1, y2 - y1
        if bw >= bh:  # horizontal gap → thin horizontal central band
            cy = (y1 + y2) // 2
            half = max(1, int(round(bh * core_frac / 2)))
            sl = image_rgb[max(y1, cy - half): min(y2, cy + half), x1:x2]
        else:         # vertical gap → thin vertical central band
            cx = (x1 + x2) // 2
            half = max(1, int(round(bw * core_frac / 2)))
            sl = image_rgb[y1:y2, max(x1, cx - half): min(x2, cx + half)]

        if sl.size == 0:
            return 0.0
        gray = cv2.cvtColor(sl, cv2.COLOR_RGB2GRAY) if sl.ndim == 3 else sl
        return float(np.percentile(gray, 90))

    @staticmethod
    def _box_back_from_cw(box: np.ndarray, roi_h_orig: int) -> np.ndarray:
        """Convert a bbox from a 90°-CW-rotated ROI back to the original ROI.

        Forward map (per pixel): (x, y) → (roi_h_orig - 1 - y, x).
        Inverse: (rx, ry) → (ry, roi_h_orig - 1 - rx).
        """
        rx1, ry1, rx2, ry2 = [float(v) for v in box]
        # Two opposite corners of the rotated bbox map back to two opposite
        # corners of the original-ROI bbox; take the axis-aligned envelope.
        xs = (ry1, ry2)
        ys = ((roi_h_orig - 1) - rx1, (roi_h_orig - 1) - rx2)
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    # ── Cut-gap post-processing / conditioned fallback ───────────────────────

    @staticmethod
    def _clip_box(box: np.ndarray | list[float], W: int, H: int) -> np.ndarray:
        x1, y1, x2, y2 = [float(v) for v in box]
        x1 = max(0.0, min(float(W - 1), x1))
        x2 = max(0.0, min(float(W - 1), x2))
        y1 = max(0.0, min(float(H - 1), y1))
        y2 = max(0.0, min(float(H - 1), y2))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    @staticmethod
    def _box_intersection_area(a: np.ndarray, b: np.ndarray) -> float:
        x1 = max(float(a[0]), float(b[0]))
        y1 = max(float(a[1]), float(b[1]))
        x2 = min(float(a[2]), float(b[2]))
        y2 = min(float(a[3]), float(b[3]))
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @staticmethod
    def _is_valid_cutgap_box(
        box: np.ndarray,
        W: int,
        H: int,
        strip_box: Optional[np.ndarray] = None,
    ) -> bool:
        """Final geometry filter for cut-gap boxes.

        It removes false positives near the image border and shapes that are not
        plausible long, thin gaps. If a strip box is provided, the cut must lie
        mostly inside that strip.
        """
        x1, y1, x2, y2 = [float(v) for v in box]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        long_side = max(bw, bh)
        short_side = min(bw, bh)
        aspect = long_side / short_side

        if aspect < 3.0:
            return False
        if long_side < 0.05 * max(W, H):
            return False
        if long_side > 0.70 * max(W, H):
            return False
        if short_side > 0.13 * min(W, H):
            return False

        # Avoid detections stuck to the image frame, common in no-cut images.
        margin_x = 0.025 * W
        margin_y = 0.025 * H
        touches_img_border = (
            x1 <= margin_x or x2 >= W - margin_x or
            y1 <= margin_y or y2 >= H - margin_y
        )
        if touches_img_border:
            return False

        if strip_box is not None:
            inter = GroundedSAMModel._box_intersection_area(box, strip_box)
            area = bw * bh
            if area <= 0 or inter / area < 0.75:
                return False

        return True

    @staticmethod
    def _make_rect_mask(box: np.ndarray, H: int, W: int) -> np.ndarray:
        mask = np.zeros((H, W), dtype=np.uint8)
        x1, y1, x2, y2 = GroundedSAMModel._clip_box(box, W, H).astype(int)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)
        return mask.astype(bool)

    @staticmethod
    def _associated_strip_box(
        edge_box: np.ndarray,
        strip_detections: list[DetectionResult],
    ) -> Optional[np.ndarray]:
        if not strip_detections:
            return None
        best = None
        best_inter = 0.0
        for strip in strip_detections:
            sbox = strip.box_xyxy.astype(np.float32)
            inter = GroundedSAMModel._box_intersection_area(edge_box, sbox)
            if inter > best_inter:
                best_inter = inter
                best = sbox
        return best

    @staticmethod
    def _edge_strength_profile(
        gray: np.ndarray,
        strip_box: np.ndarray,
        orientation: str,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        H, W = gray.shape[:2]
        sx1, sy1, sx2, sy2 = GroundedSAMModel._clip_box(strip_box, W, H).astype(int)
        roi = gray[sy1:sy2, sx1:sx2]
        if roi.size == 0:
            return None, None

        roi = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi)
        roi = cv2.GaussianBlur(roi, (5, 5), 0)
        grad_x = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)

        if orientation == "vertical":
            profile = np.mean(np.abs(grad_x), axis=0)
            coords = np.arange(sx1, sx2, dtype=np.float32)
        else:
            profile = np.mean(np.abs(grad_y), axis=1)
            coords = np.arange(sy1, sy2, dtype=np.float32)

        if len(profile) >= 9:
            profile = cv2.GaussianBlur(profile.reshape(1, -1), (1, 9), 0).ravel()
        return coords, profile

    @staticmethod
    def _complete_second_border(
        image_bgr: np.ndarray,
        edge_box: np.ndarray,
        strip_box: Optional[np.ndarray],
        H: int,
        W: int,
    ) -> np.ndarray:
        """If a detection covers only one side, expand it towards a parallel edge.

        The method searches a nearby strong parallel gradient inside the same
        strip and merges both borders into a single gap box.
        """
        if strip_box is None:
            return edge_box

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = [float(v) for v in edge_box]
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        orientation = "vertical" if bh >= bw else "horizontal"
        coords, profile = GroundedSAMModel._edge_strength_profile(gray, strip_box, orientation)
        if coords is None or profile is None or len(coords) < 10:
            return edge_box

        base = 0.5 * (x1 + x2) if orientation == "vertical" else 0.5 * (y1 + y2)
        # Search nearby; gaps in the dataset are generally thin but visible.
        search_min, search_max = 6.0, 55.0
        cand_mask = (
            ((coords >= base + search_min) & (coords <= base + search_max)) |
            ((coords <= base - search_min) & (coords >= base - search_max))
        )
        idxs = np.where(cand_mask)[0]
        if len(idxs) == 0:
            return edge_box

        local = profile[idxs]
        if local.max() < max(5.0, float(profile.mean() + 0.7 * profile.std())):
            return edge_box
        second = float(coords[idxs[int(np.argmax(local))]])

        if orientation == "vertical":
            half = max(2.0, bw * 0.35)
            second_box = np.array([second - half, y1, second + half, y2], dtype=np.float32)
        else:
            half = max(2.0, bh * 0.35)
            second_box = np.array([x1, second - half, x2, second + half], dtype=np.float32)

        merged = np.array([
            min(edge_box[0], second_box[0]),
            min(edge_box[1], second_box[1]),
            max(edge_box[2], second_box[2]),
            max(edge_box[3], second_box[3]),
        ], dtype=np.float32)
        return GroundedSAMModel._clip_box(merged, W, H)

    def _conditioned_classical_fallback(
        self,
        image_bgr: np.ndarray,
        strip_detections: list[DetectionResult],
        H: int,
        W: int,
        max_new: int = 2,
        existing_edges: Optional[list[DetectionResult]] = None,
    ) -> tuple[list[DetectionResult], list[np.ndarray]]:
        """Classical fallback restricted to strip boxes only.

        It is used only to recover missing cuts, never as a free full-image
        detector. This targets M35/M36/M37-like false negatives while reducing
        false positives in no-cut images.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        new_dets: list[DetectionResult] = []
        new_pts: list[np.ndarray] = []
        existing_edges = existing_edges or []

        for strip in strip_detections:
            if len(new_dets) >= max_new:
                break
            sbox = strip.box_xyxy.astype(np.float32)

            # Skip strips already containing a detected cut.
            already = False
            for edge in existing_edges + new_dets:
                inter = self._box_intersection_area(edge.box_xyxy, sbox)
                area = max(1.0, (edge.box_xyxy[2] - edge.box_xyxy[0]) * (edge.box_xyxy[3] - edge.box_xyxy[1]))
                if inter / area > 0.55:
                    already = True
                    break
            if already:
                continue

            sx1, sy1, sx2, sy2 = self._clip_box(sbox, W, H).astype(int)
            roi = gray[sy1:sy2, sx1:sx2]
            if roi.size == 0:
                continue
            sw, sh = max(1, sx2 - sx1), max(1, sy2 - sy1)

            strip_is_horizontal = sw >= sh
            cut_orientation = "vertical" if strip_is_horizontal else "horizontal"
            coords, profile = self._edge_strength_profile(gray, sbox, cut_orientation)
            if coords is None or profile is None or len(coords) < 20:
                continue

            valid = np.zeros_like(profile, dtype=bool)
            valid[int(0.08 * len(profile)): int(0.92 * len(profile))] = True
            local = profile.copy()
            local[~valid] = 0
            peak = float(local.max())
            if peak < max(5.0, float(profile.mean() + 1.0 * profile.std())):
                continue
            idx = int(np.argmax(local))
            coord = float(coords[idx])

            if cut_orientation == "vertical":
                box = np.array([
                    coord - 8,
                    sy1 + 0.06 * sh,
                    coord + 8,
                    sy2 - 0.06 * sh,
                ], dtype=np.float32)
            else:
                box = np.array([
                    sx1 + 0.06 * sw,
                    coord - 8,
                    sx2 - 0.06 * sw,
                    coord + 8,
                ], dtype=np.float32)

            box = self._clip_box(box, W, H)
            if not self._is_valid_cutgap_box(box, W, H, strip_box=sbox):
                continue
            box = self._complete_second_border(image_bgr, box, sbox, H, W)
            mask = self._make_rect_mask(box, H, W)
            det = DetectionResult("cut_edge_classical", 0.10, box, mask)
            new_dets.append(det)
            new_pts.append(self._extract_cut_gap_borders(mask))
            print(f"[Stage 2] Conditioned classical fallback recovered cut: {box.tolist()}")

        return new_dets, new_pts

    def _postprocess_edges(
        self,
        image_bgr: np.ndarray,
        edge_detections: list[DetectionResult],
        edge_sample_points: list[np.ndarray],
        strip_detections: list[DetectionResult],
        H: int,
        W: int,
    ) -> tuple[list[DetectionResult], list[np.ndarray]]:
        """Final filter + double-border completion."""
        filtered_edges: list[DetectionResult] = []
        filtered_points: list[np.ndarray] = []

        for det, _pts in zip(edge_detections, edge_sample_points):
            box = self._clip_box(det.box_xyxy, W, H)
            strip_box = self._associated_strip_box(box, strip_detections)

            if not self._is_valid_cutgap_box(box, W, H, strip_box=strip_box):
                print(f"[Stage 2] Removed false positive cut gap: {box.tolist()}")
                continue

            completed_box = self._complete_second_border(image_bgr, box, strip_box, H, W)
            mask = self._make_rect_mask(completed_box, H, W)

            new_det = DetectionResult(
                label=det.label,
                score=det.score,
                box_xyxy=completed_box,
                mask=mask,
            )
            filtered_edges.append(new_det)
            filtered_points.append(self._extract_cut_gap_borders(mask))

        return filtered_edges, filtered_points

    def _intensity_strip_fallback(
        self,
        image_bgr: np.ndarray,
        img_area:  float,
        H:         int,
        W:         int,
    ) -> list[DetectionResult]:
        """Find rubber strips via row/column intensity projection.

        Works when DINO fails to detect horizontal (or vertical) strips by
        finding connected dark-pixel bands along the image's long axis.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        dark = (gray < 80).astype(np.uint8)

        def _find_strips_from_projection(axis: int) -> list[DetectionResult]:
            projection = dark.sum(axis=axis)  # sum along rows (axis=1→cols) or cols (axis=0→rows)
            size_perp = dark.shape[1 - axis]  # width for row-proj, height for col-proj

            # A "strip line" has at least 8% of perpendicular size as dark pixels
            thresh = 0.08 * size_perp
            is_strip_line = projection >= thresh

            ranges: list[tuple[int, int]] = []
            in_strip = False
            for i, flag in enumerate(is_strip_line):
                if flag and not in_strip:
                    start = i
                    in_strip = True
                elif not flag and in_strip:
                    ranges.append((start, i - 1))
                    in_strip = False
            if in_strip:
                ranges.append((start, len(is_strip_line) - 1))

            # Filter: strip must span at least 5% of the full dimension
            min_span = 0.05 * len(is_strip_line)
            ranges = [(a, b) for a, b in ranges if (b - a) >= min_span]

            # Sort by span descending, keep top 2
            ranges.sort(key=lambda r: r[1] - r[0], reverse=True)
            ranges = ranges[:STRIP_MAX_DETECTIONS]

            results = []
            for a, b in ranges:
                if axis == 1:   # row projection → Y ranges
                    slab = dark[a:b + 1, :]
                    col_any = slab.sum(axis=0)
                    xs = np.where(col_any > 0)[0]
                    if len(xs) == 0:
                        continue
                    x1, x2, y1, y2 = int(xs.min()), int(xs.max()), a, b
                else:           # col projection → X ranges
                    slab = dark[:, a:b + 1]
                    row_any = slab.sum(axis=1)
                    ys = np.where(row_any > 0)[0]
                    if len(ys) == 0:
                        continue
                    y1, y2, x1, x2 = int(ys.min()), int(ys.max()), a, b

                area_frac = float((x2 - x1) * (y2 - y1)) / img_area
                if not (STRIP_MIN_AREA_FRAC <= area_frac <= STRIP_MAX_AREA_FRAC):
                    continue

                box = np.array([x1, y1, x2, y2], dtype=np.float32)
                mask = np.zeros((H, W), dtype=bool)
                mask[y1:y2 + 1, x1:x2 + 1] = dark[y1:y2 + 1, x1:x2 + 1].astype(bool)
                results.append(DetectionResult(
                    label="strip_intensity",
                    score=0.30,
                    box_xyxy=box,
                    mask=mask,
                ))
            return results

        # Try horizontal strips first (row projection), then vertical
        strips = _find_strips_from_projection(axis=1)
        if len(strips) < 2:
            strips = _find_strips_from_projection(axis=0)
        return strips

    def _intensity_valley_fallback(
        self,
        image_bgr: np.ndarray,
        strip_detections: list[DetectionResult],
        H: int,
        W: int,
        max_new: int = 2,
        existing_edges: Optional[list[DetectionResult]] = None,
    ) -> tuple[list[DetectionResult], list[np.ndarray]]:
        """Detect cuts as intensity valleys (dark crevice) in the strip ROI.

        Used for pressed-rubber images where the separation between pieces
        is a dark thin line rather than a bright white gap.  The method
        looks for a narrow column (or row) whose mean intensity is
        significantly lower than the local baseline computed by a moving
        average across the strip.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        new_dets: list[DetectionResult] = []
        new_pts:  list[np.ndarray]      = []
        existing_edges = existing_edges or []

        for strip in strip_detections:
            if len(new_dets) >= max_new:
                break
            sbox = strip.box_xyxy.astype(np.float32)

            # Skip strips that already contain a detected cut edge.
            already = False
            for edge in existing_edges + new_dets:
                inter = self._box_intersection_area(edge.box_xyxy, sbox)
                area  = max(1.0, (edge.box_xyxy[2] - edge.box_xyxy[0])
                                * (edge.box_xyxy[3] - edge.box_xyxy[1]))
                if inter / area > 0.55:
                    already = True
                    break
            if already:
                continue

            sx1, sy1, sx2, sy2 = self._clip_box(sbox, W, H).astype(int)
            roi = gray[sy1:sy2, sx1:sx2]
            if roi.size == 0:
                continue
            sw, sh = max(1, sx2 - sx1), max(1, sy2 - sy1)

            strip_is_horizontal = sw >= sh

            if strip_is_horizontal:
                # Vertical cut → column-wise mean intensity profile
                profile = roi.mean(axis=0).astype(np.float32)
                coords  = np.arange(sx1, sx2, dtype=np.float32)
            else:
                # Horizontal cut → row-wise mean intensity profile
                profile = roi.mean(axis=1).astype(np.float32)
                coords  = np.arange(sy1, sy2, dtype=np.float32)

            if len(profile) < 20:
                continue

            # Local baseline via moving average; window = ~1/5 of the profile length
            window = max(10, min(len(profile) // 5, 80))
            kernel   = np.ones(window, dtype=np.float32) / window
            baseline = np.convolve(profile, kernel, mode="same")
            darkness = baseline - profile    # positive where darker than local mean

            # Restrict search to the central 80 % to avoid rubber-edge artefacts.
            margin = max(1, int(0.10 * len(darkness)))
            valid_darkness = darkness.copy()
            valid_darkness[:margin]  = 0.0
            valid_darkness[-margin:] = 0.0

            peak_dark = float(valid_darkness.max())
            threshold = max(6.0, float(np.mean(darkness) + 1.5 * np.std(darkness)))
            if peak_dark < threshold:
                continue

            idx   = int(np.argmax(valid_darkness))
            coord = float(coords[idx])

            if strip_is_horizontal:
                box = np.array([
                    coord - 8.0,
                    sy1 + 0.06 * sh,
                    coord + 8.0,
                    sy2 - 0.06 * sh,
                ], dtype=np.float32)
            else:
                box = np.array([
                    sx1 + 0.06 * sw,
                    coord - 8.0,
                    sx2 - 0.06 * sw,
                    coord + 8.0,
                ], dtype=np.float32)

            box = self._clip_box(box, W, H)
            if not self._is_valid_cutgap_box(box, W, H, strip_box=sbox):
                continue

            mask = self._make_rect_mask(box, H, W)
            det  = DetectionResult("cut_edge_valley", 0.08, box, mask)
            new_dets.append(det)
            new_pts.append(self._extract_cut_gap_borders(mask))
            print(f"[Stage 2] Intensity-valley fallback: cut at coord={coord:.1f}  "
                  f"darkness_peak={peak_dark:.1f} (thr={threshold:.1f})")

        return new_dets, new_pts

    @staticmethod
    def _deduplicate_edges(
        edge_detections: list[DetectionResult],
        edge_sample_points: list[np.ndarray],
        mask_iou_threshold: float = 0.30,
    ) -> tuple[list[DetectionResult], list[np.ndarray]]:
        """Remove duplicate cut-edge detections whose masks overlap."""
        if len(edge_detections) <= 1:
            return edge_detections, edge_sample_points
        n = len(edge_detections)
        to_remove: set[int] = set()
        for i in range(n):
            for j in range(i + 1, n):
                if i in to_remove or j in to_remove:
                    continue
                mi = edge_detections[i].mask
                mj = edge_detections[j].mask
                if mi is None or mj is None:
                    continue
                inter = float((mi & mj).sum())
                union = float((mi | mj).sum())
                if union > 0 and inter / union >= mask_iou_threshold:
                    loser = j if edge_detections[i].score >= edge_detections[j].score else i
                    to_remove.add(loser)
        if not to_remove:
            return edge_detections, edge_sample_points
        keep = [k for k in range(n) if k not in to_remove]
        print(
            f"[Stage 2] Edge deduplication: removed {len(to_remove)} duplicate(s); "
            f"{len(keep)} edge(s) remain."
        )
        return [edge_detections[k] for k in keep], [edge_sample_points[k] for k in keep]

    # ── Edge sampling ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_cut_gap_borders(
        mask: np.ndarray,
        n_points: int = NUM_SAMPLE_POINTS,
    ) -> np.ndarray:
        """Extract 5 points from each side of the detected cut gap.

        The mask is treated as the separation between rubber bands:
        P1-P5 belong to one border, P6-P10 to the opposite border.
        """
        if mask is None or mask.sum() == 0:
            return np.empty((0, 2), dtype=np.float32)

        mask_u8 = mask.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if num_labels <= 1:
            return GroundedSAMModel._extract_edge_centerline(mask, n_points)

        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        clean = (labels == largest_label).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)
        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)

        ys, xs = np.where(clean > 0)
        if len(xs) == 0:
            return GroundedSAMModel._extract_edge_centerline(mask, n_points)

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bw = x_max - x_min + 1
        bh = y_max - y_min + 1
        horizontal_gap = bw >= bh
        n_side = max(1, n_points // 2)
        border_1, border_2 = [], []

        if horizontal_gap:
            xs_sample = np.linspace(x_min, x_max, n_side)
            for x in xs_sample:
                xi = int(round(x))
                col = clean[:, xi]
                ys_col = np.where(col > 0)[0]
                if len(ys_col) == 0:
                    ya, yb = float(y_min), float(y_max)
                else:
                    ya = float(np.percentile(ys_col, 5))
                    yb = float(np.percentile(ys_col, 95))
                border_1.append([float(x), ya])
                border_2.append([float(x), yb])
        else:
            ys_sample = np.linspace(y_min, y_max, n_side)
            for y in ys_sample:
                yi = int(round(y))
                row = clean[yi, :]
                xs_row = np.where(row > 0)[0]
                if len(xs_row) == 0:
                    xa, xb = float(x_min), float(x_max)
                else:
                    xa = float(np.percentile(xs_row, 5))
                    xb = float(np.percentile(xs_row, 95))
                border_1.append([xa, float(y)])
                border_2.append([xb, float(y)])

        pts = np.array(border_1 + border_2, dtype=np.float32)
        return pts if len(pts) else GroundedSAMModel._extract_edge_centerline(mask, n_points)

    @staticmethod
    def _extract_edge_centerline(
        mask:     np.ndarray,
        n_points: int = NUM_SAMPLE_POINTS,
    ) -> np.ndarray:
        """For a thin elongated mask, sample N points along its centre line.

        The centre line is found by:
          1. Computing the longest contour of the mask.
          2. Determining whether the mask is horizontal or vertical
             from its bounding box.
          3. Walking the long axis at evenly-spaced positions and taking
             the centre of the mask cross-section at each position.
        """
        if mask is None or mask.sum() == 0:
            return np.empty((0, 2), dtype=np.float32)

        ys, xs = np.where(mask)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bw = x_max - x_min + 1
        bh = y_max - y_min + 1

        is_horizontal = bw >= bh   # long axis = X
        pts = np.empty((n_points, 2), dtype=np.float32)

        if is_horizontal:
            xs_sample = np.linspace(x_min, x_max, n_points)
            for i, xi in enumerate(xs_sample):
                col = mask[:, int(round(xi))]
                ys_col = np.where(col)[0]
                if len(ys_col) == 0:
                    pts[i] = (xi, (y_min + y_max) / 2.0)
                else:
                    pts[i] = (xi, ys_col.mean())
        else:
            ys_sample = np.linspace(y_min, y_max, n_points)
            for i, yi in enumerate(ys_sample):
                row = mask[int(round(yi)), :]
                xs_row = np.where(row)[0]
                if len(xs_row) == 0:
                    pts[i] = ((x_min + x_max) / 2.0, yi)
                else:
                    pts[i] = (xs_row.mean(), yi)

        return pts

    # ── Backwards-compatibility shim (old name) ──────────────────────────────

    @staticmethod
    def _extract_edge_samples(mask: np.ndarray, n_points: int = NUM_SAMPLE_POINTS) -> np.ndarray:
        return GroundedSAMModel._extract_edge_centerline(mask, n_points)

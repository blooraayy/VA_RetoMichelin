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
    NUM_SAMPLE_POINTS, DEVICE,
    QR_MAX_AREA_FRAC, QR_MIN_AREA_FRAC,
    QR_ASPECT_RATIO_MIN, QR_ASPECT_RATIO_MAX,
    STRIP_MAX_AREA_FRAC, STRIP_MIN_AREA_FRAC,
    CUT_MIN_AREA_FRAC_OF_STRIP, CUT_MAX_AREA_FRAC_OF_STRIP,
    CUT_MIN_ASPECT_RATIO,
    NMS_IOU_THRESHOLD,
)
from utils.utils import bgr_to_rgb, sample_points_along_contour


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
        print(f"[Stage 1] STRIP raw={len(strip_boxes_raw)}  kept={len(strip_detections)}")
        if len(strip_detections) == 0:
            print("[Stage 1] No strip detected. Using full image as fallback ROI.")
            full_box = np.array([0, 0, W - 1, H - 1], dtype=np.float32)
            strip_detections = [
                DetectionResult(
                    label="strip_fallback_full_image",
                    score=1.0,
                    box_xyxy=full_box,
                    mask=np.ones((H, W), dtype=bool),
                )
            ]

        # ── Stage 2 : cut edge inside each strip ─────────────────────────
        edge_detections = []
        edge_sample_points = []

        for strip_idx, strip_det in enumerate(strip_detections):
            cuts = self._detect_cut_in_strip(image_rgb, strip_det, H, W, strip_idx)
            for cut in cuts:
                edge_detections.append(cut)
                pts = self._extract_edge_centerline(cut.mask)
                edge_sample_points.append(pts)

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
    ) -> list[DetectionResult]:
        """Try several prompts on the strip's ROI; keep best valid cut."""
        x1, y1, x2, y2 = strip_det.box_xyxy.astype(int)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W, x2); y2 = min(H, y2)

        if x2 <= x1 + 5 or y2 <= y1 + 5:
            return []

        roi_rgb = image_rgb[y1:y2, x1:x2]
        roi_h, roi_w = roi_rgb.shape[:2]
        roi_area = float(roi_h * roi_w)
        is_fallback_roi = strip_det.label.startswith("strip_fallback")

        # The strip's longer side dictates which orientation a cut should have.
        strip_is_horizontal = roi_w >= roi_h

        for prompt in PROMPT_CUT_CANDIDATES:
            boxes, scores = self._detect_hf(
                roi_rgb, prompt,
                box_thr=GDINO_BOX_THRESHOLD_CUT,
                text_thr=GDINO_TEXT_THRESHOLD_CUT,
            )
            if len(boxes) == 0:
                continue

            # Keep elongated boxes whose orientation is perpendicular to the strip.
            keep_idx = []
            for i, box in enumerate(boxes):
                bx1, by1, bx2, by2 = box
                bw = max(bx2 - bx1, 1.0)
                bh = max(by2 - by1, 1.0)

                margin_x = 0.05 * roi_w
                margin_y = 0.05 * roi_h

                # Avoid detections too close to the outer border of the strip ROI
                if bx1 < margin_x or bx2 > roi_w - margin_x or by1 < margin_y or by2 > roi_h - margin_y:
                    continue

                area_frac = (bw * bh) / roi_area
                aspect = max(bw, bh) / min(bw, bh)

                if not (CUT_MIN_AREA_FRAC_OF_STRIP <= area_frac <= CUT_MAX_AREA_FRAC_OF_STRIP):
                    continue
                if aspect < CUT_MIN_ASPECT_RATIO:
                    continue
                # Cut must be perpendicular to the strip's long axis:
                #  - horizontal strip → cut is vertical (bh > bw)
                #  - vertical   strip → cut is horizontal (bw > bh)
                if not is_fallback_roi:
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

            # NMS on the survivors
            from torchvision.ops import nms
            t_b = torch.from_numpy(boxes).float()
            t_s = torch.from_numpy(scores).float()
            keep = nms(t_b, t_s, iou_threshold=NMS_IOU_THRESHOLD).cpu().numpy()
            boxes  = boxes[keep]
            scores = scores[keep]

            # Keep only the best candidate for this strip
            if len(boxes) > 1:
                best_idx = int(np.argmax(scores))
                boxes = boxes[[best_idx]]
                scores = scores[[best_idx]]

            # Map ROI boxes back to full-image coords
            full_boxes = boxes.copy()
            full_boxes[:, [0, 2]] += x1
            full_boxes[:, [1, 3]] += y1

            print(f"  [Stage 2 / strip {strip_idx}] prompt='{prompt}' "
                  f"→ {len(full_boxes)} cut candidate(s) kept")

            print(f"    selected boxes full image: {full_boxes.tolist()}")
            print(f"    selected scores: {scores.tolist()}")
            
            return self._segment_boxes(full_boxes, scores, "cut_edge", H, W)

        print(f"  [Stage 2 / strip {strip_idx}] no cut detected with any prompt")
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

    # ── Edge sampling ────────────────────────────────────────────────────────

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

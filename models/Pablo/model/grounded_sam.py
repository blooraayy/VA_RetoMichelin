# =============================================================================
# model/grounded_sam.py
# Hybrid Grounded-SAM pipeline for Michelin Challenge 2.
#
# Stage 0 — Detect QR/pink markers for calibration.
# Stage 1 — Detect rubber strips with Grounding DINO + SAM.
# Stage 2 — Detect the cut gap between rubber bands with Grounding DINO + SAM.
# Fallback — If Stage 2 finds no cut gap, use a conservative classical detector.
#
# The detected cut is interpreted as the separation/gap between two rubber bands.
# Therefore, the sampling extracts 5 points from one border of the gap and
# 5 points from the opposite border.
# =============================================================================

from __future__ import annotations

import numpy as np
import torch
import cv2
from dataclasses import dataclass, field
from typing import Optional
from utils.classical_cutgap import is_valid_cutgap_box

from config.config import (
    SAM_CHECKPOINT, SAM_MODEL_TYPE,
    PROMPT_QR, PROMPT_STRIP, PROMPT_CUT_CANDIDATES,
    GDINO_BOX_THRESHOLD, GDINO_TEXT_THRESHOLD,
    GDINO_BOX_THRESHOLD_CUT, GDINO_TEXT_THRESHOLD_CUT,
    GDINO_BOX_THRESHOLD_CUT_FALLBACK, GDINO_TEXT_THRESHOLD_CUT_FALLBACK,
    GDINO_BOX_THRESHOLD_CUT_LAST_RESORT, GDINO_TEXT_THRESHOLD_CUT_LAST_RESORT,
    NUM_SAMPLE_POINTS, DEVICE,
    QR_MAX_AREA_FRAC, QR_MIN_AREA_FRAC,
    QR_ASPECT_RATIO_MIN, QR_ASPECT_RATIO_MAX,
    STRIP_MAX_AREA_FRAC, STRIP_MIN_AREA_FRAC,
    CUT_MIN_AREA_FRAC_OF_STRIP, CUT_MAX_AREA_FRAC_OF_STRIP,
    CUT_MIN_ASPECT_RATIO,
    NMS_IOU_THRESHOLD,
    STRIP_MAX_DETECTIONS,
    USE_CLASSICAL_FALLBACK,
    CLASSICAL_FALLBACK_MAX_DETECTIONS,
)
from utils.utils import bgr_to_rgb


@dataclass
class DetectionResult:
    """One detected object: QR marker, rubber strip, or cut gap."""
    label: str
    score: float
    box_xyxy: np.ndarray
    mask: Optional[np.ndarray] = None


@dataclass
class PipelineResult:
    """Full output of the Grounded-SAM pipeline for one image."""
    image_path: str
    image_shape: tuple
    qr_detections: list[DetectionResult]
    strip_detections: list[DetectionResult]
    edge_detections: list[DetectionResult]
    edge_sample_points: list[np.ndarray]
    edge_sample_points_mm: list[np.ndarray] = field(default_factory=list)
    distances_to_bottom_mm: list[np.ndarray] = field(default_factory=list)


class GroundedSAMModel:
    """Grounding DINO from HuggingFace + SAM + conservative classical fallback."""

    GDINO_MODEL_ID = "IDEA-Research/grounding-dino-base"

    def __init__(
        self,
        device: str = DEVICE,
        sam_checkpoint: str = SAM_CHECKPOINT,
        sam_model_type: str = SAM_MODEL_TYPE,
        gdino_config: str = "",
        gdino_weights: str = "",
    ):
        self.device = device
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = sam_model_type
        self._gdino_processor = None
        self._gdino_model = None
        self._sam_predictor = None

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
        """Run the full hybrid pipeline on one image."""
        self._load_models()

        image_rgb = bgr_to_rgb(image_bgr)
        H, W = image_bgr.shape[:2]
        img_area = float(H * W)

        self._sam_predictor.set_image(image_rgb)

        # ── Stage 0: QR markers ───────────────────────────────────────────
        qr_boxes_raw, qr_scores_raw = self._detect_hf(
            image_rgb,
            PROMPT_QR,
            box_thr=GDINO_BOX_THRESHOLD,
            text_thr=GDINO_TEXT_THRESHOLD,
        )
        qr_boxes, qr_scores = self._filter_and_nms(
            qr_boxes_raw,
            qr_scores_raw,
            img_area,
            min_area_frac=QR_MIN_AREA_FRAC,
            max_area_frac=QR_MAX_AREA_FRAC,
            aspect_min=QR_ASPECT_RATIO_MIN,
            aspect_max=QR_ASPECT_RATIO_MAX,
        )
        qr_detections = self._segment_boxes(qr_boxes, qr_scores, "qr_code", H, W)
        qr_detections = self._filter_qr_near_corners(qr_detections, H, W)
        qr_detections = self._score_qr_by_color(image_bgr, qr_detections)
        print(f"[Stage 0] QR raw={len(qr_boxes_raw)} kept={len(qr_detections)}")

        # ── Stage 1: rubber strips ────────────────────────────────────────
        strip_boxes_raw, strip_scores_raw = self._detect_hf(
            image_rgb,
            PROMPT_STRIP,
            box_thr=GDINO_BOX_THRESHOLD,
            text_thr=GDINO_TEXT_THRESHOLD,
        )
        strip_boxes, strip_scores = self._filter_and_nms(
            strip_boxes_raw,
            strip_scores_raw,
            img_area,
            min_area_frac=STRIP_MIN_AREA_FRAC,
            max_area_frac=STRIP_MAX_AREA_FRAC,
            aspect_min=None,
            aspect_max=None,
        )
        strip_detections = self._segment_boxes(strip_boxes, strip_scores, "strip", H, W)
        strip_detections = self._suppress_contained_strips(strip_detections)

        print(
            f"[Stage 1] STRIP raw={len(strip_boxes_raw)} "
            f"kept={len(strip_detections)}"
        )

        if len(strip_detections) > STRIP_MAX_DETECTIONS:
            strip_detections = sorted(
                strip_detections, key=lambda d: d.score, reverse=True
            )[:STRIP_MAX_DETECTIONS]
            print(f"[Stage 1] Capped to top {STRIP_MAX_DETECTIONS} strips by score.")

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

        # ── Stage 2: cut gap inside each strip ────────────────────────────
        edge_detections: list[DetectionResult] = []
        edge_sample_points: list[np.ndarray] = []
        strip_had_cut = [False] * len(strip_detections)

        # First pass: main threshold.
        for strip_idx, strip_det in enumerate(strip_detections):
            cuts = self._detect_cut_in_strip(
                image_rgb=image_rgb,
                strip_det=strip_det,
                H=H,
                W=W,
                strip_idx=strip_idx,
            )
            if cuts:
                strip_had_cut[strip_idx] = True
            for cut in cuts:
                edge_detections.append(cut)
                edge_sample_points.append(
                    self._extract_cut_gap_borders(cut.mask, n_points=NUM_SAMPLE_POINTS)
                )

        # Second pass: fallback threshold only on strips that yielded nothing,
        # and only if at least one other strip already found a cut. This recovers
        # second cuts without increasing false positives globally.
        if any(strip_had_cut):
            for strip_idx, strip_det in enumerate(strip_detections):
                if strip_had_cut[strip_idx]:
                    continue
                print(f"  [Stage 2 / strip {strip_idx}] retrying with fallback threshold")
                cuts = self._detect_cut_in_strip(
                    image_rgb=image_rgb,
                    strip_det=strip_det,
                    H=H,
                    W=W,
                    strip_idx=strip_idx,
                    box_thr=GDINO_BOX_THRESHOLD_CUT_FALLBACK,
                    text_thr=GDINO_TEXT_THRESHOLD_CUT_FALLBACK,
                )
                for cut in cuts:
                    edge_detections.append(cut)
                    edge_sample_points.append(
                        self._extract_cut_gap_borders(cut.mask, n_points=NUM_SAMPLE_POINTS)
                    )

        # Third pass: last-resort deep threshold if the whole image produced zero.
        if len(edge_detections) == 0:
            print("[Stage 2] No cut detected. Trying last-resort deep threshold...")
            for strip_idx, strip_det in enumerate(strip_detections):
                cuts = self._detect_cut_in_strip(
                    image_rgb=image_rgb,
                    strip_det=strip_det,
                    H=H,
                    W=W,
                    strip_idx=strip_idx,
                    box_thr=GDINO_BOX_THRESHOLD_CUT_LAST_RESORT,
                    text_thr=GDINO_TEXT_THRESHOLD_CUT_LAST_RESORT,
                    relaxed_border_filter=True,
                )
                for cut in cuts:
                    edge_detections.append(cut)
                    edge_sample_points.append(
                        self._extract_cut_gap_borders(cut.mask, n_points=NUM_SAMPLE_POINTS)
                    )

        # Classical fallback: only if all Grounded-SAM passes failed.
        if len(edge_detections) == 0 and USE_CLASSICAL_FALLBACK:
            print("[Stage 2] No deep cut detected. Trying classical fallback...")
            classical = self._detect_cutgap_classical(image_bgr, strip_detections, H, W)
            for det in classical:
                edge_detections.append(det)
                edge_sample_points.append(
                    self._extract_cut_gap_borders(det.mask, n_points=NUM_SAMPLE_POINTS)
                )

        if len(edge_detections) > 1:
            edge_detections, edge_sample_points = self._deduplicate_edges(
                edge_detections, edge_sample_points
            )

        # Final geometric filter: remove obvious false-positive cut gaps.
        # This is applied after all deep/fallback detections and after
        # deduplication, so it does not change the detection process itself;
        # it only rejects boxes that are geometrically incompatible with a
        # valid cut-gap.
        filtered_edges: list[DetectionResult] = []
        filtered_points: list[np.ndarray] = []

        for det, pts in zip(edge_detections, edge_sample_points):
            x1, y1, x2, y2 = det.box_xyxy

            if not is_valid_cutgap_box(
                box=[float(x1), float(y1), float(x2), float(y2)],
                img_w=W,
                img_h=H,
            ):
                print(
                    "[Stage 2] Removed false positive cut gap: "
                    f"{det.box_xyxy.tolist()}"
                )
                continue

            # Extra safety: a cut-gap must be substantially inside at least one
            # detected rubber strip. This removes many no-cut false positives
            # located on the table/image border while keeping true cuts inside
            # the rubber pieces.
            if not self._cut_is_inside_any_strip(det, strip_detections):
                print(
                    "[Stage 2] Removed cut gap outside strips: "
                    f"{det.box_xyxy.tolist()}"
                )
                continue

            filtered_edges.append(det)
            filtered_points.append(pts)

        edge_detections = filtered_edges
        edge_sample_points = filtered_points

        print(f"[Stage 2] CUT detected={len(edge_detections)}")

        return PipelineResult(
            image_path=image_path,
            image_shape=(H, W, 3),
            qr_detections=qr_detections,
            strip_detections=strip_detections,
            edge_detections=edge_detections,
            edge_sample_points=edge_sample_points,
        )

    # ── QR helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _filter_qr_near_corners(
        qr_detections: list[DetectionResult],
        H: int,
        W: int,
        outer_band: float = 0.35,
    ) -> list[DetectionResult]:
        """Keep only QR detections close to image corners/borders."""
        if len(qr_detections) == 0:
            return qr_detections

        corners = {
            "top_left": np.array([0.0, 0.0]),
            "top_right": np.array([float(W - 1), 0.0]),
            "bottom_left": np.array([0.0, float(H - 1)]),
            "bottom_right": np.array([float(W - 1), float(H - 1)]),
        }
        max_corner_dist = 0.30 * float(np.hypot(W, H))
        best_by_corner = {}

        for det in qr_detections:
            x1, y1, x2, y2 = det.box_xyxy
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            centre = np.array([cx, cy])

            near_left = cx <= outer_band * W
            near_right = cx >= (1.0 - outer_band) * W
            near_top = cy <= outer_band * H
            near_bottom = cy >= (1.0 - outer_band) * H
            if not ((near_left or near_right) and (near_top or near_bottom)):
                continue

            distances = {
                name: float(np.linalg.norm(centre - corner_pt))
                for name, corner_pt in corners.items()
            }
            corner_name = min(distances, key=distances.get)
            corner_dist = distances[corner_name]
            if corner_dist > max_corner_dist:
                continue

            quality = float(det.score) - 0.25 * (corner_dist / max_corner_dist)
            if corner_name not in best_by_corner or quality > best_by_corner[corner_name][0]:
                best_by_corner[corner_name] = (quality, det)

        filtered = [
            best_by_corner[name][1]
            for name in ["top_left", "top_right", "bottom_left", "bottom_right"]
            if name in best_by_corner
        ]
        return qr_detections if len(filtered) < 3 else filtered

    @staticmethod
    def _score_qr_by_color(
        image_bgr: np.ndarray,
        qr_detections: list[DetectionResult],
    ) -> list[DetectionResult]:
        """Re-score QR detections using pink/magenta pixel ratio."""
        if not qr_detections:
            return qr_detections

        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        H_img, W_img = image_bgr.shape[:2]
        lower_pink = np.array([130, 40, 100], dtype=np.uint8)
        upper_pink = np.array([175, 255, 255], dtype=np.uint8)
        pink_mask = cv2.inRange(image_hsv, lower_pink, upper_pink)

        updated = []
        for det in qr_detections:
            x1, y1, x2, y2 = det.box_xyxy.astype(int)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(W_img, x2); y2 = min(H_img, y2)
            if x2 <= x1 or y2 <= y1:
                updated.append(det)
                continue

            roi = pink_mask[y1:y2, x1:x2]
            pink_ratio = float(roi.sum() / 255.0) / max(1, roi.size)
            combined = min(1.0, det.score * (1.0 + pink_ratio))
            print(
                f"    [QR color] dino={det.score:.3f} "
                f"pink_ratio={pink_ratio:.3f} combined={combined:.3f}"
            )
            updated.append(DetectionResult(det.label, combined, det.box_xyxy, det.mask))
        return updated

    # ── Stage 2 helper ───────────────────────────────────────────────────────

    def _detect_cut_in_strip(
        self,
        image_rgb: np.ndarray,
        strip_det: DetectionResult,
        H: int,
        W: int,
        strip_idx: int,
        box_thr: float = GDINO_BOX_THRESHOLD_CUT,
        text_thr: float = GDINO_TEXT_THRESHOLD_CUT,
        relaxed_border_filter: bool = False,
    ) -> list[DetectionResult]:
        """Try several prompts on a strip ROI and keep the best valid cut gap."""
        x1, y1, x2, y2 = strip_det.box_xyxy.astype(int)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W, x2); y2 = min(H, y2)
        if x2 <= x1 + 5 or y2 <= y1 + 5:
            return []

        roi_rgb = image_rgb[y1:y2, x1:x2]
        roi_h, roi_w = roi_rgb.shape[:2]
        roi_area = float(roi_h * roi_w)
        is_fallback_roi = strip_det.label.startswith("strip_fallback")
        strip_is_horizontal = roi_w >= roi_h

        for prompt in PROMPT_CUT_CANDIDATES:
            boxes, scores = self._detect_hf(
                roi_rgb,
                prompt,
                box_thr=box_thr,
                text_thr=text_thr,
            )
            if len(boxes) == 0:
                continue

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

                margin_x = 0.03 * roi_w
                margin_y = 0.03 * roi_h
                touches_border = (
                    bx1 < margin_x
                    or bx2 > roi_w - margin_x
                    or by1 < margin_y
                    or by2 > roi_h - margin_y
                )
                if touches_border and not is_fallback_roi and not relaxed_border_filter:
                    continue

                if not is_fallback_roi:
                    cut_is_vertical = bh > bw
                    if strip_is_horizontal and not cut_is_vertical:
                        continue
                    if (not strip_is_horizontal) and cut_is_vertical:
                        continue

                keep_idx.append(i)

            if not keep_idx:
                continue

            boxes = boxes[keep_idx]
            scores = scores[keep_idx]

            from torchvision.ops import nms
            t_b = torch.from_numpy(boxes).float()
            t_s = torch.from_numpy(scores).float()
            keep = nms(t_b, t_s, iou_threshold=NMS_IOU_THRESHOLD).cpu().numpy()
            boxes = boxes[keep]
            scores = scores[keep]

            if len(boxes) > 1:
                best_idx = int(np.argmax(scores))
                boxes = boxes[[best_idx]]
                scores = scores[[best_idx]]

            full_boxes = boxes.copy()
            full_boxes[:, [0, 2]] += x1
            full_boxes[:, [1, 3]] += y1

            print(
                f"  [Stage 2 / strip {strip_idx}] prompt='{prompt}' "
                f"thr=({box_thr:.2f},{text_thr:.2f}) → {len(full_boxes)} cut candidate(s) kept"
            )
            print(f"    selected boxes full image: {full_boxes.tolist()}")
            print(f"    selected scores: {scores.tolist()}")

            return self._segment_boxes(full_boxes, scores, "cut_edge", H, W)

        print(f"  [Stage 2 / strip {strip_idx}] no cut detected with any prompt")
        return []

    @staticmethod
    def _cut_is_inside_any_strip(
        cut_det: DetectionResult,
        strip_detections: list[DetectionResult],
        min_center_margin_frac: float = 0.01,
    ) -> bool:
        """Return True when the cut center is inside a detected strip ROI.

        This is deliberately mild: it does not require a perfect strip mask,
        only that the cut center lies inside one strip bounding box with a tiny
        margin. It helps suppress false positives in images without cuts, where
        detections often appear on the table or the image border.
        """
        if not strip_detections:
            return True

        x1, y1, x2, y2 = [float(v) for v in cut_det.box_xyxy]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        for strip in strip_detections:
            sx1, sy1, sx2, sy2 = [float(v) for v in strip.box_xyxy]
            sw = max(1.0, sx2 - sx1)
            sh = max(1.0, sy2 - sy1)
            mx = min_center_margin_frac * sw
            my = min_center_margin_frac * sh
            if (sx1 + mx) <= cx <= (sx2 - mx) and (sy1 + my) <= cy <= (sy2 - my):
                return True

        return False


    # ── Classical fallback integration ───────────────────────────────────────

    @staticmethod
    def _detect_cutgap_classical(
        image_bgr: np.ndarray,
        strip_detections: list[DetectionResult],
        H: int,
        W: int,
    ) -> list[DetectionResult]:
        """Run the conservative classical fallback and convert results."""
        try:
            from utils.classical_cutgap import detect_cutgap_classic
        except Exception as exc:
            print(f"[Classical fallback] Could not import fallback: {exc}")
            return []

        raw = detect_cutgap_classic(
            image_bgr,
            strip_detections=strip_detections,
            max_detections=CLASSICAL_FALLBACK_MAX_DETECTIONS,
        )
        results = []
        for item in raw:
            mask = item["mask"].astype(bool)
            box = item["box_xyxy"].astype(np.float32)
            score = float(item.get("score", 0.10))
            if mask.shape != (H, W):
                mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            results.append(DetectionResult("cut_edge_classical", score, box, mask))

        print(f"[Classical fallback] recovered {len(results)} cut candidate(s)")
        return results

    # ── Detection / segmentation helpers ─────────────────────────────────────

    def _detect_hf(
        self,
        image_rgb: np.ndarray,
        prompt: str,
        box_thr: float,
        text_thr: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run Grounding DINO on an image or ROI."""
        from PIL import Image
        pil_image = Image.fromarray(image_rgb)
        H, W = image_rgb.shape[:2]
        inputs = self._gdino_processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._gdino_model(**inputs)

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

        return results["boxes"].cpu().numpy(), results["scores"].cpu().numpy()

    @staticmethod
    def _filter_and_nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        img_area: float,
        min_area_frac: float,
        max_area_frac: float,
        aspect_min: Optional[float],
        aspect_max: Optional[float],
        nms_iou: float = NMS_IOU_THRESHOLD,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Discard invalid boxes and apply NMS."""
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

        boxes = boxes[keep]
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
        boxes: np.ndarray,
        scores: np.ndarray,
        label: str,
        H: int,
        W: int,
    ) -> list[DetectionResult]:
        """Run SAM on each box."""
        results = []
        if boxes is None or len(boxes) == 0:
            return results

        input_boxes = torch.tensor(boxes, device=self.device)
        transformed = self._sam_predictor.transform.apply_boxes_torch(input_boxes, (H, W))
        masks_batch, _, _ = self._sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed,
            multimask_output=False,
        )
        for i in range(len(boxes)):
            mask = masks_batch[i, 0].cpu().numpy().astype(bool)
            results.append(DetectionResult(label, float(scores[i]), boxes[i], mask))
        return results

    # ── Cut-gap border sampling ──────────────────────────────────────────────

    @staticmethod
    def _extract_cut_gap_borders(mask: np.ndarray, n_points: int = NUM_SAMPLE_POINTS) -> np.ndarray:
        """Extract points from the two borders of the detected cut gap.

        Returns P1-P5 for one border and P6-P10 for the opposite border.
        """
        if mask is None or mask.sum() == 0:
            return np.empty((0, 2), dtype=np.float32)
        if n_points < 2:
            return GroundedSAMModel._extract_edge_centerline(mask, n_points)

        mask_u8 = mask.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if num_labels <= 1:
            return np.empty((0, 2), dtype=np.float32)

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
        is_horizontal_gap = bw >= bh
        n_side = max(1, n_points // 2)

        border_1 = []
        border_2 = []

        def get_column_pixels(xi: int, search_radius: int = 4) -> np.ndarray:
            values = []
            for dx in range(-search_radius, search_radius + 1):
                x = xi + dx
                if 0 <= x < clean.shape[1]:
                    ys_col = np.where(clean[:, x] > 0)[0]
                    if len(ys_col) > 0:
                        values.extend(ys_col.tolist())
            return np.array(values, dtype=np.float32)

        def get_row_pixels(yi: int, search_radius: int = 4) -> np.ndarray:
            values = []
            for dy in range(-search_radius, search_radius + 1):
                y = yi + dy
                if 0 <= y < clean.shape[0]:
                    xs_row = np.where(clean[y, :] > 0)[0]
                    if len(xs_row) > 0:
                        values.extend(xs_row.tolist())
            return np.array(values, dtype=np.float32)

        if is_horizontal_gap:
            xs_sample = np.linspace(x_min, x_max, n_side)
            for x in xs_sample:
                ys_cross = get_column_pixels(int(round(x)))
                if len(ys_cross) == 0:
                    y_a, y_b = float(y_min), float(y_max)
                else:
                    y_a = float(np.percentile(ys_cross, 5))
                    y_b = float(np.percentile(ys_cross, 95))
                border_1.append([float(x), y_a])
                border_2.append([float(x), y_b])
        else:
            ys_sample = np.linspace(y_min, y_max, n_side)
            for y in ys_sample:
                xs_cross = get_row_pixels(int(round(y)))
                if len(xs_cross) == 0:
                    x_a, x_b = float(x_min), float(x_max)
                else:
                    x_a = float(np.percentile(xs_cross, 5))
                    x_b = float(np.percentile(xs_cross, 95))
                border_1.append([x_a, float(y)])
                border_2.append([x_b, float(y)])

        pts = np.array(border_1 + border_2, dtype=np.float32)
        if len(pts) == 0:
            return GroundedSAMModel._extract_edge_centerline(mask, n_points)
        return pts

    @staticmethod
    def _extract_edge_centerline(mask: np.ndarray, n_points: int = NUM_SAMPLE_POINTS) -> np.ndarray:
        """Fallback method: extract N points from the centre line of a mask."""
        if mask is None or mask.sum() == 0:
            return np.empty((0, 2), dtype=np.float32)

        ys, xs = np.where(mask)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bw = x_max - x_min + 1
        bh = y_max - y_min + 1
        is_horizontal = bw >= bh
        pts = np.empty((n_points, 2), dtype=np.float32)

        if is_horizontal:
            xs_sample = np.linspace(x_min, x_max, n_points)
            for i, xi in enumerate(xs_sample):
                col = mask[:, int(round(xi))]
                ys_col = np.where(col)[0]
                pts[i] = (xi, (y_min + y_max) / 2.0 if len(ys_col) == 0 else float(ys_col.mean()))
        else:
            ys_sample = np.linspace(y_min, y_max, n_points)
            for i, yi in enumerate(ys_sample):
                row = mask[int(round(yi)), :]
                xs_row = np.where(row)[0]
                pts[i] = ((x_min + x_max) / 2.0 if len(xs_row) == 0 else float(xs_row.mean()), yi)
        return pts

    @staticmethod
    def _extract_edge_samples(mask: np.ndarray, n_points: int = NUM_SAMPLE_POINTS) -> np.ndarray:
        """Backwards-compatibility shim."""
        return GroundedSAMModel._extract_cut_gap_borders(mask, n_points)

    # ── Post-detection cleanup ───────────────────────────────────────────────

    @staticmethod
    def _suppress_contained_strips(
        strip_detections: list[DetectionResult],
        containment_threshold: float = 0.70,
    ) -> list[DetectionResult]:
        """Remove strip detections largely contained inside a larger one."""
        if len(strip_detections) <= 1:
            return strip_detections

        boxes = np.array([d.box_xyxy for d in strip_detections])
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        to_remove: set[int] = set()
        n = len(strip_detections)

        for i in range(n):
            if i in to_remove:
                continue
            for j in range(n):
                if j == i or j in to_remove:
                    continue
                if areas[j] <= areas[i]:
                    continue
                ox1 = max(boxes[i, 0], boxes[j, 0])
                oy1 = max(boxes[i, 1], boxes[j, 1])
                ox2 = min(boxes[i, 2], boxes[j, 2])
                oy2 = min(boxes[i, 3], boxes[j, 3])
                if ox2 > ox1 and oy2 > oy1:
                    inter = (ox2 - ox1) * (oy2 - oy1)
                    if inter / max(areas[i], 1e-6) >= containment_threshold:
                        to_remove.add(i)
                        break

        kept = [d for k, d in enumerate(strip_detections) if k not in to_remove]
        if to_remove:
            print(
                f"[Stage 1] Containment suppression removed {len(to_remove)} "
                f"sub-region strip(s); {len(kept)} strip(s) remain."
            )
        return kept

    @staticmethod
    def _deduplicate_edges(
        edge_detections: list[DetectionResult],
        edge_sample_points: list[np.ndarray],
        mask_iou_threshold: float = 0.30,
    ) -> tuple[list[DetectionResult], list[np.ndarray]]:
        """Remove duplicate cut-gap detections whose masks overlap."""
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
        print(f"[Stage 2] Edge deduplication: removed {len(to_remove)} duplicate(s); {len(keep)} edge(s) remain.")
        return [edge_detections[k] for k in keep], [edge_sample_points[k] for k in keep]

# =============================================================================
# model/grounded_sam.py
# Grounded-SAM pipeline using Hugging Face API for Grounding DINO + SAM.
#
# Pipeline steps:
# 1. Grounding DINO (HuggingFace) -> bounding boxes for "qr code" and "cut edge"
# 2. SAM (segment-anything)       -> pixel-level masks for each bounding box
# 3. Post-processing              -> contour extraction, edge sampling
# =============================================================================

from __future__ import annotations

import numpy as np
import torch
import cv2
from dataclasses import dataclass, field
from typing import Optional

from config.config import (
    SAM_CHECKPOINT, SAM_MODEL_TYPE,
    PROMPT_QR, PROMPT_EDGE,
    GDINO_BOX_THRESHOLD, GDINO_TEXT_THRESHOLD,
    NUM_SAMPLE_POINTS, DEVICE,
)
from utils.utils import bgr_to_rgb, sample_points_along_contour


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """One detected object (QR code or cut edge)."""
    label:    str
    score:    float
    box_xyxy: np.ndarray        # [x1, y1, x2, y2] in pixels
    mask:     Optional[np.ndarray] = None  # binary (H, W) bool mask


@dataclass
class PipelineResult:
    """Full output of the Grounded-SAM pipeline for one image."""
    image_path:          str
    image_shape:         tuple
    qr_detections:       list[DetectionResult]
    edge_detections:     list[DetectionResult]
    edge_sample_points:  list[np.ndarray]       # pixel coords (N, 2) per edge
    edge_sample_points_mm:   list[np.ndarray] = field(default_factory=list)
    distances_to_bottom_mm:  list[np.ndarray] = field(default_factory=list)


# ── Grounded-SAM pipeline ─────────────────────────────────────────────────────

class GroundedSAMModel:
    """Grounding DINO (HuggingFace) + SAM pipeline.

    Args:
        device:         'cuda' or 'cpu'
        sam_checkpoint: Path to SAM checkpoint (.pth)
        sam_model_type: 'vit_h', 'vit_l', or 'vit_b'
        gdino_model_id: HuggingFace model ID for Grounding DINO
    """

    GDINO_MODEL_ID = "IDEA-Research/grounding-dino-base"

    def __init__(
        self,
        device:         str = DEVICE,
        sam_checkpoint: str = SAM_CHECKPOINT,
        sam_model_type: str = SAM_MODEL_TYPE,
        # Legacy args kept for CLI compatibility (ignored when using HF)
        gdino_config:   str = "",
        gdino_weights:  str = "",
    ):
        self.device         = device
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = sam_model_type

        self._gdino_processor = None
        self._gdino_model     = None
        self._sam_predictor   = None

    # ── Lazy loading ──────────────────────────────────────────────────────────

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

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, image_bgr: np.ndarray, image_path: str = "") -> PipelineResult:
        """Run the full Grounded-SAM pipeline on one image."""
        self._load_models()

        image_rgb = bgr_to_rgb(image_bgr)
        H, W = image_bgr.shape[:2]

        # 1. Detect QR codes and cut edges
        qr_boxes,   qr_scores   = self._detect_hf(image_rgb, PROMPT_QR)
        edge_boxes, edge_scores = self._detect_hf(image_rgb, PROMPT_EDGE)

        # 2. SAM segmentation
        self._sam_predictor.set_image(image_rgb)

        qr_detections   = self._segment_boxes(qr_boxes,   qr_scores,   "qr_code",  H, W)
        edge_detections = self._segment_boxes(edge_boxes, edge_scores, "cut_edge", H, W)

        # 3. Extract sample points from edge masks
        edge_sample_points = []
        for det in edge_detections:
            if det.mask is not None:
                pts = self._extract_edge_samples(det.mask)
            else:
                pts = np.empty((0, 2), dtype=np.float32)
            edge_sample_points.append(pts)

        return PipelineResult(
            image_path=image_path,
            image_shape=(H, W, 3),
            qr_detections=qr_detections,
            edge_detections=edge_detections,
            edge_sample_points=edge_sample_points,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _detect_hf(
        self,
        image_rgb: np.ndarray,
        prompt:    str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run Grounding DINO via HuggingFace API.

        Returns:
            boxes_xyxy: (N, 4) numpy array in pixel coords
            scores:     (N,)   numpy array of confidence scores
        """
        from PIL import Image

        pil_image = Image.fromarray(image_rgb)
        H, W = image_rgb.shape[:2]

        inputs = self._gdino_processor(
            images=pil_image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._gdino_model(**inputs)

        # Post-process: filter by threshold and convert to pixel coords
        # HuggingFace transformers >= 4.38 uses threshold instead of box_threshold
        try:
            results = self._gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=GDINO_BOX_THRESHOLD,
                text_threshold=GDINO_TEXT_THRESHOLD,
                target_sizes=[(H, W)],
            )[0]
        except TypeError:
            results = self._gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=GDINO_BOX_THRESHOLD,
                target_sizes=[(H, W)],
            )[0]

        boxes  = results["boxes"].cpu().numpy()   # (N, 4) xyxy pixels
        scores = results["scores"].cpu().numpy()  # (N,)

        return boxes, scores

    def _segment_boxes(
        self,
        boxes:  np.ndarray,
        scores: np.ndarray,
        label:  str,
        H:      int,
        W:      int,
    ) -> list[DetectionResult]:
        """Run SAM on each bounding box and return DetectionResult list."""
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
        # masks_batch: (N, 1, H, W) bool tensor

        for i in range(len(boxes)):
            mask = masks_batch[i, 0].cpu().numpy()
            results.append(DetectionResult(
                label=label,
                score=float(scores[i]),
                box_xyxy=boxes[i],
                mask=mask,
            ))
        return results

    @staticmethod
    def _extract_edge_samples(
        mask:     np.ndarray,
        n_points: int = NUM_SAMPLE_POINTS,
    ) -> np.ndarray:
        """Extract contour from mask and sample n points along it."""
        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return np.empty((0, 2), dtype=np.float32)
        contour = max(contours, key=cv2.contourArea)
        pts = contour.reshape(-1, 2).astype(np.float32)
        return sample_points_along_contour(pts, n=n_points)

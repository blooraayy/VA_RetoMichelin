# =============================================================================
# model/grounded_sam.py
# Grounded-SAM pipeline: Grounding DINO (detection) → SAM (segmentation).
#
# Pipeline steps
# --------------
# 1. Load image
# 2. Grounding DINO → bounding boxes for "qr code" and "cut edge"
# 3. SAM           → pixel-level masks for each bounding box
# 4. Post-processing (contour extraction, edge-boundary sampling)
# =============================================================================

from __future__ import annotations

import os
import sys
import numpy as np
import torch
import cv2
from dataclasses import dataclass, field
from typing import Optional

from config.config import (
    GDINO_CONFIG, GDINO_WEIGHTS,
    SAM_CHECKPOINT, SAM_MODEL_TYPE,
    PROMPT_QR, PROMPT_EDGE,
    GDINO_BOX_THRESHOLD, GDINO_TEXT_THRESHOLD,
    NUM_SAMPLE_POINTS, DEVICE,
)
from utils.utils import bgr_to_rgb, sample_points_along_contour


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """One detected object (QR code or cut edge)."""
    label: str
    score: float
    box_xyxy: np.ndarray        # [x1, y1, x2, y2] in pixels
    mask: Optional[np.ndarray]  # binary mask H×W (bool), None before SAM


@dataclass
class PipelineResult:
    """Full output of the Grounded-SAM pipeline for one image."""
    image_path: str
    image_shape: tuple                          # (H, W, 3)
    qr_detections: list[DetectionResult]
    edge_detections: list[DetectionResult]
    # Sampled pixel coordinates along each detected edge contour
    edge_sample_points: list[np.ndarray]        # list of (N,2) arrays
    # Measurements filled in by the Controller after calibration
    edge_sample_points_mm: list[np.ndarray] = field(default_factory=list)
    distances_to_bottom_mm: list[np.ndarray] = field(default_factory=list)


# ── Grounded-SAM pipeline ────────────────────────────────────────────────────

class GroundedSAMModel:
    """Wraps Grounding DINO + SAM in a single inference pipeline.

    The model is lazy-loaded the first time ``run()`` is called so that
    importing this module does not require the weights to be present.

    Args:
        device: "cuda" or "cpu".
        gdino_config:   Path to Grounding DINO config file.
        gdino_weights:  Path to Grounding DINO checkpoint.
        sam_checkpoint: Path to SAM checkpoint.
        sam_model_type: One of "vit_h", "vit_l", "vit_b".
    """

    def __init__(
        self,
        device: str = DEVICE,
        gdino_config: str = GDINO_CONFIG,
        gdino_weights: str = GDINO_WEIGHTS,
        sam_checkpoint: str = SAM_CHECKPOINT,
        sam_model_type: str = SAM_MODEL_TYPE,
    ):
        self.device         = device
        self.gdino_config   = gdino_config
        self.gdino_weights  = gdino_weights
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = sam_model_type

        self._gdino_model = None
        self._sam_predictor = None

    # ── Lazy loading ─────────────────────────────────────────────────────────

    def _load_models(self) -> None:
        """Load Grounding DINO and SAM weights into memory."""
        if self._gdino_model is not None:
            return   # already loaded

        print("[GroundedSAMModel] Loading Grounding DINO …")
        self._gdino_model = self._load_grounding_dino()

        print("[GroundedSAMModel] Loading SAM …")
        self._sam_predictor = self._load_sam()

        print("[GroundedSAMModel] Models ready.")

    def _load_grounding_dino(self):
        """Load Grounding DINO using the GroundingDINO library."""
        try:
            from groundingdino.util.inference import load_model
        except ImportError as e:
            raise ImportError(
                "GroundingDINO not installed. "
                "Run: pip install groundingdino-py\n"
                f"Original error: {e}"
            )
        model = load_model(self.gdino_config, self.gdino_weights)
        model = model.to(self.device)
        model.eval()
        return model

    def _load_sam(self):
        """Load SAM and return a SamPredictor."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError as e:
            raise ImportError(
                "segment_anything not installed. "
                "Run: pip install segment-anything\n"
                f"Original error: {e}"
            )
        sam = sam_model_registry[self.sam_model_type](
            checkpoint=self.sam_checkpoint
        )
        sam = sam.to(self.device)
        predictor = SamPredictor(sam)
        return predictor

    # ── Public API ───────────────────────────────────────────────────────────

    def run(self, image_bgr: np.ndarray, image_path: str = "") -> PipelineResult:
        """Run the full Grounded-SAM pipeline on one image.

        Args:
            image_bgr:   BGR image (H, W, 3) as loaded by OpenCV.
            image_path:  Original file path (for bookkeeping only).

        Returns:
            PipelineResult with all detections, masks, and sampled points.
        """
        self._load_models()

        image_rgb = bgr_to_rgb(image_bgr)
        H, W = image_bgr.shape[:2]

        # ── Step 1: Grounding DINO detection ────────────────────────────────
        qr_boxes, qr_scores     = self._detect(image_rgb, PROMPT_QR)
        edge_boxes, edge_scores = self._detect(image_rgb, PROMPT_EDGE)

        # ── Step 2: SAM segmentation ─────────────────────────────────────────
        self._sam_predictor.set_image(image_rgb)

        qr_detections   = self._segment_boxes(qr_boxes,   qr_scores,   "qr_code",    H, W)
        edge_detections = self._segment_boxes(edge_boxes, edge_scores, "cut_edge",   H, W)

        # ── Step 3: Extract contours from edge masks ──────────────────────
        edge_sample_points = []
        for det in edge_detections:
            if det.mask is not None:
                pts = self._extract_edge_samples(det.mask)
                edge_sample_points.append(pts)
            else:
                edge_sample_points.append(np.empty((0, 2), dtype=np.float32))

        return PipelineResult(
            image_path=image_path,
            image_shape=(H, W, 3),
            qr_detections=qr_detections,
            edge_detections=edge_detections,
            edge_sample_points=edge_sample_points,
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    def _detect(
        self,
        image_rgb: np.ndarray,
        prompt: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Grounding DINO and return (boxes_xyxy, scores) as tensors."""
        from groundingdino.util.inference import predict
        from torchvision.ops import box_convert
        import torch

        # GroundingDINO expects a PIL image or a transformed tensor.
        # The library's ``predict`` utility handles the conversion internally.
        boxes_cxcywh, scores, _ = predict(
            model=self._gdino_model,
            image=self._preprocess_for_gdino(image_rgb),
            caption=prompt,
            box_threshold=GDINO_BOX_THRESHOLD,
            text_threshold=GDINO_TEXT_THRESHOLD,
            device=self.device,
        )

        # Convert from normalised cx,cy,w,h → absolute x1,y1,x2,y2
        H, W = image_rgb.shape[:2]
        boxes_xyxy = box_convert(
            boxes=boxes_cxcywh * torch.tensor([W, H, W, H], device=boxes_cxcywh.device),
            in_fmt="cxcywh",
            out_fmt="xyxy",
        )
        return boxes_xyxy, scores

    @staticmethod
    def _preprocess_for_gdino(image_rgb: np.ndarray):
        """Convert an RGB numpy array to the tensor format expected by
        Grounding DINO's ``predict`` helper."""
        from groundingdino.util.transforms import Compose, Normalize, ToTensor, RandomResize
        import torchvision.transforms.functional as F
        from PIL import Image

        pil_img = Image.fromarray(image_rgb)
        transform = Compose([
            RandomResize([800], max_size=1333),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img_transformed, _ = transform(pil_img, None)
        return img_transformed

    def _segment_boxes(
        self,
        boxes: "torch.Tensor",
        scores: "torch.Tensor",
        label: str,
        H: int,
        W: int,
    ) -> list[DetectionResult]:
        """For each bounding box run SAM and return DetectionResult list."""
        results = []
        if boxes is None or len(boxes) == 0:
            return results

        boxes_np = boxes.cpu().numpy()
        scores_np = scores.cpu().numpy()

        # SAM expects boxes in xyxy pixel format
        input_boxes = torch.tensor(boxes_np, device=self.device)
        transformed_boxes = self._sam_predictor.transform.apply_boxes_torch(
            input_boxes, (H, W)
        )

        masks_batch, _, _ = self._sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        # masks_batch: (N, 1, H, W) bool tensor

        for i in range(len(boxes_np)):
            mask = masks_batch[i, 0].cpu().numpy()   # (H, W) bool
            results.append(DetectionResult(
                label=label,
                score=float(scores_np[i]),
                box_xyxy=boxes_np[i],
                mask=mask,
            ))
        return results

    @staticmethod
    def _extract_edge_samples(
        mask: np.ndarray,
        n_points: int = NUM_SAMPLE_POINTS,
    ) -> np.ndarray:
        """Extract the boundary contour of a binary mask and sample *n* points.

        Returns:
            Array of shape (n, 2) with (x, y) pixel coordinates.
        """
        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return np.empty((0, 2), dtype=np.float32)

        # Use the longest contour (most likely the cut edge boundary)
        contour = max(contours, key=cv2.contourArea)

        # Keep only the bottom boundary (maximum y per x column) as the
        # "cut edge" line, since that is the separation we want to measure.
        pts = contour.reshape(-1, 2).astype(np.float32)

        return sample_points_along_contour(pts, n=n_points)

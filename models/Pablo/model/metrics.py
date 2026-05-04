# =============================================================================
# model/metrics.py
# Geometric measurement and evaluation metrics for the Michelin Challenge 2.
#
# Two categories of metrics:
#   A) Geometric measurements (always computed, output in mm)
#      - Distance from each sample point on a cut edge to the table bottom
#      - Distance between the two cut edges
#
#   B) Evaluation metrics (computed when ground-truth labels are available)
#      - RMSE, MAE between predicted and ground-truth distances
#      - IoU of predicted vs. ground-truth masks
#      - Precision / Recall / F1 for edge detection
# =============================================================================

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from model.qr_calibrator import QRCalibrator
from model.grounded_sam  import PipelineResult
from config.config       import NUM_SAMPLE_POINTS, QR_LEG_MM


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class GeometricMeasurements:
    """Metric measurements derived from the Grounded-SAM pipeline output."""
    # For each detected cut edge: N distance values (mm) to the table bottom
    distances_to_bottom_mm: list[np.ndarray] = field(default_factory=list)
    # Distance between cut-edge pair (mm), one value per pair
    inter_edge_distances_mm: list[float] = field(default_factory=list)
    # Sample points in mm space, one array (N,2) per edge
    edge_sample_points_mm: list[np.ndarray] = field(default_factory=list)


@dataclass
class EvaluationMetrics:
    """Quantitative evaluation metrics (require ground-truth annotations)."""
    rmse_mm:  Optional[float] = None
    mae_mm:   Optional[float] = None
    iou:      Optional[float] = None
    precision: Optional[float] = None
    recall:   Optional[float] = None
    f1:       Optional[float] = None


# ── Geometric measurement ─────────────────────────────────────────────────────

class MeasurementEngine:
    """Converts pixel-space pipeline results into mm-space measurements.

    Args:
        qr_calibrator: A QRCalibrator instance (provides homography).
        image_height_px: Height of the source image in pixels
                         (needed to locate the table bottom in px space).
    """

    def __init__(self, qr_calibrator: QRCalibrator, image_height_px: int):
        self.calibrator      = qr_calibrator
        self.img_height_px   = image_height_px

    def compute(
        self,
        result: PipelineResult,
        homography: np.ndarray,
    ) -> GeometricMeasurements:
        """Compute all geometric measurements from a PipelineResult.

        Args:
            result:     Output of GroundedSAMModel.run().
            homography: 3×3 affine/homography from pixel to mm space.

        Returns:
            GeometricMeasurements with all distance values in mm.
        """
        meas = GeometricMeasurements()
        W    = result.image_shape[1]

        # Table bottom in pixel space: a horizontal line at y = image_height-1
        # In mm space, that line is obtained by transforming two of its points.
        bottom_left_mm  = self.calibrator.pixel_to_mm(
            (0.0, float(self.img_height_px - 1)), homography
        )
        bottom_right_mm = self.calibrator.pixel_to_mm(
            (float(W - 1), float(self.img_height_px - 1)), homography
        )
        table_bottom_y_mm = (bottom_left_mm[1] + bottom_right_mm[1]) / 2.0

        edge_centres_y_mm: list[float] = []   # y centroid of each edge in mm

        for sample_pts_px in result.edge_sample_points:
            if len(sample_pts_px) == 0:
                meas.distances_to_bottom_mm.append(np.array([], dtype=np.float32))
                meas.edge_sample_points_mm.append(np.empty((0, 2), dtype=np.float32))
                continue

            # Convert all sample points to mm
            pts_mm = self.calibrator.pixels_to_mm_bulk(sample_pts_px, homography)
            meas.edge_sample_points_mm.append(pts_mm)

            # Distance from each sample point to the table bottom
            # (vertical distance in mm coordinate frame)
            dists = np.abs(table_bottom_y_mm - pts_mm[:, 1])
            meas.distances_to_bottom_mm.append(dists.astype(np.float32))

            # Track the mean y of this edge for inter-edge distance
            edge_centres_y_mm.append(float(pts_mm[:, 1].mean()))

        # Inter-edge distance (between each consecutive pair)
        for i in range(len(edge_centres_y_mm) - 1):
            gap = abs(edge_centres_y_mm[i + 1] - edge_centres_y_mm[i])
            meas.inter_edge_distances_mm.append(gap)

        return meas


# ── Evaluation metrics ────────────────────────────────────────────────────────

class MetricEvaluator:
    """Computes quantitative metrics against ground-truth annotations.

    Ground-truth format (numpy arrays):
        gt_distances_mm:  (N,) array of mm distances (one per sample point).
        gt_mask:          (H, W) binary bool array.
        gt_edge_pts:      (N, 2) array of annotated boundary pixels.
    """

    # ── Distance-based metrics ────────────────────────────────────────────

    @staticmethod
    def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
        """Root Mean Square Error between predicted and GT distances (mm)."""
        pred = np.asarray(pred, dtype=np.float64)
        gt   = np.asarray(gt,   dtype=np.float64)
        if len(pred) == 0 or len(gt) == 0:
            return float("nan")
        n = min(len(pred), len(gt))
        return float(np.sqrt(np.mean((pred[:n] - gt[:n]) ** 2)))

    @staticmethod
    def mae(pred: np.ndarray, gt: np.ndarray) -> float:
        """Mean Absolute Error between predicted and GT distances (mm)."""
        pred = np.asarray(pred, dtype=np.float64)
        gt   = np.asarray(gt,   dtype=np.float64)
        if len(pred) == 0 or len(gt) == 0:
            return float("nan")
        n = min(len(pred), len(gt))
        return float(np.mean(np.abs(pred[:n] - gt[:n])))

    # ── Segmentation metrics ──────────────────────────────────────────────

    @staticmethod
    def iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Intersection over Union for binary masks."""
        pred = pred_mask.astype(bool)
        gt   = gt_mask.astype(bool)
        intersection = float((pred & gt).sum())
        union        = float((pred | gt).sum())
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def precision_recall_f1(
        pred_mask: np.ndarray,
        gt_mask:   np.ndarray,
    ) -> tuple[float, float, float]:
        """Pixel-level Precision, Recall, and F1 for a binary mask."""
        pred = pred_mask.astype(bool).ravel()
        gt   = gt_mask.astype(bool).ravel()

        tp = float((pred & gt).sum())
        fp = float((pred & ~gt).sum())
        fn = float((~pred & gt).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        return precision, recall, f1

    # ── Convenience wrapper ───────────────────────────────────────────────

    def evaluate(
        self,
        pred_distances_mm: np.ndarray,
        gt_distances_mm:   np.ndarray,
        pred_mask:         Optional[np.ndarray] = None,
        gt_mask:           Optional[np.ndarray] = None,
    ) -> EvaluationMetrics:
        """Compute all available metrics and return an EvaluationMetrics object."""
        em = EvaluationMetrics()
        em.rmse_mm = self.rmse(pred_distances_mm, gt_distances_mm)
        em.mae_mm  = self.mae(pred_distances_mm,  gt_distances_mm)

        if pred_mask is not None and gt_mask is not None:
            em.iou = self.iou(pred_mask, gt_mask)
            em.precision, em.recall, em.f1 = self.precision_recall_f1(
                pred_mask, gt_mask
            )
        return em

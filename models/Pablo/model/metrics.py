# =============================================================================
# model/metrics.py
# Geometric measurement and evaluation metrics for the Michelin Challenge 2.
# =============================================================================

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from model.qr_calibrator import QRCalibrator
from model.grounded_sam  import PipelineResult
from config.config       import NUM_SAMPLE_POINTS


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class GeometricMeasurements:
    """Metric measurements derived from the Grounded-SAM pipeline output."""
    distances_to_bottom_mm:  list[np.ndarray] = field(default_factory=list)
    inter_edge_distances_mm: list[float]       = field(default_factory=list)
    edge_sample_points_mm:   list[np.ndarray]  = field(default_factory=list)


@dataclass
class EvaluationMetrics:
    """Quantitative evaluation metrics against ground truth."""
    rmse_mm:          Optional[float] = None
    mae_mm:           Optional[float] = None
    iou:              Optional[float] = None
    precision:        Optional[float] = None
    recall:           Optional[float] = None
    f1:               Optional[float] = None
    per_edge_iou:     list[float]     = field(default_factory=list)
    per_edge_rmse_mm: list[float]     = field(default_factory=list)
    per_edge_mae_mm:  list[float]     = field(default_factory=list)


# ── Geometric measurement ─────────────────────────────────────────────────────

class MeasurementEngine:
    """Converts pixel-space pipeline results into mm-space measurements."""

    def __init__(self, qr_calibrator: QRCalibrator, image_height_px: int):
        self.calibrator    = qr_calibrator
        self.img_height_px = image_height_px

    def compute(
        self,
        result:     PipelineResult,
        homography: np.ndarray,
    ) -> GeometricMeasurements:
        meas = GeometricMeasurements()
        W    = result.image_shape[1]

        bottom_left_mm  = self.calibrator.pixel_to_mm(
            (0.0, float(self.img_height_px - 1)), homography)
        bottom_right_mm = self.calibrator.pixel_to_mm(
            (float(W - 1), float(self.img_height_px - 1)), homography)
        table_bottom_y_mm = (bottom_left_mm[1] + bottom_right_mm[1]) / 2.0

        edge_centres_y_mm: list[float] = []

        for sample_pts_px in result.edge_sample_points:
            if len(sample_pts_px) == 0:
                meas.distances_to_bottom_mm.append(np.array([], dtype=np.float32))
                meas.edge_sample_points_mm.append(np.empty((0, 2), dtype=np.float32))
                continue
            pts_mm = self.calibrator.pixels_to_mm_bulk(sample_pts_px, homography)
            meas.edge_sample_points_mm.append(pts_mm)
            dists  = np.abs(table_bottom_y_mm - pts_mm[:, 1])
            meas.distances_to_bottom_mm.append(dists.astype(np.float32))
            edge_centres_y_mm.append(float(pts_mm[:, 1].mean()))

        for i in range(len(edge_centres_y_mm) - 1):
            meas.inter_edge_distances_mm.append(
                abs(edge_centres_y_mm[i + 1] - edge_centres_y_mm[i]))

        return meas


# ── Evaluation metrics ────────────────────────────────────────────────────────

class MetricEvaluator:
    """Computes quantitative metrics against COCO ground-truth annotations."""

    @staticmethod
    def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
        pred, gt = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
        if len(pred) == 0 or len(gt) == 0:
            return float("nan")
        n = min(len(pred), len(gt))
        return float(np.sqrt(np.mean((pred[:n] - gt[:n]) ** 2)))

    @staticmethod
    def mae(pred: np.ndarray, gt: np.ndarray) -> float:
        pred, gt = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
        if len(pred) == 0 or len(gt) == 0:
            return float("nan")
        n = min(len(pred), len(gt))
        return float(np.mean(np.abs(pred[:n] - gt[:n])))

    @staticmethod
    def iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        pred = pred_mask.astype(bool)
        gt   = gt_mask.astype(bool)
        intersection = float((pred & gt).sum())
        union        = float((pred | gt).sum())
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def precision_recall_f1(
        pred_mask: np.ndarray, gt_mask: np.ndarray
    ) -> tuple[float, float, float]:
        pred = pred_mask.astype(bool).ravel()
        gt   = gt_mask.astype(bool).ravel()
        tp = float((pred & gt).sum())
        fp = float((pred & ~gt).sum())
        fn = float((~pred & gt).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        return precision, recall, f1

    def evaluate_against_gt(
        self,
        pipeline_result: PipelineResult,
        measurements:    GeometricMeasurements,
        gt,                              # GTAnnotation from coco_loader
        homography:      Optional[np.ndarray]    = None,
        qr_calibrator:   Optional[QRCalibrator]  = None,
    ) -> EvaluationMetrics:
        """Compare predictions against COCO ground truth."""
        em = EvaluationMetrics()
        H, W = pipeline_result.image_shape[:2]

        pred_masks = [det.mask for det in pipeline_result.edge_detections
                      if det.mask is not None]
        gt_masks   = gt.cut_edge_masks

        # ── IoU per edge ──────────────────────────────────────────────────────
        iou_scores = []
        for pred_mask in pred_masks:
            best = 0.0
            for gt_mask in gt_masks:
                if gt_mask.shape != pred_mask.shape:
                    gt_r = cv2.resize(
                        gt_mask.astype(np.uint8),
                        (pred_mask.shape[1], pred_mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                else:
                    gt_r = gt_mask
                best = max(best, self.iou(pred_mask, gt_r))
            iou_scores.append(best)

        em.per_edge_iou = iou_scores
        em.iou = float(np.mean(iou_scores)) if iou_scores else float("nan")

        # ── Precision / Recall / F1 ───────────────────────────────────────────
        if pred_masks and gt_masks:
            pred_combined = np.zeros((H, W), bool)
            gt_combined   = np.zeros((H, W), bool)
            for m in pred_masks:
                pred_combined |= m.astype(bool)
            for m in gt_masks:
                m_r = (cv2.resize(m.astype(np.uint8), (W, H),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
                       if m.shape != (H, W) else m.astype(bool))
                gt_combined |= m_r
            em.precision, em.recall, em.f1 = \
                self.precision_recall_f1(pred_combined, gt_combined)

        # ── Distance RMSE / MAE ───────────────────────────────────────────────
        if homography is not None and qr_calibrator is not None:
            gt_dists = self._gt_to_distances(gt, homography, qr_calibrator, H, W)
            rmse_vals, mae_vals = [], []
            for i, pred_d in enumerate(measurements.distances_to_bottom_mm):
                if i < len(gt_dists) and len(pred_d) > 0:
                    rmse_vals.append(self.rmse(pred_d, gt_dists[i]))
                    mae_vals.append(self.mae(pred_d,  gt_dists[i]))
            em.per_edge_rmse_mm = rmse_vals
            em.per_edge_mae_mm  = mae_vals
            em.rmse_mm = float(np.nanmean(rmse_vals)) if rmse_vals else float("nan")
            em.mae_mm  = float(np.nanmean(mae_vals))  if mae_vals  else float("nan")

        return em

    @staticmethod
    def _gt_to_distances(
        gt, homography, qr_calibrator, H: int, W: int
    ) -> list[np.ndarray]:
        bl = qr_calibrator.pixel_to_mm((0.0, float(H - 1)), homography)
        br = qr_calibrator.pixel_to_mm((float(W - 1), float(H - 1)), homography)
        table_bottom_y_mm = (bl[1] + br[1]) / 2.0
        dists_list = []
        for pts_px in gt.cut_edge_points:
            if len(pts_px) == 0:
                dists_list.append(np.array([], np.float32))
                continue
            pts_mm = qr_calibrator.pixels_to_mm_bulk(pts_px, homography)
            dists_list.append(
                np.abs(table_bottom_y_mm - pts_mm[:, 1]).astype(np.float32))
        return dists_list

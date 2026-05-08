# =============================================================================
# view/visualiser.py
# Renders and saves all visual outputs for the Michelin Challenge 2 pipeline.
#
# Responsibilities (View layer)
# ------------------------------
# - Draw QR bounding boxes and centres
# - Overlay SAM segmentation masks (semi-transparent)
# - Draw detected cut-edge bounding boxes
# - Mark the 10 sample points on each edge
# - Annotate distance labels (mm) next to each sample point
# - Print a summary table to stdout
# - Save the annotated image and a JSON results file
# =============================================================================

from __future__ import annotations

import os
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for headless servers
import matplotlib.pyplot as plt
from datetime import datetime

from config.config import (
    COLOR_QR_BOX, COLOR_EDGE_BOX, COLOR_EDGE_MASK,
    COLOR_SAMPLE_PT, COLOR_DISTANCE,
    MASK_ALPHA, OUTPUT_DIR, SAVE_VISUALISATIONS, SHOW_VISUALISATIONS,
)
from model.grounded_sam import PipelineResult
from model.metrics      import GeometricMeasurements, EvaluationMetrics


class Visualiser:
    """Creates annotated images and console summaries of pipeline results.

    Args:
        output_dir: Directory where annotated images and JSON files are saved.
        show:       If True, display the image in an OpenCV window.
        save:       If True, write the annotated image to disk.
    """

    def __init__(
        self,
        output_dir: str  = OUTPUT_DIR,
        show: bool       = SHOW_VISUALISATIONS,
        save: bool       = SAVE_VISUALISATIONS,
    ):
        self.output_dir = output_dir
        self.show       = show
        self.save       = save
        os.makedirs(self.output_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def render(
        self,
        image_bgr:    np.ndarray,
        pipeline_res: PipelineResult,
        measurements: GeometricMeasurements,
        eval_metrics: EvaluationMetrics | None = None,
        base_name:    str = "result",
    ) -> np.ndarray:
        """Produce and (optionally) save / display the annotated image.

        Args:
            image_bgr:    Original BGR image.
            pipeline_res: Output from GroundedSAMModel.run().
            measurements: Output from MeasurementEngine.compute().
            eval_metrics: Optional evaluation metrics (printed if provided).
            base_name:    Stem for output file names.

        Returns:
            The annotated BGR image.
        """
        canvas = image_bgr.copy()

        # 1. Draw QR detections
        self._draw_qr_detections(canvas, pipeline_res)

        # 2. Draw edge masks and bounding boxes
        self._draw_edge_detections(canvas, pipeline_res)

        # 3. Draw sample points + distance labels
        self._draw_measurements(canvas, measurements)

        # 4. Console summary
        self.print_summary(pipeline_res, measurements, eval_metrics)

        # 5. Save annotated image
        if self.save:
            self._save_image(canvas, base_name)
            self._save_json(pipeline_res, measurements, eval_metrics, base_name)

        # 6. Optionally display
        if self.show:
            self._show_image(canvas, base_name)

        return canvas

    def print_summary(
        self,
        pipeline_res: PipelineResult,
        measurements: GeometricMeasurements,
        eval_metrics: EvaluationMetrics | None = None,
    ) -> None:
        """Print a formatted summary to stdout."""
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  MICHELIN CHALLENGE 2 — Grounded-SAM Results")
        print(f"  Image : {pipeline_res.image_path}")
        print(sep)

        print(f"\n  QR codes detected : {len(pipeline_res.qr_detections)}")
        print(f"  Cut edges detected: {len(pipeline_res.edge_detections)}")

        for ei, dists in enumerate(measurements.distances_to_bottom_mm):
            print(f"\n  Edge {ei + 1} — distances to table bottom (mm):")
            if len(dists) == 0:
                print("    (no sample points)")
                continue
            for j, d in enumerate(dists):
                print(f"    Point {j + 1:2d}: {d:7.2f} mm")
            print(f"    Mean : {dists.mean():.2f} mm  |  "
                  f"Std  : {dists.std():.2f} mm  |  "
                  f"Range: [{dists.min():.2f}, {dists.max():.2f}] mm")

        for k, gap in enumerate(measurements.inter_edge_distances_mm):
            print(f"\n  Inter-edge gap {k + 1}–{k + 2}: {gap:.2f} mm")

        if eval_metrics is not None:
            print(f"\n  Evaluation metrics:")
            if eval_metrics.rmse_mm is not None:
                print(f"    RMSE : {eval_metrics.rmse_mm:.3f} mm")
            if eval_metrics.mae_mm  is not None:
                print(f"    MAE  : {eval_metrics.mae_mm:.3f} mm")
            if eval_metrics.iou     is not None:
                print(f"    IoU  : {eval_metrics.iou:.4f}")
            if eval_metrics.f1      is not None:
                print(f"    F1   : {eval_metrics.f1:.4f}  "
                      f"(P={eval_metrics.precision:.4f}, "
                      f"R={eval_metrics.recall:.4f})")

        print(f"\n{sep}\n")

    # ── Private drawing helpers ───────────────────────────────────────────────

    @staticmethod
    def _draw_qr_detections(canvas: np.ndarray,
                            pipeline_res: PipelineResult) -> None:
        for det in pipeline_res.qr_detections:
            x1, y1, x2, y2 = det.box_xyxy.astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_QR_BOX, 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.drawMarker(canvas, (cx, cy), COLOR_QR_BOX,
                           cv2.MARKER_CROSS, 15, 2)
            label = f"QR marker (conf: {det.score:.2f})"
            cv2.putText(canvas, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_QR_BOX, 1,
                        cv2.LINE_AA)

    def _draw_edge_detections(self, canvas: np.ndarray,
                               pipeline_res: PipelineResult) -> None:
        overlay = canvas.copy()
        for det in pipeline_res.edge_detections:
            # Semi-transparent mask
            if det.mask is not None:
                overlay[det.mask] = COLOR_EDGE_MASK

            # Bounding box
            x1, y1, x2, y2 = det.box_xyxy.astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_EDGE_BOX, 2)
            label = f"Cut edge (conf: {det.score:.2f})"
            cv2.putText(canvas, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_EDGE_BOX, 1,
                        cv2.LINE_AA)

        cv2.addWeighted(overlay, MASK_ALPHA, canvas, 1 - MASK_ALPHA, 0, canvas)

    @staticmethod
    def _draw_measurements(canvas: np.ndarray,
                           measurements: GeometricMeasurements) -> None:
        for ei, (pts_px, dists_mm) in enumerate(
            zip(measurements.edge_sample_points_mm,
                measurements.distances_to_bottom_mm)
        ):
            if len(pts_px) == 0:
                continue
            for j, (pt_mm, d_mm) in enumerate(zip(pts_px, dists_mm)):
                # Convert mm back to pixel is not straightforward without
                # the inverse homography; we stored the pixel pts in the
                # pipeline result — here we use the mm values only for labels
                # and draw sample points from edge_sample_points (pixel).
                pass  # drawing below uses edge_sample_points directly

        # We draw using pixel coordinates from the pipeline result via
        # measurements.edge_sample_points_mm — but we need the pixel version.
        # The Controller passes both; the Visualiser uses what it gets here.
        # (See Controller for the coupling logic.)

    def draw_sample_points_px(
        self,
        canvas: np.ndarray,
        pipeline_res: PipelineResult,
        measurements: GeometricMeasurements,
    ) -> None:
        """Draw sample points using pixel coordinates and annotate with mm values.

        Call this from the Controller after ``render()`` if you want the
        per-point annotations overlaid on the canvas in pixel space.
        """
        for ei, (pts_px, dists_mm) in enumerate(
            zip(pipeline_res.edge_sample_points,
                measurements.distances_to_bottom_mm)
        ):
            if len(pts_px) == 0:
                continue
            for j, (pt, d_mm) in enumerate(zip(pts_px, dists_mm)):
                px, py = int(pt[0]), int(pt[1])
                cv2.circle(canvas, (px, py), 5, COLOR_SAMPLE_PT, -1)
                label = f"{d_mm:.1f}"
                cv2.putText(canvas, label, (px + 6, py - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            COLOR_DISTANCE, 1, cv2.LINE_AA)

    # ── File I/O ──────────────────────────────────────────────────────────────

    def _save_image(self, canvas: np.ndarray, base_name: str) -> None:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"{base_name}_{ts}.png")
        cv2.imwrite(path, canvas)
        print(f"[Visualiser] Saved annotated image → {path}")

    def _save_json(
        self,
        pipeline_res: PipelineResult,
        measurements: GeometricMeasurements,
        eval_metrics: EvaluationMetrics | None,
        base_name:    str,
    ) -> None:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"{base_name}_{ts}.json")

        data: dict = {
            "image_path": pipeline_res.image_path,
            "n_qr_detected":   len(pipeline_res.qr_detections),
            "n_edges_detected": len(pipeline_res.edge_detections),
            "edges": [],
        }

        for ei, dists in enumerate(measurements.distances_to_bottom_mm):
            edge_data = {
                "edge_id": ei + 1,
                "distances_to_bottom_mm": (dists.tolist()
                                           if len(dists) else []),
            }
            data["edges"].append(edge_data)

        data["inter_edge_distances_mm"] = measurements.inter_edge_distances_mm

        if eval_metrics is not None:
            data["evaluation"] = {
                "rmse_mm":   eval_metrics.rmse_mm,
                "mae_mm":    eval_metrics.mae_mm,
                "iou":       eval_metrics.iou,
                "precision": eval_metrics.precision,
                "recall":    eval_metrics.recall,
                "f1":        eval_metrics.f1,
            }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Visualiser] Saved JSON results    → {path}")

    @staticmethod
    def _show_image(canvas: np.ndarray, title: str) -> None:
        cv2.imshow(title, canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

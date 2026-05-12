# =============================================================================
# view/visualiser.py
# Renders and saves all visual outputs for the Michelin Challenge 2 pipeline.
#
# Responsibilities:
# - Draw QR bounding boxes and centres.
# - Overlay SAM segmentation masks.
# - Draw detected cut-gap bounding boxes.
# - Mark sampled points:
#     P1-P5  -> one border of the cut gap.
#     P6-P10 -> opposite border of the cut gap.
# - Annotate calibrated distances in mm.
# - Print a summary table.
# - Save annotated images, JSON results and CSV measurements.
# =============================================================================

from __future__ import annotations

import os
import json
import cv2
import csv
import numpy as np
from datetime import datetime

from config.config import (
    COLOR_QR_BOX, COLOR_EDGE_BOX, COLOR_EDGE_MASK,
    COLOR_SAMPLE_PT, COLOR_DISTANCE,
    MASK_ALPHA, OUTPUT_DIR, SAVE_VISUALISATIONS, SHOW_VISUALISATIONS,
)
from model.grounded_sam import PipelineResult
from model.metrics import GeometricMeasurements, EvaluationMetrics


class Visualiser:
    """Creates annotated images and console summaries."""

    def __init__(
        self,
        output_dir: str = OUTPUT_DIR,
        show: bool = SHOW_VISUALISATIONS,
        save: bool = SAVE_VISUALISATIONS,
    ):
        self.output_dir = output_dir
        self.show = show
        self.save = save

        os.makedirs(self.output_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def render(
        self,
        image_bgr: np.ndarray,
        pipeline_res: PipelineResult,
        measurements: GeometricMeasurements,
        eval_metrics: EvaluationMetrics | None = None,
        base_name: str = "result",
        calibration_valid: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create and save two visual outputs:

        1. Detection image.
        2. Measurement image.
        """
        detection_canvas = image_bgr.copy()

        per_edge_iou = (eval_metrics.per_edge_iou or []) if eval_metrics else []

        self._draw_qr_detections(detection_canvas, pipeline_res)
        self._draw_edge_detections(detection_canvas, pipeline_res, per_edge_iou)

        measurement_canvas = image_bgr.copy()

        self._draw_qr_detections(measurement_canvas, pipeline_res)
        self._draw_edge_detections(measurement_canvas, pipeline_res, per_edge_iou)
        self._draw_measurement_visuals(
            measurement_canvas,
            pipeline_res,
            measurements,
        )

        self.print_summary(pipeline_res, measurements, eval_metrics)

        if self.save:
            self._save_image(detection_canvas, base_name + "_detection")
            self._save_image(measurement_canvas, base_name + "_measurement")
            self._save_json(
                pipeline_res, measurements, eval_metrics, base_name,
                calibration_valid=calibration_valid,
            )
            self._save_measurements_csv(pipeline_res, measurements, base_name)

        if self.show:
            self._show_image(detection_canvas, base_name + "_detection")
            self._show_image(measurement_canvas, base_name + "_measurement")

        return detection_canvas, measurement_canvas

    def print_summary(
        self,
        pipeline_res: PipelineResult,
        measurements: GeometricMeasurements,
        eval_metrics: EvaluationMetrics | None = None,
    ) -> None:
        """Print a formatted summary to stdout."""
        sep = "=" * 60

        print(f"\n{sep}")
        print("  MICHELIN CHALLENGE 2 — Grounded-SAM Results")
        print(f"  Image : {pipeline_res.image_path}")
        print(sep)

        print(f"\n  QR codes detected : {len(pipeline_res.qr_detections)}")
        print(f"  Cut gaps detected : {len(pipeline_res.edge_detections)}")

        for ei, dists in enumerate(measurements.distances_to_bottom_mm):
            print(f"\n  Cut gap {ei + 1} — distances to table bottom (mm):")

            if len(dists) == 0:
                print("    (no sample points)")
                continue

            split = len(dists) // 2

            border_1 = dists[:split]
            border_2 = dists[split:]

            print("    Border 1 points:")
            for j, d in enumerate(border_1):
                print(f"      P{j + 1:2d}: {float(d):7.2f} mm")

            print("    Border 2 points:")
            for j, d in enumerate(border_2):
                print(f"      P{split + j + 1:2d}: {float(d):7.2f} mm")

            print(
                f"    Overall mean : {float(np.mean(dists)):.2f} mm | "
                f"Std: {float(np.std(dists)):.2f} mm | "
                f"Range: [{float(np.min(dists)):.2f}, "
                f"{float(np.max(dists)):.2f}] mm"
            )

            if len(border_1) > 0:
                print(
                    f"    Border 1 mean: {float(np.mean(border_1)):.2f} mm"
                )

            if len(border_2) > 0:
                print(
                    f"    Border 2 mean: {float(np.mean(border_2)):.2f} mm"
                )

        for k, gap in enumerate(measurements.inter_edge_distances_mm):
            print(f"\n  Inter-edge gap {k + 1}–{k + 2}: {float(gap):.2f} mm")

        if eval_metrics is not None:
            print("\n  Evaluation metrics:")

            if eval_metrics.rmse_mm is not None:
                print(f"    RMSE : {float(eval_metrics.rmse_mm):.3f} mm")

            if eval_metrics.mae_mm is not None:
                print(f"    MAE  : {float(eval_metrics.mae_mm):.3f} mm")

            if eval_metrics.iou is not None:
                print(f"    IoU  : {float(eval_metrics.iou):.4f}")

            if eval_metrics.f1 is not None:
                precision = (
                    None
                    if eval_metrics.precision is None
                    else float(eval_metrics.precision)
                )

                recall = (
                    None
                    if eval_metrics.recall is None
                    else float(eval_metrics.recall)
                )

                print(
                    f"    F1   : {float(eval_metrics.f1):.4f} "
                    f"(P={precision}, R={recall})"
                )

        print(f"\n{sep}\n")

    # ── Drawing helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _draw_qr_detections(
        canvas: np.ndarray,
        pipeline_res: PipelineResult,
    ) -> None:
        for det in pipeline_res.qr_detections:
            x1, y1, x2, y2 = det.box_xyxy.astype(int)

            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2, y2),
                COLOR_QR_BOX,
                2,
            )

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cv2.drawMarker(
                canvas,
                (cx, cy),
                COLOR_QR_BOX,
                cv2.MARKER_CROSS,
                15,
                2,
            )

            label = f"QR (conf:{det.score:.2f})"

            (lw, lh), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                1,
            )

            tx = max(0, min(x1, canvas.shape[1] - lw - 2))
            ty = max(lh + 2, y1 - 4)

            cv2.putText(
                canvas,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                COLOR_QR_BOX,
                1,
                cv2.LINE_AA,
            )

    def _draw_edge_detections(
        self,
        canvas: np.ndarray,
        pipeline_res: PipelineResult,
        per_edge_iou: list[float] | None = None,
    ) -> None:
        """Draw cut-gap masks and boxes."""
        overlay = canvas.copy()
        has_mask = False

        for det in pipeline_res.edge_detections:
            if det.mask is not None:
                overlay[det.mask] = COLOR_EDGE_MASK
                has_mask = True

        if has_mask:
            cv2.addWeighted(
                overlay,
                MASK_ALPHA,
                canvas,
                1 - MASK_ALPHA,
                0,
                canvas,
            )

        for i, det in enumerate(pipeline_res.edge_detections):
            x1, y1, x2, y2 = det.box_xyxy.astype(int)

            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2, y2),
                COLOR_EDGE_BOX,
                2,
            )

            iou_val = per_edge_iou[i] if per_edge_iou and i < len(per_edge_iou) else None
            if iou_val is not None:
                label = f"Cut gap  conf:{det.score:.2f}  IoU:{iou_val:.2f}"
            else:
                label = f"Cut gap (conf:{det.score:.2f})"

            (lw, lh), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                1,
            )

            tx = max(0, min(x1, canvas.shape[1] - lw - 2))
            ty = max(lh + 2, y1 - 4)

            cv2.putText(
                canvas,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                COLOR_EDGE_BOX,
                1,
                cv2.LINE_AA,
            )

    def _draw_measurement_visuals(
        self,
        canvas: np.ndarray,
        pipeline_res: PipelineResult,
        measurements: GeometricMeasurements,
    ) -> None:
        """Draw sampled points and distance lines.

        P1-P5  correspond to one border of the cut gap.
        P6-P10 correspond to the opposite border.
        """
        H, W = canvas.shape[:2]
        bottom_y = H - 1

        cv2.line(
            canvas,
            (0, bottom_y),
            (W - 1, bottom_y),
            COLOR_DISTANCE,
            2,
        )

        cv2.putText(
            canvas,
            "Reference bottom edge",
            (10, max(20, bottom_y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_DISTANCE,
            2,
            cv2.LINE_AA,
        )

        for edge_idx, (pts_px, dists_mm) in enumerate(
            zip(
                pipeline_res.edge_sample_points,
                measurements.distances_to_bottom_mm,
            )
        ):
            if len(pts_px) == 0 or len(dists_mm) == 0:
                continue

            n = min(len(pts_px), len(dists_mm))

            pts_px = np.asarray(pts_px[:n], dtype=np.float32)
            dists_mm = np.asarray(dists_mm[:n], dtype=np.float32)

            split = n // 2

            border_1 = pts_px[:split]
            border_2 = pts_px[split:]

            if len(border_1) >= 2:
                cv2.polylines(
                    canvas,
                    [border_1.astype(np.int32)],
                    isClosed=False,
                    color=COLOR_SAMPLE_PT,
                    thickness=2,
                )

            if len(border_2) >= 2:
                cv2.polylines(
                    canvas,
                    [border_2.astype(np.int32)],
                    isClosed=False,
                    color=COLOR_SAMPLE_PT,
                    thickness=2,
                )

            label_indices = set()

            if n >= 10:
                label_indices = {0, 4, 5, 9}
            elif n > 0:
                label_indices = {0, split, n - 1}

            for j, (pt, d_mm) in enumerate(zip(pts_px, dists_mm)):
                px = int(round(float(pt[0])))
                py = int(round(float(pt[1])))

                cv2.circle(
                    canvas,
                    (px, py),
                    4,
                    COLOR_SAMPLE_PT,
                    -1,
                )

                cv2.line(
                    canvas,
                    (px, py),
                    (px, bottom_y),
                    COLOR_DISTANCE,
                    1,
                )

                if j in label_indices:
                    if j < split:
                        border_name = "B1"
                        border_point_id = j + 1
                    else:
                        border_name = "B2"
                        border_point_id = j - split + 1

                    label = (
                        f"{border_name}-P{border_point_id}: "
                        f"{float(d_mm):.1f} mm"
                    )

                    text_x = px + 8

                    if text_x > W - 170:
                        text_x = max(5, px - 165)

                    text_y = py - 8 - (j % 3) * 15
                    text_y = max(18, min(H - 15, text_y))

                    cv2.putText(
                        canvas,
                        label,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        COLOR_DISTANCE,
                        1,
                        cv2.LINE_AA,
                    )

            if len(dists_mm) > 0:
                mean_d = float(np.mean(dists_mm))
                min_d = float(np.min(dists_mm))
                max_d = float(np.max(dists_mm))

                if split > 0:
                    b1_mean = float(np.mean(dists_mm[:split]))
                else:
                    b1_mean = mean_d

                if split < n:
                    b2_mean = float(np.mean(dists_mm[split:]))
                else:
                    b2_mean = mean_d

                if edge_idx < len(pipeline_res.edge_detections):
                    x1, y1, x2, y2 = (
                        pipeline_res.edge_detections[edge_idx]
                        .box_xyxy
                        .astype(int)
                    )

                    summary_x = max(5, min(x1, W - 340))
                    summary_y = max(22, min(y2 + 22, H - 25))
                else:
                    summary_x = 10
                    summary_y = 30 + edge_idx * 25

                summary = (
                    f"Gap {edge_idx + 1}: "
                    f"B1={b1_mean:.1f} mm | "
                    f"B2={b2_mean:.1f} mm | "
                    f"all={mean_d:.1f} [{min_d:.1f}, {max_d:.1f}]"
                )

                cv2.putText(
                    canvas,
                    summary,
                    (summary_x, summary_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    COLOR_DISTANCE,
                    1,
                    cv2.LINE_AA,
                )

    # ── File I/O ──────────────────────────────────────────────────────────────

    def _save_image(self, canvas: np.ndarray, base_name: str) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"{base_name}_{ts}.png")

        cv2.imwrite(path, canvas)

        print(f"[Visualiser] Saved annotated image → {path}")

    def _save_json(
        self,
        pipeline_res: PipelineResult,
        measurements: GeometricMeasurements,
        eval_metrics: EvaluationMetrics | None,
        base_name: str,
        calibration_valid: bool = True,
    ) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"{base_name}_{ts}.json")

        data: dict = {
            "image_path":        pipeline_res.image_path,
            "calibration_valid": calibration_valid,
            "n_qr_detected":     len(pipeline_res.qr_detections),
            "n_cut_gaps_detected": len(pipeline_res.edge_detections),
            "edges": [],
        }

        n_measured = len(measurements.distances_to_bottom_mm)
        n_detected = len(pipeline_res.edge_detections)

        for ei in range(n_detected):
            dists = (
                measurements.distances_to_bottom_mm[ei]
                if ei < n_measured
                else np.empty(0, dtype=np.float32)
            )
            pts_px = (
                pipeline_res.edge_sample_points[ei]
                if ei < len(pipeline_res.edge_sample_points)
                else np.empty((0, 2), dtype=np.float32)
            )

            pts_mm = (
                measurements.edge_sample_points_mm[ei]
                if ei < len(measurements.edge_sample_points_mm)
                else np.empty((0, 2), dtype=np.float32)
            )

            pts_px = np.asarray(pts_px, dtype=np.float32)
            pts_mm = np.asarray(pts_mm, dtype=np.float32)
            dists  = np.asarray(dists,  dtype=np.float32)

            # n_mm: how many points have full mm data (may be 0 if calibration failed)
            n_mm  = min(len(pts_mm), len(dists)) if len(pts_mm) > 0 and len(dists) > 0 else 0
            n     = min(len(pts_px), n_mm)

            # Keep all pixel points regardless of calibration
            pts_px_all = pts_px
            pts_px     = pts_px[:n]
            pts_mm     = pts_mm[:n]
            dists      = dists[:n]

            split = n // 2

            if len(dists):
                summary = {
                    "mean_mm": float(np.mean(dists)),
                    "std_mm": float(np.std(dists)),
                    "min_mm": float(np.min(dists)),
                    "max_mm": float(np.max(dists)),
                }
            else:
                summary = {
                    "mean_mm": None,
                    "std_mm": None,
                    "min_mm": None,
                    "max_mm": None,
                }

            if split > 0:
                border_1_summary = {
                    "mean_mm": float(np.mean(dists[:split])),
                    "std_mm": float(np.std(dists[:split])),
                    "min_mm": float(np.min(dists[:split])),
                    "max_mm": float(np.max(dists[:split])),
                }
            else:
                border_1_summary = None

            if split < n:
                border_2_summary = {
                    "mean_mm": float(np.mean(dists[split:])),
                    "std_mm": float(np.std(dists[split:])),
                    "min_mm": float(np.min(dists[split:])),
                    "max_mm": float(np.max(dists[split:])),
                }
            else:
                border_2_summary = None

            split_all = len(pts_px_all) // 2
            edge_data = {
                "edge_id": ei + 1,
                "calibration_available": n > 0,
                "interpretation": (
                    "Detected region is treated as the cut gap between two "
                    "rubber bands. Points P1-P5 belong to one border and "
                    "points P6-P10 belong to the opposite border."
                ),
                "n_sample_points": int(len(pts_px_all)),
                "sample_points_px": pts_px_all.tolist() if len(pts_px_all) else [],
                "sample_points_mm": pts_mm.tolist() if len(pts_mm) else [],
                "distances_to_bottom_mm": (
                    [float(x) for x in dists] if len(dists) else []
                ),
                "border_points_px": {
                    "border_1": pts_px_all[:split_all].tolist() if split_all > 0 else [],
                    "border_2": pts_px_all[split_all:].tolist() if split_all < len(pts_px_all) else [],
                },
                "border_points_mm": {
                    "border_1": pts_mm[:split].tolist() if split > 0 else [],
                    "border_2": pts_mm[split:].tolist() if split < n else [],
                },
                "border_distances_to_bottom_mm": {
                    "border_1": (
                        [float(x) for x in dists[:split]]
                        if split > 0
                        else []
                    ),
                    "border_2": (
                        [float(x) for x in dists[split:]]
                        if split < n
                        else []
                    ),
                },
                "distance_summary_mm": summary,
                "border_summary_mm": {
                    "border_1": border_1_summary,
                    "border_2": border_2_summary,
                },
            }

            data["edges"].append(edge_data)

        data["inter_edge_distances_mm"] = [
            float(x) for x in measurements.inter_edge_distances_mm
        ]

        if eval_metrics is not None:
            data["evaluation"] = {
                "rmse_mm": (
                    None
                    if eval_metrics.rmse_mm is None
                    else float(eval_metrics.rmse_mm)
                ),
                "mae_mm": (
                    None
                    if eval_metrics.mae_mm is None
                    else float(eval_metrics.mae_mm)
                ),
                "iou": (
                    None
                    if eval_metrics.iou is None
                    else float(eval_metrics.iou)
                ),
                "precision": (
                    None
                    if eval_metrics.precision is None
                    else float(eval_metrics.precision)
                ),
                "recall": (
                    None
                    if eval_metrics.recall is None
                    else float(eval_metrics.recall)
                ),
                "f1": (
                    None
                    if eval_metrics.f1 is None
                    else float(eval_metrics.f1)
                ),
            }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"[Visualiser] Saved JSON results    → {path}")

    def _save_measurements_csv(
        self,
        pipeline_res: PipelineResult,
        measurements: GeometricMeasurements,
        base_name: str,
    ) -> None:
        """Save one CSV row per sampled point."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(
            self.output_dir,
            f"{base_name}_measurements_{ts}.csv",
        )

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow(
                [
                    "image_path",
                    "edge_id",
                    "border_id",
                    "border_point_id",
                    "global_point_id",
                    "x_px",
                    "y_px",
                    "x_mm",
                    "y_mm",
                    "distance_to_bottom_mm",
                ]
            )

            for edge_idx, dists_mm in enumerate(
                measurements.distances_to_bottom_mm
            ):
                pts_px = (
                    pipeline_res.edge_sample_points[edge_idx]
                    if edge_idx < len(pipeline_res.edge_sample_points)
                    else np.empty((0, 2), dtype=np.float32)
                )

                pts_mm = (
                    measurements.edge_sample_points_mm[edge_idx]
                    if edge_idx < len(measurements.edge_sample_points_mm)
                    else np.empty((0, 2), dtype=np.float32)
                )

                pts_px = np.asarray(pts_px, dtype=np.float32)
                pts_mm = np.asarray(pts_mm, dtype=np.float32)
                dists_mm = np.asarray(dists_mm, dtype=np.float32)

                n = min(len(pts_px), len(pts_mm), len(dists_mm))
                split = n // 2

                for j in range(n):
                    if j < split:
                        border_id = 1
                        border_point_id = j + 1
                    else:
                        border_id = 2
                        border_point_id = j - split + 1

                    writer.writerow(
                        [
                            pipeline_res.image_path,
                            edge_idx + 1,
                            border_id,
                            border_point_id,
                            j + 1,
                            float(pts_px[j][0]),
                            float(pts_px[j][1]),
                            float(pts_mm[j][0]),
                            float(pts_mm[j][1]),
                            float(dists_mm[j]),
                        ]
                    )

        print(f"[Visualiser] Saved CSV measurements → {path}")

    @staticmethod
    def _show_image(canvas: np.ndarray, title: str) -> None:
        cv2.imshow(title, canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
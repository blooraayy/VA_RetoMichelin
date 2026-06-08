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
        per_qr_iou   = (eval_metrics.per_qr_iou   or []) if eval_metrics else []

        self._draw_qr_detections(detection_canvas, pipeline_res, per_qr_iou)
        self._draw_edge_detections(detection_canvas, pipeline_res, per_edge_iou)

        measurement_canvas = image_bgr.copy()

        self._draw_qr_detections(measurement_canvas, pipeline_res, per_qr_iou)
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
        per_qr_iou: list[float] | None = None,
    ) -> None:
        for i, det in enumerate(pipeline_res.qr_detections):
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

            iou_val = per_qr_iou[i] if per_qr_iou and i < len(per_qr_iou) else None
            if iou_val is not None:
                label = f"QR  conf:{det.score:.2f}  IoU:{iou_val:.2f}"
            else:
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
        if pipeline_res.qr_detections:
            bottom_y = int(max(det.box_xyxy[3] for det in pipeline_res.qr_detections))
        else:
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
            "qr_boxes":          [[float(v) for v in det.box_xyxy]
                                  for det in pipeline_res.qr_detections],
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
                "per_qr_iou": [float(v) for v in (eval_metrics.per_qr_iou or [])],
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

    # ── Reto visualisation ────────────────────────────────────────────────────

    def render_reto(
        self,
        image_bgr: np.ndarray,
        pipeline_res: PipelineResult,
        band_a,
        band_b,
        cut_a,
        cut_b,
        reto1_rows: list,
        reto2_rows: list,
        homography,
        qr_calibrator,
        base_name: str = "result",
    ) -> np.ndarray:
        """Draw SAM segmentation masks, bounding boxes, cut overlays, QR markers
        and Reto 2 measurement lines matching the nuevas_normas reference style."""
        canvas = image_bgr.copy()
        H_img, W = canvas.shape[:2]

        # Colors (BGR)
        COLOR_BAND_A = (0, 220, 220)    # yellow
        COLOR_BAND_B = (200, 200, 0)    # cyan
        COLOR_CUT    = (50, 50, 220)    # red
        BLACK        = (0, 0, 0)

        ALPHA_BAND = 0.18   # band SAM mask fill
        ALPHA_CUT  = 0.42   # cut SAM mask fill

        # ── Band SAM mask overlay ─────────────────────────────────────────────
        band_overlay = canvas.copy()
        for det, color in [(band_a, COLOR_BAND_A), (band_b, COLOR_BAND_B)]:
            if det is not None and det.mask is not None:
                band_overlay[det.mask] = color
        cv2.addWeighted(band_overlay, ALPHA_BAND, canvas, 1 - ALPHA_BAND, 0, canvas)

        # ── Band bounding boxes + labels ──────────────────────────────────────
        for det, color, label in [
            (band_a, COLOR_BAND_A, "Banda A"),
            (band_b, COLOR_BAND_B, "Banda B"),
        ]:
            if det is None:
                continue
            x1, y1, x2, y2 = det.box_xyxy.astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
            tx = max(0, min(x1 + 4, W - lw - 4))
            ty = max(lh + 4, y1 + lh + 4)
            cv2.putText(canvas, label, (tx + 1, ty + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, BLACK, 2, cv2.LINE_AA)
            cv2.putText(canvas, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)

        # ── Cut SAM mask overlay ──────────────────────────────────────────────
        cut_overlay = canvas.copy()
        for det in [cut_a, cut_b]:
            if det is not None and det.mask is not None:
                cut_overlay[det.mask] = COLOR_CUT
        cv2.addWeighted(cut_overlay, ALPHA_CUT, canvas, 1 - ALPHA_CUT, 0, canvas)

        # ── Cut bounding boxes + labels ───────────────────────────────────────
        for det, band_det, label in [
            (cut_a, band_a, "Corte A"),
            (cut_b, band_b, "Corte B"),
        ]:
            if det is None:
                continue
            x1, y1, x2, y2 = det.box_xyxy.astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_CUT, 2)
            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            tx = max(0, min(x1 + 2, W - lw - 4))
            ty = max(lh + 2, y1 - 4)
            cv2.putText(canvas, label, (tx + 1, ty + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, BLACK, 2, cv2.LINE_AA)
            cv2.putText(canvas, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_CUT, 1, cv2.LINE_AA)

        # ── QR detections ─────────────────────────────────────────────────────
        self._draw_qr_detections(canvas, pipeline_res)

        # ── Reto 2 measurement lines ──────────────────────────────────────────
        if (
            homography is not None
            and qr_calibrator is not None
            and reto2_rows
        ):
            self._draw_reto2_visuals(
                canvas, reto2_rows, band_a, band_b, cut_a, cut_b,
                homography, qr_calibrator, H_img, W,
            )

        # ── Save / show ───────────────────────────────────────────────────────
        if self.save:
            self._save_image(canvas, base_name + "_reto")
        if self.show:
            self._show_image(canvas, base_name + "_reto")

        return canvas

    def _draw_reto2_visuals(
        self,
        canvas: np.ndarray,
        reto2_rows: list,
        band_a,
        band_b,
        cut_a,
        cut_b,
        homography: np.ndarray,
        qr_calibrator,
        H_img: int,
        W: int,
    ) -> None:
        """Draw 10 measurement lines, SA1 segments and reference spine.

        Visual design:
        - Brown vertical spine from table (Y=0) to last measurement (YSA+235mm),
          at the X centre of the cut.  Tick marks every 25 mm from cut top.
        - (0) circle at cut top + YSA label.
        - 10 thin horizontal lines across the full band width (context grid).
        - White SA1 segment with circles at actual gap edges A and B.
        - A/B labels on line 1; SA1 value on lines 1, 5, 10.
        """
        YELLOW = (0, 220, 220)
        CYAN   = (200, 200, 0)
        WHITE  = (255, 255, 255)
        BLACK  = (0, 0, 0)
        BROWN  = (19, 69, 139)

        ANNOTATE_AT = {0, 4, 9}

        first = reto2_rows[0]
        ysa_val = getattr(first, "YSA", None)
        ysb_val = getattr(first, "YSB", None)

        bands = [
            (band_a, cut_a, ysa_val, YELLOW, "SA1", "YSA", "SA1_xa_mm", "SA1_xb_mm"),
            (band_b, cut_b, ysb_val, CYAN,   "SB1", "YSB", "SB1_xa_mm", "SB1_xb_mm"),
        ]

        for band_det, cut_det, ys_val, col, sa1_key, ys_key, xa_key, xb_key in bands:
            if band_det is None or ys_val is None:
                continue

            # ── Band X range from bounding box ────────────────────────────────
            bx1, by1, bx2, by2 = band_det.box_xyxy
            corners_px = np.array([
                [bx1, by1], [bx2, by1], [bx1, by2], [bx2, by2],
            ], dtype=np.float32)
            corners_mm = qr_calibrator.pixels_to_mm_bulk(corners_px, homography)
            x_lo = float(corners_mm[:, 0].min()) + 5.0
            x_hi = float(corners_mm[:, 0].max()) - 5.0

            # ── Cut centre X in mm ─────────────────────────────────────────────
            cut_cx_mm = None
            if cut_det is not None:
                cut_cx_px = float((cut_det.box_xyxy[0] + cut_det.box_xyxy[2]) / 2.0)
                cut_cy_px = float((cut_det.box_xyxy[1] + cut_det.box_xyxy[3]) / 2.0)
                cut_pts_mm = qr_calibrator.pixels_to_mm_bulk(
                    np.array([[cut_cx_px, cut_cy_px]], dtype=np.float32), homography
                )
                cut_cx_mm = float(cut_pts_mm[0, 0])

            # Determine spine X: prefer cut centre, else band midpoint
            if cut_cx_mm is not None:
                ref_x_mm = cut_cx_mm
            else:
                # Try to use actual gap edges from first row
                first_row = reto2_rows[0]
                xa0 = getattr(first_row, xa_key, None)
                xb0 = getattr(first_row, xb_key, None)
                if xa0 is not None and xb0 is not None:
                    ref_x_mm = (xa0 + xb0) / 2.0
                else:
                    ref_x_mm = (x_lo + x_hi) / 2.0

            # ── Vertical reference spine: Y=0 → Y=ys_val+235 ─────────────────
            spine_end_mm = ys_val + 235.0
            spine_pts_mm = np.array(
                [[ref_x_mm, 0.0], [ref_x_mm, spine_end_mm]], dtype=np.float32
            )
            spine_pts_px = qr_calibrator.mm_to_pixels_bulk(spine_pts_mm, homography)
            p_table  = (int(round(float(spine_pts_px[0, 0]))),
                        int(round(float(spine_pts_px[0, 1]))))
            p_bottom = (int(round(float(spine_pts_px[1, 0]))),
                        int(round(float(spine_pts_px[1, 1]))))
            if 0 <= p_table[1] < H_img:
                cv2.line(canvas, p_table, p_bottom, BROWN, 1)

            # ── (0) marker at cut top ─────────────────────────────────────────
            origin_pts_mm = np.array([[ref_x_mm, ys_val]], dtype=np.float32)
            origin_pts_px = qr_calibrator.mm_to_pixels_bulk(origin_pts_mm, homography)
            p_origin = (int(round(float(origin_pts_px[0, 0]))),
                        int(round(float(origin_pts_px[0, 1]))))
            if 0 <= p_origin[1] < H_img:
                cv2.circle(canvas, p_origin, 6, BROWN, -1)
                cv2.circle(canvas, p_origin, 6, BLACK, 1)
                lx = min(max(p_origin[0] + 9, 0), W - 1)
                ly = min(max(p_origin[1] + 5, 10), H_img - 1)
                cv2.putText(canvas, "(0)",
                            (lx + 1, ly + 1), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, BLACK, 2, cv2.LINE_AA)
                cv2.putText(canvas, "(0)",
                            (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, BROWN, 1, cv2.LINE_AA)
                ys_label = f"{ys_key}={ys_val:.1f}mm"
                ly2 = min(ly + 16, H_img - 1)
                cv2.putText(canvas, ys_label, (lx + 1, ly2 + 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, BLACK, 2, cv2.LINE_AA)
                cv2.putText(canvas, ys_label, (lx, ly2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, BROWN, 1, cv2.LINE_AA)

            # ── 10 measurement lines ──────────────────────────────────────────
            for line_num, row in enumerate(reto2_rows):
                y_mm    = ys_val + row.y_offset_mm
                sa1_val = getattr(row, sa1_key, None)
                xa_val  = getattr(row, xa_key,  None)
                xb_val  = getattr(row, xb_key,  None)

                # Thin full-band horizontal line (context)
                line_pts_mm = np.array(
                    [[x_lo, y_mm], [x_hi, y_mm]], dtype=np.float32
                )
                line_pts_px = qr_calibrator.mm_to_pixels_bulk(line_pts_mm, homography)
                p1 = (int(round(float(line_pts_px[0, 0]))),
                      int(round(float(line_pts_px[0, 1]))))
                p2 = (int(round(float(line_pts_px[1, 0]))),
                      int(round(float(line_pts_px[1, 1]))))
                if not (0 <= p1[1] < H_img and 0 <= p2[1] < H_img):
                    continue
                cv2.line(canvas, p1, p2, col, 1)

                # Y-offset label in white at the left end of the line
                num_txt = str(int(row.y_offset_mm))
                nx = max(0, p1[0] + 4)
                ny = max(10, p1[1] - 3)
                cv2.putText(canvas, num_txt, (nx + 1, ny + 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(canvas, num_txt, (nx, ny),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, WHITE, 1, cv2.LINE_AA)

                # Tick mark on the vertical spine (±5mm)
                tick_pts_mm = np.array(
                    [[ref_x_mm - 5.0, y_mm], [ref_x_mm + 5.0, y_mm]], dtype=np.float32
                )
                tick_pts_px = qr_calibrator.mm_to_pixels_bulk(tick_pts_mm, homography)
                t1 = (int(round(float(tick_pts_px[0, 0]))),
                      int(round(float(tick_pts_px[0, 1]))))
                t2 = (int(round(float(tick_pts_px[1, 0]))),
                      int(round(float(tick_pts_px[1, 1]))))
                cv2.line(canvas, t1, t2, BROWN, 2)

                # SA1 segment: use actual gap edges, fallback to symmetric
                if sa1_val is not None and sa1_val > 0.5:
                    if xa_val is not None and xb_val is not None:
                        gap_pts_mm = np.array(
                            [[xa_val, y_mm], [xb_val, y_mm]], dtype=np.float32
                        )
                    elif cut_cx_mm is not None:
                        half = sa1_val / 2.0
                        gap_pts_mm = np.array(
                            [[cut_cx_mm - half, y_mm], [cut_cx_mm + half, y_mm]],
                            dtype=np.float32
                        )
                    else:
                        gap_pts_mm = None

                    if gap_pts_mm is not None:
                        gap_pts_px = qr_calibrator.mm_to_pixels_bulk(
                            gap_pts_mm, homography
                        )
                        gp1 = (int(round(float(gap_pts_px[0, 0]))),
                               int(round(float(gap_pts_px[0, 1]))))
                        gp2 = (int(round(float(gap_pts_px[1, 0]))),
                               int(round(float(gap_pts_px[1, 1]))))
                        cv2.line(canvas, gp1, gp2, WHITE, 2)
                        cv2.circle(canvas, gp1, 5, WHITE, -1)
                        cv2.circle(canvas, gp1, 5, BLACK, 1)
                        cv2.circle(canvas, gp2, 5, WHITE, -1)
                        cv2.circle(canvas, gp2, 5, BLACK, 1)
                        if line_num == 0:
                            cv2.putText(canvas, "A",
                                        (max(0, gp1[0] - 12), max(10, gp1[1] - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, WHITE, 2, cv2.LINE_AA)
                            cv2.putText(canvas, "B",
                                        (min(W - 12, gp2[0] + 4), max(10, gp2[1] - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, WHITE, 2, cv2.LINE_AA)

                # SA1 value annotation at lines 1, 5, 10 (right side)
                if line_num not in ANNOTATE_AT:
                    continue
                if sa1_val is not None:
                    txt = f"{sa1_key}={sa1_val:.1f}"
                    ann_x = min(max(p2[0] + 5, 0), W - 120)
                    ann_y = min(max(p2[1] + 4, 10), H_img - 1)
                    cv2.putText(canvas, txt, (ann_x + 1, ann_y + 1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.30, BLACK, 2, cv2.LINE_AA)
                    cv2.putText(canvas, txt, (ann_x, ann_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.30, WHITE, 1, cv2.LINE_AA)

    @staticmethod
    def _show_image(canvas: np.ndarray, title: str) -> None:
        cv2.imshow(title, canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
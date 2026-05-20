# =============================================================================
# controller/pipeline_controller.py
# Orchestrates the Michelin Challenge 2 Grounded-SAM pipeline.
#
# CHANGE LOG (May 2026)
# ---------------------
# QR calibration is now performed AFTER Grounded-SAM detection, so we can
# reuse the DINO QR-marker boxes (the cv2.QRCodeDetector cannot read the
# pink stickers used in this dataset).
# =============================================================================

from __future__ import annotations

import os
import sys
import argparse
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    IMAGE_DIR, OUTPUT_DIR, DEVICE,
    GDINO_CONFIG, GDINO_WEIGHTS,
    SAM_CHECKPOINT, SAM_MODEL_TYPE,
    SAVE_VISUALISATIONS, SHOW_VISUALISATIONS,
    COCO_ANNOTATIONS,
)
from model.coco_loader   import COCOLoader
from model.qr_calibrator import QRCalibrator
from model.grounded_sam  import GroundedSAMModel
from model.metrics       import MeasurementEngine, MetricEvaluator
from view.visualiser     import Visualiser
from utils.utils         import load_image


class PipelineController:
    """Orchestrates the full Michelin Challenge 2 pipeline."""

    def __init__(
        self,
        device:          str  = DEVICE,
        gdino_config:    str  = GDINO_CONFIG,
        gdino_weights:   str  = GDINO_WEIGHTS,
        sam_checkpoint:  str  = SAM_CHECKPOINT,
        sam_model_type:  str  = SAM_MODEL_TYPE,
        output_dir:      str  = OUTPUT_DIR,
        show:            bool = SHOW_VISUALISATIONS,
        save:            bool = SAVE_VISUALISATIONS,
    ):
        # Model components
        self.qr_calibrator = QRCalibrator()
        self.gsam_model    = GroundedSAMModel(
            device=device,
            gdino_config=gdino_config,
            gdino_weights=gdino_weights,
            sam_checkpoint=sam_checkpoint,
            sam_model_type=sam_model_type,
        )
        self.metric_evaluator = MetricEvaluator()

        self.coco_loader = None
        if os.path.isfile(COCO_ANNOTATIONS):
            self.coco_loader = COCOLoader(
                coco_json=COCO_ANNOTATIONS, image_dir=IMAGE_DIR,
            )
        else:
            print(f"[Controller] WARNING — COCO annotations not found: "
                  f"{COCO_ANNOTATIONS}")

        # View
        self.visualiser = Visualiser(output_dir=output_dir, show=show, save=save)

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ── Single-image processing ──────────────────────────────────────────────

    def run_image(self, image_path: str, gt_label_path: str | None = None) -> dict:
        print(f"\n[Controller] Processing: {image_path}")
        t0 = time.perf_counter()

        image_bgr = load_image(image_path)
        H, W      = image_bgr.shape[:2]

        # 1. Grounded-SAM (detects QRs, strips and cuts in one go) ──────────
        pipeline_result = self.gsam_model.run(image_bgr, image_path=image_path)

        # 2. QR calibration FROM DINO DETECTIONS ────────────────────────────
        qr_result = self.qr_calibrator.calibrate_from_detections(
            qr_detections=pipeline_result.qr_detections,
            image_shape=(H, W),
        )
        if not qr_result.success:
            print(f"[Controller] WARNING — QR calibration failed: "
                  f"{qr_result.message}")
            print("[Controller] Proceeding without metric conversion.")
        else:
            print(f"[Controller] {qr_result.message}  "
                  f"(scale ≈ {qr_result.px_per_mm:.3f} px/mm)")

        # 3. Geometric measurements ─────────────────────────────────────────
        if qr_result.success:
            engine       = MeasurementEngine(self.qr_calibrator, H)
            measurements = engine.compute(pipeline_result, qr_result.homography)
            pipeline_result.edge_sample_points_mm  = measurements.edge_sample_points_mm
            pipeline_result.distances_to_bottom_mm = measurements.distances_to_bottom_mm
        else:
            from model.metrics import GeometricMeasurements
            measurements = GeometricMeasurements()

        # 4. Evaluation against COCO GT ─────────────────────────────────────
        eval_metrics = None
        gt = None
        if self.coco_loader is not None:
            gt = self.coco_loader.get_by_filename(image_path)

        if gt is None:
            print(f"[Controller] WARNING — No COCO GT found for {image_path}")
        else:
            eval_metrics = self.metric_evaluator.evaluate_against_gt(
                pipeline_result=pipeline_result,
                measurements=measurements,
                gt=gt,
                homography=qr_result.homography if qr_result.success else None,
                qr_calibrator=self.qr_calibrator if qr_result.success else None,
            )

        # 5. Visualise ──────────────────────────────────────────────────────
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        self.visualiser.render(
                    image_bgr=image_bgr,
                    pipeline_res=pipeline_result,
                    measurements=measurements,
                    eval_metrics=eval_metrics,
                    base_name=base_name,
                    calibration_valid=qr_result.success,
        )

        elapsed = time.perf_counter() - t0
        print(f"[Controller] Done in {elapsed:.2f}s")

        return self._build_summary_dict(
            image_path, pipeline_result, measurements, eval_metrics, elapsed,
            calibration_valid=qr_result.success,
        )

    # ── Batch processing (unchanged) ─────────────────────────────────────────

    def run_folder(
        self,
        folder_path:    str,
        label_folder:   str | None = None,
        extensions:     tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ) -> list[dict]:
        image_paths = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in extensions
        ])
        if not image_paths:
            print(f"[Controller] No images found in {folder_path}")
            return []
        print(f"[Controller] Found {len(image_paths)} image(s) in {folder_path}")

        results = []
        for img_path in image_paths:
            try:
                results.append(self.run_image(img_path))
            except Exception as exc:
                print(f"[Controller] ERROR on {img_path}: {exc}")

        self._save_aggregate_summary(results)
        return results

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_summary_dict(image_path, pipeline_result, measurements,
                             eval_metrics, elapsed_s: float,
                             calibration_valid: bool = True) -> dict:
        n_meas = len(measurements.distances_to_bottom_mm)
        edges = []
        for i, det in enumerate(pipeline_result.edge_detections):
            if i < n_meas and len(measurements.distances_to_bottom_mm[i]) > 0:
                dists = measurements.distances_to_bottom_mm[i].tolist()
            else:
                dists = []
            edges.append({"edge_id": i + 1, "distances_to_bottom_mm": dists})

        d = {
            "image_path":        image_path,
            "elapsed_s":         round(elapsed_s, 3),
            "calibration_valid": calibration_valid,
            "n_qr":              len(pipeline_result.qr_detections),
            "qr_boxes":          [[float(v) for v in det.box_xyxy]
                                  for det in pipeline_result.qr_detections],
            "n_edges":           len(pipeline_result.edge_detections),
            "edges":             edges,
            "inter_edge_distances_mm": measurements.inter_edge_distances_mm,
        }
        if eval_metrics:
            d["evaluation"] = {
                "rmse_mm":          eval_metrics.rmse_mm,
                "mae_mm":           eval_metrics.mae_mm,
                "iou":              eval_metrics.iou,
                "precision":        eval_metrics.precision,
                "recall":           eval_metrics.recall,
                "f1":               eval_metrics.f1,
                "per_edge_iou":     eval_metrics.per_edge_iou,
                "per_edge_rmse_mm": eval_metrics.per_edge_rmse_mm,
                "per_edge_mae_mm":  eval_metrics.per_edge_mae_mm,
                "per_qr_iou":       eval_metrics.per_qr_iou,
            }
        return d

    def _save_aggregate_summary(self, results: list[dict]) -> None:
        path = os.path.join(self.output_dir, "aggregate_summary.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Controller] Aggregate summary → {path}")


# ── CLI entry-point (unchanged) ──────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Michelin Challenge 2 — Grounded-SAM Pipeline"
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str,
                       help="Path to a single input image.")
    group.add_argument("--folder", type=str,
                       help="Path to a folder of input images.")

    p.add_argument("--labels",  type=str, default=None,
                   help="Path to GT label file (--image) or folder (--folder).")
    p.add_argument("--output",  type=str, default=OUTPUT_DIR)
    p.add_argument("--device",  type=str, default=DEVICE,
                   choices=["cuda", "cpu"])
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--gdino-config",   type=str, default=GDINO_CONFIG)
    p.add_argument("--gdino-weights",  type=str, default=GDINO_WEIGHTS)
    p.add_argument("--sam-checkpoint", type=str, default=SAM_CHECKPOINT)
    p.add_argument("--sam-model-type", type=str, default=SAM_MODEL_TYPE,
                   choices=["vit_h", "vit_l", "vit_b"])
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    controller = PipelineController(
        device         = args.device,
        gdino_config   = args.gdino_config,
        gdino_weights  = args.gdino_weights,
        sam_checkpoint = args.sam_checkpoint,
        sam_model_type = args.sam_model_type,
        output_dir     = args.output,
        show           = not args.no_show,
        save           = not args.no_save,
    )

    if args.image:
        controller.run_image(args.image, gt_label_path=args.labels)
    else:
        controller.run_folder(args.folder, label_folder=args.labels)


if __name__ == "__main__":
    main()
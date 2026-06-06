# =============================================================================
# infer_new_images.py
# Run the Grounded-SAM pipeline on new_images/ — no ground truth available.
#
# Processes every Pos*/  sub-folder found under --input, saves visualisations
# per position, and writes a summary CSV + JSON.
#
# Usage (from models/Pablo/):
#   python infer_new_images.py --device cuda --no-show
#   python infer_new_images.py --device cpu  --no-show --input new_images
# =============================================================================

from __future__ import annotations

import os
import sys
import csv
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import (
    DEVICE,
    GDINO_CONFIG, GDINO_WEIGHTS,
    SAM_CHECKPOINT, SAM_MODEL_TYPE,
)
from model.qr_calibrator import QRCalibrator
from model.grounded_sam  import GroundedSAMModel
from model.metrics       import MeasurementEngine, GeometricMeasurements
from view.visualiser     import Visualiser
from utils.utils         import load_image

EXTENSIONS = (".jpg", ".jpeg", ".png")

_BASE    = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_INPUT  = os.path.join(_BASE, "new_images")
_DEFAULT_OUTPUT = os.path.join(_BASE, "new_images", "outputs")


# ── Per-image inference ───────────────────────────────────────────────────────

def _process_image(
    image_path:    str,
    gsam_model:    GroundedSAMModel,
    qr_calibrator: QRCalibrator,
    visualiser:    Visualiser,
) -> dict:
    t0 = time.perf_counter()

    image_bgr = load_image(image_path)
    H, W      = image_bgr.shape[:2]

    # Stage 1-3: QR + strip + cut-edge detection
    pipeline_result = gsam_model.run(image_bgr, image_path=image_path)

    # QR calibration (pixel → mm)
    qr_result = qr_calibrator.calibrate_from_detections(
        qr_detections=pipeline_result.qr_detections,
        image_shape=(H, W),
    )
    if not qr_result.success:
        print(f"    [WARN] QR calibration failed: {qr_result.message}")

    # Geometric measurements (skipped when QR unavailable)
    if qr_result.success:
        engine       = MeasurementEngine(qr_calibrator, H)
        measurements = engine.compute(pipeline_result, qr_result.homography)
        pipeline_result.edge_sample_points_mm  = measurements.edge_sample_points_mm
        pipeline_result.distances_to_bottom_mm = measurements.distances_to_bottom_mm
    else:
        measurements = GeometricMeasurements()

    # Visualise (no eval_metrics — no ground truth)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    visualiser.render(
        image_bgr=image_bgr,
        pipeline_res=pipeline_result,
        measurements=measurements,
        eval_metrics=None,
        base_name=base_name,
        calibration_valid=qr_result.success,
    )

    elapsed = time.perf_counter() - t0

    # Serialise edge distances
    edges = []
    for i, det in enumerate(pipeline_result.edge_detections):
        if i < len(measurements.distances_to_bottom_mm):
            dists = measurements.distances_to_bottom_mm[i]
            dists_list = dists.tolist() if hasattr(dists, "tolist") else list(dists)
        else:
            dists_list = []
        edges.append({
            "edge_id":              i + 1,
            "score":                float(det.score),
            "distances_to_bottom_mm": dists_list,
            "mean_distance_mm":     round(sum(dists_list) / len(dists_list), 2)
                                    if dists_list else None,
        })

    return {
        "image":                   os.path.basename(image_path),
        "elapsed_s":               round(elapsed, 2),
        "qr_ok":                   qr_result.success,
        "n_qr":                    len(pipeline_result.qr_detections),
        "n_edges":                 len(pipeline_result.edge_detections),
        "edges":                   edges,
        "inter_edge_distances_mm": measurements.inter_edge_distances_mm,
    }


# ── CLI entry-point ───────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Michelin — Inference on new_images/ (no ground truth)"
    )
    p.add_argument("--input",   default=_DEFAULT_INPUT,
                   help="Root folder containing Pos* sub-folders.")
    p.add_argument("--output",  default=_DEFAULT_OUTPUT,
                   help="Root output folder (sub-folders created per position).")
    p.add_argument("--device",  default=DEVICE, choices=["cuda", "cpu"])
    p.add_argument("--no-show", action="store_true",
                   help="Do not display images (headless mode).")
    p.add_argument("--no-save", action="store_true",
                   help="Skip saving visualisation PNGs.")
    p.add_argument("--gdino-config",   default=GDINO_CONFIG)
    p.add_argument("--gdino-weights",  default=GDINO_WEIGHTS)
    p.add_argument("--sam-checkpoint", default=SAM_CHECKPOINT)
    p.add_argument("--sam-model-type", default=SAM_MODEL_TYPE,
                   choices=["vit_h", "vit_l", "vit_b"])
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load models once — reused across all positions
    print("[Inference] Loading models …")
    gsam_model = GroundedSAMModel(
        device         = args.device,
        gdino_config   = args.gdino_config,
        gdino_weights  = args.gdino_weights,
        sam_checkpoint = args.sam_checkpoint,
        sam_model_type = args.sam_model_type,
    )
    qr_calibrator = QRCalibrator()
    print("[Inference] Models ready.\n")

    # Discover position sub-folders
    pos_dirs = sorted([
        d for d in os.listdir(args.input)
        if os.path.isdir(os.path.join(args.input, d))
        and not d.startswith(".")
        and d != "outputs"
    ])

    if not pos_dirs:
        print(f"[Inference] No sub-folders found in {args.input}")
        return

    print(f"[Inference] Found positions: {', '.join(pos_dirs)}\n")

    all_results: list[dict] = []
    summary_rows: list[dict] = []
    t_global = time.perf_counter()

    for pos in pos_dirs:
        pos_input  = os.path.join(args.input, pos)
        pos_output = os.path.join(args.output, pos)
        os.makedirs(pos_output, exist_ok=True)

        images = sorted([
            os.path.join(pos_input, f)
            for f in os.listdir(pos_input)
            if os.path.splitext(f)[1].lower() in EXTENSIONS
        ])

        if not images:
            print(f"[{pos}] No images found — skipping.")
            continue

        print(f"[{pos}] {len(images)} image(s) → {pos_output}")

        # Fresh visualiser per position so outputs land in the right sub-folder
        vis = Visualiser(
            output_dir = pos_output,
            show       = not args.no_show,
            save       = not args.no_save,
        )

        for img_path in images:
            img_name = os.path.basename(img_path)
            try:
                result             = _process_image(img_path, gsam_model, qr_calibrator, vis)
                result["position"] = pos
                all_results.append(result)

                status = "OK    " if result["qr_ok"] else "NO-QR "
                mean1  = result["edges"][0]["mean_distance_mm"] if result["edges"] else "-"
                mean2  = result["edges"][1]["mean_distance_mm"] if len(result["edges"]) > 1 else "-"
                inter  = result["inter_edge_distances_mm"]
                print(f"  [{status}] {img_name:30s} "
                      f"edges={result['n_edges']}  "
                      f"d1={mean1}mm  d2={mean2}mm  "
                      f"inter={inter}mm  "
                      f"({result['elapsed_s']}s)")

                # Flat row for CSV
                summary_rows.append({
                    "position":              pos,
                    "image":                 img_name,
                    "elapsed_s":             result["elapsed_s"],
                    "qr_ok":                 result["qr_ok"],
                    "n_qr":                  result["n_qr"],
                    "n_edges":               result["n_edges"],
                    "mean_dist_edge1_mm":    mean1,
                    "mean_dist_edge2_mm":    mean2,
                    "inter_edge_mm":         inter,
                })

            except Exception as exc:
                print(f"  [ERROR] {img_name}: {exc}")
                import traceback
                traceback.print_exc()

        print()

    total_elapsed = time.perf_counter() - t_global

    # ── Save JSON summary ─────────────────────────────────────────────────────
    json_path = os.path.join(args.output, "inference_summary.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[Inference] JSON summary  → {json_path}")

    # ── Save CSV summary ──────────────────────────────────────────────────────
    csv_path = os.path.join(args.output, "inference_summary.csv")
    if summary_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[Inference] CSV  summary  → {csv_path}")

    # ── Final stats ───────────────────────────────────────────────────────────
    total     = len(summary_rows)
    qr_ok     = sum(1 for r in summary_rows if r["qr_ok"])
    edge_det  = sum(1 for r in summary_rows if r["n_edges"] > 0)

    print(f"\n{'='*60}")
    print(f"  Positions processed : {len(pos_dirs)}")
    print(f"  Images processed    : {total}")
    print(f"  QR calibration OK   : {qr_ok}/{total}  "
          f"({100*qr_ok//total if total else 0}%)")
    print(f"  Edge detected       : {edge_det}/{total}  "
          f"({100*edge_det//total if total else 0}%)")
    print(f"  Total time          : {total_elapsed:.1f}s  "
          f"(avg {total_elapsed/total:.1f}s/image)" if total else "")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

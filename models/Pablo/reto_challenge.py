# =============================================================================
# reto_challenge.py
# Competition-day script: process 1–3 images, output the Michelin xlsx table.
#
# Usage (from models/Pablo/):
#   python reto_challenge.py --images img1.jpg img2.jpg img3.jpg
#   python reto_challenge.py --images img1.jpg --output results.xlsx
#   python reto_challenge.py --folder path/to/folder --output results.xlsx
#
# Designed to run the full pipeline in < 15 minutes on CUDA hardware.
# =============================================================================

from __future__ import annotations

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import (
    DEVICE, GDINO_CONFIG, GDINO_WEIGHTS,
    SAM_CHECKPOINT, SAM_MODEL_TYPE,
)
from model.grounded_sam  import GroundedSAMModel
from model.qr_calibrator import QRCalibrator
from model.metrics       import MeasurementEngine, GeometricMeasurements
from model.reto_measurements import (
    RetoMeasurementEngine, assign_bands, assign_cuts_to_bands,
)
from view.visualiser  import Visualiser
from view.xlsx_writer import write_reto_xlsx
from utils.utils      import load_image

EXTENSIONS = (".jpg", ".jpeg", ".png")
_BASE = os.path.dirname(os.path.abspath(__file__))


# ── Per-image processing ──────────────────────────────────────────────────────

def _process_image(
    image_path: str,
    gsam_model: GroundedSAMModel,
    qr_calibrator: QRCalibrator,
    visualiser: Visualiser,
    verbose: bool = True,
) -> dict:
    t0 = time.perf_counter()
    img_name = os.path.basename(image_path)

    image_bgr = load_image(image_path)
    H, W      = image_bgr.shape[:2]

    # ── Detection ─────────────────────────────────────────────────────────────
    pipeline_result = gsam_model.run(image_bgr, image_path=image_path)

    # ── QR calibration ────────────────────────────────────────────────────────
    qr_result = qr_calibrator.calibrate_from_detections(
        qr_detections=pipeline_result.qr_detections,
        image_shape=(H, W),
    )
    if verbose:
        status = "OK" if qr_result.success else "FAIL"
        print(f"  [QR {status}] {qr_result.message}")

    # ── Legacy geometric measurements (for visualiser) ────────────────────────
    if qr_result.success:
        engine       = MeasurementEngine(qr_calibrator, H)
        measurements = engine.compute(pipeline_result, qr_result.homography)
        pipeline_result.edge_sample_points_mm  = measurements.edge_sample_points_mm
        pipeline_result.distances_to_bottom_mm = measurements.distances_to_bottom_mm
    else:
        measurements = GeometricMeasurements()

    # ── Reto measurements ─────────────────────────────────────────────────────
    reto1_rows = []
    reto2_rows = []

    if qr_result.success:
        # Assign strips to Band A and Band B
        band_a, band_b = assign_bands(
            pipeline_result.strip_detections,
            qr_result.homography,
            qr_calibrator,
        )
        mask_a = band_a.mask if band_a is not None else None
        mask_b = band_b.mask if band_b is not None else None

        # Assign cut edges to bands
        cut_a_det, cut_b_det = assign_cuts_to_bands(
            pipeline_result.edge_detections, band_a, band_b
        )
        cut_mask_a = cut_a_det.mask if cut_a_det is not None else None
        cut_mask_b = cut_b_det.mask if cut_b_det is not None else None

        reto_engine = RetoMeasurementEngine(
            qr_calibrator=qr_calibrator,
            homography=qr_result.homography,
            H=H, W=W,
        )
        reto1_rows = reto_engine.compute_reto1(mask_a, mask_b)
        reto2_rows = reto_engine.compute_reto2(
            cut_mask_a, cut_mask_b,
            mask_a, mask_b,
            image_bgr=image_bgr,
        )

        if verbose:
            n_none_r1 = sum(
                1 for r in reto1_rows
                if any(v is None for v in [r.DA, r.DB, r.LA, r.LB])
            )
            print(f"  [Reto1] {len(reto1_rows)} positions, "
                  f"{n_none_r1} with missing values")
            n_none_r2 = sum(
                1 for r in reto2_rows
                if any(v is None for v in [r.SA1, r.SA2, r.YSA])
            )
            print(f"  [Reto2] {len(reto2_rows)} positions, "
                  f"{n_none_r2} Band A missing, "
                  f"{sum(1 for r in reto2_rows if r.SB1 is None)} Band B missing")

    # ── Visualise ─────────────────────────────────────────────────────────────
    base_name = os.path.splitext(img_name)[0]
    visualiser.render(
        image_bgr=image_bgr,
        pipeline_res=pipeline_result,
        measurements=measurements,
        eval_metrics=None,
        base_name=base_name,
        calibration_valid=qr_result.success,
    )

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"  → done in {elapsed:.1f}s\n")

    return {
        "image_name": img_name,
        "qr_ok":      qr_result.success,
        "n_qr":       len(pipeline_result.qr_detections),
        "n_edges":    len(pipeline_result.edge_detections),
        "reto1_rows": reto1_rows,
        "reto2_rows": reto2_rows,
        "elapsed_s":  round(elapsed, 2),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Michelin Reto — competition-day xlsx generator"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", nargs="+",
                     help="One to three image paths to process.")
    src.add_argument("--folder", type=str,
                     help="Process all images in this folder.")

    p.add_argument("--output", type=str,
                   default=os.path.join(_BASE, "reto_results.xlsx"),
                   help="Output xlsx path (default: reto_results.xlsx).")
    p.add_argument("--vis-output", type=str,
                   default=os.path.join(_BASE, "data", "outputs", "reto"),
                   help="Folder for annotated visualisations.")
    p.add_argument("--device", type=str, default=DEVICE,
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

    # Collect image paths
    if args.images:
        image_paths = [p for p in args.images if os.path.isfile(p)]
    else:
        image_paths = sorted([
            os.path.join(args.folder, f)
            for f in os.listdir(args.folder)
            if os.path.splitext(f)[1].lower() in EXTENSIONS
        ])

    if not image_paths:
        print("[reto_challenge] No images found — check paths.")
        sys.exit(1)

    print(f"[reto_challenge] Processing {len(image_paths)} image(s):")
    for p in image_paths:
        print(f"  • {p}")
    print()

    # Load models (once)
    print("[reto_challenge] Loading models …")
    t_load = time.perf_counter()
    gsam_model = GroundedSAMModel(
        device        = args.device,
        gdino_config  = args.gdino_config,
        gdino_weights = args.gdino_weights,
        sam_checkpoint = args.sam_checkpoint,
        sam_model_type = args.sam_model_type,
    )
    qr_calibrator = QRCalibrator()
    print(f"[reto_challenge] Models ready in {time.perf_counter()-t_load:.1f}s\n")

    os.makedirs(args.vis_output, exist_ok=True)
    visualiser = Visualiser(
        output_dir=args.vis_output,
        show=not args.no_show,
        save=not args.no_save,
    )

    # Process images
    all_results = []
    t_total = time.perf_counter()
    for img_path in image_paths:
        print(f"[reto_challenge] → {os.path.basename(img_path)}")
        try:
            result = _process_image(
                img_path, gsam_model, qr_calibrator, visualiser
            )
            all_results.append(result)
        except Exception as exc:
            import traceback
            print(f"  [ERROR] {exc}")
            traceback.print_exc()

    # Write xlsx
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_reto_xlsx(args.output, all_results)

    elapsed = time.perf_counter() - t_total
    print(f"\n[reto_challenge] Total time: {elapsed:.1f}s")
    print(f"[reto_challenge] xlsx → {args.output}")


if __name__ == "__main__":
    main()

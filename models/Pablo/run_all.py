# =============================================================================
# run_all.py
# Batch-process all images from data/images/ and new_images/Pos*/ in one shot.
#
# Loads models once, runs the full pipeline on every image, collects Reto 1
# and Reto 2 measurements, writes a summary xlsx and prints statistics.
#
# Usage (from models/Pablo/):
#   python run_all.py
#   python run_all.py --device cpu --no-save
#   python run_all.py --output-dir data/outputs/run_all --xlsx run_all_results.xlsx
# =============================================================================

from __future__ import annotations

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import (
    DEVICE,
    GDINO_CONFIG, GDINO_WEIGHTS,
    SAM_CHECKPOINT, SAM_MODEL_TYPE,
)
from model.grounded_sam  import GroundedSAMModel
from model.qr_calibrator import QRCalibrator
from view.visualiser     import Visualiser
from view.xlsx_writer    import write_reto_xlsx

# Import the per-image processing function from reto_challenge
from reto_challenge import _process_image

EXTENSIONS = (".jpg", ".jpeg", ".png")

_BASE          = os.path.dirname(os.path.abspath(__file__))
_DATA_IMAGES   = os.path.join(_BASE, "data", "images")
_NEW_IMAGES    = os.path.join(_BASE, "new_images")
_DEFAULT_OUT   = os.path.join(_BASE, "data", "outputs", "run_all")
_DEFAULT_XLSX  = os.path.join(_BASE, "run_all_results.xlsx")


def collect_images() -> list[str]:
    """Collect all image paths from data/images/ and new_images/Pos*/."""
    paths: list[str] = []

    # ── Flat images from data/images/ ─────────────────────────────────────────
    if os.path.isdir(_DATA_IMAGES):
        for fname in sorted(os.listdir(_DATA_IMAGES)):
            if os.path.splitext(fname)[1].lower() in EXTENSIONS:
                paths.append(os.path.join(_DATA_IMAGES, fname))

    # ── new_images/Pos*/ sub-folders (recursive, skipping 'outputs') ──────────
    if os.path.isdir(_NEW_IMAGES):
        for root, dirs, files in os.walk(_NEW_IMAGES):
            # Skip any folder named 'outputs' in-place to avoid descending into it
            dirs[:] = sorted(
                d for d in dirs
                if d.lower() != "outputs" and not d.startswith(".")
            )
            # Only include files from Pos* sub-folders (not the root itself)
            rel = os.path.relpath(root, _NEW_IMAGES)
            if rel == "." :
                continue
            for fname in sorted(files):
                if os.path.splitext(fname)[1].lower() in EXTENSIONS:
                    paths.append(os.path.join(root, fname))

    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Michelin — batch-process all images and write a summary xlsx."
        )
    )
    p.add_argument(
        "--output-dir", type=str, default=_DEFAULT_OUT,
        help="Folder for annotated visualisation images.",
    )
    p.add_argument(
        "--xlsx", type=str, default=_DEFAULT_XLSX,
        help="Output xlsx path (default: run_all_results.xlsx).",
    )
    p.add_argument(
        "--device", type=str, default=DEVICE,
        choices=["cuda", "cpu"],
    )
    p.add_argument("--no-show", action="store_true",
                   help="Do not display images (headless mode).")
    p.add_argument("--no-save", action="store_true",
                   help="Skip saving visualisation PNGs.")
    p.add_argument("--gdino-config",   type=str, default=GDINO_CONFIG)
    p.add_argument("--gdino-weights",  type=str, default=GDINO_WEIGHTS)
    p.add_argument("--sam-checkpoint", type=str, default=SAM_CHECKPOINT)
    p.add_argument("--sam-model-type", type=str, default=SAM_MODEL_TYPE,
                   choices=["vit_h", "vit_l", "vit_b"])
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    # ── Collect images ─────────────────────────────────────────────────────────
    image_paths = collect_images()
    if not image_paths:
        print("[run_all] No images found — check data/images/ and new_images/.")
        sys.exit(1)

    print(f"[run_all] Found {len(image_paths)} image(s):")
    for p in image_paths:
        rel = os.path.relpath(p, _BASE)
        print(f"  • {rel}")
    print()

    # ── Load models once ───────────────────────────────────────────────────────
    print("[run_all] Loading models …")
    t_load = time.perf_counter()
    gsam_model = GroundedSAMModel(
        device         = args.device,
        gdino_config   = args.gdino_config,
        gdino_weights  = args.gdino_weights,
        sam_checkpoint = args.sam_checkpoint,
        sam_model_type = args.sam_model_type,
    )
    qr_calibrator = QRCalibrator()
    print(f"[run_all] Models ready in {time.perf_counter() - t_load:.1f}s\n")

    # ── Set up visualiser ──────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    visualiser = Visualiser(
        output_dir = args.output_dir,
        show       = not args.no_show,
        save       = not args.no_save,
    )

    # ── Process images ─────────────────────────────────────────────────────────
    all_results: list[dict] = []
    t_total = time.perf_counter()

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        print(f"[run_all] → {os.path.relpath(img_path, _BASE)}")
        try:
            result = _process_image(
                img_path, gsam_model, qr_calibrator, visualiser
            )
            all_results.append(result)
        except Exception as exc:
            import traceback
            print(f"  [ERROR] {img_name}: {exc}")
            traceback.print_exc()

    # ── Write xlsx ─────────────────────────────────────────────────────────────
    xlsx_path = args.xlsx
    os.makedirs(os.path.dirname(os.path.abspath(xlsx_path)), exist_ok=True)
    write_reto_xlsx(xlsx_path, all_results)

    # ── Summary ────────────────────────────────────────────────────────────────
    total_time  = time.perf_counter() - t_total
    total       = len(all_results)
    qr_ok_count = sum(1 for r in all_results if r.get("qr_ok"))
    edge_count  = sum(1 for r in all_results if r.get("n_edges", 0) > 0)

    print(f"\n{'=' * 60}")
    print(f"  Images processed    : {total}")
    print(f"  QR calibration OK   : {qr_ok_count}/{total}"
          f"  ({100 * qr_ok_count // total if total else 0}%)")
    print(f"  Edges detected      : {edge_count}/{total}"
          f"  ({100 * edge_count // total if total else 0}%)")
    print(f"  Total time          : {total_time:.1f}s"
          + (f"  (avg {total_time / total:.1f}s/image)" if total else ""))
    print(f"  xlsx → {xlsx_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

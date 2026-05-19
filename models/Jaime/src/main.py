"""
Pipeline completo — Michelin

Pasos en orden:
  1. QR detection       → outputs/detections/qr_homographies.json
  2. Rubber detection   → outputs/detections/rubber_detections.json
  3. Cut-edge detection → outputs/detections/cut_edge_detections.json
  4. Rubber segmentation
  5. Cut-edge segmentation
  6. Cut-edge measurement (punto único + multi-punto)

Uso:
  python main.py
  python main.py --skip-eval --skip-viz          # solo exportar JSONs y segmentar
  python main.py --skip-seg --skip-measure       # solo detección
  python main.py --weights-qr-rubber ruta/best.pt --weights-cut-edge ruta/best.pt
"""

import sys
import argparse
import json
from pathlib import Path

_SRC  = Path(__file__).resolve().parent   # models/Jaime/src/
_BASE = _SRC.parent                        # models/Jaime/

# Añade src/ al path para que los subpaquetes sean visibles como paquetes
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC.parent))  # models/Jaime/ → permite `from src.detection import ...`
    sys.path.insert(0, str(_SRC))         # models/Jaime/src/ → permite `from detection import ...`

from detection    import qr_detection          as qr_det
from detection    import rubber_detection      as rb_det
from detection    import cut_edge_detection    as ce_det
from segmentation import rubber_segmentation   as rb_seg
from segmentation import cut_edge_segmentation as ce_seg
from measurement  import cut_edge_measurement  as ce_meas


# ── Rutas por defecto ─────────────────────────────────────────────────────────
_DEFAULT_WEIGHTS_QR_RUBBER = str(_BASE / "utils" / "runs" / "data" / "runs" / "michelin_v1" / "weights" / "best.pt")
_DEFAULT_WEIGHTS_CUT_EDGE  = str(_BASE / "utils" / "runs" / "data" / "runs" / "michelin_v3-4" / "weights" / "best.pt")
_DEFAULT_WEIGHTS_SAM       = str(_BASE / "test" / "efficient_sam_vitt.pt")

_QR_JSON      = _BASE / "outputs" / "detections" / "qr_homographies.json"
_RUBBER_JSON  = _BASE / "outputs" / "detections" / "rubber_detections.json"
_CE_JSON      = _BASE / "outputs" / "detections" / "cut_edge_detections.json"
_RAW_DIR      = _BASE / "data" / "raw"
_DATA_SEG_CE  = _BASE / "data" / "segmentation_cut_edge"


# ── Helpers ───────────────────────────────────────────────────────────────────
def _step(label: str) -> None:
    print(f"\n{'━' * 60}")
    print(f"  {label}")
    print(f"{'━' * 60}")


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}\n"
            "Asegúrate de haber ejecutado el paso de detección previo."
        )
    return json.loads(path.read_text())


# ── Argparse ──────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline completo Michelin")
    p.add_argument("--weights-qr-rubber", default=_DEFAULT_WEIGHTS_QR_RUBBER,
                   metavar="PATH", help="Pesos YOLO para QR y rubber strip")
    p.add_argument("--weights-cut-edge",  default=_DEFAULT_WEIGHTS_CUT_EDGE,
                   metavar="PATH", help="Pesos YOLO para cut-edge")
    p.add_argument("--weights-sam",       default=_DEFAULT_WEIGHTS_SAM,
                   metavar="PATH", help="Pesos EfficientSAM ViT-Tiny")
    p.add_argument("--skip-eval",    action="store_true",
                   help="Omitir evaluación sobre el dataset anotado")
    p.add_argument("--skip-viz",     action="store_true",
                   help="Omitir guardado de figuras de detección")
    p.add_argument("--skip-seg",     action="store_true",
                   help="Omitir etapa de segmentación (pasos 4 y 5)")
    p.add_argument("--skip-measure", action="store_true",
                   help="Omitir etapa de medición (paso 6)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    from ultralytics import YOLO

    # ── 1. QR detection ───────────────────────────────────────────────────────
    _step("1 / 6  QR detection")
    model_qr_rubber = YOLO(args.weights_qr_rubber)
    print(f"[init] Modelo QR/rubber: {args.weights_qr_rubber}")

    if not args.skip_eval:
        qr_det.evaluate(model_qr_rubber)
    if not args.skip_viz:
        qr_det.save_figures(model_qr_rubber)
    qr_det.export_homographies(model_qr_rubber)

    # ── 2. Rubber detection ───────────────────────────────────────────────────
    _step("2 / 6  Rubber detection")
    if not args.skip_eval:
        rb_det.evaluate(model_qr_rubber)
    if not args.skip_viz:
        rb_det.save_figures(model_qr_rubber)
    rb_det.export_detections(model_qr_rubber)

    # ── 3. Cut-edge detection ─────────────────────────────────────────────────
    _step("3 / 6  Cut-edge detection")
    model_ce = YOLO(args.weights_cut_edge)
    print(f"[init] Modelo cut-edge: {args.weights_cut_edge}")

    if not args.skip_eval and _DATA_SEG_CE.exists():
        ce_det.run_evaluation(_DATA_SEG_CE, model_ce)

    if not args.skip_viz and _DATA_SEG_CE.exists():
        all_imgs_ce = [
            (img, _DATA_SEG_CE / "train" / "labels" / (img.stem + ".txt"))
            for img in sorted((_DATA_SEG_CE / "train" / "images").glob("*.jpg"))
        ] + [
            (img, _DATA_SEG_CE / "valid" / "labels" / (img.stem + ".txt"))
            for img in sorted((_DATA_SEG_CE / "valid" / "images").glob("*.jpg"))
        ]
        ce_det.save_visualizations(
            all_imgs_ce, model_ce,
            _BASE / "outputs" / "figures" / "cut_edge_detection",
        )

    ce_det.export_detections_json(_RAW_DIR, model_ce, _CE_JSON)

    if args.skip_seg and args.skip_measure:
        print("\nPipeline completado (segmentación y medición omitidas).")
        return

    # ── Cargar EfficientSAM (compartido por pasos 4-6) ────────────────────────
    _step("Cargando EfficientSAM")
    sam = ce_meas.load_model(Path(args.weights_sam))
    print(f"[init] EfficientSAM listo en {ce_meas.DEVICE}: {args.weights_sam}")

    # ── 4. Rubber segmentation ────────────────────────────────────────────────
    if not args.skip_seg:
        _step("4 / 6  Rubber segmentation")
        rubber_dets = _load_json(_RUBBER_JSON)
        rb_seg.segment_all(
            rubber_dets, _RAW_DIR, sam,
            _BASE / "outputs" / "figures" / "rubber_segmentation",
        )

    # ── 5. Cut-edge segmentation ──────────────────────────────────────────────
    if not args.skip_seg:
        _step("5 / 6  Cut-edge segmentation")
        ce_dets = _load_json(_CE_JSON)
        ce_seg.segment_all(
            ce_dets, _RAW_DIR, sam,
            _BASE / "outputs" / "figures" / "cut_edge_segmentation",
        )

    # ── 6. Cut-edge measurement ───────────────────────────────────────────────
    if not args.skip_measure:
        _step("6 / 6  Cut-edge measurement")
        qr_data = _load_json(_QR_JSON)
        ce_dets = _load_json(_CE_JSON)
        ce_meas.process_single_point(
            qr_data, ce_dets, _RAW_DIR, sam,
            _BASE / "outputs" / "figures" / "cut_edge_distances",
        )
        ce_meas.process_multipoint(
            qr_data, ce_dets, _RAW_DIR, sam,
            _BASE / "outputs" / "figures" / "cut_edge_multipoint",
        )

    _step("Pipeline completado")


if __name__ == "__main__":
    main()

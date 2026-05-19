"""
Detección de rubber strips — YOLOv9

Pipeline:
  1. Carga el modelo YOLOv9 entrenado
  2. Inferencia filtrando solo rubber_strip (clase 2)
  3. Guarda visualizaciones en outputs/figures/rubber_detection/
  4. Exporta bboxes detectados sobre imágenes raw a outputs/detections/rubber_detections.json

Uso:
  python rubber_detection.py                    # visualizar + exportar JSON
  python rubber_detection.py --only-export      # solo exportar JSON (imágenes raw)
  python rubber_detection.py --weights ruta/best.pt
"""

import argparse
import json
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

# ── Rutas ────────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).resolve().parents[2]   # models/Jaime/
DATA_ROOT = _HERE / "data" / "new_samples_qr_rubber"
RAW_DIR   = _HERE / "data" / "raw"
OUT_FIG   = _HERE / "outputs" / "figures" / "rubber_detection"
OUT_JSON  = _HERE / "outputs" / "detections" / "rubber_detections.json"

# ── Constantes ────────────────────────────────────────────────────────────────
RUBBER_CLASS  = 2
CONF_THR      = 0.65
MAX_BOX_RATIO = 0.95

COLOR_PRED = (0, 200, 0)     # verde — predicción

DEFAULT_WEIGHTS = str(_HERE / "utils" / "runs" / "data" / "runs" / "michelin_v1" / "weights" / "best.pt")


def filter_preds(preds, img_h: int, img_w: int) -> tuple:
    raw = [(b.xyxy[0], float(b.conf[0])) for b in preds.boxes if int(b.cls[0]) == RUBBER_CLASS]
    raw = [p for p in raw
           if (int(p[0][2]) - int(p[0][0])) * (int(p[0][3]) - int(p[0][1])) / (img_h * img_w) <= MAX_BOX_RATIO]
    boxes = [[*map(int, p[0])] for p in raw]
    confs = [p[1] for p in raw]
    return boxes, confs


# ── Pasos del pipeline ────────────────────────────────────────────────────────
def save_figures(model: YOLO) -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    splits = [
        (img, DATA_ROOT / "train" / "labels" / (img.stem + ".txt"))
        for img in sorted((DATA_ROOT / "train" / "images").glob("*.jpg"))
    ] + [
        (img, DATA_ROOT / "valid" / "labels" / (img.stem + ".txt"))
        for img in sorted((DATA_ROOT / "valid" / "images").glob("*.jpg"))
    ]
    print(f"[rubber] Guardando figuras ({len(splits)} imágenes)...")
    t0 = time.time()

    for img_path, lbl_path in splits:
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W    = img_rgb.shape[:2]
        preds   = model(img_bgr, verbose=False, conf=CONF_THR)[0]
        pred_boxes, pred_confs = filter_preds(preds, H, W)

        # top-2 por score
        if pred_confs:
            top2 = sorted(zip(pred_confs, pred_boxes), key=lambda x: x[0], reverse=True)[:2]
            pred_confs, pred_boxes = zip(*top2)

        overlay = img_rgb.copy()
        for box, conf in zip(pred_boxes, pred_confs):
            x1, y1, x2, y2 = box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_PRED, 2)
            cv2.putText(overlay, f"pred {conf:.2f}", (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PRED, 2)

        clean_stem = img_path.stem.split("_jpg.rf.")[0]
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(overlay)
        ax.set_title(f"{clean_stem}.jpg")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(OUT_FIG / f"{clean_stem}.jpg", dpi=100, bbox_inches="tight")
        plt.close()

    print(f"[rubber] Figuras completadas — {len(splits)} imágenes — {time.time()-t0:.1f} s")


def save_figures_raw(model: YOLO) -> None:
    """Visualización sobre imágenes raw (sin ground truth)."""
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(RAW_DIR.glob("*.jpg"))
    print(f"[rubber] Guardando figuras raw ({len(image_paths)} imágenes)...")
    t0 = time.time()

    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W    = img_bgr.shape[:2]
        preds   = model(img_bgr, verbose=False, conf=CONF_THR)[0]
        pred_boxes, pred_confs = filter_preds(preds, H, W)

        # top-2 por score
        if pred_confs:
            top2 = sorted(zip(pred_confs, pred_boxes), key=lambda x: x[0], reverse=True)[:2]
            pred_confs, pred_boxes = zip(*top2)

        overlay = img_rgb.copy()
        for box, conf in zip(pred_boxes, pred_confs):
            x1, y1, x2, y2 = box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_PRED, 2)
            cv2.putText(overlay, f"pred {conf:.2f}", (x1, max(y1 - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PRED, 2)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(overlay)
        ax.set_title(img_path.name)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(OUT_FIG / img_path.name, dpi=100, bbox_inches="tight")
        plt.close()

    print(f"[rubber] Figuras raw completadas — {len(image_paths)} imágenes — {time.time()-t0:.1f} s")


def export_detections(model: YOLO) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(RAW_DIR.glob("*.jpg"))
    print(f"[rubber] Exportando detecciones ({len(image_paths)} imágenes)...")
    t0 = time.time()

    detections = {}
    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        H, W    = img_bgr.shape[:2]
        preds   = model(img_bgr, verbose=False, conf=CONF_THR)[0]
        pred_boxes, pred_confs = filter_preds(preds, H, W)
        if pred_confs:
            top2 = sorted(zip(pred_confs, pred_boxes), key=lambda x: x[0], reverse=True)[:2]
            pred_confs, pred_boxes = zip(*top2)

        detections[img_path.name] = [
            {"bbox": box, "conf": round(conf, 4)}
            for box, conf in zip(pred_boxes, pred_confs)
        ]

    OUT_JSON.write_text(json.dumps(detections, indent=2))
    print(f"[rubber] JSON exportado — {len(image_paths)} imágenes — {time.time()-t0:.1f} s")


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rubber strip detection pipeline — Michelin")
    p.add_argument("--weights",     type=str, default=DEFAULT_WEIGHTS)
    p.add_argument("--only-export", action="store_true", help="Solo exportar JSON de detecciones")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    model = YOLO(args.weights)
    print(f"[init] Modelo cargado: {args.weights}")

    if args.only_export:
        export_detections(model)
        return

    save_figures(model)
    export_detections(model)


if __name__ == "__main__":
    main()

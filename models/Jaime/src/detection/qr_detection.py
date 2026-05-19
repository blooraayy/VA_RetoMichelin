"""
Detección de QR codes — YOLOv9

Pipeline:
  1. Carga el modelo YOLOv9 entrenado
  2. Inferencia filtrando solo qr_code (clase 1)
  3. Evalúa sobre train + valid comparando con ground truth (IoU, P, R, F1)
  4. Guarda visualizaciones en outputs/figures/qr_detection/
  5. Exporta posiciones QR y homografías a outputs/detections/qr_homographies.json

Uso:
  python qr_detection.py                        # eval + visualizar + exportar JSON
  python qr_detection.py --only-export          # solo exportar JSON (imágenes raw)
  python qr_detection.py --only-eval            # solo métricas, sin guardar figuras
  python qr_detection.py --weights ruta/best.pt
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
_HERE    = Path(__file__).resolve().parents[2]   # models/Jaime/
DATA_ROOT = _HERE / "data" / "new_samples_qr_rubber"
RAW_DIR   = _HERE / "data" / "raw"
OUT_FIG   = _HERE / "outputs" / "figures" / "qr_detection"
OUT_JSON  = _HERE / "outputs" / "detections" / "qr_homographies.json"

# ── Constantes ────────────────────────────────────────────────────────────────
QR_CLASS      = 1
CONF_THR      = 0.40
MAX_BOX_RATIO = 0.15     # descarta predicciones cuya área supere el 15 % de la imagen
OUTPUT_SIZE   = (1000, 1400)
OUT_W, OUT_H  = OUTPUT_SIZE

COLOR_GT   = (0, 220, 80)    # verde   — ground truth
COLOR_PRED = (0, 255, 0)     # verde — predicción

DEFAULT_WEIGHTS = str(_HERE / "utils" / "runs" / "data" / "runs" / "michelin_v1" / "weights" / "best.pt")


def filter_preds(preds, img_h: int, img_w: int) -> tuple:
    raw = [(b.xyxy[0], float(b.conf[0])) for b in preds.boxes if int(b.cls[0]) == QR_CLASS]
    raw = [p for p in raw
           if (int(p[0][2]) - int(p[0][0])) * (int(p[0][3]) - int(p[0][1])) / (img_h * img_w) <= MAX_BOX_RATIO]
    raw = sorted(raw, key=lambda p: p[1], reverse=True)[:3]
    boxes = [[*map(int, p[0])] for p in raw]
    confs = [p[1] for p in raw]
    return boxes, confs


# ── Helpers de homografía ────────────────────────────────────────────────────
def _label_by_position(centers: dict) -> dict:
    pts  = np.array(list(centers.values()), dtype=np.float32)
    idxs = list(centers.keys())
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    labels = {}
    labels[idxs[np.argmin(s)]]    = "TL"
    labels[idxs[np.argmax(s)]]    = "BR"
    labels[idxs[np.argmin(diff)]] = "TR"
    labels[idxs[np.argmax(diff)]] = "BL"
    return labels


def _compute_H(qr_centers: dict):
    req     = ["TL", "TR", "BR", "BL"]
    dst_map = {"TL": (0, 0), "TR": (OUT_W, 0), "BR": (OUT_W, OUT_H), "BL": (0, OUT_H)}
    available = [k for k in req if k in qr_centers]
    if len(available) == 4:
        src = np.array([qr_centers[k] for k in req], dtype=np.float32)
        dst = np.array([dst_map[k]     for k in req], dtype=np.float32)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        return H
    if len(available) >= 3:
        src = np.array([qr_centers[k] for k in available[:3]], dtype=np.float32)
        dst = np.array([dst_map[k]     for k in available[:3]], dtype=np.float32)
        A   = cv2.getAffineTransform(src, dst)
        return np.vstack([A, [0, 0, 1]]).astype(np.float32)
    return None


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

    print(f"[qr] Guardando figuras ({len(splits)} imágenes)...")
    t0 = time.time()

    for img_path, _ in splits:
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W    = img_rgb.shape[:2]
        preds   = model(img_bgr, verbose=False, conf=CONF_THR)[0]
        pred_boxes, pred_confs = filter_preds(preds, H, W)

        overlay = img_rgb.copy()
        for box, conf in zip(pred_boxes, pred_confs):
            x1, y1, x2, y2 = box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_PRED, 2)
            cv2.putText(overlay, f"pred {conf:.2f}", (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PRED, 2)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(overlay)
        ax.axis("off")
        clean_stem = img_path.stem.split("_jpg.rf.")[0]
        fig.suptitle(f"{clean_stem}.jpg", fontsize=10, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUT_FIG / f"{clean_stem}.jpg", dpi=100, bbox_inches="tight")
        plt.close()

    print(f"[qr] Figuras completadas — {len(splits)} imágenes — {time.time()-t0:.1f} s")


def export_homographies(model: YOLO) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    image_paths  = sorted(RAW_DIR.glob("*.jpg"))
    homographies = {}
    print(f"[qr] Exportando homografías ({len(image_paths)} imágenes)...")
    t0 = time.time()

    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        ih, iw  = img_bgr.shape[:2]
        preds   = model(img_bgr, verbose=False, conf=CONF_THR)[0]
        boxes   = [b for b in preds.boxes
                   if int(b.cls[0]) == QR_CLASS
                   and (int(b.xyxy[0][2]) - int(b.xyxy[0][0])) *
                       (int(b.xyxy[0][3]) - int(b.xyxy[0][1])) / (ih * iw) <= MAX_BOX_RATIO]
        if len(boxes) < 3:
            print(f"  [SKIP] {img_path.name}: solo {len(boxes)} QRs detectados")
            continue

        centers = {i: (float((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                       float((b.xyxy[0][1] + b.xyxy[0][3]) / 2))
                   for i, b in enumerate(boxes)}
        labels  = _label_by_position(centers)
        qr_pos  = {lbl: list(centers[idx]) for idx, lbl in labels.items()}
        H_mat   = _compute_H(qr_pos)

        if H_mat is None:
            print(f"  [ERR]  {img_path.name}: no se pudo calcular homografía")
            continue

        homographies[img_path.name] = {"qrs": qr_pos, "H": H_mat.tolist()}

    OUT_JSON.write_text(json.dumps(homographies, indent=2))
    print(f"[qr] Homografías exportadas — {len(homographies)}/{len(image_paths)} imágenes — {time.time()-t0:.1f} s")


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QR detection pipeline — Michelin")
    p.add_argument("--weights",     type=str, default=DEFAULT_WEIGHTS)
    p.add_argument("--only-export", action="store_true", help="Solo exportar JSON de homografías")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    model = YOLO(args.weights)
    print(f"[init] Modelo cargado: {args.weights}")

    if args.only_export:
        export_homographies(model)
        return

    save_figures(model)
    export_homographies(model)


if __name__ == "__main__":
    main()

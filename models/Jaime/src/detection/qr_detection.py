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
IOU_THR       = 0.50
MAX_BOX_RATIO = 0.15     # descarta predicciones cuya área supere el 15 % de la imagen
OUTPUT_SIZE   = (1000, 1400)
OUT_W, OUT_H  = OUTPUT_SIZE

COLOR_GT   = (0, 220, 80)    # verde   — ground truth
COLOR_PRED = (255, 220, 0)   # amarillo — predicción

DEFAULT_WEIGHTS = str(_HERE / "data" / "runs" / "michelin_v1" / "weights" / "best.pt")


# ── Helpers de métricas ───────────────────────────────────────────────────────
def load_gt_qr(label_path: Path, img_w: int, img_h: int) -> list:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if not parts or int(parts[0]) != QR_CLASS:
            continue
        cx, cy, w, h = map(float, parts[1:])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])
    return boxes


def bbox_iou(a: list, b: list) -> float:
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    inter  = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_predictions(gt_boxes: list, pred_boxes: list, iou_thr: float = IOU_THR):
    matched_gt, matched_pred, tp_ious = set(), set(), []
    for pi, pred in enumerate(pred_boxes):
        best_iou, best_gi = 0.0, -1
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = bbox_iou(pred, gt)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= iou_thr:
            tp_ious.append(best_iou)
            matched_gt.add(best_gi)
            matched_pred.add(pi)
    n_fp = len(pred_boxes) - len(matched_pred)
    n_fn = len(gt_boxes)   - len(matched_gt)
    return tp_ious, n_fp, n_fn


def filter_preds(preds, img_h: int, img_w: int) -> tuple:
    raw = [(b.xyxy[0], float(b.conf[0])) for b in preds.boxes if int(b.cls[0]) == QR_CLASS]
    raw = [p for p in raw
           if (int(p[0][2]) - int(p[0][0])) * (int(p[0][3]) - int(p[0][1])) / (img_h * img_w) <= MAX_BOX_RATIO]
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
def evaluate(model: YOLO) -> None:
    splits = [
        (img, DATA_ROOT / "train" / "labels" / (img.stem + ".txt"))
        for img in sorted((DATA_ROOT / "train" / "images").glob("*.jpg"))
    ] + [
        (img, DATA_ROOT / "valid" / "labels" / (img.stem + ".txt"))
        for img in sorted((DATA_ROOT / "valid" / "images").glob("*.jpg"))
    ]
    print(f"[eval] Total imágenes: {len(splits)}")

    total_tp = total_fp = total_fn = 0
    all_ious, per_image = [], []

    for img_path, lbl_path in splits:
        img_bgr = cv2.imread(str(img_path))
        H, W    = img_bgr.shape[:2]
        gt      = load_gt_qr(lbl_path, W, H)
        preds   = model(img_bgr, verbose=False, conf=CONF_THR)[0]
        pred_boxes, _ = filter_preds(preds, H, W)

        tp_ious, n_fp, n_fn = match_predictions(gt, pred_boxes)
        total_tp += len(tp_ious); total_fp += n_fp; total_fn += n_fn
        all_ious.extend(tp_ious)
        per_image.append(dict(name=img_path.name, gt=len(gt), pred=len(pred_boxes),
                              tp=len(tp_ious), fp=n_fp, fn=n_fn,
                              iou_mean=float(np.mean(tp_ious)) if tp_ious else None))

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'='*55}")
    print(f"TP: {total_tp}  FP: {total_fp}  FN: {total_fn}")
    print(f"Precision : {precision:.3f}")
    print(f"Recall    : {recall:.3f}")
    print(f"F1        : {f1:.3f}")
    if all_ious:
        print(f"IoU medio : {np.mean(all_ious):.3f}")

    print(f"\n{'Imagen':<50} {'GT':>4} {'Pred':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'IoU':>6}")
    print("-" * 80)
    for r in per_image:
        iou_str = f"{r['iou_mean']:.3f}" if r["iou_mean"] is not None else "  —  "
        print(f"{r['name']:<50} {r['gt']:>4} {r['pred']:>5} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4} {iou_str:>6}")


def save_figures(model: YOLO) -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    splits = [
        (img, DATA_ROOT / "train" / "labels" / (img.stem + ".txt"))
        for img in sorted((DATA_ROOT / "train" / "images").glob("*.jpg"))
    ] + [
        (img, DATA_ROOT / "valid" / "labels" / (img.stem + ".txt"))
        for img in sorted((DATA_ROOT / "valid" / "images").glob("*.jpg"))
    ]

    for img_path, lbl_path in splits:
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W    = img_rgb.shape[:2]
        gt      = load_gt_qr(lbl_path, W, H)
        preds   = model(img_bgr, verbose=False, conf=CONF_THR)[0]
        pred_boxes, pred_confs = filter_preds(preds, H, W)
        tp_ious, n_fp, n_fn    = match_predictions(gt, pred_boxes)

        overlay = img_rgb.copy()
        for box in gt:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_GT, 3)
            cv2.putText(overlay, "GT", (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_GT, 2)
        for box, conf in zip(pred_boxes, pred_confs):
            x1, y1, x2, y2 = box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_PRED, 2)
            cv2.putText(overlay, f"pred {conf:.2f}", (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PRED, 2)

        iou_str  = f"IoU={np.mean(tp_ious):.3f}" if tp_ious else "IoU=nulo"
        subtitle = f"GT={len(gt)}  Pred={len(pred_boxes)}  TP={len(tp_ious)}  FP={n_fp}  FN={n_fn}  {iou_str}"

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(overlay)
        ax.set_title("GT (verde) vs Pred (amarillo)")
        ax.axis("off")
        fig.suptitle(f"{img_path.name}\n{subtitle}", fontsize=10, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUT_FIG / f"{img_path.stem}.jpg", dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  {img_path.name} → {subtitle}")

    print(f"[viz] Figuras guardadas en: {OUT_FIG.resolve()}")


def export_homographies(model: YOLO) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    image_paths  = sorted(RAW_DIR.glob("*.jpg"))
    homographies = {}
    print(f"[export] Imágenes a procesar: {len(image_paths)}")

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
        print(f"  {img_path.name}: QRs={list(qr_pos.keys())}")

    OUT_JSON.write_text(json.dumps(homographies, indent=2))
    print(f"[export] JSON guardado en: {OUT_JSON.resolve()}")


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QR detection pipeline — Michelin")
    p.add_argument("--weights",     type=str, default=DEFAULT_WEIGHTS)
    p.add_argument("--only-export", action="store_true", help="Solo exportar JSON de homografías")
    p.add_argument("--only-eval",   action="store_true", help="Solo métricas, sin guardar figuras")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    model = YOLO(args.weights)
    print(f"[init] Modelo cargado: {args.weights}")

    if args.only_export:
        export_homographies(model)
        return

    if args.only_eval:
        evaluate(model)
        return

    evaluate(model)
    save_figures(model)
    export_homographies(model)


if __name__ == "__main__":
    main()

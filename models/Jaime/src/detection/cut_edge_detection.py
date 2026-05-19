import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

# ── Configuración ────────────────────────────────────────────────────────────
WEIGHTS        = "models/Jaime/utils/runs/data/runs/michelin_v3-4/weights/best.pt"
CUT_EDGE_CLASS = 0
CONF_THR       = 0.65
IOU_THR        = 0.50
MAX_BOX_RATIO  = 0.10
NMS_IOU        = 0.30

COLOR_GT   = (0, 220, 80)
COLOR_PRED = (255, 220, 0)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_gt_cut_edge(label_path, img_w, img_h):
    """
    Soporta dos formatos de Roboflow:
      - YOLO detección estándar:    class cx cy w h
      - YOLO segmentación con ID:   ann_id  class x1 y1 x2 y2 ...
    Para polígonos calcula el bbox como min/max de los vértices.
    """
    boxes = []
    if not Path(label_path).exists():
        return boxes
    for line in Path(label_path).read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue

        if len(parts) >= 6:
            try:
                ann_id = int(parts[0])
                cls    = int(parts[1])
                coords = list(map(float, parts[2:]))
            except ValueError:
                continue
        else:
            try:
                cls    = int(parts[0])
                coords = list(map(float, parts[1:]))
            except ValueError:
                continue

        if cls != CUT_EDGE_CLASS:
            continue

        if len(coords) == 4:
            cx, cy, w, h = coords
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
        else:
            xs = [coords[i] * img_w for i in range(0, len(coords), 2)]
            ys = [coords[i] * img_h for i in range(1, len(coords), 2)]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        boxes.append([x1, y1, x2, y2])
    return boxes


def bbox_iou(a, b):
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    inter  = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_preds(raw_preds, iou_thr=NMS_IOU):
    raw_preds = sorted(raw_preds, key=lambda p: float(p[1]), reverse=True)
    keep = []
    for box_i, conf_i in raw_preds:
        bi = [int(v) for v in box_i]
        if any(bbox_iou(bi, [int(v) for v in bj]) >= iou_thr for bj, _ in keep):
            continue
        keep.append((box_i, conf_i))
    return keep


def match_predictions(gt_boxes, pred_boxes, iou_thr=IOU_THR):
    matched_gt   = set()
    matched_pred = set()
    tp_ious = []
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


# ── Pipeline principal ────────────────────────────────────────────────────────

def run_evaluation(data_root: Path, model: YOLO):
    """Evalúa el modelo sobre train + valid y devuelve métricas globales."""
    all_imgs = [
        (img, data_root / "train" / "labels" / (img.stem + ".txt"))
        for img in sorted((data_root / "train" / "images").glob("*.jpg"))
    ] + [
        (img, data_root / "valid" / "labels" / (img.stem + ".txt"))
        for img in sorted((data_root / "valid" / "images").glob("*.jpg"))
    ]
    print(f"Total imágenes: {len(all_imgs)}")

    total_tp, total_fp, total_fn = 0, 0, 0
    all_tp_ious = []
    per_image   = []

    for img_path, lbl_path in all_imgs:
        img_bgr  = cv2.imread(str(img_path))
        H, W     = img_bgr.shape[:2]
        gt_boxes = load_gt_cut_edge(lbl_path, W, H)
        preds    = model(img_bgr, verbose=False, conf=CONF_THR)[0]

        raw_preds = [
            (b.xyxy[0], float(b.conf[0]))
            for b in preds.boxes
            if int(b.cls[0]) == CUT_EDGE_CLASS
            and (int(b.xyxy[0][2])-int(b.xyxy[0][0])) * (int(b.xyxy[0][3])-int(b.xyxy[0][1])) / (H * W) <= MAX_BOX_RATIO
        ]
        raw_preds  = nms_preds(raw_preds)
        pred_boxes = [[*map(int, p[0])] for p in raw_preds]

        tp_ious, n_fp, n_fn = match_predictions(gt_boxes, pred_boxes)
        total_tp += len(tp_ious)
        total_fp += n_fp
        total_fn += n_fn
        all_tp_ious.extend(tp_ious)

        per_image.append({
            "name":     img_path.name,
            "gt":       len(gt_boxes),
            "pred":     len(pred_boxes),
            "tp":       len(tp_ious),
            "fp":       n_fp,
            "fn":       n_fn,
            "iou_mean": np.mean(tp_ious) if tp_ious else None,
        })

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'='*55}")
    print(f"TP: {total_tp}  FP: {total_fp}  FN: {total_fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")
    if all_tp_ious:
        print(f"IoU medio (TP): {np.mean(all_tp_ious):.3f}")

    print(f"\n{'Imagen':<50} {'GT':>4} {'Pred':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'IoU':>6}")
    print("-" * 80)
    for r in per_image:
        iou_str = f"{r['iou_mean']:.3f}" if r["iou_mean"] is not None else "  — "
        print(f"{r['name']:<50} {r['gt']:>4} {r['pred']:>5} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4} {iou_str:>6}")

    return per_image, all_tp_ious


def save_visualizations(all_imgs, model: YOLO, out_dir: Path):
    """Guarda visualizaciones GT vs predicciones en out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path in all_imgs:
        img_bgr  = cv2.imread(str(img_path))
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W     = img_rgb.shape[:2]
        gt_boxes = load_gt_cut_edge(lbl_path, W, H)
        preds    = model(img_bgr, verbose=False, conf=CONF_THR)[0]

        raw_preds = [
            (b.xyxy[0], float(b.conf[0]))
            for b in preds.boxes
            if int(b.cls[0]) == CUT_EDGE_CLASS
            and (int(b.xyxy[0][2])-int(b.xyxy[0][0])) * (int(b.xyxy[0][3])-int(b.xyxy[0][1])) / (H * W) <= MAX_BOX_RATIO
        ]
        raw_preds  = nms_preds(raw_preds)
        pred_boxes = [[*map(int, p[0])] for p in raw_preds]
        pred_confs = [p[1] for p in raw_preds]

        tp_ious, n_fp, n_fn = match_predictions(gt_boxes, pred_boxes)

        overlay = img_rgb.copy()
        for box in gt_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_GT, 3)
            cv2.putText(overlay, "GT", (x1, max(y1-8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_GT, 2)
        for box, conf in zip(pred_boxes, pred_confs):
            x1, y1, x2, y2 = box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_PRED, 2)
            cv2.putText(overlay, f"pred {conf:.2f}", (x1, y2+18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PRED, 2)

        iou_str  = f"IoU={np.mean(tp_ious):.3f}" if tp_ious else "IoU=nulo"
        subtitle = f"GT={len(gt_boxes)}  Pred={len(pred_boxes)}  TP={len(tp_ious)}  FP={n_fp}  FN={n_fn}  {iou_str}"

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(overlay)
        ax.set_title("GT (verde) vs Pred (amarillo)")
        ax.axis("off")
        fig.suptitle(f"{img_path.name}\n{subtitle}", fontsize=10, fontweight="bold")
        plt.tight_layout()

        plt.savefig(out_dir / f"{img_path.stem}.jpg", dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  {img_path.name} → {subtitle}")

    print(f"\nFiguras guardadas en: {out_dir.resolve()}")


def export_detections_json(raw_dir: Path, model: YOLO, out_json: Path):
    """Corre inferencia sobre imágenes raw y exporta bboxes a JSON."""
    out_json.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(raw_dir.glob("*.jpg"))
    print(f"Imágenes a procesar: {len(image_paths)}")

    detections = {}
    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        H, W    = img_bgr.shape[:2]
        preds   = model(img_bgr, verbose=False, conf=CONF_THR)[0]

        raw = [
            (b.xyxy[0], float(b.conf[0]))
            for b in preds.boxes
            if int(b.cls[0]) == CUT_EDGE_CLASS
            and (int(b.xyxy[0][2])-int(b.xyxy[0][0])) * (int(b.xyxy[0][3])-int(b.xyxy[0][1])) / (H * W) <= MAX_BOX_RATIO
        ]
        raw = nms_preds(raw)

        detections[img_path.name] = [
            {"bbox": [*map(int, box)], "conf": round(conf, 4)}
            for box, conf in raw
        ]
        print(f"  {img_path.name}: {len(detections[img_path.name])} detecciones")

    out_json.write_text(json.dumps(detections, indent=2))
    print(f"Guardado en: {out_json.resolve()}")
    return detections


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = YOLO(WEIGHTS)
    print(f"Modelo cargado: {WEIGHTS}")

    DATA_ROOT = Path("models/Jaime/data/segmentation_cut_edge")
    OUT_DIR   = Path("models/Jaime/outputs/figures/cut_edge_detection")
    RAW_DIR   = Path("models/Jaime/data/raw")
    OUT_JSON  = Path("models/Jaime/outputs/detections/cut_edge_detections.json")

    all_imgs = [
        (img, DATA_ROOT / "train" / "labels" / (img.stem + ".txt"))
        for img in sorted((DATA_ROOT / "train" / "images").glob("*.jpg"))
    ] + [
        (img, DATA_ROOT / "valid" / "labels" / (img.stem + ".txt"))
        for img in sorted((DATA_ROOT / "valid" / "images").glob("*.jpg"))
    ]

    run_evaluation(DATA_ROOT, model)
    save_visualizations(all_imgs, model, OUT_DIR)
    export_detections_json(RAW_DIR, model, OUT_JSON)

import json
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

# ── Configuración ────────────────────────────────────────────────────────────
WEIGHTS        = "models/Jaime/utils/runs/data/runs/michelin_v3-4/weights/best.pt"
CUT_EDGE_CLASS = 0
CONF_THR       = 0.65
MAX_BOX_RATIO  = 0.10
NMS_IOU        = 0.30

COLOR_PRED = (0, 200, 0)


# ── Helpers ──────────────────────────────────────────────────────────────────

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


# ── Pipeline principal ────────────────────────────────────────────────────────

def save_visualizations(all_imgs, model: YOLO, out_dir: Path):
    """Guarda visualizaciones de predicciones en out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cut_edge] Guardando figuras ({len(all_imgs)} imágenes)...")
    t0 = time.time()

    for img_path, lbl_path in all_imgs:
        img_bgr  = cv2.imread(str(img_path))
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W     = img_rgb.shape[:2]
        preds    = model(img_bgr, verbose=False, conf=CONF_THR)[0]

        raw_preds = [
            (b.xyxy[0], float(b.conf[0]))
            for b in preds.boxes
            if int(b.cls[0]) == CUT_EDGE_CLASS
            and (int(b.xyxy[0][2])-int(b.xyxy[0][0])) * (int(b.xyxy[0][3])-int(b.xyxy[0][1])) / (H * W) <= MAX_BOX_RATIO
        ]
        raw_preds  = nms_preds(raw_preds)[:2]
        pred_boxes = [[*map(int, p[0])] for p in raw_preds]
        pred_confs = [p[1] for p in raw_preds]

        overlay = img_rgb.copy()
        for box, conf in zip(pred_boxes, pred_confs):
            x1, y1, x2, y2 = box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_PRED, 2)
            cv2.putText(overlay, f"pred {conf:.2f}", (x1, y2+18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PRED, 2)

        clean_stem = img_path.stem.split("_jpg.rf.")[0]
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(overlay)
        ax.set_title(f"{clean_stem}.jpg")
        ax.axis("off")
        plt.tight_layout()

        plt.savefig(out_dir / f"{clean_stem}.jpg", dpi=100, bbox_inches="tight")
        plt.close()

    print(f"[cut_edge] Figuras completadas — {len(all_imgs)} imágenes — {time.time()-t0:.1f} s")


def export_detections_json(raw_dir: Path, model: YOLO, out_json: Path):
    """Corre inferencia sobre imágenes raw y exporta bboxes a JSON."""
    out_json.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(raw_dir.glob("*.jpg"))
    print(f"[cut_edge] Exportando detecciones ({len(image_paths)} imágenes)...")
    t0 = time.time()

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
        raw = nms_preds(raw)[:2]

        detections[img_path.name] = [
            {"bbox": [*map(int, box)], "conf": round(conf, 4)}
            for box, conf in raw
        ]

    out_json.write_text(json.dumps(detections, indent=2))
    print(f"[cut_edge] JSON exportado — {len(image_paths)} imágenes — {time.time()-t0:.1f} s")
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

    save_visualizations(all_imgs, model, OUT_DIR)
    export_detections_json(RAW_DIR, model, OUT_JSON)

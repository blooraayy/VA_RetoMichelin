import json
import time
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# ── Configuración ────────────────────────────────────────────────────────────
DETECTIONS_JSON = Path("../outputs/detections/rubber_detections.json")
RAW_DIR         = Path("../data/raw")
ESAM_WEIGHTS    = Path("efficient_sam_vitt.pt")

COLOR_BBOX = (255, 220,   0)
COLOR_MASK = (255,  80,   0)
ALPHA      = 0.50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Carga de modelo ───────────────────────────────────────────────────────────

def load_model(weights_path: Path):
    from efficient_sam.efficient_sam import build_efficient_sam
    sam = build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint=str(weights_path),
    )
    sam.eval()
    return sam.to(DEVICE)


# ── Segmentación ──────────────────────────────────────────────────────────────

def run_efficient_sam(image_np: np.ndarray, box_xyxy, sam) -> np.ndarray:
    """
    Segmenta la región delimitada por box_xyxy usando esquinas TL y BR como prompt.
    Devuelve una máscara booleana del tamaño original de la imagen.
    """
    H, W = image_np.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]

    pad = 15
    x1p = max(0, x1 - pad);  y1p = max(0, y1 - pad)
    x2p = min(W, x2 + pad);  y2p = min(H, y2 + pad)
    crop = image_np[y1p:y2p, x1p:x2p]

    bx1 = float(x1 - x1p);  by1 = float(y1 - y1p)
    bx2 = float(x2 - x1p);  by2 = float(y2 - y1p)

    img_t = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
    img_t = img_t.unsqueeze(0).to(DEVICE)

    pts = torch.tensor([[[[bx1, by1], [bx2, by2]]]], dtype=torch.float32).to(DEVICE)
    lbl = torch.tensor([[[2, 3]]],                   dtype=torch.int32).to(DEVICE)

    with torch.no_grad():
        masks, scores = sam(img_t, pts, lbl)

    best      = scores[0, 0].argmax().item()
    crop_mask = masks[0, 0, best].cpu().numpy() > 0.5

    full_mask = np.zeros((H, W), dtype=bool)
    full_mask[y1p:y2p, x1p:x2p] = crop_mask
    return full_mask


def overlay_mask(base: np.ndarray, mask: np.ndarray, bbox, conf: float) -> np.ndarray:
    overlay = base.astype(np.float32)
    colored = np.zeros_like(overlay)
    colored[mask] = COLOR_MASK
    overlay = np.where(mask[:, :, None], overlay * (1 - ALPHA) + colored * ALPHA, overlay)

    ov_u8 = overlay.astype(np.uint8)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(ov_u8, (x1, y1), (x2, y2), COLOR_BBOX, 2)
    cv2.putText(ov_u8, f"rubber {conf:.2f}",
                (x1, max(y1 - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_BBOX, 2)
    return ov_u8


# ── Pipeline principal ────────────────────────────────────────────────────────

def segment_all(detections: dict, raw_dir: Path, sam, out_dir: Path):
    """Segmenta todas las imágenes y guarda las visualizaciones en out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(detections)
    print(f"[rubber_seg] Segmentando ({n} imágenes)...")
    t0 = time.time()

    for img_name, bboxes in sorted(detections.items()):
        img_path = raw_dir / img_name
        if not img_path.exists():
            print(f"  [SKIP] {img_name}: imagen no encontrada")
            continue

        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        canvas  = img_rgb.copy()

        for det in bboxes:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["conf"]
            try:
                mask   = run_efficient_sam(img_rgb, [x1, y1, x2, y2], sam)
                canvas = overlay_mask(canvas, mask, (x1, y1, x2, y2), conf)
            except Exception as e:
                print(f"  [WARN] {img_name} bbox={det['bbox']}: {e}")

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(canvas)
        ax.set_title("EfficientSAM — rubber strip")
        ax.axis("off")
        fig.text(0.5, 0.01, f"{len(bboxes)} rubber strip(s) segmentado(s)",
                 ha="center", fontsize=8, color="#444")
        fig.suptitle(img_name, fontsize=11, fontweight="bold")
        plt.tight_layout(rect=[0, 0.03, 1, 1])

        plt.savefig(out_dir / (Path(img_name).stem + ".jpg"), dpi=100, bbox_inches="tight")
        plt.close()

    print(f"[rubber_seg] Segmentación completada — {n} imágenes — {time.time()-t0:.1f} s")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    assert DETECTIONS_JSON.exists(), (
        f"No se encontró {DETECTIONS_JSON.resolve()}\n"
        "Ejecuta primero rubber_detection.py para generar el JSON."
    )
    assert ESAM_WEIGHTS.exists(), f"Pesos no encontrados: {ESAM_WEIGHTS.resolve()}"

    sam = load_model(ESAM_WEIGHTS)
    print(f"EfficientSAM listo en {DEVICE}")

    detections = json.loads(DETECTIONS_JSON.read_text())
    print(f"Imágenes con detecciones: {len(detections)}")
    print(f"Total bboxes: {sum(len(v) for v in detections.values())}")

    OUT_DIR = Path("../outputs/figures/rubber_segmentation")
    segment_all(detections, RAW_DIR, sam, OUT_DIR)

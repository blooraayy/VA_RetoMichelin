import json
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Configuración ────────────────────────────────────────────────────────────
CANONICAL_W = 1000
CANONICAL_H = 1400
CATETO_MM   = 800
SCALE_X = CATETO_MM / CANONICAL_W
SCALE_Y = CATETO_MM / CANONICAL_H

QR_JSON      = Path("../outputs/detections/qr_homographies.json")
CUTEDGE_JSON = Path("../outputs/detections/cut_edge_detections.json")
RAW_DIR      = Path("../data/raw")
ESAM_WEIGHTS = Path("efficient_sam_vitt.pt")

COLOR_MASK  = np.array([0, 200, 255], dtype=np.uint8)
COLOR_BBOX  = (255, 220, 0)
ALPHA       = 0.45
SIDE_COLORS = {"left": "deepskyblue", "right": "tomato",
               "top":  "deepskyblue", "bottom": "tomato"}

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


# ── Helpers ──────────────────────────────────────────────────────────────────

def run_esam(image_np: np.ndarray, box_xyxy, sam, pad: int = 15) -> np.ndarray:
    """Devuelve máscara booleana (H×W) para la bbox dada."""
    H, W = image_np.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    x1p = max(0, x1 - pad);  y1p = max(0, y1 - pad)
    x2p = min(W, x2 + pad);  y2p = min(H, y2 + pad)
    crop = image_np[y1p:y2p, x1p:x2p]

    bx1, by1 = float(x1 - x1p), float(y1 - y1p)
    bx2, by2 = float(x2 - x1p), float(y2 - y1p)

    img_t = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
    img_t = img_t.unsqueeze(0).to(DEVICE)
    pts   = torch.tensor([[[[bx1, by1], [bx2, by2]]]], dtype=torch.float32).to(DEVICE)
    lbl   = torch.tensor([[[2, 3]]],                   dtype=torch.int32).to(DEVICE)

    with torch.no_grad():
        masks, scores = sam(img_t, pts, lbl)

    best      = scores[0, 0].argmax().item()
    crop_mask = masks[0, 0, best].cpu().numpy() > 0.5
    full_mask = np.zeros((H, W), dtype=bool)
    full_mask[y1p:y2p, x1p:x2p] = crop_mask
    return full_mask


def apply_H(H_mat, pts):
    """Proyecta puntos (N×2) imagen → espacio canónico."""
    H_mat = np.array(H_mat, dtype=np.float64)
    pts   = np.atleast_2d(pts).astype(np.float64)
    hom   = np.hstack([pts, np.ones((len(pts), 1))])
    proj  = (H_mat @ hom.T).T
    return proj[:, :2] / proj[:, 2:3]


def mask_bottom_point(mask: np.ndarray):
    """Devuelve el punto más bajo de la máscara y su x central."""
    rows = np.any(mask, axis=1)
    if not rows.any():
        return None
    y_bot = int(np.where(rows)[0].max())
    cols  = np.where(mask[y_bot])[0]
    return (int(cols.mean()), y_bot)


def mask_edge_sample_points(mask: np.ndarray, n_inter: int = 3) -> dict:
    """
    Muestrea n_inter+2 puntos por borde adaptándose a la orientación:
    - altura >= anchura → bordes izquierdo y derecho (muestreo en Y)
    - anchura >  altura → bordes superior e inferior  (muestreo en X)
    """
    rows_any = np.any(mask, axis=1)
    cols_any = np.any(mask, axis=0)
    if not rows_any.any():
        return None
    y_min = int(np.where(rows_any)[0].min())
    y_max = int(np.where(rows_any)[0].max())
    x_min = int(np.where(cols_any)[0].min())
    x_max = int(np.where(cols_any)[0].max())
    height = y_max - y_min
    width  = x_max - x_min

    pts_a, pts_b = [], []
    if height >= width:
        y_vals = np.linspace(y_min, y_max, n_inter + 2).astype(int)
        for y in y_vals:
            c = np.where(mask[y])[0]
            if len(c) == 0:
                continue
            pts_a.append((int(c.min()), int(y)))
            pts_b.append((int(c.max()), int(y)))
        return {"left": pts_a, "right": pts_b}
    else:
        x_vals = np.linspace(x_min, x_max, n_inter + 2).astype(int)
        for x in x_vals:
            r = np.where(mask[:, x])[0]
            if len(r) == 0:
                continue
            pts_a.append((int(x), int(r.min())))
            pts_b.append((int(x), int(r.max())))
        return {"top": pts_a, "bottom": pts_b}


def build_overlay(img_rgb: np.ndarray, masks: list, contours: bool = False) -> np.ndarray:
    overlay = img_rgb.copy().astype(np.float32)
    contours_list = []
    for mask in masks:
        colored = np.zeros_like(overlay)
        colored[mask] = COLOR_MASK
        overlay = np.where(mask[:, :, None],
                           overlay * (1 - ALPHA) + colored * ALPHA, overlay)
        if contours:
            mask_u8 = mask.astype(np.uint8) * 255
            ctrs, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_list.append(ctrs)
    overlay = overlay.astype(np.uint8)
    for ctrs in contours_list:
        cv2.drawContours(overlay, list(ctrs), -1, (255, 215, 0), 2)
    return overlay


def draw_qr_triangle(ax, qr_centers: dict):
    order = ["TL", "TR", "BL", "TL"]
    for a, b in zip(order, order[1:]):
        if a in qr_centers and b in qr_centers:
            ax.plot([qr_centers[a][0], qr_centers[b][0]],
                    [qr_centers[a][1], qr_centers[b][1]],
                    color="lime", lw=1.8, zorder=3)
    for lbl, (cx, cy) in qr_centers.items():
        ax.plot(cx, cy, "o", color="yellow", ms=10, zorder=5)
        ax.text(cx + 14, cy - 14, lbl, color="yellow", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.55))


# ── Pipeline 1: distancia de punto más bajo hasta BL ─────────────────────────

def process_single_point(qr_data: dict, cut_edges: dict, raw_dir: Path,
                         sam, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'Imagen':<25} {'Borde':>6} {'conf':>5}  {'dist_BL (mm)':>13}")
    print("-" * 58)

    for img_name in sorted(qr_data.keys()):
        if img_name not in cut_edges or not cut_edges[img_name]:
            continue

        img_bgr    = cv2.imread(str(raw_dir / img_name))
        img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H_mat      = qr_data[img_name]["H"]
        qr_centers = qr_data[img_name]["qrs"]
        detections = cut_edges[img_name]
        BL_x, BL_y = qr_centers["BL"]

        masks = []
        for det in detections:
            try:
                masks.append(run_esam(img_rgb, det["bbox"], sam))
            except Exception as e:
                print(f"  [WARN] {img_name}: {e}")
                masks.append(np.zeros(img_rgb.shape[:2], dtype=bool))

        overlay = build_overlay(img_rgb, masks)

        fig, ax = plt.subplots(figsize=(11, 11))
        ax.imshow(overlay)
        draw_qr_triangle(ax, qr_centers)
        ax.axhline(BL_y, color="orange", lw=2.2, linestyle="--")
        ax.text(10, BL_y - 14, "Borde inferior mesa", color="orange",
                fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55))

        for i, (mask, det) in enumerate(zip(masks, detections)):
            pt = mask_bottom_point(mask)
            if pt is None:
                continue
            px, py  = pt
            canon   = apply_H(H_mat, [[px, py]])[0]
            dist_mm = max(0.0, (CANONICAL_H - canon[1]) * SCALE_Y)

            ax.plot([px, px], [py, BL_y], color="violet", lw=2, zorder=4)
            ax.plot(px, py, "^", color="violet", ms=8, zorder=6)
            mid_y = (py + BL_y) / 2
            ax.text(px + 8, mid_y, f"{dist_mm:.1f} mm",
                    color="violet", fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.6), zorder=7)
            print(f"{img_name:<25} {'#'+str(i+1):>6} {det['conf']:>5.2f}  {dist_mm:>13.1f}")

        ax.set_title("Segmentación bordes de corte · Línea BL · Distancias en mm", fontsize=10)
        ax.axis("off")
        fig.suptitle(img_name, fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_dir / f"{Path(img_name).stem}_distances.jpg",
                    dpi=100, bbox_inches="tight")
        plt.close()

    print(f"\nGuardado en: {out_dir.resolve()}")


# ── Pipeline 2: multi-punto (5 puntos por lado por segmentación) ──────────────

def process_multipoint(qr_data: dict, cut_edges: dict, raw_dir: Path,
                       sam, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'Imagen':<25} {'#':>4} {'Lado':<7} {'Punto':<8} {'dist_BL (mm)':>14}")
    print("-" * 65)

    for img_name in sorted(qr_data.keys()):
        if img_name not in cut_edges or not cut_edges[img_name]:
            continue

        img_bgr    = cv2.imread(str(raw_dir / img_name))
        img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H_mat      = qr_data[img_name]["H"]
        qr_centers = qr_data[img_name]["qrs"]
        detections = cut_edges[img_name]
        BL_x, BL_y = qr_centers["BL"]

        masks = []
        for det in detections:
            try:
                masks.append(run_esam(img_rgb, det["bbox"], sam))
            except Exception as e:
                print(f"  [WARN] {img_name}: {e}")
                masks.append(np.zeros(img_rgb.shape[:2], dtype=bool))

        overlay = build_overlay(img_rgb, masks, contours=True)

        fig, ax = plt.subplots(figsize=(13, 13))
        ax.imshow(overlay)
        draw_qr_triangle(ax, qr_centers)
        ax.axhline(BL_y, color="orange", lw=2.2, linestyle="--")
        ax.text(10, BL_y - 14, "Borde inferior mesa", color="orange",
                fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55))

        for i, (mask, det) in enumerate(zip(masks, detections)):
            edge_pts = mask_edge_sample_points(mask)
            if edge_pts is None:
                continue
            n_pts = len(next(iter(edge_pts.values())))
            for side, pts in edge_pts.items():
                color = SIDE_COLORS[side]
                for j, (px, py) in enumerate(pts):
                    canon   = apply_H(H_mat, [[px, py]])[0]
                    dist_mm = max(0.0, (CANONICAL_H - canon[1]) * SCALE_Y)

                    ax.plot([px, px], [py, BL_y], color=color, lw=1.5,
                            linestyle=":", alpha=0.75, zorder=4)
                    marker = "^" if j == 0 else ("v" if j == n_pts - 1 else "o")
                    ax.plot(px, py, marker, color=color, ms=8, zorder=6,
                            markeredgecolor="black", markeredgewidth=0.7)
                    if j in (0, n_pts - 1):
                        ax.text(px + 6, (py + BL_y) / 2, f"{dist_mm:.1f}",
                                color=color, fontsize=8, fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
                                zorder=7)

                    label = "sup" if j == 0 else ("inf" if j == n_pts - 1 else f"int{j}")
                    print(f"{img_name:<25} {'#'+str(i+1):>4} {side:<7} {label:<8} {dist_mm:>14.1f}")

        patches = [
            mpatches.Patch(color="deepskyblue", label="Lado izquierdo"),
            mpatches.Patch(color="tomato",      label="Lado derecho"),
            mpatches.Patch(color="orange",      label="Borde inferior (BL)"),
        ]
        ax.legend(handles=patches, loc="upper right", fontsize=9,
                  facecolor="black", labelcolor="white", framealpha=0.7)
        ax.set_title("Multi-punto · 5 puntos × lado × segmentación", fontsize=10)
        ax.axis("off")
        fig.suptitle(img_name, fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_dir / f"{Path(img_name).stem}_multipoint.jpg",
                    dpi=100, bbox_inches="tight")
        plt.close()

    print(f"\nGuardado en: {out_dir.resolve()}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    assert QR_JSON.exists(),      f"No se encontró {QR_JSON.resolve()}"
    assert CUTEDGE_JSON.exists(), f"No se encontró {CUTEDGE_JSON.resolve()}"
    assert ESAM_WEIGHTS.exists(), f"Pesos no encontrados: {ESAM_WEIGHTS.resolve()}"

    with open(QR_JSON)      as f: qr_data   = json.load(f)
    with open(CUTEDGE_JSON) as f: cut_edges = json.load(f)

    sam = load_model(ESAM_WEIGHTS)
    print(f"EfficientSAM listo en {DEVICE}")
    print(f"QR data:       {len(qr_data)} imágenes")
    print(f"Cut-edge data: {len(cut_edges)} imágenes")
    print(f"SCALE_Y = {SCALE_Y:.4f} mm/px canónico")

    process_single_point(
        qr_data, cut_edges, RAW_DIR, sam,
        out_dir=Path("../outputs/figures/cut_edge_distances"),
    )
    process_multipoint(
        qr_data, cut_edges, RAW_DIR, sam,
        out_dir=Path("../outputs/figures/cut_edge_multipoint"),
    )

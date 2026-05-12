"""
Genera data/merged/ combinando:
  - new_samples_qr_rubber  → clases 1 (qr_code) y 2 (rubber_strip), labels bbox
  - new_samples_cut_edge   → clase  0 (cut_edge), labels polígono → bbox
                             + renombrado de ficheros al estilo multimedia_XX_jpg.rf.HASH

Estructura de salida:
  data/merged/
  ├── train/images/  train/labels/
  ├── valid/images/  valid/labels/
  └── data.yaml
"""

import re
import random
import shutil
from pathlib import Path

# ── Rutas ────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent
DATA_DIR      = SCRIPT_DIR / "data"
SRC_QR_RUBBER = DATA_DIR / "new_samples_qr_rubber"
SRC_CUT_EDGE  = DATA_DIR / "new_samples_cut_edge"
DST           = DATA_DIR / "merged"

# ── Clases a conservar por dataset ───────────────────────────────────────────
KEEP_QR_RUBBER = {1, 2}   # qr_code, rubber_strip  (descarta cut_edge=0)
KEEP_CUT_EDGE  = {0}       # cut_edge               (descarta qr_code=1)

# ── Split para cut_edge (no tiene valid propio) ───────────────────────────────
VALID_RATIO = 0.20
SEED        = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def rename_cut_edge(filename: str) -> str:
    """
    Multimedia-31-_jpg.rf.HASH.jpg  →  multimedia_31_jpg.rf.HASH.jpg
    """
    stem, sep, rest = filename.partition(".rf.")
    stem_norm = re.sub(r"_+", "_", stem.lower().replace("-", "_")).strip("_")
    return f"{stem_norm}{sep}{rest}"


def polygon_to_bbox(parts: list[str]) -> str:
    """
    Convierte una línea de polígono YOLO a bbox YOLO.
    parts = [class_id, x1, y1, x2, y2, ...]
    """
    cls    = parts[0]
    coords = list(map(float, parts[1:]))
    xs     = coords[0::2]
    ys     = coords[1::2]
    cx     = (min(xs) + max(xs)) / 2
    cy     = (min(ys) + max(ys)) / 2
    w      = max(xs) - min(xs)
    h      = max(ys) - min(ys)
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"


def filter_bbox(label_path: Path, keep: set) -> list[str]:
    """Devuelve las líneas bbox cuya clase está en keep."""
    out = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if parts and int(parts[0]) in keep:
            out.append(line + "\n")
    return out


def filter_polygon(label_path: Path, keep: set) -> list[str]:
    """Devuelve líneas de polígono filtradas y convertidas a bbox."""
    out = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if not parts or int(parts[0]) not in keep:
            continue
        if len(parts) > 5:          # polígono → convertir
            out.append(polygon_to_bbox(parts))
        else:                        # ya es bbox
            out.append(line + "\n")
    return out


# ── Crear estructura de salida ────────────────────────────────────────────────
for split in ("train", "valid"):
    (DST / split / "images").mkdir(parents=True, exist_ok=True)
    (DST / split / "labels").mkdir(parents=True, exist_ok=True)

print("── new_samples_qr_rubber ─────────────────────────────────────")
for split in ("train", "valid"):
    img_dir = SRC_QR_RUBBER / split / "images"
    lbl_dir = SRC_QR_RUBBER / split / "labels"
    if not img_dir.exists():
        continue
    imgs = sorted(img_dir.glob("*.jpg"))
    for img in imgs:
        shutil.copy2(img, DST / split / "images" / img.name)
        lbl = lbl_dir / (img.stem + ".txt")
        dst_lbl = DST / split / "labels" / (img.stem + ".txt")
        if lbl.exists():
            lines = filter_bbox(lbl, KEEP_QR_RUBBER)
            dst_lbl.write_text("".join(lines))
        else:
            dst_lbl.write_text("")
    print(f"  {split}: {len(imgs)} imágenes copiadas")

print("\n── new_samples_cut_edge ──────────────────────────────────────")
cut_imgs = sorted((SRC_CUT_EDGE / "train" / "images").glob("*.jpg"))
lbl_dir  = SRC_CUT_EDGE / "train" / "labels"

random.seed(SEED)
random.shuffle(cut_imgs)
n_val = max(1, int(len(cut_imgs) * VALID_RATIO))
split_map = {"valid": cut_imgs[:n_val], "train": cut_imgs[n_val:]}

for split, imgs in split_map.items():
    for img in imgs:
        new_name = rename_cut_edge(img.name)
        shutil.copy2(img, DST / split / "images" / new_name)
        lbl     = lbl_dir / (img.stem + ".txt")
        dst_lbl = DST / split / "labels" / (Path(new_name).stem + ".txt")
        if lbl.exists():
            lines = filter_polygon(lbl, KEEP_CUT_EDGE)
            dst_lbl.write_text("".join(lines))
        else:
            dst_lbl.write_text("")
    print(f"  {split}: {len(imgs)} imágenes copiadas/renombradas")

print("\n── data.yaml ─────────────────────────────────────────────────")
yaml_path = DST / "data.yaml"
yaml_path.write_text(
    "train: train/images\n"
    "val:   valid/images\n"
    "\n"
    "nc: 3\n"
    "names: ['cut_edge', 'qr_code', 'rubber_strip']\n"
)
print(f"  Escrito en {yaml_path}")

# ── Resumen ───────────────────────────────────────────────────────────────────
print("\n── Resumen ───────────────────────────────────────────────────")
for split in ("train", "valid"):
    n_imgs = len(list((DST / split / "images").glob("*.jpg")))
    n_lbls = len(list((DST / split / "labels").glob("*.txt")))
    print(f"  {split}: {n_imgs} imágenes, {n_lbls} labels")
print(f"\nDataset merged listo en: {DST.resolve()}")

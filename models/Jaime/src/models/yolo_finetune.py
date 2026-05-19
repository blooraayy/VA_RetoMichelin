"""
YOLOv9 fine-tune — Michelin Challenge

Clases:
  0: cut_edge      — borde de corte
  1: qr_code       — código QR
  2: rubber_strip  — banda de goma completa

Uso:
  python yolo_finetune.py                  # split + train + eval + infer
  python yolo_finetune.py --skip-split     # salta el split si ya está hecho
  python yolo_finetune.py --only-eval      # solo evaluación con pesos existentes
  python yolo_finetune.py --only-infer     # solo inferencia visual
"""

import argparse
import random
import shutil
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

# ── Rutas (relativas a models/Jaime/) ────────────────────────────────────────
_HERE     = Path(__file__).resolve().parents[2]   # models/Jaime/
DATA_ROOT = _HERE / "data" / "segmentation_cut_edge"
RAW_DIR   = _HERE / "data" / "raw"
RUNS_DIR  = _HERE / "data" / "runs"
OUT_DIR   = _HERE / "outputs" / "figures" / "yolo_infer"

# ── Defaults de entrenamiento ────────────────────────────────────────────────
DEFAULTS = dict(
    model      = "yolov9c.pt",
    epochs     = 100,
    imgsz      = 640,
    batch      = 8,
    run_name   = "michelin_v1",
    valid_ratio= 0.2,
    seed       = 42,
    patience   = 20,
)

CLASS_COLORS = {0: (0, 200, 255), 1: (255, 220, 0), 2: (255, 140, 0)}
CLASS_NAMES  = {0: "cut_edge", 1: "qr_code", 2: "rubber_strip"}


# ── 1. Split train / val ──────────────────────────────────────────────────────
def split_dataset(valid_ratio: float = 0.2, seed: int = 42) -> None:
    train_img = DATA_ROOT / "train" / "images"
    train_lbl = DATA_ROOT / "train" / "labels"
    valid_img = DATA_ROOT / "valid" / "images"
    valid_lbl = DATA_ROOT / "valid" / "labels"

    valid_img.mkdir(parents=True, exist_ok=True)
    valid_lbl.mkdir(parents=True, exist_ok=True)

    all_imgs = sorted(train_img.glob("*.jpg")) + sorted(train_img.glob("*.png"))
    print(f"[split] Imágenes en train: {len(all_imgs)}")

    random.seed(seed)
    random.shuffle(all_imgs)
    n_val = max(1, int(len(all_imgs) * valid_ratio))
    val_imgs = all_imgs[:n_val]

    moved = 0
    for img in val_imgs:
        lbl = train_lbl / (img.stem + ".txt")
        shutil.move(str(img), valid_img / img.name)
        if lbl.exists():
            shutil.move(str(lbl), valid_lbl / lbl.name)
        moved += 1

    print(f"[split] Movidas a valid: {moved}  |  Quedan en train: {len(all_imgs) - moved}")


# ── 2. Fine-tune ──────────────────────────────────────────────────────────────
def train(cfg: dict) -> Path:
    model = YOLO(cfg["model"])
    model.train(
        data      = str(DATA_ROOT / "data.yaml"),
        epochs    = cfg["epochs"],
        imgsz     = cfg["imgsz"],
        batch     = cfg["batch"],
        project   = str(RUNS_DIR),
        name      = cfg["run_name"],
        device    = 0,
        patience  = cfg["patience"],
        augment   = True,
        degrees   = 15,
        translate = 0.1,
        scale     = 0.3,
        fliplr    = 0.5,
        mosaic    = 0.5,
    )
    best = RUNS_DIR / cfg["run_name"] / "weights" / "best.pt"
    print(f"[train] Pesos guardados en: {best}")
    return best


# ── 3. Evaluación ─────────────────────────────────────────────────────────────
def evaluate(weights: Path) -> None:
    model   = YOLO(str(weights))
    metrics = model.val(data=str(DATA_ROOT / "data.yaml"), imgsz=640)
    print(f"[eval] mAP50:    {metrics.box.map50:.3f}")
    print(f"[eval] mAP50-95: {metrics.box.map:.3f}")


# ── 4. Inferencia visual ──────────────────────────────────────────────────────
def infer(weights: Path, n_images: int = 6) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model      = YOLO(str(weights))
    test_imgs  = sorted(RAW_DIR.glob("*.jpg"))[:n_images]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, img_path in zip(axes, test_imgs):
        img_bgr = cv2.imread(str(img_path))
        preds   = model(img_bgr, verbose=False)[0]
        overlay = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        for box in preds.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            col  = CLASS_COLORS.get(cls, (255, 255, 255))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), col, 3)
            cv2.putText(overlay, f"{CLASS_NAMES.get(cls, cls)} {conf:.2f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        ax.imshow(overlay)
        ax.set_title(img_path.name)
        ax.axis("off")

    plt.tight_layout()
    out_path = OUT_DIR / "inference_grid.jpg"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.show()
    print(f"[infer] Guardado en: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv9 fine-tune pipeline — Michelin")
    p.add_argument("--skip-split",  action="store_true", help="No dividir train/val")
    p.add_argument("--only-eval",   action="store_true", help="Solo evaluar")
    p.add_argument("--only-infer",  action="store_true", help="Solo inferencia visual")
    p.add_argument("--weights",     type=str, default=None, help="Pesos para eval/infer")
    p.add_argument("--model",       type=str, default=DEFAULTS["model"])
    p.add_argument("--epochs",      type=int, default=DEFAULTS["epochs"])
    p.add_argument("--imgsz",       type=int, default=DEFAULTS["imgsz"])
    p.add_argument("--batch",       type=int, default=DEFAULTS["batch"])
    p.add_argument("--run-name",    type=str, default=DEFAULTS["run_name"])
    p.add_argument("--valid-ratio", type=float, default=DEFAULTS["valid_ratio"])
    p.add_argument("--seed",        type=int, default=DEFAULTS["seed"])
    p.add_argument("--patience",    type=int, default=DEFAULTS["patience"])
    p.add_argument("--n-images",    type=int, default=6, help="Imágenes para inferencia")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = dict(
        model       = args.model,
        epochs      = args.epochs,
        imgsz       = args.imgsz,
        batch       = args.batch,
        run_name    = args.run_name,
        valid_ratio = args.valid_ratio,
        seed        = args.seed,
        patience    = args.patience,
    )

    weights = Path(args.weights) if args.weights else None

    if args.only_eval:
        assert weights, "--weights requerido para --only-eval"
        evaluate(weights)
        return

    if args.only_infer:
        assert weights, "--weights requerido para --only-infer"
        infer(weights, args.n_images)
        return

    if not args.skip_split:
        split_dataset(cfg["valid_ratio"], cfg["seed"])

    weights = train(cfg)
    evaluate(weights)
    infer(weights, args.n_images)


if __name__ == "__main__":
    main()

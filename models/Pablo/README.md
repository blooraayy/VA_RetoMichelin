# Michelin Challenge 2 — Grounded-SAM Pipeline

**Cut-edge detection, segmentation and metric estimation**
Computer Vision (Visión Artificial) · Universidad de León · 2025–2026
Reto Cluster FACyL + Michelin Aranda — Challenge 2 (cut-edge detection on assembled truck-tyre rubber strips)

Author: **Pablo Ruiz Morán** (deep-learning track, branch `paboct`).
Team comparison (see `../`): Jaime (DL) · Pablo (DL) · Miguel (classical) · David (classical).

---

## 1 — Problem statement

In Michelin Aranda's truck-tyre assembly workshop, an operator manually places rubber strips on a cylindrical drum, cuts them and joins their borders. The company wants to automate two of those manual checks via computer vision. This repository solves **Challenge 2**:

1. Detect / segment **QR markers** placed on the table → calibrate pixel ↔ mm.
2. Detect / segment **cut edges** (the gap between two pieces of rubber strip on the drum).
3. Estimate the **position / distance** of the cut edge from the bottom of the table at 10 sample points along the gap.

Images are taken with a smartphone, so a single global pixel-to-mm conversion is not always valid. Calibration is performed per image from the QR triangle (legs of 800 mm in the real world).

---

## 2 — Course context (rubric mapping)

The course project (Eduardo Fidalgo, ULE) is split in two blocks. This repository covers **Block 2: Proof of concept**:

| Course requirement | Where it lives in this repo |
|---|---|
| Selected method = one deep-learning approach per student | Grounded-SAM (Grounding DINO + SAM) — this folder |
| Detect / segment QRs | `model/grounded_sam.py` Stage 0, `model/qr_calibrator.py` |
| Detect / segment cut edges | `model/grounded_sam.py` Stages 1–2 |
| 10 measurements per detected object | `model/metrics.py` (`MeasurementEngine`) |
| Compare predictions vs labelled GT (MAE between predictions and reality, plus segmentation metrics) | `evaluate.py`, GT in `data/labels/_annotations.coco.json` |
| README.TXT with install/run + dataset link | `README.TXT` (next to this file) |
| 1-page summary PDF with results table & recommendation | Pending — to be added to the delivery ZIP |
| Block 2 deadline | **20/05/2026** |
| Written evaluation | **26/05/2026** |

The Block 1 research proposal (papers + presentation) lives in `../../docs/` and `../../docs/2_overleaf/`.

---

## 3 — Pipeline overview

```
┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│  Stage 0 — QRs     │   │  Stage 1 — Strips  │   │  Stage 2 — Cut gap │
│  Grounding DINO    │ → │  Grounding DINO    │ → │  Grounding DINO    │
│  + SAM masks       │   │  + SAM masks       │   │  + SAM (per ROI)   │
│  → calibration     │   │  → orientation     │   │  → 10 sample pts   │
│                    │   │     decision       │   │  → distance to     │
│                    │   │     (rotate 90° ?) │   │     table bottom   │
└────────────────────┘   └────────────────────┘   └────────────────────┘
```

Key design decisions taken in development:

- **Calibration runs *after* DINO QR detection**, not with `cv2.QRCodeDetector` — the pink stickers on the Michelin table are not standard QRs.
- **Orientation decision** (rotate 90° CW or not before Stage 2): when ≥2 strips are detected, compare the spread of their centres (`spread_x > spread_y · 1.2` ⇒ strips side-by-side ⇒ rotate). With 1 strip, fallback to the aspect ratio of the **SAM mask** (not the DINO bbox). Other heuristics (bbox aspect ratio, Sobel) were tried and discarded.
- **Brightness filter on cut candidates** (`CUT_GAP_BRIGHTNESS_MIN = 120`): real gaps expose the white table at their core, dark-on-dark false positives are rejected.
- **Classical fallback** (`utils/classical_cutgap.py`) only fires when Grounded-SAM finds zero cut gaps.

---

## 4 — Current results (22-image evaluation set)

Single source of truth: `data/outputs/evaluation_report/evaluation_report.csv`.

### Cut-edge segmentation (the main task)

| Metric | Value |
|---|---|
| Pixel Precision | **90.63 %** |
| Pixel Recall | **85.29 %** |
| Pixel F1 | **87.88 %** |
| Mean Pixel IoU | **0.686** |
| MAE (mm, predicted vs GT sample points) | **16.34 mm** |
| RMSE (mm) | **19.68 mm** |
| Vertical strips correctly handled | 7 / 9 |
| Horizontal strips correctly handled | 5 / 8 |
| No-cut images (48–52) with zero FP | 5 / 5 |

Breakdown vs the baseline run (no rotation decision + no brightness filter): **+18 pts in F1**, **+0.12 IoU**, FP eliminated on the 5 no-cut images.

### QR detection (used for calibration)

See tables `08_summary_table_qr_markers.png` and figures `11_*.png … 17_*.png` in `data/outputs/evaluation_report/`.

### Known failure case

**Multimedia_35.jpg** — Stage 1 splits the strip into a tall main band + a near-square detached chunk, the orientation decision does not fire and the small strip confuses the brightness filter. Documented; candidate improvements listed in §10.

---

## 5 — Project structure (MVC)

```
models/Pablo/
├── config/
│   └── config.py                ← all tuneable parameters (thresholds, prompts, paths)
├── controller/
│   └── pipeline_controller.py   ← CLI + orchestration (Controller)
├── model/
│   ├── qr_calibrator.py         ← QR detection & homography to mm
│   ├── grounded_sam.py          ← Grounding DINO + SAM, 3-stage pipeline
│   ├── coco_loader.py           ← COCO GT reader for evaluation
│   └── metrics.py               ← measurements + IoU / MAE evaluators
├── view/
│   └── visualiser.py            ← OpenCV / matplotlib rendering, file output
├── utils/
│   ├── utils.py                 ← shared helpers (image I/O, colour space, …)
│   ├── cutgap_postprocess.py    ← brightness filter, NMS, geometric checks
│   └── classical_cutgap.py      ← fallback when DINO finds zero gaps
├── data/
│   ├── images/                  ← 22 Michelin images (Multimedia_31..52)
│   ├── labels/_annotations.coco.json
│   └── outputs/                 ← per-image JSON/CSV/PNG + evaluation_report/
├── weights/
│   ├── sam_vit_h_4b8939.pth     ← SAM ViT-H checkpoint (must be downloaded)
│   └── groundingdino_swint_ogc.pth (legacy — NOT used, DINO is loaded from HuggingFace)
├── evaluate.py                  ← offline evaluation report + 17 charts + CSV
├── rename_dataset.py            ← one-off utility to normalise filenames
├── README.md
└── README.TXT                   ← Block 2 deliverable (install/run + dataset)
```

---

## 6 — Installation

Tested on **Python 3.10**, **CUDA 12.x**, NVIDIA RTX-class GPU. CPU mode works but Stage 1+2 takes ~30 s/img.

All dependencies are pinned in [`requirements.txt`](./requirements.txt).

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# GPU only: install the CUDA 12.1 build of PyTorch FIRST so the next step
# does not pull the CPU wheel. Skip this line on CPU-only setups.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Everything else (and torch/torchvision on CPU-only setups):
pip install -r requirements.txt
```

`requirements.txt` covers: `torch`, `torchvision`, `transformers` (Grounding DINO), `segment-anything` (from GitHub), `opencv-python`, `numpy`, `pandas`, `matplotlib`, `Pillow`, `pycocotools`.

Grounding DINO is loaded automatically from HuggingFace (`IDEA-Research/grounding-dino-base`) on first run and cached under `~/.cache/huggingface/`. **You do not need to download it manually.**

The only weight you must download yourself is the SAM ViT-H checkpoint:

| File | URL |
|------|-----|
| `sam_vit_h_4b8939.pth` | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth |

Place it under `weights/sam_vit_h_4b8939.pth`.

---

## 7 — Running the pipeline

### Full folder (recommended for evaluation)

```bash
python controller/pipeline_controller.py --folder data/images/ --device cuda --no-show
```

This processes the 22 images, writes per-image annotated PNGs + JSON + CSV under `data/outputs/`, and produces `aggregate_summary.json`.

### Single image

```bash
python controller/pipeline_controller.py --image data/images/Multimedia_31.jpg --device cuda
```

### CPU only

```bash
python controller/pipeline_controller.py --folder data/images/ --device cpu --no-show
```

### Generate evaluation report (after running the pipeline)

```bash
python evaluate.py
```

Produces under `data/outputs/evaluation_report/`:
- 17 PNG charts (per-image bar charts, distributions, PR scatter, AP curve, processing time, separate QR tables)
- `evaluation_report.csv` — full per-image metrics table

---

## 8 — Tuning parameters

All thresholds and paths live in `config/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GDINO_BOX_THRESHOLD` | 0.12 | Min. detection confidence (QR + strip) |
| `GDINO_TEXT_THRESHOLD` | 0.10 | Min. text-alignment score (QR + strip) |
| `GDINO_BOX_THRESHOLD_CUT` | 0.17 | Min. confidence for cut-edge detection |
| `CUT_GAP_BRIGHTNESS_MIN` | 120.0 | 90th-percentile grayscale needed to accept a gap |
| `QR_MAX_AREA_FRAC` | 0.012 | Max. QR bbox area (filters rubber-strip FPs) |
| `NUM_SAMPLE_POINTS` | 10 | Sample points per cut edge (rubric requirement) |
| `QR_LEG_MM` | 800.0 | Physical length of QR triangle legs (mm) |
| `STRIP_MAX_DETECTIONS` | 2 | Max rubber strips passed to Stage 2 |

---

## 9 — Output files

Per image (`data/outputs/`):
- `<stem>_detection_<ts>.png` — bboxes, QR markers and cut-edge masks
- `<stem>_measurement_<ts>.png` — annotated distances to table bottom (mm)
- `<stem>_<ts>.json` — all numeric results for this image
- `<stem>_measurements_<ts>.csv` — one row per sampled point (px + mm + distance)

Aggregated:
- `aggregate_summary.json` — combined results for the full folder run
- `evaluation_report/*` — charts + CSV (after `evaluate.py`)

---

## 10 — Known limitations & future work

- **Multimedia_35** consistently fails (see §4). Fix candidates: improve Stage 1 strip segmentation, or take the orientation decision from the SAM mask of the tallest strip.
- Smartphone images lack a global pixel-to-mm scale; calibration must be redone per image from QRs. Images where the QR triangle is partially out of frame can degrade calibration.
- Grounded-SAM is heavy (≈ 0.5 GB DINO + 2.5 GB SAM). For a lighter solution, see Miguel/David's classical (Canny + watershed) baselines under `../Miguel/` and `../David/`.
- The legacy `weights/groundingdino_swint_ogc.pth` is from an early attempt at running the original DINO repo; it is no longer used and can be deleted to save 700 MB.

---

## 11 — Reproducing the reported numbers

```bash
python controller/pipeline_controller.py --folder data/images/ --device cuda --no-show
python evaluate.py
# Open data/outputs/evaluation_report/08_summary_table_cut_edges.png
```

Results in this README match commit `3982b6c` (`Threshold enhanced`).

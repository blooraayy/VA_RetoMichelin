# Michelin Challenge 2 — Grounded-SAM Pipeline
**Cut-edge detection, segmentation and metric estimation**  
Computer Vision · Universidad de León · 2025-2026

---

## Project structure (MVC)

```
michelin_grounded_sam/
├── config/
│   └── config.py                   ← All tuneable parameters
├── controller/
│   └── pipeline_controller.py      ← CLI + orchestration (Controller)
├── model/
│   ├── qr_calibrator.py            ← QR detection & homography (Model)
│   ├── grounded_sam.py             ← Grounding DINO + SAM (Model)
│   └── metrics.py                  ← Measurements & evaluation (Model)
├── view/
│   └── visualiser.py               ← Rendering & file output (View)
├── utils/
│   └── utils.py                    ← Shared helpers
├── data/
│   ├── images/                     ← Input images (place Michelin dataset here)
│   ├── labels/                     ← COCO ground-truth annotations
│   └── outputs/                    ← Annotated images + JSON/CSV results
├── weights/                        ← SAM model checkpoint (see below)
├── evaluate.py                     ← Offline evaluation report + charts
└── README.md
```

---

## 1 — Install dependencies

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# Core libraries (CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python matplotlib numpy pandas Pillow

# Grounding DINO — loaded automatically from HuggingFace (no manual download needed)
pip install transformers

# Segment Anything Model (SAM)
pip install git+https://github.com/facebookresearch/segment-anything.git
```

---

## 2 — Download model weights

Only the SAM checkpoint needs to be downloaded manually.  
Create a `weights/` directory and place the file there:

| File | URL |
|------|-----|
| `sam_vit_h_4b8939.pth` | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth |

```bash
mkdir weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P weights/
```

Grounding DINO (`IDEA-Research/grounding-dino-base`) is downloaded automatically from HuggingFace on first run and cached locally.

---

## 3 — Add images

Copy the Michelin dataset images into `data/images/`.  
Place the COCO ground-truth file at `data/labels/_annotations.coco.json`.

---

## 4 — Run the pipeline

### Full folder (recommended)
```bash
python controller/pipeline_controller.py --folder data/images/ --device cuda --no-show
```

### Single image
```bash
python controller/pipeline_controller.py --image data/images/Multimedia_31.jpg --device cuda
```

### CPU-only
```bash
python controller/pipeline_controller.py --folder data/images/ --device cpu --no-show
```

### Generate evaluation report and charts
```bash
python evaluate.py
```

---

## 5 — Tune parameters

All thresholds and paths live in `config/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GDINO_BOX_THRESHOLD` | 0.12 | Min. detection confidence (QR + strip) |
| `GDINO_TEXT_THRESHOLD` | 0.10 | Min. text-alignment score (QR + strip) |
| `GDINO_BOX_THRESHOLD_CUT` | 0.17 | Min. confidence for cut-edge detection |
| `QR_MAX_AREA_FRAC` | 0.012 | Max. QR bounding-box area (filters rubber-strip FPs) |
| `NUM_SAMPLE_POINTS` | 10 | Points sampled per cut edge (5 per border) |
| `QR_LEG_MM` | 800.0 | Physical length of QR triangle legs (mm) |
| `STRIP_MAX_DETECTIONS` | 2 | Max rubber strips passed to Stage 2 |

---

## 6 — Ground-truth format

Ground truth is provided as a single COCO JSON file:

```
data/labels/_annotations.coco.json
```

It contains polygon segmentation masks for each `cut_edge` instance across all images. The pipeline loads this file automatically when running in evaluation mode.

---

## 7 — Output files

For each processed image the pipeline writes to `data/outputs/`:

- `<stem>_detection_<timestamp>.png` — bounding boxes, QR markers and cut-edge masks
- `<stem>_measurement_<timestamp>.png` — annotated distances to table bottom (mm)
- `<stem>_<timestamp>.json` — all numeric results per image
- `<stem>_measurements_<timestamp>.csv` — one row per sampled point (px + mm + distance)
- `aggregate_summary.json` — combined results for full-folder runs

Running `evaluate.py` generates in `data/outputs/evaluation_report/`:

- `01_iou_f1_per_image.png` — pixel IoU and F1 per image
- `02_mae_rmse_per_image.png` — MAE and RMSE in mm per image
- `03_precision_recall_f1_per_image.png` — pixel-level P/R/F1
- `04_detection_level_per_image.png` — TP/FP/FN and detection P/R/F1
- `05_metric_distributions.png` — histograms of all metrics
- `06_pr_scatter.png` — precision–recall scatter with F1 iso-curves
- `07_detection_counts.png` — predicted vs ground-truth count per image
- `08_summary_table.png` — global summary table
- `09_ap_curve.png` — IoU-threshold sensitivity curve (AUC, AP50, AP75)
- `10_processing_time.png` — processing time per image (s/img, imgs/min)
- `evaluation_report.csv` — full per-image metrics table

---

## Dependencies summary

```
torch >= 2.0
torchvision
transformers          (Grounding DINO via HuggingFace)
segment-anything      (from GitHub)
opencv-python
matplotlib
numpy
pandas
Pillow
```

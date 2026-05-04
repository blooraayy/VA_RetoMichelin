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
│   ├── outputs/                    ← Annotated images + JSON results
│   └── labels/                     ← Optional ground-truth JSON labels
├── weights/                        ← Model checkpoints (see below)
└── README.md
```

---

## 1 — Install dependencies

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# Core libraries
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python matplotlib numpy

# Grounding DINO
pip install groundingdino-py
# If the above fails, install from source:
# git clone https://github.com/IDEA-Research/GroundingDINO
# cd GroundingDINO && pip install -e .

# Segment Anything Model (SAM)
pip install git+https://github.com/facebookresearch/segment-anything.git

# Optional: pyzbar (fallback QR decoder)
pip install pyzbar
```

---

## 2 — Download model weights

Create a `weights/` directory at the project root and download:

| File | URL |
|------|-----|
| `groundingdino_swint_ogc.pth` | https://github.com/IDEA-Research/GroundingDINO/releases |
| `sam_vit_h_4b8939.pth` | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth |

```bash
mkdir weights
# Grounding DINO config is bundled with the package — check your install path
# and update GDINO_CONFIG in config/config.py accordingly.
```

---

## 3 — Add images

Copy the Michelin dataset images into `data/images/`.

---

## 4 — Run the pipeline

### Single image
```bash
python controller/pipeline_controller.py \
    --image data/images/Multimedia__31_.jpg \
    --output data/outputs/
```

### Full folder
```bash
python controller/pipeline_controller.py \
    --folder data/images/ \
    --output data/outputs/
```

### With ground-truth labels (evaluation mode)
```bash
python controller/pipeline_controller.py \
    --image  data/images/Multimedia__31_.jpg \
    --labels data/labels/Multimedia__31_.json \
    --output data/outputs/
```

### Headless / batch (no window)
```bash
python controller/pipeline_controller.py \
    --folder data/images/ \
    --no-show
```

### CPU-only
```bash
python controller/pipeline_controller.py \
    --image data/images/Multimedia__31_.jpg \
    --device cpu
```

---

## 5 — Tune parameters

All thresholds and paths live in `config/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GDINO_BOX_THRESHOLD` | 0.30 | Min. detection confidence |
| `GDINO_TEXT_THRESHOLD`| 0.25 | Min. text-alignment score |
| `PROMPT_EDGE` | `"rubber strip cut edge . cutting edge . edge of rubber band"` | Text prompt for cut edges |
| `NUM_SAMPLE_POINTS` | 10 | Points sampled per edge |
| `QR_LEG_MM` | 800.0 | Physical length of QR triangle legs (mm) |

---

## 6 — Ground-truth label format

JSON files in `data/labels/` must match the image filename stem:

```json
{
  "image": "Multimedia__31_.jpg",
  "edges": [
    {
      "edge_id": 1,
      "distances_to_bottom_mm": [12.3, 11.9, 12.1, 12.4, 12.2,
                                  12.0, 11.8, 12.3, 12.5, 12.1]
    },
    {
      "edge_id": 2,
      "distances_to_bottom_mm": [45.1, 45.3, 45.0, 45.2, 45.4,
                                  45.1, 44.9, 45.2, 45.3, 45.0]
    }
  ]
}
```

---

## 7 — Output files

For each processed image the pipeline writes to `data/outputs/`:

- `<stem>_<timestamp>.png`  — annotated image with masks, boxes, distances
- `<stem>_<timestamp>.json` — all numeric results
- `aggregate_summary.json`  — combined results for folder runs (batch)

---

## Dependencies summary

```
torch >= 2.0
torchvision
opencv-python
matplotlib
numpy
groundingdino-py  (or from source)
segment-anything  (from GitHub)
pyzbar            (optional, fallback QR decoder)
```

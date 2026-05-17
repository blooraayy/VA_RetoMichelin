# Adaptive Classical Vision Pipeline — Michelin Challenge 2
### Cut Edge Detection and Metric Estimation

**Author:** David Morán Gorgojo  
**Subject:** Visión Artificial — Grado en Ingeniería de Datos e Inteligencia Artificial  
**Universidad de León — Course 2025-2026**  
**Challenge:** Reto Michelin 2 — Detection and segmentation of cut edges in rubber strips

---

## Overview

This pipeline implements a fully classical, training-free computer vision solution for the automated detection of cut edges in rubber strips and the metric estimation of their positions relative to the bottom border of the inspection table.

The method is based on adaptive Canny edge detection with Otsu thresholding, QR-code-based geometric calibration, and gap boundary extraction via morphological analysis. It requires no GPU, no training data, and no deep learning framework.

---

## Results

| Metric | Value |
|--------|-------|
| Precision (cut_edge) | 78.1% |
| Recall (cut_edge) | 73.5% |
| F1 (cut_edge) | 75.8% |
| IoU (pixel-level) | 0.467 |
| F1 (pixel-level) | 0.583 |
| Mean inference time | ~0.024 s/image |
| Throughput | ~41 images/second (CPU) |

Evaluated against manually annotated ground truth (COCO format) provided by the group.

---

## Pipeline Structure

The pipeline is divided into 6 sequential notebooks. Each notebook reads from the outputs of the previous one and saves its results for the next.

```
data/images/
    │
    ▼
01_preprocessing.ipynb
    └── outputs/01_preprocessed/          (grayscale + Gaussian blur)
    │
    ▼
02_qr_calibration.ipynb
    └── outputs/02_rectified/             (800×800 px rectified images, 1 px = 1 mm)
    └── outputs/02_rectified/calibration.json
    │
    ▼
03_canny.ipynb
    └── outputs/03_canny/                 (binary edge maps)
    │
    ▼
04_gap_boundary.ipynb
    └── outputs/04_gap_boundaries/gap_boundaries.json
    │
    ▼
05_metric.ipynb
    └── outputs/05_metric/measurements.json
    └── outputs/05_metric/measurements_flat.csv
    └── outputs/05_metric/samples_flat.csv
    │
    ▼
06_evaluation.ipynb
    └── outputs/06_evaluation/report.json
    └── outputs/06_evaluation/summary.json
```

---

## Notebook Descriptions

### `01_preprocessing.ipynb`
**Input:** `data/images/`  
**Output:** `outputs/01_preprocessed/`

Converts each raw image to grayscale and applies a Gaussian blur (5×5 kernel) to suppress high-frequency noise. Multiple preprocessing variants (with/without CLAHE) are generated and compared. The selected variant (`gray_blur`) is saved as the main output.

The selected variant deliberately omits CLAHE because the raw grayscale histogram already presents a well-separated bimodal distribution (dark rubber ~50, white table ~210), which is ideal for Otsu thresholding in the next stages.

---

### `02_qr_calibration.ipynb`
**Input:** `outputs/01_preprocessed/`  
**Output:** `outputs/02_rectified/`, `calibration.json`

Detects the three QR codes present in each image using a visual detection approach based on:
- High internal edge density (characteristic of QR patterns)
- White surrounding ring (QR codes are printed on the white table)

Since the QR codes are too blurry to decode, they are detected purely visually. Each candidate is assigned to the nearest image corner (TL, TR, BL, BR) and the three best candidates are selected.

The exterior corners of the detected QR bounding boxes are used as reference points. These define a right triangle with legs of 800 mm. An affine transform is computed mapping the three reference points to a canonical 800×800 mm metric space, producing a rectified image where **1 pixel = 1 mm**.

Calibration data (affine matrix, px/mm scale, QR positions) is saved to `calibration.json` and reused by all subsequent notebooks.

---

### `03_canny.ipynb`
**Input:** `outputs/02_rectified/`  
**Output:** `outputs/03_canny/`

Applies intensity-based segmentation to separate the dark rubber band from the white table background, followed by Canny edge detection on the resulting binary mask.

Steps:
1. Otsu thresholding on the rectified grayscale image → binary rubber mask
2. Morphological closing (5×5 ellipse kernel) to fill small holes in the mask
3. Canny edge detection on the mask (fixed thresholds 50/150, which are reliable on a binary input)

Applying Canny on the mask rather than the raw image eliminates all internal rubber texture noise, producing clean contour edges of the rubber band boundaries.

---

### `04_gap_boundary.ipynb`
**Input:** `outputs/02_rectified/`, `outputs/03_canny/`  
**Output:** `outputs/04_gap_boundaries/gap_boundaries.json`

Detects the gaps between rubber strip segments (the cut edges) using a morphological gap detection approach that handles both vertical and horizontal orientations automatically.

Steps:
1. Build a strict bright-pixel mask (pixels brighter than threshold = possible gap/table)
2. Require rubber support on both sides of each bright pixel (side support filter)
3. Apply morphological closing to connect fragmented gap regions
4. Extract connected components and filter by geometry (length, thickness, area)
5. Validate each candidate using context (rubber must be present on both sides)
6. Extract paired boundary curves (left/right for vertical gaps, top/bottom for horizontal)
7. Sample 10 equally spaced measurement points along each detected gap

---

### `05_metric.ipynb`
**Input:** `outputs/04_gap_boundaries/gap_boundaries.json`, `calibration.json`  
**Output:** `outputs/05_metric/measurements.json`

Converts the detected gap boundaries into metric measurements. For each of the 10 sample points along each detected cut edge, computes the **distance from the sample point to the bottom border of the table** (y = 800 mm in the rectified coordinate system).

Since the rectified image has 1 px = 1 mm:
```
distance_to_bottom_mm = 800 - y_px
```

Robust statistics are computed per gap: mean, median, trimmed mean (removing the most extreme values), standard deviation, and coefficient of variation. The recommended final distance uses the trimmed mean when enough samples are available, falling back to median or mean otherwise.

---

### `06_evaluation.ipynb`
**Input:** `outputs/05_metric/measurements.json`, `calibration.json`, `data/labels/_annotations.coco.json`  
**Output:** `outputs/06_evaluation/`

Evaluates the pipeline outputs against manually annotated ground truth in COCO format. Ground truth bounding boxes are transformed into the rectified 800×800 coordinate system using the same affine matrix computed in NB02.

Metrics computed:
- **Precision, Recall, F1** — detection quality at IoU ≥ 0.30 and IoU ≥ 0.50
- **IoU (pixel-level)** — mask overlap between predicted and GT regions
- **MAE, RMSE** — metric accuracy of distance measurements in mm
- **Processing speed** — mean inference time per image and throughput (images/second)

---

## Installation

```bash
pip install opencv-python numpy matplotlib pandas pycocotools
```

No GPU or deep learning framework required. All processing runs on CPU.

---

## Dataset

Images are provided by **Michelin Aranda de Duero** as part of the II Premio al Talento Joven promoted by FACyL — Cluster de Automoción y Movilidad de Castilla y León.

Download link (provided by Michelin):  
https://drive.google.com/file/d/1xhwequDsqBpoJPC2JjYo3izU5yU6h_xy/view?usp=sharing

Place the images in `data/images/` and the COCO annotations file in `data/labels/_annotations.coco.json`.

Expected folder structure:
```
project/
├── data/
│   ├── images/
│   │   ├── Multimedia_31.jpg
│   │   ├── Multimedia_32.jpg
│   │   └── ...
│   └── labels/
│       └── _annotations.coco.json
├── outputs/              (created automatically by the notebooks)
├── 01_preprocessing.ipynb
├── 02_qr_calibration.ipynb
├── 03_canny.ipynb
├── 04_gap_boundary.ipynb
├── 05_metric.ipynb
├── 06_evaluation.ipynb
└── README.md
```

---

## Execution Order

Run the notebooks in order from 01 to 06. Each notebook saves its outputs before the next one starts. No notebook needs to be re-run unless its inputs change.

```
01_preprocessing → 02_qr_calibration → 03_canny → 04_gap_boundary → 05_metric → 06_evaluation
```

---

## Method Summary

| Step | Method | Tool |
|------|--------|------|
| Preprocessing | Grayscale + Gaussian blur | OpenCV |
| QR detection | Visual: edge density + white ring filter | OpenCV |
| Calibration | Affine transform to 800×800 mm space | `cv2.getAffineTransform` |
| Segmentation | Otsu threshold + morphological closing | OpenCV |
| Edge detection | Canny on binary mask | `cv2.Canny` |
| Gap detection | Morphological gap mask + connected components | OpenCV |
| Measurement | Distance to bottom border at 10 sample points | NumPy |
| Evaluation | IoU, Precision, Recall, F1, MAE, RMSE | NumPy + pycocotools |

---

## Key Design Decisions

**Why classical instead of deep learning?**  
The dataset is small (~22 images) and initially unlabelled. Classical methods require no training data, no GPU, and are fully interpretable and deployable in a factory environment with standard hardware.

**Why affine transform instead of full homography?**  
With only 3 QR reference points, only an affine transform can be estimated (homography requires 4 correspondences). The affine transform corrects translation, rotation, scale and shear, which is sufficient given the relatively constrained camera positions.

**Why Canny on the binary mask?**  
Applying Canny directly on the raw image picks up all internal rubber texture. Segmenting first and running Canny on the clean mask reduces noise drastically and produces reliable gap boundary contours.

**Why distance to bottom border instead of gap width?**  
The Reto 2 specification requires measuring distances from points along the cut edge to the bottom border of the table. This is the physically meaningful measurement for the robotic arm positioning task.

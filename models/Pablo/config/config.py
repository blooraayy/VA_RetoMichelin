# =============================================================================
# config/config.py
# Central configuration for the Michelin Challenge 2 - Grounded-SAM pipeline
# =============================================================================

import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
IMAGE_DIR   = os.path.join(DATA_DIR, "images")
OUTPUT_DIR  = os.path.join(DATA_DIR, "outputs")
LABEL_DIR   = os.path.join(DATA_DIR, "labels")
COCO_ANNOTATIONS = os.path.join(LABEL_DIR, "_annotations.coco.json")

# ── Grounding DINO ───────────────────────────────────────────────────────────
GDINO_CONFIG  = ""
GDINO_WEIGHTS = ""

# ── Prompts ──────────────────────────────────────────────────────────────────
PROMPT_QR = "pink square sticker . magenta square marker"
PROMPT_STRIP = "black rubber strip on white table"

PROMPT_CUT_CANDIDATES = [
    "gap between rubber pieces .",
    "white gap between black rubber pieces .",
    "separation between rubber pieces .",
    "cut edge of rubber strip .",
    "edge between two rubber strips .",
]

PROMPT_EDGE = PROMPT_STRIP

# ── Detection thresholds ─────────────────────────────────────────────────────
GDINO_BOX_THRESHOLD  = 0.12
GDINO_TEXT_THRESHOLD = 0.10

GDINO_BOX_THRESHOLD_CUT  = 0.17
GDINO_TEXT_THRESHOLD_CUT = 0.14

GDINO_BOX_THRESHOLD_CUT_FALLBACK  = 0.13
GDINO_TEXT_THRESHOLD_CUT_FALLBACK = 0.11

GDINO_BOX_THRESHOLD_CUT_LAST_RESORT  = 0.14
GDINO_TEXT_THRESHOLD_CUT_LAST_RESORT = 0.12

# Classical fallback is only used when Grounded-SAM detects no cut gap.
USE_CLASSICAL_FALLBACK = True
CLASSICAL_FALLBACK_MAX_DETECTIONS = 2

# ── Post-detection filters ───────────────────────────────────────────────────
QR_MAX_AREA_FRAC      = 0.012
QR_MIN_AREA_FRAC      = 1e-5
QR_ASPECT_RATIO_MIN   = 0.6
QR_ASPECT_RATIO_MAX   = 1.7

STRIP_MAX_AREA_FRAC   = 0.65
STRIP_MIN_AREA_FRAC   = 0.02

CUT_MIN_AREA_FRAC_OF_STRIP = 0.002
CUT_MAX_AREA_FRAC_OF_STRIP = 0.12
CUT_MIN_ASPECT_RATIO       = 8.0

NMS_IOU_THRESHOLD     = 0.5
STRIP_MAX_DETECTIONS  = 2

EDGE_MAX_AREA_FRAC    = STRIP_MAX_AREA_FRAC
EDGE_MIN_AREA_FRAC    = STRIP_MIN_AREA_FRAC

# ── SAM ──────────────────────────────────────────────────────────────────────
SAM_CHECKPOINT = "weights/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"

# ── QR calibration ───────────────────────────────────────────────────────────
QR_LEG_MM       = 800.0
QR_MIN_AREA_PX  = 50

# ── Sampling ─────────────────────────────────────────────────────────────────
NUM_SAMPLE_POINTS = 10

# ── Runtime ──────────────────────────────────────────────────────────────────
DEVICE = "cuda"
SAVE_VISUALISATIONS = True
SHOW_VISUALISATIONS = True

# ── Visualisation colours (BGR for OpenCV) ───────────────────────────────────
COLOR_QR_BOX    = (0,   255,   0)
COLOR_EDGE_BOX  = (0,   0,   255)
COLOR_EDGE_MASK = (0,   0,   255)
COLOR_SAMPLE_PT = (255, 0,     0)
COLOR_DISTANCE  = (255, 255,   0)
MASK_ALPHA      = 0.4

COLOR_STRIP_BOX  = (255, 200,   0)
COLOR_STRIP_MASK = (255, 200,   0)

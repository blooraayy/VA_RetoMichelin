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

# ── Grounding DINO ────────────────────────────────────────────────────────────
GDINO_CONFIG  = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_WEIGHTS = "weights/groundingdino_swint_ogc.pth"

# Text prompts sent to Grounding DINO
# Tip: use ". " as separator between prompts
PROMPT_QR   = "qr code"
PROMPT_EDGE = "rubber strip cut edge . cutting edge . edge of rubber band"

# Detection thresholds
GDINO_BOX_THRESHOLD  = 0.30   # minimum confidence for a bounding box
GDINO_TEXT_THRESHOLD = 0.25   # minimum confidence for text-box alignment

# ── SAM ───────────────────────────────────────────────────────────────────────
SAM_CHECKPOINT = "weights/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"          # "vit_h" | "vit_l" | "vit_b"

# ── QR calibration ────────────────────────────────────────────────────────────
# The three QR codes form a right triangle.
# The two legs (horizontal and vertical) measure exactly 800 mm each.
QR_LEG_MM = 800.0

# Minimum area (px²) for a QR detection to be considered valid
QR_MIN_AREA_PX = 500

# ── Measurement ───────────────────────────────────────────────────────────────
NUM_SAMPLE_POINTS = 10   # number of points sampled along each detected edge

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda"   # "cuda" | "cpu"

# ── Visualisation ─────────────────────────────────────────────────────────────
SAVE_VISUALISATIONS = True
SHOW_VISUALISATIONS = True    # set False for batch / headless runs

# Colours (BGR for OpenCV)
COLOR_QR_BOX    = (0,   255,   0)   # green
COLOR_EDGE_BOX  = (0,   0,   255)   # red
COLOR_EDGE_MASK = (0,   0,   255)   # red (semi-transparent)
COLOR_SAMPLE_PT = (255, 0,     0)   # blue
COLOR_DISTANCE  = (255, 255,   0)   # cyan
MASK_ALPHA      = 0.4

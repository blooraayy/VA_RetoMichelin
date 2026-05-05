# =============================================================================
# config/config.py
# Central configuration for the Michelin Challenge 2 - Grounded-SAM pipeline
#
# TWO-STAGE STRATEGY
# ------------------
# Stage 1 — DINO+SAM detects the rubber strip (large, high-contrast object)
# Stage 2 — DINO+SAM detects the fine cut line WITHIN each strip bbox
#
# The Stage 2 ground-truth (from COCO) shows the cut_edge is:
#   * very thin (width ~13-43 px) and long (~320-390 px)
#   * aspect ratio ~13:1 (range 7:1 to 28:1)
#   * area ~0.2 - 0.85 % of the full image (~5-15 % of one strip)
#   * orientation perpendicular to the strip's long axis
# =============================================================================

import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
IMAGE_DIR   = os.path.join(DATA_DIR, "images")
OUTPUT_DIR  = os.path.join(DATA_DIR, "outputs")
LABEL_DIR   = os.path.join(DATA_DIR, "labels")

# COCO ground-truth annotations (single file with all images, Roboflow-style)
COCO_ANNOTATIONS = os.path.join(LABEL_DIR, "_annotations.coco.json")

# ── Grounding DINO ───────────────────────────────────────────────────────────
# Loaded from HuggingFace; legacy paths kept for CLI backwards compatibility.
GDINO_CONFIG  = ""
GDINO_WEIGHTS = ""

# ── Prompts ──────────────────────────────────────────────────────────────────
# Stage 0 — QR markers (pink/magenta squares with white asterisk, NOT black-and-white QRs)
PROMPT_QR = "pink square sticker . magenta square marker"

# Stage 1 — Rubber strip (whole band)
PROMPT_STRIP = "black rubber strip on white table"

# Stage 2 — Cut edge line WITHIN a strip ROI.
# Tried sequentially; first prompt that yields a detection wins.
# Visual concepts work better than abstract ones for thin features.
PROMPT_CUT_CANDIDATES = [
    "gap between rubber pieces .",
    "white gap between black rubber pieces .",
    "separation between rubber pieces .",
    "cut edge of rubber strip .",
    "edge between two rubber strips .",
    "vertical separation between rubber pieces .",
    "horizontal separation between rubber pieces .",
    "seam between rubber pieces .",
]

# Legacy alias (some old code still imports PROMPT_EDGE)
PROMPT_EDGE = PROMPT_STRIP

# ── Detection thresholds ─────────────────────────────────────────────────────
# Stage 1 (strip) - confident detections only
GDINO_BOX_THRESHOLD  = 0.12
GDINO_TEXT_THRESHOLD = 0.10

# Stage 2 (cut) - lower thresholds: thin line is harder to detect
GDINO_BOX_THRESHOLD_CUT  = 0.08
GDINO_TEXT_THRESHOLD_CUT = 0.08

# ── Post-detection filters (applied AFTER Grounding DINO, BEFORE SAM) ────────
# Constants are derived from the dataset statistics summarised at the top.

# QR codes — small, ~square
QR_MAX_AREA_FRAC      = 0.05    # max 5 % of image area
QR_MIN_AREA_FRAC      = 1e-5
QR_ASPECT_RATIO_MIN   = 0.6
QR_ASPECT_RATIO_MAX   = 1.7

# Rubber strip — large but never the whole image
STRIP_MAX_AREA_FRAC   = 0.65
STRIP_MIN_AREA_FRAC   = 0.02

# Cut edge — relative to the parent STRIP bbox (not to the full image)
CUT_MIN_AREA_FRAC_OF_STRIP = 0.002
CUT_MAX_AREA_FRAC_OF_STRIP = 0.12
CUT_MIN_ASPECT_RATIO       = 6.0
# Max aspect not enforced (some are 28:1)

# Non-Maximum Suppression IoU threshold (per prompt, per stage)
NMS_IOU_THRESHOLD     = 0.5

# Backwards-compat aliases (used by old name in grounded_sam.py if any)
EDGE_MAX_AREA_FRAC    = STRIP_MAX_AREA_FRAC
EDGE_MIN_AREA_FRAC    = STRIP_MIN_AREA_FRAC

# ── SAM ──────────────────────────────────────────────────────────────────────
SAM_CHECKPOINT = "weights/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"          # "vit_h" | "vit_l" | "vit_b"

# ── QR calibration ───────────────────────────────────────────────────────────
QR_LEG_MM       = 800.0
QR_MIN_AREA_PX  = 50

# ── Sampling ─────────────────────────────────────────────────────────────────
NUM_SAMPLE_POINTS = 10

# ── Runtime ──────────────────────────────────────────────────────────────────
DEVICE             = "cuda"
SAVE_VISUALISATIONS = True
SHOW_VISUALISATIONS = True

# ── Visualisation colours (BGR for OpenCV) ───────────────────────────────────
COLOR_QR_BOX    = (0,   255,   0)   # green
COLOR_EDGE_BOX  = (0,   0,   255)   # red
COLOR_EDGE_MASK = (0,   0,   255)   # red (semi-transparent)
COLOR_SAMPLE_PT = (255, 0,     0)   # blue
COLOR_DISTANCE  = (255, 255,   0)   # cyan
MASK_ALPHA      = 0.4

# Optional: colour for the Stage-1 strip overlay (only used if the visualiser
# is updated to draw strips; the current visualiser ignores it).
COLOR_STRIP_BOX  = (255, 200,   0)   # orange-ish
COLOR_STRIP_MASK = (255, 200,   0)


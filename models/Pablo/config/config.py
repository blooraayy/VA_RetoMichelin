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
]

# Legacy alias (some old code still imports PROMPT_EDGE)
PROMPT_EDGE = PROMPT_STRIP

# ── Detection thresholds ─────────────────────────────────────────────────────
# Stage 1 (strip) - confident detections only
GDINO_BOX_THRESHOLD  = 0.12
GDINO_TEXT_THRESHOLD = 0.10

# Stage 2 (cut) - lower thresholds: thin line is harder to detect
# Raised from 0.08 → 0.14: high recall but too many false positives,
# especially in images with prominent vertical striations on the rubber.
# Raised from 0.08 → 0.14: too many FP.
# Raised from 0.14 → 0.22: too aggressive, real TP lost (images 40,44-47).
# Settled at 0.17: images 48-52 (no GT) stay at 0 predictions while
# real cuts in 40,44-47 are recovered.
GDINO_BOX_THRESHOLD_CUT  = 0.17
GDINO_TEXT_THRESHOLD_CUT = 0.14

# Fallback thresholds used ONLY on strips where no cut was found in the first
# pass, and only when at least one other strip already yielded a cut.
# This recovers the second cut per image without increasing FP globally.
GDINO_BOX_THRESHOLD_CUT_FALLBACK  = 0.13
GDINO_TEXT_THRESHOLD_CUT_FALLBACK = 0.11

# Last-resort thresholds: used on ALL strips when the entire image produced
# zero detections after both previous passes. Targets images like M35/36/37
# where cuts are subtle (confidence ~0.14) and no first-pass anchor exists.
GDINO_BOX_THRESHOLD_CUT_LAST_RESORT  = 0.14
GDINO_TEXT_THRESHOLD_CUT_LAST_RESORT = 0.12

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
# Raised from 6.0 → 8.0: real cuts have aspect ratio ~13:1; filtering
# out less elongated shapes reduces FP from texture/shadow artefacts.
CUT_MIN_ASPECT_RATIO       = 8.0
# Max aspect not enforced (some are 28:1)

# Non-Maximum Suppression IoU threshold (per prompt, per stage)
NMS_IOU_THRESHOLD     = 0.5
# Reduced from 3 → 2: dataset analysis showed images 31 and 36 produced
# 3 predictions vs 2 GT cuts because all 3 strips were passed to Stage 2.
STRIP_MAX_DETECTIONS  = 2     # max strips passed to Stage 2 (keeps highest-score ones)

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


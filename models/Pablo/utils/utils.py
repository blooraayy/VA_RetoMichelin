# =============================================================================
# utils/utils.py
# Shared utility functions used across the MVC layers
# =============================================================================

import os
import cv2
import numpy as np
from datetime import datetime


# ── Image I/O ────────────────────────────────────────────────────────────────

def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk in BGR format (OpenCV convention).

    Args:
        image_path: Absolute or relative path to the image file.

    Returns:
        BGR numpy array of shape (H, W, 3).

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If the image cannot be decoded.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not decode image: {image_path}")
    return img


def save_image(image: np.ndarray, output_path: str) -> None:
    """Save a BGR numpy array to disk, creating parent directories as needed."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR OpenCV image to RGB (required by PyTorch/PIL pipelines)."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image back to BGR."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


# ── Bounding-box helpers ──────────────────────────────────────────────────────

def xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [x, y, w, h]."""
    x1, y1, x2, y2 = box
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)


def box_center(box: np.ndarray) -> tuple:
    """Return the (cx, cy) centre of a [x1, y1, x2, y2] box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_area(box: np.ndarray) -> float:
    """Return the area of a [x1, y1, x2, y2] box."""
    x1, y1, x2, y2 = box
    return max(0.0, float((x2 - x1) * (y2 - y1)))


# ── Geometry ─────────────────────────────────────────────────────────────────

def euclidean_distance(p1: tuple, p2: tuple) -> float:
    """Euclidean distance between two 2-D points."""
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def sample_points_along_contour(contour: np.ndarray, n: int = 10) -> np.ndarray:
    """Sample *n* equally-spaced points along a contour.

    Args:
        contour: Array of shape (N, 1, 2) or (N, 2) of pixel coordinates.
        n:       Number of sample points.

    Returns:
        Array of shape (n, 2) with the sampled (x, y) pixel coordinates.
    """
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 2:
        return pts

    # Compute cumulative arc-length
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cumlen[-1]

    if total_length == 0:
        return pts[:n] if len(pts) >= n else pts

    sample_lens = np.linspace(0.0, total_length, n)
    sampled = np.zeros((n, 2), dtype=np.float32)
    for i, sl in enumerate(sample_lens):
        idx = np.searchsorted(cumlen, sl, side="right") - 1
        idx = np.clip(idx, 0, len(pts) - 2)
        seg_start = cumlen[idx]
        seg_end   = cumlen[idx + 1]
        t = (sl - seg_start) / (seg_end - seg_start + 1e-9)
        sampled[i] = pts[idx] + t * (pts[idx + 1] - pts[idx])
    return sampled


# ── Output naming ─────────────────────────────────────────────────────────────

def timestamped_filename(base_name: str, extension: str = ".png") -> str:
    """Return e.g. 'result_20260426_153012.png'."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{ts}{extension}"

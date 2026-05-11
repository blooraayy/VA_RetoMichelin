# =============================================================================
# utils/classical_cutgap.py
# Conservative classical fallback for cut-gap detection.
#
# This module is only used when Grounded-SAM fails to detect any cut gap.
# It searches for elongated light gaps between dark rubber strips using
# classical OpenCV preprocessing and geometric filters.
# =============================================================================

from __future__ import annotations

import cv2
import numpy as np


def is_valid_cutgap_box(box, img_w: int, img_h: int) -> bool:
    """Final geometric filter for obvious false positive cut gaps.

    Parameters
    ----------
    box:
        [x1, y1, x2, y2] box in full-image pixel coordinates.
    img_w, img_h:
        Image width and height.

    Returns
    -------
    bool
        True if the candidate is geometrically plausible.
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return False

    # Avoid detections stuck to the image border. Most false positives in
    # no-cut images appear near the outer border of the scene.
    margin = 25
    if x1 < margin or y1 < margin or x2 > img_w - margin or y2 > img_h - margin:
        return False

    # Avoid huge boxes.
    area = w * h
    img_area = float(img_w * img_h)
    if img_area <= 0:
        return False
    if area / img_area > 0.08:
        return False

    # A cut gap should be an elongated strip-like region.
    aspect_ratio = max(w, h) / max(min(w, h), 1.0)
    if aspect_ratio < 3.0:
        return False

    # Avoid lines almost spanning the whole image.
    if h > 0.85 * img_h:
        return False
    if w > 0.85 * img_w:
        return False

    return True


def _preprocess_roi(roi_bgr: np.ndarray) -> np.ndarray:
    """Return an edge map for a rubber-strip ROI."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Canny is intentionally conservative. The fallback should recover clear
    # missed gaps, not generate many new detections.
    edges = cv2.Canny(blur, 40, 120)

    return edges


def _candidate_from_contour(cnt, roi_w: int, roi_h: int):
    """Convert a contour to a filtered ROI-local candidate box."""
    x, y, w, h = cv2.boundingRect(cnt)

    if w <= 0 or h <= 0:
        return None

    area = w * h
    roi_area = float(roi_w * roi_h)
    if roi_area <= 0:
        return None

    aspect = max(w, h) / max(min(w, h), 1)
    area_frac = area / roi_area

    # The cut gap should be long and relatively thin.
    if aspect < 3.0:
        return None
    if area_frac < 0.001:
        return None
    if area_frac > 0.15:
        return None

    # Avoid candidates exactly on the ROI border.
    margin_x = 0.03 * roi_w
    margin_y = 0.03 * roi_h
    if x < margin_x or y < margin_y or (x + w) > roi_w - margin_x or (y + h) > roi_h - margin_y:
        return None

    return x, y, x + w, y + h


def detect_cutgap_classic(
    image_bgr: np.ndarray,
    strip_detections=None,
    max_detections: int = 1,
):
    """Detect cut gaps with a conservative classical fallback.

    Parameters
    ----------
    image_bgr:
        Full input image in OpenCV BGR format.
    strip_detections:
        Optional list of DetectionResult strip detections. If provided, the
        fallback searches only inside those strip ROIs. If not provided, it uses
        the full image.
    max_detections:
        Maximum number of returned candidates.

    Returns
    -------
    list[dict]
        Each dict contains:
        - mask: bool mask of shape (H, W)
        - box_xyxy: np.ndarray [x1, y1, x2, y2]
        - score: fallback confidence
        - source: "classical"
    """
    H, W = image_bgr.shape[:2]

    rois = []
    if strip_detections:
        for det in strip_detections:
            x1, y1, x2, y2 = det.box_xyxy.astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)
            if x2 > x1 + 5 and y2 > y1 + 5:
                rois.append((x1, y1, x2, y2))
    else:
        rois.append((0, 0, W, H))

    candidates = []

    for rx1, ry1, rx2, ry2 in rois:
        roi = image_bgr[ry1:ry2, rx1:rx2]
        roi_h, roi_w = roi.shape[:2]
        if roi_h <= 5 or roi_w <= 5:
            continue

        edges = _preprocess_roi(roi)

        # Connect fragmented lines. Use two kernels because gaps can be either
        # mostly horizontal or mostly vertical depending on image orientation.
        kernels = [
            cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 17)),
        ]

        merged = np.zeros_like(edges)
        for k in kernels:
            dil = cv2.dilate(edges, k, iterations=1)
            merged = cv2.bitwise_or(merged, dil)

        contours, _ = cv2.findContours(
            merged,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for cnt in contours:
            local_box = _candidate_from_contour(cnt, roi_w, roi_h)
            if local_box is None:
                continue

            lx1, ly1, lx2, ly2 = local_box
            full_x1 = int(rx1 + lx1)
            full_y1 = int(ry1 + ly1)
            full_x2 = int(rx1 + lx2)
            full_y2 = int(ry1 + ly2)

            box = np.array([full_x1, full_y1, full_x2, full_y2], dtype=np.float32)

            if not is_valid_cutgap_box(box, W, H):
                continue

            # IMPORTANT FIX:
            # OpenCV drawing functions do not support bool arrays.
            # Draw on uint8, then convert to bool.
            full_mask_u8 = np.zeros((H, W), dtype=np.uint8)
            cv2.rectangle(
                full_mask_u8,
                (full_x1, full_y1),
                (full_x2, full_y2),
                1,
                -1,
            )
            full_mask = full_mask_u8.astype(bool)

            # Prefer longer and thinner candidates.
            bw = max(full_x2 - full_x1, 1)
            bh = max(full_y2 - full_y1, 1)
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            score = min(0.20, 0.08 + 0.01 * aspect)

            candidates.append({
                "mask": full_mask,
                "box_xyxy": box,
                "score": float(score),
                "source": "classical",
            })

    if not candidates:
        return []

    # Sort by score and keep a small number of candidates.
    candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)
    return candidates[:max_detections]

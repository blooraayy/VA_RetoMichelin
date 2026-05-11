# =============================================================================
# utils/classical_cutgap.py
# Conservative classical fallback for cut-gap detection.
#
# Use this file as: utils/classical_cutgap.py
# It is still a fallback: Grounded-SAM remains the main/deep method.
# =============================================================================

from __future__ import annotations

import cv2
import numpy as np


def is_valid_cutgap_box(box, img_w: int, img_h: int) -> bool:
    """Geometric filter for obvious false-positive cut gaps."""
    x1, y1, x2, y2 = [float(v) for v in box]
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return False

    # Reject detections stuck to the image borders. This is especially useful
    # for images without cuts, where the model/fallback sometimes locks onto
    # the lower edge of the table or rubber piece.
    margin = 35
    if x1 < margin or y1 < margin or x2 > img_w - margin or y2 > img_h - margin:
        return False

    area = w * h
    img_area = float(img_w * img_h)
    if img_area <= 0:
        return False
    if area / img_area > 0.06:
        return False

    aspect = max(w, h) / max(min(w, h), 1.0)
    if aspect < 3.0:
        return False

    # Avoid almost full-image lines.
    if h > 0.80 * img_h:
        return False
    if w > 0.80 * img_w:
        return False

    return True


def _box_intersection_frac(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    return inter / area_a


def _candidate_has_gap_contrast(gray_roi: np.ndarray, box, pad: int = 8) -> bool:
    """Check that candidate is a bright gap with dark rubber on both sides.

    This rejects many false positives on outer borders. For a real cut gap,
    the candidate zone is usually brighter than the rubber immediately on both
    sides of the gap.
    """
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    H, W = gray_roi.shape[:2]
    x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W, x2))
    y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        return False

    bw = x2 - x1
    bh = y2 - y1
    gap = gray_roi[y1:y2, x1:x2]
    if gap.size == 0:
        return False

    gap_med = float(np.median(gap))
    is_horizontal = bw >= bh

    if is_horizontal:
        p = max(pad, int(1.5 * bh))
        above = gray_roi[max(0, y1 - p):y1, x1:x2]
        below = gray_roi[y2:min(H, y2 + p), x1:x2]
        if above.size == 0 or below.size == 0:
            return False
        side_med = (float(np.median(above)) + float(np.median(below))) / 2.0
    else:
        p = max(pad, int(1.5 * bw))
        left = gray_roi[y1:y2, max(0, x1 - p):x1]
        right = gray_roi[y1:y2, x2:min(W, x2 + p)]
        if left.size == 0 or right.size == 0:
            return False
        side_med = (float(np.median(left)) + float(np.median(right))) / 2.0

    # Real gap should be visibly brighter than adjacent rubber.
    return (gap_med - side_med) >= 18.0


def _candidate_from_component(x: int, y: int, w: int, h: int, roi_w: int, roi_h: int):
    if w <= 0 or h <= 0:
        return None

    area = w * h
    roi_area = float(roi_w * roi_h)
    if roi_area <= 0:
        return None

    aspect = max(w, h) / max(min(w, h), 1)
    area_frac = area / roi_area

    if aspect < 3.0:
        return None
    if area_frac < 0.0005:
        return None
    if area_frac > 0.12:
        return None

    # Do not reject candidates only because they are near the strip ROI border:
    # some true cuts are close to an end of a strip. Image-border rejection is
    # handled later in full-image coordinates.
    return x, y, x + w, y + h


def _extract_bright_gap_candidates(roi_bgr: np.ndarray):
    """Return ROI-local boxes of elongated bright gap candidates."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Dynamic bright threshold. It targets white/pale cut gaps but adapts to
    # exposure differences.
    p85 = np.percentile(blur, 85)
    p95 = np.percentile(blur, 95)
    thr = max(115, min(205, 0.55 * p85 + 0.45 * p95))
    bright = (blur >= thr).astype(np.uint8) * 255

    # Remove tiny noise.
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    candidates = []

    # Merge elongated structures in both orientations.
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (31, 5)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 31)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (19, 3)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 19)),
    ]

    for kernel in kernels:
        merged = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            local_box = _candidate_from_component(x, y, w, h, W, H)
            if local_box is None:
                continue
            if not _candidate_has_gap_contrast(gray, local_box):
                continue
            candidates.append(local_box)

    return candidates


def _extract_edge_gap_candidates(roi_bgr: np.ndarray):
    """Secondary Canny-based candidates for weak/low-contrast gaps."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blur, 35, 110)

    candidates = []
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25)),
    ]
    for kernel in kernels:
        merged = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            local_box = _candidate_from_component(x, y, w, h, W, H)
            if local_box is None:
                continue
            if not _candidate_has_gap_contrast(gray, local_box, pad=10):
                continue
            candidates.append(local_box)
    return candidates


def _nms_candidates(candidates, iou_thr: float = 0.35):
    if not candidates:
        return []

    boxes = np.array([c["box_xyxy"] for c in candidates], dtype=np.float32)
    scores = np.array([c["score"] for c in candidates], dtype=np.float32)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = inter / union
        order = order[1:][iou <= iou_thr]

    return [candidates[i] for i in keep]


def detect_cutgap_classic(
    image_bgr: np.ndarray,
    strip_detections=None,
    max_detections: int = 1,
):
    """Detect cut gaps with a conservative classical fallback.

    Returns a list of dictionaries with mask, box_xyxy, score and source.
    """
    H, W = image_bgr.shape[:2]

    rois = []
    if strip_detections:
        for det in strip_detections:
            x1, y1, x2, y2 = det.box_xyxy.astype(int)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(W, x2); y2 = min(H, y2)
            if x2 > x1 + 10 and y2 > y1 + 10:
                rois.append((x1, y1, x2, y2))
    else:
        rois.append((0, 0, W, H))

    candidates = []

    for rx1, ry1, rx2, ry2 in rois:
        roi = image_bgr[ry1:ry2, rx1:rx2]
        roi_h, roi_w = roi.shape[:2]
        if roi_h <= 10 or roi_w <= 10:
            continue

        local_boxes = []
        local_boxes.extend(_extract_bright_gap_candidates(roi))
        local_boxes.extend(_extract_edge_gap_candidates(roi))

        for lx1, ly1, lx2, ly2 in local_boxes:
            full_x1 = int(rx1 + lx1)
            full_y1 = int(ry1 + ly1)
            full_x2 = int(rx1 + lx2)
            full_y2 = int(ry1 + ly2)
            box = np.array([full_x1, full_y1, full_x2, full_y2], dtype=np.float32)

            if not is_valid_cutgap_box(box, W, H):
                continue

            # Candidate must mostly lie inside its strip ROI.
            if _box_intersection_frac(box, [rx1, ry1, rx2, ry2]) < 0.80:
                continue

            full_mask_u8 = np.zeros((H, W), dtype=np.uint8)
            cv2.rectangle(full_mask_u8, (full_x1, full_y1), (full_x2, full_y2), 1, -1)
            full_mask = full_mask_u8.astype(bool)

            bw = max(full_x2 - full_x1, 1)
            bh = max(full_y2 - full_y1, 1)
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            length_frac = max(bw / max(roi_w, 1), bh / max(roi_h, 1))
            score = min(0.34, 0.10 + 0.012 * aspect + 0.08 * length_frac)

            candidates.append({
                "mask": full_mask,
                "box_xyxy": box,
                "score": float(score),
                "source": "classical",
            })

    if not candidates:
        return []

    candidates = _nms_candidates(candidates, iou_thr=0.35)
    candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)
    return candidates[:max_detections]

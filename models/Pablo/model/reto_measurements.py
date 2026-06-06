# =============================================================================
# model/reto_measurements.py
# Compute Reto 1 and Reto 2 measurements as specified in nuevas_normas.pdf.
#
# Reto 1 — 9 fixed X positions (80…720 mm from QR origin along the top edge):
#   DA  Distance from table origin (Y=0) to nearest edge of Band A
#   DB  Same for Band B
#   LA  Width of Band A (Y extent of the rubber strip)
#   LB  Width of Band B
#
# Reto 2 — 10 fixed Y offsets (5…230 mm from the TOP of each band's cut zone):
#   SA1  Gap width between upper cut borders of Band A at that Y slice
#   SA2  Gap including chamfer (upper-to-lower border distance); ≈SA1 for
#        top-view images where the chamfer is not visible
#   YSA  Global Y distance (from table edge, Y=0) to nearest cut border in A
#   SB1  Same metrics for Band B
#   SB2
#   YSB
# =============================================================================

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from config.config import RETO1_X_MM, RETO2_Y_OFFSETS_MM, RETO2_BAND_WIDTH_MM
from model.qr_calibrator import QRCalibrator


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Reto1Row:
    x_mm:  float
    DA:    Optional[float] = None
    DB:    Optional[float] = None
    LA:    Optional[float] = None
    LB:    Optional[float] = None


@dataclass
class Reto2Row:
    y_offset_mm: float          # distance from top of cut zone
    SA1: Optional[float] = None
    SA2: Optional[float] = None
    YSA: Optional[float] = None
    SB1: Optional[float] = None
    SB2: Optional[float] = None
    YSB: Optional[float] = None


# ── Engine ─────────────────────────────────────────────────────────────────────

class RetoMeasurementEngine:
    """Converts segmentation masks + homography into Reto 1 / Reto 2 tables."""

    # Resolution of the mm-space scan grid (smaller = more accurate but slower)
    _Y_SCAN_STEP_MM = 1.0
    _X_SCAN_STEP_MM = 1.0
    _Y_SCAN_MAX_MM  = 1200.0   # table + band < 1200 mm
    _X_SCAN_MAX_MM  = 1200.0   # same upper bound for horizontal scans

    def __init__(self, qr_calibrator: QRCalibrator, homography: np.ndarray,
                 H: int, W: int):
        self.calibrator  = qr_calibrator
        self.H_mat       = homography          # px → mm
        self.H_inv       = np.linalg.inv(homography)  # mm → px
        self.img_H       = H
        self.img_W       = W

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_reto1(
        self,
        strip_a_mask: Optional[np.ndarray],
        strip_b_mask: Optional[np.ndarray],
    ) -> list[Reto1Row]:
        rows = []
        for x_mm in RETO1_X_MM:
            da, la = self._band_edges_at_x(strip_a_mask, x_mm)
            db, lb = self._band_edges_at_x(strip_b_mask, x_mm)
            rows.append(Reto1Row(x_mm=x_mm, DA=da, DB=db, LA=la, LB=lb))
        return rows

    def compute_reto2(
        self,
        cut_a_mask:   Optional[np.ndarray],
        cut_b_mask:   Optional[np.ndarray],
        strip_a_mask: Optional[np.ndarray],
        strip_b_mask: Optional[np.ndarray],
        image_bgr:    Optional[np.ndarray] = None,
    ) -> list[Reto2Row]:
        # Find the global Y (in mm) of the top edge of each band's cut zone.
        # This is the top edge of the band itself (= DA from Reto 1).
        top_y_a = self._band_top_y_mm(strip_a_mask)
        top_y_b = self._band_top_y_mm(strip_b_mask)

        # Y of nearest cut border in each band (YSA / YSB):
        # = minimum global Y of any pixel belonging to the cut mask.
        ysa = self._nearest_cut_y_mm(cut_a_mask)
        ysb = self._nearest_cut_y_mm(cut_b_mask)

        rows = []
        for y_off in RETO2_Y_OFFSETS_MM:
            sa1 = sa2 = sb1 = sb2 = None

            if cut_a_mask is not None and top_y_a is not None:
                y_global = top_y_a + y_off
                sa1 = self._gap_width_at_y(cut_a_mask, y_global)
                sa2 = self._gap_upper_to_lower_at_y(
                    cut_a_mask, y_global, image_bgr, band="A"
                )

            if cut_b_mask is not None and top_y_b is not None:
                y_global = top_y_b + y_off
                sb1 = self._gap_width_at_y(cut_b_mask, y_global)
                sb2 = self._gap_upper_to_lower_at_y(
                    cut_b_mask, y_global, image_bgr, band="B"
                )

            rows.append(Reto2Row(
                y_offset_mm=y_off,
                SA1=sa1, SA2=sa2, YSA=ysa,
                SB1=sb1, SB2=sb2, YSB=ysb,
            ))
        return rows

    # ── Reto 1 helpers ────────────────────────────────────────────────────────

    def _band_edges_at_x(
        self,
        mask: Optional[np.ndarray],
        x_mm: float,
    ) -> tuple[Optional[float], Optional[float]]:
        """Return (nearest_y_mm, width_mm) of the band at a given X position.

        Scans a vertical line in mm space, maps each candidate point to a
        pixel, looks up the mask, and finds the Y extents of the band.
        Returns (None, None) if the mask is absent or has no pixels here.
        """
        if mask is None or not mask.any():
            return None, None

        y_vals = np.arange(0.0, self._Y_SCAN_MAX_MM, self._Y_SCAN_STEP_MM)
        pts_mm = np.column_stack([
            np.full(len(y_vals), x_mm, dtype=np.float32),
            y_vals.astype(np.float32),
        ])
        pts_px = self.calibrator.mm_to_pixels_bulk(pts_mm, self.H_mat)

        xi = np.round(pts_px[:, 0]).astype(int)
        yi = np.round(pts_px[:, 1]).astype(int)
        valid = (xi >= 0) & (xi < self.img_W) & (yi >= 0) & (yi < self.img_H)
        if not valid.any():
            return None, None

        in_band = mask[yi[valid], xi[valid]]
        hit_y   = y_vals[valid][in_band]

        if len(hit_y) < 3:
            return None, None

        nearest = float(hit_y.min())
        width   = float(hit_y.max() - hit_y.min())
        return nearest, width

    # ── Reto 2 helpers ────────────────────────────────────────────────────────

    def _band_top_y_mm(self, mask: Optional[np.ndarray]) -> Optional[float]:
        """Global Y (mm) of the nearest edge of a band to the table origin."""
        if mask is None or not mask.any():
            return None
        ys, xs = np.where(mask)
        pts_px = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
        pts_mm = self.calibrator.pixels_to_mm_bulk(pts_px, self.H_mat)
        return float(pts_mm[:, 1].min())

    def _nearest_cut_y_mm(self, cut_mask: Optional[np.ndarray]) -> Optional[float]:
        """Global Y (mm) of the cut border closest to the table edge."""
        return self._band_top_y_mm(cut_mask)

    def _gap_width_at_y(
        self,
        cut_mask: Optional[np.ndarray],
        y_mm: float,
    ) -> Optional[float]:
        """Horizontal gap width (mm) of the cut mask at a given global Y.

        Scans a horizontal line in mm space and measures the X extent of the
        cut mask pixels at that Y level.
        """
        if cut_mask is None or not cut_mask.any():
            return None

        x_vals = np.arange(0.0, self._X_SCAN_MAX_MM, self._X_SCAN_STEP_MM)
        pts_mm = np.column_stack([
            x_vals.astype(np.float32),
            np.full(len(x_vals), y_mm, dtype=np.float32),
        ])
        pts_px = self.calibrator.mm_to_pixels_bulk(pts_mm, self.H_mat)

        xi = np.round(pts_px[:, 0]).astype(int)
        yi = np.round(pts_px[:, 1]).astype(int)
        valid = (xi >= 0) & (xi < self.img_W) & (yi >= 0) & (yi < self.img_H)
        if not valid.any():
            return None

        in_cut = cut_mask[yi[valid], xi[valid]]
        hit_x  = x_vals[valid][in_cut]

        if len(hit_x) < 2:
            return None

        return float(hit_x.max() - hit_x.min())

    def _gap_upper_to_lower_at_y(
        self,
        cut_mask: Optional[np.ndarray],
        y_mm: float,
        image_bgr: Optional[np.ndarray],
        band: str = "A",
    ) -> Optional[float]:
        """SA2: distance between the upper and lower black rubber edges of the cut.

        For a top-view image where the chamfer is not visible, this equals the
        gap width (SA1). We detect the rubber edges by looking at the dark
        pixels flanking the white gap along a horizontal scan line.
        If image_bgr is provided we use image intensity for a more accurate
        estimate; otherwise we fall back to SA1.
        """
        sa1 = self._gap_width_at_y(cut_mask, y_mm)
        if sa1 is None:
            return None
        if image_bgr is None:
            return sa1

        # Map the target Y (in mm) to pixel space and scan horizontally.
        x_vals = np.arange(0.0, self._X_SCAN_MAX_MM, self._X_SCAN_STEP_MM)
        pts_mm = np.column_stack([
            x_vals.astype(np.float32),
            np.full(len(x_vals), y_mm, dtype=np.float32),
        ])
        pts_px = self.calibrator.mm_to_pixels_bulk(pts_mm, self.H_mat)
        xi = np.round(pts_px[:, 0]).astype(int)
        yi = np.round(pts_px[:, 1]).astype(int)
        valid = (xi >= 0) & (xi < self.img_W) & (yi >= 0) & (yi < self.img_H)
        if not valid.any():
            return sa1

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        intensity = gray[yi[valid], xi[valid]].astype(np.float32)
        x_scan = x_vals[valid]

        # Black rubber threshold: pixels below 80 are rubber edges
        DARK_THR = 80
        dark_mask = intensity < DARK_THR
        if dark_mask.sum() < 2:
            return sa1

        # Find the outermost dark pixels surrounding the white gap.
        # The gap is identified by the cut_mask. We look for the first dark
        # pixel to the left of the gap and the first to the right.
        in_cut = cut_mask[yi[valid], xi[valid]]
        cut_indices = np.where(in_cut)[0]
        if len(cut_indices) == 0:
            return sa1

        left_edge = cut_indices[0]
        right_edge = cut_indices[-1]

        # Search leftward for the nearest dark (rubber) pixel
        left_dark = np.where(dark_mask[:left_edge])[0]
        right_dark = np.where(dark_mask[right_edge + 1:])[0]

        if len(left_dark) == 0 or len(right_dark) == 0:
            return sa1

        x_left_rubber  = float(x_scan[left_dark[-1]])
        x_right_rubber = float(x_scan[right_edge + 1 + right_dark[0]])

        sa2 = x_right_rubber - x_left_rubber
        return round(float(sa2), 2) if sa2 > 0 else sa1


# ── Band assignment helper (used by pipeline) ─────────────────────────────────

def assign_bands(
    strip_detections: list,
    homography: np.ndarray,
    qr_calibrator: QRCalibrator,
) -> tuple[Optional[object], Optional[object]]:
    """Return (band_A_det, band_B_det) where A is the strip closest to Y=0.

    'Closest to Y=0' means smallest mean Y in mm space (nearest table edge).
    """
    if not strip_detections:
        return None, None
    if len(strip_detections) == 1:
        return strip_detections[0], None

    def mean_y_mm(det) -> float:
        if det.mask is not None and det.mask.any():
            ys, xs = np.where(det.mask)
        else:
            box = det.box_xyxy
            ys = np.array([float(box[1]), float(box[3])])
            xs = np.array([float(box[0]), float(box[2])])
        pts_px = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
        pts_mm = qr_calibrator.pixels_to_mm_bulk(pts_px, homography)
        return float(pts_mm[:, 1].mean())

    scored = sorted(strip_detections, key=mean_y_mm)
    return scored[0], scored[1]


def assign_cuts_to_bands(
    edge_detections: list,
    band_a,
    band_b,
) -> tuple[Optional[object], Optional[object]]:
    """Assign each cut edge to Band A or Band B based on bounding-box overlap.

    Returns (cut_for_A, cut_for_B).  If only one cut is detected it is
    assigned to whichever band overlaps it most.
    """
    if not edge_detections:
        return None, None

    def overlap(edge_det, band_det) -> float:
        if band_det is None:
            return 0.0
        ex1, ey1, ex2, ey2 = edge_det.box_xyxy
        bx1, by1, bx2, by2 = band_det.box_xyxy
        ix1, iy1 = max(ex1, bx1), max(ey1, by1)
        ix2, iy2 = min(ex2, bx2), min(ey2, by2)
        return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    cut_a = cut_b = None
    for edge in edge_detections:
        ov_a = overlap(edge, band_a)
        ov_b = overlap(edge, band_b)
        if ov_a >= ov_b:
            if cut_a is None or ov_a > overlap(cut_a, band_a):
                cut_a = edge
        else:
            if cut_b is None or ov_b > overlap(cut_b, band_b):
                cut_b = edge

    return cut_a, cut_b

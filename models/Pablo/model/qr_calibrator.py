# =============================================================================
# model/qr_calibrator.py
# QR-code detection and homography-based pixel-to-mm calibration.
#
# The three QR codes on the inspection surface form a right triangle whose
# two legs measure exactly QR_LEG_MM mm each (800 mm by default).
# Origin (0, 0) is defined as the centre of the top-left QR code.
# =============================================================================

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from config.config import QR_LEG_MM, QR_MIN_AREA_PX


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class QRResult:
    """All information extracted from the three QR codes."""
    centres_px: list[tuple[float, float]]   # pixel centres (x, y), one per QR
    homography: Optional[np.ndarray]        # 3×3 H that maps pixel → mm
    px_per_mm: float                        # average scale factor
    success: bool                           # True if calibration succeeded
    message: str = ""                       # human-readable status


# ── Detector ─────────────────────────────────────────────────────────────────

class QRCalibrator:
    """Detects the three QR codes in an image and builds a homography matrix
    so that any pixel coordinate can be expressed in millimetres relative to
    the top-left QR code.

    Usage
    -----
    calibrator = QRCalibrator()
    result     = calibrator.calibrate(bgr_image)
    mm_point   = calibrator.pixel_to_mm((px, py), result.homography)
    """

    def __init__(self, qr_leg_mm: float = QR_LEG_MM,
                 min_area_px: float = QR_MIN_AREA_PX):
        self.qr_leg_mm   = qr_leg_mm
        self.min_area_px = min_area_px

        # OpenCV QR decoder
        self._detector = cv2.QRCodeDetector()

    # ── Public API ───────────────────────────────────────────────────────────

    def calibrate(self, image_bgr: np.ndarray) -> QRResult:
        """Detect QR codes and compute the homography.

        Args:
            image_bgr: Full-resolution BGR image from the smartphone.

        Returns:
            QRResult with homography matrix and scale factor.
        """
        centres = []

        if len(centres) < 3:
            return QRResult(
                centres_px=centres,
                homography=None,
                px_per_mm=0.0,
                success=False,
                message=f"Only {len(centres)} QR code(s) found; need 3."
            )

        # Use only the three most prominent QR codes
        centres = centres[:3]

        try:
            H, px_per_mm = self._build_homography(centres)
        except Exception as exc:
            return QRResult(
                centres_px=centres,
                homography=None,
                px_per_mm=0.0,
                success=False,
                message=f"Homography failed: {exc}"
            )

        return QRResult(
            centres_px=centres,
            homography=H,
            px_per_mm=px_per_mm,
            success=True,
            message="Calibration OK"
        )

    @staticmethod
    def pixel_to_mm(point_px: tuple[float, float],
                    homography: np.ndarray) -> tuple[float, float]:
        """Map a single pixel coordinate to mm using the homography.

        Args:
            point_px:   (x, y) in pixel space.
            homography: 3×3 matrix H (pixel → mm).

        Returns:
            (x_mm, y_mm) in the QR reference frame.
        """
        pt = np.array([[[point_px[0], point_px[1]]]], dtype=np.float32)
        pt_mm = cv2.perspectiveTransform(pt, homography)
        return float(pt_mm[0, 0, 0]), float(pt_mm[0, 0, 1])

    @staticmethod
    def pixels_to_mm_bulk(points_px: np.ndarray,
                          homography: np.ndarray) -> np.ndarray:
        """Map an array of (N, 2) pixel coordinates to mm in one call."""
        pts = points_px.reshape(-1, 1, 2).astype(np.float32)
        pts_mm = cv2.perspectiveTransform(pts, homography)
        return pts_mm.reshape(-1, 2)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _detect_qr_centres(self, image_bgr: np.ndarray
                           ) -> list[tuple[float, float]]:
        """Return pixel centres of all detected QR codes, largest first."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # cv2.QRCodeDetector can find multiple codes via detectAndDecodeMulti
        retval, decoded_info, points, _ = \
            self._detector.detectAndDecodeMulti(gray)

        centres: list[tuple[float, float]] = []
        if not retval or points is None:
            # Fallback: try pyzbar if available
            centres = self._detect_with_pyzbar(image_bgr)
            return centres

        for poly in points:
            # poly shape: (4, 2) corners of one QR code
            cx = float(poly[:, 0].mean())
            cy = float(poly[:, 1].mean())
            area = cv2.contourArea(poly.astype(np.float32))
            if area >= self.min_area_px:
                centres.append((cx, cy))

        # Sort by area (descending) for reproducibility
        if len(points) > 1:
            areas = [cv2.contourArea(p.astype(np.float32)) for p in points]
            centres = [c for _, c in sorted(
                zip(areas, centres), key=lambda x: x[0], reverse=True)]

        return centres

    @staticmethod
    def _detect_with_pyzbar(image_bgr: np.ndarray
                            ) -> list[tuple[float, float]]:
        """Fallback QR detection using pyzbar."""
        try:
            from pyzbar.pyzbar import decode as pyzbar_decode
            from pyzbar.pyzbar import ZBarSymbol
        except ImportError:
            return []

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        barcodes = pyzbar_decode(gray, symbols=[ZBarSymbol.QRCODE])
        centres = []
        for bc in barcodes:
            pts = np.array(bc.polygon, dtype=np.float32)
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
            centres.append((cx, cy))
        return centres

    def _build_homography(self, centres_px: list[tuple[float, float]]
                          ) -> tuple[np.ndarray, float]:
        """Compute the homography H (pixel → mm) from three QR centres.

        The assignment of QR codes to triangle vertices (top-left, top-right,
        bottom-left) is done automatically by spatial position.

        Returns:
            (H, px_per_mm)  where H is the 3×3 homography matrix.
        """
        pts = np.array(centres_px, dtype=np.float64)  # shape (3, 2)

        # ── Assign corners ─────────────────────────────────────────────────
        # top-left  → smallest x+y
        # top-right → largest  x, smallest y
        # bottom-left → smallest x, largest y
        # (the fourth corner of the right angle is not present)

        idx_tl = int(np.argmin(pts[:, 0] + pts[:, 1]))

        remaining = [i for i in range(3) if i != idx_tl]
        # top-right: largest x among remaining
        idx_tr = remaining[int(np.argmax(pts[remaining, 0]))]
        # bottom-left: the last one
        idx_bl = [i for i in remaining if i != idx_tr][0]

        src = np.float32([pts[idx_tl], pts[idx_tr], pts[idx_bl]])

        # Destination in mm: origin at top-left QR
        # top-left  → (0, 0)
        # top-right → (QR_LEG_MM, 0)
        # bottom-left → (0, QR_LEG_MM)
        L = self.qr_leg_mm
        dst = np.float32([[0.0, 0.0],
                          [L,   0.0],
                          [0.0, L  ]])

        # With only 3 points we use an affine transform (exact solution)
        H_affine = cv2.getAffineTransform(src, dst)
        # Embed in 3×3 for perspectiveTransform compatibility
        H = np.vstack([H_affine, [0.0, 0.0, 1.0]])

        # Estimate px_per_mm from the horizontal leg length in pixels
        leg_px = float(np.linalg.norm(pts[idx_tr] - pts[idx_tl]))
        px_per_mm = leg_px / L

        return H, px_per_mm

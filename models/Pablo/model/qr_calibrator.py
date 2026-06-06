# =============================================================================
# model/qr_calibrator.py
# QR-marker calibration using Grounding DINO detections.
#
# The pink/magenta square stickers are NOT standard black-and-white QR codes,
# so cv2.QRCodeDetector and pyzbar do not see them. Grounding DINO already
# detects them well — we reuse its bounding boxes here.
#
# Three QR markers form a right triangle whose two legs measure exactly
# QR_LEG_MM mm each (default 800 mm). Origin (0, 0) is the top-left QR centre.
# =============================================================================

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from config.config import QR_LEG_MM


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class QRResult:
    """All information extracted from the three QR markers."""
    centres_px: list[tuple[float, float]]
    homography: Optional[np.ndarray]
    px_per_mm:  float
    success:    bool
    message:    str = ""


# ── Calibrator ───────────────────────────────────────────────────────────────

class QRCalibrator:
    """Builds a pixel→mm homography from the QR markers detected by Grounded-SAM.

    New design (May 2026)
    ---------------------
    The previous version called ``cv2.QRCodeDetector`` directly on the image,
    but the markers in this dataset are pink stickers (not real QRs), so the
    OpenCV decoder and pyzbar both failed silently. Grounding DINO already
    detects them with the prompt ``"pink square sticker"``, and we reuse those
    detections here.

    Two entry points:
      * ``calibrate_from_detections(...)``  — preferred, uses DINO output
      * ``calibrate(image_bgr)``            — legacy fallback (cv2 decoder)
    """

    def __init__(self, qr_leg_mm: float = QR_LEG_MM):
        self.qr_leg_mm = qr_leg_mm
        self._detector = cv2.QRCodeDetector()  # legacy fallback only

    # ── New API : reuse Grounding DINO detections ────────────────────────────

    def calibrate_from_detections(
        self,
        qr_detections,                    # list[DetectionResult] from Grounded-SAM
        image_shape:    tuple[int, int],  # (H, W) of the source image
    ) -> QRResult:
        """Build the homography from the QR detections returned by Grounded-SAM.

        Filters out spurious DINO QR detections (e.g. one that lands inside the
        rubber strip — see Multimedia_47) by checking how close each detection
        is to the image border. Real QRs are near the corners; false positives
        tend to appear near the centre.
        """
        H, W = image_shape

        # Convert each DetectionResult into (cx, cy, score, distance_to_centre)
        candidates = []
        for det in qr_detections:
            x1, y1, x2, y2 = det.box_xyxy
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            # Distance from image centre, normalised to half-diagonal (0..1).
            half_diag = 0.5 * float(np.hypot(W, H))
            dist_centre = float(np.hypot(cx - W / 2.0, cy - H / 2.0)) / half_diag
            candidates.append((cx, cy, float(det.score), dist_centre))

        # A real QR sits in the outer ~30 % of the image (dist_centre > 0.55).
        DIST_TO_CENTRE_MIN = 0.55
        filtered = [c for c in candidates if c[3] >= DIST_TO_CENTRE_MIN]

        if len(filtered) < 3:
            # Not enough peripheral QRs — fall back to top-3 by score (legacy
            # behaviour) so calibration still has a chance of succeeding.
            filtered = sorted(candidates, key=lambda c: c[2], reverse=True)[:3]

        # Sort by distance-to-centre (descending) — strongest peripheral first.
        filtered.sort(key=lambda c: c[3], reverse=True)
        filtered = filtered[:3]

        if len(filtered) < 3:
            return QRResult(
                centres_px=[(c[0], c[1]) for c in candidates],
                homography=None,
                px_per_mm=0.0,
                success=False,
                message=f"Need 3 QR markers, only {len(filtered)} usable.",
            )

        centres = [(c[0], c[1]) for c in filtered]

        try:
            Hm, px_per_mm = self._build_homography(centres)
        except Exception as exc:
            return QRResult(
                centres_px=centres,
                homography=None,
                px_per_mm=0.0,
                success=False,
                message=f"Homography failed: {exc}",
            )

        # Sanity check: transformed image corners must have a reasonable Y range.
        # If the QR triangle is degenerate (wrong corners assigned), the Y range
        # in mm space will be wildly larger than the physical leg length.
        H_img, W_img = image_shape
        test_px = np.float32([
            [0.0,             float(H_img - 1)],
            [float(W_img - 1), float(H_img - 1)],
            [0.0,             0.0],
            [float(W_img - 1), 0.0],
        ])
        test_mm = cv2.perspectiveTransform(
            test_px.reshape(-1, 1, 2), Hm
        ).reshape(-1, 2)
        y_range_mm = float(test_mm[:, 1].max() - test_mm[:, 1].min())
        if y_range_mm > 2.5 * self.qr_leg_mm:
            return QRResult(
                centres_px=centres,
                homography=None,
                px_per_mm=0.0,
                success=False,
                message=(
                    f"Calibration sanity check failed: transformed Y range "
                    f"({y_range_mm:.0f} mm) > 2.5 × QR leg "
                    f"({self.qr_leg_mm:.0f} mm). "
                    f"Likely wrong QR_LEG_MM or misidentified marker corners."
                ),
            )

        return QRResult(
            centres_px=centres,
            homography=Hm,
            px_per_mm=px_per_mm,
            success=True,
            message="Calibration OK (from Grounding DINO detections)",
        )

    # ── Legacy API : direct cv2 / pyzbar decoding ────────────────────────────

    def calibrate(self, image_bgr: np.ndarray) -> QRResult:
        """Legacy entry point — kept for backwards compatibility.

        Tries cv2.QRCodeDetector then pyzbar. Will almost certainly fail on
        the Michelin pink-marker dataset; prefer ``calibrate_from_detections``.
        """
        centres = self._detect_qr_centres(image_bgr)
        if len(centres) < 3:
            return QRResult(
                centres_px=centres,
                homography=None,
                px_per_mm=0.0,
                success=False,
                message=f"Only {len(centres)} QR code(s) found via cv2/pyzbar; "
                        f"need 3.",
            )

        centres = centres[:3]
        try:
            H, px_per_mm = self._build_homography(centres)
        except Exception as exc:
            return QRResult(centres, None, 0.0, False, f"Homography failed: {exc}")

        return QRResult(centres, H, px_per_mm, True, "Calibration OK (legacy)")

    # ── Static helpers (unchanged from previous version) ─────────────────────

    @staticmethod
    def pixel_to_mm(point_px, homography):
        pt = np.array([[[point_px[0], point_px[1]]]], dtype=np.float32)
        pt_mm = cv2.perspectiveTransform(pt, homography)
        return float(pt_mm[0, 0, 0]), float(pt_mm[0, 0, 1])

    @staticmethod
    def pixels_to_mm_bulk(points_px, homography):
        pts = points_px.reshape(-1, 1, 2).astype(np.float32)
        pts_mm = cv2.perspectiveTransform(pts, homography)
        return pts_mm.reshape(-1, 2)

    # ── Private ──────────────────────────────────────────────────────────────

    def _detect_qr_centres(self, image_bgr: np.ndarray) -> list[tuple[float, float]]:
        """Legacy cv2/pyzbar detection — kept verbatim from previous version."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if max(h, w) > 2000:
            scale = 2000 / max(h, w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
        gray = cv2.equalizeHist(gray)

        retval, _, points, _ = self._detector.detectAndDecodeMulti(gray)
        if not retval or points is None:
            return self._detect_with_pyzbar(image_bgr)

        centres = [(float(p[:, 0].mean()), float(p[:, 1].mean())) for p in points]
        return centres

    @staticmethod
    def _detect_with_pyzbar(image_bgr: np.ndarray) -> list[tuple[float, float]]:
        try:
            from pyzbar.pyzbar import decode as pyzbar_decode
            from pyzbar.pyzbar import ZBarSymbol
        except Exception:
            return []
        try:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            barcodes = pyzbar_decode(gray, symbols=[ZBarSymbol.QRCODE])
            return [
                (float(np.array(bc.polygon)[:, 0].mean()),
                 float(np.array(bc.polygon)[:, 1].mean()))
                for bc in barcodes
            ]
        except Exception:
            return []

    def _build_homography(self, centres_px) -> tuple[np.ndarray, float]:
        """Compute affine pixel→mm transform from 3 QR centres.

        Handles two triangle layouts automatically:
          TL / TR / BL  →  (0,0) / (L,0) / (0,L)   [old dataset, pink stickers]
          TL / TR / BR  →  (0,0) / (L,0) / (L,L)   [new dataset, real QR codes]

        In both cases the X axis runs along the top edge (TL→TR) and the Y
        axis runs perpendicular into the table.  The origin (0,0) is always
        the TL corner so X measurements are referenced from the LEFT side.
        """
        pts = np.array(centres_px, dtype=np.float64)
        L   = self.qr_leg_mm

        idx_tl = int(np.argmin(pts[:, 0] + pts[:, 1]))
        remaining = [i for i in range(3) if i != idx_tl]
        idx_tr = remaining[int(np.argmax(pts[remaining, 0]))]
        idx_3rd = [i for i in remaining if i != idx_tr][0]

        # Determine whether the 3rd point is bottom-LEFT (x ≈ TL) or
        # bottom-RIGHT (x ≈ TR). Midpoint of TL→TR is the threshold.
        mid_x = 0.5 * (pts[idx_tl, 0] + pts[idx_tr, 0])
        if pts[idx_3rd, 0] <= mid_x:
            # BL layout (original dataset)
            src = np.float32([pts[idx_tl], pts[idx_tr], pts[idx_3rd]])
            dst = np.float32([[0.0, 0.0], [L, 0.0], [0.0, L]])
        else:
            # BR layout (new dataset — third QR is below TR, not below TL)
            src = np.float32([pts[idx_tl], pts[idx_tr], pts[idx_3rd]])
            dst = np.float32([[0.0, 0.0], [L, 0.0], [L, L]])

        H_aff = cv2.getAffineTransform(src, dst)
        H = np.vstack([H_aff, [0.0, 0.0, 1.0]])

        leg_px    = float(np.linalg.norm(pts[idx_tr] - pts[idx_tl]))
        px_per_mm = leg_px / L

        if not (0.2 <= px_per_mm <= 15.0):
            raise ValueError(
                f"px_per_mm={px_per_mm:.4f} outside [0.2, 15.0]; "
                f"check QR_LEG_MM ({L:.0f} mm) or marker positions."
            )

        return H, px_per_mm

    @staticmethod
    def mm_to_pixels_bulk(points_mm: np.ndarray, homography: np.ndarray) -> np.ndarray:
        """Inverse of pixels_to_mm_bulk: convert mm coordinates back to pixels."""
        H_inv = np.linalg.inv(homography)
        pts = points_mm.reshape(-1, 1, 2).astype(np.float32)
        pts_px = cv2.perspectiveTransform(pts, H_inv)
        return pts_px.reshape(-1, 2)

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
from itertools import combinations
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

        # A real QR sits in the outer ~30 % of the image.
        # New competition images have QRs slightly closer to centre → 0.40.
        DIST_TO_CENTRE_MIN = 0.40
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
        # Threshold 3.5× instead of 2.5× — some camera angles show significant grey
        # background beyond the table, pushing full-image corners to ~3× the QR leg
        # while the triangle itself is still correct (e.g. Pos1 images, range ~2.6–3.1×).
        # Genuinely degenerate homographies produce ranges of 5–10×.
        if y_range_mm > 3.5 * self.qr_leg_mm:
            # Primary 3-QR selection failed the sanity check.
            # Try all combinations of 3 from the top-10 candidates (by peripherality)
            # before giving up — one of the top-3 might be a false positive.
            top_n = sorted(candidates, key=lambda c: c[3], reverse=True)[:min(10, len(candidates))]
            print(f"[QR] Sanity failed — retrying with {len(list(combinations(range(len(top_n)), 3)))} combos from top-{len(top_n)} candidates")
            best_y = y_range_mm
            best_result = None
            for trio in combinations(range(len(top_n)), 3):
                trial_centres = [(top_n[i][0], top_n[i][1]) for i in trio]
                try:
                    Hm_t, px_t = self._build_homography(trial_centres)
                except Exception:
                    continue
                test_mm_t = cv2.perspectiveTransform(
                    test_px.reshape(-1, 1, 2), Hm_t
                ).reshape(-1, 2)
                y_range_t = float(test_mm_t[:, 1].max() - test_mm_t[:, 1].min())
                if y_range_t < best_y:
                    best_y = y_range_t
                    best_result = QRResult(
                        centres_px=trial_centres,
                        homography=Hm_t,
                        px_per_mm=px_t,
                        success=y_range_t <= 3.5 * self.qr_leg_mm,
                        message=(
                            "Calibration OK (QR combo retry)"
                            if y_range_t <= 3.5 * self.qr_leg_mm
                            else f"Best combo still failed: Y range {y_range_t:.0f} mm"
                        ),
                    )
                if y_range_t <= 3.5 * self.qr_leg_mm:
                    print(f"[QR] Combo retry succeeded: Y range {y_range_t:.0f} mm")
                    return best_result
            if best_result is not None:
                print(f"[QR] All combos failed; best Y range {best_y:.0f} mm")
                if best_result.success:
                    return best_result
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

        Handles all three realised triangle layouts automatically:
          TL / TR / BL  →  (0,0) / (L,0) / (0,L)   [old dataset, pink stickers]
          TL / TR / BR  →  (0,0) / (L,0) / (L,L)   [new dataset, real QR codes]
          TL / BL / BR  →  (0,0) / (0,L) / (L,L)   [Pos5-style, only one top QR]

        In all cases X runs along the top edge (or the single top→right leg) and Y
        runs perpendicular into the table.  Origin (0,0) is always the TL corner.
        """
        pts = np.array(centres_px, dtype=np.float64)
        L   = self.qr_leg_mm

        # Classify markers into "top row" vs "bottom row" by splitting at the
        # largest Y gap.  This handles all layouts including TL/BL/BR (Pos5)
        # where median-based splitting would misclassify BL as a top point.
        ys_sorted = np.sort(pts[:, 1])
        y_gaps    = np.diff(ys_sorted)                         # 2 values for 3 pts
        split_y   = 0.5 * (ys_sorted[int(np.argmax(y_gaps))]
                           + ys_sorted[int(np.argmax(y_gaps)) + 1])
        top_mask  = pts[:, 1] < split_y
        n_top     = int(top_mask.sum())

        if n_top == 2:
            # Standard layouts: two markers at the top, one at the bottom.
            top_pts = pts[top_mask]
            bot_pt  = pts[~top_mask][0]

            tl = top_pts[int(np.argmin(top_pts[:, 0]))]
            tr = top_pts[int(np.argmax(top_pts[:, 0]))]
            mid_x = 0.5 * (tl[0] + tr[0])
            if bot_pt[0] <= mid_x:
                # TL / TR / BL
                src = np.float32([tl, tr, bot_pt])
                dst = np.float32([[0.0, 0.0], [L, 0.0], [0.0, L]])
            else:
                # TL / TR / BR
                src = np.float32([tl, tr, bot_pt])
                dst = np.float32([[0.0, 0.0], [L, 0.0], [L, L]])
            leg_px = float(np.linalg.norm(tr - tl))

        elif n_top == 1:
            # TL / BL / BR layout: only the top-left QR is visible at the top.
            tl      = pts[top_mask][0]
            bot_pts = pts[~top_mask]
            bl      = bot_pts[int(np.argmin(bot_pts[:, 0]))]
            br      = bot_pts[int(np.argmax(bot_pts[:, 0]))]
            src = np.float32([tl, bl, br])
            dst = np.float32([[0.0, 0.0], [0.0, L], [L, L]])
            # Scale estimate: use the longer bottom leg (BL→BR ≈ L)
            leg_px = float(np.linalg.norm(br - bl))

        else:
            # All three in one row — degenerate; fall back to original logic.
            idx_tl  = int(np.argmin(pts[:, 0] + pts[:, 1]))
            remaining = [i for i in range(3) if i != idx_tl]
            idx_tr  = remaining[int(np.argmax(pts[remaining, 0]))]
            idx_3rd = [i for i in remaining if i != idx_tr][0]
            mid_x   = 0.5 * (pts[idx_tl, 0] + pts[idx_tr, 0])
            if pts[idx_3rd, 0] <= mid_x:
                src = np.float32([pts[idx_tl], pts[idx_tr], pts[idx_3rd]])
                dst = np.float32([[0.0, 0.0], [L, 0.0], [0.0, L]])
            else:
                src = np.float32([pts[idx_tl], pts[idx_tr], pts[idx_3rd]])
                dst = np.float32([[0.0, 0.0], [L, 0.0], [L, L]])
            leg_px = float(np.linalg.norm(pts[idx_tr] - pts[idx_tl]))

        H_aff = cv2.getAffineTransform(src, dst)
        H = np.vstack([H_aff, [0.0, 0.0, 1.0]])

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

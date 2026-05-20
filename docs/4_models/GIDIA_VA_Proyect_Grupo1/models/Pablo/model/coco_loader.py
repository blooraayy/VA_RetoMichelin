# =============================================================================
# model/coco_loader.py
# Loads Roboflow COCO Segmentation annotations and generates ground truth:
#   - cut_edge masks and contours
#   - 10 auto-sampled measurement points per cut_edge
#   - qr_code bounding boxes
# =============================================================================

from __future__ import annotations

import os
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional

from pycocotools.coco import COCO
from utils.utils import sample_points_along_contour
from config.config import COCO_ANNOTATIONS, IMAGE_DIR, NUM_SAMPLE_POINTS


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class GTAnnotation:
    """Ground truth for one image."""
    image_id:   int
    file_name:  str
    image_path: str
    width:      int
    height:     int
    # cut_edge
    cut_edge_masks:   list[np.ndarray] = field(default_factory=list)  # (H,W) bool
    cut_edge_points:  list[np.ndarray] = field(default_factory=list)  # (10,2) px
    cut_edge_boxes:   list[np.ndarray] = field(default_factory=list)  # (4,) xyxy
    # qr_code
    qr_boxes:         list[np.ndarray] = field(default_factory=list)  # (4,) xyxy


# ── Loader ────────────────────────────────────────────────────────────────────

class COCOLoader:
    """Loads the Roboflow COCO Segmentation JSON and provides ground truth
    annotations per image.

    Args:
        coco_json:  Path to _annotations.coco.json
        image_dir:  Directory containing the image files
        n_points:   Number of measurement points to sample per cut_edge
    """

    def __init__(
        self,
        coco_json:  str = COCO_ANNOTATIONS,
        image_dir:  str = IMAGE_DIR,
        n_points:   int = NUM_SAMPLE_POINTS,
    ):
        self.image_dir = image_dir
        self.n_points  = n_points

        print(f"[COCOLoader] Loading annotations from {coco_json}")
        self.coco = COCO(coco_json)

        # Build category map
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat_map = {c['id']: c['name'] for c in cats}

        self.cut_edge_id = self._get_cat_id('cut_edge')
        self.qr_code_id  = self._get_cat_id('qr_code')

        print(f"[COCOLoader] Found {len(self.coco.getImgIds())} images")
        print(f"[COCOLoader] cut_edge annotations: "
              f"{len(self.coco.getAnnIds(catIds=[self.cut_edge_id]))}")
        print(f"[COCOLoader] qr_code annotations:  "
              f"{len(self.coco.getAnnIds(catIds=[self.qr_code_id]))}")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_all(self) -> list[GTAnnotation]:
        """Return ground truth for every image in the dataset."""
        return [self.get_by_id(img_id)
                for img_id in self.coco.getImgIds()]

    def get_by_filename(self, filename: str) -> Optional[GTAnnotation]:
        """Return ground truth for the image matching *filename*."""
        # Roboflow appends a hash suffix — match by original stem
        stem = os.path.splitext(os.path.basename(filename))[0].lower()
        for img_id in self.coco.getImgIds():
            info = self.coco.loadImgs(img_id)[0]
            rf_stem = os.path.splitext(info['file_name'])[0].lower()
            # Roboflow format: "Original Name_jpg.rf.HASH"
            # Strip the ".rf.HASH" suffix for matching
            base = rf_stem.split('.rf.')[0] if '.rf.' in rf_stem else rf_stem
            if stem in base or base in stem:
                return self.get_by_id(img_id)
        return None

    def get_by_id(self, image_id: int) -> GTAnnotation:
        """Return ground truth for one image by COCO image id."""
        info      = self.coco.loadImgs(image_id)[0]
        img_path  = os.path.join(self.image_dir, info['file_name'])

        gt = GTAnnotation(
            image_id   = image_id,
            file_name  = info['file_name'],
            image_path = img_path,
            width      = info['width'],
            height     = info['height'],
        )

        # ── cut_edge ─────────────────────────────────────────────────────────
        ce_ann_ids = self.coco.getAnnIds(
            imgIds=[image_id], catIds=[self.cut_edge_id]
        )
        for ann in self.coco.loadAnns(ce_ann_ids):
            mask   = self.coco.annToMask(ann).astype(bool)
            box    = self._ann_to_xyxy(ann)
            points = self._mask_to_sample_points(mask)

            gt.cut_edge_masks.append(mask)
            gt.cut_edge_boxes.append(box)
            gt.cut_edge_points.append(points)

        # ── qr_code ───────────────────────────────────────────────────────────
        qr_ann_ids = self.coco.getAnnIds(
            imgIds=[image_id], catIds=[self.qr_code_id]
        )
        for ann in self.coco.loadAnns(qr_ann_ids):
            gt.qr_boxes.append(self._ann_to_xyxy(ann))

        return gt

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_cat_id(self, name: str) -> int:
        for cid, cname in self.cat_map.items():
            if cname == name:
                return cid
        raise ValueError(f"Category '{name}' not found in annotations. "
                         f"Available: {list(self.cat_map.values())}")

    def _mask_to_sample_points(self, mask: np.ndarray) -> np.ndarray:
        """Extract the longest contour from a binary mask and sample N points."""
        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return np.empty((0, 2), dtype=np.float32)
        contour = max(contours, key=cv2.contourArea)
        return sample_points_along_contour(contour, n=self.n_points)

    @staticmethod
    def _ann_to_xyxy(ann: dict) -> np.ndarray:
        """Convert COCO bbox [x, y, w, h] to [x1, y1, x2, y2]."""
        x, y, w, h = [float(v) for v in ann['bbox']]
        return np.array([x, y, x + w, y + h], dtype=np.float32)

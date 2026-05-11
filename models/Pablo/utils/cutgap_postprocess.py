
"""
cutgap_postprocess.py

Postproceso para Grounded-SAM:
1) Refuerza detecciones de cut_gap con un segundo borde paralelo cuando solo aparece uno.
2) Aplica fallback clásico condicionado por strip, evitando usar el clásico como detector libre.
3) Filtra falsos positivos en imágenes sin corte real probable.

Uso esperado dentro de grounded_sam.py:
    from utils.cutgap_postprocess import (
        postprocess_cutgap_detections,
        conditional_classical_cutgap_fallback,
    )
"""

from __future__ import annotations

import cv2
import numpy as np


def _clip_box(box, w, h):
    x1, y1, x2, y2 = map(float, box)
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _box_size(box):
    x1, y1, x2, y2 = box
    return max(1.0, x2 - x1), max(1.0, y2 - y1)


def _orientation(box):
    bw, bh = _box_size(box)
    return "vertical" if bh >= bw else "horizontal"


def _box_center(box):
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _box_area(box):
    bw, bh = _box_size(box)
    return bw * bh


def _inside_strip(box, strip_box, margin=20):
    x1, y1, x2, y2 = box
    sx1, sy1, sx2, sy2 = strip_box
    return (
        x1 >= sx1 - margin and x2 <= sx2 + margin and
        y1 >= sy1 - margin and y2 <= sy2 + margin
    )


def is_valid_cutgap_box(
    box,
    img_w,
    img_h,
    strip_box=None,
    min_long_ratio=0.06,
    max_long_ratio=0.55,
    max_short_ratio=0.10,
    border_margin_ratio=0.025,
):
    """
    Filtro geométrico conservador para cut_gap.
    No debe eliminar cortes válidos centrales, pero sí reduce detecciones pegadas a bordes externos.
    """
    x1, y1, x2, y2 = _clip_box(box, img_w, img_h)
    bw, bh = _box_size([x1, y1, x2, y2])

    long_side = max(bw, bh)
    short_side = min(bw, bh)

    img_long = max(img_w, img_h)
    img_short = min(img_w, img_h)

    if long_side < min_long_ratio * img_long:
        return False

    if long_side > max_long_ratio * img_long:
        return False

    if short_side > max_short_ratio * img_short:
        return False

    # Evita falsos positivos completamente pegados al borde de la imagen.
    margin_x = border_margin_ratio * img_w
    margin_y = border_margin_ratio * img_h
    if x1 <= margin_x or x2 >= img_w - margin_x or y1 <= margin_y or y2 >= img_h - margin_y:
        # Solo permitimos borde si también está dentro de strip.
        if strip_box is None or not _inside_strip([x1, y1, x2, y2], strip_box, margin=10):
            return False

    if strip_box is not None and not _inside_strip([x1, y1, x2, y2], strip_box, margin=30):
        return False

    return True


def _mask_from_det(det, h, w):
    mask = det.get("mask", None)
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def _edge_strength_profile(gray, strip_box, orientation):
    """
    Devuelve perfil de intensidad de borde dentro del strip.
    Para corte vertical: perfil por columnas.
    Para corte horizontal: perfil por filas.
    """
    h, w = gray.shape[:2]
    sx1, sy1, sx2, sy2 = map(int, _clip_box(strip_box, w, h))
    roi = gray[sy1:sy2, sx1:sx2]
    if roi.size == 0:
        return None, None

    roi = cv2.GaussianBlur(roi, (5, 5), 0)
    roi = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi)

    grad_x = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)

    if orientation == "vertical":
        profile = np.mean(np.abs(grad_x), axis=0)
        coords = np.arange(sx1, sx2)
    else:
        profile = np.mean(np.abs(grad_y), axis=1)
        coords = np.arange(sy1, sy2)

    profile = cv2.GaussianBlur(profile.reshape(1, -1), (1, 9), 0).ravel()
    return coords, profile


def _find_parallel_edge(gray, box, strip_box, search_min=8, search_max=45):
    """
    Busca un segundo borde paralelo cerca del borde detectado.
    No crea un corte nuevo lejos; solo completa uno ya detectado.
    """
    h, w = gray.shape[:2]
    orientation = _orientation(box)
    coords, profile = _edge_strength_profile(gray, strip_box, orientation)
    if coords is None or profile is None or len(coords) < 5:
        return None

    cx, cy = _box_center(box)

    if orientation == "vertical":
        base_coord = cx
        candidates_mask = (
            ((coords >= base_coord + search_min) & (coords <= base_coord + search_max)) |
            ((coords <= base_coord - search_min) & (coords >= base_coord - search_max))
        )
    else:
        base_coord = cy
        candidates_mask = (
            ((coords >= base_coord + search_min) & (coords <= base_coord + search_max)) |
            ((coords <= base_coord - search_min) & (coords >= base_coord - search_max))
        )

    candidate_indices = np.where(candidates_mask)[0]
    if len(candidate_indices) == 0:
        return None

    local_profile = profile[candidate_indices]
    if local_profile.max() < max(5.0, 0.55 * profile.max()):
        return None

    best_idx = candidate_indices[int(np.argmax(local_profile))]
    second_coord = float(coords[best_idx])

    x1, y1, x2, y2 = box
    bw, bh = _box_size(box)

    if orientation == "vertical":
        half_w = max(2.0, bw * 0.35)
        return [second_coord - half_w, y1, second_coord + half_w, y2]
    else:
        half_h = max(2.0, bh * 0.35)
        return [x1, second_coord - half_h, x2, second_coord + half_h]


def _merge_two_edges_to_gap(box_a, box_b, img_w, img_h):
    """
    Une dos bordes paralelos en una única caja de cut_gap más completa.
    """
    x1 = min(box_a[0], box_b[0])
    y1 = min(box_a[1], box_b[1])
    x2 = max(box_a[2], box_b[2])
    y2 = max(box_a[3], box_b[3])
    return _clip_box([x1, y1, x2, y2], img_w, img_h)


def _make_rect_mask(box, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = map(int, _clip_box(box, w, h))
    cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)
    return mask.astype(bool)


def postprocess_cutgap_detections(
    image_bgr,
    detections,
    strip_detections,
    min_conf_keep=0.12,
    complete_second_edge=True,
):
    """
    Postproceso principal.
    Entrada:
      detections: lista de detecciones Grounded-SAM.
      strip_detections: detecciones de piezas/tiras de goma.
    Salida:
      detections filtradas/refinadas.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    cut_dets = [d for d in detections if d.get("label") == "cut_gap"]
    other_dets = [d for d in detections if d.get("label") != "cut_gap"]

    refined = []

    for det in cut_dets:
        score = float(det.get("score", det.get("confidence", 0.0)))
        box = _clip_box(det["box"], w, h)

        if score < min_conf_keep:
            continue

        # Asociar al strip más cercano/donde cae.
        best_strip = None
        best_overlap_area = 0.0
        for s in strip_detections:
            sbox = _clip_box(s["box"], w, h)
            ix1 = max(box[0], sbox[0])
            iy1 = max(box[1], sbox[1])
            ix2 = min(box[2], sbox[2])
            iy2 = min(box[3], sbox[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter > best_overlap_area:
                best_overlap_area = inter
                best_strip = sbox

        if not is_valid_cutgap_box(box, w, h, strip_box=best_strip):
            continue

        final_box = box

        if complete_second_edge and best_strip is not None:
            second = _find_parallel_edge(gray, box, best_strip)
            if second is not None and is_valid_cutgap_box(second, w, h, strip_box=best_strip):
                final_box = _merge_two_edges_to_gap(box, second, w, h)

        new_det = dict(det)
        new_det["box"] = final_box

        # Si no hay máscara o queremos coherencia, generamos máscara rectangular refinada.
        new_det["mask"] = _make_rect_mask(final_box, h, w)

        refined.append(new_det)

    return other_dets + refined


def _classic_candidate_from_strip(image_bgr, strip_det):
    """
    Fallback clásico condicionado por strip.
    Busca un posible corte SOLO dentro del strip.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    sbox = _clip_box(strip_det["box"], w, h)
    sx1, sy1, sx2, sy2 = map(int, sbox)
    roi = gray[sy1:sy2, sx1:sx2]
    if roi.size == 0:
        return None

    roi_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi)
    roi_blur = cv2.GaussianBlur(roi_eq, (5, 5), 0)

    # Estimar orientación del strip.
    sw, sh = _box_size(sbox)
    expected_orientation = "vertical" if sw >= sh else "horizontal"
    # Si la tira es horizontal, el corte suele ser vertical; si la tira es vertical, el corte suele ser horizontal.
    cut_orientation = "vertical" if expected_orientation == "horizontal" else "horizontal"

    coords, profile = _edge_strength_profile(gray, sbox, cut_orientation)
    if coords is None:
        return None

    # Ignorar extremos del strip para evitar falsos bordes de pieza.
    n = len(profile)
    if n < 20:
        return None
    valid = np.zeros_like(profile, dtype=bool)
    valid[int(0.08 * n): int(0.92 * n)] = True

    local = profile.copy()
    local[~valid] = 0

    peak = float(local.max())
    if peak < max(6.0, float(profile.mean() + 1.2 * profile.std())):
        return None

    idx = int(np.argmax(local))
    coord = float(coords[idx])

    if cut_orientation == "vertical":
        # altura dentro del strip casi completa, anchura estrecha
        box = [coord - 8, sy1 + 0.05 * (sy2 - sy1), coord + 8, sy2 - 0.05 * (sy2 - sy1)]
    else:
        box = [sx1 + 0.05 * (sx2 - sx1), coord - 8, sx2 - 0.05 * (sx2 - sx1), coord + 8]

    box = _clip_box(box, w, h)
    if not is_valid_cutgap_box(box, w, h, strip_box=sbox):
        return None

    return {
        "label": "cut_gap",
        "score": 0.10,
        "box": box,
        "mask": _make_rect_mask(box, h, w),
        "source": "classical_conditioned_fallback",
    }


def conditional_classical_cutgap_fallback(
    image_bgr,
    detections,
    strip_detections,
    max_new_per_image=2,
):
    """
    Solo activa clásico cuando:
    - No hay cut_gap suficiente, o
    - Hay menos cortes que strips.
    No busca fuera de strips, por lo que reduce falsos positivos.
    """
    h, w = image_bgr.shape[:2]

    existing_cuts = [d for d in detections if d.get("label") == "cut_gap"]
    other = [d for d in detections if d.get("label") != "cut_gap"]

    # Si ya hay al menos un corte por strip relevante, no tocamos.
    if len(existing_cuts) >= min(len(strip_detections), 2):
        return detections

    new_cuts = list(existing_cuts)

    for strip in strip_detections:
        if len(new_cuts) >= max_new_per_image:
            break

        sbox = _clip_box(strip["box"], w, h)

        # Saltar strips que ya tienen cut_gap dentro.
        already = False
        for c in new_cuts:
            if _inside_strip(_clip_box(c["box"], w, h), sbox, margin=25):
                already = True
                break
        if already:
            continue

        cand = _classic_candidate_from_strip(image_bgr, strip)
        if cand is not None:
            new_cuts.append(cand)

    return other + new_cuts

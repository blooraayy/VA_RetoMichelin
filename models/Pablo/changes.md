# Cambios realizados en el proyecto (sesión junio 2026)

## Resumen

Se han implementado dos fases de mejora sobre el pipeline original para adaptarlo
al formato exigido por `nuevas_normas.pdf` (junio 2026) y para que funcione con
las nuevas imágenes del dataset.

---

## Phase 1 — Formato de salida xlsx (commit `d1f3033` + `7f0ac52`)

### Archivos nuevos

| Archivo | Descripción |
|---|---|
| `model/reto_measurements.py` | Motor de medición Reto 1 y Reto 2. Dataclasses `Reto1Row` y `Reto2Row`. Clase `RetoMeasurementEngine` que convierte máscaras SAM + homografía en las tablas de medidas. Funciones `assign_bands()` y `assign_cuts_to_bands()`. |
| `view/xlsx_writer.py` | Genera el `.xlsx` con exactamente el formato que Michelin usará para comparar contra su ground truth: hojas "Foto 1 / 2 / 3", cabeceras numéricas 1–10, Reto 1 con 9 columnas (col. 10 vacía), Reto 2 con 10 columnas, valores `float` o celda vacía (no "N/A"), leyenda al final. |
| `reto_challenge.py` | Script principal del día de competición. Carga modelos una vez, procesa 1–3 imágenes y escribe el xlsx. |
| `infer_new_images.py` | Inferencia masiva sobre `new_images/` sin ground truth. Escribe JSON + CSV de resumen por posición. |

### Archivos modificados

| Archivo | Cambio |
|---|---|
| `config/config.py` | Añadidos `RETO1_X_MM`, `RETO2_Y_OFFSETS_MM`, `RETO2_BAND_WIDTH_MM`. Prompt QR actualizado para detectar QR codes reales (además de las pegatinas rosas antiguas). |
| `model/qr_calibrator.py` | `_build_homography` ahora detecta automáticamente si el tercer QR es BL (dataset antiguo) o BR (dataset nuevo) y calcula la homografía correcta en ambos casos. Añadido `mm_to_pixels_bulk()`. |
| `requirements.txt` | Añadido `openpyxl`. |

---

## Phase 2 — Robustez para nuevas imágenes (commit `e883651`)

Las nuevas imágenes presentan dos problemas que el pipeline original no manejaba:
- **Bandas horizontales** (en lugar de verticales): DINO no las detectaba con suficiente confianza.
- **Cortes casi invisibles** (goma presionada): DINO no detectaba el gap y el fallback clásico nunca se activaba.

### Cambios en `model/grounded_sam.py`

**1. `_intensity_strip_fallback()` (método nuevo)**

Cuando el Stage 1 de DINO devuelve 0 bandas, se activa un fallback basado en proyección de píxeles oscuros por filas/columnas:
- Umbraliza la imagen (píxeles < 80 = goma).
- Calcula la proyección a lo largo del eje largo de la imagen.
- Localiza rangos de filas/columnas con suficiente densidad de goma.
- Construye las bounding boxes y máscaras directamente.
- Funciona para bandas horizontales (Pos4, Pos7) y layouts de 4 cuadrantes con corte en cruz (Pos1, Pos3).

**2. Fallback clásico sin condición (Stage 2)**

El fallback de gradiente clásico pasaba por la condición `any(strip_had_cut)` — solo se ejecutaba si DINO había encontrado al menos un corte. Esto hacía que los cortes apenas visibles no se detectaran nunca.

La condición ahora es simplemente: *"hay bandas detectadas y faltan cortes por encontrar"*. El filtro `_is_valid_cutgap_box` sigue evitando falsos positivos.

### Cambios en `model/qr_calibrator.py`

`DIST_TO_CENTRE_MIN`: **0.55 → 0.40**

Varias imágenes nuevas mostraban `n_qr=2` con el umbral anterior. El QR válido estaba
ligeramente más cerca del centro de lo que el filtro permitía.

### Cambios en `config/config.py`

`CUT_MIN_ASPECT_RATIO`: **8.0 → 4.0**

Los cortes verticales en bandas horizontales cortas tienen un ratio altura/anchura
de ~5–7, que quedaba por debajo del umbral anterior.

---

## Estructura de la salida xlsx

Exactamente el formato de `nuevas_normas.pdf`:

```
Foto 1  —  nombre_imagen.jpg  [QR OK / SIN QR]
        1    2    3    ...    9    (vacío)
Distancia a borde (mm)    80  160  ...  720   ""
DA      <float> ...
DB      ...
LA      ...
LB      ...

        1    2    3    ...   10
Distancia a borde producto (mm)    5   30  ...  230
SA1     ...
SA2     ...
YSA     ...
SB1     ...
SB2     ...
YSB     ...

Leyenda
DA  Distancia a borde mesa ...
...
```

Los valores ausentes son celdas vacías (no texto "N/A") para que Michelin pueda
comparar numéricamente contra su ground truth.

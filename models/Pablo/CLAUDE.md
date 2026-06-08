# CLAUDE.md — Contexto completo del proyecto Reto Michelin
# (sesión junio 2026 — actualizado 2026-06-08 sesión tarde)

---

## Lo primero al arrancar sesión

**Ejecutar el pipeline completo y revisar las imágenes `_reto.png` de salida:**

```bash
conda run -n michelin python reto_challenge.py --folder all_images/ --no-show
```

Salida en `data/outputs/reto/`. Revisar especialmente Pos1, Pos3, Pos4, Pos5, Pos7.

---

## Descripción

Pipeline de Computer Vision para la **Competición Michelin Challenge** (día 10 de junio 2026).
Dado un conjunto de 1–3 fotografías de bandas de goma sobre mesa, el sistema debe:
1. Detectar los QR de calibración y calcular la homografía píxel↔mm.
2. Detectar las dos bandas de goma (Banda A y Banda B).
3. Detectar el corte en cada banda (si existe y es visible).
4. Medir distancias según `nuevas_normas.pdf` (Reto 1 y Reto 2).
5. Escribir una tabla xlsx con el formato exacto que Michelin comparará contra su ground truth.

**Repositorio**: `C:\Users\ruizm\Escritorio\VA_RetoMichelin\models\Pablo\`
**Rama activa**: `main`
**Autor**: Paboct (Pablo Ruiz)

---

## Comando de competición (día 10 de junio, 9:15h)

```bash
# Poner las 3 fotos en competition_images/ y ejecutar:
conda run -n michelin python reto_challenge.py --folder competition_images/ --no-show

# Genera: reto_results.xlsx  (subir a Agora)
# Tiempo estimado: 5-15 min GPU, 30-60 min CPU
```

Para verificar en todo el dataset:
```bash
conda run -n michelin python reto_challenge.py --folder all_images/ --no-show
# Salida: data/outputs/reto/  +  reto_results.xlsx
```

---

## Arquitectura del pipeline

```
reto_challenge.py  (orquestador principal)
    ├── model/grounded_sam.py       (detección: QR + strips + cuts)
    ├── model/qr_calibrator.py      (homografía píxel↔mm)
    ├── model/reto_measurements.py  (motor de medición Reto 1 y Reto 2)
    ├── view/visualiser.py          (visualización: _detection, _measurement, _reto)
    ├── view/xlsx_writer.py         (escritura del xlsx formato Michelin)
    └── config/config.py            (todos los umbrales y parámetros)
```

### Flujo por imagen

1. `GroundedSAMModel.run()` → devuelve `PipelineResult` con strips, cuts, QRs
2. `QRCalibrator` → homografía H (píxeles → mm) a partir de 3 QR corners
3. `RetoMeasurementEngine.compute_reto1/2()` → listas de `Reto1Row` / `Reto2Row`
4. `Visualiser.render()` + `render_reto()` → imágenes anotadas
5. `write_reto_xlsx()` → xlsx final

---

## Sistema de detección de cortes (5 pasadas)

```
Pasada 1: DINO normal          (BOX=0.17, TEXT=0.14)
Pasada 2: DINO umbral bajo     (BOX=0.13, TEXT=0.11)
Pasada 3: Gradient clásico     (Sobel+CLAHE, max(3.5, mean+0.8σ)) + brightness p90>120
Pasada 4: Intensity valley     (valle oscuro, min(6.0, max(2.5, mean+0.7σ))) SIN brightness check
Pasada 5: Bright-gap fallback  (columna brillante, max(5.0, mean+1.2σ) + p90≥130)
```

Cada pasada solo se ejecuta si la anterior no encontró corte.
Pasada 4: umbral CAPEADO a 6.0 — fix para Pos5 (fondo metálico oscuro con σ alto por textura).

### Strip fallback (cuando DINO devuelve 0 bandas)

`_intensity_strip_fallback()`: proyección de píxeles oscuros por filas/columnas.
1. Aplica CLAHE (clipLimit=3) → umbral fijo `gray_eq < 95`
2. Si <2 bandas: reintento con umbral adaptativo `raw_gray < (p20 + 0.35*(p50-p20))` (máx 110)

---

## QR Calibration (qr_calibrator.py)

- Umbral sanity check: `y_range_mm > 3.5 * qr_leg_mm` (era 2.5×, subido para Pos1 con fondo gris grande)
- **Combo retry**: si falla el sanity check, prueba todas las C(n,3) combinaciones de los top-10 candidatos QR
- Layout TL/BL/BR soportado: `n_top==1` → `dst=[(0,0),(0,L),(L,L)]`

---

## Parámetros clave (config/config.py)

| Parámetro | Valor | Descripción |
|---|---|---|
| `PROMPT_STRIP` | `"black rubber strip . dark rubber band on table . rubber piece on metal surface"` | Soporte fondo blanco, gris y metálico |
| `GDINO_BOX_THRESHOLD` | 0.10 | Umbral de detección de strips |
| `CUT_GAP_BRIGHTNESS_MIN` | 120.0 | p90 mínimo para aceptar un gap (pasadas 1–3) |
| `RETO1_X_MM` | [80,160,...,720] | 9 posiciones X para Reto 1 |
| `RETO2_Y_OFFSETS_MM` | [5,30,55,...,230] | 10 offsets Y desde top del corte |
| `RETO2_BAND_WIDTH_MM` | 250.0 | Ancho nominal de banda |

---

## Especificación Reto 2 (nuevas_normas.pdf)

Las 10 medidas se toman en posiciones fijas desde la **parte superior del corte de cada banda** (origen = top del corte, marcado como `(0)`):

| Posición | Offset Y desde top del corte |
|---|---|
| 1 | 5 mm |
| 2 | 30 mm |
| 3 | 55 mm |
| ... | +25 mm cada una |
| 10 | 230 mm |

**SA1**: distancia horizontal entre el borde superior izquierdo (punto A) y el borde superior derecho (punto B) del corte en cada posición Y.
**SA2**: distancia entre borde superior e inferior del corte (chaflán). En vista cenital ≈ SA1. **Negativo** si el borde inferior está tapado por el superior (solapamiento).
**YSA/YSB**: distancia global Y desde el borde de la mesa hasta el top del corte de cada banda.

---

## Visualización `_reto.png` (diseño sesión 2026-06-08)

### `render_reto()` — capas en orden de pintado:
1. **Overlay SAM bandas** (alpha=0.18): máscara SAM de Banda A en amarillo `(0,220,220)`, Banda B en cian `(200,200,0)`
2. **Bounding box bandas** (2px) + etiquetas "Banda A" / "Banda B"
3. **Overlay SAM cortes** (alpha=0.42): máscaras SAM de Corte A y Corte B en rojo `(50,50,220)`
4. **Bounding box cortes** (2px, rojo) + etiquetas "Corte A" / "Corte B"
5. **QR detections** (markers + bbox)
6. **Líneas de medida Reto 2** vía `_draw_reto2_visuals()`

### `_draw_reto2_visuals()`:
- **Línea vertical marrón** `(19,69,139)` desde borde de mesa (Y=0mm) hasta top del corte (YSA/YSB)
- **Círculo `(0)`** marrón con borde negro en el punto origen + etiqueta `YSA=xxx.xmm`
- **10 líneas horizontales** finas (1px, color banda) sin tick marks, desde left a right de la banda
- **Segmento SA1 blanco** (2px) en el centro del corte en cada línea
- **Puntos A y B** (círculos blancos, borde negro) en los extremos del segmento SA1; etiquetas "A"/"B" solo en línea 1
- **Anotaciones** `"N. SA1=x.x"` solo en líneas 1, 5 y 10 (índices 0, 4, 9) al borde derecho
- Si una banda **no tiene corte detectado** → no se dibujan líneas (correcto según spec)

---

## Estado de detección por tipo de imagen

| Posición | Descripción | Estado | Notas |
|---|---|---|---|
| Multimedia_31–52 | Mesa blanca, bandas verticales | ✅ | Sin falsos positivos |
| Pos1 | 4 cuadrantes, fondo blanco/gris | ⚠️ | QR fix aplicado (3.5×); solo mitad de banda detectada (sin merge) |
| Pos2 | Bandas verticales, fondo gris | ⚠️ | A veces asignación A/B incorrecta (111118) |
| Pos3 | Bandas horizontales largas, gris | ⚠️ | Mejorado (valley fallback): de 0 a ~3/4 con CUT=2 |
| Pos4 | Bandas horizontales + goma presionada | ⚠️ | Valley+CLAHE aplicado; pendiente validar en nuevo run |
| Pos5 | 4 cuadrantes, mesa metálica gris oscura | ⚠️ | Valley cap 6.0 aplicado; pendiente validar en nuevo run |
| Pos6 | Bandas verticales largas, gris | ⚠️ | Solo 1 de 2 cortes detectados |
| Pos7 | Bandas horizontales, gris | ⚠️ | Gradiente+bright-gap aplicado; pendiente validar |
| Pos8 | Bandas horizontales, cortes tenues | ⚠️ | Mejorado; 112247 aún falla |

---

## Problemas pendientes

### P1 — Pos1 solo detecta mitad de banda
- DINO detecta las 4 piezas por separado; falta merge de trozos adyacentes del mismo lado.
- Archivo: `model/grounded_sam.py` → necesita lógica de merge de bounding boxes solapados.

### P2 — Asignación A/B incorrecta (Pos2, imagen 111118)
- `assign_bands()` en `model/reto_measurements.py` usa `mean_y_mm` → falla si la calibración QR es imprecisa.

### P3 — Pos6 solo 1 corte
- El segundo corte es muy difuso. Sin fix conocido todavía.

### P4 — 112247 (Pos8) fallo total
- Ninguna detección en esa imagen. Sin fix conocido todavía.

---

## Estructura de salida

```
data/outputs/reto/
    <nombre>_detection_<ts>.png     # QR + strip boxes + cut boxes (SAM masks)
    <nombre>_measurement_<ts>.png   # Reto 1 sample points + distancias
    <nombre>_reto_<ts>.png          # Vista Reto 2: masks + bboxes + 10 líneas
    <nombre>_measurements_<ts>.csv  # CSV por imagen
    <nombre>_<ts>.json              # JSON completo de resultados

reto_results.xlsx                   # Tabla final formato Michelin
```

### Formato xlsx (nuevas_normas.pdf)

```
Hoja "Foto N" por imagen:
  Reto 1: 4 filas (DA, DB, LA, LB) × 9 columnas (80–720 mm) + col 10 vacía
  Reto 2: 6 filas (SA1, SA2, YSA, SB1, SB2, YSB) × 10 columnas (5–230 mm)
  Valores ausentes = celda vacía (no "N/A")
```

---

## Definición de métricas (nuevas_normas.pdf)

| Métrica | Definición |
|---|---|
| **DA** | Distancia en Y desde borde de mesa (QR baseline) al borde superior de Banda A |
| **DB** | Ídem para Banda B |
| **LA** | Anchura de Banda A en mm (Y extent de la máscara) |
| **LB** | Ídem para Banda B |
| **SA1** | Distancia entre punto A (borde superior izq. del corte) y punto B (borde superior dcho.) en cada Y |
| **SA2** | Distancia entre borde superior e inferior del corte (chaflán). Negativo si se solapan. ≈SA1 en vista cenital |
| **YSA** | Y global desde borde de mesa hasta top del corte de Banda A (origen `(0)` de las 10 divisiones) |
| **SB1/SB2/YSB** | Equivalentes para Banda B |

---

## Instalación

```bash
# GPU (CUDA 12.4) — entorno conda "michelin"
conda run -n michelin python reto_challenge.py --folder all_images/ --no-show

# Pesos SAM ViT-H en weights/sam_vit_h_4b8939.pth
# Grounding DINO se descarga automático de HuggingFace
```

GPU en uso: NVIDIA GeForce RTX 4070 Laptop, 8 GB VRAM, CUDA 12.4.

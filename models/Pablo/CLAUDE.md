# CLAUDE.md — Contexto completo del proyecto Reto Michelin
# (sesión junio 2026 — actualizado 2026-06-07)

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
python reto_challenge.py --folder competition_images/ --no-show

# Genera: reto_results.xlsx  (subir a Agora)
# Tiempo estimado: 5-15 min GPU, 30-60 min CPU
```

Para verificar en todo el dataset:
```bash
python reto_challenge.py --folder all_images/ --no-show
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

## Sistema de detección de cortes (4 pasadas)

```
Pasada 1: DINO normal          (BOX=0.17, TEXT=0.14)
Pasada 2: DINO umbral bajo     (BOX=0.13, TEXT=0.11)
Pasada 3: Gradient clásico     (Sobel+CLAHE, max(5.0, mean+1.0σ))
Pasada 4: Intensity valley     (valle oscuro con ventana deslizante, max(6.0, mean+1.5σ))
```

Cada pasada solo se ejecuta si la anterior no encontró corte.
Todos los candidatos pasan por `_is_valid_cutgap_box()` (geometría) y `_gap_centre_brightness()` (p90 > 120, para descartar bordes oscuros).

### Strip fallback (cuando DINO devuelve 0 bandas)

`_intensity_strip_fallback()`: proyección de píxeles oscuros (`gray < 95`) por filas/columnas.
Funciona para bandas horizontales y layouts de 4 cuadrantes en cruz.

---

## Parámetros clave (config/config.py)

| Parámetro | Valor | Descripción |
|---|---|---|
| `PROMPT_STRIP` | `"black rubber strip . dark rubber band on table"` | Soporte fondo blanco y gris |
| `PROMPT_QR` | `"qr code . black square qr code . ... . pink square sticker"` | QR reales + pegatinas antiguas |
| `CUT_GAP_BRIGHTNESS_MIN` | 120.0 | p90 mínimo para aceptar un gap (mesa blanca visible) |
| `CUT_MIN_ASPECT_RATIO` | 4.0 | Ratio mínimo para validar un corte |
| `RETO1_X_MM` | [80,160,...,720] | 9 posiciones X para Reto 1 |
| `RETO2_Y_OFFSETS_MM` | [5,30,55,...,230] | 10 offsets Y desde top del corte para Reto 2 |
| `RETO2_BAND_WIDTH_MM` | 250.0 | Ancho nominal de banda |
| `DIST_TO_CENTRE_MIN` | 0.40 | En QR calibrator — filtro de QRs válidos |

---

## Estado de detección por tipo de imagen (evaluado 2026-06-07)

### ✅ Multimedia_31 – Multimedia_52 (mesa blanca, bandas verticales)
- Detección correcta en la mayoría. Ambas bandas, cortes detectados cuando son visibles.
- Las 10 líneas de medida son correctas y abarcan el ancho completo de la banda.
- Multimedia_48 es el caso modelo más limpio.
- Multimedia_52: Banda A sin corte visible → no se muestran líneas (comportamiento correcto).
- **Sin falsos positivos** tras el check de brillo en el fallback clásico.

### ✅/⚠️ Pos1 (4 cuadrantes, bandas verticales, fondo blanco/gris)
- Bandas detectadas pero a veces solo la mitad derecha de cada banda (un trozo en lugar de dos).
- El corte sí se detecta cuando el gap es visible.
- **Problema**: DINO detecta las 4 piezas por separado; el merge de trozos adyacentes de la misma banda no se hace.

### ✅/⚠️ Pos2 (bandas verticales, fondo gris)
- Algunos ejemplos funcionan perfectamente (111123_1, 111134_1).
- **Problema conocido**: En 111118, la Banda B se asigna a una posición completamente errónea (esquina superior izquierda de la imagen). Es un error de asignación A/B cuando los dos strips detectados están mal ordenados.

### ❌ Pos3 (2 bandas horizontales largas, fondo gris)
- Bandas detectadas pero **0 cortes** en todos los ejemplos.
- Los cortes son arrugas o líneas muy sutiles; ni DINO ni los fallbacks los detectan.

### ❌ Pos4 (bandas horizontales largas, cortes de goma presionada, fondo gris)
- Inconsistente: algunos ejemplos detectan bandas, otros fallan completamente.
- **Nunca detecta los cortes** (goma presionada = línea oscura sin hueco blanco).
- La Pasada 4 (intensity valley) no es suficientemente sensible para estos cortes.

### ❌ Pos5 (4 cuadrantes, fondo metálico gris oscuro)
- **Fallo total**: ni bandas ni cortes en ninguno de los 5 ejemplos.
- El caucho negro sobre mesa metálica gris oscura apenas tiene contraste.
- Umbral `gray < 95` no es suficiente. La goma puede ser similar a 80-90 y la mesa 70-85.
- **Prioridad alta** para mañana: este layout puede aparecer en las imágenes de competición.

### ⚠️ Pos6 (bandas verticales largas, fondo gris)
- Bandas detectadas. Solo una de las dos bandas tiene corte detectado en la mayoría de casos.
- La otra banda tiene corte pero muy difuso.

### ⚠️ Pos7 (bandas horizontales, fondo gris)
- Solo se detecta una banda de las dos en la mayoría de ejemplos.
- El strip fallback no termina de funcionar bien para estos layouts.

### ⚠️ Pos8 (bandas horizontales, cortes visibles pero tenues)
- Algunos ejemplos bien (112218: bandas detectadas, corte de banda A visible).
- **112247: fallo total** — ninguna detección.
- Cuando el corte está en posición muy alta de la imagen, las 10 líneas de medida caen fuera del área visible (pero el cálculo numérico es correcto).

---

## Problemas prioritarios para próximas sesiones

### P1 (crítico) — Pos5 fallo total
- Fondo gris metálico oscuro, umbral 95 insuficiente.
- **Posible solución A**: subir umbral a 110 y usar detección de contraste relativo en lugar de absoluto (percentil 20 de la imagen como referencia).
- **Posible solución B**: Preprocesar con CLAHE antes del fallback de intensidad para normalizar el histograma.
- **Archivo**: `model/grounded_sam.py` → `_intensity_strip_fallback()` (línea ~911)

### P2 (importante) — Pos4 cortes de goma presionada
- El valley fallback existe pero no es suficientemente sensible.
- La goma presionada crea una depresión de ~2-5 píxeles de ancho y ~3-8 de intensidad más oscura.
- **Posible solución**: reducir la ventana mínima del valley a 1-2 px, bajar el umbral a `mean + 0.8σ` y confiar más en `_is_valid_cutgap_box` para filtrar falsos positivos.
- **Archivo**: `model/grounded_sam.py` → `_intensity_valley_fallback()`

### P3 (importante) — Pos7 solo detecta una banda
- El strip fallback debería encontrar ambas bandas horizontales.
- **Posible causa**: las dos bandas tienen proyecciones muy cercanas y el algoritmo las fusiona en una sola.
- **Archivo**: `model/grounded_sam.py` → `_intensity_strip_fallback()` → `_find_strips_from_projection()`

### P4 (moderado) — Asignación A/B incorrecta (Pos2)
- La asignación Banda A = strip más a la izquierda / Banda B = más a la derecha (o arriba/abajo para horizontales) falla cuando hay detecciones espurias.
- **Archivo**: `model/grounded_sam.py` → la lógica de asignación de strips tras `run()`
- En `reto_challenge.py`: `band_a, band_b = strips[0], strips[1]` — ordenar por posición X o Y consistentemente

### P5 (leve) — Pos1 solo detecta mitad de banda
- Para layouts de 4 cuadrantes, DINO detecta 4 trozos pero el pipeline toma solo 2.
- Solución: merge de bounding boxes de trozos del mismo lado (similar X range, separación < umbral en Y).

---

## Estructura de salida

```
data/outputs/reto/
    <nombre>_detection_<ts>.png     # QR + strip boxes + cut boxes
    <nombre>_measurement_<ts>.png   # Reto 1 sample points + distancias
    <nombre>_reto_<ts>.png          # Vista Reto 2 con 10 líneas de medida
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
| **LA** | Longitud/anchura de Banda A en mm |
| **LB** | Ídem para Banda B |
| **SA1** | Ancho horizontal del gap (corte) de Banda A a cada posición Y |
| **SA2** | Ídem incluyendo chaflán (en vista cenital ≈ SA1) |
| **YSA** | Distancia global en Y desde borde de mesa al top del corte de Banda A (origen de las 10 divisiones) |
| **SB1/SB2/YSB** | Equivalentes para Banda B |

**Origen Reto 2**: `ysa` = top de la máscara del corte (no top del strip). Ver `model/reto_measurements.py`.

---

## Visualización _reto.png (diseño actual)

- Relleno de banda: alpha=0.12 (muy tenue, el patrón de la goma es visible)
- Líneas horizontales: 2 px, abarcan el **ancho completo de la banda** (no solo la zona de corte)
- Marca vertical de 3 px en el borde izquierdo de cada línea
- Marcador `(0)` en la posición YSA/YSB (círculo + texto)
- Flecha YSA/YSB desde borde de mesa hasta top del corte
- Segmento SA1 en blanco (3 px) con marcas en los extremos
- Anotaciones numeradas `"N. SA1=x.x"` al borde derecho con sombra negra
- Si una banda **no tiene corte detectado** → no se dibujan líneas (correcto según spec)

---

## Historial de commits relevantes

```
a2e8fbd  docs: add Phase 5 to changes.md
0710a0c  feat(viz+detection): match reference image format and fix false positives
26be798  feat(detection): add intensity-valley fallback + all_images/ folder
518a669  docs: add Phase 3 (render_reto + run_all) to changes.md
e68adf3  feat: add render_reto visualisation and run_all batch processor
8ecd0dc  fix(reto2): use top of cut zone (ysa) as measurement origin, not top of strip
e883651  feat(phase2): robust strip+cut detection for new competition images
d1f3033  Formato Michelin nuevas_normas: xlsx, mediciones Reto1/2, homografia BR
```

---

## Instalación

```bash
# GPU (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Solo CPU
pip install -r requirements.txt

# Pesos SAM ViT-H (deben estar en weights/sam_vit_h_4b8939.pth)
# Grounding DINO se descarga automático de HuggingFace
```

GPU en uso: NVIDIA GeForce RTX 4070 Laptop, 8 GB VRAM, CUDA 12.4.

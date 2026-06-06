# Comandos del proyecto

Todos los comandos se ejecutan desde `models/Pablo/`.

---

## Dia de competicion (10 de junio, 9:15)

### Paso 1 — Descarga las 3 imágenes de Agora y ponlas en esta carpeta

```
models/Pablo/competition_images/
    foto1.jpg
    foto2.jpg
    foto3.jpg
```

### Paso 2 — Ejecuta el script principal

```bash
python reto_challenge.py --folder competition_images/ --no-show
```

Esto genera `reto_results.xlsx` en `models/Pablo/`. Súbelo a Agora.

**Tiempo estimado:** ~5–15 min en GPU (CUDA), ~30–60 min en CPU.

**Opciones adicionales:**

```bash
# Especificar nombre del xlsx de salida
python reto_challenge.py --folder competition_images/ --no-show --output resultado_final.xlsx

# Ejecutar en CPU (si no hay GPU disponible)
python reto_challenge.py --folder competition_images/ --no-show --device cpu

# Guardar también las imágenes anotadas (para revisar detecciones)
python reto_challenge.py --folder competition_images/
```

---

## Proceso batch — todas las imágenes (data/images/ + new_images/Pos*/)

Procesa en un solo paso todas las imágenes del dataset original **y** las nuevas.
Genera visualizaciones Reto 2 (`*_reto.png`) + tabla xlsx de resultados.

```bash
python run_all.py --no-show
```

Con CPU:

```bash
python run_all.py --device cpu --no-show
```

Opciones adicionales:

```bash
# Especificar carpeta de salida y xlsx
python run_all.py --no-show --output-dir data/outputs/run_all --xlsx run_all_results.xlsx

# Sin guardar imágenes anotadas (solo xlsx)
python run_all.py --no-show --no-save
```

Salida:
- `data/outputs/run_all/` — imágenes `*_detection.png`, `*_measurement.png`, `*_reto.png`
- `run_all_results.xlsx` — tabla Reto 1 / Reto 2 de todas las imágenes

---

## Inferencia sobre new_images/ (exploración, sin ground truth)

Procesa todas las imágenes de las carpetas `Pos1/`, `Pos2/`, etc. y guarda
visualizaciones + resumen JSON/CSV en `new_images/outputs/`.

```bash
python infer_new_images.py --no-show
```

Con CPU:

```bash
python infer_new_images.py --device cpu --no-show
```

Salida:
- `new_images/outputs/PosX/` — imágenes anotadas por posición
- `new_images/outputs/inference_summary.json` — resumen completo
- `new_images/outputs/inference_summary.csv` — tabla resumen

---

## Evaluacion con ground truth (dataset antiguo)

```bash
python evaluate.py
```

---

## Instalacion del entorno

### Con GPU (CUDA 12.1)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Solo CPU

```bash
pip install -r requirements.txt
```

### Pesos del modelo SAM

Los pesos SAM ViT-H deben estar en `weights/sam_vit_h_4b8939.pth`.
Si no están, descárgalos:

```bash
mkdir -p weights
curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -o weights/sam_vit_h_4b8939.pth
```

El modelo Grounding DINO (`IDEA-Research/grounding-dino-base`) se descarga
automáticamente de HuggingFace la primera vez.

---

## Git — commits relevantes de la sesion

```
e68adf3  feat: add render_reto visualisation and run_all batch processor
2d33fa4  docs: sync changes.md and prompts.md with latest commits
765f072  chore: add competition_images/ folder and update prompts.md
8ecd0dc  fix(reto2): use top of cut zone (ysa) as measurement origin, not top of strip
f1010e3  docs: add changes.md and prompts.md
e883651  feat(phase2): robust strip+cut detection for new competition images
7f0ac52  fix(xlsx_writer): match exact Michelin PDF table format
d1f3033  Formato Michelin nuevas_normas: xlsx, mediciones Reto1/2, homografia BR
```

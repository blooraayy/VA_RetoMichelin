# Comandos del proyecto

Todos los comandos se ejecutan desde `models/Pablo/`.

---

## Dia de competicion (10 de junio, 9:15)

### Paso 1 — Pon las 3 imágenes descargadas de Agora en una carpeta

Crea una carpeta (p.ej. `competition_images/`) y copia las 3 fotos dentro:

```
models/Pablo/competition_images/
    foto1.jpg
    foto2.jpg
    foto3.jpg
```

### Paso 2 — Ejecuta el script principal

```bash
python reto_challenge.py --images competition_images/foto1.jpg competition_images/foto2.jpg competition_images/foto3.jpg
```

Esto genera `reto_results.xlsx` en la misma carpeta (`models/Pablo/`).

**Opciones útiles:**

```bash
# Especificar ruta de salida del xlsx
python reto_challenge.py --images foto1.jpg foto2.jpg foto3.jpg --output mi_resultado.xlsx

# Procesar toda una carpeta de imágenes
python reto_challenge.py --folder competition_images/

# Ejecutar en CPU (si no hay GPU disponible)
python reto_challenge.py --images foto1.jpg --device cpu

# Sin ventanas de visualización (útil en servidor/headless)
python reto_challenge.py --images foto1.jpg foto2.jpg foto3.jpg --no-show

# Sin guardar imágenes anotadas (solo el xlsx)
python reto_challenge.py --images foto1.jpg foto2.jpg foto3.jpg --no-show --no-save
```

**Tiempo estimado:** ~5–15 min en GPU (CUDA), ~30–60 min en CPU.

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
e883651  feat(phase2): robust strip+cut detection for new competition images
7f0ac52  fix(xlsx_writer): match exact Michelin PDF table format
d1f3033  Formato Michelin nuevas_normas: xlsx, mediciones Reto1/2, homografia BR
```

# Pipeline de visión artificial — Reto Michelin (Jaime)

Sistema de inspección automática de bandas de goma en el montaje de neumáticos. Detecta, segmenta y mide la posición de los bordes de corte respecto al borde inferior de la mesa de trabajo, usando marcadores QR como referencia métrica.

El proyecto se organiza en dos capas:

- **`src/`** — pipeline modular listo para ejecutar en producción (`python src/main.py`) (se puede ejecutar paso a paso cada script por separado)
- **`test/`** — notebooks de exploración, evaluación de métricas y visualización paso a paso

---

## Pipeline completo (`src/`)

```
Imagen raw (data/raw/)
        │
        ├──► detection/qr_detection.py
        │         YOLOv9 (michelin_v1, clase qr_code índice 1, conf≥0.40)
        │         Etiqueta TL / TR / BL por posición relativa
        │         Homografía afín (3 QRs) o completa (4 QRs) → imagen → canvas 1000×1400 px
        │         ──► outputs/detections/qr_homographies.json
        │
        ├──► detection/rubber_detection.py
        │         YOLOv9 (michelin_v1, clase rubber_strip índice 2, conf≥0.65)
        │         Top-2 detecciones por confianza
        │         ──► outputs/detections/rubber_detections.json
        │
        ├──► detection/cut_edge_detection.py
        │         YOLOv9 (michelin_v3-4, clase cut_edge índice 0, conf≥0.65)
        │         Filtro área ≤ 10 % de la imagen + NMS adicional IoU=0.30
        │         Top-2 detecciones por confianza
        │         ──► outputs/detections/cut_edge_detections.json
        │
        ├──► segmentation/rubber_segmentation.py
        │         EfficientSAM ViT-Tiny — prompt: esquinas TL y BR de cada bbox
        │         ──► outputs/figures/rubber_segmentation/
        │
        ├──► segmentation/cut_edge_segmentation.py
        │         EfficientSAM ViT-Tiny — prompt: esquinas TL y BR de cada bbox
        │         ──► outputs/figures/cut_edge_segmentation/
        │
        └──► measurement/cut_edge_measurement.py
                  Re-segmenta bordes de corte con EfficientSAM
                  Pipeline 1 — punto único: borde más bajo de cada máscara
                  │    ──► outputs/figures/cut_edge_distances/
                  Pipeline 2 — multi-punto: 5 puntos × lado de la máscara
                       ──► outputs/figures/cut_edge_multipoint/
                       ──► outputs/detections/cut_edge_measurements.json
```

## Módulos (`src/`)

### `detection/qr_detection.py`
- Clase `qr_code` (índice 1), conf ≥ 0.40, área máxima 15 % de la imagen
- Etiqueta los QRs como TL / TR / BL / BR por suma y diferencia de coordenadas
- Con 3 QRs calcula transformación afín; con 4, homografía completa (RANSAC)
- Canvas de referencia: TL=(0,0), TR=(1000,0), BL=(0,1400), BR=(1000,1400)

### `detection/rubber_detection.py`
- Clase `rubber_strip` (índice 2), conf ≥ 0.65, área máxima 95 % de la imagen
- Devuelve las 2 detecciones de mayor confianza

### `detection/cut_edge_detection.py`
- Clase `cut_edge` (índice 0), conf ≥ 0.65, área máxima 10 % de la imagen
- NMS adicional con IoU = 0.30 para suprimir detecciones solapadas
- Devuelve las 2 detecciones de mayor confianza

### `segmentation/rubber_segmentation.py` y `cut_edge_segmentation.py`
- EfficientSAM ViT-Tiny con prompt de esquinas TL y BR del bbox
- Recorte con padding de 15 px antes de inferencia
- Devuelve máscara booleana del tamaño original de la imagen

### `measurement/cut_edge_measurement.py`
- EfficientSAM compartido con las etapas de segmentación (cargado una sola vez en `main.py`)
- **Pipeline 1 — punto único**: calcula la distancia del punto más bajo de cada máscara al borde BL y la visualiza en la figura
- **Pipeline 2 — multi-punto**: muestrea n+2 puntos por cada lado de la máscara (izquierdo/derecho en cortes verticales, superior/inferior en horizontales) y exporta todas las distancias a JSON

#### Geometría de calibración

Los 3 QRs forman un triángulo rectángulo con catetos de 800 mm. La homografía mapea imagen → espacio canónico donde:

```
TL → (0,    0)    TR → (1000,    0)
BL → (0, 1400)    BR → (1000, 1400)
```

Escalas resultantes:
- Eje X: 800 mm / 1000 px = **0.800 mm/px canónico**
- Eje Y: 800 mm / 1400 px ≈ **0.571 mm/px canónico**

Fórmula de distancia al borde de la mesa (referencia BL):

```
canon   = H · [px, py, 1]ᵀ
dist_mm = (1400 - canon_y) · 0.571
```

#### Formato del JSON de medidas (`cut_edge_measurements.json`)

```json
{
  "multimedia_01.jpg": [
    {
      "conf": 0.9191,
      "bbox": [x1, y1, x2, y2],
      "sides": {
        "left":  [{"x": 312, "y": 863, "dist_mm": 270.42}, ...],
        "right": [{"x": 313, "y": 863, "dist_mm": 270.41}, ...]
      }
    }
  ]
}
```

Cada imagen contiene una lista de cortes detectados. Cada corte incluye la confianza YOLO, su bbox y los puntos muestreados por cada lado con su distancia en mm al borde de la mesa.

---

## Notebooks de exploración y métricas (`test/`)

| Notebook | Propósito |
|----------|-----------|
| `qr_detection.ipynb` | Evaluación de detección QR: P, R, F1, IoU medio sobre train+valid |
| `rubber_detection.ipynb` | Evaluación de detección rubber strip: P, R, F1, IoU medio |
| `cut_edge_detection.ipynb` | Evaluación de detección cut edge: P, R, F1, IoU medio. Visualización GT vs predicción |
| `rubber_segmentation.ipynb` | Pruebas de segmentación EfficientSAM sobre rubber strips |
| `cut_edge_segmentation.ipynb` | Pruebas de segmentación EfficientSAM sobre bordes de corte |
| `cut_edge_measurement_test.ipynb` | Pruebas del pipeline de medición multi-punto y visualización de distancias |
| `yolo_finetune.ipynb` | Fine-tuning de YOLOv9 sobre el dataset de Michelin |
| `efficientSAM.ipynb` | Experimentos con EfficientSAM: prompts de bbox, evaluación de IoU de máscara |


## Entrenamiento (`src/models/yolo_finetune.py`)

Script para fine-tuning de YOLOv9 sobre el dataset de Michelin.

```bash
# Entrenamiento completo (split + train + eval + inferencia visual)
python src/models/yolo_finetune.py

Configuración por defecto: YOLOv9c, 100 épocas, imgsz=640, batch=8, patience=20, augmentación activada.

---

## Estructura de directorios

```
models/Jaime/
├── src/                                    # Pipeline modular
│   ├── main.py                             # Punto de entrada del pipeline completo
│   ├── detection/
│   │   ├── qr_detection.py
│   │   ├── rubber_detection.py
│   │   └── cut_edge_detection.py
│   ├── segmentation/
│   │   ├── rubber_segmentation.py
│   │   └── cut_edge_segmentation.py
│   ├── measurement/
│   │   └── cut_edge_measurement.py
│   └── models/
│       └── yolo_finetune.py
│
├── test/                                  # Notebooks de evaluación y pruebas
│   ├── outputs/                           # JSON, imágenes de salida de las pruebas y GT (misma estructura que el otro outputs)  
│   ├── qr_detection.ipynb
│   ├── rubber_detection.ipynb
│   ├── cut_edge_detection.ipynb
│   ├── rubber_segmentation.ipynb
│   ├── cut_edge_segmentation.ipynb
│   ├── cut_edge_measurement_test.ipynb
│   ├── yolo_finetune.ipynb
│   ├── efficientSAM.ipynb
│   └── efficient_sam_vitt.pt              # Pesos EfficientSAM ViT-Tiny
│
├── utils/
│   └── runs/data/runs/
│       ├── michelin_v1/weights/best.pt    # YOLO — QR + rubber strip
│       └── michelin_v3-4/weights/best.pt  # YOLO — cut edge
│
├── data/
│   ├── raw/                               # 22 imágenes originales (multimedia_01..22.jpg)
│   ├── new_samples_qr_rubber/             # Dataset etiquetado QR + rubber (train/valid)
│   └── segmentation_cut_edge/             # Dataset etiquetado cut edge (train/valid)
│
├── outputs/
│   ├── detections/
│   │   ├── qr_homographies.json           # Centros QR + matrices H por imagen
│   │   ├── rubber_detections.json         # Bboxes de rubber strips por imagen
│   │   ├── cut_edge_detections.json       # Bboxes de bordes de corte por imagen
│   │   └── cut_edge_measurements.json     # Distancias en mm por borde y por punto
│   └── figures/
│       ├── qr_detection/                  # Visualizaciones detección QR
│       ├── rubber_detection/              # Visualizaciones detección rubber
│       ├── rubber_segmentation/           # Máscaras de rubber strips
│       ├── cut_edge_detection/            # Visualizaciones detección cut edge
│       ├── cut_edge_segmentation/         # Máscaras de bordes de corte
│       ├── cut_edge_distances/            # Figuras punto único con distancias en mm
│       └── cut_edge_multipoint/           # Figuras multi-punto con distancias en mm
│
└── docs/                                  # Documentación y notas de diseño
```

---

## Requisitos

Ver `requirements.txt`. EfficientSAM requiere instalación local desde su repositorio:

```bash
git clone https://github.com/yformer/EfficientSAM.git
cd EfficientSAM && pip install -e .
```

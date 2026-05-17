# Pipeline de visión artificial — Reto Michelin

Sistema de inspección automática de bandas de goma en el montaje de neumáticos. Detecta, segmenta y mide la posición de los bordes de corte respecto al borde inferior de la mesa de trabajo, usando marcadores QR como referencia métrica.

---

## Flujo completo del pipeline

```
Imagen raw (data/raw/)
        │
        ├──► qr_detection.ipynb
        │         Detecta los 3 QRs de calibración (YOLOv9 michelin_v2)
        │         Etiqueta TL / TR / BL por posición relativa
        │         Calcula la homografía H (imagen → espacio canónico 1000×1400 px)
        │         ──► outputs/detections/qr_homographies.json
        │
        ├──► rubber_detection.ipynb
        │         Detecta las bandas de goma (YOLOv9 michelin_v2, clase rubber_strip)
        │         ──► outputs/detections/rubber_detections.json
        │
        ├──► rubber_segmentation.ipynb
        │         Segmenta píxel a píxel cada banda de goma (EfficientSAM ViT-Tiny)
        │         Entrada: rubber_detections.json
        │         ──► outputs/figures/rubber_segmentation/
        │
        ├──► cut_edge_detection.ipynb
        │         Detecta los bordes de corte (YOLOv9 michelin_v3-4, clase cut_edge)
        │         Aplica NMS adicional para eliminar detecciones solapadas
        │         ──► outputs/detections/cut_edge_detections.json
        │
        ├──► cut_edge_segmentation.ipynb
        │         Segmenta píxel a píxel cada borde de corte (EfficientSAM ViT-Tiny)
        │         Entrada: cut_edge_detections.json
        │         ──► outputs/figures/cut_edge_segmentation/
        │
        └──► cut_edge_measurement.ipynb
                  Une los centros QR y traza el triángulo de calibración
                  Dibuja la línea de referencia horizontal en BL (borde inferior mesa)
                  Re-segmenta bordes de corte con EfficientSAM
                  Mide la distancia en mm desde cada segmentación hasta la línea BL
                  ──► outputs/figures/cut_edge_distances/
```

---

## Notebooks

Todos los notebooks están en `test/`. Ejecutar en el orden indicado.

### 1. `qr_detection.ipynb`
Detecta los 3 marcadores QR de calibración en cada imagen raw.

- **Modelo**: YOLOv9 `michelin_v2`, clase `qr_code` (índice 1)
- **Etiquetado posicional**: asigna TL / TR / BL / BR según la suma y diferencia de coordenadas
- **Homografía**: con 3 QRs calcula una transformación afín; con 4, una homografía completa. Mapea imagen → canvas de 1000 × 1400 px donde TL=(0,0), TR=(1000,0), BL=(0,1400)
- **Salida**: `outputs/detections/qr_homographies.json` con posiciones de centros y matriz H
- **Métricas** (dataset train+valid): P=0.901 · R=1.000 · F1=0.948 · IoU medio=0.866

### 2. `rubber_detection.ipynb`
Detecta las bandas de goma (rubber strips) en las imágenes raw.

- **Modelo**: YOLOv9 `michelin_v2`, clase `rubber_strip` (índice 2), conf≥0.65
- **Salida**: `outputs/detections/rubber_detections.json`
- **Métricas**: P=0.978 · R=1.000 · F1=0.989 · IoU medio=0.864

### 3. `rubber_segmentation.ipynb`
Genera máscaras píxel a píxel de cada banda de goma.

- **Modelo**: EfficientSAM ViT-Tiny (`efficient_sam_vitt.pt`)
- **Prompt**: esquinas TL y BR de cada bbox de `rubber_detections.json`
- **Salida**: visualizaciones en `outputs/figures/rubber_segmentation/`

### 4. `cut_edge_detection.ipynb`
Detecta los bordes de corte de las bandas de goma.

- **Modelo**: YOLOv9 `michelin_v3-4`, clase `cut_edge` (índice 0), conf≥0.65
- **Filtros**: ratio de área ≤ 10% de la imagen (los bordes son tiras estrechas); NMS adicional con IoU=0.30 para suprimir detecciones duplicadas
- **Salida**: `outputs/detections/cut_edge_detections.json`

### 5. `cut_edge_segmentation.ipynb`
Genera máscaras píxel a píxel de cada borde de corte.

- **Modelo**: EfficientSAM ViT-Tiny
- **Prompt**: esquinas TL y BR de cada bbox de `cut_edge_detections.json`
- **Salida**: visualizaciones en `outputs/figures/cut_edge_segmentation/`

### 6. `cut_edge_measurement.ipynb`
Mide la distancia en milímetros desde cada borde de corte hasta el borde inferior de la mesa.

**Geometría de calibración**

Los 3 QRs forman un triángulo rectángulo con catetos de 800 mm. La homografía H mapea el espacio imagen al espacio canónico donde:

```
TL → (0,    0)      TR → (1000,    0)
BL → (0, 1400)      BR → (1000, 1400)
```

Esto da las escalas:
- Eje X: 800 mm / 1000 px = **0.800 mm/px canónico**
- Eje Y: 800 mm / 1400 px ≈ **0.571 mm/px canónico**

**Referencia "borde inferior de la mesa"**

Se traza una línea horizontal en la coordenada Y del QR BL en el espacio imagen. En el espacio canónico corresponde a y = 1400.

**Cálculo de distancia**

Para el punto más bajo de cada máscara de borde de corte `(px, py)`:

```
canon = H · [px, py, 1]ᵀ          # proyección al espacio canónico
dist_mm = (1400 - canon_y) · 0.571 # distancia hasta BL en mm
```

**Salida**: imágenes anotadas en `outputs/figures/cut_edge_distances/` con:
- Triángulo QR en verde
- Línea de referencia BL en naranja
- Máscara de cada borde de corte en cian
- Línea vertical violeta desde el borde de la máscara hasta la línea BL con la distancia en mm

---

## Notebooks auxiliares

| Notebook | Propósito |
|---|---|
| `yolo_finetune.ipynb` | Fine-tuning de YOLOv9 sobre el dataset de Michelin |
| `efficientSAM.ipynb` | Experimentos y pruebas de EfficientSAM |
| `check_esam.py` | Verificación rápida de la instalación de EfficientSAM |

---

## Estructura de directorios

```
models/Jaime/
├── test/                          # Notebooks del pipeline
│   ├── qr_detection.ipynb
│   ├── rubber_detection.ipynb
│   ├── rubber_segmentation.ipynb
│   ├── cut_edge_detection.ipynb
│   ├── cut_edge_segmentation.ipynb
│   ├── cut_edge_measurement.ipynb
│   ├── efficient_sam_vitt.pt      # Pesos EfficientSAM ViT-Tiny
│   └── runs/                      # Pesos YOLOv9 entrenados
│       └── data/runs/
│           ├── michelin_v2/weights/best.pt   # QR + rubber strip
│           └── michelin_v3-4/weights/best.pt # cut edge
│
├── data/
│   ├── raw/                        # 22 imágenes originales (multimedia_01..22.jpg)
│   ├── new_samples_qr_rubber/      # Dataset etiquetado QR + rubber (train/valid)
│   └── segmentation_cut_edge/      # Dataset etiquetado cut edge (train/valid)
│
└── outputs/
    ├── detections/
    │   ├── qr_homographies.json    # Centros QR + matrices H
    │   ├── rubber_detections.json  # Bboxes de rubber strips
    │   └── cut_edge_detections.json # Bboxes de bordes de corte
    └── figures/
        ├── qr_detection/           # Visualizaciones QR (GT vs predicción)
        ├── rubber_detection/       # Visualizaciones rubber (predicciones)
        ├── rubber_segmentation/    # Máscaras de rubber strips
        ├── cut_edge_detection/     # Visualizaciones cut edge
        ├── cut_edge_segmentation/  # Máscaras de bordes de corte
        └── cut_edge_distances/     # Imágenes con distancias medidas en mm
```

---

## Requisitos

```
ultralytics       # YOLOv9
torch             # PyTorch (CUDA recomendado para EfficientSAM)
opencv-python
matplotlib
numpy
efficient_sam     # instalación local desde el repositorio EfficientSAM
```

Los notebooks asumen que el kernel se lanza desde `test/` (rutas relativas `../data/`, `../outputs/`).

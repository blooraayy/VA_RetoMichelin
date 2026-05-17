================================================================================
 RETO MICHELIN 2 — DETECCION Y MEDIDA DE BORDES DE CORTE EN BANDAS DE GOMA
 Asignatura : Vision Artificial
 Titulacion : Grado en Ingenieria de Datos e Inteligencia Artificial
 Universidad: Universidad de Leon · Curso 2025-2026
 Reto       : II Premio al Talento Joven - Cluster FACyL & Michelin Aranda
================================================================================

CONTENIDO
---------
 1. Que hace este repositorio
 2. Estructura del repositorio
 3. Dataset (descarga GDrive)
 4. Requisitos generales
 5. Instalacion por pipeline
      5.1 Pablo  - Grounded-SAM           (Deep Learning, GPU recomendada)
      5.2 Jaime  - YOLOv9 + EfficientSAM  (Deep Learning, GPU recomendada)
      5.3 Miguel - Clasico OpenCV         (CPU)
      5.4 David  - Clasico Canny adaptativo (CPU)
 6. Ejecucion de cada pipeline
 7. Resultados resumidos
 8. Documentacion ampliada


1. QUE HACE ESTE REPOSITORIO
----------------------------
Sistema de inspeccion automatica de bandas de goma en el montaje de neumaticos
de camion. Cada pipeline detecta y segmenta:
  - los marcadores QR (calibracion pixel <-> mm),
  - las bandas de goma (rubber strips) sobre la mesa,
  - los bordes de corte (cut edges) entre tramos contiguos,
y estima la distancia metrica desde 10 puntos de cada borde hasta la linea
inferior de la mesa de trabajo.

El reto se ha abordado con cuatro metodos en paralelo (dos deep learning y
dos clasicos) para comparar precision, robustez y coste computacional.


2. ESTRUCTURA DEL REPOSITORIO
-----------------------------
  VA_RetoMichelin/
  |
  +-- data/
  |   `-- michelin_dataset/      # 22 imagenes JPG del puesto Michelin
  |
  +-- docs/                      # Enunciado, papers y entregables del reto
  |   |-- SUMMARY.docx           # Resumen 1 pagina (tabla + recomendacion)
  |   |-- 2526.GIDeIA.VA.00.Course Projects_Requirements_v3_MICHELIN.pdf
  |   |-- RETO_Cluster_FACYL_MICHELIN.pdf
  |   |-- 1_papers/  2_overleaf/  3_presentation/  0_templates/
  |
  +-- models/
  |   |-- Pablo/                 # Pipeline deep learning Grounded-SAM
  |   |-- Jaime/                 # Pipeline deep learning YOLOv9 + EfficientSAM
  |   |-- Miguel/                # Pipeline clasico OpenCV
  |   |-- David/                 # Pipeline clasico Canny adaptativo
  |   `-- GroundingDINO/         # Repo auxiliar (no usado en runtime)
  |
  +-- README.md                  # Indice de papers consultados por cada autor
  +-- JAIMEreadme.md             # README detallado del pipeline de Jaime
  `-- README.TXT                 # (este archivo)


3. DATASET (DESCARGA GDRIVE)
---------------------------
Las 22 imagenes Multimedia_31..52 y las anotaciones COCO ya se distribuyen
en este repositorio dentro de:

    data/michelin_dataset/                          (imagenes)
    models/Pablo/data/labels/_annotations.coco.json (ground truth COCO)
    models/Jaime/data/                              (subsets etiquetados)
    models/Miguel/ground_truth/_annotations.coco.json
    models/David/data/labels/_annotations.coco.json

Si necesitas re-descargar el dataset original facilitado por Michelin:

    Dataset Michelin (22 imagenes + COCO):
    https://drive.google.com/file/d/1xhwequDsqBpoJPC2JjYo3izU5yU6h_xy/view?usp=sharing

Coloca las imagenes descargadas en   data/michelin_dataset/
y los _annotations.coco.json en las rutas indicadas arriba para cada pipeline.


4. REQUISITOS GENERALES
-----------------------
 - Sistema operativo : Windows 10/11, Linux o macOS.
 - Python            : 3.10 recomendado (3.9-3.11 testeados).
 - Git               : para clonar el repositorio y submodulos opcionales.
 - GPU NVIDIA con CUDA 12.x (RTX-class, >=8 GB VRAM)
                      : OBLIGATORIA en la practica para los pipelines DL
                        (Pablo y Jaime). Sin GPU pueden tardar ~30 s/imagen.
 - ~6 GB de disco para pesos descargados + cache de HuggingFace.

Se recomienda un entorno virtual por pipeline (los DL comparten torch, pero
los clasicos no requieren torch):

    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/macOS
    source venv/bin/activate


5. INSTALACION POR PIPELINE
---------------------------

5.1 PABLO - GROUNDED-SAM   (models/Pablo)
-----------------------------------------
Detecta QRs + rubber strips + cut edges con Grounding DINO (HuggingFace) y
segmenta con SAM ViT-H. Mide la distancia metrica de 10 puntos por borde.

  cd models/Pablo

  # 1) Solo GPU: instalar PyTorch CUDA 12.1 PRIMERO
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

  # 2) Resto de dependencias (incluye torch CPU si no se hizo el paso 1)
  pip install -r requirements.txt

  # 3) Descargar el peso SAM ViT-H (~2.4 GB) en weights/
  #    Windows PowerShell:
  New-Item -ItemType Directory -Force weights
  Invoke-WebRequest `
    -Uri https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth `
    -OutFile weights\sam_vit_h_4b8939.pth

  #    Linux/macOS:
  mkdir -p weights
  wget -P weights/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

Grounding DINO se descarga solo de HuggingFace (IDEA-Research/grounding-dino-base)
la primera vez que se ejecuta el pipeline; queda cacheado en ~/.cache/huggingface/.

Requisitos instalados : torch, torchvision, transformers, opencv-python, numpy,
                        pandas, matplotlib, Pillow, pycocotools, segment-anything.


5.2 JAIME - YOLOv9 + EFFICIENTSAM   (models/Jaime)
--------------------------------------------------
Pipeline en 6 notebooks: QR detect -> rubber detect -> rubber seg ->
cut-edge detect -> cut-edge seg -> measurement.

  cd models/Jaime

  # No hay requirements.txt en esta carpeta; instalar manualmente:
  pip install ultralytics torch torchvision opencv-python matplotlib numpy

  # EfficientSAM (clonar el repo oficial localmente):
  git clone https://github.com/yformer/EfficientSAM efficient_sam_repo
  cd efficient_sam_repo && pip install -e . && cd ..

Pesos necesarios (ya incluidos en el repo):
    test/efficient_sam_vitt.pt                                  (~40 MB)
    test/runs/data/runs/michelin_v2/weights/best.pt             (QR + rubber)
    test/runs/data/runs/michelin_v3-4/weights/best.pt           (cut edge)


5.3 MIGUEL - CLASICO OPENCV   (models/Miguel)
---------------------------------------------
Pipeline 100% clasico (OpenCV + SciPy), sin entrenamiento ni GPU.

  cd models/Miguel
  pip install opencv-python>=4.8 numpy>=1.24 scipy>=1.11 matplotlib>=3.7 pandas>=2.0

Verificar que el QR detector de OpenCV esta disponible:
  python -c "import cv2; cv2.QRCodeDetector(); print('OpenCV', cv2.__version__)"


5.4 DAVID - CLASICO CANNY ADAPTATIVO   (models/David)
-----------------------------------------------------
Pipeline classical adaptativo (Canny + Otsu + morfologia), tambien CPU-only.

  cd models/David
  pip install -r requirements.txt
  # Contenido: opencv-python, numpy, matplotlib, pandas, pycocotools


6. EJECUCION DE CADA PIPELINE
-----------------------------

6.1 PABLO   (Grounded-SAM, GPU recomendada)
-------------------------------------------
  cd models/Pablo

  # Procesar las 22 imagenes (~3 min en RTX 4070, ~6.85 s/img):
  python controller/pipeline_controller.py --folder data/images/ --device cuda --no-show

  # Una sola imagen:
  python controller/pipeline_controller.py --image data/images/Multimedia_31.jpg --device cuda

  # CPU (mucho mas lento, ~30 s/img):
  python controller/pipeline_controller.py --folder data/images/ --device cpu --no-show

  # Generar el informe de evaluacion (17 graficas + CSV):
  python evaluate.py
  # Salidas: data/outputs/<imagen>_*.png/json/csv
  #          data/outputs/evaluation_report/01_iou_f1_per_image.png ... 17_*.png
  #          data/outputs/evaluation_report/evaluation_report.csv


6.2 JAIME   (YOLOv9 + EfficientSAM, GPU)
----------------------------------------
Ejecutar los notebooks de  models/Jaime/test/  en orden:

  1. qr_detection.ipynb            -> outputs/detections/qr_homographies.json
  2. rubber_detection.ipynb        -> outputs/detections/rubber_detections.json
  3. rubber_segmentation.ipynb     -> outputs/figures/rubber_segmentation/
  4. cut_edge_detection.ipynb      -> outputs/detections/cut_edge_detections.json
  5. cut_edge_segmentation.ipynb   -> outputs/figures/cut_edge_segmentation/
  6. cut_edge_measurement.ipynb    -> outputs/figures/cut_edge_distances/

Lanzar el kernel desde  models/Jaime/test/  (los notebooks usan rutas relativas
../data/  y  ../outputs/).


6.3 MIGUEL   (Clasico OpenCV, CPU)
----------------------------------
  jupyter lab     # o jupyter notebook
  # Abrir   models/Miguel/final_model_complete.ipynb
  # Ejecutar todas las celdas en orden (Section 1 -> Section 7).

Salidas:
  outputs/evaluation_report/cut_edge_summary_metrics.{json,png}
  outputs/evaluation_report/qr_code_summary_metrics.{json,png}
  outputs/evaluation_report/measurements_10_points_all/*.csv


6.4 DAVID   (Clasico adaptativo, CPU)
-------------------------------------
Ejecutar los notebooks de  models/David/  en orden:

  01_preprocessing.ipynb -> 02_qr_calibration.ipynb -> 03_canny.ipynb
  -> 04_gap_boundary.ipynb -> 05_metric.ipynb -> 06_evaluation.ipynb

Cada notebook lee la salida del anterior. El reporte final queda en
  models/David/outputs/06_evaluation/coco_gt_report.json
  models/David/outputs/06_evaluation/final_report_table.csv


7. RESULTADOS RESUMIDOS (22 imagenes)
-------------------------------------
                            Det F1   Pixel IoU   MAE (mm)   Tiempo / img
  Grounded-SAM (Pablo)      0.879    0.530       11.14      ~6.85 s   (GPU)
  YOLOv9+ESAM  (Jaime)      QR 0.948 / Strip 0.989 (cut_edge: validacion visual)
  Clasico OCV  (Miguel)     0.879    0.574       24.63      <0.1 s    (CPU)
  Clasico Canny(David)      0.758    0.467       21.06      0.024 s   (CPU)

Recomendacion:
  - Mejor por metricas      -> Grounded-SAM (Pablo). Mejor MAE (11.14 mm) y F1
                               de deteccion mas alto empatado con Miguel; los
                               5 casos sin corte se resuelven sin falsos
                               positivos gracias al filtro de brillo.
  - Mejor por tiempo        -> Clasico adaptativo (David). ~24 ms/imagen
                               (~41 fps) en CPU, sin GPU ni framework DL.

Detalle ampliado en  docs/SUMMARY.docx.

================================================================================
 Autores
   - Pablo Ruiz Moran   (Deep learning track)
   - Jaime              (Deep learning track)
   - Miguel             (Classical track)
   - David Moran Gorgojo (Classical track)
================================================================================

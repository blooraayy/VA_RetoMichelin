================================================================================
  RUBBER STRIP SEPARATION & QR CODE DETECTION PIPELINE
  README — Setup, Dependencies & Dataset
================================================================================

PROJECT OVERVIEW
----------------
This project detects two types of objects in tabletop images:

  1. cut_edge   — Separations (gaps / overlaps) between rubber strip pieces.
  2. qr_code    — QR codes placed at the four corners of a white table surface.

Detection is done entirely with classical computer vision (OpenCV + SciPy).
Results are evaluated against a COCO-format Ground Truth exported from Roboflow.
Evaluation reports (CSV, PNG charts) are saved to outputs/evaluation_report/.


--------------------------------------------------------------------------------
RECOMMENDED PROJECT STRUCTURE
--------------------------------------------------------------------------------

  project_root/
  ├── models/
  │   └── Miguel/
  │       ├── final_model_complete.ipynb   ← main notebook
  │       └── ground_truth/
  │           └── _annotations.coco.json  ← COCO annotation file (see below)
  ├── data/
  │   └── michelin_dataset/               ← dataset images (see below)
  │       ├── Multimedia (31).jpg
  │       ├── Multimedia (32).jpg
  │       └── ...
  └── outputs/
      └── evaluation_report/              ← auto-created by the notebook
          └── measurements_10_points_all/

The notebook uses relative paths (../../../data/) so this layout must be
respected. You can adjust DATASET_PATH in Section 1 if your layout differs.


--------------------------------------------------------------------------------
DATASETS — GOOGLE DRIVE LINKS
--------------------------------------------------------------------------------

  [1] Image Dataset (michelin_dataset/)
      >>> https://drive.google.com/drive/folders/REPLACE_WITH_DATASET_LINK <<<
      Download and place all .jpg files inside:
        data/michelin_dataset/

  [2] COCO Ground Truth Annotations (_annotations.coco.json)
      >>> https://drive.google.com/file/d/REPLACE_WITH_ANNOTATIONS_LINK <<<
      Download and place at:
        models/Miguel/ground_truth/_annotations.coco.json

  NOTE TO MAINTAINER: Replace the placeholder URLs above with the actual
  shared Google Drive links before distributing this README.


--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------

Python version
  Python 3.9 or higher is recommended (tested on 3.10 and 3.11).

Core dependencies
  opencv-python       >= 4.8.0
  numpy               >= 1.24.0
  scipy               >= 1.11.0
  matplotlib          >= 3.7.0
  pandas              >= 2.0.0

All standard library modules used (os, json, csv, math, re) are included
with Python and require no separate installation.


--------------------------------------------------------------------------------
INSTALLATION
--------------------------------------------------------------------------------

Step 1 — (Recommended) Create a virtual environment
  python -m venv venv

  Activate it:
    Windows:  venv\Scripts\activate
    macOS/Linux:  source venv/bin/activate

Step 2 — Install dependencies

  Option A: install from requirements.txt (provided below)
    pip install -r requirements.txt

  Option B: install manually
    pip install opencv-python numpy scipy matplotlib pandas

Step 3 — Verify OpenCV QR detector availability
  The notebook uses cv2.QRCodeDetector(), which is included in the standard
  opencv-python package. No extra build flags are needed.

  Quick check:
    python -c "import cv2; d = cv2.QRCodeDetector(); print('OpenCV OK:', cv2.__version__)"


--------------------------------------------------------------------------------
REQUIREMENTS.TXT (copy to a file named requirements.txt)
--------------------------------------------------------------------------------

opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.7.0
pandas>=2.0.0


--------------------------------------------------------------------------------
RUNNING THE NOTEBOOK
--------------------------------------------------------------------------------

Step 1 — Start Jupyter
  jupyter notebook
  or
  jupyter lab

Step 2 — Open the notebook
  Navigate to models/Miguel/final_model_complete.ipynb and open it.

Step 3 — Run all cells in order (top to bottom)

  Section 1  — Configuration & path verification.
                Check the printed output to confirm:
                  · Dataset found: N images
                  · GT COCO found: ground_truth/_annotations.coco.json

  Section 2  — Loads the rubber-strip separation detection pipeline.
  Section 2B — Loads the QR detection module.
  Section 3  — Runs both detectors on all images.
                Prints per-image detection summary.
  Section 4  — Visualises separation predictions overlaid on images.
  Section 4B — Visualises QR predictions overlaid on images.
  Section 5  — Loads Ground Truth from COCO JSON and computes metrics
                (Precision, Recall, F1) for cut_edge and qr_code separately.
  Section 5B — Generates full evaluation reports with plots and CSVs saved to
                outputs/evaluation_report/.
  Section 5C — Generates combined (cut_edge + qr_code) evaluation report.
  Section 6  — Visualises GT vs prediction for failed images (QR).
  Section 7  — Takes 10-point measurements along each detected separation and
                saves annotated images + CSV files to
                outputs/evaluation_report/measurements_10_points_all/.

Step 4 — Find results
  After running, check:
    outputs/evaluation_report/
      cut_edge_summary_metrics.png
      cut_edge_per_image_metrics.csv
      qr_code_summary_metrics.png
      qr_code_per_image_metrics.csv
      combined_cut_edge_qr_code_summary_metrics.png
      ...
    outputs/evaluation_report/measurements_10_points_all/
      all_measurements.csv
      <image_name>_measurements_10_points.png
      <image_name>_measurements_10_points.csv


--------------------------------------------------------------------------------
CONFIGURATION OPTIONS (Section 1 of the notebook)
--------------------------------------------------------------------------------

  DATASET_PATH       Path to the folder containing .jpg images.
                     Default: ../../data/michelin_dataset

  COCO_GT_PATHS      List of candidate paths where the COCO JSON will be
                     searched. The first existing path is used.

  MATCH_DIST_PX      Maximum centre-to-centre distance (pixels) to count a
                     predicted separation as a True Positive.
                     Default: 150 px

  QR_MATCH_DIST_PX   Same threshold for QR code matching.
                     Default: 80 px

  PX_TO_MM           Pixel-to-mm conversion factor used in measurement reports.
                     Set to the real calibration value if known; default is 1.0
                     (results reported in pixels).

  METHOD_NAME        Label used in chart titles. Default: 'Pipeline'


--------------------------------------------------------------------------------
TROUBLESHOOTING
--------------------------------------------------------------------------------

"ERROR: no se encuentra '<DATASET_PATH>'"
  → Check that DATASET_PATH in Section 1 points to the correct folder.
    Run os.listdir('../../data') in a cell to see what folders exist.

"AVISO: no se ha encontrado _annotations.coco.json"
  → The COCO Ground Truth file is missing. Download it from the GDrive link
    in the DATASETS section above and place it at:
    models/Miguel/ground_truth/_annotations.coco.json

"No existe la categoría 'cut_edge' en el COCO"
  → The annotation file may be from a different export. Ensure the file was
    exported from Roboflow with categories named 'cut_edge' and 'qr_code'.

ImportError: No module named 'cv2'
  → Run: pip install opencv-python

Images appear in wrong order / metrics seem off
  → Filenames must be sortable (e.g. 'Multimedia (31).jpg').
    The notebook handles both 'Multimedia (N).jpg' and 'Multimedia_N.jpg'.


--------------------------------------------------------------------------------
CONTACT / MAINTAINER
--------------------------------------------------------------------------------

  Author : Miguel
  Project: Michelin rubber-strip detection
  Date   : 2025


================================================================================

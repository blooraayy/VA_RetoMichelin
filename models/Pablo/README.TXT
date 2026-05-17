================================================================================
 Michelin Challenge 2 - Grounded-SAM Pipeline
 Author: Pablo Ruiz Moran
================================================================================

CONTENT
-------
1. What this code does
2. Requirements
3. Install dependencies
4. Model weights
5. Dataset
6. Run the pipeline
7. Generate the evaluation report
8. Outputs you will get
9. Where to read the full documentation


1. WHAT THIS CODE DOES
----------------------
Detects and segments QR markers (used for pixel-to-mm calibration), rubber
strips, and the cut gap between rubber pieces on Michelin truck-tyre assembly
images. Returns 10 sample points along each cut and their distance in
millimetres to the bottom of the table. 
Includes an offline evaluation script that compares predictions against the ground truth 
and produces charts + a CSV report.


2. REQUIREMENTS
---------------
- Python 3.10 (tested) - 3.9 / 3.11 should also work.
- NVIDIA GPU with CUDA 12.x recommended (RTX-class, >= 8 GB VRAM).
  CPU mode works but is much slower (~30 s/image vs ~5 s/image on GPU).
- ~5 GB free disk for model weights + HuggingFace cache.
- OS: Windows 10/11, Linux or macOS.


3. INSTALL DEPENDENCIES
-----------------------
All Python dependencies are pinned in requirements.txt (next to this README).

    # GPU users only — install the CUDA 12.1 build of PyTorch first, so that
    # the subsequent "pip install -r requirements.txt" does not pull the CPU
    # wheel:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    # Then install everything else (and torch/torchvision on CPU-only setups):
    pip install -r requirements.txt

What requirements.txt installs:
    torch, torchvision                  PyTorch backbone
    transformers                        Grounding DINO (HuggingFace)
    segment-anything (from GitHub)      SAM ViT-H
    opencv-python, numpy, pandas        image + numeric stack
    matplotlib, Pillow                  rendering + image I/O
    pycocotools                         COCO ground-truth loader


4. MODEL WEIGHTS
----------------
You only need to download ONE weight file manually. Grounding DINO is loaded
automatically from HuggingFace ("IDEA-Research/grounding-dino-base") the first
time you run the pipeline and is cached under ~/.cache/huggingface/ .

Download the SAM ViT-H checkpoint into the "weights/" folder:

    File: sam_vit_h_4b8939.pth   (~2.4 GB)
    URL : https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

    # Linux / macOS
    mkdir -p weights
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P weights/

    # Windows PowerShell
    New-Item -ItemType Directory -Force weights
    Invoke-WebRequest `
      -Uri https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth `
      -OutFile weights\sam_vit_h_4b8939.pth

Note: the file weights/groundingdino_swint_ogc.pth (if present) is legacy and
NOT used by the current code; you can safely delete it to save space.


5. DATASET
----------
The Michelin dataset used for evaluation in this Block 2 SHIPS WITH THE CODE:

    data/images/         22 images
    data/labels/_annotations.coco.json
                         COCO ground-truth annotations (QR markers and cut
                         edges) labelled by the team.

No download is required - the images and labels are already inside this
folder. However, you can access to the dataset with this link:
    https://drive.google.com/file/d/1xhwequDsqBpoJPC2JjYo3izU5yU6h_xy/view?usp=sharing


6. RUN THE PIPELINE
-------------------
Full folder (recommended, ~3 minutes on RTX 4070):

    python controller/pipeline_controller.py --folder data/images/ --device cuda --no-show

Single image:

    python controller/pipeline_controller.py --image data/images/Multimedia_31.jpg --device cuda

CPU only:

    python controller/pipeline_controller.py --folder data/images/ --device cpu --no-show

Useful CLI flags (see "python controller/pipeline_controller.py --help"):
    --image  <path>     run on a single image
    --folder <path>     run on every image in the folder
    --labels <path>     override the GT label path/folder
    --output <path>     override the output folder (default: data/outputs/)
    --device {cuda,cpu}
    --no-show           do not pop up image windows (mandatory for headless)
    --no-save           do not write annotated PNG files


7. GENERATE THE EVALUATION REPORT
---------------------------------
After running the pipeline at least once (so aggregate_summary.json exists):

    python evaluate.py

This compares predictions against data/labels/_annotations.coco.json and writes
17 PNG charts + a CSV under data/outputs/evaluation_report/ .


8. OUTPUTS YOU WILL GET
-----------------------
Per image, under data/outputs/:
    <stem>_detection_<timestamp>.png     bboxes, QR markers, cut-edge masks
    <stem>_measurement_<timestamp>.png   annotated distances in mm
    <stem>_<timestamp>.json              all numeric results
    <stem>_measurements_<timestamp>.csv  one row per sampled point

After "evaluate.py", under data/outputs/evaluation_report/:
    01_iou_f1_per_image.png ... 17_qr_ap_curve.png
    evaluation_report.csv                full per-image metrics table

Headline results on the 22-image set (commit 3982b6c):
    Pixel Precision  90.63 %
    Pixel Recall     85.29 %
    Pixel F1         87.88 %
    Mean Pixel IoU   0.686
    MAE              16.34 mm
    RMSE             19.68 mm


9. WHERE TO READ THE FULL DOCUMENTATION
---------------------------------------
- README.md (this folder)         full project documentation, rubric mapping,
                                  design decisions, limitations, future work.
- ../../docs/                     course requirements PDFs, Michelin challenge
                                  brief, Overleaf research proposal.
- config/config.py                every tuneable threshold with comments.

================================================================================

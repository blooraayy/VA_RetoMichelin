# =============================================================================
# evaluate.py
# Offline evaluation report for the Grounded-SAM pipeline.
#
# Reads aggregate_summary.json (already produced by the pipeline) plus the
# COCO ground-truth annotations and generates:
#   - Per-image metric bar charts
#   - Distribution histograms
#   - Precision-Recall scatter with F1 iso-curves
#   - Detection-level TP/FP/FN breakdown per image
#   - Summary CSV report
#
# Usage:
#   python evaluate.py [--summary path/to/aggregate_summary.json]
#                      [--coco    path/to/_annotations.coco.json]
#                      [--outdir  path/to/output/folder]
#                      [--iou-threshold 0.3]
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.config import COCO_ANNOTATIONS, OUTPUT_DIR


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_summary(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_gt_counts_for_category(coco_json: str, category_name: str) -> dict[str, int]:
    """Return {image_stem: n_annotations} for the given category name."""
    with open(coco_json, encoding="utf-8") as f:
        coco = json.load(f)

    cat_id = None
    for c in coco.get("categories", []):
        if c["name"] == category_name:
            cat_id = c["id"]
            break
    if cat_id is None:
        return {}

    img_map = {img["id"]: img["file_name"] for img in coco["images"]}
    counts: dict[str, int] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] != cat_id:
            continue
        fname = img_map.get(ann["image_id"], "")
        stem  = os.path.splitext(os.path.basename(fname))[0].lower()
        stem = stem.split(".rf.")[0] if ".rf." in stem else stem
        counts[stem] = counts.get(stem, 0) + 1
    return counts


def load_gt_counts(coco_json: str) -> dict[str, int]:
    """Return {image_stem: n_cut_edge_annotations}."""
    return _load_gt_counts_for_category(coco_json, "cut_edge")


def load_gt_qr_counts(coco_json: str) -> dict[str, int]:
    """Return {image_stem: n_qr_code_annotations}."""
    return _load_gt_counts_for_category(coco_json, "qr_code")


def stem_of(image_path: str) -> str:
    return os.path.splitext(os.path.basename(image_path))[0].lower()


def build_dataframe(summary: list[dict], gt_counts: dict[str, int],
                    gt_qr_counts: dict[str, int],
                    iou_thr: float) -> pd.DataFrame:
    rows = []
    for rec in summary:
        ev = rec.get("evaluation")
        st = stem_of(rec["image_path"])
        n_gt = gt_counts.get(st, 0)
        n_pred = rec.get("n_edges", 0)

        per_iou = (ev.get("per_edge_iou") or []) if ev else []

        n_tp = sum(1 for v in per_iou if v >= iou_thr)
        n_fp = sum(1 for v in per_iou if v < iou_thr)
        n_fn = max(0, n_gt - n_tp)

        det_p  = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else float("nan")
        det_r  = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else float("nan")
        det_f1 = (2 * det_p * det_r / (det_p + det_r)
                  if (det_p + det_r) > 0 else float("nan"))

        # ── QR metrics ──────────────────────────────────────────────────────
        n_qr_gt   = gt_qr_counts.get(st, 0)
        n_qr_pred = rec.get("n_qr", 0)
        per_qr_iou = (ev.get("per_qr_iou") or []) if ev else []

        qr_tp = sum(1 for v in per_qr_iou if v >= iou_thr)
        qr_fp = sum(1 for v in per_qr_iou if v < iou_thr)
        qr_fn = max(0, n_qr_gt - qr_tp)

        qr_mean_iou = float(np.mean(per_qr_iou)) if per_qr_iou else float("nan")

        qr_det_p  = qr_tp / (qr_tp + qr_fp) if (qr_tp + qr_fp) > 0 else float("nan")
        qr_det_r  = qr_tp / (qr_tp + qr_fn) if (qr_tp + qr_fn) > 0 else float("nan")
        qr_det_f1 = (2 * qr_det_p * qr_det_r / (qr_det_p + qr_det_r)
                     if (qr_det_p + qr_det_r) > 0 else float("nan"))

        rows.append({
            "image":      os.path.basename(rec["image_path"]),
            "calib_ok":   rec.get("calibration_valid", False),
            "n_qr":       n_qr_pred,
            "n_pred":     n_pred,
            "n_gt":       n_gt,
            "n_tp":       n_tp,
            "n_fp":       n_fp,
            "n_fn":       n_fn,
            # pixel-level from pipeline
            "iou_px":     ev["iou"]       if ev else float("nan"),
            "precision_px": ev["precision"] if ev else float("nan"),
            "recall_px":  ev["recall"]    if ev else float("nan"),
            "f1_px":      ev["f1"]        if ev else float("nan"),
            "mae_mm":     ev["mae_mm"]    if ev else float("nan"),
            "rmse_mm":    ev["rmse_mm"]   if ev else float("nan"),
            # detection-level (cut edges)
            "det_precision": det_p,
            "det_recall":    det_r,
            "det_f1":        det_f1,
            # QR detection
            "n_qr_gt":       n_qr_gt,
            "qr_mean_iou":   qr_mean_iou,
            "qr_tp":         qr_tp,
            "qr_fp":         qr_fp,
            "qr_fn":         qr_fn,
            "qr_det_p":      qr_det_p,
            "qr_det_r":      qr_det_r,
            "qr_det_f1":     qr_det_f1,
            "elapsed_s":     rec.get("elapsed_s", float("nan")),
        })

    return pd.DataFrame(rows)


# ── Plot helpers ──────────────────────────────────────────────────────────────

PALETTE = {
    "deep":    "#2196F3",
    "tp":      "#4CAF50",
    "fp":      "#F44336",
    "fn":      "#9E9E9E",
    "neutral": "#607D8B",
}

def _save(fig, outdir: str, name: str) -> None:
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


def _short(image_name: str) -> str:
    return image_name.replace("Multimedia_", "M").replace(".jpg", "")


# ── Individual chart functions ────────────────────────────────────────────────

def plot_iou_f1_per_image(df: pd.DataFrame, outdir: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    x = np.arange(len(df))
    labels = [_short(n) for n in df["image"]]

    axes[0].bar(x, df["iou_px"], color=PALETTE["deep"], alpha=0.85)
    axes[0].axhline(df["iou_px"].mean(), color="red", lw=1.5,
                    linestyle="--", label=f"Mean = {df['iou_px'].mean():.3f}")
    axes[0].set_ylabel("Pixel IoU")
    axes[0].set_title("Pixel-level IoU per image")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 1)

    axes[1].bar(x, df["f1_px"], color=PALETTE["neutral"], alpha=0.85)
    axes[1].axhline(df["f1_px"].mean(), color="red", lw=1.5,
                    linestyle="--", label=f"Mean = {df['f1_px'].mean():.3f}")
    axes[1].set_ylabel("Pixel F1")
    axes[1].set_title("Pixel-level F1 per image")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    fig.suptitle("Pixel-level segmentation metrics per image", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, outdir, "01_iou_f1_per_image.png")


def plot_mae_rmse_per_image(df: pd.DataFrame, outdir: str) -> None:
    """Plot both MAE and RMSE per image.

    The previous version only plotted MAE while the title mentioned both
    metrics, which made the chart misleading.
    """
    valid = df.dropna(subset=["mae_mm", "rmse_mm"])
    x = np.arange(len(valid))
    labels = [_short(n) for n in valid["image"]]
    width = 0.36

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width/2, valid["mae_mm"],  width,
           label="MAE",  color=PALETTE["deep"], alpha=0.85)
    ax.bar(x + width/2, valid["rmse_mm"], width,
           label="RMSE", color=PALETTE["neutral"], alpha=0.85)

    mean_mae  = valid["mae_mm"].mean()
    mean_rmse = valid["rmse_mm"].mean()
    ax.axhline(mean_mae, color=PALETTE["deep"], lw=1.5, linestyle="--",
               label=f"Mean MAE = {mean_mae:.1f} mm")
    ax.axhline(mean_rmse, color=PALETTE["neutral"], lw=1.5, linestyle="--",
               label=f"Mean RMSE = {mean_rmse:.1f} mm")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Error (mm)")
    ax.set_title("MAE and RMSE per image  (dashed = mean)", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    _save(fig, outdir, "02_mae_rmse_per_image.png")


def plot_precision_recall_f1_per_image(df: pd.DataFrame, outdir: str) -> None:
    x = np.arange(len(df))
    labels = [_short(n) for n in df["image"]]
    width = 0.27

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width, df["precision_px"], width, label="Precision", color="#42A5F5", alpha=0.9)
    ax.bar(x,         df["recall_px"],    width, label="Recall",    color="#AB47BC", alpha=0.9)
    ax.bar(x + width, df["f1_px"],        width, label="F1",        color=PALETTE["neutral"], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Pixel-level Precision / Recall / F1 per image", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, outdir, "03_precision_recall_f1_per_image.png")


def plot_detection_level_per_image(df: pd.DataFrame, outdir: str,
                                   iou_thr: float) -> None:
    x = np.arange(len(df))
    labels = [_short(n) for n in df["image"]]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Stacked bar: TP / FP / FN
    axes[0].bar(x, df["n_tp"], label="TP", color=PALETTE["tp"])
    axes[0].bar(x, df["n_fp"], bottom=df["n_tp"], label="FP", color=PALETTE["fp"])
    axes[0].bar(x, df["n_fn"],
                bottom=df["n_tp"] + df["n_fp"],
                label="FN", color=PALETTE["fn"])
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Detection-level TP / FP / FN per image  (mask IoU ≥ {iou_thr})")
    axes[0].legend(fontsize=9)

    width = 0.27
    axes[1].bar(x - width, df["det_precision"], width, label="Precision", color="#42A5F5", alpha=0.9)
    axes[1].bar(x,         df["det_recall"],    width, label="Recall",    color="#AB47BC", alpha=0.9)
    axes[1].bar(x + width, df["det_f1"],        width, label="F1",        color=PALETTE["neutral"], alpha=0.9)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Detection-level P / R / F1 per image")
    axes[1].legend(fontsize=9)

    fig.suptitle("Detection-level evaluation (deep learning)", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, outdir, "04_detection_level_per_image.png")


def plot_metric_distributions(df: pd.DataFrame, outdir: str) -> None:
    metrics = [
        ("iou_px",      "Pixel IoU",           (0, 1)),
        ("f1_px",       "Pixel F1",            (0, 1)),
        ("precision_px","Pixel Precision",     (0, 1)),
        ("recall_px",   "Pixel Recall",        (0, 1)),
        ("mae_mm",      "MAE (mm)",            None),
        ("rmse_mm",     "RMSE (mm)",           None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for ax, (col, label, xlim) in zip(axes, metrics):
        data = df[col].dropna()
        ax.hist(data, bins=10, color=PALETTE["deep"], edgecolor="white", alpha=0.85)
        ax.axvline(data.mean(), color="red",    lw=1.5, linestyle="--",
                   label=f"Mean={data.mean():.3f}")
        ax.axvline(data.median(), color="orange", lw=1.5, linestyle="-.",
                   label=f"Median={data.median():.3f}")
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.legend(fontsize=7)

    fig.suptitle("Distribution of evaluation metrics across all images",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, outdir, "05_metric_distributions.png")


def plot_precision_recall_scatter(df: pd.DataFrame, outdir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (p_col, r_col, title) in zip(axes, [
        ("precision_px", "recall_px",    "Pixel-level P–R"),
        ("det_precision", "det_recall",  "Detection-level P–R"),
    ]):
        valid = df.dropna(subset=[p_col, r_col])

        # F1 iso-curves
        pp = np.linspace(0.01, 1.0, 300)
        for f1_val in [0.2, 0.4, 0.6, 0.8]:
            rr = f1_val * pp / (2 * pp - f1_val + 1e-9)
            mask = (rr >= 0) & (rr <= 1)
            ax.plot(pp[mask], rr[mask], "gray", lw=0.6, alpha=0.5)
            idx = np.argmin(np.abs(pp[mask] - 0.85))
            ax.text(pp[mask][idx], rr[mask][idx], f"F1={f1_val}",
                    fontsize=6, color="gray", alpha=0.7)

        sc = ax.scatter(valid[p_col], valid[r_col],
                        c=valid["mae_mm"].reindex(valid.index),
                        cmap="RdYlGn_r", s=60, zorder=3, edgecolors="white")
        for _, row in valid.iterrows():
            ax.annotate(_short(row["image"]), (row[p_col], row[r_col]),
                        fontsize=6, xytext=(4, 2), textcoords="offset points")

        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Precision")
        ax.set_ylabel("Recall")
        ax.set_title(title)
        plt.colorbar(sc, ax=ax, label="MAE (mm)")

    fig.suptitle("Precision–Recall space  (colour = MAE)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, outdir, "06_pr_scatter.png")


def plot_detections_counts(df: pd.DataFrame, outdir: str) -> None:
    """Plot predicted and ground-truth cut-gap counts per image."""
    x = np.arange(len(df))
    labels = [_short(n) for n in df["image"]]
    width = 0.36

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width/2, df["n_pred"], width, label="Predicted",
           color=PALETTE["deep"], alpha=0.85)
    ax.bar(x + width/2, df["n_gt"], width, label="Ground truth",
           color=PALETTE["neutral"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of cut gaps")
    ax.set_title("Predicted vs ground-truth cut-gap count per image")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, outdir, "07_detection_counts.png")


def plot_summary_table(df: pd.DataFrame, outdir: str) -> None:
    total_tp = df["n_tp"].sum()
    total_fp = df["n_fp"].sum()
    total_gt = df["n_gt"].sum()
    total_fn = max(0, total_gt - total_tp)

    deep_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    deep_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    deep_f1 = (2 * deep_p * deep_r / (deep_p + deep_r)
               if (deep_p + deep_r) > 0 else 0.0)
    deep_mae  = df["mae_mm"].mean()
    deep_rmse = df["rmse_mm"].mean()

    qr_total_tp = df["qr_tp"].sum()
    qr_total_fp = df["qr_fp"].sum()
    qr_total_fn = df["qr_fn"].sum()
    qr_p  = qr_total_tp / (qr_total_tp + qr_total_fp) if (qr_total_tp + qr_total_fp) > 0 else 0.0
    qr_r  = qr_total_tp / (qr_total_tp + qr_total_fn) if (qr_total_tp + qr_total_fn) > 0 else 0.0
    qr_f1 = (2 * qr_p * qr_r / (qr_p + qr_r) if (qr_p + qr_r) > 0 else 0.0)
    qr_mean_iou = df["qr_mean_iou"].mean()

    def _make_table(ax, headers, rows, section_rows: set):
        tbl = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1.4, 1.8)
        for j in range(len(headers)):
            tbl[0, j].set_facecolor("#37474F")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        for i in range(1, len(rows) + 1):
            if (i - 1) in section_rows:
                for j in range(len(headers)):
                    tbl[i, j].set_facecolor("#546E7A")
                    tbl[i, j].set_text_props(color="white", fontweight="bold")
            else:
                color = "#E3F2FD" if i % 2 == 0 else "#FAFAFA"
                for j in range(len(headers)):
                    tbl[i, j].set_facecolor(color)

    # ── Cut-edge summary table ────────────────────────────────────────────────
    cut_rows = [
        ["Precision (det.)",  f"{deep_p:.1%}"],
        ["Recall (det.)",     f"{deep_r:.1%}"],
        ["F1 (det.)",         f"{deep_f1:.1%}"],
        ["Mean IoU (pixel)",  f"{df['iou_px'].mean():.4f}"],
        ["Mean F1 (pixel)",   f"{df['f1_px'].mean():.4f}"],
        ["MAE (mm)",          f"{deep_mae:.1f}"],
        ["RMSE (mm)",         f"{deep_rmse:.1f}"],
        ["TP / FP / FN",      f"{int(total_tp)} / {int(total_fp)} / {int(total_fn)}"],
        ["Images evaluated",  str(len(df))],
    ]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axis("off")
    _make_table(ax, ["Metric", "Grounded-SAM"], cut_rows, section_rows=set())
    fig.suptitle("Cut edges — Summary metrics", fontsize=13, fontweight="bold", y=0.97)
    fig.tight_layout()
    _save(fig, outdir, "08_summary_table_cut_edges.png")

    # ── QR summary table ──────────────────────────────────────────────────────
    qr_rows = [
        ["Precision (det.)",  f"{qr_p:.1%}"],
        ["Recall (det.)",     f"{qr_r:.1%}"],
        ["F1 (det.)",         f"{qr_f1:.1%}"],
        ["Mean bbox IoU",     f"{qr_mean_iou:.4f}"],
        ["TP / FP / FN",      f"{int(qr_total_tp)} / {int(qr_total_fp)} / {int(qr_total_fn)}"],
        ["Images evaluated",  str(len(df))],
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axis("off")
    _make_table(ax, ["Metric", "Grounded-SAM"], qr_rows, section_rows=set())
    fig.suptitle("QR markers — Summary metrics", fontsize=13, fontweight="bold", y=0.97)
    fig.tight_layout()
    _save(fig, outdir, "08_summary_table_qr_markers.png")

    return deep_p, deep_r, deep_f1, deep_mae, deep_rmse, total_tp, total_fp, total_fn


def plot_ap_curve(summary: list[dict], gt_counts: dict[str, int],
                  outdir: str) -> None:
    """IoU-threshold sensitivity curve.

    This is useful for analysis, but it is not a full COCO mAP computation.
    It varies the mask-IoU threshold and recomputes global Precision / Recall / F1.
    The area under this curve is therefore reported as AUC, not as COCO AP.
    """
    thresholds = np.arange(0.05, 0.96, 0.05)
    precisions, recalls, f1s = [], [], []

    for thr in thresholds:
        total_tp = total_fp = total_fn = 0
        for rec in summary:
            ev  = rec.get("evaluation")
            st  = stem_of(rec["image_path"])
            n_gt = gt_counts.get(st, 0)
            per_iou = (ev.get("per_edge_iou") or []) if ev else []

            tp = sum(1 for v in per_iou if v >= thr)
            fp = sum(1 for v in per_iou if v < thr)
            fn = max(0, n_gt - tp)
            total_tp += tp; total_fp += fp; total_fn += fn

        p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
        r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p); recalls.append(r); f1s.append(f1)

    # AUC over the P-R points obtained by sweeping the IoU threshold.
    # This is a compact sensitivity summary, not COCO mAP.
    pairs    = sorted(zip(recalls, precisions))
    r_sorted = np.array([x[0] for x in pairs])
    p_sorted = np.array([x[1] for x in pairs])
    auc_pr = float(np.trapezoid(p_sorted, r_sorted))

    thr_list = thresholds.tolist()
    idx_50 = min(range(len(thr_list)), key=lambda i: abs(thr_list[i] - 0.50))
    idx_75 = min(range(len(thr_list)), key=lambda i: abs(thr_list[i] - 0.75))
    p_iou50, r_iou50 = precisions[idx_50], recalls[idx_50]
    p_iou75, r_iou75 = precisions[idx_75], recalls[idx_75]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel — P-R curve across IoU thresholds
    axes[0].plot(r_sorted, p_sorted, color=PALETTE["deep"], lw=2.5, marker="o",
                 markersize=4, label=f"AUC = {auc_pr:.3f}")
    axes[0].fill_between(r_sorted, p_sorted, alpha=0.15, color=PALETTE["deep"])
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("P–R curve obtained by sweeping the IoU threshold")
    axes[0].set_xlim(0, 1.02); axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    # Annotate precision/recall at IoU 0.50 and 0.75.
    for r_val, p_val, col, lbl in [
        (r_iou50, p_iou50, "orange", f"P@IoU=0.50: {p_iou50:.2f}"),
        (r_iou75, p_iou75, "red",    f"P@IoU=0.75: {p_iou75:.2f}"),
    ]:
        axes[0].scatter([r_val], [p_val], s=80, color=col, zorder=5)
        axes[0].annotate(lbl, (r_val, p_val), textcoords="offset points",
                         xytext=(6, 4), fontsize=8, color=col)

    # Right panel — P / R / F1 vs IoU threshold
    axes[1].plot(thresholds, precisions, color="#42A5F5", lw=2, marker="o",
                 markersize=4, label="Precision")
    axes[1].plot(thresholds, recalls,    color="#AB47BC", lw=2, marker="o",
                 markersize=4, label="Recall")
    axes[1].plot(thresholds, f1s,        color=PALETTE["neutral"], lw=2, marker="o",
                 markersize=4, label="F1")
    axes[1].axvline(0.30, color="orange", lw=1.5, linestyle="--",
                    label="IoU=0.30 (chosen TP threshold)")
    axes[1].axvline(0.50, color="red", lw=1.5, linestyle="--",
                    label="IoU=0.50")
    axes[1].set_xlabel("Mask IoU threshold")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Precision / Recall / F1 at each IoU threshold")
    axes[1].set_xlim(0.0, 1.0); axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"IoU-threshold sensitivity  |  AUC={auc_pr:.3f}  "
        f"P@0.50={p_iou50:.3f}  P@0.75={p_iou75:.3f}",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    _save(fig, outdir, "09_ap_curve.png")
    print(f"  AUC={auc_pr:.4f}  P@IoU0.50={p_iou50:.4f}  P@IoU0.75={p_iou75:.4f}")


# ── QR charts (mirror of cut-edge charts 01/03/04/05/06/07/09) ───────────────

def plot_qr_iou_per_image(df: pd.DataFrame, outdir: str) -> None:
    """QR bbox IoU per image — mirrors chart 01."""
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(df))
    labels = [_short(n) for n in df["image"]]

    ax.bar(x, df["qr_mean_iou"].fillna(0), color=PALETTE["deep"], alpha=0.85)
    mean_val = df["qr_mean_iou"].mean()
    ax.axhline(mean_val, color="red", lw=1.5, linestyle="--",
               label=f"Mean = {mean_val:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean bbox IoU")
    ax.set_title("QR markers — mean bbox IoU per image")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.suptitle("QR marker bbox IoU per image", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, outdir, "11_qr_iou_per_image.png")


def plot_qr_precision_recall_f1_per_image(df: pd.DataFrame, outdir: str,
                                          iou_thr: float) -> None:
    """QR detection P/R/F1 per image — mirrors chart 03."""
    x = np.arange(len(df))
    labels = [_short(n) for n in df["image"]]
    width = 0.27

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width, df["qr_det_p"], width, label="Precision", color="#42A5F5", alpha=0.9)
    ax.bar(x,         df["qr_det_r"], width, label="Recall",    color="#AB47BC", alpha=0.9)
    ax.bar(x + width, df["qr_det_f1"],width, label="F1",        color=PALETTE["neutral"], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(f"QR markers — detection Precision / Recall / F1 per image"
                 f"  (bbox IoU ≥ {iou_thr})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, outdir, "12_qr_precision_recall_f1_per_image.png")


def plot_qr_detection_level_per_image(df: pd.DataFrame, outdir: str,
                                      iou_thr: float) -> None:
    """QR TP/FP/FN stacked bar + P/R/F1 per image — mirrors chart 04."""
    x = np.arange(len(df))
    labels = [_short(n) for n in df["image"]]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].bar(x, df["qr_tp"], label="TP", color=PALETTE["tp"])
    axes[0].bar(x, df["qr_fp"], bottom=df["qr_tp"], label="FP", color=PALETTE["fp"])
    axes[0].bar(x, df["qr_fn"],
                bottom=df["qr_tp"] + df["qr_fp"],
                label="FN", color=PALETTE["fn"])
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"QR markers — TP / FP / FN per image  (bbox IoU ≥ {iou_thr})")
    axes[0].legend(fontsize=9)

    width = 0.27
    axes[1].bar(x - width, df["qr_det_p"],  width, label="Precision", color="#42A5F5", alpha=0.9)
    axes[1].bar(x,         df["qr_det_r"],  width, label="Recall",    color="#AB47BC", alpha=0.9)
    axes[1].bar(x + width, df["qr_det_f1"], width, label="F1",        color=PALETTE["neutral"], alpha=0.9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_ylabel("Score")
    axes[1].set_title("QR markers — detection P / R / F1 per image")
    axes[1].legend(fontsize=9)

    fig.suptitle("QR marker detection evaluation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, outdir, "13_qr_detection_level_per_image.png")


def plot_qr_metric_distributions(df: pd.DataFrame, outdir: str) -> None:
    """QR metric histograms — mirrors chart 05."""
    metrics = [
        ("qr_mean_iou", "Mean bbox IoU",   (0, 1)),
        ("qr_det_p",    "Precision",        (0, 1)),
        ("qr_det_r",    "Recall",           (0, 1)),
        ("qr_det_f1",   "F1",               (0, 1)),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (col, label, xlim) in zip(axes, metrics):
        data = df[col].dropna()
        ax.hist(data, bins=8, color=PALETTE["deep"], edgecolor="white", alpha=0.85)
        ax.axvline(data.mean(),   color="red",    lw=1.5, linestyle="--",
                   label=f"Mean={data.mean():.3f}")
        ax.axvline(data.median(), color="orange", lw=1.5, linestyle="-.",
                   label=f"Median={data.median():.3f}")
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.legend(fontsize=7)

    fig.suptitle("Distribution of QR marker evaluation metrics",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, outdir, "14_qr_metric_distributions.png")


def plot_qr_pr_scatter(df: pd.DataFrame, outdir: str) -> None:
    """QR P–R scatter with F1 iso-curves — mirrors chart 06."""
    valid = df.dropna(subset=["qr_det_p", "qr_det_r"])

    fig, ax = plt.subplots(figsize=(7, 6))

    pp = np.linspace(0.01, 1.0, 300)
    for f1_val in [0.2, 0.4, 0.6, 0.8]:
        rr = f1_val * pp / (2 * pp - f1_val + 1e-9)
        mask = (rr >= 0) & (rr <= 1)
        ax.plot(pp[mask], rr[mask], "gray", lw=0.6, alpha=0.5)
        idx = np.argmin(np.abs(pp[mask] - 0.85))
        ax.text(pp[mask][idx], rr[mask][idx], f"F1={f1_val}",
                fontsize=6, color="gray", alpha=0.7)

    sc = ax.scatter(valid["qr_det_p"], valid["qr_det_r"],
                    c=valid["qr_mean_iou"].reindex(valid.index),
                    cmap="RdYlGn", s=60, zorder=3, edgecolors="white")
    for _, row in valid.iterrows():
        ax.annotate(_short(row["image"]), (row["qr_det_p"], row["qr_det_r"]),
                    fontsize=6, xytext=(4, 2), textcoords="offset points")

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_title("QR markers — Precision–Recall space")
    plt.colorbar(sc, ax=ax, label="Mean bbox IoU")

    fig.suptitle("QR marker P–R scatter  (colour = mean bbox IoU)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, outdir, "15_qr_pr_scatter.png")


def plot_qr_detection_counts(df: pd.DataFrame, outdir: str) -> None:
    """Predicted vs GT QR count per image — mirrors chart 07."""
    x = np.arange(len(df))
    labels = [_short(n) for n in df["image"]]
    width = 0.36

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width/2, df["n_qr"],    width, label="Predicted",
           color=PALETTE["deep"], alpha=0.85)
    ax.bar(x + width/2, df["n_qr_gt"], width, label="Ground truth",
           color=PALETTE["neutral"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of QR markers")
    ax.set_title("QR markers — predicted vs ground-truth count per image")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, outdir, "16_qr_detection_counts.png")


def plot_qr_ap_curve(summary: list[dict], gt_qr_counts: dict[str, int],
                     outdir: str) -> None:
    """QR bbox-IoU sensitivity curve — mirrors chart 09."""
    thresholds = np.arange(0.05, 0.96, 0.05)
    precisions, recalls, f1s = [], [], []

    for thr in thresholds:
        total_tp = total_fp = total_fn = 0
        for rec in summary:
            ev  = rec.get("evaluation")
            st  = stem_of(rec["image_path"])
            n_gt = gt_qr_counts.get(st, 0)
            per_qr_iou = (ev.get("per_qr_iou") or []) if ev else []

            tp = sum(1 for v in per_qr_iou if v >= thr)
            fp = sum(1 for v in per_qr_iou if v < thr)
            fn = max(0, n_gt - tp)
            total_tp += tp; total_fp += fp; total_fn += fn

        p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
        r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p); recalls.append(r); f1s.append(f1)

    pairs    = sorted(zip(recalls, precisions))
    r_sorted = np.array([x[0] for x in pairs])
    p_sorted = np.array([x[1] for x in pairs])
    auc_pr   = float(np.trapezoid(p_sorted, r_sorted))

    thr_list = thresholds.tolist()
    idx_50 = min(range(len(thr_list)), key=lambda i: abs(thr_list[i] - 0.50))
    idx_75 = min(range(len(thr_list)), key=lambda i: abs(thr_list[i] - 0.75))
    p_iou50, r_iou50 = precisions[idx_50], recalls[idx_50]
    p_iou75, r_iou75 = precisions[idx_75], recalls[idx_75]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(r_sorted, p_sorted, color=PALETTE["deep"], lw=2.5, marker="o",
                 markersize=4, label=f"AUC = {auc_pr:.3f}")
    axes[0].fill_between(r_sorted, p_sorted, alpha=0.15, color=PALETTE["deep"])
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_title("QR — P–R curve (sweeping bbox IoU threshold)")
    axes[0].set_xlim(0, 1.02); axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=10)

    for r_val, p_val, col, lbl in [
        (r_iou50, p_iou50, "orange", f"P@IoU=0.50: {p_iou50:.2f}"),
        (r_iou75, p_iou75, "red",    f"P@IoU=0.75: {p_iou75:.2f}"),
    ]:
        axes[0].scatter([r_val], [p_val], s=80, color=col, zorder=5)
        axes[0].annotate(lbl, (r_val, p_val), textcoords="offset points",
                         xytext=(6, 4), fontsize=8, color=col)

    axes[1].plot(thresholds, precisions, color="#42A5F5", lw=2, marker="o",
                 markersize=4, label="Precision")
    axes[1].plot(thresholds, recalls,    color="#AB47BC", lw=2, marker="o",
                 markersize=4, label="Recall")
    axes[1].plot(thresholds, f1s,        color=PALETTE["neutral"], lw=2, marker="o",
                 markersize=4, label="F1")
    axes[1].axvline(0.30, color="orange", lw=1.5, linestyle="--",
                    label="IoU=0.30 (chosen TP threshold)")
    axes[1].axvline(0.50, color="red",    lw=1.5, linestyle="--", label="IoU=0.50")
    axes[1].set_xlabel("Bbox IoU threshold"); axes[1].set_ylabel("Score")
    axes[1].set_title("QR — Precision / Recall / F1 at each bbox IoU threshold")
    axes[1].set_xlim(0.0, 1.0); axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"QR bbox-IoU sensitivity  |  AUC={auc_pr:.3f}  "
        f"P@0.50={p_iou50:.3f}  P@0.75={p_iou75:.3f}",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    _save(fig, outdir, "17_qr_ap_curve.png")
    print(f"  [QR] AUC={auc_pr:.4f}  P@IoU0.50={p_iou50:.4f}  P@IoU0.75={p_iou75:.4f}")


def plot_processing_time(df: pd.DataFrame, outdir: str) -> None:
    valid = df.dropna(subset=["elapsed_s"])
    x = np.arange(len(valid))
    labels = [_short(n) for n in valid["image"]]

    # First image includes model loading — flag it
    first_img = labels[0] if len(labels) > 0 else ""
    colors = [
        "#FF8A65" if lbl == first_img else PALETTE["deep"]
        for lbl in labels
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(x, valid["elapsed_s"], color=colors, alpha=0.88)

    mean_all  = valid["elapsed_s"].mean()
    mean_rest = valid["elapsed_s"].iloc[1:].mean() if len(valid) > 1 else mean_all

    ax.axhline(mean_all,  color="red",    lw=1.5, linestyle="--",
               label=f"Mean (all) = {mean_all:.2f} s")
    ax.axhline(mean_rest, color="orange", lw=1.5, linestyle="-.",
               label=f"Mean (excl. 1st) = {mean_rest:.2f} s  "
                     f"≈ {60/mean_rest:.1f} imgs/min")

    # Value labels on bars
    for bar, val in zip(bars, valid["elapsed_s"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Time (s)")
    ax.set_title(
        "Processing time per image  "
        f"(orange = includes model loading,  mean excl. 1st ≈ {mean_rest:.2f} s/img)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, outdir, "10_processing_time.png")


def save_csv_report(df: pd.DataFrame, outdir: str) -> None:
    path = os.path.join(outdir, "evaluation_report.csv")
    df.to_csv(path, index=False, float_format="%.4f")
    print(f"  Saved -> {path}")


def print_aggregate_summary(df: pd.DataFrame, iou_thr: float) -> None:
    total_tp = df["n_tp"].sum()
    total_fp = df["n_fp"].sum()
    total_gt = df["n_gt"].sum()
    total_fn = max(0, total_gt - total_tp)

    deep_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    deep_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    deep_f1 = (2 * deep_p * deep_r / (deep_p + deep_r)
               if (deep_p + deep_r) > 0 else 0.0)

    qr_total_tp = df["qr_tp"].sum()
    qr_total_fp = df["qr_fp"].sum()
    qr_total_fn = df["qr_fn"].sum()
    qr_p  = qr_total_tp / (qr_total_tp + qr_total_fp) if (qr_total_tp + qr_total_fp) > 0 else 0.0
    qr_r  = qr_total_tp / (qr_total_tp + qr_total_fn) if (qr_total_tp + qr_total_fn) > 0 else 0.0
    qr_f1 = (2 * qr_p * qr_r / (qr_p + qr_r) if (qr_p + qr_r) > 0 else 0.0)

    sep = "=" * 60
    print(f"\n{sep}")
    print("  EVALUATION SUMMARY - Grounded-SAM (Deep Learning)")
    print(sep)
    print(f"  Images evaluated : {len(df)}")
    print(f"  Total GT edges   : {int(total_gt)}")
    print()
    print(f"  -- Pixel-level segmentation (cut edges) --")
    print(f"  Mean IoU         : {df['iou_px'].mean():.4f}")
    print(f"  Mean Precision   : {df['precision_px'].mean():.4f}")
    print(f"  Mean Recall      : {df['recall_px'].mean():.4f}")
    print(f"  Mean F1          : {df['f1_px'].mean():.4f}")
    print()
    print(f"  -- Detection-level cut edges  (IoU >= {iou_thr}) --")
    print(f"  TP = {int(total_tp)}  FP = {int(total_fp)}  FN = {int(total_fn)}")
    print(f"  Precision        : {deep_p:.4f}")
    print(f"  Recall           : {deep_r:.4f}")
    print(f"  F1               : {deep_f1:.4f}")
    print()
    print(f"  -- Measurement error --")
    print(f"  Mean MAE (mm)    : {df['mae_mm'].mean():.2f}")
    print(f"  Mean RMSE (mm)   : {df['rmse_mm'].mean():.2f}")
    print()
    print(f"  -- QR marker detection  (bbox IoU >= {iou_thr}) --")
    print(f"  Mean bbox IoU    : {df['qr_mean_iou'].mean():.4f}")
    print(f"  TP = {int(qr_total_tp)}  FP = {int(qr_total_fp)}  FN = {int(qr_total_fn)}")
    print(f"  Precision        : {qr_p:.4f}")
    print(f"  Recall           : {qr_r:.4f}")
    print(f"  F1               : {qr_f1:.4f}")
    print()
    print(f"  {'-'*40}")
    print(f"{sep}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Offline evaluation report for the Grounded-SAM pipeline."
    )
    p.add_argument("--summary", default=os.path.join(OUTPUT_DIR, "aggregate_summary.json"),
                   help="Path to aggregate_summary.json")
    p.add_argument("--coco",    default=COCO_ANNOTATIONS,
                   help="Path to COCO annotations JSON")
    p.add_argument("--outdir",  default=os.path.join(OUTPUT_DIR, "evaluation_report"),
                   help="Output directory for charts and CSV")
    p.add_argument("--iou-threshold", type=float, default=0.3,
                   dest="iou_thr",
                   help="IoU threshold for detection-level TP/FP/FN")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"\n[Evaluate] Summary    : {args.summary}")
    print(f"[Evaluate] COCO GT    : {args.coco}")
    print(f"[Evaluate] Output dir : {args.outdir}")
    print(f"[Evaluate] IoU thr    : {args.iou_thr}\n")

    summary      = load_summary(args.summary)
    gt_counts    = load_gt_counts(args.coco)    if os.path.isfile(args.coco) else {}
    gt_qr_counts = load_gt_qr_counts(args.coco) if os.path.isfile(args.coco) else {}
    if not gt_counts:
        print("[Evaluate] WARNING — COCO GT not found; GT counts set to 0.")

    df = build_dataframe(summary, gt_counts, gt_qr_counts, args.iou_thr)

    print(f"[Evaluate] Generating charts...\n")
    plot_iou_f1_per_image(df, args.outdir)
    plot_mae_rmse_per_image(df, args.outdir)
    plot_precision_recall_f1_per_image(df, args.outdir)
    plot_detection_level_per_image(df, args.outdir, args.iou_thr)
    plot_metric_distributions(df, args.outdir)
    plot_precision_recall_scatter(df, args.outdir)
    plot_detections_counts(df, args.outdir)
    plot_summary_table(df, args.outdir)
    plot_ap_curve(summary, gt_counts, args.outdir)
    plot_processing_time(df, args.outdir)
    # ── QR charts (mirrors of cut-edge charts) ──────────────────────────────
    plot_qr_iou_per_image(df, args.outdir)
    plot_qr_precision_recall_f1_per_image(df, args.outdir, args.iou_thr)
    plot_qr_detection_level_per_image(df, args.outdir, args.iou_thr)
    plot_qr_metric_distributions(df, args.outdir)
    plot_qr_pr_scatter(df, args.outdir)
    plot_qr_detection_counts(df, args.outdir)
    plot_qr_ap_curve(summary, gt_qr_counts, args.outdir)
    save_csv_report(df, args.outdir)

    print_aggregate_summary(df, args.iou_thr)
    print(f"[Evaluate] Done. All outputs in: {args.outdir}\n")


if __name__ == "__main__":
    main()

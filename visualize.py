"""
CT Scan Denoising – Visualisation & Results Report
====================================================
Generates publication-quality comparison figures and a full metric summary
report from `denoising_metrics.csv`.

Usage:
    # Full report from existing metrics CSV
    python visualize.py --metrics denoising_metrics.csv --output_dir ./results

    # Sample grid from a folder of paired images (no model needed)
    python visualize.py --noisy_dir ./Noisy --clean_dir ./Clean --output_dir ./results
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
from matplotlib import gridspec  # pylint: disable=wrong-import-position
import seaborn as sns  # pylint: disable=wrong-import-position
import cv2  # pylint: disable=wrong-import-position

# ── Style ──────────────────────────────────────────────────────────────────────
DARK_BG   = "#0d0d14"
CARD_BG   = "#151520"
ACCENT1   = "#7c6efd"   # purple
ACCENT2   = "#2ecc71"   # green
ACCENT3   = "#e74c3c"   # red
ACCENT4   = "#f39c12"   # amber
TEXT      = "#e8e8f0"
SUBTEXT   = "#888898"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    CARD_BG,
    "axes.edgecolor":    "#2a2a3a",
    "axes.labelcolor":   TEXT,
    "xtick.color":       SUBTEXT,
    "ytick.color":       SUBTEXT,
    "text.color":        TEXT,
    "grid.color":        "#1e1e2e",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
})


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_gray(path):
    """Load an image as [0, 1] float32 grayscale."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 255.0


def ensure(directory):
    """Create directory tree if it does not exist."""
    os.makedirs(directory, exist_ok=True)
    return directory


# ── 1. Metric Distribution Plots ──────────────────────────────────────────────

def plot_metric_distributions(df, out_dir):
    """Plot histograms of PSNR, SSIM and MSE reduction."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=DARK_BG)
    fig.suptitle("Metric Distributions — Noisy vs Denoised", color=TEXT,
                 fontsize=16, fontweight="bold", y=1.02)

    # PSNR
    ax = axes[0]
    ax.hist(df["noisy_psnr"],    bins=30, color=ACCENT3, alpha=0.75, label="Noisy")
    ax.hist(df["denoised_psnr"], bins=30, color=ACCENT2, alpha=0.75, label="Denoised")
    ax.set_title("PSNR (dB)")
    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel("Count")
    ax.legend(facecolor=CARD_BG, labelcolor=TEXT)
    ax.grid(True)

    # SSIM
    ax = axes[1]
    ax.hist(df["noisy_ssim"],    bins=30, color=ACCENT3, alpha=0.75, label="Noisy")
    ax.hist(df["denoised_ssim"], bins=30, color=ACCENT2, alpha=0.75, label="Denoised")
    ax.set_title("SSIM")
    ax.set_xlabel("SSIM")
    ax.legend(facecolor=CARD_BG, labelcolor=TEXT)
    ax.grid(True)

    # MSE Reduction %
    ax = axes[2]
    ax.hist(df["mse_reduction_percent"], bins=30, color=ACCENT1, alpha=0.85)
    mse_mean = df['mse_reduction_percent'].mean()
    ax.axvline(mse_mean, color=ACCENT4,
               linestyle="--", linewidth=2,
               label=f"Mean: {mse_mean:.1f}%")
    ax.set_title("MSE Reduction (%)")
    ax.set_xlabel("MSE Reduction (%)")
    ax.legend(facecolor=CARD_BG, labelcolor=TEXT)
    ax.grid(True)

    plt.tight_layout()
    out = os.path.join(out_dir, "metric_distributions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  \u2713 metric_distributions.png")
    return out


# ── 2. Scatter: Noisy PSNR vs Improvement ─────────────────────────────────────

def plot_improvement_scatter(df: pd.DataFrame, out_dir: str) -> str:
    """Scatter plot of PSNR improvement vs starting noise level.

    Args:
        df (pd.DataFrame): DataFrame containing 'noisy_psnr',
                           'psnr_improvement', and 'ssim_improvement' columns.
        out_dir (str): Directory to save the plot.

    Returns:
        str: Path to the saved plot.
    """
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)

    sc = ax.scatter(df["noisy_psnr"], df["psnr_improvement"],
                    c=df["ssim_improvement"], cmap="plasma",
                    s=30, alpha=0.7, edgecolors="none")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("SSIM Improvement", color=TEXT)
    cb.ax.yaxis.set_tick_params(color=SUBTEXT)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=SUBTEXT)

    ax.set_title("PSNR Improvement vs Starting Noisiness", fontweight="bold")
    ax.set_xlabel("Noisy Input PSNR (dB)")
    ax.set_ylabel("PSNR Improvement (dB)")
    ax.grid(True)

    plt.tight_layout()
    out = os.path.join(out_dir, "improvement_scatter.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  \u2713 improvement_scatter.png")
    return out


# ── 3. Summary Dashboard ───────────────────────────────────────────────────────

def plot_summary_dashboard(df: pd.DataFrame, out_dir: str) -> str:
    """Create a multi-panel summary dashboard with KPIs and charts.

    Args:
        df (pd.DataFrame): DataFrame containing 'psnr_improvement',
                           'ssim_improvement', and 'mse_reduction_percent'
                           columns.
        out_dir (str): Directory to save the plot.

    Returns:
        str: Path to the saved plot.
    """
    fig = plt.figure(figsize=(16, 10), facecolor=DARK_BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── KPI cards (top row) ──
    kpis = [
        ("Avg PSNR Gain", f"+{df['psnr_improvement'].mean():.2f} dB", ACCENT2),
        ("Avg SSIM Gain", f"+{df['ssim_improvement'].mean():.4f}", ACCENT1),
        ("Avg MSE Reduction", f"{df['mse_reduction_percent'].mean():.1f}%",
         ACCENT4),
    ]
    for col, (label, value, colour) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(CARD_BG)
        ax.text(0.5, 0.62, value, ha="center", va="center", fontsize=26,
                fontweight="bold", color=colour, transform=ax.transAxes)
        ax.text(0.5, 0.28, label, ha="center", va="center", fontsize=12,
                color=SUBTEXT, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(colour)
            s.set_linewidth(2)

    # ── Box-plots (bottom left + centre) ──
    ax_psnr = fig.add_subplot(gs[1, :2])
    data = pd.DataFrame({
        "Noisy PSNR":    df["noisy_psnr"],
        "Denoised PSNR": df["denoised_psnr"],
    })
    sns.boxplot(data=data, palette={"Noisy PSNR": ACCENT3,
                                    "Denoised PSNR": ACCENT2},
                width=0.4, ax=ax_psnr, flierprops={"markerfacecolor": SUBTEXT})
    ax_psnr.set_title("PSNR Before & After Denoising")
    ax_psnr.set_ylabel("PSNR (dB)")
    ax_psnr.grid(True, axis="y")

    # ── MSE Reduction histogram (bottom right) ──
    ax_mse = fig.add_subplot(gs[1, 2])
    ax_mse.hist(df["mse_reduction_percent"], bins=25, color=ACCENT1, alpha=0.85)
    ax_mse.axvline(df["mse_reduction_percent"].mean(), color=ACCENT4,
                   linestyle="--", linewidth=2)
    ax_mse.set_title("MSE Reduction (%)")
    ax_mse.set_xlabel("%")
    ax_mse.grid(True)

    fig.suptitle("CT Scan Denoising — Model Performance Summary",
                 color=TEXT, fontsize=17, fontweight="bold", y=1.01)

    out = os.path.join(out_dir, "summary_dashboard.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  \u2713 summary_dashboard.png")
    return out


# ── 4. Image Grid (Noisy | Denoised | Clean) ──────────────────────────────────

def plot_image_grid(noisy_dir: str, clean_dir: str, out_dir: str,
                    n_samples: int = 5, denoised_dir: str = None) -> str | None:
    """Create a grid of n_samples rows: Noisy | (Denoised) | Clean.

    Args:
        noisy_dir (str): Directory containing noisy images.
        clean_dir (str): Directory containing clean ground truth images.
        out_dir (str): Directory to save the plot.
        n_samples (int, optional): Number of sample images to display.
                                   Defaults to 5.
        denoised_dir (str, optional): Directory containing denoised images.
                                      If None, the denoised column is omitted.
                                      Defaults to None.

    Returns:
        str | None: Path to the saved plot, or None if no images were found.
    """
    exts = {".png", ".jpg", ".jpeg"}
    files = sorted([f for f in os.listdir(noisy_dir)
                    if Path(f).suffix.lower() in exts])[:n_samples]
    if not files:
        print("  ! No images found for grid plot.")
        return None

    cols = 3 if denoised_dir else 2
    col_labels = ["🔴  Noisy Input", "🟢  Denoised", "🔵  Clean Ground Truth"] \
                 if denoised_dir else ["🔴  Noisy Input",
                                       "🔵  Clean Ground Truth"]

    fig, axes = plt.subplots(len(files), cols,
                             figsize=(5 * cols, 4 * len(files)),
                             facecolor=DARK_BG)
    if len(files) == 1:
        axes = [axes]

    colours = [ACCENT3, ACCENT2, "#3498db"] if denoised_dir else \
              [ACCENT3, "#3498db"]

    for row_i, fname in enumerate(files):
        imgs = [load_gray(os.path.join(noisy_dir, fname))]
        if denoised_dir and os.path.exists(path.join(denoised_dir, fname)):
            imgs.append(load_gray(os.path.join(denoised_dir, fname)))
        elif denoised_dir:
            imgs.append(np.zeros((256, 256), np.float32))
        imgs.append(load_gray(os.path.join(clean_dir, fname))
                    if os.path.exists(path.join(clean_dir, fname))
                    else np.zeros((256, 256), np.float32))

        for col_i, (img, col, clabel) in enumerate(zip(imgs, colours,
                                                       col_labels)):
            ax = axes[row_i][col_i]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if row_i == 0:
                ax.set_title(clabel, color=col, fontsize=12,
                             fontweight="bold", pad=6)
            for sp in ax.spines.values():
                sp.set_edgecolor(col)
                sp.set_linewidth(1.5)

    fig.suptitle("Sample Image Comparison", color=TEXT, fontsize=16,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(out_dir, "image_grid.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  \u2713 image_grid.png")
    return out


# ── 5. Training History ────────────────────────────────────────────────────────

def plot_training_history(history_png: str, out_dir: str) -> str | None:
    """Copies the training history PNG to the output directory.

    Args:
        history_png (str): Path to the source training history PNG.
        out_dir (str): Directory to save the copied PNG.

    Returns:
        str | None: Path to the copied PNG, or None if the source file
                    does not exist.
    """
    if not os.path.exists(history_png):
        return None
    img = cv2.imread(history_png)
    out = os.path.join(out_dir, "training_history.png")
    cv2.imwrite(out, img)
    print("  \u2713 training_history.png (copied)")
    return out


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    p = argparse.ArgumentParser(
        description="CT Denoising — Visualisation & Reporting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--metrics",      default="denoising_metrics.csv",
                   help="Path to metrics CSV from train.py")
    p.add_argument("--output_dir",   default="./results",
                   help="Directory to save all output figures")
    p.add_argument("--noisy_dir",    help="(Optional) Folder of noisy images for image grid")
    p.add_argument("--clean_dir",    help="(Optional) Folder of clean images for image grid")
    p.add_argument("--denoised_dir", help="(Optional) Folder of denoised images for image grid")
    p.add_argument("--n_samples",    type=int, default=5,
                   help="Number of sample images in comparison grid")
    p.add_argument("--history_png",  default="training_history.png",
                   help="Path to training history PNG from train.py")
    return p


def main():
    """CLI entry point for visualisation."""
    args = build_parser().parse_args()
    out  = ensure(args.output_dir)
    print(f"\n📊 Generating visualisations → {out}\n")

    if os.path.exists(args.metrics):
        df = pd.read_csv(args.metrics)
        print(f"Loaded {len(df)} rows from {args.metrics}")
        plot_metric_distributions(df, out)
        plot_improvement_scatter(df, out)
        plot_summary_dashboard(df, out)
    else:
        print(f"⚠  Metrics file not found: {args.metrics}  (skipping metric plots)")

    if args.noisy_dir and args.clean_dir:
        plot_image_grid(args.noisy_dir, args.clean_dir, out,
                        n_samples=args.n_samples,
                        denoised_dir=args.denoised_dir)

    plot_training_history(args.history_png, out)

    print(f"\n✅  All figures saved to: {out}")


if __name__ == "__main__":
    main()

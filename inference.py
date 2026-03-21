"""
CT Scan Denoising - Inference Script
=====================================
Run denoising on a single image or a folder of images using the trained U-Net model.

Usage:
    # Single image
    python inference.py --input noisy_ct.png --output denoised_ct.png

    # Batch folder
    python inference.py --input_dir ./noisy_images/ --output_dir ./denoised_results/

    # Show metrics alongside denoised output (if clean reference is available)
    python inference.py --input noisy.png --output denoised.png --reference clean.png
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

# ─── Constants ────────────────────────────────────────────────────────────────
IMG_HEIGHT = 256
IMG_WIDTH  = 256
IMG_CHANNELS = 1
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "denoising_model.h5")


# ─── Image I/O ────────────────────────────────────────────────────────────────
def load_image(path: str) -> np.ndarray:
    """Load a grayscale image, resize to model input size, normalise to [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    return img


def save_image(path: str, img: np.ndarray):
    """Save a [0, 1] float image as an 8-bit PNG."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, (img * 255).clip(0, 255).astype(np.uint8))


# ─── Model ────────────────────────────────────────────────────────────────────
def load_model(model_path: str = DEFAULT_MODEL_PATH) -> tf.keras.Model:
    """Load the trained Keras model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'.\n"
            "Train the model first by running:  python train.py"
        )
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    print("Model loaded successfully.")
    return model


def denoise(model: tf.keras.Model, noisy_img: np.ndarray) -> np.ndarray:
    """Run inference on a single (H, W) float32 image. Returns (H, W) float32."""
    inp = noisy_img.reshape(1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    pred = model.predict(inp, verbose=0)
    return pred[0].reshape(IMG_HEIGHT, IMG_WIDTH)


# ─── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray) -> dict:
    """Compute PSNR, SSIM, MSE before and after denoising."""
    noisy_psnr   = psnr(clean, noisy,    data_range=1.0)
    denoised_psnr = psnr(clean, denoised, data_range=1.0)
    noisy_ssim   = ssim(clean, noisy,    data_range=1.0)
    denoised_ssim = ssim(clean, denoised, data_range=1.0)
    noisy_mse    = np.mean((clean - noisy)    ** 2)
    denoised_mse = np.mean((clean - denoised) ** 2)
    return {
        "noisy_psnr":    round(noisy_psnr,   4),
        "denoised_psnr": round(denoised_psnr, 4),
        "psnr_gain":     round(denoised_psnr - noisy_psnr, 4),
        "noisy_ssim":    round(noisy_ssim,   4),
        "denoised_ssim": round(denoised_ssim, 4),
        "ssim_gain":     round(denoised_ssim - noisy_ssim, 4),
        "noisy_mse":     round(float(noisy_mse),    6),
        "denoised_mse":  round(float(denoised_mse), 6),
        "mse_reduction": round((1 - denoised_mse / noisy_mse) * 100, 2),
    }


def print_metrics(metrics):
    """Pretty-print PSNR / SSIM / MSE comparison table."""
    print("\n" + "=" * 48)
    print(f"  {'Metric':<22} {'Noisy':>10} {'Denoised':>10}")
    print("-" * 48)
    print(
        f"  {'PSNR (dB)':<22} {metrics['noisy_psnr']:>10.2f}"
        f" {metrics['denoised_psnr']:>10.2f}"
        f"  (+{metrics['psnr_gain']:.2f})"
    )
    print(
        f"  {'SSIM':<22} {metrics['noisy_ssim']:>10.4f}"
        f" {metrics['denoised_ssim']:>10.4f}"
        f"  (+{metrics['ssim_gain']:.4f})"
    )
    print(
        f"  {'MSE':<22} {metrics['noisy_mse']:>10.6f}"
        f" {metrics['denoised_mse']:>10.6f}"
    )
    print(
        f"  {'MSE Reduction':<22} {'':>10}"
        f" {metrics['mse_reduction']:>9.2f}%"
    )
    print("=" * 48 + "\n")


# ─── Visualisation ────────────────────────────────────────────────────────────
def save_comparison(noisy: np.ndarray, denoised: np.ndarray,
                    clean: np.ndarray | None, out_path: str,
                    title: str = ""):
    """Save a side-by-side comparison figure."""
    cols = 3 if clean is not None else 2
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5), facecolor="#0f0f14")
    subplot_args = [("Noisy Input", noisy, "#e74c3c"),
                    ("Denoised Output", denoised, "#2ecc71")]
    if clean is not None:
        subplot_args.append(("Ground Truth (Clean)", clean, "#3498db"))

    for ax, (label, img, colour) in zip(axes, subplot_args):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(label, color=colour, fontsize=13, pad=8, fontweight="bold")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(colour)
            spine.set_linewidth(2)

    if title:
        fig.suptitle(title, color="white", fontsize=15, y=1.01)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Comparison saved → {out_path}")


# ─── Single-image mode ────────────────────────────────────────────────────────
def run_single(args, model):
    """Denoise a single image and optionally compute metrics."""
    noisy = load_image(args.input)
    denoised = denoise(model, noisy)
    save_image(args.output, denoised)
    print(f"Denoised image saved → {args.output}")

    clean = None
    if args.reference:
        clean = load_image(args.reference)
        m = compute_metrics(clean, noisy, denoised)
        print_metrics(m)

    if args.compare:
        compare_path = (
            args.output
            .replace(".png", "_comparison.png")
            .replace(".jpg", "_comparison.png")
        )
        save_comparison(noisy, denoised, clean, compare_path,
                        title=os.path.basename(args.input))


# ─── Batch / folder mode ──────────────────────────────────────────────────────
def run_batch(args, model):
    """Denoise all images in a directory."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    files = [f for f in os.listdir(args.input_dir)
             if os.path.splitext(f)[1].lower() in exts]
    if not files:
        print("No image files found in the input directory.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Processing {len(files)} images …")

    for fname in tqdm(files):
        in_path  = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.output_dir, fname)
        try:
            noisy    = load_image(in_path)
            denoised = denoise(model, noisy)
            save_image(out_path, denoised)
            if args.compare:
                cmp_path = os.path.join(args.output_dir,
                                        os.path.splitext(fname)[0] + "_comparison.png")
                save_comparison(noisy, denoised, None, cmp_path)
        except (FileNotFoundError, OSError) as exc:
            print(f"  [SKIP] {fname}: {exc}")

    print(f"\nAll denoised images saved to: {args.output_dir}")


# ─── CLI ──────────────────────────────────────────────────────────────────────
def build_parser():
    """Build the argparse CLI parser."""
    p = argparse.ArgumentParser(
        description="CT Scan Denoising — Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,  # pylint: disable=undefined-variable
    )
    p.add_argument("--model",       default=DEFAULT_MODEL_PATH,
                   help="Path to trained model (.h5 / .keras). Default: denoising_model.h5")

    # Single-image mode
    single = p.add_argument_group("Single-image mode")
    single.add_argument("--input",     help="Path to noisy input image")
    single.add_argument("--output",    help="Path to save denoised image")
    single.add_argument("--reference", help="(Optional) Path to clean reference image for metrics")

    # Batch mode
    batch = p.add_argument_group("Batch / folder mode")
    batch.add_argument("--input_dir",  help="Folder of noisy images")
    batch.add_argument("--output_dir", help="Folder to save denoised images")

    p.add_argument("--compare", action="store_true",
                   help="Save side-by-side comparison PNG(s)")
    return p


def main():
    """CLI entry point for inference."""
    args = build_parser().parse_args()
    model = load_model(args.model)

    if args.input and args.output:
        run_single(args, model)
    elif args.input_dir and args.output_dir:
        run_batch(args, model)
    else:
        print("ERROR: Provide either (--input + --output) or (--input_dir + --output_dir).")
        build_parser().print_help()


if __name__ == "__main__":
    main()

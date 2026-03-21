"""
CT Scan Denoising - Preprocessing Pipeline
===========================================
Converts a raw dataset of lung CT/X-ray images into paired (noisy, clean) sets
ready for training.

Preprocessing steps applied:
  1. Grayscale conversion
  2. Resize to target resolution
  3. CLAHE contrast enhancement
  4. Gaussian denoising (clean images) / synthetic noise injection (noisy images)
  5. Normalisation [0, 1]
  6. Augmentation  (rotation, flip, translation)
  7. Save paired images to output directory

Usage:
    python preprocess.py \
        --input_dir  /path/to/raw_images \
        --output_dir /path/to/processed  \
        --img_size   256 \
        --augment    4
"""

import os
import argparse
import random
import numpy as np
import cv2
from tqdm import tqdm

# ─── Defaults ─────────────────────────────────────────────────────────────────
IMG_SIZE = 256
AUGMENT_FACTOR = 4      # number of augmented copies per raw image
NOISE_LEVEL_RANGE = (0.05, 0.25)   # random Gaussian noise σ


# ─── Core transforms ──────────────────────────────────────────────────────────

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR or RGBA image to single-channel grayscale."""
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    elif img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def resize_image(img, size):
    """Resize image to a square of the given size."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0,
                tile_grid: tuple = (8, 8)) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(img)


def normalise(img: np.ndarray) -> np.ndarray:
    """Scale pixel values to [0.0, 1.0] float32."""
    return img.astype(np.float32) / 255.0


def denormalize(img: np.ndarray) -> np.ndarray:
    """Convert [0.0, 1.0] float back to uint8."""
    return (img * 255).clip(0, 255).astype(np.uint8)


def add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    """Add Gaussian noise to a [0, 1] float32 image."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)


def add_speckle_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    """Add multiplicative speckle noise (common in CT images)."""
    noise = np.random.normal(1.0, sigma, img.shape).astype(np.float32)
    return np.clip(img * noise, 0.0, 1.0)


def add_mixed_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    """Mix of Gaussian + speckle noise for realistic CT degradation."""
    img = add_gaussian_noise(img, sigma * 0.6)
    img = add_speckle_noise(img, sigma * 0.4)
    return img


# ─── Augmentation ─────────────────────────────────────────────────────────────

def augment_image(img: np.ndarray, idx: int) -> np.ndarray:
    """Apply a deterministic augmentation based on the augmentation index."""
    aug_idx = idx % 8
    if aug_idx == 0:
        return img                                           # original
    if aug_idx == 1:
        return np.fliplr(img)                               # horizontal flip
    if aug_idx == 2:
        return np.flipud(img)                               # vertical flip
    if aug_idx == 3:
        return np.rot90(img, 1)                             # +90°
    if aug_idx == 4:
        return np.rot90(img, 2)                             # 180°
    if aug_idx == 5:
        return np.rot90(img, 3)                             # -90°
    if aug_idx == 6:
        # Small rotation ±15°
        angle = random.uniform(-15, 15)
        h, w = img.shape
        rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, rot_mat, (w, h))
    # Default: random translation ±10%
    h, w = img.shape
    tx = random.randint(-w // 10, w // 10)
    ty = random.randint(-h // 10, h // 10)
    trans_mat = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, trans_mat, (w, h))


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def preprocess_single(raw_path: str, img_size: int) -> np.ndarray | None:
    """Full preprocessing for one raw image. Returns float32 [0,1] array or None."""
    img = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    img = to_grayscale(img)
    img = resize_image(img, img_size)
    img = apply_clahe(img)
    img = normalise(img)
    return img


def process_dataset(input_dir: str, output_clean_dir: str, output_noisy_dir: str,
                    img_size: int, augment_factor: int, noise_type: str):
    """Process all images in input_dir and output paired clean/noisy sets."""
    os.makedirs(output_clean_dir, exist_ok=True)
    os.makedirs(output_noisy_dir, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    raw_files = [f for f in sorted(os.listdir(input_dir))
                 if os.path.splitext(f)[1].lower() in exts]

    if not raw_files:
        print("No image files found in the input directory. Exiting.")
        return

    print(f"Found {len(raw_files)} raw images.")
    n_expected = len(raw_files) * augment_factor
    print(
        f"Augmentation factor: {augment_factor}  "
        f"→  ~{n_expected} output pairs"
    )

    noise_fn = {
        "gaussian": add_gaussian_noise,
        "speckle":  add_speckle_noise,
        "mixed":    add_mixed_noise,
    }.get(noise_type, add_mixed_noise)

    saved_pairs = 0

    for fname in tqdm(raw_files, desc="Processing"):
        base, _ = os.path.splitext(fname)
        img = preprocess_single(os.path.join(input_dir, fname), img_size)
        if img is None:
            print(f"  [SKIP] Could not read: {fname}")
            continue

        for aug_i in range(augment_factor):
            aug_clean = augment_image(img, aug_i)
            sigma = random.uniform(*NOISE_LEVEL_RANGE)
            aug_noisy = noise_fn(aug_clean, sigma)

            suffix = "" if aug_i == 0 else f"_aug{aug_i}"
            out_name = f"{base}{suffix}.png"

            cv2.imwrite(
                os.path.join(output_clean_dir, out_name),
                denormalize(aug_clean)
            )
            cv2.imwrite(
                os.path.join(output_noisy_dir, out_name),
                denormalize(aug_noisy)
            )
            saved_pairs += 1

    print(f"\n✅ Done! Saved {saved_pairs} paired image sets.")
    print(f"   Clean images → {output_clean_dir}")
    print(f"   Noisy images → {output_noisy_dir}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_parser():
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        description="CT Scan Preprocessing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_dir",   required=True,
                   help="Directory of raw CT/X-ray images")
    p.add_argument("--output_dir",  required=True,
                   help="Root output directory (Clean/ and Noisy/ subfolders will be created)")
    p.add_argument("--img_size",    type=int, default=IMG_SIZE,
                   help="Target image size (square)")
    p.add_argument("--augment",     type=int, default=AUGMENT_FACTOR,
                   help="Number of augmented copies per raw image (1 = no augmentation)")
    p.add_argument("--noise_type",  choices=["gaussian", "speckle", "mixed"],
                   default="mixed",
                   help="Type of synthetic noise to inject into the noisy images")
    return p


def main():
    """CLI entry point for preprocessing."""
    args = build_parser().parse_args()
    clean_dir = os.path.join(args.output_dir, "Clean")
    noisy_dir = os.path.join(args.output_dir, "Noisy")
    process_dataset(
        input_dir=args.input_dir,
        output_clean_dir=clean_dir,
        output_noisy_dir=noisy_dir,
        img_size=args.img_size,
        augment_factor=args.augment,
        noise_type=args.noise_type,
    )


if __name__ == "__main__":
    main()

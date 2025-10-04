#!/usr/bin/env python3
import cv2
import os
import glob
import numpy as np


def unblur_image(img):
    """Simple Laplacian-based unblurring (deblurring effect)."""
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sharp = cv2.convertScaleAbs(img - 0.3 * laplacian)  # 0.3 controls strength
    return sharp


def enhance_image(img):
    """Apply natural rug enhancement pipeline with unblur."""

    # --- 1. Mild Unsharp Masking ---
    gaussian = cv2.GaussianBlur(img, (5, 5), 1)
    mild_unsharp = cv2.addWeighted(img, 1.2, gaussian, -0.2, 0)

    # --- 2. Bilateral filter + Sharpen ---
    bilateral = cv2.bilateralFilter(mild_unsharp, d=9, sigmaColor=75, sigmaSpace=75)
    gaussian_bi = cv2.GaussianBlur(bilateral, (5, 5), 1)
    bilateral_sharp = cv2.addWeighted(bilateral, 1.2, gaussian_bi, -0.2, 0)

    # --- 3. Unblur (Laplacian) ---
    final = unblur_image(bilateral_sharp)

    return final


def process_folder(input_folder, output_folder):
    """Batch process all PNG/JPG images in a folder."""
    os.makedirs(output_folder, exist_ok=True)

    files = glob.glob(os.path.join(input_folder, "*.png")) + \
            glob.glob(os.path.join(input_folder, "*.jpg")) + \
            glob.glob(os.path.join(input_folder, "*.jpeg"))

    if not files:
        print(f"No images found in {input_folder}")
        return

    for path in files:
        img = cv2.imread(path)
        if img is None:
            print(f"Skipping {path}, could not load.")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enhanced = enhance_image(img_rgb)

        # back to BGR for saving
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

        filename = os.path.basename(path)
        name, ext = os.path.splitext(filename)
        out_path = os.path.join(output_folder, f"{name}_enhanced{ext}")

        cv2.imwrite(out_path, enhanced_bgr)
        print(f"Saved {out_path}")

    print("✅ Enhancement complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Natural rug enhancement with unblur")
    parser.add_argument("--input", type=str, required=True, help="Input folder with rug images")
    parser.add_argument("--output", type=str, required=True, help="Output folder for enhanced images")
    args = parser.parse_args()

    process_folder(args.input, args.output)

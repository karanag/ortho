#!/usr/bin/env python3
import cv2
import numpy as np
import requests
import time
import os
from pathlib import Path
import argparse


# ──────────────────────────────────────────────
# FINAL: Fully Automatic Vibrance & Exposure Enhancement
# ──────────────────────────────────────────────# ──────────────────────────────────────────────
# FINAL: Fully Automatic Vibrance & Exposure Enhancement (Corrected)
# ──────────────────────────────────────────────
def enhance_rug_vibrance_auto_exposure(
    img_rgba: np.ndarray,
    saturation_scale: float = 1.15,
    target_brightness: int = 115,  # The desired median brightness (0-255)
    max_exposure_boost: float = 0.25, # The maximum boost to apply (e.g., 0.25 = max 25% boost)
    blend_strength: float = 0.85,
    clahe_limit: float = 1.2,
    sharpness_amount: float = 0.4
) -> np.ndarray:
    """
    Automatically enhances a rug's vibrance and exposure based on its content.

    Args:
        img_rgba (np.ndarray): The RGBA image from the background removal service.
        saturation_scale (float): Multiplier for saturation.
        target_brightness (int): The desired median luminance for the rug.
        max_exposure_boost (float): The upper limit for the automatic exposure boost.
        blend_strength (float): Overall intensity of the enhancement.
        clahe_limit (float): Clip limit for CLAHE.
        sharpness_amount (float): Amount for unsharp masking.

    Returns:
        np.ndarray: The enhanced BGR image.
    """
    if img_rgba is None or img_rgba.shape[2] < 4:
        # Handle cases with no valid image
        return img_rgba[:, :, :3] if img_rgba is not None else np.zeros((1,1,3), dtype=np.uint8)

    # 1. Separate original BGR and the alpha mask
    original_bgr = img_rgba[:, :, :3]
    alpha_mask = img_rgba[:, :, 3]
    is_rug = alpha_mask > 0

    # --- Step 1: Boost Color Vibrance (Saturation) ---
    hsv = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s_float = s.astype(np.float32)
    s_float[is_rug] *= saturation_scale
    s = np.clip(s_float, 0, 255).astype(np.uint8)
    vibrant_hsv = cv2.merge([h, s, v])
    vibrant_bgr = cv2.cvtColor(vibrant_hsv, cv2.COLOR_HSV2BGR)

    # --- Step 2: Automatic Exposure Calculation & Boost ---
    lab = cv2.cvtColor(vibrant_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Calculate the median brightness of ONLY the rug pixels
    current_median_brightness = np.median(l[is_rug])
    
    exposure_boost = 0.0
    if current_median_brightness < target_brightness:
        # Calculate the required boost to reach the target
        boost_needed = (target_brightness - current_median_brightness) / 255.0
        # We cap the boost at the maximum allowed value to prevent extreme changes
        exposure_boost = min(max_exposure_boost, boost_needed)
    
    print(f"    - Median Brightness: {current_median_brightness:.1f} -> Target: {target_brightness} -> Applied Boost: {exposure_boost:.2f}")

    # Apply the calculated exposure boost (if any)
    if exposure_boost > 0:
        l_float = l.astype(np.float32)
        l_float[is_rug] += (exposure_boost * 255)
        l = np.clip(l_float, 0, 255).astype(np.uint8)

    # Merge the potentially boosted L channel with original A and B channels from the vibrant image
    exposed_lab = cv2.merge([l, a, b])
    exposed_bgr = cv2.cvtColor(exposed_lab, cv2.COLOR_LAB2BGR)

    # --- Step 3: Gentle Contrast (CLAHE) on the exposed image ---
    lab_for_clahe = cv2.cvtColor(exposed_bgr, cv2.COLOR_BGR2LAB)
    l_clahe, a_clahe, b_clahe = cv2.split(lab_for_clahe)
    clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l_clahe)
    merged_lab = cv2.merge([cl, a_clahe, b_clahe])
    contrast_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    # --- Step 4: Controlled Sharpening (Unsharp Mask) ---
    blurred = cv2.GaussianBlur(contrast_bgr, (0, 0), 3)
    # CORRECTED LINE: The variable is now correctly spelled as 'sharpness_amount'
    sharp_bgr = cv2.addWeighted(contrast_bgr, 1.0 + sharpness_amount, blurred, -sharpness_amount, 0)
    
    # --- Step 5: Final Blend for Subtlety ---
    final_bgr = cv2.addWeighted(sharp_bgr, blend_strength, original_bgr, 1 - blend_strength, 0)

    return final_bgr
# ──────────────────────────────────────────────
# NEW: Vibrance & Richness Enhancement
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# UPDATED: Vibrance & Richness Enhancement with Exposure Boost
# ──────────────────────────────────────────────


def enhance_rug_vibrance(
    img_rgba: np.ndarray,
    saturation_scale: float = 1.15,
    exposure_boost: float = 0.1,  # NEW PARAMETER: Controls global brightness boost
    blend_strength: float = 0.8,
    clahe_limit: float = 1.2,
    sharpness_amount: float = 0.4
) -> np.ndarray:
    """
    Enhances a rug's vibrance and richness with an optional exposure boost,
    without altering its natural color balance.

    Args:
        img_rgba (np.ndarray): The RGBA image from the background removal service.
        saturation_scale (float): Multiplier for saturation. 1.0 is no change, 1.15 is a 15% boost.
        exposure_boost (float): A value (e.g., 0.1 or 0.2) to linearly brighten the image.
                                Applied to the L channel in LAB space.
        blend_strength (float): Overall intensity of the enhancement (0.0 to 1.0).
        clahe_limit (float): Clip limit for CLAHE. Lower is more subtle.
        sharpness_amount (float): Amount for unsharp masking.

    Returns:
        np.ndarray: The enhanced BGR image, ready for compositing.
    """
    if img_rgba is None or img_rgba.shape[2] < 4:
        print("⚠️ Warning: Invalid RGBA image passed to enhancer. Skipping.")
        return img_rgba[:, :, :3] if img_rgba is not None else np.zeros((1,1,3), dtype=np.uint8)

    # 1. Separate original BGR and the alpha mask
    original_bgr = img_rgba[:, :, :3]
    alpha_mask = img_rgba[:, :, 3]
    is_rug = alpha_mask > 0

    # --- Step 1: Boost Color Vibrance (Saturation) ---
    hsv = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    s_float = s.astype(np.float32)
    s_float[is_rug] = s_float[is_rug] * saturation_scale
    s = np.clip(s_float, 0, 255).astype(np.uint8)
    
    vibrant_hsv = cv2.merge([h, s, v])
    vibrant_bgr = cv2.cvtColor(vibrant_hsv, cv2.COLOR_HSV2BGR)

    # --- Step 2: Apply Global Exposure Boost ---
    # Convert the vibrant image to LAB color space
    lab_for_exposure = cv2.cvtColor(vibrant_bgr, cv2.COLOR_BGR2LAB)
    l_exp, a_exp, b_exp = cv2.split(lab_for_exposure)

    # Apply the exposure boost directly to the L-channel
    # Multiply by 255 to get a pixel value between 0-255 (L*1.0 = 100% brightness)
    # Ensure we only apply to rug pixels.
    l_exp_float = l_exp.astype(np.float32)
    l_exp_float[is_rug] = l_exp_float[is_rug] + (exposure_boost * 255) 
    
    l_exp = np.clip(l_exp_float, 0, 255).astype(np.uint8)
    
    # Merge channels and convert back to BGR
    exposed_lab = cv2.merge([l_exp, a_exp, b_exp])
    exposed_bgr = cv2.cvtColor(exposed_lab, cv2.COLOR_LAB2BGR)


    # --- Step 3: Gentle Contrast (CLAHE) on the exposed image ---
    # CLAHE works best on the L-channel of a LAB image
    lab_for_clahe = cv2.cvtColor(exposed_bgr, cv2.COLOR_BGR2LAB)
    l_clahe, a_clahe, b_clahe = cv2.split(lab_for_clahe)
    clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l_clahe)
    merged_lab = cv2.merge([cl, a_clahe, b_clahe])
    contrast_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    # --- Step 4: Controlled Sharpening (Unsharp Mask) ---
    blurred = cv2.GaussianBlur(contrast_bgr, (0, 0), 3)
    sharp_bgr = cv2.addWeighted(contrast_bgr, 1.0 + sharpness_amount, blurred, -sharpness_amount, 0)
    
    # --- Step 5: Final Blend for Subtlety ---
    final_bgr = cv2.addWeighted(sharp_bgr, blend_strength, original_bgr, 1 - blend_strength, 0)

    return final_bgr

# ──────────────────────────────────────────────
# Subtle & Natural Image Enhancement
# ──────────────────────────────────────────────
def enhance_rug_subtle(
    img_rgba: np.ndarray,
    blend_strength: float = 0.5,
    clahe_limit: float = 1.1,
    sharpness_amount: float = 0.5
) -> np.ndarray:
    """
    Applies a more subtle and natural set of enhancements to a rug image.

    Args:
        img_rgba (np.ndarray): The RGBA image from the background removal service.
        blend_strength (float): Overall intensity of the enhancement (0.0 to 1.0).
        clahe_limit (float): Clip limit for CLAHE. Lower is more subtle.
        sharpness_amount (float): Amount for unsharp masking.

    Returns:
        np.ndarray: The enhanced BGR image, ready for compositing.
    """
    if img_rgba is None or img_rgba.shape[2] < 4:
        print("⚠️ Warning: Invalid RGBA image passed to enhancer. Skipping.")
        return img_rgba[:, :, :3] if img_rgba is not None else np.zeros((1,1,3), dtype=np.uint8)

    # 1. Separate original BGR and the alpha mask
    original_bgr = img_rgba[:, :, :3]
    alpha_mask = img_rgba[:, :, 3]
    
    # Create a boolean mask for efficient indexing
    is_rug = alpha_mask > 0

    # --- White Balance (LAB Color Correction) ---
    # This remains effective, calculated only on rug pixels.
    lab = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Calculate average 'a' and 'b' from rug pixels only
    avg_a = np.average(a_channel[is_rug])
    avg_b = np.average(b_channel[is_rug])
    
    # Apply correction. The adjustment is based on rug averages.
    l_channel_float = l_channel.astype(np.float32)
    a_channel_float = a_channel.astype(np.float32)
    b_channel_float = b_channel.astype(np.float32)

    a_channel_float[is_rug] -= (avg_a - 128) * (l_channel_float[is_rug] / 255.0) * 1.1
    b_channel_float[is_rug] -= (avg_b - 128) * (l_channel_float[is_rug] / 255.0) * 1.1

    a_channel = np.clip(a_channel_float, 0, 255).astype(np.uint8)
    b_channel = np.clip(b_channel_float, 0, 255).astype(np.uint8)
    
    balanced_lab = cv2.merge([l_channel, a_channel, b_channel])
    balanced_bgr = cv2.cvtColor(balanced_lab, cv2.COLOR_LAB2BGR)

    # --- Gentle Contrast (CLAHE) ---
    # Apply CLAHE to the luminance channel of the new white-balanced image
    lab_for_clahe = cv2.cvtColor(balanced_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_for_clahe)
    clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged_lab = cv2.merge([cl, a, b])
    contrast_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    # --- Controlled Sharpening (Unsharp Mask) ---
    # This method gives a more natural sharpness than a simple kernel.
    blurred = cv2.GaussianBlur(contrast_bgr, (0, 0), 3)
    sharp_bgr = cv2.addWeighted(contrast_bgr, 1.0 + sharpness_amount, blurred, -sharpness_amount, 0)
    
    # --- Final Blend for Subtlety ---
    # This is the key step. We blend the fully processed image with the original.
    # A blend_strength of 0.7 means 70% of the enhancement is applied.
    final_bgr = cv2.addWeighted(sharp_bgr, blend_strength, original_bgr, 1 - blend_strength, 0)

    return final_bgr

# ──────────────────────────────────────────────
# PhotoRoom API Helper
# ──────────────────────────────────────────────
def load_token(token_file: str) -> str:
    path = Path(token_file)
    if not path.exists():
        raise FileNotFoundError(f"Token file not found: {token_file}")
    return path.read_text().strip()


def remove_bg_photoroom(img_bgr: np.ndarray, token: str, retries=2, delay=2.0) -> np.ndarray:
    """Send BGR image to Photoroom API and get RGBA result."""
    if img_bgr is None:
        raise ValueError("Empty image passed to remove_bg_photoroom()")
    api_url = "https://sdk.photoroom.com/v1/segment"
    headers = {"x-api-key": token}

    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Encoding error before API request")

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(api_url, headers=headers, files={"image_file": ("image.png", buf.tobytes(), "image/png")}, timeout=60)
            if resp.status_code == 200:
                data = np.frombuffer(resp.content, np.uint8)
                img_rgba = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
                if img_rgba is None or img_rgba.shape[2] < 4:
                    raise RuntimeError("Photoroom returned invalid RGBA")
                return img_rgba
            else:
                print(f"⚠️ API error {resp.status_code}, retrying ({attempt}/{retries})...")
        except Exception as e:
            print(f"⚠️ Request failed ({attempt}/{retries}): {e}")
        time.sleep(delay)
    raise RuntimeError("Background removal failed after retries")


# ──────────────────────────────────────────────
# Alpha + White Background Composite
# ──────────────────────────────────────────────
def composite_on_white(img_rgba: np.ndarray) -> np.ndarray:
    """Composite RGBA onto white background, preserving edges."""
    if img_rgba.shape[2] == 3:
        return img_rgba
    bgr, alpha = img_rgba[:, :, :3], img_rgba[:, :, 3] / 255.0
    white = np.ones_like(bgr, dtype=np.float32) * 255
    out = (bgr.astype(np.float32) * alpha[..., None] + white * (1 - alpha[..., None]))
    return np.clip(out, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────
# Folder / Batch Runner
# ──────────────────────────────────────────────
def process_folder(src, dst, token_file):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    token = load_token(token_file)

    files = sorted(list(src.glob("*.jpg")) + list(src.glob("*.jpeg")) + list(src.glob("*.png")))
    print(f"Found {len(files)} files in {src}")

    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name} ...", flush=True)
        img = cv2.imread(str(f))
        rgba = remove_bg_photoroom(img, token)
        enhanced_bgr = enhance_rug_vibrance_auto_exposure(rgba)
        enhanced_rgba = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2BGRA)
        enhanced_rgba[:, :, 3] = rgba[:, :, 3] # Carry over the original alpha mask

        # 4. Composite onto white background
        final = composite_on_white(enhanced_rgba)
        
        out_path = dst / f.with_suffix(".png").name # Ensure output is PNG
        cv2.imwrite(str(out_path), final) # No need for PNG_COMPRESSION 0 unless you want uncompressed
        print(f"✅ Saved → {out_path}")


# ──────────────────────────────────────────────
# CLI Entrypoint
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove background and composite over white using Photoroom API.")
    parser.add_argument("--src", required=True, help="Input file or folder")
    parser.add_argument("--out", required=True, help="Output file or folder")
    parser.add_argument("--bg-token-file", required=True, help="Photoroom API token file")
    args = parser.parse_args()

    src = Path(args.src)
    if src.is_dir():
        process_folder(src, args.out, args.bg_token_file)
    else:
        token = load_token(args.bg_token_file)
        img = cv2.imread(str(src))
        rgba = remove_bg_photoroom(img, token)
        enhanced_bgr = enhance_rug_vibrance_auto_exposure(rgba)
        enhanced_rgba = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2BGRA)
        enhanced_rgba[:, :, 3] = rgba[:, :, 3]
        final = composite_on_white(enhanced_rgba)

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), final)
        print(f"✅ Saved → {out_path}")
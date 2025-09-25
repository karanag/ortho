import cv2
import numpy as np
import os
from pathlib import Path

# --- Stage 1: Dynamic Vignette Correction Functions ---

def create_vignette_correction_map_from_image(image: np.ndarray) -> np.ndarray:
    """
    Computes a gain map to correct vignetting by heavily blurring the image itself.
    """
    # Convert to grayscale to work with brightness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use a very large blur kernel to smooth out all texture and detail
    # The kernel size should be odd and large relative to image size
    h, w = gray.shape
    kernel_size = int(min(h, w) / 4) * 2 + 1 # e.g., 1/4 of the smallest dimension
    
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    
    # Calculate the average brightness of the blurred field
    avg_brightness = np.mean(blurred)
    
    # Create the gain map
    gain_map = avg_brightness / (blurred.astype(np.float64) + 1e-6)
    
    # Return a 3-channel map for easy multiplication with a color image
    return cv2.merge([gain_map, gain_map, gain_map])

def apply_correction(image: np.ndarray, gain_map: np.ndarray) -> np.ndarray:
    """Applies a pre-calculated gain map to an image."""
    corrected_image = image.astype(np.float64) * gain_map
    return np.clip(corrected_image, 0, 255).astype(np.uint8)

# --- Stage 2: White Balance Correction Functions ---

def calculate_gray_world_factors(image: np.ndarray) -> tuple[float, float, float]:
    """Calculates BGR scaling factors based on the Gray World assumption."""
    avg_b, avg_g, avg_r = np.mean(image.astype(np.float64), axis=(0, 1))
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    
    scale_b = avg_gray / (avg_b + 1e-6)
    scale_g = avg_gray / (avg_g + 1e-6)
    scale_r = avg_gray / (avg_r + 1e-6)
    
    return (scale_b, scale_g, scale_r)

def apply_wb_factors(image: np.ndarray, factors: tuple[float, float, float]) -> np.ndarray:
    """Applies pre-calculated BGR scaling factors to an image."""
    scale_b, scale_g, scale_r = factors
    corrected_image = image.astype(np.float64)
    corrected_image[:, :, 0] *= scale_b
    corrected_image[:, :, 1] *= scale_g
    corrected_image[:, :, 2] *= scale_r
    return np.clip(corrected_image, 0, 255).astype(np.uint8)

# --- Stage 3: Histogram Matching Function ---

def match_histogram(source_image: np.ndarray, reference_image: np.ndarray) -> np.ndarray:
    """
    Matches the histogram of a source image to a reference image.
    Works on the V channel (Value/Brightness) in HSV color space to avoid color shifts.
    """
    source_hsv = cv2.cvtColor(source_image, cv2.COLOR_BGR2HSV)
    reference_hsv = cv2.cvtColor(reference_image, cv2.COLOR_BGR2HSV)
    
    hist_src, _ = np.histogram(source_hsv[:, :, 2].flatten(), 256, [0, 256])
    hist_ref, _ = np.histogram(reference_hsv[:, :, 2].flatten(), 256, [0, 256])
    
    cdf_src = hist_src.cumsum()
    cdf_ref = hist_ref.cumsum()
    
    cdf_src_normalized = cdf_src * hist_ref.sum() / (cdf_src.sum() + 1e-6)
    
    lookup_table = np.zeros(256, dtype='uint8')
    g = 0
    for f in range(256):
        while g < 256 and cdf_src_normalized[f] > cdf_ref[g]:
            g += 1
        lookup_table[f] = g
    
    matched_v = cv2.LUT(source_hsv[:, :, 2], lookup_table)
    matched_hsv = cv2.merge([source_hsv[:, :, 0], source_hsv[:, :, 1], matched_v])
    
    return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2BGR)

# --- Main Processing Pipeline ---

def preprocess_image_pipeline(input_dir: str, output_dir: str, do_vignette_correction: bool = True, do_histogram_matching: bool = True):
    """
    Runs the full pre-processing pipeline on a folder of images.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_files = sorted([f for f in in_path.iterdir() if f.suffix.lower() in image_extensions])

    if not image_files:
        print(f"No images found in '{input_dir}'")
        return
        
    # --- Step 1: Perform per-image vignette correction on all images first ---
    vignette_corrected_images = {}
    if do_vignette_correction:
        print("--- Stage 1: Performing Vignette Correction ---")
        for i, image_file in enumerate(image_files):
            print(f"  [{i+1}/{len(image_files)}] Analyzing {image_file.name} for vignetting...")
            img = cv2.imread(str(image_file))
            if img is not None:
                gain_map = create_vignette_correction_map_from_image(img)
                vignette_corrected_images[image_file.name] = apply_correction(img, gain_map)
    else:
        print("--- Stage 1: Vignette Correction SKIPPED ---")
        # If skipped, just load the original images
        for image_file in image_files:
            vignette_corrected_images[image_file.name] = cv2.imread(str(image_file))

    # --- Step 2: Analyze ONE reference image to get WB and Histogram targets ---
    print("\n--- Stage 2 & 3: Applying Consistent WB and Histogram Matching ---")
    reference_image_name = image_files[0].name
    print(f"Using '{reference_image_name}' as the reference for WB and Histogram.")
    
    # Get the already vignette-corrected version of our reference image
    ref_img_for_analysis = vignette_corrected_images[reference_image_name]
    
    # Calculate WB factors from it
    wb_factors = calculate_gray_world_factors(ref_img_for_analysis)
    print(f"Calculated WB Correction Factors (B, G, R): {wb_factors}")
    
    # Create the final "golden reference" for histogram matching
    golden_reference_image = apply_wb_factors(ref_img_for_analysis, wb_factors)

    # --- Step 3: Process All Images Consistently ---
    for i, image_file in enumerate(image_files):
        print(f"  [{i+1}/{len(image_files)}] Final processing for {image_file.name}...")
        
        # Get the vignette-corrected image from our dictionary
        current_img = vignette_corrected_images[image_file.name]
        
        # Apply Consistent White Balance
        current_img = apply_wb_factors(current_img, wb_factors)
        
        # Apply Histogram Matching
        if do_histogram_matching:
            current_img = match_histogram(current_img, golden_reference_image)
            
        # Save the final processed image
        save_path = out_path / image_file.name
        cv2.imwrite(str(save_path), current_img)

    print(f"\nProcessing complete. Corrected images saved in '{output_dir}'")

if __name__ == '__main__':
    # --- Configuration ---
    INPUT_FOLDER = 'images/3'
    OUTPUT_FOLDER = 'images/3/a'
    
    # Set to True to enable the automated, per-image vignette correction
    DO_VIGNETTE_CORRECTION = True
    
    # Set to True to enable histogram matching, False to disable.
    DO_HISTOGRAM_MATCHING = False
    # -------------------
    
    preprocess_image_pipeline(INPUT_FOLDER, OUTPUT_FOLDER, DO_VIGNETTE_CORRECTION, DO_HISTOGRAM_MATCHING)
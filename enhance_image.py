import cv2
import numpy as np

# ──────────────────────────────────────────────
# Subtle & Natural Image Enhancement
# ──────────────────────────────────────────────
def enhance_rug_subtle(
    img_rgba: np.ndarray,
    blend_strength: float = 0.75,
    clahe_limit: float = 1.2,
    sharpness_amount: float = 0.15
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

def enhance_rug_image_with_masking(input_path, output_path, bg_color_low=(200, 200, 200), bg_color_high=(255, 255, 255)):
    """
    Applies smart enhancements to a rug image after masking out a specific background color range.

    Args:
        input_path (str): The file path for the input image.
        output_path (str): The file path to save the enhanced image.
        bg_color_low (tuple): Lower BGR bound for background color. Default for white.
        bg_color_high (tuple): Upper BGR bound for background color. Default for white.
    """
    # 1. Read the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return

    original_shape = img.shape
    
    # --- 2. Create a mask to separate rug from background ---
    # We assume the background is within the specified BGR range (e.g., white)
    # The mask will be 255 for background pixels, 0 for rug pixels.
    mask_bg = cv2.inRange(img, bg_color_low, bg_color_high)
    
    # Invert the mask to get the rug itself (255 for rug, 0 for background)
    mask_rug = cv2.bitwise_not(mask_bg)

    # Optional: Apply some morphological operations to clean up the mask
    # This helps in removing small specks of noise or closing small gaps in the mask
    kernel = np.ones((3,3), np.uint8)
    mask_rug = cv2.morphologyEx(mask_rug, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_rug = cv2.morphologyEx(mask_rug, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Create an empty image for the enhanced rug, filling background with black for now
    isolated_rug = cv2.bitwise_and(img, img, mask=mask_rug)

    # Convert the isolated rug to LAB color space for enhancements
    # We will only apply these to the rug area, ignoring the black background introduced by masking
    isolated_rug_lab = cv2.cvtColor(isolated_rug, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image (L, A, B channels)
    l, a, b = cv2.split(isolated_rug_lab)

    # --- 3. Smart White Balance (Gray World Algorithm) - ONLY ON RUG PIXELS ---
    # We need to compute averages only from the rug pixels.
    # Convert mask_rug to a boolean array to use for indexing
    rug_pixels_mask_l = mask_rug.astype(bool)
    rug_pixels_mask_a = mask_rug.astype(bool)
    rug_pixels_mask_b = mask_rug.astype(bool)

    # Calculate average 'a' and 'b' values only for rug pixels
    avg_a_rug = np.average(a[rug_pixels_mask_a]) if np.any(rug_pixels_mask_a) else 128
    avg_b_rug = np.average(b[rug_pixels_mask_b]) if np.any(rug_pixels_mask_b) else 128

    # Apply scaling for white balance *only to rug pixels*
    # We apply the white balance transformation to the full channels, but the averages are based on rug_pixels
    # This transformation affects all pixels, but the adjustment is calculated to balance the rug
    # Note: L channel scaling in the original gray world implementation is slightly more complex if doing fully masked,
    # for simplicity here we let CLAHE handle luminance balance.
    
    # We create temporary channels for adjustment to avoid modifying 'a' and 'b' directly before CLAHE
    temp_a = a.copy().astype(np.float32)
    temp_b = b.copy().astype(np.float32)

    # Ensure to only modify pixels where the mask is active
    temp_a[rug_pixels_mask_a] = temp_a[rug_pixels_mask_a] - ((avg_a_rug - 128) * (l[rug_pixels_mask_l] / 255.0) * 1.1)
    temp_b[rug_pixels_mask_b] = temp_b[rug_pixels_mask_b] - ((avg_b_rug - 128) * (l[rug_pixels_mask_l] / 255.0) * 1.1)

    # Clip values to valid LAB range [0, 255]
    temp_a = np.clip(temp_a, 0, 255).astype(np.uint8)
    temp_b = np.clip(temp_b, 0, 255).astype(np.uint8)

    # --- 4. Smart Contrast & Exposure (CLAHE) - ONLY ON RUG PIXELS (via L channel) ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl_rug = clahe.apply(l) # Apply CLAHE to the L-channel of the isolated rug
    
    # Merge the enhanced L channel with the (now white-balanced) A and B channels
    merged_lab_rug = cv2.merge([cl_rug, temp_a, temp_b])
    enhanced_rug_color_balanced_bgr = cv2.cvtColor(merged_lab_rug, cv2.COLOR_LAB2BGR)

    # Re-mask the enhanced rug to ensure background is clean before sharpening
    enhanced_rug_pixels_only = cv2.bitwise_and(enhanced_rug_color_balanced_bgr, enhanced_rug_color_balanced_bgr, mask=mask_rug)

    # --- 5. Sharpening - ONLY ON RUG PIXELS ---
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    
    # Apply sharpening only to the rug pixels. We do this by applying to the isolated part
    # and then blending back.
    sharpened_rug_part = cv2.filter2D(src=enhanced_rug_pixels_only, ddepth=-1, kernel=kernel)

    # Create an empty canvas with the original background color (e.g., white)
    final_output_img = np.full(original_shape, (255, 255, 255), dtype=np.uint8) # White background

    # Combine the sharpened rug with the new background
    # We only copy pixels where mask_rug is active
    final_output_img = np.where(mask_rug[:, :, None] == 255, sharpened_rug_part, final_output_img)


    # 6. Save the final processed image
    cv2.imwrite(output_path, final_output_img)
    print(f"Successfully enhanced image and saved to {output_path}")


# --- How to use the script ---
if __name__ == '__main__':
    # Define the input image path (the one from your pipeline)
    input_image_path = 'cleaned_seedha.png' # Replace with your actual image file
    
    # Define where you want to save the final result
    output_image_path = 'enhanced_rug_masked_final.png'

    # Define the background color range (BGR). For pure white, this range is suitable.
    # Adjust if your "white" background has slight variations or tints.
    # For example, a light grey background might be (200,200,200), (230,230,230)
    # Be careful not to include actual rug colors in this range.
    bg_low = np.array([230, 230, 230], dtype=np.uint8) # Lower BGR bound for "white"
    bg_high = np.array([255, 255, 255], dtype=np.uint8) # Upper BGR bound for "white"
    
    # Run the enhancement function
    enhance_rug_image_with_masking(input_image_path, output_image_path, bg_color_low=bg_low, bg_color_high=bg_high)
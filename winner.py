import cv2
import numpy as np
import glob
import os

warp_dir = "./out"
scale = 0.3   # set to 1.0 for full resolution

# Load warps + masks
warp_files = sorted(glob.glob(os.path.join(warp_dir, "warp_*.jpeg.png")))
mask_files = sorted(glob.glob(os.path.join(warp_dir, "mask_*.jpeg.png")))

images = [cv2.imread(f) for f in warp_files]  # keep 8-bit for blender
masks  = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in mask_files]

print(f"Loaded {len(images)} images")

# Scale down images + masks for faster preview
if scale != 1.0:
    resized_images = []
    resized_masks = []
    for img, m in zip(images, masks):
        h, w = img.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        resized_images.append(cv2.resize(img, (new_w, new_h)))
        resized_masks.append(cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST))
    images, masks = resized_images, resized_masks
    print(f"Resized to {images[0].shape[1]}x{images[0].shape[0]}")

# Convert masks to binary 8-bit
masks = [cv2.threshold(m, 1, 255, cv2.THRESH_BINARY)[1] for m in masks]

# All images are already aligned at (0,0)
corners = [(0, 0)] * len(images)

# --- Seam finder needs float32 ---
images_f32 = [img.astype(np.float32) for img in images]
seam_finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
masks = seam_finder.find(images_f32, corners, masks)

# --- Multiband blender ---
blender = cv2.detail_MultiBandBlender()
blender.setNumBands(3)  # reduce bands for speed
dst_sz = (images[0].shape[1], images[0].shape[0])
blender.prepare((0, 0, dst_sz[0], dst_sz[1]))

for i, (img, mask) in enumerate(zip(images, masks)):
    # Convert UMat → numpy if needed
    if isinstance(mask, cv2.UMat):
        mask = mask.get()
    mask = mask.astype(np.uint8)

    print(f"Feeding image {i+1}/{len(images)} into blender...")
    blender.feed(img, mask, (0, 0))
    cv2.imwrite(f"debug_mask_{i}.png", mask)  # save seam mask

print("Blending...")
result, result_mask = blender.blend(None, None)
result = np.clip(result, 0, 255).astype(np.uint8)

out_file = "fused_seam_preview.png" if scale != 1.0 else "fused_seam.png"
cv2.imwrite(out_file, result)
print("Saved", out_file)

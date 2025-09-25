import cv2
import numpy as np
import glob
import os

warp_dir = "./out"
scale = 0.3   # for seam finding (fast). Set to 1.0 to skip low-res phase

# --- Load warps + masks ---
warp_files = sorted(glob.glob(os.path.join(warp_dir, "warp_*.jpeg.png")))
mask_files = sorted(glob.glob(os.path.join(warp_dir, "mask_*.jpeg.png")))

images_full = [cv2.imread(f) for f in warp_files]   # 8-bit full res
masks_full  = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in mask_files]

print(f"Loaded {len(images_full)} full-res images")

# --- Step 1: Low-res seam finding ---
if scale != 1.0:
    images_small = []
    masks_small = []
    for img, m in zip(images_full, masks_full):
        h, w = img.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        images_small.append(cv2.resize(img.astype(np.float32), (new_w, new_h)))
        masks_small.append(cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST))

    # binarize masks
    masks_small = [cv2.threshold(m, 1, 255, cv2.THRESH_BINARY)[1] for m in masks_small]

    # corners (aligned)
    corners = [(0, 0)] * len(images_small)

    seam_finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
    seam_masks_small = seam_finder.find(images_small, corners, masks_small)

    # --- Step 2: Upscale seam masks ---
    seam_masks_full = []
    for i, sm in enumerate(seam_masks_small):
        if isinstance(sm, cv2.UMat):
            sm = sm.get()
        h, w = images_full[i].shape[:2]
        seam_masks_full.append(cv2.resize(sm, (w, h), interpolation=cv2.INTER_NEAREST))
        cv2.imwrite(f"debug_seam_mask_{i}.png", seam_masks_full[-1])

else:
    # If scale=1.0, run seam finder directly at full res
    images_f32 = [img.astype(np.float32) for img in images_full]
    masks_bin = [cv2.threshold(m, 1, 255, cv2.THRESH_BINARY)[1] for m in masks_full]
    corners = [(0, 0)] * len(images_full)
    seam_finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
    seam_masks_full = seam_finder.find(images_f32, corners, masks_bin)

print("Seam masks ready.")

# --- Step 3: Full-res blending ---
blender = cv2.detail_MultiBandBlender()
blender.setNumBands(3)  # tune
h, w = images_full[0].shape[:2]
blender.prepare((0, 0, w, h))

for i, (img, mask) in enumerate(zip(images_full, seam_masks_full)):
    if isinstance(mask, cv2.UMat):
        mask = mask.get()
    mask = mask.astype(np.uint8)
    print(f"Feeding image {i+1}/{len(images_full)}...")
    blender.feed(img, mask, (0, 0))

print("Blending...")
result, _ = blender.blend(None, None)
result = np.clip(result, 0, 255).astype(np.uint8)

out_file = "fused_two_phase.png"
cv2.imwrite(out_file, result)
print("Saved", out_file)

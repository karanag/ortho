import cv2, numpy as np
from pathlib import Path

def match_color_lab(ref_img, target_img, clip_limit=2.0, grid_size=(8,8)):
    ref_lab = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype(np.float32)

    ref_mean, ref_std = cv2.meanStdDev(ref_lab)
    tgt_mean, tgt_std = cv2.meanStdDev(tgt_lab)

    norm = (tgt_lab - tgt_mean.T) * (ref_std.T / (tgt_std.T + 1e-6)) + ref_mean.T
    norm = np.clip(norm, 0, 255).astype(np.uint8)

    lab_split = list(cv2.split(norm))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    lab_split[0] = clahe.apply(lab_split[0])
    norm = cv2.merge(lab_split)

    return cv2.cvtColor(norm, cv2.COLOR_LAB2BGR)

def normalize_pairwise(img_dir, out_dir="balanced_pairwise"):
    img_dir, out_dir = Path(img_dir), Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    paths = sorted(img_dir.glob("warp_*.png"))
    imgs = [cv2.imread(str(p)) for p in paths]

    corrected = [imgs[0]]  # first image as base reference
    cv2.imwrite(str(out_dir / paths[0].name), imgs[0])
    print(f"✓ Base reference: {paths[0].name}")

    for i in range(1, len(imgs)):
        ref = corrected[-1]
        target = imgs[i]
        corr = match_color_lab(ref, target)
        corrected.append(corr)
        cv2.imwrite(str(out_dir / paths[i].name), corr)
        print(f"→ Matched {paths[i].name} to {paths[i-1].name}")

    print("\n✅ All done. Balanced tiles saved in:", out_dir)

if __name__ == "__main__":
    normalize_pairwise("input")

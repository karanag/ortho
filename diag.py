import cv2, numpy as np
from pathlib import Path

def analyze_color_stats(img_dir):
    paths = sorted(Path(img_dir).glob("*.png"))
    print(f"Analyzing {len(paths)} images in {img_dir}...\n")

    for p in paths:
        img = cv2.imread(str(p))
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        mean, std = cv2.meanStdDev(lab)
        print(f"{p.name:20s} | L_mean={mean[0,0]:.1f} | a={mean[1,0]:.1f} | b={mean[2,0]:.1f} | L_std={std[0,0]:.1f}")

from skimage.metrics import structural_similarity as ssim

def ssim_between_pairs(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

if __name__ == "__main__":
    analyze_color_stats("harmonized_auto/03_final")  # folder containing balanced images


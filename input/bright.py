# bright_auto.py
import cv2
import numpy as np
from pathlib import Path
import argparse

# ---------------------------- Utils ----------------------------

def split_alpha(img):
    """Return (bgr, alpha or None)."""
    if img is None:
        return None, None
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR), img[:, :, 3]
    return img, None

def merge_alpha(bgr, alpha):
    if alpha is None:
        return bgr
    out = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    out[:, :, 3] = alpha
    return out

def lab_with_alpha(img):
    """Return (lab(float32), alpha or None)."""
    bgr, a = split_alpha(img)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab, a

def mask_from_alpha(alpha, shape_hw):
    if alpha is None:
        return np.ones(shape_hw, dtype=bool)
    return alpha > 0

def l_stats(l_chan, mask_bool):
    vals = l_chan[mask_bool].astype(np.float32)
    if vals.size == 0:
        vals = l_chan.reshape(-1).astype(np.float32)
    mean = float(vals.mean())
    std  = float(vals.std())
    p03  = float(np.percentile(vals, 3.0))
    p97  = float(np.percentile(vals, 97.0))
    return mean, std, p03, p97

def apply_L_scale(lab, scale):
    L = lab[:, :, 0] * scale
    np.clip(L, 0, 255, out=L)
    lab[:, :, 0] = L
    return lab

def mild_local_contrast_if_flat(bgr, alpha, flat_std_thresh=22.0, clip_limit=0.8, grid=(12, 12)):
    """Only add mild local contrast if L std is too low (avoids 'mechanical' CLAHE look)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    mask = mask_from_alpha(alpha, L.shape)
    mean, std, _, _ = l_stats(L, mask)
    if std < flat_std_thresh:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
        # Apply on all, but it’s mild; masking not necessary post-matting
        lab[:, :, 0] = clahe.apply(L)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

def low_freq_match_L(prev_bgr, curr_bgr, alpha_curr, beta=0.25, blur_ks=61, blur_sigma=0):
    """
    Neighbor-aware low-frequency smoothing:
    Add the low-frequency difference from prev to curr:
        L_curr += beta * (blur(prev) - blur(curr))
    Preserves texture; avoids over-smoothing details.
    """
    prev_lab = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    curr_lab = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    Lp = prev_lab[:, :, 0]
    Lc = curr_lab[:, :, 0]

    # big-kernel Gaussian for low-freq illumination
    Lp_blur = cv2.GaussianBlur(Lp, (blur_ks, blur_ks), blur_sigma)
    Lc_blur = cv2.GaussianBlur(Lc, (blur_ks, blur_ks), blur_sigma)

    L_new = Lc + beta * (Lp_blur - Lc_blur)
    np.clip(L_new, 0, 255, out=L_new)
    curr_lab[:, :, 0] = L_new
    out_bgr = cv2.cvtColor(curr_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return merge_alpha(out_bgr, alpha_curr)

# ------------------------ Core algorithms ----------------------

def pairwise_match_lab(ref_img, tgt_img, clahe_clip=0.0):
    """
    Mask-aware (via alpha) LAB mean/std match from tgt -> ref.
    Mild optional CLAHE on L (disabled by default to avoid 'mechanical' look).
    """
    ref_bgr, ref_a = split_alpha(ref_img)
    tgt_bgr, tgt_a = split_alpha(tgt_img)

    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    ref_mask = mask_from_alpha(ref_a, ref_lab.shape[:2])
    tgt_mask = mask_from_alpha(tgt_a, tgt_lab.shape[:2])

    # mean/std over masked pixels
    ref_mean, ref_std = cv2.meanStdDev(ref_lab, mask=ref_mask.astype(np.uint8) * 255)
    tgt_mean, tgt_std = cv2.meanStdDev(tgt_lab, mask=tgt_mask.astype(np.uint8) * 255)

    eps = 1e-6
    norm = (tgt_lab - tgt_mean.T) * (ref_std.T / (tgt_std.T + eps)) + ref_mean.T
    norm = np.clip(norm, 0, 255).astype(np.uint8)

    if clahe_clip > 0:
        l, a, b = cv2.split(norm)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(12, 12))
        l = clahe.apply(l)
        norm = cv2.merge([l, a, b])

    out_bgr = cv2.cvtColor(norm, cv2.COLOR_LAB2BGR)
    return merge_alpha(out_bgr, tgt_a)

def adaptive_gain(scale_desired, meanL, p03, p97,
                  min_gain=0.85, max_gain=1.15,
                  max_highlight=240.0, min_shadow=4.0):
    """
    Clamp scale to protect highlights & shadows, and keep a natural range.
    """
    # highlight safety if brightening
    hi_cap = max_gain
    if p97 > 1.0:
        hi_cap = min(hi_cap, max_highlight / p97)

    # shadow safety if darkening
    lo_cap = min_gain
    if p03 > 1.0:
        lo_floor = min_shadow / p03
        lo_cap = max(lo_cap, lo_floor)

    scale = float(np.clip(scale_desired, lo_cap, hi_cap))
    clamped = (scale != scale_desired)
    return scale, clamped, lo_cap, hi_cap

def progressive_targets(means, anchor_strength=0.6):
    """
    Build a smooth target curve:
      - move endpoints toward global median by 'anchor_strength'
      - linearly interpolate targets for in-between frames.
    """
    N = len(means)
    if N == 1:
        return [means[0]]

    med = float(np.median(means))
    t0 = means[0] + anchor_strength * (med - means[0])
    tN = means[-1] + anchor_strength * (med - means[-1])

    targets = []
    for i in range(N):
        t = t0 + (tN - t0) * (i / (N - 1))
        targets.append(float(t))
    return targets, med, t0, tN

# ------------------------ Pipeline -----------------------------

def process_folder(inp_dir, out_dir,
                   drift_slope_thresh=0.5,
                   neighbor_beta=0.25,
                   chroma_cap=3.0,
                   enable_pairwise_clahe=False):
    inp = Path(inp_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = sorted([p for p in inp.glob("*.png")])
    if not paths:
        print("No PNGs found in:", inp_dir)
        return

    imgs = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in paths]

    # --- Step 0: Enhance the first frame slightly so it isn't dull
    # (very mild local contrast if needed)
    base_bgr, base_a = split_alpha(imgs[0])
    base_bgr = mild_local_contrast_if_flat(base_bgr, base_a, flat_std_thresh=22.0, clip_limit=0.7, grid=(12, 12))
    imgs[0] = merge_alpha(base_bgr, base_a)
    cv2.imwrite(str(out / paths[0].name), imgs[0])
    print(f"✓ Base reference enhanced & saved: {paths[0].name}")

    # --- Step 1: Pairwise LAB mean/std match (mask-aware)
    corrected = [imgs[0]]
    for i in range(1, len(imgs)):
        ref, tgt = corrected[-1], imgs[i]
        corr = pairwise_match_lab(ref, tgt, clahe_clip=(0.6 if enable_pairwise_clahe else 0.0))
        corrected.append(corr)
        cv2.imwrite(str(out / paths[i].name), corr)
        print(f"→ Matched {paths[i].name} to {paths[i-1].name}")

    # --- Step 2: Compute luminance metrics
    print("\n🔧 Computing initial luminance metrics...")
    means, stds, p3s, p97s = [], [], [], []
    for im in corrected:
        lab, a = lab_with_alpha(im)
        mask = mask_from_alpha(a, lab.shape[:2])
        m, s, p3, p97 = l_stats(lab[:, :, 0], mask)
        means.append(m); stds.append(s); p3s.append(p3); p97s.append(p97)
    print("   Initial L_means:", [round(x, 1) for x in means])

    # --- Step 3: Flatten linear brightness drift (if present)
    x = np.arange(len(means))
    slope, intercept = np.polyfit(x, means, 1)
    if abs(slope) >= drift_slope_thresh:
        print("📈 Flattening linear brightness drift first...")
        predicted = slope * x + intercept
        drift = predicted - np.mean(predicted)
        drift_scale = 1 - (drift / np.mean(means))
        print(f"   Fit line: L = {slope:.3f}*i + {intercept:.3f}")
        print(f"   Drift scale factors:", [round(s, 3) for s in drift_scale])

        # apply drift compensation on L
        for i, (im, ds) in enumerate(zip(corrected, drift_scale)):
            lab, a = lab_with_alpha(im)
            lab = apply_L_scale(lab, ds)
            corrected[i] = merge_alpha(cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR), a)
    else:
        print("📉 Drift is negligible; skipping drift flattening.")

    # recompute means after flatten
    means = []
    for im in corrected:
        lab, a = lab_with_alpha(im)
        mask = mask_from_alpha(a, lab.shape[:2])
        m, *_ = l_stats(lab[:, :, 0], mask)
        means.append(m)

    # --- Step 4: Progressive target equalization with adaptive limiter
    targets, med, t0, tN = progressive_targets(means, anchor_strength=0.6)
    print("\n🔧 Progressive luminance targeting...")
    print(f"   Global median = {med:.2f} | Anchors → first: {t0:.2f}, last: {tN:.2f}")

    for i, (im, target) in enumerate(zip(corrected, targets)):
        lab, a = lab_with_alpha(im)
        mask = mask_from_alpha(a, lab.shape[:2])
        meanL, _, p03, p97 = l_stats(lab[:, :, 0], mask)

        desired = target / (meanL + 1e-6)
        scale, clamped, lo_cap, hi_cap = adaptive_gain(
            desired, meanL, p03, p97,
            min_gain=0.85, max_gain=1.15,
            max_highlight=240.0, min_shadow=4.0
        )
        if clamped:
            print(f"   [{i+1}] {paths[i].name}: desired×{desired:.3f} → clamped to ×{scale:.3f} (bounds {lo_cap:.3f}–{hi_cap:.3f})")
        else:
            print(f"   [{i+1}] {paths[i].name}: scale ×{scale:.3f}")

        lab = apply_L_scale(lab, scale)
        corrected[i] = merge_alpha(cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR), a)
        cv2.imwrite(str(out / paths[i].name), corrected[i])

    # --- Step 5: Neighbor-aware low-frequency alignment (preserve details)
    print("🤝 Applying neighbor-aware low-freq smoothing...")
    for i in range(1, len(corrected)):
        prev_bgr, _ = split_alpha(corrected[i - 1])
        curr_bgr, curr_a = split_alpha(corrected[i])
        blended = low_freq_match_L(prev_bgr, curr_bgr, curr_a, beta=neighbor_beta, blur_ks=61, blur_sigma=0)
        corrected[i] = blended
        cv2.imwrite(str(out / paths[i].name), blended)

    # --- Step 6: Gentle global chroma alignment (±3 cap)
    print("🎨 Aligning global chroma (a/b channels, gently)...")
    a_means, b_means = [], []
    for im in corrected:
        bgr, _ = split_alpha(im)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        a_means.append(float(lab[:, :, 1].mean()))
        b_means.append(float(lab[:, :, 2].mean()))
    target_a, target_b = np.median(a_means), np.median(b_means)
    for i, im in enumerate(corrected):
        bgr, a = split_alpha(im)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        da = np.clip(target_a - lab[:, :, 1].mean(), -chroma_cap, chroma_cap)
        db = np.clip(target_b - lab[:, :, 2].mean(), -chroma_cap, chroma_cap)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + da, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + db, 0, 255)
        bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        corrected[i] = merge_alpha(bgr, a)
        cv2.imwrite(str(out / paths[i].name), corrected[i])

    # --- Step 7: Mild local contrast only if needed (final polish)
    print("✨ Final mild local-contrast polish (only if flat)...")
    for i, im in enumerate(corrected):
        bgr, a = split_alpha(im)
        bgr = mild_local_contrast_if_flat(bgr, a, flat_std_thresh=22.0, clip_limit=0.7, grid=(12, 12))
        corrected[i] = merge_alpha(bgr, a)
        cv2.imwrite(str(out / paths[i].name), corrected[i])

    print("\n✅ Final polished, color-consistent images saved to:", out_dir)

# ------------------------ CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Natural exposure/color normalization (auto, safe limits).")
    ap.add_argument("--inp", type=str, default="no_bg_photoroom", help="Input folder of PNGs")
    ap.add_argument("--out", type=str, default="balanced_auto", help="Output folder")
    ap.add_argument("--drift_thresh", type=float, default=0.5, help="Min |slope| to flatten drift")
    ap.add_argument("--neighbor_beta", type=float, default=0.25, help="Strength of low-freq neighbor smoothing")
    ap.add_argument("--chroma_cap", type=float, default=3.0, help="Max a/b offset per image")
    ap.add_argument("--pairwise_clahe", action="store_true", help="Enable mild CLAHE during pairwise step")
    args = ap.parse_args()

    process_folder(
        inp_dir=args.inp,
        out_dir=args.out,
        drift_slope_thresh=args.drift_thresh,
        neighbor_beta=args.neighbor_beta,
        chroma_cap=args.chroma_cap,
        enable_pairwise_clahe=args.pairwise_clahe
    )

if __name__ == "__main__":
    main()

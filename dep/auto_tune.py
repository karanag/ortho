import os
import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ---------------------------- Utils ----------------------------

def split_alpha(img):
    """Return (bgr, alpha or None)."""
    if img is None:
        return None, None
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, :3], img[:, :, 3]
    return img, None

def ensure_bgra(img):
    """Ensure 4 channels (BGRA)."""
    if img.ndim == 3 and img.shape[2] == 4:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        h, w = img.shape[:2]
        a = np.full((h, w), 255, dtype=np.uint8)
        return np.dstack([img, a])
    raise ValueError("Unsupported image shape")

def merge_alpha(bgr, alpha):
    if alpha is None:
        return bgr
    out = np.dstack([bgr, alpha])
    return out

def lab_with_alpha(img):
    """Return (lab(float32), alpha or None)."""
    bgr, a = split_alpha(img)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab, a

def mask_from_alpha(alpha, shape_hw, threshold=5):
    if alpha is None:
        return np.ones(shape_hw, dtype=bool)
    return alpha.astype(np.uint8) > threshold

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
        lab[:, :, 0] = clahe.apply(L)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

def low_freq_match_L(prev_bgr, curr_bgr, alpha_curr, beta=0.25, blur_ks=61, blur_sigma=0):
    """
    Neighbor-aware low-frequency smoothing:
      L_curr += beta * (blur(prev) - blur(curr))
    Preserves texture; avoids over-smoothing details.
    """
    prev_lab = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    curr_lab = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    Lp = prev_lab[:, :, 0]
    Lc = curr_lab[:, :, 0]

    # big-kernel Gaussian for low-freq illumination
    blur_ks = (blur_ks // 2 * 2) + 1  # ensure odd
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
    ref_mask_u8 = (ref_mask.astype(np.uint8) * 255)
    tgt_mask_u8 = (tgt_mask.astype(np.uint8) * 255)
    ref_mean, ref_std = cv2.meanStdDev(ref_lab, mask=ref_mask_u8)
    tgt_mean, tgt_std = cv2.meanStdDev(tgt_lab, mask=tgt_mask_u8)

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
        return [means[0]], float(np.median(means)), means[0], means[0]

    med = float(np.median(means))
    t0 = means[0] + anchor_strength * (med - means[0])
    tN = means[-1] + anchor_strength * (med - means[-1])

    targets = []
    for i in range(N):
        t = t0 + (tN - t0) * (i / (N - 1))
        targets.append(float(t))
    return targets, med, t0, tN

def soft_mask(mask, blur=21):
    """Feather mask edges to avoid halos; returns [0..1] float mask."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if blur > 0:
        k = (blur // 2 * 2) + 1  # ensure odd
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    mask = mask.astype(np.float32) / (mask.max() + 1e-6)
    return np.clip(mask, 0, 1)

def color_transfer_lab_masked(source, target, feather=21):
    """Transparency-safe LAB color transfer with edge feathering on the target mask."""
    # --- Separate alpha and build masks ---
    if source.shape[2] == 4:
        src_bgr, src_mask = source[..., :3], source[..., 3]
    else:
        src_bgr = source
        src_mask = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    if target.shape[2] == 4:
        tgt_bgr, tgt_mask = target[..., :3], target[..., 3]
    else:
        tgt_bgr = target
        tgt_mask = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2GRAY)

    src_mask = (src_mask > 5).astype(np.uint8) * 255
    tgt_mask = (tgt_mask > 5).astype(np.uint8) * 255

    # --- LAB conversion ---
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # --- Stats only on visible areas ---
    ref_mask_u8 = src_mask
    tar_mask_u8 = tgt_mask
    s_mean, s_std = cv2.meanStdDev(src_lab, mask=ref_mask_u8)
    t_mean, t_std = cv2.meanStdDev(tgt_lab, mask=tar_mask_u8)

    # --- Apply transfer ---
    eps = 1e-6
    norm = (tgt_lab - t_mean.T) * (s_std.T / (t_std.T + eps)) + s_mean.T
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(norm, cv2.COLOR_LAB2BGR)

    # --- Blend result only inside soft mask ---
    alpha = soft_mask(tgt_mask, blur=feather)
    out = (out_bgr * alpha[..., None] + tgt_bgr * (1 - alpha[..., None])).astype(np.uint8)

    # --- Re-attach transparency if present ---
    if target.shape[2] == 4:
        out = np.dstack([out, tgt_mask])
    return out

# ------------------------ Stage 1: Brightness pipeline -----------------------------

def stage1_bright_normalize(imgs, paths, out_dir=None,
                            drift_slope_thresh=0.5,
                            neighbor_beta=0.25,
                            chroma_cap=3.0,
                            enable_pairwise_clahe=False):
    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    # Step 0: Enhance the first frame slightly so it isn't dull
    base_bgr, base_a = split_alpha(imgs[0])
    base_bgr = mild_local_contrast_if_flat(base_bgr, base_a, flat_std_thresh=22.0, clip_limit=0.7, grid=(12, 12))
    imgs[0] = merge_alpha(base_bgr, base_a)
    if out_path is not None:
        cv2.imwrite(str(out_path / paths[0].name), imgs[0])
    print(f"✓ [S1] Base reference enhanced: {paths[0].name}")

    # Step 1: Pairwise LAB mean/std match (mask-aware)
    corrected = [imgs[0]]
    for i in range(1, len(imgs)):
        ref, tgt = corrected[-1], imgs[i]
        corr = pairwise_match_lab(ref, tgt, clahe_clip=(0.6 if enable_pairwise_clahe else 0.0))
        corrected.append(corr)
        if out_path is not None:
            cv2.imwrite(str(out_path / paths[i].name), corr)
        print(f"→ [S1] Matched {paths[i].name} to {paths[i-1].name}")

    # Step 2: Compute luminance metrics
    print("\n🔧 [S1] Computing initial luminance metrics...")
    means, stds, p3s, p97s = [], [], [], []
    for im in corrected:
        lab, a = lab_with_alpha(im)
        mask = mask_from_alpha(a, lab.shape[:2])
        m, s, p3, p97 = l_stats(lab[:, :, 0], mask)
        means.append(m); stds.append(s); p3s.append(p3); p97s.append(p97)
    print("   [S1] Initial L_means:", [round(x, 1) for x in means])

    # Step 3: Flatten linear brightness drift (if present)
    x = np.arange(len(means))
    slope, intercept = np.polyfit(x, means, 1)
    if abs(slope) >= drift_slope_thresh:
        print("📈 [S1] Flattening linear brightness drift...")
        predicted = slope * x + intercept
        drift = predicted - np.mean(predicted)
        drift_scale = 1 - (drift / np.mean(means))
        print(f"   [S1] Fit line: L = {slope:.3f}*i + {intercept:.3f}")
        print(f"   [S1] Drift scale factors:", [round(s, 3) for s in drift_scale])

        # apply drift compensation on L
        for i, (im, ds) in enumerate(zip(corrected, drift_scale)):
            lab, a = lab_with_alpha(im)
            lab = apply_L_scale(lab, ds)
            corrected[i] = merge_alpha(cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR), a)
            if out_path is not None:
                cv2.imwrite(str(out_path / paths[i].name), corrected[i])
    else:
        print("📉 [S1] Drift negligible; skipping drift flattening.")

    # recompute means after flatten
    means = []
    for im in corrected:
        lab, a = lab_with_alpha(im)
        mask = mask_from_alpha(a, lab.shape[:2])
        m, *_ = l_stats(lab[:, :, 0], mask)
        means.append(m)

    # Step 4: Progressive target equalization with adaptive limiter
    targets, med, t0, tN = progressive_targets(means, anchor_strength=0.6)
    print("\n🔧 [S1] Progressive luminance targeting...")
    print(f"   [S1] Global median = {med:.2f} | Anchors → first: {t0:.2f}, last: {tN:.2f}")
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
            print(f"   [S1:{i+1}] {paths[i].name}: desired×{desired:.3f} → clamped to ×{scale:.3f} (bounds {lo_cap:.3f}–{hi_cap:.3f})")
        else:
            print(f"   [S1:{i+1}] {paths[i].name}: scale ×{scale:.3f}")

        lab = apply_L_scale(lab, scale)
        corrected[i] = merge_alpha(cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR), a)
        if out_path is not None:
            cv2.imwrite(str(out_path / paths[i].name), corrected[i])

    # Step 5: Neighbor-aware low-frequency alignment
    print("🤝 [S1] Applying neighbor-aware low-freq smoothing...")
    for i in range(1, len(corrected)):
        prev_bgr, _ = split_alpha(corrected[i - 1])
        curr_bgr, curr_a = split_alpha(corrected[i])
        blended = low_freq_match_L(prev_bgr, curr_bgr, curr_a, beta=neighbor_beta, blur_ks=61, blur_sigma=0)
        corrected[i] = blended
        if out_path is not None:
            cv2.imwrite(str(out_path / paths[i].name), blended)

    # Step 6: Gentle global chroma alignment (±chroma_cap)
    print("🎨 [S1] Aligning global chroma (a/b channels, gently)...")
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
        if out_path is not None:
            cv2.imwrite(str(out_path / paths[i].name), corrected[i])

    # Step 7: Mild local contrast only if needed (final polish of Stage 1)
    print("✨ [S1] Final mild local-contrast polish (only if flat)...")
    for i, im in enumerate(corrected):
        bgr, a = split_alpha(im)
        bgr = mild_local_contrast_if_flat(bgr, a, flat_std_thresh=22.0, clip_limit=0.7, grid=(12, 12))
        corrected[i] = merge_alpha(bgr, a)
        if out_path is not None:
            cv2.imwrite(str(out_path / paths[i].name), corrected[i])

    return corrected

# ------------------------ Stage 2: Color transfer -----------------------------

def choose_reference_index(imgs):
    """Choose the tile whose L* mean is closest to the median; break ties by larger valid area."""
    L_means = []
    val_counts = []
    for im in imgs:
        lab, a = lab_with_alpha(im)
        mask = mask_from_alpha(a, lab.shape[:2])
        m, *_ = l_stats(lab[:, :, 0], mask)
        L_means.append(m)
        val_counts.append(int(mask.sum()))
    med = np.median(L_means)
    diffs = [abs(m - med) for m in L_means]
    best = np.argsort(diffs)
    top = best[:min(3, len(imgs))]
    # among top few, pick the one with max valid area
    top_best = max(top, key=lambda i: val_counts[i])
    return int(top_best)

def stage2_color_transfer(imgs, paths, out_dir=None, feather=21):
    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    ref_idx = choose_reference_index(imgs)
    ref = ensure_bgra(imgs[ref_idx])
    print(f"🎯 [S2] Reference chosen: index {ref_idx} → {paths[ref_idx].name}")

    outputs = []
    for i, (p, img) in enumerate(zip(paths, imgs)):
        if i == ref_idx:
            out = ref
        else:
            out = color_transfer_lab_masked(ref, ensure_bgra(img), feather=feather)
        outputs.append(out)
        if out_path is not None:
            cv2.imwrite(str(out_path / p.name), out)
        print(f"✓ [S2] Color-aligned: {p.name}")
    return outputs, ref_idx

# ------------------------ Stage 3: Auto global chroma bias correction -----------------------------

def stage3_global_unify(imgs, paths, out_dir=None, chroma_cap=2.0, bias_threshold=1.5):
    """
    Detect residual global a/b bias; if above threshold, gently center all tiles toward global medians.
    """
    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    a_means, b_means = [], []
    for im in imgs:
        bgr, _ = split_alpha(im)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        a_means.append(float(lab[:, :, 1].mean()))
        b_means.append(float(lab[:, :, 2].mean()))
    med_a, med_b = float(np.median(a_means)), float(np.median(b_means))

    # magnitude of overall chroma drift (median vs neutral 128 is not relevant; we center each image to group median)
    # Here we check spread: if median absolute deviation across tiles is large, we unify.
    mad_a = float(np.median(np.abs(np.array(a_means) - med_a)))
    mad_b = float(np.median(np.abs(np.array(b_means) - med_b)))
    spread = mad_a + mad_b

    if spread < bias_threshold:
        print(f"🌈 [S3] Residual chroma spread {spread:.2f} < threshold {bias_threshold:.2f}. Skipping global unify.")
        # Still save the inputs into out_dir for consistency
        if out_path is not None:
            for p, im in zip(paths, imgs):
                cv2.imwrite(str(out_path / p.name), im)
        return imgs

    print(f"🌈 [S3] Residual chroma spread {spread:.2f} ≥ threshold {bias_threshold:.2f}. Applying gentle unify...")
    unified = []
    for i, im in enumerate(imgs):
        bgr, a = split_alpha(im)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        da = np.clip(med_a - lab[:, :, 1].mean(), -chroma_cap, chroma_cap)
        db = np.clip(med_b - lab[:, :, 2].mean(), -chroma_cap, chroma_cap)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + da, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + db, 0, 255)
        bgr2 = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        # final polish per-frame if very flat
        bgr2 = mild_local_contrast_if_flat(bgr2, a, flat_std_thresh=22.0, clip_limit=0.7, grid=(12, 12))
        out = merge_alpha(bgr2, a)
        unified.append(out)
        if out_path is not None:
            cv2.imwrite(str(out_path / paths[i].name), out)
        print(f"✓ [S3] Unified: {paths[i].name}")
    return unified

# ------------------------ Orchestrator -----------------------------

def harmonize_images(
    images: Sequence[np.ndarray],
    names: Optional[Sequence[str]] = None,
    *,
    out_root: Optional[Path] = None,
    drift_slope_thresh: float = 0.5,
    neighbor_beta: float = 0.25,
    chroma_cap: float = 3.0,
    enable_pairwise_clahe: bool = False,
    feather: int = 21,
    s3_chroma_cap: float = 2.0,
    s3_bias_threshold: float = 1.5,
) -> Tuple[List[np.ndarray], int]:
    """Run the harmonization pipeline on in-memory images."""

    if not images:
        return [], -1

    if names is None:
        path_tags = [Path(f"image_{idx:02d}.png") for idx in range(len(images))]
    else:
        path_tags = [Path(str(name)) for name in names]

    prepared: List[np.ndarray] = [img.copy() for img in images]

    root_dir = Path(out_root) if out_root is not None else None
    if root_dir is not None:
        root_dir.mkdir(parents=True, exist_ok=True)

    s1_dir = root_dir / "01_bright" if root_dir is not None else None
    s1 = stage1_bright_normalize(
        prepared,
        path_tags,
        out_dir=s1_dir,
        drift_slope_thresh=drift_slope_thresh,
        neighbor_beta=neighbor_beta,
        chroma_cap=chroma_cap,
        enable_pairwise_clahe=enable_pairwise_clahe,
    )

    s2_dir = root_dir / "02_color_aligned" if root_dir is not None else None
    s2, ref_idx = stage2_color_transfer(
        s1,
        path_tags,
        out_dir=s2_dir,
        feather=feather,
    )

    s3_dir = root_dir / "03_final" if root_dir is not None else None
    s3 = stage3_global_unify(
        s2,
        path_tags,
        out_dir=s3_dir,
        chroma_cap=s3_chroma_cap,
        bias_threshold=s3_bias_threshold,
    )

    return s3, ref_idx


def run_pipeline(inp_dir, out_root,
                 drift_slope_thresh=0.5,
                 neighbor_beta=0.25,
                 chroma_cap=3.0,
                 enable_pairwise_clahe=False,
                 feather=21,
                 s3_chroma_cap=2.0,
                 s3_bias_threshold=1.5):
    inp = Path(inp_dir)
    out_root_path = Path(out_root)
    out_root_path.mkdir(parents=True, exist_ok=True)

    paths = sorted([p for p in inp.glob("*.png")])
    if not paths:
        print("No PNGs found in:", inp_dir)
        return

    imgs = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in paths]

    _, ref_idx = harmonize_images(
        imgs,
        names=[p.name for p in paths],
        out_root=out_root_path,
        drift_slope_thresh=drift_slope_thresh,
        neighbor_beta=neighbor_beta,
        chroma_cap=chroma_cap,
        enable_pairwise_clahe=enable_pairwise_clahe,
        feather=feather,
        s3_chroma_cap=s3_chroma_cap,
        s3_bias_threshold=s3_bias_threshold,
    )

    print("\n✅ Done. Outputs:")
    print(f"   Stage 1: {out_root_path / '01_bright'}")
    print(f"   Stage 2: {out_root_path / '02_color_aligned'}")
    print(f"   Stage 3: {out_root_path / '03_final'}")
    if 0 <= ref_idx < len(paths):
        print(f"   Reference tile (Stage 2): index {ref_idx} → {paths[ref_idx].name}")
    else:
        print("   Reference tile (Stage 2): unavailable")

# ------------------------ CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Auto Studio Harmonizer: exposure/color normalization + alpha-safe color transfer + global unify.")
    ap.add_argument("--inp", type=str, default="no_bg_photoroom", help="Input folder of PNGs")
    ap.add_argument("--out", type=str, default="harmonized_auto", help="Output root folder")
    ap.add_argument("--drift_thresh", type=float, default=0.5, help="Min |slope| to flatten drift (Stage 1)")
    ap.add_argument("--neighbor_beta", type=float, default=0.25, help="Strength of low-freq neighbor smoothing (Stage 1)")
    ap.add_argument("--chroma_cap", type=float, default=3.0, help="Max a/b offset per image (Stage 1)")
    ap.add_argument("--pairwise_clahe", action="store_true", help="Enable mild CLAHE during pairwise step (Stage 1)")
    ap.add_argument("--feather", type=int, default=21, help="Feather kernel for target mask during color transfer (Stage 2)")
    ap.add_argument("--s3_chroma_cap", type=float, default=2.0, help="Max a/b delta per image in global unify (Stage 3)")
    ap.add_argument("--s3_bias_threshold", type=float, default=1.5, help="MAD(a)+MAD(b) threshold to trigger Stage 3")
    args = ap.parse_args()

    run_pipeline(
        inp_dir=args.inp,
        out_root=args.out,
        drift_slope_thresh=args.drift_thresh,
        neighbor_beta=args.neighbor_beta,
        chroma_cap=args.chroma_cap,
        enable_pairwise_clahe=args.pairwise_clahe,
        feather=args.feather,
        s3_chroma_cap=args.s3_chroma_cap,
        s3_bias_threshold=args.s3_bias_threshold
    )

if __name__ == "__main__":
    main()


'''
python auto_tune.py \
  --inp input/no_bg_photoroom \
  --out harmonized_auto \
  --drift_thresh 0.5 \
  --neighbor_beta 0.25 \
  --chroma_cap 3.0 \
  --pairwise_clahe


'''

import cv2, numpy as np
from pathlib import Path

def image_stats_lab_masked(img_lab, mask):
    l, a, b = cv2.split(img_lab)
    valid = mask > 0
    return (
        l[valid].mean(), l[valid].std(),
        a[valid].mean(), a[valid].std(),
        b[valid].mean(), b[valid].std()
    )

def soft_mask(mask, blur=21):
    """Feather mask edges to avoid halos."""
    if blur > 0:
        mask = cv2.GaussianBlur(mask.astype(np.float32), (blur, blur), 0)
    mask = mask / (mask.max() + 1e-6)
    return np.clip(mask, 0, 1)

def color_transfer_lab_masked(source, target):
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
    sL, sLs, sA, sAs, sB, sBs = image_stats_lab_masked(src_lab, src_mask)
    tL, tLs, tA, tAs, tB, tBs = image_stats_lab_masked(tgt_lab, tgt_mask)

    # --- Apply transfer ---
    l, a, b = cv2.split(tgt_lab)
    l = ((l - tL) * (sLs / (tLs + 1e-6))) + sL
    a = ((a - tA) * (sAs / (tAs + 1e-6))) + sA
    b = ((b - tB) * (sBs / (tBs + 1e-6))) + sB
    out_lab = cv2.merge([np.clip(l,0,255), np.clip(a,0,255), np.clip(b,0,255)])
    out_bgr = cv2.cvtColor(out_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # --- Blend result only inside soft mask ---
    alpha = soft_mask(tgt_mask)
    out = (out_bgr * alpha[..., None] + tgt_bgr * (1 - alpha[..., None])).astype(np.uint8)

    # --- Re-attach transparency if present ---
    if target.shape[2] == 4:
        out = np.dstack([out, tgt_mask])
    return out

def batch_color_transfer(img_dir=".", out_dir="ct_out"):
    img_dir, out_dir = Path(img_dir), Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    paths = sorted(img_dir.glob("*.png"))
    imgs = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in paths]
    ref_idx = len(imgs)//2
    ref = imgs[ref_idx]

    for i, (p, img) in enumerate(zip(paths, imgs)):
        out = ref if i == ref_idx else color_transfer_lab_masked(ref, img)
        cv2.imwrite(str(out_dir / p.name), out)
        print("✓ Color-balanced:", p.name)

if __name__ == "__main__":
    batch_color_transfer("input/balanced_auto")

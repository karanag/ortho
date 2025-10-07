import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

import ssl
import urllib3

# Force using system SSL context, fallback to legacy
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
print("⚠️ Running in legacy SSL mode with LibreSSL fallback.")

# photoroom_bridge.py
import os, time, requests
from pathlib import Path
from typing import Optional, List, Tuple
import cv2
import numpy as np

PHOTOROOM_URL = "https://sdk.photoroom.com/v1/segment"  # returns transparent PNG

def _api_key(explicit: Optional[str]) -> str:
    key = (explicit or os.environ.get("PHOTOROOM_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("PhotoRoom API key missing. Set PHOTOROOM_API_KEY or pass api_key=...")
    return key

def photoroom_segment_dir(
    src_dir: Path,
    out_dir: Path,
    api_key: Optional[str] = None,
    pattern: str = "warp_*.png",
    save_alpha_mask: bool = False,
    retries: int = 2,
    delay_s: float = 1.5,
    hd_edges: bool = False,
) -> Tuple[Path, Optional[Path], List[Path]]:
    """
    Sends images to PhotoRoom and writes the returned PNG bytes verbatim.
    If save_alpha_mask=True, also writes alpha into out_dir/'masks'/*.png (kept separate).
    """
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = out_dir / "masks"
    if save_alpha_mask:
        mask_dir.mkdir(parents=True, exist_ok=True)

    headers = {"x-api-key": _api_key(api_key)}
    if hd_edges:
        # harmless if not recognized; matches your CLI help
        headers["pr-hd-background-removal"] = "auto"

    files = sorted(src_dir.glob(pattern))
    written: List[Path] = []

    for i, p in enumerate(files, 1):
        print(f"[PR] {i}/{len(files)}  {p.name}")
        ok = False
        for attempt in range(1, retries + 1):
            try:
                with open(p, "rb") as fh:
                    resp = requests.post(
                        PHOTOROOM_URL,
                        headers=headers,
                        files={"image_file": fh},
                        timeout=120,
                    )
                if resp.status_code == 200:
                    out_png = out_dir / f"{p.stem}.png"
                    with open(out_png, "wb") as out_f:
                        out_f.write(resp.content)  # write AS-IS
                    written.append(out_png)

                    if save_alpha_mask:
                        img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_UNCHANGED)
                        if img is not None and img.ndim == 3 and img.shape[2] == 4:
                            a = img[:, :, 3]
                            cv2.imwrite(str(mask_dir / f"{p.stem}_mask.png"), a)
                    ok = True
                    break
                else:
                    print(f"   ⚠️  {resp.status_code}: {resp.text[:120]}")
            except Exception as e:
                print(f"   ⚠️  attempt {attempt} failed: {e}")
            time.sleep(delay_s)

        if not ok:
            print(f"   ❌ skipped {p.name}")

    return out_dir, (mask_dir if save_alpha_mask else None), written


# Convenience wrapper used by orthomosaic.py
def photoroom_cutout_folder(
    src_dir: Path,
    out_dir: Path,
    api_key: Optional[str] = None,
    pattern: str = "warp_*.png",
    hd_auto: bool = False,
) -> Tuple[Path, Optional[Path], List[Path]]:
    return photoroom_segment_dir(
        src_dir=src_dir,
        out_dir=out_dir,
        api_key=api_key,
        pattern=pattern,
        save_alpha_mask=False,
        hd_edges=hd_auto,
    )


def merge_alpha_with_mask(bgra_img: np.ndarray, geom_mask_u8: np.ndarray, erode_ks: int = 3):
    """
    Combine PhotoRoom alpha with your geometric coverage mask (intersection).
    Returns (bgr, merged_alpha).
    """
    if bgra_img.ndim != 3 or bgra_img.shape[2] < 4:
        raise ValueError("merge_alpha_with_mask expects BGRA image")
    bgr = bgra_img[:, :, :3].copy()
    a_pr = bgra_img[:, :, 3].astype(np.uint8)

    a_geom = geom_mask_u8.astype(np.uint8)
    if erode_ks and erode_ks > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_ks, erode_ks))
        a_geom = cv2.erode(a_geom, k)

    # Intersection — keep only pixels that are valid in BOTH
    merged = cv2.min(a_pr, a_geom)
    return bgr, merged

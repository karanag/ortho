#!/usr/bin/env python3
import cv2
import numpy as np
import requests
import time
import os
from pathlib import Path
import argparse

# ──────────────────────────────────────────────
# PhotoRoom API Helper
# ──────────────────────────────────────────────
def load_token(token_file: str) -> str:
    path = Path(token_file)
    if not path.exists():
        raise FileNotFoundError(f"Token file not found: {token_file}")
    return path.read_text().strip()


def remove_bg_photoroom(img_bgr: np.ndarray, token: str, retries=2, delay=2.0) -> np.ndarray:
    """Send BGR image to Photoroom API and get RGBA result."""
    if img_bgr is None:
        raise ValueError("Empty image passed to remove_bg_photoroom()")
    api_url = "https://sdk.photoroom.com/v1/segment"
    headers = {"x-api-key": token}

    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Encoding error before API request")

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(api_url, headers=headers, files={"image_file": ("image.png", buf.tobytes(), "image/png")}, timeout=60)
            if resp.status_code == 200:
                data = np.frombuffer(resp.content, np.uint8)
                img_rgba = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
                if img_rgba is None or img_rgba.shape[2] < 4:
                    raise RuntimeError("Photoroom returned invalid RGBA")
                return img_rgba
            else:
                print(f"⚠️ API error {resp.status_code}, retrying ({attempt}/{retries})...")
        except Exception as e:
            print(f"⚠️ Request failed ({attempt}/{retries}): {e}")
        time.sleep(delay)
    raise RuntimeError("Background removal failed after retries")


# ──────────────────────────────────────────────
# Alpha + White Background Composite
# ──────────────────────────────────────────────
def composite_on_white(img_rgba: np.ndarray) -> np.ndarray:
    """Composite RGBA onto white background, preserving edges."""
    if img_rgba.shape[2] == 3:
        return img_rgba
    bgr, alpha = img_rgba[:, :, :3], img_rgba[:, :, 3] / 255.0
    white = np.ones_like(bgr, dtype=np.float32) * 255
    out = (bgr.astype(np.float32) * alpha[..., None] + white * (1 - alpha[..., None]))
    return np.clip(out, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────
# Folder / Batch Runner
# ──────────────────────────────────────────────
def process_folder(src, dst, token_file):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    token = load_token(token_file)

    files = sorted(list(src.glob("*.jpg")) + list(src.glob("*.jpeg")) + list(src.glob("*.png")))
    print(f"Found {len(files)} files in {src}")

    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name} ...", flush=True)
        img = cv2.imread(str(f))
        rgba = remove_bg_photoroom(img, token)
        final = composite_on_white(rgba)
        out_path = dst / f.name
        cv2.imwrite(str(out_path), final, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"✅ Saved → {out_path}")


# ──────────────────────────────────────────────
# CLI Entrypoint
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove background and composite over white using Photoroom API.")
    parser.add_argument("--src", required=True, help="Input file or folder")
    parser.add_argument("--out", required=True, help="Output file or folder")
    parser.add_argument("--bg-token-file", required=True, help="Photoroom API token file")
    args = parser.parse_args()

    src = Path(args.src)
    if src.is_dir():
        process_folder(src, args.out, args.bg_token_file)
    else:
        token = load_token(args.bg_token_file)
        img = cv2.imread(str(src))
        rgba = remove_bg_photoroom(img, token)
        final = composite_on_white(rgba)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), final, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"✅ Saved → {out_path}")

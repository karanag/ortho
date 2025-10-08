import time
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Union

import cv2
import numpy as np
import requests

try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.simplefilter("ignore", NotOpenSSLWarning)
except Exception:
    pass

DEFAULT_TOKEN_PATH: Optional[Union[str, Path]] = None


def _ensure_token_path(token_file: Optional[Union[str, Path]]) -> Path:
    if token_file is None:
        raise ValueError(
            "Photoroom API token path is not set. Provide --bg-token-file or pass token explicitly."
        )
    return Path(token_file)


def load_token(token_file: Optional[Union[str, Path]] = DEFAULT_TOKEN_PATH) -> str:
    """Read API token from a file"""
    path = _ensure_token_path(token_file)
    with path.open("r", encoding="utf-8") as f:
        return f.read().strip()


def remove_background_image(
    image_bgr: np.ndarray,
    *,
    token: Optional[str] = None,
    token_file: Optional[Union[str, Path]] = DEFAULT_TOKEN_PATH,
    retries: int = 2,
    delay: float = 2.0,
    timeout: float = 60.0,
    session: Optional[requests.Session] = None,
) -> Optional[np.ndarray]:
    """Remove background from a single image (BGR input) using Photoroom."""

    if image_bgr is None:
        return None
    if image_bgr.dtype != np.uint8:
        raise ValueError("remove_background_image expects uint8 image")

    token = token or load_token(token_file)
    api_url = "https://sdk.photoroom.com/v1/segment"
    headers = {"x-api-key": token}

    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode image for Photoroom request")
    payload = encoded.tobytes()

    if session is None:
        session = requests.Session()

    files = {"image_file": ("image.png", payload, "image/png")}

    for attempt in range(1, retries + 1):
        try:
            response = session.post(api_url, headers=headers, files=files, timeout=timeout)
            if response.status_code == 200:
                data = np.frombuffer(response.content, dtype=np.uint8)
                return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if delay > 0:
                time.sleep(delay)
        except Exception as exc:
            print(f"   ⚠️  PhotoRoom request failed (attempt {attempt}/{retries}): {exc}", flush=True)
            if delay > 0:
                time.sleep(delay)
    return None


def remove_background_batch(
    images_bgr: Sequence[np.ndarray],
    names: Optional[Sequence[str]] = None,
    *,
    token: Optional[str] = None,
    token_file: Optional[Union[str, Path]] = DEFAULT_TOKEN_PATH,
    retries: int = 2,
    delay: float = 2.0,
    timeout: float = 60.0,
    save_dir: Optional[Path] = None,
    session: Optional[requests.Session] = None,
    verbose: bool = True,
) -> List[Optional[np.ndarray]]:
    """Remove background for a batch of images provided as numpy arrays."""

    if not images_bgr:
        return []

    token = token or load_token(token_file)
    session = session or requests.Session()
    save_dir_path = Path(save_dir) if save_dir is not None else None
    if save_dir_path is not None:
        save_dir_path.mkdir(parents=True, exist_ok=True)

    results: List[Optional[np.ndarray]] = []
    total = len(images_bgr)
    for idx, img in enumerate(images_bgr, start=1):
        name = names[idx - 1] if names is not None and idx - 1 < len(names) else f"image_{idx:04d}.png"
        if verbose:
            print(f"[{idx}/{total}] Removing background for {name}...", flush=True)
        if img is None:
            results.append(None)
            if verbose:
                print(f"   ⚠️ Skipping {name}: empty image")
            continue
        out = remove_background_image(
            img,
            token=token,
            token_file=token_file,
            retries=retries,
            delay=delay,
            timeout=timeout,
            session=session,
        )
        if out is not None:
            results.append(out)
            if save_dir_path is not None:
                cv2.imwrite(str(save_dir_path / Path(name).name), out)
            if verbose:
                print(f"   ✅ Saved processed result for {name}", flush=True)
        else:
            results.append(None)
            if verbose:
                print(f"   ❌ Failed to process {name}", flush=True)
    return results


def remove_background_photoroom(
    src,
    dst="no_bg_photoroom",
    token_file: Optional[Union[str, Path]] = DEFAULT_TOKEN_PATH,
    retries: int = 2,
    delay: float = 2,
):
    """Send images in a folder to Photoroom API for background removal."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(exist_ok=True)
    token = load_token(token_file)

    files = sorted(list(src.glob("*.jpg")) + list(src.glob("*.jpeg")) + list(src.glob("*.png")))
    print(f"Found {len(files)} files in {src}")
    if not files:
        print("⚠️ No matching files found.")
        return

    session = requests.Session()
    images = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in files]
    names = [p.name for p in files]
    remove_background_batch(
        images,
        names,
        token=token,
        token_file=token_file,
        retries=retries,
        delay=delay,
        save_dir=dst,
        session=session,
        verbose=True,
    )
    print(f"\n🎯 All done → saved in: {dst}")

if __name__ == "__main__":
    remove_background_photoroom("img")

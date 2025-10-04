#!/usr/bin/env python3
"""Generate a grid of color/exposure variants for the COLMAP orthomosaic output.

The script expects that ``run.sh`` or ``main.py`` has already produced the debug
artifacts (warps/masks) under ``--debug-dir`` (defaults to ``out``).  It reads the
per-tile warped images, applies a suite of colour normalisation strategies and
exposure compensation options, rebends via the existing seam-hybrid blender, and
stores the resulting mosaics (with logs/metrics) in ``--output-root``.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from orthomosaic import (  # type: ignore
    seamhybrid_ortho_blend,
    compute_seam_masks_lowres,
    blend_fullres_with_masks,
    _ensure_binary,
)


@dataclass
class TileStats:
    """Basic colour statistics for a warped tile within its valid mask."""

    image: str
    area_px: int
    mean_bgr: Tuple[float, float, float]
    std_bgr: Tuple[float, float, float]
    mean_lab: Tuple[float, float, float]
    std_lab: Tuple[float, float, float]


@dataclass
class SweepResult:
    """Metadata captured for a single grid combination."""

    identifier: str
    color_mode: str
    color_params: Dict[str, float]
    exposure_mode: str
    output_dir: str
    mosaic_path: str
    mask_path: str
    tile_stats: List[TileStats]
    global_mean_bgr: Tuple[float, float, float]
    global_mean_lab: Tuple[float, float, float]
    global_std_bgr: Tuple[float, float, float]
    mosaic_rms_vs_baseline: float
    mask_area: int
    notes: str = ""


def _masked_pixels(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return pixels inside ``mask`` as (N, C) float array."""
    idx = mask > 0
    if not np.any(idx):
        return np.zeros((0, image.shape[2]), dtype=np.float32)
    return image[idx].reshape(-1, image.shape[2]).astype(np.float32)


def _masked_mean_std(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std on masked region (float32)."""
    pixels = _masked_pixels(image, mask)
    if pixels.size == 0:
        c = image.shape[2]
        return np.zeros(c, dtype=np.float32), np.zeros(c, dtype=np.float32)
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)
    return mean, std


def _to_lab(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_LAB2BGR)


def _apply_gray_world(image: np.ndarray, mask: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Gray-world white balance with optional blending strength."""
    mean_bgr, _ = _masked_mean_std(image, mask)
    if np.allclose(mean_bgr, 0):
        return image
    gray = float(np.mean(mean_bgr))
    scales = gray / (mean_bgr + 1e-6)
    scales = 1.0 + strength * (scales - 1.0)
    adjusted = image.astype(np.float32)
    for c in range(3):
        adjusted[..., c] *= scales[c]
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    adjusted[mask == 0] = image[mask == 0]
    return adjusted


def _apply_lab_match(
    image: np.ndarray,
    mask: np.ndarray,
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    """Match LAB mean/std of ``image`` inside mask to reference statistics."""
    lab = _to_lab(image).astype(np.float32)
    mask_idx = mask > 0
    if not np.any(mask_idx):
        return image
    for c in range(3):
        channel = lab[..., c]
        src_vals = channel[mask_idx]
        if src_vals.size == 0:
            continue
        src_mean = float(src_vals.mean())
        src_std = float(src_vals.std())
        if src_std < 1e-6:
            src_std = 1.0
        ref_mean_c = float(ref_mean[c])
        ref_std_c = float(ref_std[c])
        target = (src_vals - src_mean) * (ref_std_c / src_std) + ref_mean_c
        blended = src_vals * (1.0 - strength) + target * strength
        channel[mask_idx] = blended
        lab[..., c] = channel
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    result = _to_bgr(lab)
    result[mask == 0] = image[mask == 0]
    return result


def _build_histogram_lut(src_vals: np.ndarray, ref_vals: np.ndarray) -> np.ndarray:
    """Create LUT that maps ``src_vals`` histogram onto ``ref_vals`` histogram."""
    src_hist, _ = np.histogram(src_vals, bins=256, range=(0, 256))
    ref_hist, _ = np.histogram(ref_vals, bins=256, range=(0, 256))
    src_cdf = np.cumsum(src_hist).astype(np.float64)
    ref_cdf = np.cumsum(ref_hist).astype(np.float64)
    if src_cdf[-1] == 0 or ref_cdf[-1] == 0:
        return np.arange(256, dtype=np.uint8)
    src_cdf /= src_cdf[-1]
    ref_cdf /= ref_cdf[-1]
    lut = np.zeros(256, dtype=np.uint8)
    ref_idx = 0
    for val in range(256):
        while ref_idx < 255 and ref_cdf[ref_idx] < src_cdf[val]:
            ref_idx += 1
        lut[val] = ref_idx
    return lut


def _apply_hist_match(
    image: np.ndarray,
    mask: np.ndarray,
    ref_image: np.ndarray,
    ref_mask: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    """Histogram match the V-channel in HSV space to the reference tile."""
    mask_idx = mask > 0
    ref_idx = ref_mask > 0
    if not np.any(mask_idx) or not np.any(ref_idx):
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ref_hsv = cv2.cvtColor(ref_image, cv2.COLOR_BGR2HSV)
    src_vals = hsv[..., 2][mask_idx]
    ref_vals = ref_hsv[..., 2][ref_idx]
    lut = _build_histogram_lut(src_vals, ref_vals)
    mapped = cv2.LUT(hsv[..., 2], lut)
    blended = (1.0 - strength) * hsv[..., 2].astype(np.float32) + strength * mapped.astype(np.float32)
    hsv[..., 2] = np.clip(blended, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    result[mask == 0] = image[mask == 0]
    return result


def _apply_clahe_lab(image: np.ndarray, mask: np.ndarray, clip_limit: float = 2.0, grid: int = 8, strength: float = 1.0) -> np.ndarray:
    """CLAHE on the L channel in LAB space with masked blending."""
    lab = _to_lab(image)
    clahe = cv2.createCLAHE(clipLimit=max(0.1, clip_limit), tileGridSize=(grid, grid))
    L = lab[..., 0]
    enhanced = clahe.apply(L)
    blended = (1.0 - strength) * L.astype(np.float32) + strength * enhanced.astype(np.float32)
    lab[..., 0] = np.clip(blended, 0, 255).astype(np.uint8)
    out = _to_bgr(lab)
    out[mask == 0] = image[mask == 0]
    return out


def _apply_exposure_compensation(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    mode: str,
) -> List[np.ndarray]:
    """Apply OpenCV exposure compensator in-place and return adjusted list."""
    if mode == "none":
        return [img.copy() for img in images]
    type_lookup = {
        "gain": getattr(cv2.detail, "ExposureCompensator_GAIN", None),
        "gain_blocks": getattr(cv2.detail, "ExposureCompensator_GAIN_BLOCKS", None),
        "channels": getattr(cv2.detail, "ExposureCompensator_CHANNELS", None),
        "channels_blocks": getattr(cv2.detail, "ExposureCompensator_CHANNELS_BLOCKS", None),
    }
    comp_type = type_lookup.get(mode)
    if comp_type is None:
        raise ValueError(f"Unsupported exposure mode '{mode}'")
    comp = cv2.detail.ExposureCompensator_createDefault(comp_type)
    corners = [(0, 0)] * len(images)
    masks_bin = [_ensure_binary(m.copy()) for m in masks]
    comp.feed(corners, [img.astype(np.int16) for img in images], masks_bin)
    adjusted: List[np.ndarray] = []
    for idx, img in enumerate(images):
        out = img.copy()
        comp.apply(idx, (0, 0), out, masks_bin[idx])
        adjusted.append(out)
    return adjusted


def _enumerate_color_modes(reference: np.ndarray, reference_mask: np.ndarray) -> List[Tuple[str, Dict[str, float]]]:
    """Define the colour adjustment grid (name, params)."""
    # Entries are (mode_name, parameters...) to plug into adjustment functions.
    return [
        ("identity", {}),
        ("grayworld_50", {"strength": 0.5}),
        ("grayworld_100", {"strength": 1.0}),
        ("labmatch_70", {"strength": 0.7}),
        ("labmatch_100", {"strength": 1.0}),
        ("hismatch_70", {"strength": 0.7}),
        ("hismatch_100", {"strength": 1.0}),
        ("clahe_soft", {"clip_limit": 1.5, "grid": 8, "strength": 0.6}),
        ("clahe_strong", {"clip_limit": 2.5, "grid": 8, "strength": 1.0}),
    ]


def _select_reference_tile(warps: Sequence[np.ndarray], masks: Sequence[np.ndarray]) -> int:
    areas = [int(np.sum(mask > 0)) for mask in masks]
    if not areas:
        raise RuntimeError("No masks available to select reference tile")
    return int(np.argmax(areas))


def _compute_tile_stats(image: np.ndarray, mask: np.ndarray, image_name: str) -> TileStats:
    mean_bgr, std_bgr = _masked_mean_std(image, mask)
    lab = _to_lab(image)
    mean_lab, std_lab = _masked_mean_std(lab, mask)
    area = int(np.sum(mask > 0))
    return TileStats(
        image=image_name,
        area_px=area,
        mean_bgr=tuple(float(x) for x in mean_bgr),
        std_bgr=tuple(float(x) for x in std_bgr),
        mean_lab=tuple(float(x) for x in mean_lab),
        std_lab=tuple(float(x) for x in std_lab),
    )


def _rms_diff(image_a: np.ndarray, image_b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if image_a.shape != image_b.shape:
        raise ValueError("Images must have identical shapes for RMS diff")
    diff = image_a.astype(np.float32) - image_b.astype(np.float32)
    if mask is not None:
        idx = mask > 0
        if np.any(idx):
            diff = diff[idx]
    diff = diff.reshape(-1, diff.shape[-1])
    if diff.size == 0:
        return 0.0
    mse = float(np.mean(np.square(diff)))
    return math.sqrt(mse)


def _load_warps_and_masks(debug_dir: Path) -> Tuple[List[str], List[np.ndarray], List[np.ndarray]]:
    warp_paths = sorted(debug_dir.glob("warp_*.png"))
    mask_paths = sorted(debug_dir.glob("mask_*.png"))
    if not warp_paths or not mask_paths:
        raise FileNotFoundError(f"No warp/mask images found under {debug_dir}")
    if len(warp_paths) != len(mask_paths):
        raise RuntimeError("Mismatch between number of warp and mask images")
    image_names: List[str] = []
    warps: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    for warp_path, mask_path in zip(warp_paths, mask_paths):
        base = warp_path.name[len("warp_") : -len(".png")]
        # Format is 01_original-name.ext
        parts = base.split("_", 1)
        image_name = parts[1] if len(parts) > 1 else base
        warp = cv2.imread(str(warp_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if warp is None or mask is None:
            raise RuntimeError(f"Failed to read warp/mask pair: {warp_path}, {mask_path}")
        warps.append(warp)
        masks.append(mask)
        image_names.append(image_name)
    return image_names, warps, masks


def _ensure_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            for child in path.iterdir():
                if child.is_file() or child.is_symlink():
                    child.unlink()
                else:
                    _ensure_dir(child, overwrite=True)
                    child.rmdir()
        else:
            raise FileExistsError(f"Directory {path} already exists; use --overwrite to replace")
    path.mkdir(parents=True, exist_ok=True)


def _global_stats(tile_stats: Sequence[TileStats]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    if not tile_stats:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    total_area = max(1, sum(stat.area_px for stat in tile_stats))
    mean_bgr = np.zeros(3, dtype=np.float64)
    mean_lab = np.zeros(3, dtype=np.float64)
    std_bgr = np.zeros(3, dtype=np.float64)
    for stat in tile_stats:
        weight = stat.area_px / total_area
        mean_bgr += weight * np.array(stat.mean_bgr)
        mean_lab += weight * np.array(stat.mean_lab)
        std_bgr += weight * np.array(stat.std_bgr)
    return (
        tuple(float(x) for x in mean_bgr),
        tuple(float(x) for x in mean_lab),
        tuple(float(x) for x in std_bgr),
    )


def run_sweep(args: argparse.Namespace) -> Tuple[List[SweepResult], Path]:
    debug_dir = args.debug_dir.resolve()
    output_root = args.output_root.resolve()
    _ensure_dir(output_root, overwrite=args.overwrite)

    image_names, warps, masks = _load_warps_and_masks(debug_dir)
    reference_idx = _select_reference_tile(warps, masks)
    reference_image = warps[reference_idx]
    reference_mask = masks[reference_idx]
    ref_lab = _to_lab(reference_image).astype(np.float32)
    ref_mean_lab, ref_std_lab = _masked_mean_std(ref_lab, reference_mask)

    baseline_path = debug_dir / "orthomosaic_colmap.png"
    baseline = None
    if baseline_path.exists():
        baseline = cv2.imread(str(baseline_path), cv2.IMREAD_COLOR)

    color_modes = _enumerate_color_modes(reference_image, reference_mask)
    exposure_modes = ["none", "gain", "channels"]
    if args.exposure_modes:
        exposure_modes = args.exposure_modes
    if args.color_modes:
        color_modes = [(name, params) for name, params in color_modes if name in args.color_modes]
        missing = set(args.color_modes) - {name for name, _ in color_modes}
        if missing:
            raise ValueError(f"Unknown color modes requested: {sorted(missing)}")

    results: List[SweepResult] = []

    seam_masks_cached: List[np.ndarray] | None = None
    if args.reuse_seams:
        print("Precomputing seam masks once (reuse enabled)…")
        seam_masks_cached = compute_seam_masks_lowres(
            [img.copy() for img in warps],
            [_ensure_binary(m.copy()) for m in masks],
            scale=args.seam_scale,
            seam_method=args.seam_method,
            debug_dir=None,
        )

    for combo_idx, (color_name, color_params) in enumerate(color_modes, start=1):
        for exposure_mode in exposure_modes:
            identifier = f"{combo_idx:02d}_{color_name}__exp_{exposure_mode}"
            combo_dir = output_root / identifier
            _ensure_dir(combo_dir, overwrite=True)

            print(f"[sweep] {identifier} → processing …")
            adjusted_tiles: List[np.ndarray] = []
            for img, mask in zip(warps, masks):
                local = img.copy()
                if color_name.startswith("grayworld"):
                    strength = float(color_params.get("strength", 1.0))
                    local = _apply_gray_world(local, mask, strength=strength)
                elif color_name.startswith("labmatch"):
                    strength = float(color_params.get("strength", 1.0))
                    local = _apply_lab_match(local, mask, ref_mean_lab, ref_std_lab, strength=strength)
                elif color_name.startswith("hismatch"):
                    strength = float(color_params.get("strength", 1.0))
                    local = _apply_hist_match(local, mask, reference_image, reference_mask, strength=strength)
                elif color_name.startswith("clahe"):
                    local = _apply_clahe_lab(
                        local,
                        mask,
                        clip_limit=float(color_params.get("clip_limit", 2.0)),
                        grid=int(color_params.get("grid", 8)),
                        strength=float(color_params.get("strength", 1.0)),
                    )
                # identity leaves local untouched
                adjusted_tiles.append(local)

            exposed_tiles = _apply_exposure_compensation(adjusted_tiles, masks, exposure_mode)

            seam_debug_dir = combo_dir / "seams" if args.save_seam_debug else None
            mosaic_bgr: np.ndarray
            mosaic_mask: np.ndarray

            if seam_masks_cached is not None:
                if seam_debug_dir is not None:
                    seam_debug_dir.mkdir(parents=True, exist_ok=True)
                    for idx, seam_mask in enumerate(seam_masks_cached, start=1):
                        cv2.imwrite(
                            str(seam_debug_dir / f"seam_mask_{idx:02d}.png"),
                            _ensure_binary(seam_mask.copy()),
                        )
                mosaic_bgr, mosaic_mask = blend_fullres_with_masks(
                    [img.copy() for img in exposed_tiles],
                    [mask.copy() for mask in seam_masks_cached],
                    blender="multiband",
                    bands=args.bands,
                    feather_sharpness=args.feather_sharpness,
                    do_exposure=False,
                )
            else:
                if seam_debug_dir is not None:
                    seam_debug_dir.mkdir(parents=True, exist_ok=True)
                mosaic_bgr, mosaic_mask = seamhybrid_ortho_blend(
                    exposed_tiles,
                    [_ensure_binary(m.copy()) for m in masks],
                    seam_scale=args.seam_scale,
                    seam_method=args.seam_method,
                    blender="multiband",
                    bands=args.bands,
                    feather_sharpness=args.feather_sharpness,
                    do_exposure=False,
                    debug_dir=seam_debug_dir,
                )

            mosaic_path = combo_dir / "mosaic.png"
            mask_path = combo_dir / "mask.png"
            cv2.imwrite(str(mosaic_path), mosaic_bgr)
            cv2.imwrite(str(mask_path), mosaic_mask)

            tile_stats = [_compute_tile_stats(img, mask, name) for img, mask, name in zip(exposed_tiles, masks, image_names)]
            global_mean_bgr, global_mean_lab, global_std_bgr = _global_stats(tile_stats)
            rms_vs_baseline = 0.0
            if baseline is not None and baseline.shape == mosaic_bgr.shape:
                rms_vs_baseline = _rms_diff(mosaic_bgr, baseline, mosaic_mask)

            result = SweepResult(
                identifier=identifier,
                color_mode=color_name,
                color_params={k: float(v) for k, v in color_params.items()},
                exposure_mode=exposure_mode,
                output_dir=str(combo_dir.relative_to(output_root.parent if output_root.parent != output_root else output_root)),
                mosaic_path=str(mosaic_path.relative_to(output_root.parent if output_root.parent != output_root else output_root)),
                mask_path=str(mask_path.relative_to(output_root.parent if output_root.parent != output_root else output_root)),
                tile_stats=tile_stats,
                global_mean_bgr=global_mean_bgr,
                global_mean_lab=global_mean_lab,
                global_std_bgr=global_std_bgr,
                mosaic_rms_vs_baseline=rms_vs_baseline,
                mask_area=int(np.sum(mosaic_mask > 0)),
            )

            with open(combo_dir / "metadata.json", "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "identifier": result.identifier,
                        "color_mode": result.color_mode,
                        "color_params": result.color_params,
                        "exposure_mode": result.exposure_mode,
                        "tile_stats": [asdict(ts) for ts in result.tile_stats],
                        "global_mean_bgr": result.global_mean_bgr,
                        "global_mean_lab": result.global_mean_lab,
                        "global_std_bgr": result.global_std_bgr,
                        "mosaic_rms_vs_baseline": result.mosaic_rms_vs_baseline,
                        "mask_area": result.mask_area,
                    },
                    fh,
                    indent=2,
                )

            results.append(result)

    summary_path = output_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            [
                {
                    "identifier": r.identifier,
                    "color_mode": r.color_mode,
                    "color_params": r.color_params,
                    "exposure_mode": r.exposure_mode,
                    "mosaic_path": r.mosaic_path,
                    "mask_path": r.mask_path,
                    "global_mean_bgr": r.global_mean_bgr,
                    "global_mean_lab": r.global_mean_lab,
                    "global_std_bgr": r.global_std_bgr,
                    "mosaic_rms_vs_baseline": r.mosaic_rms_vs_baseline,
                    "mask_area": r.mask_area,
                }
                for r in results
            ],
            fh,
            indent=2,
        )

    return results, summary_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Grid sweep of colour/exposure settings for orthomosaic warps")
    p.add_argument("--debug-dir", type=Path, default=Path("out"), help="Directory containing warp_*.png/mask_*.png artefacts")
    p.add_argument("--output-root", type=Path, default=Path("out/sweeps"), help="Where to place sweep results")
    p.add_argument("--seam-scale", type=float, default=0.30, help="Downscale factor for seam finder")
    p.add_argument("--seam-method", type=str, default="graphcut", choices=["graphcut", "dp"], help="Seam finder variant")
    p.add_argument("--bands", type=int, default=3, help="Number of bands for multiband blender (hybrid stage)")
    p.add_argument("--feather-sharpness", type=float, default=0.02, help="Feather sharpness used during hybrid fallback")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory")
    p.add_argument("--color-modes", nargs="*", default=None, help="Subset of colour mode names to evaluate")
    p.add_argument("--exposure-modes", nargs="*", default=None, help="Subset of exposure modes to evaluate")
    p.add_argument("--save-seam-debug", action="store_true", help="Persist seam masks per combo for inspection")
    p.add_argument("--no-reuse-seams", dest="reuse_seams", action="store_false", help="Recompute seam masks for each combo instead of reusing baseline seams")
    p.set_defaults(reuse_seams=True)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    results, summary_path = run_sweep(args)
    print(f"\nGenerated {len(results)} mosaics. Summary: {summary_path}")
    for res in results:
        print(
            f" - {res.identifier}: color={res.color_mode}, exposure={res.exposure_mode}, "
            f"rms∆={res.mosaic_rms_vs_baseline:.2f}, mosaic={res.mosaic_path}"
        )


if __name__ == "__main__":
    main()

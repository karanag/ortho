#!/usr/bin/env python3
"""Colour harmonisation and mosaicking for COLMAP ortho tiles.

This script revisits the exposure/white-balance problem by solving for
per-tile corrections *before* the final seam blending:

1. Load warped ortho tiles (`warp_*.png`) and their masks (`mask_*.png`).
2. Analyse overlaps between every tile pair to measure relative brightness
   and colour biases directly where they intersect.
3. Solve a global least-squares problem for:
     * multiplicative gains on the LAB lightness channel (exposure), and
     * additive shifts on LAB a/b channels (white balance).
4. Apply these corrections to each tile, optionally followed by a light
   global grey-world pass for stability.
5. Blend the adjusted tiles with the existing seam-hybrid routine while
   reusing baseline seam masks (so geometric seams remain untouched).

Outputs are written to the requested directory (default `out/26sep`) and
include the harmonised mosaic, the per-tile adjustments, and diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from orthomosaic import (  # type: ignore
    compute_seam_masks_lowres,
    blend_fullres_with_masks,
    seamhybrid_ortho_blend,
    _ensure_binary,
)


@dataclass
class PairStats:
    tile_i: str
    tile_j: str
    overlap_px: int
    mean_L_i: float
    mean_L_j: float
    mean_a_i: float
    mean_a_j: float
    mean_b_i: float
    mean_b_j: float


def _ensure_dir(path: Path, overwrite: bool = False) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Directory {path} already exists; use --overwrite to replace")
        for item in sorted(path.iterdir(), reverse=True):
            if item.is_file() or item.is_symlink():
                item.unlink()
            else:
                _ensure_dir(item, overwrite=True)
                item.rmdir()
    path.mkdir(parents=True, exist_ok=True)


def _load_tiles(debug_dir: Path) -> Tuple[List[str], List[np.ndarray], List[np.ndarray]]:
    warp_paths = sorted(debug_dir.glob("warp_*.png"))
    mask_paths = sorted(debug_dir.glob("mask_*.png"))
    if not warp_paths or not mask_paths:
        raise FileNotFoundError(f"Expected warp_*.png and mask_*.png under {debug_dir}")
    if len(warp_paths) != len(mask_paths):
        raise RuntimeError("Mismatch between warp and mask counts")

    names: List[str] = []
    warps: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    for warp_path, mask_path in zip(warp_paths, mask_paths):
        base = warp_path.name[len("warp_") : -len(".png")]
        parts = base.split("_", 1)
        name = parts[1] if len(parts) > 1 else base
        warp = cv2.imread(str(warp_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if warp is None or mask is None:
            raise RuntimeError(f"Failed to load {warp_path} / {mask_path}")
        warps.append(warp)
        masks.append(mask)
        names.append(name)
    return names, warps, masks


def _lab_tiles(images: Sequence[np.ndarray]) -> List[np.ndarray]:
    return [cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32) for img in images]


def _sample_overlap(
    mask_i: np.ndarray,
    mask_j: np.ndarray,
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    overlap = (mask_i > 0) & (mask_j > 0)
    if not np.any(overlap):
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)
    ys, xs = np.where(overlap)
    if ys.size > max_samples:
        idx = np.random.default_rng(0).choice(ys.size, size=max_samples, replace=False)
        return ys[idx], xs[idx]
    return ys, xs


def _collect_pair_stats(
    names: Sequence[str],
    lab_tiles: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    min_overlap: int,
    max_samples: int,
) -> List[PairStats]:
    stats: List[PairStats] = []
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            ys, xs = _sample_overlap(masks[i], masks[j], max_samples=max_samples)
            count = ys.size
            if count < min_overlap:
                continue
            lab_i = lab_tiles[i]
            lab_j = lab_tiles[j]
            mean_L_i = float(lab_i[ys, xs, 0].mean())
            mean_L_j = float(lab_j[ys, xs, 0].mean())
            mean_a_i = float(lab_i[ys, xs, 1].mean())
            mean_a_j = float(lab_j[ys, xs, 1].mean())
            mean_b_i = float(lab_i[ys, xs, 2].mean())
            mean_b_j = float(lab_j[ys, xs, 2].mean())
            stats.append(
                PairStats(
                    tile_i=names[i],
                    tile_j=names[j],
                    overlap_px=count,
                    mean_L_i=mean_L_i,
                    mean_L_j=mean_L_j,
                    mean_a_i=mean_a_i,
                    mean_a_j=mean_a_j,
                    mean_b_i=mean_b_i,
                    mean_b_j=mean_b_j,
                )
            )
    if not stats:
        raise RuntimeError("No overlapping tiles found; cannot harmonise colours")
    return stats


def _solve_linear_system(
    pairs: Iterable[Tuple[int, int, float, float]],
    num_tiles: int,
    anchor_idx: int,
    reg_lambda: float = 0.0,
) -> np.ndarray:
    rows: List[np.ndarray] = []
    rhs: List[float] = []
    weights: List[float] = []
    for i, j, value, weight in pairs:
        row = np.zeros(num_tiles, dtype=np.float64)
        row[i] = 1.0
        row[j] = -1.0
        rows.append(row)
        rhs.append(value)
        weights.append(weight)

    # Anchor selected tile at zero to stabilise the system
    anchor_row = np.zeros(num_tiles, dtype=np.float64)
    anchor_row[anchor_idx] = 1.0
    rows.append(anchor_row)
    rhs.append(0.0)
    weights.append(1.0)

    if reg_lambda > 0.0:
        for idx in range(num_tiles):
            row = np.zeros(num_tiles, dtype=np.float64)
            row[idx] = 1.0
            rows.append(row)
            rhs.append(0.0)
            weights.append(reg_lambda)

    A = np.vstack(rows)
    b = np.array(rhs, dtype=np.float64)
    w = np.sqrt(np.clip(np.array(weights, dtype=np.float64), 1e-6, None))
    A *= w[:, None]
    b *= w

    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return sol.astype(np.float32)


def _harmonise_tiles(
    names: Sequence[str],
    lab_tiles: List[np.ndarray],
    masks: Sequence[np.ndarray],
    stats: Sequence[PairStats],
    reference_idx: int,
    exposure_clip: Tuple[float, float],
    gain_reg: float,
    color_reg: float,
) -> Tuple[List[np.ndarray], Dict[str, Dict[str, float]]]:
    index_map = {name: idx for idx, name in enumerate(names)}

    pairs_L: List[Tuple[int, int, float, float]] = []
    pairs_a: List[Tuple[int, int, float, float]] = []
    pairs_b: List[Tuple[int, int, float, float]] = []

    for st in stats:
        i = index_map[st.tile_i]
        j = index_map[st.tile_j]
        weight = float(st.overlap_px)
        if st.mean_L_i <= 1e-3 or st.mean_L_j <= 1e-3:
            continue
        log_diff = math.log(st.mean_L_j) - math.log(st.mean_L_i)
        pairs_L.append((i, j, log_diff, weight))
        pairs_a.append((i, j, st.mean_a_j - st.mean_a_i, weight))
        pairs_b.append((i, j, st.mean_b_j - st.mean_b_i, weight))

    gains_log = _solve_linear_system(pairs_L, len(names), reference_idx, reg_lambda=gain_reg)
    offsets_a = _solve_linear_system(pairs_a, len(names), reference_idx, reg_lambda=color_reg)
    offsets_b = _solve_linear_system(pairs_b, len(names), reference_idx, reg_lambda=color_reg)

    gains_log = gains_log - float(np.median(gains_log))
    offsets_a = offsets_a - float(np.median(offsets_a))
    offsets_b = offsets_b - float(np.median(offsets_b))

    gains = np.clip(np.exp(gains_log), exposure_clip[0], exposure_clip[1])
    corrections: Dict[str, Dict[str, float]] = {}
    adjusted_tiles: List[np.ndarray] = []

    for idx, name in enumerate(names):
        lab = lab_tiles[idx].copy()
        lab[..., 0] *= gains[idx]
        lab[..., 0] = np.clip(lab[..., 0], 0.0, 255.0)
        lab[..., 1] += offsets_a[idx]
        lab[..., 2] += offsets_b[idx]
        lab[..., 1] = np.clip(lab[..., 1], 0.0, 255.0)
        lab[..., 2] = np.clip(lab[..., 2], 0.0, 255.0)
        adjusted_tiles.append(lab)
        corrections[name] = {
            "gain_L": float(gains[idx]),
            "offset_a": float(offsets_a[idx]),
            "offset_b": float(offsets_b[idx]),
        }

    return adjusted_tiles, corrections


def _lab_to_bgr(tiles_lab: Sequence[np.ndarray]) -> List[np.ndarray]:
    return [cv2.cvtColor(tile.astype(np.uint8), cv2.COLOR_LAB2BGR) for tile in tiles_lab]


def _gray_world_global(images: Sequence[np.ndarray], masks: Sequence[np.ndarray], strength: float) -> List[np.ndarray]:
    adjusted: List[np.ndarray] = []
    for img, mask in zip(images, masks):
        masked = img[mask > 0]
        if masked.size == 0:
            adjusted.append(img)
            continue
        means = masked.mean(axis=0)
        gray = float(means.mean())
        scales = gray / (means + 1e-6)
        scales = 1.0 + strength * (scales - 1.0)
        out = img.astype(np.float32)
        for c in range(3):
            out[..., c] *= scales[c]
        out = np.clip(out, 0, 255).astype(np.uint8)
        out[mask == 0] = img[mask == 0]
        adjusted.append(out)
    return adjusted


def harmonise_and_blend(args: argparse.Namespace) -> Path:
    debug_dir = args.debug_dir.resolve()
    output_dir = args.output_dir.resolve()
    _ensure_dir(output_dir, overwrite=args.overwrite)

    names, warps_bgr, masks = _load_tiles(debug_dir)
    masks = [_ensure_binary(m) for m in masks]
    lab_tiles = _lab_tiles(warps_bgr)

    reference_idx = int(np.argmax([mask.sum() for mask in masks]))

    pair_stats = _collect_pair_stats(
        names,
        lab_tiles,
        masks,
        min_overlap=args.min_overlap,
        max_samples=args.max_overlap_samples,
    )

    harmonised_lab, corrections = _harmonise_tiles(
        names,
        lab_tiles,
        masks,
        pair_stats,
        reference_idx,
        exposure_clip=(args.min_gain, args.max_gain),
        gain_reg=args.gain_regularization,
        color_reg=args.color_regularization,
    )

    harmonised_bgr = _lab_to_bgr(harmonised_lab)

    if args.gray_world_strength > 0:
        harmonised_bgr = _gray_world_global(harmonised_bgr, masks, strength=args.gray_world_strength)

    # Reuse baseline seam layout unless explicitly disabled.
    if args.reuse_seams:
        seam_masks = compute_seam_masks_lowres(
            warps_bgr,
            masks,
            scale=args.seam_scale,
            seam_method=args.seam_method,
            debug_dir=None,
        )
        mosaic_bgr, mosaic_mask = blend_fullres_with_masks(
            harmonised_bgr,
            seam_masks,
            blender="multiband",
            bands=args.bands,
            feather_sharpness=args.feather_sharpness,
            do_exposure=False,
        )
    else:
        mosaic_bgr, mosaic_mask = seamhybrid_ortho_blend(
            harmonised_bgr,
            masks,
            seam_scale=args.seam_scale,
            seam_method=args.seam_method,
            blender="multiband",
            bands=args.bands,
            feather_sharpness=args.feather_sharpness,
            do_exposure=False,
            debug_dir=None,
        )

    mosaic_path = output_dir / "orthomosaic_harmonised.png"
    mask_path = output_dir / "orthomosaic_mask.png"
    cv2.imwrite(str(mosaic_path), mosaic_bgr)
    cv2.imwrite(str(mask_path), mosaic_mask)

    adjustments_path = output_dir / "tile_adjustments.json"
    with open(adjustments_path, "w", encoding="utf-8") as fh:
        json.dump(corrections, fh, indent=2)

    stats_path = output_dir / "pair_stats.json"
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump([ps.__dict__ for ps in pair_stats], fh, indent=2)

    preview_dir = output_dir / "tiles_adjusted"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for name, img in zip(names, harmonised_bgr):
        cv2.imwrite(str(preview_dir / f"{name}.png"), img)

    log = {
        "reference_tile": names[reference_idx],
        "num_tiles": len(names),
        "min_overlap": args.min_overlap,
        "max_overlap_samples": args.max_overlap_samples,
        "seam_scale": args.seam_scale,
        "seam_method": args.seam_method,
        "bands": args.bands,
        "gray_world_strength": args.gray_world_strength,
        "reuse_seams": args.reuse_seams,
        "gain_regularization": args.gain_regularization,
        "color_regularization": args.color_regularization,
        "output_mosaic": str(mosaic_path),
    }
    with open(output_dir / "harmonise_log.json", "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)

    return mosaic_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Solve global colour/exposure harmonisation for ortho tiles")
    p.add_argument("--debug-dir", type=Path, default=Path("out"), help="Directory with warp_*.png/mask_*.png artefacts")
    p.add_argument("--output-dir", type=Path, default=Path("out/26sep"), help="Destination directory for harmonised outputs")
    p.add_argument("--min-overlap", type=int, default=25000, help="Minimum overlapping pixels to use a tile pair")
    p.add_argument("--max-overlap-samples", type=int, default=200000, help="Max pixels sampled per overlap region")
    p.add_argument("--min-gain", type=float, default=0.6, help="Lower clamp for exposure gain")
    p.add_argument("--max-gain", type=float, default=1.8, help="Upper clamp for exposure gain")
    p.add_argument("--gain-regularization", type=float, default=0.05, help="L2 regularisation weight for gain solving (higher = closer to 1)")
    p.add_argument("--color-regularization", type=float, default=0.05, help="L2 regularisation weight for colour offsets (higher = closer to 0)")
    p.add_argument("--gray-world-strength", type=float, default=0.15, help="Strength for optional gray-world pass (0 to disable)")
    p.add_argument("--seam-scale", type=float, default=0.30, help="Downscale factor for seam finder")
    p.add_argument("--seam-method", type=str, default="graphcut", choices=["graphcut", "dp"], help="Seam finder type")
    p.add_argument("--bands", type=int, default=4, help="Number of bands for multiband blending")
    p.add_argument("--feather-sharpness", type=float, default=0.02, help="Feather sharpness when fallback blending is used")
    p.add_argument("--reuse-seams", action="store_true", default=True, help="Reuse baseline seam masks via blend_fullres_with_masks")
    p.add_argument("--overwrite", action="store_true", help="Replace output directory if it exists")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    mosaic_path = harmonise_and_blend(args)
    print(f"Saved harmonised mosaic to {mosaic_path}")


if __name__ == "__main__":
    main()

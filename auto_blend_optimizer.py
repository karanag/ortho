from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import json
import numpy as np


# --- simple Unix/macOS timeout helper ---
import signal
class _CandidateTimeout(Exception): pass
def _timebox(seconds, fn):
    def _raise_timeout(signum, frame):  # noqa: ARG002
        raise _CandidateTimeout()
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(int(seconds))
    try:
        return fn()
    finally:
        signal.alarm(0)
# --- end helper ---


@dataclass
class AutoBlendParams:
    lfe_strength: float = 0.60
    lfe_ksize: int = 121  # must be odd
    do_exposure: bool = True

    blender: str = "multiband"
    bands: int = 4
    seam_method: str = "graphcut"
    seam_scale: float = 0.30
    seam_cost: str = "gradient"
    seam_gradient_weight: float = 10.0
    candidate_timeout_s: int = 90

    flow_enable: bool = False
    flow_method: str = "farneback_slow"
    flow_max_px: float = 1.0
    flow_smooth_ksize: int = 31


@dataclass
class AutoBlendScores:
    sharpness_global: float
    seam_contrast: float
    color_mismatch: float
    composite: float


@dataclass
class AutoBlendChoice:
    params: AutoBlendParams
    scores: AutoBlendScores
    details: Dict[str, Any]


def _odd(value: int) -> int:
    return (value // 2) * 2 + 1


def _variance_of_laplacian(img_gray: np.ndarray) -> float:
    lap = cv2.Laplacian(img_gray, cv2.CV_32F)
    return float(lap.var())


def _lowfreq_equalize_tiles(
    warps_rgb: List[np.ndarray],
    masks_u8: List[np.ndarray],
    ksize: int = 121,
    strength: float = 0.60,
) -> List[np.ndarray]:
    k = _odd(int(ksize))
    eps = 1e-6

    blurs = [cv2.GaussianBlur(w.astype(np.float32), (k, k), 0) for w in warps_rgb]

    from orthomosaic import blend_fullres_with_masks, _ensure_binary

    blur_bgr = [
        cv2.cvtColor(np.clip(b, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        for b in blurs
    ]
    illum_bgr, _ = blend_fullres_with_masks(
        blur_bgr,
        [_ensure_binary(m) for m in masks_u8],
        blender="feather",
        bands=3,
        feather_sharpness=0.01,
        do_exposure=False,
    )
    illum = cv2.cvtColor(illum_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    equalized: List[np.ndarray] = []
    for warped, blurred in zip(warps_rgb, blurs):
        gain = (illum + eps) / (blurred + eps)
        gain = cv2.GaussianBlur(gain, (k, k), 0)
        gain = (1.0 - strength) + strength * gain
        gain = np.clip(gain, 0.6, 1.6)
        corrected = np.clip(warped.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        equalized.append(corrected)

    return equalized


def _measure_lowfreq_mismatch(
    warps_rgb: List[np.ndarray],
    masks_u8: List[np.ndarray],
) -> float:
    k = 101
    blurs = [cv2.GaussianBlur(w, (k, k), 0).astype(np.float32) for w in warps_rgb]
    diff_sum = 0.0
    pair_count = 0.0
    for i in range(len(blurs) - 1):
        for j in range(i + 1, len(blurs)):
            overlap = (masks_u8[i] > 0) & (masks_u8[j] > 0)
            if overlap.sum() < 256:
                continue
            da = blurs[i][overlap]
            db = blurs[j][overlap]
            diff_sum += float(np.mean(np.abs(da - db)))
            pair_count += 1.0
    return diff_sum / (pair_count + 1e-6)


def _evaluate_mosaic(
    mosaic_rgb: np.ndarray,
    seam_masks: Optional[List[np.ndarray]] = None,
) -> AutoBlendScores:
    gray = cv2.cvtColor(mosaic_rgb, cv2.COLOR_RGB2GRAY)
    sharpness = _variance_of_laplacian(gray)

    seam_contrast = 0.0
    if seam_masks is not None and len(seam_masks) >= 2:
        belt = np.zeros_like(seam_masks[0], dtype=np.uint8)
        for mask in seam_masks:
            belt = cv2.bitwise_or(belt, (mask > 0).astype(np.uint8) * 255)
        dil = cv2.dilate(belt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        ero = cv2.erode(belt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        seam_line = cv2.bitwise_xor(dil, ero)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        vals = mag[seam_line > 0]
        seam_contrast = float(np.mean(vals)) if vals.size else 0.0

    blurred = cv2.GaussianBlur(gray, (101, 101), 0).astype(np.float32)
    color_mismatch = float(blurred.std())

    alpha, beta = 0.4, 0.15
    composite = sharpness - alpha * seam_contrast - beta * color_mismatch
    return AutoBlendScores(sharpness, seam_contrast, color_mismatch, composite)


def auto_optimize_blend(
    warps_rgb: List[np.ndarray],
    masks_u8: List[np.ndarray],
    debug_dir: Optional[Path] = None,
    try_flow: bool = False,  # placeholder to extend later
    base_params: Optional[AutoBlendParams] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    from orthomosaic import (
        compute_seam_masks_lowres,
        blend_fullres_with_masks,
        _ensure_binary,
    )

    if base_params is None:
        base_params = AutoBlendParams()

    strengths = [0.50, 0.55, 0.60, 0.65]
    best_strength = base_params.lfe_strength
    best_metric = float("inf")
    for strength in strengths:
        print(f"[AUTO] Evaluating LFE strength {strength:.2f}")
        equalized = _lowfreq_equalize_tiles(
            warps_rgb, masks_u8, ksize=base_params.lfe_ksize, strength=strength
        )
        mismatch = _measure_lowfreq_mismatch(equalized, masks_u8)
        if mismatch < best_metric:
            best_metric = mismatch
            best_strength = strength
    lfe_warps = _lowfreq_equalize_tiles(
        warps_rgb, masks_u8, ksize=base_params.lfe_ksize, strength=best_strength
    )
    print(f"[AUTO] Selected LFE strength {best_strength:.2f} (mismatch={best_metric:.4f})")

    lfe_warps_bgr = [cv2.cvtColor(warp, cv2.COLOR_RGB2BGR) for warp in lfe_warps]
    masks_binary = [_ensure_binary(m) for m in masks_u8]

    search_method = base_params.seam_method.lower()
    seam_cost = base_params.seam_cost.lower()
    grad_weight = float(base_params.seam_gradient_weight)

    BAND_OPTIONS = [base_params.bands if hasattr(base_params, "bands") else 3]
    SEAM_SCALES = [0.5, 0.4]
    SEARCH_METHODS = [search_method]

    search_candidates: List[Dict[str, Any]] = []
    for bands_candidate in BAND_OPTIONS:
        search_bands = max(1, min(int(bands_candidate), 3))
        for search_method_candidate in SEARCH_METHODS:
            for seam_scale in SEAM_SCALES:
                print(
                    f"[AUTO] Evaluating seam_scale={seam_scale:.2f} "
                    f"(search_method={search_method_candidate}, bands={search_bands})"
                )
                def _run_candidate() -> Tuple[np.ndarray, List[np.ndarray]]:
                    seam_masks_local = compute_seam_masks_lowres(
                        lfe_warps_bgr,
                        masks_binary,
                        scale=seam_scale,
                        seam_method=search_method_candidate,
                        seam_cost=seam_cost,
                        seam_gradient_weight=grad_weight,
                        debug_dir=None,
                    )
                    mosaic_bgr_local, _ = blend_fullres_with_masks(
                        lfe_warps_bgr,
                        seam_masks_local,
                        blender=base_params.blender,
                        bands=search_bands,
                        feather_sharpness=0.02,
                        do_exposure=base_params.do_exposure,
                    )
                    return mosaic_bgr_local, seam_masks_local
                try:
                    mosaic_bgr, seam_masks = _timebox(
                        base_params.candidate_timeout_s
                        if hasattr(base_params, "candidate_timeout_s")
                        else 90,
                        _run_candidate,
                    )
                except _CandidateTimeout:
                    print(
                        f"[AUTO] Candidate timed out (method={search_method_candidate}, "
                        f"scale={seam_scale}, bands={search_bands}); skipping."
                    )
                    continue
                mosaic_rgb = cv2.cvtColor(mosaic_bgr, cv2.COLOR_BGR2RGB)
                scores = _evaluate_mosaic(mosaic_rgb, seam_masks=seam_masks)
                search_candidates.append(
                    {
                        "seam_scale": seam_scale,
                        "scores": scores,
                        "search_method": search_method_candidate,
                        "bands": search_bands,
                        "lowfreq_mismatch": best_metric,
                    }
                )

    if not search_candidates:
        raise RuntimeError("[AUTO] No seam candidates evaluated.")

    best_search = max(search_candidates, key=lambda c: c["scores"].composite)
    selected_scale = float(best_search["seam_scale"])
    final_seam_method = best_search["search_method"]

    selected_params = AutoBlendParams(
        lfe_strength=best_strength,
        lfe_ksize=base_params.lfe_ksize,
        do_exposure=base_params.do_exposure,
        blender=base_params.blender,
        bands=base_params.bands,
        seam_method=final_seam_method,
        seam_scale=selected_scale,
        seam_cost=base_params.seam_cost,
        seam_gradient_weight=grad_weight,
        flow_enable=base_params.flow_enable,
        flow_method=base_params.flow_method,
        flow_max_px=base_params.flow_max_px,
        flow_smooth_ksize=base_params.flow_smooth_ksize,
    )
    print("[AUTO] Best search candidate:", selected_params)
    print(
        f"[AUTO] Re-blending at full resolution with seam_method={final_seam_method}, "
        f"bands={base_params.bands}, seam_scale={selected_scale:.2f}"
    )

    final_debug_dir: Optional[Path] = None
    if debug_dir is not None:
        final_debug_dir = Path(debug_dir)
        final_debug_dir.mkdir(parents=True, exist_ok=True)

    seam_debug_dir: Optional[Path] = None
    if final_debug_dir is not None:
        seam_debug_dir = final_debug_dir / "seams_final"

    final_seam_masks = compute_seam_masks_lowres(
        lfe_warps_bgr,
        masks_binary,
        scale=selected_scale,
        seam_method=final_seam_method,
        seam_cost=seam_cost,
        seam_gradient_weight=grad_weight,
        debug_dir=seam_debug_dir,
    )
    mosaic_bgr_final, _ = blend_fullres_with_masks(
        lfe_warps_bgr,
        final_seam_masks,
        blender=base_params.blender,
        bands=base_params.bands,
        feather_sharpness=0.02,
        do_exposure=base_params.do_exposure,
    )
    mosaic_rgb_final = cv2.cvtColor(mosaic_bgr_final, cv2.COLOR_BGR2RGB)
    final_scores = _evaluate_mosaic(mosaic_rgb_final, seam_masks=final_seam_masks)

    best_choice = AutoBlendChoice(
        params=selected_params,
        scores=final_scores,
        details={
            "lowfreq_mismatch": best_metric,
            "search_method": final_seam_method,
            "search_bands": best_search["bands"],
        },
    )

    report = {
        "selected_params": asdict(best_choice.params),
        "scores": best_choice.scores.__dict__,
        "search": {
            "method": final_seam_method,
            "bands": best_search["bands"],
            "candidates": [
                {
                    "seam_scale": cand["seam_scale"],
                    "bands": cand["bands"],
                    "search_method": cand["search_method"],
                    "scores": cand["scores"].__dict__,
                    "lowfreq_mismatch": cand["lowfreq_mismatch"],
                }
                for cand in sorted(search_candidates, key=lambda c: -c["scores"].composite)
            ],
        },
    }

    if final_debug_dir is not None:
        with (final_debug_dir / "auto_blend_report.json").open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

    return mosaic_rgb_final, report

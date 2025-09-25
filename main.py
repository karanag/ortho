#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path
from typing import Optional

import pycolmap

# only orchestrate; all heavy lifting remains in orthomosaic.py
from orthomosaic import run_dense_reconstruction, build_orthomosaic


def run(cmd: list[str]):
    print("Running:", " ".join(cmd))
    code = os.system(" ".join(cmd))
    if code != 0:
        raise SystemExit(f"Command failed with exit code {code}: {' '.join(cmd)}")


def select_best_reconstruction(sparse_dir: Path) -> tuple[Path, tuple[int, int]]:
    """Return the reconstruction directory with the most registered images (tie-breaking by points)."""
    candidates = [d for d in sorted(sparse_dir.iterdir()) if d.is_dir()]
    if not candidates:
        raise SystemExit(f"COLMAP mapper did not create any reconstructions under {sparse_dir}")

    best_dir: Optional[Path] = None
    best_stats = (-1, -1)
    for candidate in candidates:
        reconstruction = pycolmap.Reconstruction(str(candidate))
        stats = (len(reconstruction.images), len(reconstruction.points3D))
        if stats > best_stats:
            best_dir = candidate
            best_stats = stats

    if best_dir is None:
        raise SystemExit("Unable to select a reconstruction from mapper output.")

    return best_dir, best_stats


def main():
    parser = argparse.ArgumentParser(description="COLMAP SfM + MVS + Orthomosaic")
    parser.add_argument("--images", type=Path, default=Path(__file__).resolve().parent / "images")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "output")

    # Matching options
    parser.add_argument("--sequential", action="store_true", help="Use sequential matcher (recommended for sweep/pan)")
    parser.add_argument("--exhaustive", action="store_true", help="Use exhaustive matcher instead of sequential")
    parser.add_argument("--overlap", type=int, default=3, help="Sequential overlap")
    parser.add_argument("--single_camera", type=int, default=1, help="COLMAP ImageReader.single_camera (1 or 0)")

    # Mosaic & blending
    parser.add_argument("--max_mosaic_px", type=int, default=4000,
                        help="Target max width/height of final mosaic (orthorectified)")
    parser.add_argument("--blend", type=str, default="multiband",
                        choices=["multiband", "feather", "enfuse", "seamhybrid"],
                        help="Blending mode for mosaic")
    parser.add_argument("--bands", type=int, default=6, help="Num bands for multiband blender")
    parser.add_argument("--feather-sharpness", type=float, default=0.02,
                        help="Feather blender sharpness (only used if feather blender is selected)")

    # Seam-hybrid knobs (used when --blend seamhybrid)
    parser.add_argument("--seam-scale", type=float, default=0.30,
                        help="Downscale for seam finding (e.g., 0.3 = 30%)")
    parser.add_argument("--seam-method", type=str, default="graphcut",
                        choices=["graphcut", "dp"], help="Seam finder type for seamhybrid")

    # Optional flow refinement (your existing options)
    parser.add_argument("--fresh", action="store_true", help="Clear DB and outputs in --out before running")
    parser.add_argument("--no-flow-refine", action="store_true", help="Disable optical-flow refinement")
    parser.add_argument("--flow-method", type=str, default="farneback_slow",
                        choices=["farneback_slow", "farneback", "dis"])
    parser.add_argument("--flow-downscale", type=float, default=1.0)
    parser.add_argument("--flow-max-px", type=float, default=2.5)
    parser.add_argument("--flow-smooth-ksize", type=int, default=13)

    # Stripe splitting & APAP options (existing)
    parser.add_argument("--split-stripes", action="store_true")
    parser.add_argument("--stripe-threshold", type=float, default=0.5)
    parser.add_argument("--stripe-flow-max-px", type=float, default=6.0)
    parser.add_argument("--warp-model", type=str, default="homography", choices=["homography", "apap"])
    parser.add_argument("--apap-cell-size", type=int, default=80)
    parser.add_argument("--apap-sigma", type=float, default=120.0)
    parser.add_argument("--apap-min-weight", type=float, default=1e-4)
    parser.add_argument("--apap-regularization", type=float, default=0.05)

    # (Legacy) seam-cost flags your code already had; keep them for back-compat
    parser.add_argument("--seam-cost", type=str, default="color", choices=["color", "gradient"])
    parser.add_argument("--seam-gradient-weight", type=float, default=8.0)

    parser.add_argument("--debug-dir", type=Path, default=None)
    args = parser.parse_args()

    images = args.images
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    db = out / "db.db"
    sparse = out / "sparse"
    dense = out / "dense"
    orthomosaic_path = out / "orthomosaic_colmap.png"

    colmap = shutil.which("colmap") or "colmap"

    # Clean previous run if requested
    if args.fresh:
        if db.exists():
            db.unlink()
        if sparse.exists():
            shutil.rmtree(sparse)
        if dense.exists():
            shutil.rmtree(dense)

    # 1) Feature extraction
    run([
        colmap,
        "feature_extractor",
        f"--database_path={db}",
        f"--image_path={images}",
        f"--ImageReader.single_camera={args.single_camera}",
        "--ImageReader.camera_model=SIMPLE_RADIAL",
        "--SiftExtraction.use_gpu=1",
    ])

    # 2) Matching
    if args.exhaustive:
        run([
            colmap, "exhaustive_matcher",
            f"--database_path={db}",
            "--SiftMatching.guided_matching=1",
            "--SiftMatching.max_num_matches=50000",
        ])
    else:
        # default to sequential if not explicitly exhaustive
        run([
            colmap, "sequential_matcher",
            f"--database_path={db}",
            f"--SequentialMatching.overlap={args.overlap}",
            "--SiftMatching.use_gpu=1",
        ])

    # 3) Incremental mapping
    sparse.mkdir(exist_ok=True)
    run([
        colmap, "mapper",
        f"--database_path={db}",
        f"--image_path={images}",
        f"--output_path={sparse}",
        "--Mapper.ba_local_max_num_iterations=50",
        "--Mapper.ba_global_max_num_iterations=100",
        "--Mapper.init_min_num_inliers=60",
    ])

    model0, (num_registered_images, num_points3d) = select_best_reconstruction(sparse)
    print(f"Selected reconstruction {model0.name} with "
          f"{num_registered_images} registered images and {num_points3d} points.")

    # 4) Undistort + dense stereo
    dense.mkdir(exist_ok=True)
    run_dense_reconstruction(colmap, images, model0, dense)

    # 5) Orthorectify & mosaic
    undist_sparse = dense / "sparse"
    undist_images = dense / "images"
    run([
        colmap, "image_undistorter",
        f"--image_path={images}",
        f"--input_path={model0}",
        f"--output_path={dense}",
        "--output_type", "COLMAP",
        "--max_image_size=-1",
    ])

    # Delegate to orthomosaic.py; it should implement seamhybrid internally
    build_orthomosaic(
        undist_sparse,
        undist_images,
        model0,
        orthomosaic_path,
        target_max_size_px=args.max_mosaic_px,
        blend_mode=args.blend,
        num_bands=args.bands,
        feather_sharpness=args.feather_sharpness,   # NEW ➜ pass-through
        seam_scale=args.seam_scale,                 # NEW ➜ pass-through
        seam_method=args.seam_method,               # NEW ➜ pass-through
        flow_refine=not args.no_flow_refine,
        flow_method=args.flow_method,
        flow_downscale=args.flow_downscale,
        flow_max_px=args.flow_max_px,
        flow_smooth_ksize=args.flow_smooth_ksize,
        split_stripes=args.split_stripes,
        stripe_threshold=args.stripe_threshold,
        stripe_flow_max_px=args.stripe_flow_max_px,
        warp_model=args.warp_model,
        apap_cell_size=args.apap_cell_size,
        apap_sigma=args.apap_sigma,
        apap_min_weight=args.apap_min_weight,
        apap_regularization=args.apap_regularization,
        seam_cost=args.seam_cost,
        seam_gradient_weight=args.seam_gradient_weight,
        debug_dir=args.debug_dir,
    )

    print("\nDone. Outputs:")
    print(f"- Sparse model: {model0}")
    print(f"- Dense workspace: {dense}")
    print(f"- Orthomosaic: {orthomosaic_path}")
    if args.debug_dir:
        print(f"- Diagnostics: {args.debug_dir}")


if __name__ == "__main__":
    main()

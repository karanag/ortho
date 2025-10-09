import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import pycolmap
from auto_blend_optimizer import auto_optimize_blend, AutoBlendParams


def _depth_path_candidates(depth_dir: Path, image_name: str) -> List[Path]:
    stem = Path(image_name).stem
    return [depth_dir / f'{stem}.geometric.bin', depth_dir / f'{stem}.photometric.bin', depth_dir / f'{image_name}.geometric.bin', depth_dir / f'{image_name}.photometric.bin', depth_dir / f'{stem}.bin', depth_dir / f'{image_name}.bin']

def _read_colmap_depth(depth_path: Path) -> Optional[np.ndarray]:
    try:
        arr = pycolmap.read_array(str(depth_path))
        return np.asarray(arr, dtype=np.float32)
    except Exception:
        return None

def _dense_corr_from_depth(depth: np.ndarray, K: np.ndarray, R_cw: np.ndarray, t_cw: np.ndarray, n: np.ndarray, d_plane: float, X0: np.ndarray, u_axis: np.ndarray, v_axis: np.ndarray, step: int=8, max_samples: int=250000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (plane_raw Nx2, image_uv Nx2, height Nx1) from depth samples.
    - plane_raw: [u,v] in plane coords
    - image_uv: [x,y] in image (undistorted) pixels
    - height: signed distance to plane (n·X + d)
    """
    H, W = depth.shape
    ys = np.arange(0, H, step, dtype=np.int32)
    xs = np.arange(0, W, step, dtype=np.int32)
    Xg, Yg = np.meshgrid(xs, ys)
    us = Xg.ravel()
    vs = Yg.ravel()
    ds = depth[vs, us].reshape(-1)
    valid = np.isfinite(ds) & (ds > 0)
    if not np.any(valid):
        return (np.zeros((0, 2), dtype=float), np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float))
    us = us[valid].astype(np.float32)
    vs = vs[valid].astype(np.float32)
    ds = ds[valid].astype(np.float32)
    Kinv = np.linalg.inv(K)
    pix_h = np.stack([us, vs, np.ones_like(us)], axis=0)
    rays = Kinv @ pix_h
    X_cam = rays * ds
    R_wc = R_cw.T
    t_wc = -R_wc @ t_cw
    X_w = (R_wc @ X_cam).T + t_wc[None, :]
    height = X_w @ n + d_plane
    X_proj = X_w - height[:, None] * n[None, :]
    uv = X_proj - X0[None, :]
    u_vals = uv @ u_axis
    v_vals = uv @ v_axis
    N = u_vals.shape[0]
    if N > max_samples:
        idx = np.random.default_rng(0).choice(N, size=max_samples, replace=False)
        u_vals = u_vals[idx]
        v_vals = v_vals[idx]
        us = us[idx]
        vs = vs[idx]
        height = height[idx]
    plane_raw = np.stack([u_vals, v_vals], axis=1).astype(np.float32)
    img_uv = np.stack([us, vs], axis=1).astype(np.float32)
    return (plane_raw, img_uv, height.astype(np.float32))

def run_dense_reconstruction(colmap_bin: str, image_dir: Path, sfm_dir: Path, dense_dir: Path):
    undist_cmd = [colmap_bin, 'image_undistorter', f'--image_path={image_dir}', f'--input_path={sfm_dir}', f'--output_path={dense_dir}', '--max_image_size=4000']
    print('Running:', ' '.join(undist_cmd))
    os.system(' '.join(undist_cmd))
    pm_cmd = [colmap_bin, 'patch_match_stereo', f'--workspace_path={dense_dir}', f'--workspace_format=COLMAP', f'--PatchMatchStereo.geom_consistency=true']
    print('Running:', ' '.join(pm_cmd))
    os.system(' '.join(pm_cmd))
    fuse_cmd = [colmap_bin, 'stereo_fusion', f'--workspace_path={dense_dir}', f'--workspace_format=COLMAP', f'--input_type=geometric', f"--output_path={dense_dir / 'fused.ply'}"]
    print('Running:', ' '.join(fuse_cmd))
    os.system(' '.join(fuse_cmd))

def fit_plane_ransac(X: np.ndarray, n_iter: int=2000, thresh: Optional[float]=None):
    N = X.shape[0]
    if N < 3:
        raise ValueError('Not enough 3D points to fit a plane.')
    if thresh is None:
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        diag = np.linalg.norm(maxs - mins)
        thresh = max(diag * 0.01, 1e-06)
    best_inliers = None
    rng = np.random.default_rng(42)
    best_model = None
    for _ in range(n_iter):
        idx = rng.choice(N, size=3, replace=False)
        p1, p2, p3 = X[idx]
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-09:
            continue
        n = n / norm
        d = -np.dot(n, p1)
        dist = np.abs(X @ n + d)
        inliers = dist < thresh
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_model = (n, d)
    if best_inliers is None or best_inliers.sum() < 3:
        raise RuntimeError('Plane RANSAC failed.')
    Xin = X[best_inliers]
    centroid = Xin.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xin - centroid, full_matrices=False)
    n_refined = Vt[-1]
    n_refined = n_refined / np.linalg.norm(n_refined)
    d_refined = -np.dot(n_refined, centroid)
    return (n_refined, d_refined, best_inliers)

def _to_np(u):
    return u.get() if isinstance(u, cv2.UMat) else u

def _ensure_binary(m):
    m = _to_np(m)
    m = cv2.threshold(m, 1, 255, cv2.THRESH_BINARY)[1]
    return m.astype(np.uint8)

def compute_seam_masks_lowres(
    images_u8,
    masks_u8,
    scale=0.3,
    seam_method='graphcut',
    seam_cost: str='color',
    seam_gradient_weight: float=0.0,
    debug_dir: Optional[Path]=None
):
    """
    images_u8: list of 8-bit BGR tiles already reprojected to a *single plane* (true ortho)
    masks_u8:  list of 8-bit 0/255 valid-region masks for each tile
    Returns: list of full-res seam masks (uint8 0/255), one per image
    """
    assert len(images_u8) == len(masks_u8) and len(images_u8) > 0
    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
    imgs_s, msks_s = ([], [])
    full_sizes: List[Tuple[int, int]] = []
    for img, m in zip(images_u8, masks_u8):
        ih, iw = img.shape[:2]
        nw, nh = (int(iw * scale), int(ih * scale))
        imgs_s.append(cv2.resize(img.astype(np.float32), (nw, nh), interpolation=cv2.INTER_LANCZOS4))
        msks_s.append(cv2.resize(_ensure_binary(m), (nw, nh), interpolation=cv2.INTER_LINEAR))
        full_sizes.append((ih, iw))
    seam_method = (seam_method or 'graphcut').lower()
    seam_cost = (seam_cost or 'color').lower()
    use_gradient = seam_cost == 'gradient' and seam_gradient_weight > 0.0
    if seam_method.startswith('dp'):
        cost_mode = 'COLOR_GRAD' if use_gradient else 'COLOR'
        seam_finder = cv2.detail_DpSeamFinder(cost_mode)
        try:
            seam_finder.setCostFunction(cost_mode)
        except Exception:
            pass
    else:
        cost_mode = 'COST_COLOR_GRAD' if use_gradient else 'COST_COLOR'
        seam_finder = cv2.detail_GraphCutSeamFinder(cost_mode)
    corners = [(0, 0)] * len(imgs_s)
    seam_masks_small = seam_finder.find(imgs_s, corners, msks_s)
    seam_masks_full = []
    for i, sm in enumerate(seam_masks_small):
        sm = _to_np(sm)
        ih, iw = full_sizes[i]
        full_mask = cv2.resize(sm, (iw, ih), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        seam_masks_full.append(_ensure_binary(full_mask))
        if debug_dir is not None:
            out_path = debug_dir / f'seam_mask_{i:02d}.png'
            cv2.imwrite(str(out_path), full_mask)
    return [_ensure_binary(m) for m in seam_masks_full]

def blend_fullres_with_masks(images_u8, seam_masks_u8, blender='multiband', bands=4, feather_sharpness=0.01, do_exposure=True):
    """
    Blend full-res ortho tiles using *fixed* seam masks.
    Returns: (mosaic_u8, mosaic_mask_u8)
    """
    assert len(images_u8) == len(seam_masks_u8) and len(images_u8) > 0
    h, w = images_u8[0].shape[:2]
    corners = [(0, 0)] * len(images_u8)
    if do_exposure:
        comp = cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_GAIN_BLOCKS)
        comp.feed(corners, [img.astype(np.int16) for img in images_u8], seam_masks_u8)
    b = cv2.detail_MultiBandBlender()
    b.setNumBands(max(6, bands))
    b.prepare((0, 0, w, h))
    for i, (img, m) in enumerate(zip(images_u8, seam_masks_u8)):
        m = _ensure_binary(m)
        if do_exposure:
            comp.apply(i, (0, 0), img, m)
        b.feed(img, m, (0, 0))
    mosaic, mos_mask = b.blend(None, None)
    mosaic = np.clip(mosaic, 0, 255).astype(np.uint8)
    mos_mask = _ensure_binary(mos_mask)
    return (mosaic, mos_mask)

def seamhybrid_ortho_blend(images_u8, masks_u8, seam_scale=0.3, seam_method='graphcut', seam_cost: str='color', seam_gradient_weight: float=0.0, blender='multiband', bands=4, feather_sharpness=0.02, do_exposure=True, debug_dir: Optional[Path]=None):
    """
    Convenience wrapper: find seams fast at low-res, upscale, then blend at full-res.
    """
    seam_masks = compute_seam_masks_lowres(
        images_u8,
        masks_u8,
        scale=seam_scale,
        seam_method=seam_method,
        seam_cost=seam_cost,
        seam_gradient_weight=seam_gradient_weight,
        debug_dir=debug_dir,
    )
    mosaic, mos_mask = blend_fullres_with_masks(images_u8, seam_masks, blender=blender, bands=bands, feather_sharpness=feather_sharpness, do_exposure=do_exposure)
    return (mosaic, mos_mask)

def camera_matrix_from_colmap(cam: pycolmap.Camera) -> np.ndarray:
    params = cam.params
    model = cam.model.name
    if model in ('PINHOLE', 'OPENCV', 'OPENCV_FISHEYE', 'FULL_OPENCV'):
        fx, fy, cx, cy = (params[0], params[1], params[2], params[3])
    elif model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL'):
        fx, fy, cx, cy = (params[0], params[0], params[1], params[2])
    elif len(params) >= 4:
        fx, fy, cx, cy = (params[0], params[1], params[2], params[3])
    else:
        raise NotImplementedError(f'Unsupported camera model: {model}')
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=float)
    return K

def _ensure_numpy_rotation(pose) -> np.ndarray:
    rot = pose.rotation
    rot = rot() if callable(rot) else rot
    if hasattr(rot, 'matrix'):
        rot = rot.matrix() if callable(rot.matrix) else rot.matrix
    return np.asarray(rot, dtype=float)

def _ensure_numpy_translation(pose) -> np.ndarray:
    trans = pose.translation
    trans = trans() if callable(trans) else trans
    return np.asarray(trans, dtype=float).reshape(3)

def homography_img_to_plane(K: np.ndarray, R_wc: np.ndarray, t_wc: np.ndarray, X0: np.ndarray, u_axis: np.ndarray, v_axis: np.ndarray) -> np.ndarray:
    Rc_u = R_wc @ u_axis
    Rc_v = R_wc @ v_axis
    Rc_X0 = R_wc @ X0 + t_wc
    H_p2i = K @ np.column_stack([Rc_u, Rc_v, Rc_X0])
    H_i2p = np.linalg.inv(H_p2i)
    return H_i2p

def build_orthomosaic(undistorted_sparse_dir: Path, undistorted_images_dir: Path, sfm_dir: Path, mosaic_path: Path, target_max_size_px: int=8000, blend_mode: str='feather', num_bands: int=6, flow_refine: bool=True, flow_method: str='farneback_slow', flow_downscale: float=1.0, flow_max_px: float=2.5, flow_smooth_ksize: int=13, split_stripes: bool=False, stripe_threshold: float=0.5, stripe_flow_max_px: float=6.0, warp_model: str='homography', apap_cell_size: int=80, apap_sigma: float=120.0, apap_min_weight: float=0.0001, apap_regularization: float=0.05, seam_cost: str='color', seam_gradient_weight: float=8.0, debug_dir: Optional[Path]=None, feather_sharpness: float=0.02, seam_scale: float=0.3, seam_method: str='graphcut', remove_background: bool=False, background_token_file: Optional[Path]=None, background_retries: int=2, background_retry_delay: float=2.0, color_harmonize: bool=False, harmonize_output_dir: Optional[Path]=None,auto_blend: bool = False):
    rec = pycolmap.Reconstruction(str(sfm_dir))
    pts = [p.xyz for _, p in rec.points3D.items()]
    X = np.array(pts, dtype=float)
    if X.shape[0] < 50:
        raise RuntimeError('Not enough 3D points for plane fitting.')
    n, d, inliers = fit_plane_ransac(X)
    X0 = X[inliers].mean(axis=0)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u_axis = np.cross(n, a)
    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(n, u_axis)
    v_axis /= np.linalg.norm(v_axis)
    heights = X @ n + d
    abs_heights = np.abs(heights)
    inlier_ratio = float(inliers.sum() / len(inliers))
    mean_abs_height = float(abs_heights.mean())
    p95_abs_height = float(np.percentile(abs_heights, 95))
    max_abs_height = float(abs_heights.max())
    print('Plane fit:', f'pts={X.shape[0]}', f'inliers={int(inliers.sum())} ({inlier_ratio:.1%})', f'|height| mean={mean_abs_height:.4f}', f'p95={p95_abs_height:.4f}', f'max={max_abs_height:.4f}')
    plane_stats: Dict[str, object] = {}
    if debug_dir is not None:
        debug_dir = debug_dir.expanduser().resolve()
        debug_dir.mkdir(parents=True, exist_ok=True)
        plane_stats = {'num_points3D': int(X.shape[0]), 'plane_normal': n.tolist(), 'plane_offset': float(d), 'inliers': int(inliers.sum()), 'inlier_ratio': float(inliers.sum() / len(inliers)), 'height_stats': {'mean_abs': mean_abs_height, 'p95_abs': p95_abs_height, 'max_abs': max_abs_height}}
        point_errors = np.array([p.error for _, p in rec.points3D.items()], dtype=float)
        if point_errors.size:
            plane_stats['reprojection_error_px'] = {'mean': float(point_errors.mean()), 'median': float(np.median(point_errors)), 'p90': float(np.percentile(point_errors, 90)), 'max': float(point_errors.max())}
        plane_heights = {pid: float(np.dot(p.xyz, n) + d) for pid, p in rec.points3D.items()}
        per_image_stats: List[Dict[str, object]] = []
        for img in rec.images.values():
            residuals = [abs(plane_heights[p.point3D_id]) for p in img.points2D if p.has_point3D()]
            pose = img.cam_from_world
            pose = pose() if callable(pose) else pose
            R_wc = _ensure_numpy_rotation(pose)
            t_wc = _ensure_numpy_translation(pose)
            cam_center = -R_wc.T @ t_wc
            dist = abs(np.dot(n, cam_center) + d)
            if residuals:
                res_arr = np.array(residuals, dtype=float)
                per_image_stats.append({'image': img.name, 'points3D': int(res_arr.size), 'plane_residual_mean': float(res_arr.mean()), 'plane_residual_p95': float(np.percentile(res_arr, 95)), 'plane_residual_max': float(res_arr.max()), 'camera_distance_to_plane': float(dist)})
            else:
                per_image_stats.append({'image': img.name, 'points3D': 0, 'plane_residual_mean': None, 'plane_residual_p95': None, 'plane_residual_max': None, 'camera_distance_to_plane': float(dist)})
        plane_stats['per_image'] = per_image_stats
    undist_rec = pycolmap.Reconstruction(str(undistorted_sparse_dir))
    corners_all, img_entries = ([], [])
    image_correspondences: Dict[str, Dict[str, np.ndarray]] = {}
    for _, img in undist_rec.images.items():
        name = img.name
        path = undistorted_images_dir / name
        if not path.exists():
            continue
        cam = undist_rec.cameras[img.camera_id]
        K = camera_matrix_from_colmap(cam)
        T = img.cam_from_world
        T = T() if callable(T) else T
        R_wc = _ensure_numpy_rotation(T)
        t_wc = _ensure_numpy_translation(T)
        H_i2p = homography_img_to_plane(K, R_wc, t_wc, X0, u_axis, v_axis)
        h = cam.height() if callable(cam.height) else cam.height
        w = cam.width() if callable(cam.width) else cam.width
        corners = np.array([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]], dtype=float).T
        plane = H_i2p @ corners
        plane /= plane[2:3, :]
        plane_xy = plane[:2, :].T
        corners_all.append(plane_xy)
        x_min = float(plane_xy[:, 0].min())
        plane_center = plane_xy.mean(axis=0)
        img_entries.append({'name': name, 'path': path, 'H': H_i2p, 'x_min': x_min, 'plane_xy': plane_xy, 'plane_center': plane_center})
        corr_plane: List[List[float]] = []
        corr_image: List[List[float]] = []
        corr_height: List[float] = []
        for p2d in img.points2D:
            if not p2d.has_point3D():
                continue
            if p2d.point3D_id not in undist_rec.points3D:
                continue
            pt3d = undist_rec.points3D[p2d.point3D_id]
            X3d = np.asarray(pt3d.xyz, dtype=float)
            height = float(np.dot(X3d, n) + d)
            X_proj = X3d - height * n
            uv_vec = X_proj - X0
            u_val = float(np.dot(uv_vec, u_axis))
            v_val = float(np.dot(uv_vec, v_axis))
            if hasattr(p2d, 'xy'):
                xy = np.asarray(p2d.xy, dtype=float)
            else:
                xy = np.array([float(p2d.x), float(p2d.y)], dtype=float)
            corr_plane.append([u_val, v_val])
            corr_image.append([float(xy[0]), float(xy[1])])
            corr_height.append(height)
        image_correspondences[name] = {'plane_raw': np.array(corr_plane, dtype=float) if corr_plane else np.zeros((0, 2), dtype=float), 'image': np.array(corr_image, dtype=float) if corr_image else np.zeros((0, 2), dtype=float), 'height': np.array(corr_height, dtype=float) if corr_height else np.zeros((0,), dtype=float)}
    dense_root = undistorted_sparse_dir.parent / 'stereo' / 'depth_maps'
    dense_upgraded = 0
    if dense_root.exists():
        for entry in img_entries:
            name = entry['name']
            path = entry['path']
            if not path.exists():
                continue
            depth_path = None
            for cand in _depth_path_candidates(dense_root, name):
                if cand.exists():
                    depth_path = cand
                    break
            if depth_path is None:
                continue
            depth = _read_colmap_depth(depth_path)
            if depth is None or depth.size == 0:
                continue
            img_u = undist_rec.images[name] if name in undist_rec.images else None
            if img_u is None:
                continue
            cam = undist_rec.cameras[img_u.camera_id]
            K = camera_matrix_from_colmap(cam)
            T = img_u.cam_from_world
            T = T() if callable(T) else T
            R_cw = _ensure_numpy_rotation(T)
            t_cw = _ensure_numpy_translation(T)
            plane_raw, img_uv, heights = _dense_corr_from_depth(depth, K, R_cw, t_cw, n, d, X0, u_axis, v_axis, step=8, max_samples=250000)
            if plane_raw.shape[0] < 4:
                continue
            image_correspondences[name] = {'plane_raw': plane_raw, 'image': img_uv, 'height': heights}
            dense_upgraded += 1
    if dense_upgraded > 0:
        print(f'Dense depth available: upgraded correspondences for {dense_upgraded}/{len(img_entries)} images.')
    else:
        print('Dense depth not found or unreadable; using sparse correspondences.')
    if not corners_all:
        raise RuntimeError('No undistorted images found for mosaic.')
    all_xy = np.vstack(corners_all)
    min_xy = all_xy.min(axis=0)
    max_xy = all_xy.max(axis=0)
    span = max_xy - min_xy
    scale = target_max_size_px / max(span[0], span[1]) if max(span) > 0 else 1.0
    out_w = max(int(np.ceil(span[0] * scale)), 8)
    out_h = max(int(np.ceil(span[1] * scale)), 8)
    print(f'Orthomosaic size: {out_w}x{out_h} (scale={scale:.3f})')
    offset_T = np.array([[1, 0, -min_xy[0]], [0, 1, -min_xy[1]], [0, 0, 1]], dtype=float)
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=float)
    plane_to_mosaic = S @ offset_T
    mosaic_to_plane = np.linalg.inv(plane_to_mosaic)
    for entry in img_entries:
        name = entry['name']
        if name not in image_correspondences:
            continue
        corr = image_correspondences[name]
        plane_raw = corr.get('plane_raw')
        if plane_raw is not None and plane_raw.size:
            plane_h = np.column_stack([plane_raw, np.ones(len(plane_raw), dtype=float)])
            mosaic_pts_h = (plane_to_mosaic @ plane_h.T).T
            mosaic_pts = mosaic_pts_h[:, :2] / mosaic_pts_h[:, 2:3]
            corr['plane_mosaic'] = mosaic_pts.astype(float)
        else:
            corr['plane_mosaic'] = np.zeros((0, 2), dtype=float)
    img_entries.sort(key=lambda e: e['x_min'])
    use_multiband = blend_mode == 'multiband'
    flow_downscale = float(max(flow_downscale, 0.001))
    blender = None



    if use_multiband:
        try:
            blender = cv2.detail_MultiBandBlender()
            blender.setNumBands(max(1, int(num_bands)))
            blender.prepare((0, 0, out_w, out_h))
        except Exception as e:
            print(f'MultiBand blender unavailable ({e}); falling back to feather.')
            blender = None
            use_multiband = False
    warp_model = (warp_model or 'homography').lower()
    seam_cost = (seam_cost or 'color').lower()
    apap_cell_size = max(8, int(apap_cell_size))
    apap_sigma = max(1.0, float(apap_sigma))
    apap_min_weight = max(0.0, float(apap_min_weight))
    apap_regularization = max(0.0, float(apap_regularization))
    seam_gradient_weight = max(0.0, float(seam_gradient_weight))

    def _apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 1:
            pts = pts[None, :]
        ones = np.ones((pts.shape[0], 1), dtype=float)
        pts_h = np.hstack([pts, ones])
        proj = (H @ pts_h.T).T
        denom = proj[:, 2]
        safe = np.abs(denom) > 1e-12
        out = np.zeros((pts.shape[0], 2), dtype=float)
        if np.any(safe):
            out[safe] = proj[safe, :2] / denom[safe, None]
        if np.any(~safe):
            out[~safe] = proj[~safe, :2]
        return out

    def _solve_weighted_homography(src_pts: np.ndarray, dst_pts: np.ndarray, weights: np.ndarray) -> Optional[np.ndarray]:
        src_pts = np.asarray(src_pts, dtype=float)
        dst_pts = np.asarray(dst_pts, dtype=float)
        weights = np.asarray(weights, dtype=float)
        valid = weights > 1e-12
        if valid.sum() < 4:
            return None
        src_pts = src_pts[valid]
        dst_pts = dst_pts[valid]
        weights = weights[valid]
        A_rows: List[List[float]] = []
        for (sx, sy), (dx, dy), w in zip(src_pts, dst_pts, weights):
            w_sqrt = math.sqrt(float(max(w, 1e-12)))
            A_rows.append([-sx * w_sqrt, -sy * w_sqrt, -1.0 * w_sqrt, 0.0, 0.0, 0.0, dx * sx * w_sqrt, dx * sy * w_sqrt, dx * w_sqrt])
            A_rows.append([0.0, 0.0, 0.0, -sx * w_sqrt, -sy * w_sqrt, -1.0 * w_sqrt, dy * sx * w_sqrt, dy * sy * w_sqrt, dy * w_sqrt])
        A = np.asarray(A_rows, dtype=float)
        if A.shape[0] < 8:
            return None
        try:
            _, _, vt = np.linalg.svd(A)
        except np.linalg.LinAlgError:
            return None
        h = vt[-1]
        H = h.reshape(3, 3)
        if abs(H[2, 2]) > 1e-08:
            H = H / H[2, 2]
        return H

    def _compute_roi_from_homography(H_img_to_mosaic: np.ndarray, img_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int, np.ndarray]]:
        h_img, w_img = img_shape
        corners = np.array([[0, 0], [w_img - 1, 0], [w_img - 1, h_img - 1], [0, h_img - 1]], dtype=float)
        mosaic_corners = _apply_homography(H_img_to_mosaic, corners)
        if not np.isfinite(mosaic_corners).all():
            return None
        x_min = int(math.floor(mosaic_corners[:, 0].min()) - 3)
        x_max = int(math.ceil(mosaic_corners[:, 0].max()) + 3)
        y_min = int(math.floor(mosaic_corners[:, 1].min()) - 3)
        y_max = int(math.ceil(mosaic_corners[:, 1].max()) + 3)
        x_min = max(0, min(out_w - 1, x_min))
        y_min = max(0, min(out_h - 1, y_min))
        x_max = min(out_w, max(0, x_max))
        y_max = min(out_h, max(0, y_max))
        if x_max - x_min < 4 or y_max - y_min < 4:
            return None
        return (x_min, y_min, x_max, y_max, mosaic_corners)

    def _apap_warp_image(image_rgb: np.ndarray, H_img_to_mosaic: np.ndarray, corr_mosaic: np.ndarray, corr_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        fallback_info: Dict[str, object] = {'mode': 'homography_fallback', 'roi': None, 'corr_total': int(corr_mosaic.shape[0]) if corr_mosaic is not None else 0}
        roi_info = _compute_roi_from_homography(H_img_to_mosaic, image_rgb.shape[:2])
        if roi_info is None:
            warped = cv2.warpPerspective(image_rgb, H_img_to_mosaic, (out_w, out_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            mask = (warped.sum(axis=2) > 0).astype(np.uint8) * 255
            fallback_info['mode'] = 'homography_invalid_roi'
            return (warped, mask, fallback_info)
        roi_x0, roi_y0, roi_x1, roi_y1, mosaic_corners = roi_info
        fallback_info['roi'] = [roi_x0, roi_y0, roi_x1, roi_y1]
        if corr_mosaic is None or corr_image is None or len(corr_mosaic) < 4:
            warped = cv2.warpPerspective(image_rgb, H_img_to_mosaic, (out_w, out_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            mask = (warped.sum(axis=2) > 0).astype(np.uint8) * 255
            fallback_info['mode'] = 'homography_no_corr'
            return (warped, mask, fallback_info)
        margin = float(apap_cell_size) * 1.5
        corr_mosaic = np.asarray(corr_mosaic, dtype=float)
        corr_image = np.asarray(corr_image, dtype=float)
        within = (corr_mosaic[:, 0] >= roi_x0 - margin) & (corr_mosaic[:, 0] <= roi_x1 + margin) & (corr_mosaic[:, 1] >= roi_y0 - margin) & (corr_mosaic[:, 1] <= roi_y1 + margin)
        corr_mosaic_local = corr_mosaic[within]
        corr_image_local = corr_image[within]
        if corr_mosaic_local.shape[0] < 4:
            warped = cv2.warpPerspective(image_rgb, H_img_to_mosaic, (out_w, out_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            mask = (warped.sum(axis=2) > 0).astype(np.uint8) * 255
            fallback_info['mode'] = 'homography_sparse_corr'
            fallback_info['corr_used'] = int(corr_mosaic_local.shape[0])
            return (warped, mask, fallback_info)
        H_mosaic_to_image = np.linalg.inv(H_img_to_mosaic)
        roi_w = roi_x1 - roi_x0
        roi_h = roi_y1 - roi_y0
        xs = np.arange(roi_x0, roi_x1, dtype=float)
        ys = np.arange(roi_y0, roi_y1, dtype=float)
        Xg, Yg = np.meshgrid(xs, ys)
        roi_pts = np.column_stack([Xg.ravel(), Yg.ravel()])
        global_map = _apply_homography(H_mosaic_to_image, roi_pts).reshape(roi_h, roi_w, 2)
        map_x = global_map[..., 0].astype(np.float32)
        map_y = global_map[..., 1].astype(np.float32)
        grid_x_vals = np.arange(roi_x0, roi_x1 + apap_cell_size, apap_cell_size, dtype=float)
        grid_y_vals = np.arange(roi_y0, roi_y1 + apap_cell_size, apap_cell_size, dtype=float)
        if grid_x_vals[-1] < roi_x1:
            grid_x_vals = np.append(grid_x_vals, float(roi_x1))
        if grid_y_vals[-1] < roi_y1:
            grid_y_vals = np.append(grid_y_vals, float(roi_y1))
        grid_h = len(grid_y_vals)
        grid_w = len(grid_x_vals)
        vertex_map_x = np.zeros((grid_h, grid_w), dtype=float)
        vertex_map_y = np.zeros((grid_h, grid_w), dtype=float)
        vertex_mask = np.zeros((grid_h, grid_w), dtype=np.uint8)
        corr_mosaic_local = corr_mosaic_local.astype(float)
        corr_image_local = corr_image_local.astype(float)
        sigma2 = apap_sigma ** 2 * 2.0
        for iy, gy in enumerate(grid_y_vals):
            for ix, gx in enumerate(grid_x_vals):
                p = np.array([gx, gy], dtype=float)
                diff = corr_mosaic_local - p
                dist2 = np.sum(diff * diff, axis=1)
                weights = np.exp(-dist2 / sigma2)
                if weights.sum() < max(apap_min_weight, 1e-06):
                    continue
                H_local_input_src = corr_mosaic_local
                H_local_input_dst = corr_image_local
                H_weights = weights.copy()
                if apap_regularization > 0.0:
                    global_xy = _apply_homography(H_mosaic_to_image, p)
                    H_local_input_src = np.vstack([H_local_input_src, p[None, :]])
                    H_local_input_dst = np.vstack([H_local_input_dst, global_xy])
                    H_weights = np.hstack([H_weights, np.array([apap_regularization], dtype=float)])
                H_local = _solve_weighted_homography(H_local_input_src, H_local_input_dst, H_weights)
                if H_local is None:
                    continue
                mapped = _apply_homography(H_local, p)
                vertex_map_x[iy, ix] = mapped[0, 0]
                vertex_map_y[iy, ix] = mapped[0, 1]
                vertex_mask[iy, ix] = 1
        apap_used = vertex_mask.sum()
        if apap_used == 0:
            warped = cv2.warpPerspective(image_rgb, H_img_to_mosaic, (out_w, out_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            mask = (warped.sum(axis=2) > 0).astype(np.uint8) * 255
            fallback_info['mode'] = 'homography_no_vertices'
            fallback_info['corr_used'] = int(corr_mosaic_local.shape[0])
            return (warped, mask, fallback_info)
        apap_mask_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)
        for iy in range(grid_h - 1):
            y0 = grid_y_vals[iy]
            y1 = grid_y_vals[iy + 1]
            if y1 <= y0 + 1e-06:
                continue
            for ix in range(grid_w - 1):
                x0 = grid_x_vals[ix]
                x1 = grid_x_vals[ix + 1]
                if x1 <= x0 + 1e-06:
                    continue
                if vertex_mask[iy, ix] == 0 or vertex_mask[iy, ix + 1] == 0 or vertex_mask[iy + 1, ix] == 0 or (vertex_mask[iy + 1, ix + 1] == 0):
                    continue
                xi0 = max(int(math.floor(x0)), roi_x0)
                xi1 = min(int(math.ceil(x1)), roi_x1)
                yi0 = max(int(math.floor(y0)), roi_y0)
                yi1 = min(int(math.ceil(y1)), roi_y1)
                if xi1 <= xi0 or yi1 <= yi0:
                    continue
                cell_dx = x1 - x0
                cell_dy = y1 - y0
                tl_x = vertex_map_x[iy, ix]
                tl_y = vertex_map_y[iy, ix]
                tr_x = vertex_map_x[iy, ix + 1]
                tr_y = vertex_map_y[iy, ix + 1]
                bl_x = vertex_map_x[iy + 1, ix]
                bl_y = vertex_map_y[iy + 1, ix]
                br_x = vertex_map_x[iy + 1, ix + 1]
                br_y = vertex_map_y[iy + 1, ix + 1]
                for yy in range(yi0, yi1):
                    ty = (yy - y0) / cell_dy
                    ty = min(max(ty, 0.0), 1.0)
                    y_index = yy - roi_y0
                    for xx in range(xi0, xi1):
                        tx = (xx - x0) / cell_dx
                        tx = min(max(tx, 0.0), 1.0)
                        x_index = xx - roi_x0
                        top_x = tl_x * (1 - tx) + tr_x * tx
                        top_y = tl_y * (1 - tx) + tr_y * tx
                        bottom_x = bl_x * (1 - tx) + br_x * tx
                        bottom_y = bl_y * (1 - tx) + br_y * tx
                        map_x[y_index, x_index] = float(top_x * (1 - ty) + bottom_x * ty)
                        map_y[y_index, x_index] = float(top_y * (1 - ty) + bottom_y * ty)
                        apap_mask_roi[y_index, x_index] = 1
        src_mask = np.ones(image_rgb.shape[:2], dtype=np.uint8) * 255
        warped_roi = cv2.remap(image_rgb, map_x, map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        mask_roi = cv2.remap(src_mask, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask_roi = _ensure_binary(mask_roi)
        warped_full = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        mask_full = np.zeros((out_h, out_w), dtype=np.uint8)
        warped_full[roi_y0:roi_y1, roi_x0:roi_x1] = warped_roi
        mask_full[roi_y0:roi_y1, roi_x0:roi_x1] = mask_roi
        deviation = np.sqrt((map_x - global_map[..., 0].astype(np.float32)) ** 2 + (map_y - global_map[..., 1].astype(np.float32)) ** 2)
        apap_pixels = apap_mask_roi > 0
        deviation_stats = {'mean': float(deviation[apap_pixels].mean()) if apap_pixels.any() else 0.0, 'p90': float(np.percentile(deviation[apap_pixels], 90)) if apap_pixels.any() else 0.0, 'max': float(deviation[apap_pixels].max()) if apap_pixels.any() else 0.0}
        info: Dict[str, object] = {'mode': 'apap', 'roi': [roi_x0, roi_y0, roi_x1, roi_y1], 'corr_total': int(corr_mosaic.shape[0]), 'corr_used': int(corr_mosaic_local.shape[0]), 'mesh_cells': int((grid_h - 1) * (grid_w - 1)), 'apap_vertices': int(apap_used), 'apap_pixel_ratio': float(apap_mask_roi.mean()), 'deviation': deviation_stats}
        return (warped_full, mask_full, info)
    warps, masks = ([], [])
    warp_grays: List[np.ndarray] = []
    raw_masks: List[np.ndarray] = []
    image_names: List[str] = []
    plane_v_list: List[float] = []
    plane_uv_map: Dict[str, Tuple[float, float]] = {}
    apap_diagnostics: List[Dict[str, object]] = []
    seam_records: List[Dict[str, object]] = []
    gradient_seam_masks: Optional[List[np.ndarray]] = None
    for entry in img_entries:
        name = entry['name']
        path = entry['path']
        H_i2p = entry['H']
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        Hi = plane_to_mosaic @ H_i2p
        if warp_model == 'apap':
            corr = image_correspondences.get(name, {})
            corr_mosaic = np.asarray(corr.get('plane_mosaic', np.zeros((0, 2), dtype=float)))
            corr_image = np.asarray(corr.get('image', np.zeros((0, 2), dtype=float)))
            warped, mask, apap_info = _apap_warp_image(img, Hi, corr_mosaic, corr_image)
            apap_info['image'] = name
            if corr and 'height' in corr:
                heights = np.asarray(corr.get('height', np.zeros((0,), dtype=float)))
                if heights.size:
                    apap_info['height_stats'] = {'mean': float(np.mean(heights)), 'mean_abs': float(np.mean(np.abs(heights))), 'p90_abs': float(np.percentile(np.abs(heights), 90)), 'max_abs': float(np.max(np.abs(heights)))}
            apap_diagnostics.append(apap_info)
        else:
            warped = cv2.warpPerspective(img, Hi, (out_w, out_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            mask = (warped.sum(axis=2) > 0).astype(np.uint8) * 255
        if mask.max() == 0:
            continue
        try:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.erode(mask, k)
        except Exception:
            pass
        warps.append(warped)
        masks.append(mask)
        raw_masks.append(mask.copy())
        image_names.append(name)
        warp_gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
        warp_grays.append(warp_gray)
        plane_center = entry['plane_center']
        plane_uv_map[name] = (float(plane_center[0]), float(plane_center[1]))
        plane_v_list.append(float(plane_center[1]))
        if debug_dir is not None:
            warp_path = debug_dir / f'warp_{len(warps):02d}_{name}.png'
            mask_path = debug_dir / f'mask_{len(warps):02d}_{name}.png'
            cv2.imwrite(str(warp_path), cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(mask_path), mask)

    def _mask_to_alpha(mask: np.ndarray) -> np.ndarray:
        return np.where(mask > 0, 255, 0).astype(np.uint8)

    def _warp_to_bgra(rgb_img: np.ndarray, mask_img: np.ndarray) -> np.ndarray:
        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        alpha = _mask_to_alpha(mask_img)
        return np.dstack([bgr, alpha])

    def _shrink_mask(mask: np.ndarray, ksize: int=5) -> np.ndarray:
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            return cv2.erode(mask, kernel)
        except Exception:
            return mask

    final_bgra: Optional[List[np.ndarray]] = None
    harmonized_ref = -1
    background_updated = False
    if (remove_background or color_harmonize) and warps:
        base_bgra = [_warp_to_bgra(warps[idx], masks[idx]) for idx in range(len(warps))]
        if remove_background:
            try:
                from bg import remove_background_batch
            except ImportError:
                print('Background removal requested, but bg module is unavailable. Skipping background removal step.')
                if color_harmonize:
                    final_bgra = base_bgra.copy()
            else:
                valid_indices = [idx for idx, m in enumerate(masks) if int(m.sum()) > 0]
                if not valid_indices:
                    print('Background removal skipped: no valid mask coverage across warped tiles.')
                    if color_harmonize:
                        final_bgra = base_bgra.copy()
                else:
                    warps_bgr = [cv2.cvtColor(warps[idx], cv2.COLOR_RGB2BGR) for idx in valid_indices]
                    names_for_api = [f'{idx:02d}_{image_names[idx]}.png' for idx in valid_indices]
                    save_dir = (debug_dir / 'bg_removed') if debug_dir is not None else None
                    batch_kwargs = {
                        'images_bgr': warps_bgr,
                        'names': names_for_api,
                        'retries': background_retries,
                        'delay': background_retry_delay,
                        'save_dir': save_dir,
                        'verbose': True,
                    }
                    if background_token_file is not None:
                        batch_kwargs['token_file'] = str(background_token_file)
                    bg_results = remove_background_batch(**batch_kwargs)
                    final_bgra = base_bgra.copy()
                    for idx, result in zip(valid_indices, bg_results):
                        if result is None:
                            continue
                        if result.ndim == 2:
                            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGRA)
                        elif result.shape[2] == 3:
                            result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
                        final_bgra[idx] = result
                        background_updated = True
        if color_harmonize:
            try:
                from auto_tune import harmonize_images
            except ImportError:
                print('Color harmonization requested, but auto_tune module is unavailable. Skipping color correction.')
                if final_bgra is None:
                    final_bgra = base_bgra.copy()
            else:
                if final_bgra is None:
                    final_bgra = base_bgra.copy()
                harmonize_dir = harmonize_output_dir
                if harmonize_dir is None and debug_dir is not None:
                    harmonize_dir = debug_dir / 'harmonized_auto'
                names_for_harmonize = [f'{idx:02d}_{name}.png' for idx, name in enumerate(image_names)]
                final_bgra, harmonized_ref = harmonize_images(
                    final_bgra,
                    names=names_for_harmonize,
                    out_root=harmonize_dir,
                )
                if 0 <= harmonized_ref < len(image_names):
                    print(f'Harmonization reference tile: index {harmonized_ref} ({image_names[harmonized_ref]})')
        if final_bgra is None and remove_background:
            final_bgra = base_bgra.copy()
        if final_bgra is not None:
            new_warps: List[np.ndarray] = []
            new_masks: List[np.ndarray] = []
            for idx, bgra in enumerate(final_bgra):
                if bgra is None:
                    new_warps.append(warps[idx])
                    new_masks.append(raw_masks[idx])
                    continue
                if bgra.ndim == 2:
                    bgra = cv2.cvtColor(bgra, cv2.COLOR_GRAY2BGRA)
                elif bgra.shape[2] == 3:
                    bgra = cv2.cvtColor(bgra, cv2.COLOR_BGR2BGRA)
                bgr = bgra[:, :, :3]
                alpha = bgra[:, :, 3]
                if background_updated:
                    alpha_mask = _mask_to_alpha(alpha)
                    combined_mask = cv2.bitwise_and(alpha_mask, masks[idx])
                    combined_mask = _shrink_mask(combined_mask, ksize=5)
                else:
                    combined_mask = masks[idx].copy()
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                new_warps.append(rgb)
                new_masks.append(combined_mask)
            warps = new_warps
            masks = new_masks
            raw_masks = [m.copy() for m in new_masks]
            warp_grays = [cv2.cvtColor(w, cv2.COLOR_RGB2GRAY) for w in new_warps]
    if warp_model == 'apap' and apap_diagnostics:
        print('APAP warp diagnostics:')
        for rec in apap_diagnostics:
            image_label = rec.get('image', '?')
            mode = rec.get('mode', 'apap')
            corr_used = rec.get('corr_used', rec.get('corr_total', 0))
            deviation = rec.get('deviation', {})
            mean_dev = deviation.get('mean', 0.0)
            max_dev = deviation.get('max', 0.0)
            roi = rec.get('roi')
            print(f'  {image_label}: mode={mode}, corr_used={corr_used}, mean_dev={mean_dev:.3f}px, max_dev={max_dev:.3f}px, roi={roi}')
    
    # >>> AUTO-BLEND EARLY EXIT (after warps/masks are ready, before any seam/blend code) >>>
    if auto_blend and warps and masks:
        auto_dir = (debug_dir / "auto_opt") if debug_dir is not None else None
        try:
            from auto_blend_optimizer import auto_optimize_blend, AutoBlendParams
            base = AutoBlendParams(
                seam_method=seam_method,
                bands=int(num_bands),
                seam_scale=seam_scale,
                seam_cost=seam_cost,
                seam_gradient_weight=seam_gradient_weight,
                flow_enable=flow_refine,
                flow_method=flow_method,
                flow_max_px=flow_max_px,
                flow_smooth_ksize=flow_smooth_ksize,
            )
            if hasattr(AutoBlendParams, "seam_scales"):
                base.seam_scales = (0.5, 0.4)
            if hasattr(base, "candidate_timeout_s"):
                base.candidate_timeout_s = 90
            mosaic_rgb, auto_report = auto_optimize_blend(
                warps,
                masks,
                debug_dir=auto_dir,
                try_flow=False,
                base_params=base
            )
            mosaic_bgr = cv2.cvtColor(mosaic_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(mosaic_path), mosaic_bgr)
            print(f"[AUTO] Saved orthomosaic to {mosaic_path}")
            return
        except Exception as e:
            print(f"[AUTO] Optimizer failed ({e}); falling back to manual blend.")
    # <<< AUTO-BLEND EARLY EXIT <<<

    
    plane_vs = np.array(plane_v_list, dtype=float) if plane_v_list else np.empty(0, dtype=float)
    groups = np.zeros(len(plane_vs), dtype=int)
    split_mode = False
    stripe_groups: List[Dict[str, object]] = []
    group_pair_metrics_before: List[Dict[str, object]] = []
    group_alignment_record: Optional[Dict[str, object]] = None
    stripe_flow_limit = max(flow_max_px, stripe_flow_max_px)
    original_image_names = image_names.copy()
    if plane_vs.size >= 2 and split_stripes:
        v_range = float(plane_vs.max() - plane_vs.min()) if plane_vs.size else 0.0
        if v_range >= stripe_threshold or stripe_threshold <= 0:
            median_v = float(np.median(plane_vs))
            groups = np.array([0 if v <= median_v else 1 for v in plane_vs], dtype=int)
            if len(np.unique(groups)) >= 2:
                split_mode = True
                stripe_groups = [{'image': image_names[idx], 'group': int(groups[idx]), 'plane_u': float(plane_uv_map[image_names[idx]][0]), 'plane_v': float(plane_uv_map[image_names[idx]][1])} for idx in range(len(image_names))]

    def _measure_pair(gray_a: np.ndarray, gray_b: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray, name_a: str, name_b: str) -> Optional[Dict[str, object]]:
        mask_overlap = ((mask_a > 0) & (mask_b > 0)).astype(np.uint8)
        overlap_px = int(mask_overlap.sum())
        if overlap_px < 500:
            return None
        ys, xs = np.where(mask_overlap)
        y0, y1 = (ys.min(), ys.max() + 1)
        x0, x1 = (xs.min(), xs.max() + 1)
        roi_a = gray_a[y0:y1, x0:x1]
        roi_b = gray_b[y0:y1, x0:x1]
        roi_mask = mask_overlap[y0:y1, x0:x1]
        try:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            roi_mask = cv2.erode(roi_mask, k)
        except Exception:
            pass
        valid = roi_mask > 0
        if int(valid.sum()) < 100:
            return None
        ys2, xs2 = np.where(valid)
        y0b, y1b = (ys2.min(), ys2.max() + 1)
        x0b, x1b = (xs2.min(), xs2.max() + 1)
        roi_a = roi_a[y0b:y1b, x0b:x1b]
        roi_b = roi_b[y0b:y1b, x0b:x1b]
        if roi_a.size == 0 or roi_b.size == 0:
            return None
        roi_a_f = roi_a.astype(np.float32)
        roi_b_f = roi_b.astype(np.float32)
        try:
            window = cv2.createHanningWindow((roi_a.shape[1], roi_a.shape[0]), cv2.CV_32F)
            shift, response = cv2.phaseCorrelate(roi_a_f * window, roi_b_f * window)
        except Exception:
            shift, response = cv2.phaseCorrelate(roi_a_f, roi_b_f)
        shift_x = float(shift[0])
        shift_y = float(shift[1])
        mae = float(np.mean(np.abs(roi_a_f - roi_b_f)))
        return {'image_i': name_a, 'image_j': name_b, 'shift_x': shift_x, 'shift_y': shift_y, 'shift_mag': float(math.hypot(shift_x, shift_y)), 'phase_corr_response': float(response), 'mae': mae, 'overlap_px': overlap_px}

    def _collect_pair_metrics(gray_list: List[np.ndarray], mask_list: List[np.ndarray], names: List[str]) -> List[Dict[str, object]]:
        metrics: List[Dict[str, object]] = []
        for i in range(len(gray_list) - 1):
            for j in range(i + 1, len(gray_list)):
                res = _measure_pair(gray_list[i], gray_list[j], mask_list[i], mask_list[j], names[i], names[j])
                if res is not None:
                    metrics.append(res)
        return metrics

    def _compose_subset(indices: List[int], warp_list: List[np.ndarray], mask_list: List[np.ndarray], group_label: Optional[str]=None) -> Tuple[np.ndarray, np.ndarray]:
        if not indices:
            return (np.zeros((out_h, out_w, 3), dtype=np.uint8), np.zeros((out_h, out_w), dtype=np.uint8))
        if len(indices) == 1:
            idx = indices[0]
            return (warp_list[idx].astype(np.uint8), _ensure_binary(mask_list[idx]))
        subset_warps_rgb = [warp_list[i].astype(np.uint8) for i in indices]
        subset_masks = [_ensure_binary(mask_list[i]) for i in indices]
        seam_dir = None
        if debug_dir is not None and group_label is not None:
            seam_dir = debug_dir / f'{group_label}_seams_lowres'
        try:
            subset_warps_bgr = [cv2.cvtColor(w, cv2.COLOR_RGB2BGR) for w in subset_warps_rgb]
            mosaic_bgr, mos_mask = seamhybrid_ortho_blend(
                subset_warps_bgr,
                [m.copy() for m in subset_masks],
                seam_scale=seam_scale,
                seam_method=seam_method,
                seam_cost=seam_cost,
                seam_gradient_weight=seam_gradient_weight,
                blender='multiband',
                bands=3,
                feather_sharpness=feather_sharpness,
                do_exposure=False,
                debug_dir=seam_dir,
            )
            mosaic_rgb = cv2.cvtColor(mosaic_bgr, cv2.COLOR_BGR2RGB)
            return (mosaic_rgb.astype(np.uint8), _ensure_binary(mos_mask))
        except Exception as e:
            print(f'Group seam blend failed ({e}); falling back to distance-weighted average.')
        acc = np.zeros((out_h, out_w, 3), dtype=np.float64)
        wsum = np.zeros((out_h, out_w), dtype=np.float64)
        union_mask = np.zeros((out_h, out_w), dtype=np.uint8)
        for idx in indices:
            warped = warp_list[idx].astype(np.float32)
            mask = _ensure_binary(mask_list[idx])
            union_mask = cv2.bitwise_or(union_mask, mask)
            dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
            w = dist / (dist.max() + 1e-06) if dist.max() > 0 else (mask > 0).astype(np.float32)
            acc += warped * w[..., None]
            wsum += w
        wsum[wsum == 0] = 1.0
        mosaic = (acc / wsum[..., None]).clip(0, 255).astype(np.uint8)
        return (mosaic, union_mask)

    def _dp_horizontal_seam(diff: np.ndarray) -> Optional[np.ndarray]:
        h, w = diff.shape
        if h == 0 or w == 0:
            return None
        cost = diff.astype(np.float64)
        back = np.zeros((h, w), dtype=np.int16)
        for x in range(1, w):
            prev = cost[:, x - 1]
            for y in range(h):
                min_cost = prev[y]
                idx = y
                if y > 0 and prev[y - 1] < min_cost:
                    min_cost = prev[y - 1]
                    idx = y - 1
                if y + 1 < h and prev[y + 1] < min_cost:
                    min_cost = prev[y + 1]
                    idx = y + 1
                cost[y, x] += min_cost
                back[y, x] = idx
        seam = np.zeros(w, dtype=np.int32)
        seam[-1] = int(np.argmin(cost[:, -1]))
        for x in range(w - 2, -1, -1):
            seam[x] = back[seam[x + 1], x + 1]
        return seam

    def _compute_binary_seam(mask_a: np.ndarray, mask_b: np.ndarray, img_a: np.ndarray, img_b: np.ndarray, gradient_weight: float=0.0) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, float]]]:
        overlap = (mask_a > 0) & (mask_b > 0)
        if int(overlap.sum()) < 100:
            return None
        ys, xs = np.where(overlap)
        y0, y1 = (ys.min(), ys.max() + 1)
        x0, x1 = (xs.min(), xs.max() + 1)
        roi_mask = overlap[y0:y1, x0:x1]
        if roi_mask.shape[0] < 10 or roi_mask.shape[1] < 10:
            return None
        patch_a = img_a[y0:y1, x0:x1].astype(np.float32)
        patch_b = img_b[y0:y1, x0:x1].astype(np.float32)
        diff = np.linalg.norm(patch_a - patch_b, axis=2)
        cost = diff.copy()
        grad_penalty = None
        if gradient_weight > 0.0:
            gray_a = cv2.cvtColor(patch_a.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
            gray_b = cv2.cvtColor(patch_b.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
            grad_ax = cv2.Sobel(gray_a, cv2.CV_32F, 1, 0, ksize=3)
            grad_ay = cv2.Sobel(gray_a, cv2.CV_32F, 0, 1, ksize=3)
            grad_bx = cv2.Sobel(gray_b, cv2.CV_32F, 1, 0, ksize=3)
            grad_by = cv2.Sobel(gray_b, cv2.CV_32F, 0, 1, ksize=3)
            grad_a_mag = cv2.magnitude(grad_ax, grad_ay)
            grad_b_mag = cv2.magnitude(grad_bx, grad_by)
            grad_penalty = 0.5 * (grad_a_mag + grad_b_mag)
            cost = cost + gradient_weight * grad_penalty
        seam = _dp_horizontal_seam(cost)
        if seam is None:
            return None
        mask_top = mask_a.copy()
        mask_bottom = mask_b.copy()
        mask_top[y0:y1, x0:x1] = 0
        mask_bottom[y0:y1, x0:x1] = 0
        ys_a = np.where(mask_a > 0)[0]
        ys_b = np.where(mask_b > 0)[0]
        center_a = float(ys_a.mean()) if ys_a.size else (y0 + y1) / 2.0
        center_b = float(ys_b.mean()) if ys_b.size else (y0 + y1) / 2.0
        a_is_lower = center_a > center_b
        for idx, x in enumerate(range(x0, x1)):
            y_seam = seam[idx] + y0
            if a_is_lower:
                mask_bottom[y0:y_seam + 1, x] = 255
                mask_top[y_seam + 1:y1, x] = 255
            else:
                mask_top[y0:y_seam + 1, x] = 255
                mask_bottom[y_seam + 1:y1, x] = 255
        mask_top = cv2.bitwise_and(mask_top, mask_a)
        mask_bottom = cv2.bitwise_and(mask_bottom, mask_b)
        if int(mask_top.sum()) == 0 or int(mask_bottom.sum()) == 0:
            return None
        columns = np.arange(cost.shape[1])
        seam_indices = np.clip(seam, 0, cost.shape[0] - 1)
        seam_cost = float(np.mean(cost[seam_indices, columns])) if cost.size else 0.0
        gradient_along = 0.0
        if grad_penalty is not None and grad_penalty.size:
            gradient_along = float(np.mean(grad_penalty[seam_indices, columns]))
        out_diag = {'seam_cost': seam_cost, 'gradient_cost': gradient_along, 'overlap_px': int(overlap.sum()), 'roi_width': int(x1 - x0), 'roi_height': int(y1 - y0)}
        return (mask_top, mask_bottom, out_diag)

    def _log_pair_metrics(label: str, metrics: List[Dict[str, object]]):
        if not metrics:
            print(f'{label}: no overlapping pairs analysed')
            return
        shifts = np.array([float(m.get('shift_mag', 0.0)) for m in metrics], dtype=float)
        maes = np.array([float(m.get('mae', 0.0)) for m in metrics if 'mae' in m], dtype=float)
        print(f'{label}: pairs={len(metrics)}, shift_mean={np.mean(shifts):.3f}px, shift_p90={np.percentile(shifts, 90):.3f}px, shift_max={np.max(shifts):.3f}px')
        if maes.size:
            print(f'{label}: mae_mean={np.mean(maes):.3f}, mae_p90={np.percentile(maes, 90):.3f}, mae_max={np.max(maes):.3f}')
    alignment_offsets: List[Dict[str, object]] = []
    per_image_offsets: List[Dict[str, float]] = []
    pair_metrics_before: List[Dict[str, object]] = []
    if len(warps) >= 2:
        analysis_masks = [m.copy() for m in raw_masks]
        pair_metrics_before = _collect_pair_metrics(warp_grays, analysis_masks, image_names)
        _log_pair_metrics('Pair metrics before alignment', pair_metrics_before)
        if not split_mode:
            name_to_index = {name: idx for idx, name in enumerate(image_names)}
            num_imgs = len(image_names)
            max_overlap = max((metric['overlap_px'] for metric in pair_metrics_before), default=1)
            edges: List[Tuple[int, int, float, float, float, float, int]] = []
            for metric in pair_metrics_before:
                i = name_to_index[metric['image_i']]
                j = name_to_index[metric['image_j']]
                measured_shift_x = metric['shift_x']
                measured_shift_y = metric['shift_y']
                applied_x = -measured_shift_x
                applied_y = -measured_shift_y
                base_w = abs(metric['phase_corr_response'])
                overlap_frac = metric['overlap_px'] / max_overlap if max_overlap > 0 else 1.0
                weight = base_w * overlap_frac
                if abs(i - j) == 1:
                    weight *= 3.0
                weight = float(np.clip(weight, 0.05, 5.0))
                edges.append((i, j, applied_x, applied_y, weight, metric['phase_corr_response'], metric['overlap_px']))
            offsets: List[Tuple[float, float]] = [(0.0, 0.0)] * num_imgs
            if edges:
                A = np.zeros((len(edges) + 1, num_imgs), dtype=np.float64)
                b_x = np.zeros(len(edges) + 1, dtype=np.float64)
                b_y = np.zeros(len(edges) + 1, dtype=np.float64)
                for row, (i, j, applied_x, applied_y, weight, _, _) in enumerate(edges):
                    A[row, i] = -weight
                    A[row, j] = weight
                    b_x[row] = weight * applied_x
                    b_y[row] = weight * applied_y
                A[-1, 0] = 1.0
                b_x[-1] = 0.0
                b_y[-1] = 0.0
                sol_x, *_ = np.linalg.lstsq(A, b_x, rcond=None)
                sol_y, *_ = np.linalg.lstsq(A, b_y, rcond=None)
                offsets = list(zip(sol_x.tolist(), sol_y.tolist()))
            per_image_offsets = [{'image': image_names[idx], 'offset_x': offsets[idx][0], 'offset_y': offsets[idx][1]} for idx in range(num_imgs)]
            for idx, (dx, dy) in enumerate(offsets):
                if abs(dx) < 1e-06 and abs(dy) < 1e-06:
                    continue
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                warps[idx] = cv2.warpAffine(warps[idx], M, (out_w, out_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                masks[idx] = cv2.warpAffine(masks[idx], M, (out_w, out_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                raw_masks[idx] = cv2.warpAffine(raw_masks[idx], M, (out_w, out_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                warp_grays[idx] = cv2.warpAffine(warp_grays[idx], M, (out_w, out_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                masks[idx] = np.where(masks[idx] > 0, 255, 0).astype(np.uint8)
                raw_masks[idx] = np.where(raw_masks[idx] > 0, 255, 0).astype(np.uint8)
            for i, j, applied_x, applied_y, weight, response, overlap in edges:
                residual_x = offsets[j][0] - offsets[i][0] - applied_x
                residual_y = offsets[j][1] - offsets[i][1] - applied_y
                alignment_offsets.append({'image_i': image_names[i], 'image_j': image_names[j], 'measured_shift_x': -applied_x, 'measured_shift_y': -applied_y, 'applied_offset_i': offsets[i], 'applied_offset_j': offsets[j], 'residual_x': residual_x, 'residual_y': residual_y, 'residual_mag': float(math.hypot(residual_x, residual_y)), 'weight': weight, 'phase_corr_response': response, 'overlap_px': overlap})
        else:
            per_image_offsets = [{'image': name, 'offset_x': 0.0, 'offset_y': 0.0} for name in image_names]
    if alignment_offsets:
        residuals = np.array([float(entry.get('residual_mag', 0.0)) for entry in alignment_offsets], dtype=float)
        print('Alignment residuals:', f'mean={np.mean(residuals):.3f}px', f'p90={np.percentile(residuals, 90):.3f}px', f'max={np.max(residuals):.3f}px')

    def _run_seam_finder(images_u8: List[np.ndarray], masks_in: List[np.ndarray]) -> Optional[List[np.ndarray]]:
        if not images_u8:
            return None
        corners = [(0, 0)] * len(images_u8)

        def _attempt(create_fn, args):
            try:
                finder = create_fn(*args)
            except Exception:
                return None
            try:
                finder.find(images_u8, corners, masks_in)
                return [m.copy() for m in masks_in]
            except Exception:
                return None
        seam_masks = _attempt(cv2.detail_GraphCutSeamFinder, ('COST_COLOR_GRAD',))
        if seam_masks is None:
            seam_masks = _attempt(cv2.detail_DpSeamFinder, ('COLOR_GRAD',))
        return seam_masks
    seam_masks_before = None
    if seam_cost != 'gradient' and use_multiband and (blender is not None) and (len(warps) >= 2):
        imgs_8u = [w.astype(np.uint8) for w in warps]
        seam_masks_before = _run_seam_finder(imgs_8u, masks)
        if seam_masks_before is None:
            print('Seam finder unavailable; falling back to feather-only blending.')
    warps_for_final = [w.copy() for w in warps]
    masks_for_final = [m.copy() for m in raw_masks]
    if split_mode:
        source_warps = [w.copy() for w in warps_for_final]
        source_masks = [m.copy() for m in masks_for_final]
        unique_groups = sorted(set(groups.tolist()))
        group_warps: List[np.ndarray] = []
        group_masks: List[np.ndarray] = []
        group_names: List[str] = []
        for gid_idx, gid in enumerate(unique_groups):
            idxs = [i for i, g in enumerate(groups) if g == gid]
            if not idxs:
                continue
            mosaic, mask_union = _compose_subset(idxs, source_warps, source_masks, group_label=f'group{gid_idx}')
            group_warps.append(mosaic)
            group_masks.append(np.where(mask_union > 0, 255, 0).astype(np.uint8))
            group_names.append(f'group{gid_idx}')
            if debug_dir is not None:
                cv2.imwrite(str(debug_dir / f'group_{gid_idx}_mosaic.png'), cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(debug_dir / f'group_{gid_idx}_mask.png'), mask_union)
        group_grays = [cv2.cvtColor(gw, cv2.COLOR_RGB2GRAY) for gw in group_warps]
        group_pair_metrics_before = _collect_pair_metrics(group_grays, group_masks, group_names)
        if len(group_warps) == 2:
            res = _measure_pair(group_grays[0], group_grays[1], group_masks[0], group_masks[1], group_names[0], group_names[1])
            if res is not None:
                dx = -res['shift_x']
                dy = -res['shift_y']
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                group_warps[1] = cv2.warpAffine(group_warps[1], M, (out_w, out_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                group_masks[1] = cv2.warpAffine(group_masks[1], M, (out_w, out_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                group_masks[1] = np.where(group_masks[1] > 0, 255, 0).astype(np.uint8)
                group_grays[1] = cv2.warpAffine(group_grays[1], M, (out_w, out_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                group_alignment_record = {'measured_shift_x': res['shift_x'], 'measured_shift_y': res['shift_y'], 'applied_shift_x': dx, 'applied_shift_y': dy, 'shift_mag': res['shift_mag'], 'phase_corr_response': res['phase_corr_response'], 'overlap_px': res['overlap_px']}
        warps = group_warps
        masks = [m.copy() for m in group_masks]
        raw_masks = [m.copy() for m in group_masks]
        image_names = group_names
        warp_grays = [g.copy() for g in group_grays]
        if len(warps) == 2:
            seam_pair = _compute_binary_seam(group_masks[0], group_masks[1], group_warps[0], group_warps[1], gradient_weight=seam_gradient_weight if seam_cost == 'gradient' else 0.0)
            if seam_pair is not None:
                mask0, mask1, seam_diag = seam_pair
                masks = [mask0, mask1]
                raw_masks = [mask0.copy(), mask1.copy()]
                seam_diag.update({'pair': [group_names[0], group_names[1]], 'mode': 'stripe', 'cost_model': seam_cost})
                seam_records.append(seam_diag)
                if debug_dir is not None:
                    cv2.imwrite(str(debug_dir / 'stripe_seam_mask_group0.png'), masks[0])
                    cv2.imwrite(str(debug_dir / 'stripe_seam_mask_group1.png'), masks[1])
            elif use_multiband and blender is not None:
                imgs_8u = [w.astype(np.uint8) for w in warps]
                stripe_seams = _run_seam_finder(imgs_8u, masks)
                if stripe_seams is not None:
                    masks = stripe_seams
                    raw_masks = [m.copy() for m in stripe_seams]
                else:
                    print('Stripe seam finder unavailable; using feather between stripe mosaics.')
        elif use_multiband and blender is not None and (len(warps) >= 2):
            imgs_8u = [w.astype(np.uint8) for w in warps]
            stripe_seams = _run_seam_finder(imgs_8u, masks)
            if stripe_seams is not None:
                masks = stripe_seams
                raw_masks = [m.copy() for m in stripe_seams]
    if seam_cost == 'gradient' and len(warps) >= 2:
        for idx in range(len(warps) - 1):
            seam_pair = _compute_binary_seam(masks[idx], masks[idx + 1], warps[idx], warps[idx + 1], gradient_weight=seam_gradient_weight)
            if seam_pair is None:
                continue
            mask_a, mask_b, seam_diag = seam_pair
            masks[idx] = mask_a
            masks[idx + 1] = mask_b
            raw_masks[idx] = mask_a.copy()
            raw_masks[idx + 1] = mask_b.copy()
            seam_diag.update({'pair': [image_names[idx], image_names[idx + 1]], 'mode': 'adjacent', 'cost_model': seam_cost})
            seam_records.append(seam_diag)
        gradient_seam_masks = [m.copy() for m in masks]
    if seam_records:
        print(f'Seam evaluations ({seam_cost}):')
        for rec in seam_records:
            pair = rec.get('pair', ['?', '?'])
            seam_cost_val = rec.get('seam_cost', 0.0)
            grad_cost_val = rec.get('gradient_cost', 0.0)
            overlap_px = rec.get('overlap_px', 0)
            mode = rec.get('mode', 'adjacent')
            print(f'  {pair[0]} ↔ {pair[1]} ({mode}): cost={seam_cost_val:.3f}, grad_penalty={grad_cost_val:.3f}, overlap_px={overlap_px}')

    def refine_with_flow(ref_img, ref_mask, cur_img, cur_mask, method: str='farneback_slow', max_px: float=2.5, smooth_ksize: int=13):
        method = (method or 'farneback_slow').lower()
        max_px = max(float(max_px), 0.0)
        overlap = ((ref_mask > 0) & (cur_mask > 0)).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        overlap = cv2.erode(overlap, k)
        if overlap.sum() < 500:
            return cur_img
        x, y, w, h = cv2.boundingRect(overlap)
        if w < 16 or h < 16:
            return cur_img
        ref_crop = ref_img[y:y + h, x:x + w]
        cur_crop = cur_img[y:y + h, x:x + w]
        ref_gray = cv2.cvtColor(ref_crop, cv2.COLOR_RGB2GRAY)
        cur_gray = cv2.cvtColor(cur_crop, cv2.COLOR_RGB2GRAY)
        if abs(flow_downscale - 1.0) > 0.001:
            ds = max(flow_downscale, 0.001)
            size = (max(8, int(w / ds)), max(8, int(h / ds)))
            ref_small = cv2.resize(ref_gray, size, interpolation=cv2.INTER_AREA)
            cur_small = cv2.resize(cur_gray, size, interpolation=cv2.INTER_AREA)
        else:
            ref_small, cur_small = (ref_gray, cur_gray)
        if method == 'dis' and hasattr(cv2, 'DISOpticalFlow_create'):
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
            flow = dis.calc(cur_small, ref_small, None)
        elif method in ('farneback', 'farneback_slow'):
            pyr_scale = 0.5
            levels = 3 if method == 'farneback' else 5
            winsize = 21 if method == 'farneback' else 31
            iters = 5 if method == 'farneback' else 7
            poly_n = 7 if method == 'farneback' else 9
            flow = cv2.calcOpticalFlowFarneback(cur_small, ref_small, None, pyr_scale, levels, winsize, iters, poly_n, 1.5, 0)
        else:
            flow = cv2.calcOpticalFlowFarneback(cur_small, ref_small, None, 0.5, 5, 31, 7, 7, 1.5, 0)
        if abs(flow_downscale - 1.0) > 0.001:
            fx = cv2.resize(flow[..., 0], (w, h), interpolation=cv2.INTER_LINEAR) * (w / flow.shape[1])
            fy = cv2.resize(flow[..., 1], (w, h), interpolation=cv2.INTER_LINEAR) * (h / flow.shape[0])
            flow_full = np.dstack([fx, fy]).astype(np.float32)
        else:
            flow_full = flow.astype(np.float32)
        try:
            k = max(3, int(abs(smooth_ksize)) | 1)
            flow_full[..., 0] = cv2.GaussianBlur(flow_full[..., 0], (k, k), 0)
            flow_full[..., 1] = cv2.GaussianBlur(flow_full[..., 1], (k, k), 0)
        except Exception:
            pass
        mag = np.sqrt(flow_full[..., 0] ** 2 + flow_full[..., 1] ** 2)
        scale = np.ones_like(mag)
        eps = 1e-06
        scale = np.minimum(1.0, (max_px + eps) / (mag + eps))
        flow_full[..., 0] *= scale
        flow_full[..., 1] *= scale
        cur_bin = (cur_mask[y:y + h, x:x + w] > 0).astype(np.uint8)
        ref_bin = (ref_mask[y:y + h, x:x + w] > 0).astype(np.uint8)
        dt_cur = cv2.distanceTransform(cur_bin, cv2.DIST_L2, 3).astype(np.float32)
        dt_ref = cv2.distanceTransform(ref_bin, cv2.DIST_L2, 3).astype(np.float32)
        wt = dt_cur / (dt_cur + dt_ref + 1e-06)
        wt = wt[..., None]
        flow_full *= wt
        full_flow = np.zeros((cur_img.shape[0], cur_img.shape[1], 2), dtype=np.float32)
        full_flow[y:y + h, x:x + w, :] = flow_full
        yy, xx = np.mgrid[0:cur_img.shape[0], 0:cur_img.shape[1]].astype(np.float32)
        map_x = xx + full_flow[..., 0] * (cur_mask > 0)
        map_y = yy + full_flow[..., 1] * (cur_mask > 0)
        warped = cv2.remap(cur_img, map_x, map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        return warped
    if split_mode and use_multiband and (blender is not None) and (len(warps) >= 2):
        try:
            seam_finder = cv2.detail_GraphCutSeamFinder('COST_COLOR')
            corners = [(0, 0)] * len(warps)
            imgs_8u = [w.astype(np.uint8) for w in warps]
            seam_finder.find(imgs_8u, corners, masks)
        except Exception as e:
            print(f'Seam finder unavailable for stripes ({e}); proceeding without.')
    flow_max_for_refine = flow_max_px if not split_mode else stripe_flow_limit
    if flow_refine and len(warps) >= 2:
        refined = [warps[0]]
        ref_mask = masks[0]
        ref_img = warps[0]
        for i in range(1, len(warps)):
            cur_img = warps[i]
            cur_mask = masks[i]
            cur_refined = refine_with_flow(ref_img, ref_mask, cur_img, cur_mask, method=flow_method, max_px=flow_max_for_refine, smooth_ksize=flow_smooth_ksize)
            refined.append(cur_refined)
        warps = refined
    pair_metrics_after: List[Dict[str, object]] = []
    if len(warps) >= 2:
        warp_grays = [cv2.cvtColor(w, cv2.COLOR_RGB2GRAY) for w in warps]
        pair_metrics_after = _collect_pair_metrics(warp_grays, raw_masks, image_names)
        _log_pair_metrics('Pair metrics after refinement', pair_metrics_after)
    if debug_dir is not None and len(warps) >= 2:
        for i in range(len(warps) - 1):
            j = i + 1
            overlay = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            overlay[..., 1] = warp_grays[i]
            overlay[..., 2] = warp_grays[j]
            overlay *= (raw_masks[i] > 0)[..., None]
            overlay *= (raw_masks[j] > 0)[..., None]
            overlay_path = debug_dir / f'overlay_{i + 1:02d}_{image_names[i]}__{j + 1:02d}_{image_names[j]}.png'
            cv2.imwrite(str(overlay_path), overlay)
    if debug_dir is not None:
        diag = {'plane': plane_stats, 'pairwise_alignment_before': pair_metrics_before, 'pairwise_alignment_after': pair_metrics_after, 'alignment_offsets': alignment_offsets, 'image_offsets': per_image_offsets, 'warp_model': warp_model, 'seam_cost_model': seam_cost, 'image_plane_uv': [{'image': name, 'plane_u': float(plane_uv_map.get(name, (0.0, 0.0))[0]) if name in plane_uv_map else None, 'plane_v': float(plane_uv_map.get(name, (0.0, 0.0))[1]) if name in plane_uv_map else None} for name in plane_uv_map.keys()], 'image_order': image_names, 'original_image_order': original_image_names, 'canvas_size': [out_w, out_h], 'scale': float(scale)}
        if stripe_groups:
            diag['stripe_groups'] = stripe_groups
        if group_pair_metrics_before:
            diag['group_pair_alignment_before'] = group_pair_metrics_before
        if group_alignment_record is not None:
            diag['group_alignment'] = group_alignment_record
        if split_mode:
            diag['group_pair_alignment_after'] = pair_metrics_after
        if apap_diagnostics:
            diag['apap'] = apap_diagnostics
        if seam_records:
            diag['seam_records'] = seam_records
        if warp_model == 'apap':
            diag['apap_params'] = {'cell_size': int(apap_cell_size), 'sigma': float(apap_sigma), 'min_weight': float(apap_min_weight), 'regularization': float(apap_regularization)}
        if seam_cost == 'gradient':
            diag['seam_params'] = {'gradient_weight': float(seam_gradient_weight)}
        diag_path = debug_dir / 'diagnostics.json'
        with open(diag_path, 'w', encoding='utf-8') as fh:
            json.dump(diag, fh, indent=2)
        if seam_masks_before is not None:
            for idx, seam_mask in enumerate(seam_masks_before, start=1):
                name_idx = min(idx - 1, len(original_image_names) - 1)
                cv2.imwrite(str(debug_dir / f'seam_mask_{idx:02d}_{original_image_names[name_idx]}.png'), seam_mask)
        if gradient_seam_masks is not None:
            for idx, seam_mask in enumerate(gradient_seam_masks, start=1):
                name_idx = min(idx - 1, len(image_names) - 1)
                cv2.imwrite(str(debug_dir / f'seam_mask_gradient_{idx:02d}_{image_names[name_idx]}.png'), seam_mask)
    if blend_mode == 'seamhybrid':
        seam_debug_dir = debug_dir / 'seams_lowres' if debug_dir is not None else None
        warps_input = warps_for_final if warps_for_final else warps
        masks_input = masks_for_final if masks_for_final else raw_masks
        warps_bgr = [cv2.cvtColor(w.astype(np.uint8), cv2.COLOR_RGB2BGR) for w in warps_input]
        masks_for_seams = [_ensure_binary(m.copy()) for m in masks_input]
        bands_for_hybrid = 3
        mosaic = None
        mos_mask = None
        try:
            mosaic_bgr, mos_mask = seamhybrid_ortho_blend(
                warps_bgr,
                masks_for_seams,
                seam_scale=seam_scale,
                seam_method=seam_method,
                seam_cost=seam_cost,
                seam_gradient_weight=seam_gradient_weight,
                blender='multiband',
                bands=bands_for_hybrid,
                feather_sharpness=feather_sharpness,
                do_exposure=False,
                debug_dir=seam_debug_dir,
            )
            mosaic = cv2.cvtColor(mosaic_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f'Hybrid seam blend failed ({e}); falling back to weighted average.')
            acc = np.zeros((out_h, out_w, 3), dtype=np.float64)
            wsum = np.zeros((out_h, out_w), dtype=np.float64)
            for warped, mask in zip(warps, masks):
                dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
                w = dist / (dist.max() + 1e-06) if dist.max() > 0 else (mask > 0).astype(np.float32)
                acc += warped.astype(np.float64) * w[..., None]
                wsum += w
            wsum[wsum == 0] = 1.0
            mosaic = (acc / wsum[..., None]).clip(0, 255).astype(np.uint8)
    elif blend_mode == 'multiband':
        if blender is None:
            blender_mb = cv2.detail_MultiBandBlender()
            blender_mb.setNumBands(max(1, int(num_bands)))
            blender_mb.prepare((0, 0, out_w, out_h))
        else:
            blender_mb = blender
        active_masks = seam_masks_before if seam_masks_before is not None else masks
        try:
            comp = cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_GAIN_BLOCKS)
            comp.feed([(0, 0)] * len(warps), [w.astype(np.int16) for w in warps], [_ensure_binary(m) for m in active_masks])
            use_comp = True
        except Exception:
            use_comp = False
        for i, (img, m) in enumerate(zip(warps, active_masks)):
            img8 = img.astype(np.uint8)
            m8 = _ensure_binary(m)
            if use_comp:
                comp.apply(i, (0, 0), img8, m8)
            blender_mb.feed(img8, m8, (0, 0))
        mosaic, mos_mask = blender_mb.blend(None, None)
        mosaic = np.clip(mosaic, 0, 255).astype(np.uint8)
    elif blend_mode == 'feather':
        blender_f = cv2.detail_FeatherBlender()
        blender_f.setSharpness(float(feather_sharpness))
        blender_f.prepare((0, 0, out_w, out_h))
        try:
            comp = cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_GAIN_BLOCKS)
            comp.feed([(0, 0)] * len(warps), [w.astype(np.int16) for w in warps], [_ensure_binary(m) for m in masks])
            use_comp = True
        except Exception:
            use_comp = False
        for i, (img, m) in enumerate(zip(warps, masks)):
            img8 = img.astype(np.uint8)
            m8 = _ensure_binary(m)
            if use_comp:
                comp.apply(i, (0, 0), img8, m8)
            blender_f.feed(img8, m8, (0, 0))
        mosaic, mos_mask = blender_f.blend(None, None)
        mosaic = np.clip(mosaic, 0, 255).astype(np.uint8)
    else:
        acc = np.zeros((out_h, out_w, 3), dtype=np.float64)
        wsum = np.zeros((out_h, out_w), dtype=np.float64)
        for warped, mask in zip(warps, masks):
            dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
            w = dist / (dist.max() + 1e-06) if dist.max() > 0 else (mask > 0).astype(np.float32)
            acc += warped.astype(np.float64) * w[..., None]
            wsum += w
        wsum[wsum == 0] = 1.0
        mosaic = (acc / wsum[..., None]).clip(0, 255).astype(np.uint8)
    mosaic_bgr = cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(mosaic_path), mosaic_bgr)
    print(f'Saved orthomosaic to {mosaic_path}')

"""Notebook SMPL-X mesh preview: Motion-X++ arrays → smplx forward → pyrender (optional deps)."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.smplx_to_opensim import SMPLX_MOTION_DIM, SMPLX_SLICES

logger = logging.getLogger(__name__)


def motion_row_to_smplx_pose_arrays(
    row: np.ndarray, *, num_expression_coeffs: int
) -> dict[str, np.ndarray]:
    """Map one Motion-X++ motion row to arrays for ``smplx.SMPLX`` forward (axis-angle).

    Layout indices come from ``SMPLX_SLICES`` in ``src.smplx_to_opensim`` (single source
    of truth). Hands are split evenly along ``pose_hand`` (left then right).

    Args:
        row: Single frame, shape ``[D]`` with ``D == len(SMPLX layout)`` (see
            ``SMPLX_SLICES``), ``float32`` / ``float64``.
        num_expression_coeffs: SMPL-X model ``num_expression_coeffs`` (e.g. 10 or 50).

    Returns:
        Dict of 1D numpy arrays suitable for ``torch.as_tensor(...).unsqueeze(0)``.
    """
    row = np.asarray(row, dtype=np.float64)
    if row.shape != (SMPLX_MOTION_DIM,):
        raise ValueError(f"row must have shape [{SMPLX_MOTION_DIM}], got {row.shape}")
    sl = SMPLX_SLICES
    expr_src = row[sl["face_expr"]].astype(np.float32)
    n = int(num_expression_coeffs)
    if n <= 0:
        raise ValueError("num_expression_coeffs must be positive")
    if expr_src.shape[0] >= n:
        expression = expr_src[:n].astype(np.float64)
    else:
        expression = np.zeros(n, dtype=np.float64)
        expression[: expr_src.shape[0]] = expr_src.astype(np.float64)
    hand = row[sl["pose_hand"]]
    half = int(hand.shape[0] // 2)
    if hand.shape[0] != 2 * half:
        raise ValueError(f"pose_hand length must be even, got {hand.shape[0]}")
    return {
        "global_orient": row[sl["root_orient"]].astype(np.float64),
        "body_pose": row[sl["pose_body"]].astype(np.float64),
        "left_hand_pose": hand[:half].astype(np.float64),
        "right_hand_pose": hand[half:].astype(np.float64),
        "jaw_pose": row[sl["pose_jaw"]].astype(np.float64),
        "expression": expression,
        "transl": row[sl["trans"]].astype(np.float64),
        "betas": row[sl["betas"]].astype(np.float64),
    }


def _camera_pose_world_from_camera_gl(
    eye: np.ndarray, center: np.ndarray, world_up: np.ndarray
) -> np.ndarray:
    """4×4 world-from-camera matrix (OpenGL / pyrender: camera looks along local -Z)."""
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    world_up = np.asarray(world_up, dtype=np.float64).reshape(3)
    forward = center - eye
    fn = float(np.linalg.norm(forward))
    if fn < 1e-9:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        forward = forward / fn
    right = np.cross(world_up, forward)
    rn = float(np.linalg.norm(right))
    if rn < 1e-9:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right = right / rn
    up = np.cross(forward, right)
    pose = np.eye(4, dtype=np.float64)
    pose[0:3, 0] = right
    pose[0:3, 1] = up
    pose[0:3, 2] = -forward
    pose[0:3, 3] = eye
    if float(np.linalg.det(pose[:3, :3])) < 0:
        pose[0:3, 0] *= -1.0
    return pose


def _orbit_eye(center: np.ndarray, distance: float, elev_deg: float, azim_deg: float) -> np.ndarray:
    el = np.radians(float(elev_deg))
    az = np.radians(float(azim_deg))
    # Y-up: azimuth in XZ, elevation above XZ plane
    c = np.asarray(center, dtype=np.float64).reshape(3)
    offset = np.array(
        [
            float(np.cos(el) * np.sin(az)),
            float(np.sin(el)),
            float(np.cos(el) * np.cos(az)),
        ],
        dtype=np.float64,
    )
    return c + float(distance) * offset


def _fit_scene_bounds_vertices(
    model: Any,
    motion: np.ndarray,
    device: Any,
    frame_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Min/max vertex bounds over a subset of frames (numpy)."""
    import torch

    vmin = np.full(3, np.inf, dtype=np.float64)
    vmax = np.full(3, -np.inf, dtype=np.float64)
    n_expr = int(model.num_expression_coeffs)
    for t in frame_indices:
        row = motion[int(t)]
        kw = motion_row_to_smplx_pose_arrays(row, num_expression_coeffs=n_expr)
        batch = {k: torch.as_tensor(v, dtype=torch.float32, device=device).unsqueeze(0) for k, v in kw.items()}
        out = model(return_verts=True, return_full_pose=False, **batch)
        v = out.vertices.detach().cpu().numpy()[0]
        vmin = np.minimum(vmin, v.min(axis=0))
        vmax = np.maximum(vmax, v.max(axis=0))
    return vmin, vmax


def show_smplx_mesh_preview(smplx_motion: np.ndarray, config: dict[str, Any]) -> Any:
    """Display an ipywidgets UI with SMPL-X mesh frames rendered off-screen (pyrender).

    Requires optional dependencies: ``torch``, ``smplx``, ``pyrender``, ``trimesh``,
    and ``PIL`` (usually pulled in with matplotlib). Install mesh extras, for example:

        conda env update -f environment.yml --prune

    Set ``paths.smplx_model_folder`` to the **parent** directory that contains a
    ``smplx`` subfolder (same layout as ``smplx.create``: e.g. ``.../models`` with
    ``.../models/smplx/SMPLX_NEUTRAL.npz``).

    Args:
        smplx_motion: Array ``[T, D]`` in Motion-X++ layout (``D = SMPLX_MOTION_DIM``).
        config: Full project configuration.

    Returns:
        ``ipywidgets.VBox`` with slider and ``Image``, after ``display()`` in notebooks.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:
        raise RuntimeError("ipywidgets and IPython are required for mesh preview.") from exc
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for SMPL-X mesh preview. Install optional mesh deps, "
            "e.g. mesh deps in `environment.yml` (torch, smplx)."
        ) from exc
    try:
        import smplx
    except ImportError as exc:
        raise RuntimeError(
            "smplx is required for mesh preview. Install optional mesh deps, "
            "e.g. mesh deps in `environment.yml` (torch, smplx)."
        ) from exc
    try:
        import pyrender
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "pyrender and trimesh are required for mesh preview. Install project dependencies."
        ) from exc
    try:
        from PIL import Image as PILImage
    except ImportError as exc:
        raise RuntimeError("Pillow (PIL) is required to encode PNG frames for the notebook widget.") from exc

    if smplx_motion.ndim != 2 or smplx_motion.shape[1] != SMPLX_MOTION_DIM:
        raise ValueError(f"smplx_motion must have shape [T, {SMPLX_MOTION_DIM}]")
    paths_cfg = config.get("paths", {}) or {}
    raw_folder = paths_cfg.get("smplx_model_folder")
    if not raw_folder:
        raise RuntimeError(
            "config paths.smplx_model_folder is not set. Use the parent directory that "
            "contains a `smplx` subfolder (see smplx.create layout in the SMPL-X download)."
        )
    model_folder = Path(raw_folder).expanduser()
    if not model_folder.is_dir():
        raise RuntimeError(f"smplx_model_folder is not a directory: {model_folder}")

    vis = config.get("visualization", {}) or {}
    n_betas = int(vis.get("mesh_preview_num_betas", 10))
    n_expr = int(vis.get("mesh_preview_num_expression_coeffs", 50))
    gender = str(vis.get("mesh_preview_gender", "neutral"))
    width = int(vis.get("mesh_preview_width", 640))
    height = int(vis.get("mesh_preview_height", 480))
    yfov = float(vis.get("mesh_preview_yfov", 0.8))
    elev = float(vis.get("mesh_preview_elevation_deg", 18.0))
    azim = float(vis.get("mesh_preview_azimuth_deg", -65.0))
    pad = float(vis.get("mesh_preview_distance_pad", 1.75))
    max_fit = int(vis.get("mesh_preview_max_fit_frames", 400))
    device = torch.device(str(vis.get("mesh_preview_device", "cpu")))

    model = smplx.create(
        str(model_folder),
        model_type="smplx",
        gender=gender,
        use_face_contour=False,
        num_betas=n_betas,
        num_expression_coeffs=n_expr,
        ext="npz",
        use_pca_hands=False,
        flat_hand_mean=True,
    ).to(device)

    faces_t = model.faces
    if hasattr(faces_t, "detach"):
        faces = np.asarray(faces_t.detach().cpu().numpy(), dtype=np.int64)
    else:
        faces = np.asarray(faces_t, dtype=np.int64)
    T = int(smplx_motion.shape[0])
    if T < 1:
        raise ValueError("smplx_motion must have at least one frame")
    idx_fit = np.linspace(0, T - 1, num=min(T, max_fit), dtype=int)
    vmin, vmax = _fit_scene_bounds_vertices(model, smplx_motion, device, idx_fit)
    center = 0.5 * (vmin + vmax)
    diag = float(np.linalg.norm(vmax - vmin)) + 1e-3
    dist = diag * pad
    eye0 = _orbit_eye(center, dist, elev, azim)
    cam_pose = _camera_pose_world_from_camera_gl(eye0, center, np.array([0.0, 1.0, 0.0], dtype=np.float64))

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=float(width) / float(height))
    light = pyrender.DirectionalLight(color=np.ones(3, dtype=np.float64), intensity=4.0)
    light_pose = np.eye(4, dtype=np.float64)

    def _render_frame(ti: int) -> bytes:
        row = smplx_motion[int(ti) % T]
        kw = motion_row_to_smplx_pose_arrays(row, num_expression_coeffs=n_expr)
        batch = {k: torch.as_tensor(v, dtype=torch.float32, device=device).unsqueeze(0) for k, v in kw.items()}
        with torch.no_grad():
            out = model(return_verts=True, return_full_pose=False, **batch)
        verts = out.vertices.detach().cpu().numpy()[0]
        tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh = pyrender.Mesh.from_trimesh(tri, smooth=False)
        scene = pyrender.Scene(ambient_light=(0.35, 0.35, 0.35), bg_color=[255, 255, 255, 255])
        scene.add(mesh)
        scene.add(camera, pose=cam_pose)
        scene.add(light, pose=light_pose)
        color, _depth = renderer.render(scene)
        buf = io.BytesIO()
        PILImage.fromarray(color[:, :, :3]).save(buf, format="PNG")
        return buf.getvalue()

    img = widgets.Image(value=_render_frame(0), format="png")
    title = widgets.Label(value="frame 0")
    slider = widgets.IntSlider(value=0, min=0, max=T - 1, step=1, description="Frame")

    def _on_change(change: dict[str, Any]) -> None:
        ti = int(change.get("new", 0))
        img.value = _render_frame(ti)
        title.value = f"frame {ti}"

    slider.observe(_on_change, names="value")
    ui = widgets.VBox([title, img, slider])
    display(ui)
    logger.info(
        "SMPL-X mesh preview displayed (%s frames, model_folder=%s).",
        T,
        model_folder,
    )
    return ui

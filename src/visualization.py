"""Activation plots and interactive motion previews."""

from __future__ import annotations

import logging
from typing import Any, Optional

import matplotlib.animation as animation
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.transform import Rotation as R

from src.smplx_to_opensim import SMPLX_MOTION_DIM, SMPLX_SLICES

logger = logging.getLogger(__name__)

# SMPL 24-joint parent indices (pelvis at 0).
_SMPL_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=int,
)

# Approximate child offsets in parent-local rest frame (meters), shape [24, 3].
_SMPL_OFFSETS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.07, -0.08, 0.0],
        [-0.07, -0.08, 0.0],
        [0.0, 0.10, 0.0],
        [0.0, -0.38, 0.0],
        [0.0, -0.38, 0.0],
        [0.0, 0.12, 0.0],
        [0.0, -0.40, 0.0],
        [0.0, -0.40, 0.0],
        [0.0, 0.15, 0.0],
        [0.0, -0.05, 0.10],
        [0.0, -0.05, 0.10],
        [0.0, 0.20, 0.0],
        [0.10, 0.12, 0.0],
        [-0.10, 0.12, 0.0],
        [0.0, 0.12, 0.0],
        [0.15, 0.0, 0.0],
        [-0.15, 0.0, 0.0],
        [0.0, -0.28, 0.0],
        [0.0, -0.28, 0.0],
        [0.0, -0.25, 0.0],
        [0.0, -0.25, 0.0],
        [0.0, -0.10, 0.0],
        [0.0, -0.10, 0.0],
    ],
    dtype=np.float64,
)

# (SMPL parent joint index, child joint index) → muscle name substrings for segment coloring.
# Substring match only (works across Rajagopal / model variants).
_SEGMENT_TO_MUSCLE_SUBSTRINGS: dict[tuple[int, int], list[str]] = {
    (1, 4): [
        "glut",
        "psoas",
        "iliacus",
        "rect_fem_r",
        "vas_",
        "semimem_r",
        "semiten_r",
        "bflh_r",
        "bfsh_r",
        "grac_r",
        "sart_r",
        "tfl_r",
        "add_",
    ],
    (2, 5): [
        "glut",
        "psoas",
        "iliacus",
        "rect_fem_l",
        "vas_",
        "semimem_l",
        "semiten_l",
        "bflh_l",
        "bfsh_l",
        "grac_l",
        "sart_l",
        "tfl_l",
        "add_",
    ],
    (4, 7): [
        "gas_med_r",
        "gas_lat_r",
        "soleus_r",
        "tib_ant_r",
        "tib_post_r",
        "per_brev_r",
        "per_long_r",
    ],
    (5, 8): [
        "gas_med_l",
        "gas_lat_l",
        "soleus_l",
        "tib_ant_l",
        "tib_post_l",
        "per_brev_l",
        "per_long_l",
    ],
    (7, 10): [
        "gas_med_r",
        "gas_lat_r",
        "soleus_r",
        "tib_ant_r",
        "tib_post_r",
        "per_brev_r",
        "per_long_r",
    ],
    (8, 11): [
        "gas_med_l",
        "gas_lat_l",
        "soleus_l",
        "tib_ant_l",
        "tib_post_l",
        "per_brev_l",
        "per_long_l",
    ],
    (0, 3): ["lumbar", "psoas", "iliacus", "erec_sp", "mult", "rect_abd"],
    (3, 6): ["erec_sp", "mult"],
    (9, 16): [
        "delt",
        "supraspin",
        "infraspin",
        "teres",
        "subscap",
        "pect_maj",
        "bic_brev_r",
        "bic_long_r",
        "tric_r",
    ],
    (9, 17): [
        "delt",
        "supraspin",
        "infraspin",
        "teres",
        "subscap",
        "pect_maj",
        "bic_brev_l",
        "bic_long_l",
        "tric_l",
    ],
    (16, 18): ["bic_brev_r", "bic_long_r", "tric_r", "pron_teres_r"],
    (17, 19): ["bic_brev_l", "bic_long_l", "tric_l", "pron_teres_l"],
}


def _mean_act_for_segment(
    parent: int,
    child: int,
    frame_idx: int,
    activations: np.ndarray,
    muscle_names: list[str],
) -> float:
    """Mean activation over muscles associated with a skeleton segment, else 0."""
    substrings = _SEGMENT_TO_MUSCLE_SUBSTRINGS.get((parent, child), [])
    if not substrings:
        return 0.0
    idxs = [i for i, n in enumerate(muscle_names) if any(s in n for s in substrings)]
    if not idxs:
        return 0.0
    mid = int(np.clip(frame_idx, 0, activations.shape[0] - 1))
    return float(np.mean(activations[mid, idxs]))


def get_smplx_skeleton_joints(
    smplx_frame: np.ndarray,
    align_rotation: Optional[R] = None,
) -> np.ndarray:
    """Approximate SMPL-24 joint positions from a single SMPL-X frame.

    Args:
        smplx_frame: Array of shape ``[D]`` in Motion-X++ layout (``D`` =
            ``SMPLX_MOTION_DIM``).
        align_rotation: If set (e.g. ``R_align`` from ``smplx_to_mot``), applied to
            root orientation and translation before FK to match OpenSim alignment.

    Returns:
        Array ``[24, 3]`` joint positions in meters (approximate FK).
    """
    if smplx_frame.shape != (SMPLX_MOTION_DIM,):
        raise ValueError(f"smplx_frame must have shape [{SMPLX_MOTION_DIM}]")
    sl = SMPLX_SLICES
    root_aa = smplx_frame[sl["root_orient"]].astype(np.float64, copy=False)
    body = smplx_frame[sl["pose_body"]].reshape(21, 3)
    trans = smplx_frame[sl["trans"]].astype(np.float64, copy=False)

    if align_rotation is not None:
        r_root = align_rotation * R.from_rotvec(root_aa)
        root_aa = r_root.as_rotvec()
        trans = np.asarray(align_rotation.apply(trans), dtype=np.float64)

    rotvec = np.zeros((24, 3), dtype=np.float64)
    rotvec[0] = root_aa
    rotvec[1:22] = body
    rotvec[22] = rotvec[20]
    rotvec[23] = rotvec[21]

    global_R: list[np.ndarray] = []
    for i in range(24):
        r_local = R.from_rotvec(rotvec[i]).as_matrix()
        p = int(_SMPL_PARENTS[i])
        if p < 0:
            global_R.append(r_local)
        else:
            global_R.append(global_R[p] @ r_local)

    joints = np.zeros((24, 3), dtype=np.float64)
    joints[0] = trans
    for i in range(1, 24):
        p = int(_SMPL_PARENTS[i])
        off = _SMPL_OFFSETS[i]
        joints[i] = joints[p] + global_R[p] @ off
    return joints.astype(np.float32)


def _skeleton_bounds_over_frames(
    frames: np.ndarray, align_rotation: Optional[R] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Min/max world coordinates over all frames (for stable 3D axis limits)."""
    lo = np.full(3, np.inf, dtype=np.float64)
    hi = np.full(3, -np.inf, dtype=np.float64)
    for t in range(int(frames.shape[0])):
        j = get_smplx_skeleton_joints(frames[t], align_rotation=align_rotation)
        lo = np.minimum(lo, j.min(axis=0).astype(np.float64))
        hi = np.maximum(hi, j.max(axis=0).astype(np.float64))
    return lo, hi


def _cubic_skeleton_bounds(
    lo: np.ndarray,
    hi: np.ndarray,
    pad_ratio: float,
    min_half_extent: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand min/max bounds to a padded cube for stable 3D scaling."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    center = (lo + hi) / 2.0
    max_half = float(np.max(hi - lo)) / 2.0
    max_half = max(max_half, float(min_half_extent))
    half = max_half * (1.0 + 2.0 * float(pad_ratio))
    lo_b = center - half
    hi_b = center + half
    return lo_b, hi_b


def plot_activation_topk(
    activations: np.ndarray, muscle_names: list[str], k: int
) -> matplotlib.figure.Figure:
    """Plot variance bar chart and time series for the top-``k`` muscles.

    Args:
        activations: Array ``[T, N]``.
        muscle_names: Names length ``N``.
        k: Number of muscles to highlight.

    Returns:
        Matplotlib figure object (no ``plt.show()``).
    """
    if activations.ndim != 2:
        raise ValueError("activations must be 2D")
    var = np.var(activations, axis=0)
    k = min(k, activations.shape[1])
    idx = np.argsort(-var)[:k]
    names_k = [muscle_names[i] for i in idx]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    ax_top.barh(names_k[::-1], var[idx][::-1], color="steelblue")
    ax_top.set_title("Top-k muscle activation variance")
    ax_top.set_xlabel("Variance")

    cmap = plt.get_cmap("tab10")
    t = np.arange(activations.shape[0])
    for rank, i in enumerate(idx):
        ax_bot.plot(t, activations[:, i], label=muscle_names[i], color=cmap(rank % 10))
    ax_bot.set_xlabel("Frame")
    ax_bot.set_ylabel("Activation")
    ax_bot.legend(loc="upper right", fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_bot, fraction=0.046, pad=0.04)
    cbar.set_label("Muscle Activation [0-1]")
    return fig


# Mapping from OpenSim coordinate name -> (SMPL joint index, axis index).
_OPENSIM_COORD_TO_SMPL: dict[str, tuple[int, int]] = {
    "pelvis_tilt": (0, 0),
    "pelvis_list": (0, 1),
    "pelvis_rotation": (0, 2),
    "lumbar_extension": (3, 0),
    "lumbar_bending": (3, 1),
    "lumbar_rotation": (3, 2),
    "hip_flexion_r": (1, 0),
    "hip_adduction_r": (1, 1),
    "hip_rotation_r": (1, 2),
    "hip_flexion_l": (2, 0),
    "hip_adduction_l": (2, 1),
    "hip_rotation_l": (2, 2),
    "knee_angle_r": (4, 0),
    "knee_angle_l": (5, 0),
    "ankle_angle_r": (7, 0),
    "ankle_angle_l": (8, 0),
    "arm_flex_r": (16, 0),
    "arm_add_r": (16, 1),
    "arm_rot_r": (16, 2),
    "arm_flex_l": (17, 0),
    "arm_add_l": (17, 1),
    "arm_rot_l": (17, 2),
    "elbow_flex_r": (18, 0),
    "elbow_flex_l": (19, 0),
}


def coords_to_skeleton_joints(
    coords_frame: dict[str, float],
    pelvis_translation: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct approximate SMPL-24 joints from one OpenSim coordinate frame."""
    rotvec = np.zeros((24, 3), dtype=np.float64)
    for name, val in coords_frame.items():
        mapping = _OPENSIM_COORD_TO_SMPL.get(name)
        if mapping is None:
            continue
        joint_idx, axis_idx = mapping
        rotvec[joint_idx, axis_idx] += float(val)

    rotvec[22] = rotvec[20]
    rotvec[23] = rotvec[21]

    global_R: list[np.ndarray] = []
    for i in range(24):
        r_local = R.from_rotvec(rotvec[i]).as_matrix()
        p = int(_SMPL_PARENTS[i])
        global_R.append(r_local if p < 0 else global_R[p] @ r_local)

    if pelvis_translation is None:
        pelvis_translation = np.array([0.0, 0.9, 0.0], dtype=np.float64)

    joints = np.zeros((24, 3), dtype=np.float64)
    joints[0] = np.asarray(pelvis_translation, dtype=np.float64)
    for i in range(1, 24):
        p = int(_SMPL_PARENTS[i])
        joints[i] = joints[p] + global_R[p] @ _SMPL_OFFSETS[i]
    return joints.astype(np.float32)


def load_mot_coords(mot_path: "str | Path") -> tuple[list[dict[str, float]], list[float]]:
    """Parse an OpenSim Storage .mot file into per-frame coordinate dicts."""
    from pathlib import Path as _Path

    p = _Path(mot_path)
    if not p.exists():
        raise FileNotFoundError(f".mot file not found: {p}")

    lines = p.read_text(encoding="utf-8").splitlines()
    header_end = next(
        (i for i, ln in enumerate(lines) if ln.strip().lower() == "endheader"),
        None,
    )
    if header_end is None:
        raise ValueError(f"No 'endheader' found in {p}")
    if header_end + 1 >= len(lines):
        raise ValueError(f"Missing columns row after header in {p}")

    col_names = lines[header_end + 1].strip().split("\t")
    if not col_names or col_names[0].lower() != "time":
        got = col_names[0] if col_names else ""
        raise ValueError(f"Expected first column 'time', got '{got}'")

    frames: list[dict[str, float]] = []
    times: list[float] = []
    for ln in lines[header_end + 2 :]:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split("\t")
        if len(parts) != len(col_names):
            continue
        vals = [float(v) for v in parts]
        times.append(float(vals[0]))
        frames.append({col_names[j]: float(vals[j]) for j in range(1, len(col_names))})
    return frames, times


def animate_motion_interactive(
    mot_path: "str | Path",
    activations: np.ndarray,
    muscle_names: list[str],
    config: dict[str, Any],
) -> animation.FuncAnimation:
    """Interactive animation driven by OpenSim .mot coordinates + activations."""
    try:
        import ipywidgets as widgets
        from IPython.display import HTML, display
    except ImportError as exc:
        raise RuntimeError("ipywidgets/IPython required for interactive animation.") from exc

    vis = config.get("visualization", {}) or {}
    frame_interval_ms = int(vis.get("frame_interval_ms", 33))
    fig_w = float(vis.get("figure_width", 6.0))
    fig_h = float(vis.get("figure_height", 6.0))
    x_lim_abs = float(vis.get("x_limit_abs", 1.2))
    y_min = float(vis.get("y_min", 0.0))
    y_max = float(vis.get("y_max", 2.2))
    use_3d = bool(vis.get("stick_figure_use_3d", True))
    elev = float(vis.get("stick_figure_elevation", 22.0))
    azim = float(vis.get("stick_figure_azimuth", 55.0))
    pad_ratio = float(vis.get("stick_figure_axis_padding_ratio", 0.08))
    coord_frames, times = load_mot_coords(mot_path)
    n_mot = len(coord_frames)
    n_act = int(activations.shape[0])
    if n_mot == 0:
        raise ValueError(f"No frames found in .mot file: {mot_path}")
    if n_act == 0:
        raise ValueError("activations must contain at least one frame")
    if n_mot != n_act:
        logger.warning(
            ".mot has %d frames but activations have %d frames; trimming to min.",
            n_mot,
            n_act,
        )
    n_frames = min(n_mot, n_act)
    coord_frames = coord_frames[:n_frames]
    times = times[:n_frames]
    activations = activations[:n_frames]

    def _joints_for_frame(idx: int) -> np.ndarray:
        cf = coord_frames[int(idx)]
        tx = float(cf.get("pelvis_tx", 0.0))
        ty = float(cf.get("pelvis_ty", 0.9))
        tz = float(cf.get("pelvis_tz", 0.0))
        return coords_to_skeleton_joints(
            cf, pelvis_translation=np.array([tx, ty, tz], dtype=np.float64)
        )

    lo = np.full(3, np.inf, dtype=np.float64)
    hi = np.full(3, -np.inf, dtype=np.float64)
    for fi in range(n_frames):
        j = _joints_for_frame(fi).astype(np.float64)
        lo = np.minimum(lo, j.min(axis=0))
        hi = np.maximum(hi, j.max(axis=0))
    lo_b, hi_b = _cubic_skeleton_bounds(lo, hi, pad_ratio)

    segs = [(int(_SMPL_PARENTS[i]), i) for i in range(24) if _SMPL_PARENTS[i] >= 0]
    joints0 = _joints_for_frame(0)

    if use_3d:
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_subplot(111, projection="3d")
        # Matplotlib's visual "vertical" axis is Z; map world Y-up to mpl Z.
        ax.set_xlim(lo_b[0], hi_b[0])
        ax.set_ylim(lo_b[2], hi_b[2])
        ax.set_zlim(lo_b[1], hi_b[1])
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y (up)")
        ax.set_box_aspect((1.0, 1.0, 1.0))

        seg_arr = np.stack(
            [
                np.column_stack([joints0[[p, c], 0], joints0[[p, c], 2], joints0[[p, c], 1]])
                for p, c in segs
            ],
            axis=0,
        )
        lines = Line3DCollection(
            list(seg_arr),
            colors=plt.cm.coolwarm(np.linspace(0.2, 0.8, len(segs))),
            linewidths=3,
        )
        ax.add_collection3d(lines)
        scat = ax.scatter(
            joints0[:, 0], joints0[:, 2], joints0[:, 1], c="k", s=15, depthshade=False
        )
        ax.view_init(elev=elev, azim=azim)
    else:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_aspect("equal")
        ax.set_xlim(-x_lim_abs, x_lim_abs)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("X")
        ax.set_ylabel("Y (up)")
        lines = LineCollection(
            [[joints0[p, [0, 1]], joints0[c, [0, 1]]] for p, c in segs],
            colors=plt.cm.coolwarm(np.linspace(0.2, 0.8, len(segs))),
            linewidths=3,
        )
        ax.add_collection(lines)
        scat = ax.scatter(joints0[:, 0], joints0[:, 1], c="k", s=15)

    title = ax.set_title("frame 0")

    def update(frame_idx: int):
        idx = int(frame_idx)
        j = _joints_for_frame(idx)
        colors = []
        for p, c in segs:
            v = _mean_act_for_segment(
                p, c, idx, activations, muscle_names
            )
            colors.append(plt.cm.coolwarm(float(np.clip(v, 0.0, 1.0))))
        if use_3d:
            seg_list = [
                np.column_stack([j[[p, c], 0], j[[p, c], 2], j[[p, c], 1]])
                for p, c in segs
            ]
            lines.set_segments(seg_list)
            lines.set_colors(colors)
            scat._offsets3d = (j[:, 0], j[:, 2], j[:, 1])
            ax.view_init(elev=elev, azim=azim)
        else:
            lines.set_segments([[j[p, [0, 1]], j[c, [0, 1]]] for p, c in segs])
            lines.set_colors(colors)
            scat.set_offsets(np.c_[j[:, 0], j[:, 1]])
        title.set_text(f"frame {idx}  t={times[idx]:.2f}s")
        return lines, scat, title

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=frame_interval_ms, blit=False
    )

    play = widgets.Play(
        value=0,
        min=0,
        max=n_frames - 1,
        step=1,
        interval=frame_interval_ms,
    )
    slider = widgets.IntSlider(value=0, min=0, max=n_frames - 1, step=1, description="Frame")
    widgets.jslink((play, "value"), (slider, "value"))

    def on_slider(change):
        update(int(change["new"]))
        fig.canvas.draw_idle()

    slider.observe(on_slider, names="value")

    pause = widgets.Button(description="Pause")
    restart = widgets.Button(description="Restart")

    def pause_click(_b):
        play.playing = False

    def restart_click(_b):
        play.value = 0
        slider.value = 0

    pause.on_click(pause_click)
    restart.on_click(restart_click)

    backend = plt.get_backend().lower()
    if "inline" in backend:
        # Inline backend renders a static canvas; show JS animation fallback.
        logger.info("Inline Matplotlib backend detected; using JS animation fallback.")
        display(HTML(anim.to_jshtml()))
        # Inline flushes all open figures at cell end; close ours so only the JS
        # player remains (avoids a second non-interactive duplicate of the same plot).
        plt.close(fig)
    else:
        display(widgets.VBox([widgets.HBox([play, pause, restart]), slider]))
        display(fig.canvas)

    try:
        import pyrender  # noqa: F401

        logger.info(
            "pyrender is importable; this function still uses the Matplotlib skeleton. "
            "Mesh-based preview is not wired here yet."
        )
    except ImportError:
        logger.info(
            "pyrender not available; using Matplotlib stick figure only. "
            "For a more human-like preview later, install deps (e.g. poetry install) "
            "and add SMPL-X mesh rendering alongside pyrender."
        )

    return anim

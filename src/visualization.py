"""Activation plots and interactive motion previews."""

from __future__ import annotations

import logging
from typing import Any

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


def get_smplx_skeleton_joints(smplx_frame: np.ndarray) -> np.ndarray:
    """Approximate SMPL-24 joint positions from a single SMPL-X frame.

    Args:
        smplx_frame: Array of shape ``[D]`` in Motion-X++ layout (``D`` =
            ``SMPLX_MOTION_DIM``).

    Returns:
        Array ``[24, 3]`` joint positions in meters (approximate FK).
    """
    if smplx_frame.shape != (SMPLX_MOTION_DIM,):
        raise ValueError(f"smplx_frame must have shape [{SMPLX_MOTION_DIM}]")
    sl = SMPLX_SLICES
    root_aa = smplx_frame[sl["root_orient"]]
    body = smplx_frame[sl["pose_body"]].reshape(21, 3)
    trans = smplx_frame[sl["trans"]]

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


def _skeleton_bounds_over_frames(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Min/max world coordinates over all frames (for stable 3D axis limits)."""
    lo = np.full(3, np.inf, dtype=np.float64)
    hi = np.full(3, -np.inf, dtype=np.float64)
    for t in range(int(frames.shape[0])):
        j = get_smplx_skeleton_joints(frames[t])
        lo = np.minimum(lo, j.min(axis=0).astype(np.float64))
        hi = np.maximum(hi, j.max(axis=0).astype(np.float64))
    return lo, hi


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


def animate_motion_interactive(
    smplx_motion: np.ndarray,
    activations: np.ndarray,
    muscle_names: list[str],
    config: dict[str, Any],
) -> animation.FuncAnimation:
    """Interactive animation with Matplotlib stick figure (primary backend).

    Args:
        smplx_motion: Array ``[T, D]`` where ``D`` is ``SMPLX_MOTION_DIM`` in
            ``src.smplx_to_opensim``.
        activations: Array ``[T, N]``.
        muscle_names: Muscle names length ``N``.
        config: Full configuration (uses ``visualization`` section).

    Returns:
        ``matplotlib.animation.FuncAnimation`` wired to ipywidgets controls.

    Note:
        On the inline Matplotlib backend, the figure is closed after embedding the
        JS player so Jupyter does not also flush a duplicate static snapshot of the
        same axes. A shaded SMPL-X mesh in pyrender would look more human-like than
        this simplified skeleton; that path is not implemented here yet.
    """
    _ = muscle_names  # reserved for segment coloring by muscle groups
    try:
        import ipywidgets as widgets
        from IPython.display import HTML, display
    except ImportError as exc:
        raise RuntimeError("ipywidgets/IPython required for interactive animation.") from exc

    vis = config.get("visualization", {}) or {}
    blend_n = int(vis.get("tpose_blend_frames", 10))
    frame_interval_ms = int(vis.get("frame_interval_ms", 33))
    fig_w = float(vis.get("figure_width", 6.0))
    fig_h = float(vis.get("figure_height", 6.0))
    x_lim_abs = float(vis.get("x_limit_abs", 1.2))
    y_min = float(vis.get("y_min", 0.0))
    y_max = float(vis.get("y_max", 2.2))
    use_3d = bool(vis.get("stick_figure_use_3d", True))
    elev = float(vis.get("stick_figure_elevation", 18.0))
    azim = float(vis.get("stick_figure_azimuth", -65.0))
    pad_ratio = float(vis.get("stick_figure_axis_padding_ratio", 0.08))
    top_muscles_for_segment = int(vis.get("segment_activation_top_muscles", 8))
    tpose = np.zeros_like(smplx_motion[0])
    tpose[309:312] = smplx_motion[0, 309:312]

    blended = []
    for i in range(min(blend_n, smplx_motion.shape[0])):
        a = (i + 1) / float(max(blend_n, 1))
        blended.append(((1 - a) * tpose + a * smplx_motion[0]).astype(np.float32))
    frames = np.vstack([np.stack(blended), smplx_motion[1:]])

    segs = []
    for i in range(24):
        p = int(_SMPL_PARENTS[i])
        if p < 0:
            continue
        segs.append((p, i))

    joints0 = get_smplx_skeleton_joints(frames[0])

    if use_3d:
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_subplot(111, projection="3d")
        lo, hi = _skeleton_bounds_over_frames(frames)
        span = np.maximum(hi - lo, 1e-3)
        pad = pad_ratio * span
        lo_b = lo - pad
        hi_b = hi + pad
        ax.set_xlim(lo_b[0], hi_b[0])
        ax.set_ylim(lo_b[1], hi_b[1])
        ax.set_zlim(lo_b[2], hi_b[2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y (up)")
        ax.set_zlabel("Z")
        r = hi_b - lo_b
        ax.set_box_aspect((r[0] / r.max(), r[1] / r.max(), r[2] / r.max()))

        seg_arr = np.stack([np.vstack([joints0[p], joints0[c]]) for p, c in segs], axis=0)
        lines = Line3DCollection(
            list(seg_arr),
            colors=plt.cm.coolwarm(np.linspace(0.2, 0.8, len(segs))),
            linewidths=3,
        )
        ax.add_collection3d(lines)
        scat = ax.scatter(
            joints0[:, 0], joints0[:, 1], joints0[:, 2], c="k", s=15, depthshade=False
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

    def _mean_act_for_segment(_parent: int, _child: int, frame_idx: int) -> float:
        mid = int(np.clip(frame_idx, 0, activations.shape[0] - 1))
        use_n = min(top_muscles_for_segment, activations.shape[1])
        return float(np.mean(activations[mid, :use_n]))

    title = ax.set_title("frame 0")

    def update(frame_idx: int):
        j = get_smplx_skeleton_joints(frames[int(frame_idx) % frames.shape[0]])
        colors = []
        for p, c in segs:
            v = _mean_act_for_segment(p, c, int(frame_idx))
            colors.append(plt.cm.coolwarm(float(np.clip(v, 0.0, 1.0))))
        if use_3d:
            seg_list = [np.vstack([j[p], j[c]]) for p, c in segs]
            lines.set_segments(seg_list)
            lines.set_colors(colors)
            scat._offsets3d = (j[:, 0], j[:, 1], j[:, 2])
            ax.view_init(elev=elev, azim=azim)
        else:
            lines.set_segments([[j[p, [0, 1]], j[c, [0, 1]]] for p, c in segs])
            lines.set_colors(colors)
            scat.set_offsets(np.c_[j[:, 0], j[:, 1]])
        title.set_text(f"frame {int(frame_idx)}")
        return lines, scat, title

    anim = animation.FuncAnimation(
        fig, update, frames=frames.shape[0], interval=frame_interval_ms, blit=False
    )

    play = widgets.Play(
        value=0,
        min=0,
        max=frames.shape[0] - 1,
        step=1,
        interval=frame_interval_ms,
    )
    slider = widgets.IntSlider(value=0, min=0, max=frames.shape[0] - 1, step=1, description="Frame")
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

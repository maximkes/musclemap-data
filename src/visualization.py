"""Activation plots and interactive motion previews."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from src.smplx_to_opensim import SMPLX_MOTION_DIM, SMPLX_SLICES

logger = logging.getLogger(__name__)

# SMPL 24-joint parent indices (pelvis at 0).
_SMPL_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=int,
)

# Child offsets in parent-local REST frame (meters).
#
# SMPL-X "0-pose" (all rotations zero) has arms hanging straight down.
# That means:
#   - joints 13/14 (collars) are left/right of spine3, slightly up
#   - joints 16/17 (shoulders) are left/right of collars — still horizontal
#   - joints 18/19 (elbows) are BELOW the shoulders along -Y (arms hang down)
#   - joints 20/21 (wrists) are BELOW elbows along -Y
#   - joints 22/23 (hands)  are BELOW wrists along -Y
#
# The confusion: "crossed arms" happens when the elbow offset points along -X
# (inward) instead of staying along -Y (downward). In 0-pose they must be -Y.
# Rotations applied by coords_to_skeleton_joints will swing them into position.
_SMPL_OFFSETS = np.array(
    [
        [ 0.000,  0.000,  0.000],  #  0 pelvis (root)
        [ 0.090, -0.090,  0.000],  #  1 R hip
        [-0.090, -0.090,  0.000],  #  2 L hip
        [ 0.000,  0.120,  0.000],  #  3 spine1
        [ 0.000, -0.400,  0.000],  #  4 R knee
        [ 0.000, -0.400,  0.000],  #  5 L knee
        [ 0.000,  0.120,  0.000],  #  6 spine2
        [ 0.000, -0.400,  0.000],  #  7 R ankle
        [ 0.000, -0.400,  0.000],  #  8 L ankle
        [ 0.000,  0.150,  0.000],  #  9 spine3
        [ 0.000, -0.060,  0.100],  # 10 R foot
        [ 0.000, -0.060,  0.100],  # 11 L foot
        [ 0.000,  0.180,  0.000],  # 12 neck
        [-0.060,  0.050,  0.000],  # 13 L collar (left+up from spine3)
        [ 0.060,  0.050,  0.000],  # 14 R collar (right+up from spine3)
        [ 0.000,  0.120,  0.000],  # 15 head
        # 16 L shoulder: left of L collar. In 0-pose arms hang down so the
        #    shoulder joint sits at the top of the upper arm — offset is
        #    purely lateral from the collar.
        [-0.150,  0.000,  0.000],  # 16 L shoulder
        [ 0.150,  0.000,  0.000],  # 17 R shoulder
        # 18/19 elbows: BELOW the shoulder in 0-pose (arms straight down).
        #    -Y is correct here. Do NOT use ±X — that would cross the arms.
        [ 0.000, -0.270,  0.000],  # 18 L elbow
        [ 0.000, -0.270,  0.000],  # 19 R elbow
        # 20/21 wrists: continue downward.
        [ 0.000, -0.250,  0.000],  # 20 L wrist
        [ 0.000, -0.250,  0.000],  # 21 R wrist
        # 22/23 hands: fingertips below wrist.
        [ 0.000, -0.100,  0.000],  # 22 L hand
        [ 0.000, -0.100,  0.000],  # 23 R hand
    ],
    dtype=np.float64,
)

# Segment → muscle substring mapping (unchanged).
_SEGMENT_TO_MUSCLE_SUBSTRINGS: dict[tuple[int, int], list[str]] = {
    (1, 4): ["glut", "psoas", "iliacus", "rect_fem_r", "vas_",
             "semimem_r", "semiten_r", "bflh_r", "bfsh_r",
             "grac_r", "sart_r", "tfl_r", "add_"],
    (2, 5): ["glut", "psoas", "iliacus", "rect_fem_l", "vas_",
             "semimem_l", "semiten_l", "bflh_l", "bfsh_l",
             "grac_l", "sart_l", "tfl_l", "add_"],
    (4, 7): ["gas_med_r", "gas_lat_r", "soleus_r", "tib_ant_r",
             "tib_post_r", "per_brev_r", "per_long_r"],
    (5, 8): ["gas_med_l", "gas_lat_l", "soleus_l", "tib_ant_l",
             "tib_post_l", "per_brev_l", "per_long_l"],
    (7, 10): ["gas_med_r", "gas_lat_r", "soleus_r", "tib_ant_r",
              "tib_post_r", "per_brev_r", "per_long_r"],
    (8, 11): ["gas_med_l", "gas_lat_l", "soleus_l", "tib_ant_l",
              "tib_post_l", "per_brev_l", "per_long_l"],
    (0, 3):  ["lumbar", "psoas", "iliacus", "erec_sp", "mult", "rect_abd"],
    (3, 6):  ["erec_sp", "mult"],
    (9, 16): ["delt", "supraspin", "infraspin", "teres", "subscap",
              "pect_maj", "bic_brev_r", "bic_long_r", "tric_r"],
    (9, 17): ["delt", "supraspin", "infraspin", "teres", "subscap",
              "pect_maj", "bic_brev_l", "bic_long_l", "tric_l"],
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
    """FK from one SMPL-X 322-dim frame → 24 world joint positions."""
    if smplx_frame.shape != (SMPLX_MOTION_DIM,):
        raise ValueError(f"smplx_frame must have shape [{SMPLX_MOTION_DIM}]")

    sl = SMPLX_SLICES
    root_aa = smplx_frame[sl["root_orient"]].astype(np.float64, copy=False)
    body = smplx_frame[sl["pose_body"]].reshape(21, 3)
    trans = smplx_frame[sl["trans"]].astype(np.float64, copy=False)

    rotvec = np.zeros((24, 3), dtype=np.float64)
    rotvec[0] = root_aa
    rotvec[1:22] = body
    rotvec[22] = rotvec[20]
    rotvec[23] = rotvec[21]

    global_R: list[np.ndarray] = []
    for i in range(24):
        # SMPL-X body_pose values ARE genuine axis-angle vectors → from_rotvec is correct.
        r_local = R.from_rotvec(rotvec[i]).as_matrix()
        p = int(_SMPL_PARENTS[i])
        global_R.append(r_local if p < 0 else global_R[p] @ r_local)

    joints = np.zeros((24, 3), dtype=np.float64)
    joints[0] = trans
    for i in range(1, 24):
        p = int(_SMPL_PARENTS[i])
        joints[i] = joints[p] + global_R[p] @ _SMPL_OFFSETS[i]

    if align_rotation is not None:
        joints = np.asarray(align_rotation.apply(joints), dtype=np.float64)
    return joints.astype(np.float32)


def _skeleton_bounds_over_frames(
    frames: np.ndarray, align_rotation: Optional[R] = None
) -> tuple[np.ndarray, np.ndarray]:
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
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    center = (lo + hi) / 2.0
    max_half = float(np.max(hi - lo)) / 2.0
    max_half = max(max_half, float(min_half_extent))
    half = max_half * (1.0 + 2.0 * float(pad_ratio))
    return center - half, center + half


def plot_activation_topk(
    activations: np.ndarray, muscle_names: list[str], k: int
) -> matplotlib.figure.Figure:
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


# OpenSim coordinate → (SMPL joint index, Euler axis index 0=X 1=Y 2=Z).
#
# Shoulder convention for 0-pose (arms hanging down):
#   arm_flex  = raise arm forward (sagittal) → rotates around LOCAL Z of shoulder
#               because in 0-pose the arm offset is -Y, and R_z(-Y) → ±X (forward)
#   arm_add   = raise arm sideways (frontal)  → rotates around LOCAL X
#   arm_rot   = axial spin                    → rotates around LOCAL Y
#
# Left/right: SMPL joint 16 = L shoulder (collar at -X), 17 = R shoulder (+X).
# OpenSim uses _r/_l suffix consistently; arm_flex_r drives the RIGHT shoulder.
_OPENSIM_COORD_TO_SMPL: dict[str, tuple[int, int]] = {
    "pelvis_tilt":      (0, 0),
    "pelvis_list":      (0, 1),
    "pelvis_rotation":  (0, 2),
    "lumbar_extension": (9, 0),
    "lumbar_bending":   (9, 1),
    "lumbar_rotation":  (9, 2),
    "hip_flexion_r":    (1, 0),
    "hip_adduction_r":  (1, 1),
    "hip_rotation_r":   (1, 2),
    "hip_flexion_l":    (2, 0),
    "hip_adduction_l":  (2, 1),
    "hip_rotation_l":   (2, 2),
    # Knee/ankle: 1-DOF sagittal flex around local X.
    "knee_angle_r":     (4, 0),
    "knee_angle_l":     (5, 0),
    "ankle_angle_r":    (7, 0),
    "ankle_angle_l":    (8, 0),
    # Shoulders: arm offset is -Y in 0-pose.
    #   flex (forward raise) → R_z rotates -Y → +X  → axis 2 ✓
    #   add  (side raise)    → R_x rotates -Y → ±Z  → axis 0 ✓
    #   rot  (axial spin)    → R_y spins arm  → axis 1 ✓
    # joint 17 = R shoulder (+X collar), joint 16 = L shoulder (-X collar)
    "arm_flex_r":       (17, 2),
    "arm_add_r":        (17, 0),
    "arm_rot_r":        (17, 1),
    "arm_flex_l":       (16, 2),
    "arm_add_l":        (16, 0),
    "arm_rot_l":        (16, 1),
    # Elbow flexion in sagittal plane around local X.
    "elbow_flex_r":     (19, 0),
    "elbow_flex_l":     (18, 0),
}

# Sign correction for OpenSim coordinate conventions used in notebook sanity checks.
_OPENSIM_COORD_SIGN: dict[str, float] = {
    "hip_flexion_r": -1.0,
    "hip_flexion_l": -1.0,
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
        sign = float(_OPENSIM_COORD_SIGN.get(name, 1.0))
        rotvec[joint_idx, axis_idx] += sign * float(val)

    rotvec[22] = rotvec[20]
    rotvec[23] = rotvec[21]

    global_R: list[np.ndarray] = []
    for i in range(24):
        # Each rotvec[i] stores independent Euler angles written by separate
        # OpenSim DOFs.  Composing R_x @ R_y @ R_z is correct here.
        # (For SMPL-X axis-angle data, R.from_rotvec is used instead — see
        # get_smplx_skeleton_joints above.)
        ax, ay, az = float(rotvec[i, 0]), float(rotvec[i, 1]), float(rotvec[i, 2])
        r_local = (
            R.from_euler("x", ax).as_matrix()
            @ R.from_euler("y", ay).as_matrix()
            @ R.from_euler("z", az).as_matrix()
        )
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
    for ln in lines[header_end + 2:]:
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


def _to_mpl(j: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World [N,3] → matplotlib 3D axes.

    World:      X=right  Y=up   Z=backward
    Matplotlib: x=right  y=forward (negated Z)  z=up (Y)
    """
    return j[:, 0], -j[:, 2], j[:, 1]


def animate_motion_interactive(
    mot_path: "str | Path",
    activations: np.ndarray,
    muscle_names: list[str],
    config: dict[str, Any],
) -> None:
    """Interactive animation logging to Rerun from .mot + activations."""
    try:
        import rerun as rr
    except ImportError as exc:
        raise RuntimeError(
            "rerun-sdk is required for interactive animation. Ensure conda env includes "
            "rerun-sdk (see environment.yml pip section)."
        ) from exc

    vis = config.get("visualization", {}) or {}
    app_id = str(vis.get("rerun_app_id", "musclemap"))
    recording_id = vis.get("rerun_recording_id", None)
    spawn = bool(vis.get("rerun_spawn", True))
    frame_interval_ms = int(vis.get("frame_interval_ms", 33))

    rr.init(app_id, recording_id=recording_id, spawn=spawn)
    if not spawn:
        rr.notebook_show()

    coord_frames, times = load_mot_coords(mot_path)
    if len(coord_frames) == 0:
        raise ValueError(f"No frames found in .mot file: {mot_path}")
    if int(activations.shape[0]) == 0:
        raise ValueError("activations must contain at least one frame")
    if len(coord_frames) != int(activations.shape[0]):
        logger.warning(
            ".mot has %d frames but activations have %d frames; trimming to min.",
            len(coord_frames),
            int(activations.shape[0]),
        )
    n_frames = min(len(coord_frames), int(activations.shape[0]))
    coord_frames = coord_frames[:n_frames]
    times = times[:n_frames]
    activations = activations[:n_frames].astype(np.float32, copy=False)

    segs = [(int(_SMPL_PARENTS[i]), i) for i in range(24) if _SMPL_PARENTS[i] >= 0]
    tab20 = plt.get_cmap("tab20")
    muscle_color_map: dict[str, list[int]] = {}
    for mi, mname in enumerate(muscle_names):
        rgba = tab20(mi % 20)
        muscle_color_map[mname] = [int(v * 255) for v in rgba]
    rr.log(
        "info/muscle_color_map",
        rr.TextDocument(str(muscle_color_map)),
        static=True,
    )

    coolwarm = plt.get_cmap("coolwarm")
    fps = 1000.0 / max(float(frame_interval_ms), 1.0)
    logger.info("Logging %d frames to Rerun at ~%.2f fps.", n_frames, fps)
    for t in range(n_frames):
        rr.set_time_seconds("time", float(times[t]))
        rr.set_time_sequence("frame", int(t))

        cf = coord_frames[t]
        tx = float(cf.get("pelvis_tx", 0.0))
        ty = float(cf.get("pelvis_ty", 0.9))
        tz = float(cf.get("pelvis_tz", 0.0))
        joints = coords_to_skeleton_joints(
            cf, pelvis_translation=np.array([tx, ty, tz], dtype=np.float64)
        ).astype(np.float32, copy=False)

        rr.log(
            "skeleton/joints",
            rr.Points3D(
                positions=joints,
                radii=0.015,
                colors=[[80, 80, 80, 255]] * 24,
            ),
        )

        for parent_idx, child_idx in segs:
            act = _mean_act_for_segment(parent_idx, child_idx, t, activations, muscle_names)
            rgba = [int(v * 255) for v in coolwarm(float(act))]
            seg_name = f"skeleton/bones/{parent_idx}_{child_idx}"
            rr.log(
                seg_name,
                rr.LineStrips3D(
                    strips=[[joints[parent_idx].tolist(), joints[child_idx].tolist()]],
                    radii=0.008,
                    colors=[rgba],
                ),
            )

        for mi, mname in enumerate(muscle_names):
            rr.log(f"activations/{mname}", rr.Scalar(float(activations[t, mi])))

    rr.log(
        "info/model",
        rr.TextDocument(f"Model: {config.get('paths', {}).get('opensim_model', 'unknown')}"),
        static=True,
    )
    return None


def build_rerun_smplx_animation(
    smplx_motion: np.ndarray,
    config: dict[str, Any],
    activations: np.ndarray | None = None,
    muscle_names: list[str] | None = None,
) -> None:
    """Log raw SMPL-X motion (no OpenSim) to Rerun.

    If activations + muscle_names are provided, also logs activation scalars.
    Useful for previewing the source motion before the OpenSim pipeline.
    """
    try:
        import rerun as rr
    except ImportError as exc:
        raise RuntimeError(
            "rerun-sdk is required for interactive animation. Ensure conda env includes "
            "rerun-sdk (see environment.yml pip section)."
        ) from exc

    vis = config.get("visualization", {}) or {}
    app_id = str(vis.get("rerun_app_id", "musclemap"))
    recording_id = vis.get("rerun_recording_id", None)
    spawn = bool(vis.get("rerun_spawn", True))
    frame_interval_ms = int(vis.get("frame_interval_ms", 33))

    rr.init(app_id, recording_id=recording_id, spawn=spawn)
    if not spawn:
        rr.notebook_show()

    fps = 1000.0 / max(float(frame_interval_ms), 1.0)
    segs = [(int(_SMPL_PARENTS[i]), i) for i in range(24) if _SMPL_PARENTS[i] >= 0]
    motion = np.asarray(smplx_motion)
    for t in range(int(motion.shape[0])):
        rr.set_time_seconds("time", float(t / fps))
        rr.set_time_sequence("frame", int(t))
        joints = get_smplx_skeleton_joints(motion[t]).astype(np.float32, copy=False)
        rr.log("smplx/joints", rr.Points3D(joints, radii=0.015))
        strips = [[joints[p].tolist(), joints[c].tolist()] for p, c in segs]
        rr.log(
            "smplx/bones",
            rr.LineStrips3D(
                strips,
                radii=0.008,
                colors=[[100, 160, 220, 255]] * len(segs),
            ),
        )
        if activations is not None and muscle_names is not None:
            for mi, mname in enumerate(muscle_names):
                rr.log(f"activations/{mname}", rr.Scalar(float(activations[t, mi])))
    return None


def show_dash_app(
    mot_path: "str | Path",
    activations: np.ndarray,
    muscle_names: list[str],
    config: dict[str, Any],
    smplx_motion: np.ndarray | None = None,
    port: int = 8050,
    inline: bool | None = None,
) -> None:
    """Launch the Plotly/Dash visualiser.

    Args:
        mot_path: Path to OpenSim .mot file.
        activations: [T, N_muscles] float32 array.
        muscle_names: List of muscle name strings.
        config: Project config dict.
        smplx_motion: Optional [T, 322] raw SMPL-X frames for a second trace.
        port: Local port for the Dash server (default 8050).
        inline: True = embed in notebook via jupyter_dash,
                False = open in browser tab,
                None = auto-detect (True if IPython kernel detected).
    """
    from src.dash_app import build_dash_app

    vis = config.get("visualization", {}) or {}
    if inline is None:
        try:
            from IPython import get_ipython

            inline = get_ipython() is not None
        except ImportError:
            inline = False
    if port == 8050 and "dash_port" in vis:
        port = int(vis.get("dash_port", 8050))
    if inline is None and vis.get("dash_inline", None) is not None:
        inline = bool(vis.get("dash_inline"))

    app = build_dash_app(mot_path, activations, muscle_names, config, smplx_motion)

    if inline:
        try:
            from jupyter_dash import JupyterDash  # type: ignore

            # Patch: JupyterDash needs the server from the existing app.
            japp = JupyterDash(__name__)
            japp.layout = app.layout
            for cb in app.callback_map.values():
                pass  # callbacks already registered on app; run app directly
            app.run(jupyter_mode="inline", port=port)
        except ImportError:
            logger.warning(
                "jupyter_dash not installed; falling back to browser tab. "
                "Install with: pip install jupyter-dash"
            )
            app.run(host="127.0.0.1", port=port, debug=False)
    else:
        import threading
        import webbrowser

        def _open():
            import time

            time.sleep(1.2)
            webbrowser.open(f"http://127.0.0.1:{port}")

        threading.Thread(target=_open, daemon=True).start()
        app.run(host="127.0.0.1", port=port, debug=False)


def show_dash_smplx_motion(
    smplx_motion: np.ndarray,
    config: dict[str, Any],
    port: int = 8050,
    inline: bool | None = None,
) -> None:
    """Launch Dash visualisation directly from raw SMPL-X motion [T, 322]."""
    motion = np.asarray(smplx_motion)
    if motion.ndim != 2 or motion.shape[1] != SMPLX_MOTION_DIM:
        raise ValueError(f"smplx_motion must have shape [T, {SMPLX_MOTION_DIM}]")
    if motion.shape[0] < 1:
        raise ValueError("smplx_motion must have at least one frame")

    import tempfile

    from src.smplx_to_opensim import smplx_to_mot

    tmp_dir = Path(tempfile.mkdtemp(prefix="dash_motion_"))
    mot_path = tmp_dir / "motion_preview.mot"
    smplx_to_mot(motion, config, mot_path)

    dummy_acts = np.zeros((motion.shape[0], 1), dtype=np.float32)
    show_dash_app(
        mot_path=mot_path,
        activations=dummy_acts,
        muscle_names=["motion_only"],
        config=config,
        smplx_motion=motion,
        port=port,
        inline=inline,
    )
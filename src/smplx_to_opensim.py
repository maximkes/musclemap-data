"""Load Motion-X++ SMPL-X arrays and write OpenSim ``.mot`` coordinate files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

from src.smplx_joint_regressor import apply_rom_limits, get_opensim_coords, load_regressor
from src.utils import joint_velocity_clamp

logger = logging.getLogger(__name__)

SMPLX_SLICES: dict[str, slice] = {
    "root_orient": slice(0, 3),
    "pose_body": slice(3, 66),
    "pose_hand": slice(66, 156),
    "pose_jaw": slice(156, 159),
    "face_expr": slice(159, 209),
    "face_shape": slice(209, 309),
    "trans": slice(309, 312),
    "betas": slice(312, 322),
}

SMPLX_MOTION_DIM: int = max(sl.stop for sl in SMPLX_SLICES.values())


def load_smplx_motion(path: Path) -> dict[str, np.ndarray]:
    """Load a ``[T, D]`` SMPL-X motion array and split into named components.

    ``D`` equals ``SMPLX_MOTION_DIM`` (derived from ``SMPLX_SLICES``).

    Args:
        path: Path to ``.npy`` motion file.

    Returns:
        Dictionary of component name to ``float32`` array.
    """
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != SMPLX_MOTION_DIM:
        raise ValueError(f"Expected motion [T, {SMPLX_MOTION_DIM}], got {arr.shape}")
    arr = arr.astype(np.float32, copy=False)
    return {name: arr[:, sl] for name, sl in SMPLX_SLICES.items()}


def axis_angle_to_euler(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle ``[..., 3]`` to intrinsic XYZ Euler angles (radians)."""
    from scipy.spatial.transform import Rotation as R

    shape = aa.shape
    flat = aa.reshape(-1, 3)
    euler = R.from_rotvec(flat).as_euler("xyz")
    return euler.reshape(shape[:-1] + (3,))


def butterworth_filter(
    data: np.ndarray, order: int, cutoff_hz: float, fps: float
) -> np.ndarray:
    """Low-pass Butterworth filter along time (axis 0).

    Args:
        data: Array with time on axis 0.
        order: Filter order.
        cutoff_hz: Cutoff frequency in Hz.
        fps: Sampling frequency in Hz.

    Returns:
        Filtered array (or input unchanged if too short for ``filtfilt``).
    """
    if data.shape[0] < 3:
        return data
    nyq = 0.5 * float(fps)
    wn = min(cutoff_hz / nyq, 0.99)
    b, a = signal.butter(order, wn, btype="low")
    # Match SciPy filtfilt default-style stability threshold for short sequences.
    padlen = 3 * max(len(a), len(b))
    if data.shape[0] < padlen:
        logger.warning(
            "Sequence length %s < padlen %s; skipping Butterworth filter.",
            data.shape[0],
            padlen,
        )
        return data
    return signal.filtfilt(b, a, data, axis=0, padlen=padlen)


def write_mot_file(
    coords: dict[str, np.ndarray], fps: float, output_path: Path
) -> None:
    """Write OpenSim ``Storage`` format ``.mot`` (coordinates, radians).

    Args:
        coords: Coordinate name → length-``T`` arrays.
        fps: Samples per second (time column uses ``0..(T-1)/fps``).
        output_path: Destination ``.mot`` path.
    """
    if not coords:
        raise ValueError("coords must be non-empty")
    names = sorted(coords.keys())
    t0 = next(iter(coords.values())).shape[0]
    for n in names:
        if coords[n].shape[0] != t0:
            raise ValueError("All coordinate arrays must share the same length")
    t = t0
    ncols = len(names) + 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Coordinates\n")
        f.write(f"nRows={t}\n")
        f.write(f"nColumns={ncols}\n")
        f.write("inDegrees=no\n")
        f.write("endheader\n")
        header = "time\t" + "\t".join(names)
        f.write(header + "\n")
        for i in range(t):
            time_val = (i / float(fps)) if t > 0 else 0.0
            row = [f"{time_val:.8f}"] + [f"{float(coords[k][i]):.8f}" for k in names]
            f.write("\t".join(row) + "\n")


def _upsample_coords(
    coords: dict[str, np.ndarray],
    src_fps: float,
    dst_fps: float,
    method: str,
) -> dict[str, np.ndarray]:
    """Resample coordinate trajectories to ``dst_fps``."""
    if method == "none" or abs(src_fps - dst_fps) < 1e-6:
        return {k: v.astype(np.float32, copy=False) for k, v in coords.items()}
    t_src = coords[next(iter(coords))].shape[0]
    if t_src < 2:
        return {k: v.astype(np.float32, copy=False) for k, v in coords.items()}
    x_old = np.arange(t_src, dtype=np.float64) / float(src_fps)
    t_new = int(round(t_src * float(dst_fps) / float(src_fps)))
    if t_new < 2:
        t_new = t_src
    x_new = np.arange(t_new, dtype=np.float64) / float(dst_fps)
    kind = "cubic" if method == "cubic" else "linear"
    out: dict[str, np.ndarray] = {}
    for name, series in coords.items():
        y = np.asarray(series, dtype=np.float64)
        f = interp1d(
            x_old,
            y,
            axis=0,
            kind=kind if y.shape[0] >= 4 and kind == "cubic" else "linear",
            fill_value="extrapolate",
        )
        out[name] = np.asarray(f(x_new), dtype=np.float32)
    return out


def smplx_to_mot(
    motion_npy: np.ndarray | Path, config: dict, output_path: Path
) -> Tuple[Path, Rotation]:
    """Convert SMPL-X ``[T, D]`` motion to an OpenSim ``.mot`` file.

    Pipeline: split arrays → OpenSim coordinates → Butterworth filter →
    velocity clamp → ROM clamp (in regressor) → optional upsample → write.

    Args:
        motion_npy: Raw motion array ``[T, D]`` or path to ``.npy`` file; ``D`` is
            ``SMPLX_MOTION_DIM`` (see ``src.smplx_to_opensim``).
        config: Full configuration dictionary.
        output_path: Path to write the ``.mot`` file.

    Returns:
        ``(output_path, r_align)`` after writing, where ``r_align`` is the
        canonical frame alignment rotation from ``get_opensim_coords`` (pelvis
        block) for reuse by visualization forward kinematics.
    """
    if isinstance(motion_npy, Path):
        motion_arr = np.load(motion_npy)
    else:
        motion_arr = np.asarray(motion_npy)
    if motion_arr.ndim != 2 or motion_arr.shape[1] != SMPLX_MOTION_DIM:
        raise ValueError(f"motion_npy must be [T, {SMPLX_MOTION_DIM}]")
    conv = config.get("conversion", {})
    dataset = config.get("dataset", {})
    src_fps = float(dataset.get("fps", 30))
    dst_fps = float(conv.get("target_fps", src_fps))
    output_fps = float(conv.get("output_fps", src_fps))
    upsample_method = str(conv.get("upsample_method", "cubic"))

    parts = {k: motion_arr[:, sl] for k, sl in SMPLX_SLICES.items() if k in SMPLX_SLICES}
    body_pose = parts["pose_body"]
    root_orient = parts["root_orient"]
    trans = parts["trans"]

    paths_cfg = config.get("paths", {}) or {}
    reg_path = paths_cfg.get("smplx_to_opensim_regressor")
    reg = load_regressor(Path(reg_path) if reg_path else None)

    coords, r_align = get_opensim_coords(body_pose, root_orient, trans, config, reg)

    order = int(conv.get("filter_order", 4))
    cutoff = float(conv.get("filter_cutoff_hz", 6.0))
    max_vel = float(conv.get("max_joint_velocity_rad_s", 15.0))

    names = sorted(coords.keys())
    stacked = np.stack([coords[n] for n in names], axis=1).astype(np.float64)
    stacked = butterworth_filter(stacked, order, cutoff, src_fps)
    stacked = joint_velocity_clamp(stacked.astype(np.float32), max_vel, src_fps)
    processed = {n: stacked[:, i].astype(np.float32) for i, n in enumerate(names)}
    processed = apply_rom_limits(processed, config)

    processed = _upsample_coords(processed, src_fps, dst_fps, upsample_method)
    processed = _upsample_coords(processed, dst_fps, output_fps, upsample_method)

    write_mot_file(processed, output_fps, output_path)
    return output_path, r_align

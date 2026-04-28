"""SMPL-X body pose to OpenSim generalized coordinates (geometric mapping).

Coordinate mapping here uses raw Motion-X++ axis-angle and translation as stored.
The interactive stick figure in ``src.visualization`` applies a separate canonical
−90° X pre-rotation on the **root** only for upright display; it does not change
these OpenSim coordinates.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

# SMPL-X body joint index (1..21 body_pose blocks) -> Rajagopal-style coordinate names.
JOINT_CORRESPONDENCE: dict[int, list[str]] = {
    1: ["hip_flexion_r", "hip_adduction_r", "hip_rotation_r"],
    2: ["hip_flexion_l", "hip_adduction_l", "hip_rotation_l"],
    4: ["knee_angle_r"],
    5: ["knee_angle_l"],
    7: ["ankle_angle_r"],
    8: ["ankle_angle_l"],
    10: ["subtalar_angle_r"],
    11: ["subtalar_angle_l"],
    16: ["arm_flex_r", "arm_add_r", "arm_rot_r"],
    17: ["arm_flex_l", "arm_add_l", "arm_rot_l"],
    18: ["elbow_flex_r"],
    19: ["elbow_flex_l"],
    20: ["pro_sup_r"],
    21: ["pro_sup_l"],
}

# SMPL-X body joint indices (1..21) for lumbar chain; slice ``bp[:, j-1, :]``.
LUMBAR_JOINTS: tuple[int, ...] = (3, 6, 9)
LUMBAR_WEIGHTS: np.ndarray = np.array([0.5, 0.3, 0.2], dtype=np.float32)

# Map common config.yaml ROM keys to coordinate names emitted here.
_ROM_KEY_ALIASES: dict[str, str] = {
    "knee_flex_r": "knee_angle_r",
    "knee_flex_l": "knee_angle_l",
    "hip_flex_r": "hip_flexion_r",
    "hip_flex_l": "hip_flexion_l",
    "hip_add_r": "hip_adduction_r",
    "hip_add_l": "hip_adduction_l",
    "hip_rot_r": "hip_rotation_r",
    "hip_rot_l": "hip_rotation_l",
}


def load_regressor(path: Optional[Path]) -> Optional[object]:
    """Load optional SMPL-X→OpenSim regressor weights.

    Args:
        path: Path to regressor file, or ``None``.

    Returns:
        Loaded object, or ``None`` if path is missing / unreadable (geometric fallback).
    """
    if path is None:
        logger.info("No SMPL-X regressor path configured; using geometric fallback.")
        return None
    p = Path(path)
    if not p.is_file():
        logger.info("Regressor file not found at %s; using geometric fallback.", p)
        return None
    try:
        data = np.load(p, allow_pickle=True)
    except OSError as exc:
        logger.info("Could not load regressor %s (%s); geometric fallback.", p, exc)
        return None
    logger.info("Loaded SMPL-X regressor from %s", p)
    return data


def _axis_angle_to_xyz_euler(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle vectors ``[..., 3]`` to intrinsic XYZ Euler radians."""
    flat = aa.reshape(-1, 3)
    euler = R.from_rotvec(flat).as_euler("xyz")
    return euler.reshape(aa.shape[:-1] + (3,))


def _weighted_average_rotvec(rotvecs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean rotation in SO(3) per time step, returned as axis-angle ``[T, 3]``.

    Args:
        rotvecs: Axis-angle stack ``[T, N, 3]``.
        weights: Non-negative weights ``[N]`` (typically sum to 1).

    Returns:
        Mean rotation as rotvec ``float32`` ``[T, 3]``.
    """
    rotvecs = np.asarray(rotvecs, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if rotvecs.ndim != 3 or rotvecs.shape[2] != 3:
        raise ValueError("rotvecs must have shape [T, N, 3]")
    t, n, _ = rotvecs.shape
    if weights.shape != (n,):
        raise ValueError(f"weights must have shape [{n}]")
    out = np.zeros((t, 3), dtype=np.float32)
    for ti in range(t):
        rots = R.from_rotvec(rotvecs[ti])
        out[ti] = rots.mean(weights=weights).as_rotvec().astype(np.float32)
    return out


def apply_rom_limits(
    coords: dict[str, np.ndarray], config: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Clamp coordinate trajectories using per-joint and global ROM settings.

    Args:
        coords: Mapping of coordinate name → ``float32``/``float64`` array ``[T]``.
        config: Full configuration (reads ``conversion.joint_rom_limits``).

    Returns:
        New mapping with clamped ``float32`` series.
    """
    conv = config.get("conversion", {}) or {}
    limits = conv.get("joint_rom_limits") or {}
    global_clamp = float(conv.get("global_rom_clamp_rad", np.pi))
    out: dict[str, np.ndarray] = {}
    for name, series in coords.items():
        arr = np.asarray(series, dtype=np.float64)
        lo, hi = -global_clamp, global_clamp
        for cfg_key, span in limits.items():
            if not isinstance(span, (list, tuple)) or len(span) != 2:
                continue
            target = _ROM_KEY_ALIASES.get(str(cfg_key), str(cfg_key))
            if target != name:
                continue
            lo, hi = float(span[0]), float(span[1])
            break
        out[name] = np.clip(arr, lo, hi).astype(np.float32, copy=False)
    return out


def get_opensim_coords(
    body_pose: np.ndarray,
    root_orient: np.ndarray,
    trans: np.ndarray,
    config: dict,
    regressor: Optional[object] = None,
) -> Tuple[dict[str, np.ndarray], R]:
    """Map SMPL-X pose arrays to OpenSim coordinate time series.

    Geometric fallback: body joints use axis-angle → intrinsic ``xyz`` Euler
    (same as ``_axis_angle_to_xyz_euler``). Pelvis orientation uses intrinsic
    ``xyz`` Euler to match Rajagopal-style pelvis tilt (X), list (Y), rotation
    (Z) ordering.

    Root orientation and translation are **canonically aligned** to frame 0:
    ``R_align = R(root_orient[0]).inv()`` is applied to the entire root rotation
    sequence and to translations, so frame 0 has zero pelvis Euler angles when
    expressed in that aligned frame (upright reference) and ``trans[0]`` maps to
    zero pelvis translation components at ``t=0``.

    Args:
        body_pose: Array ``[T, 63]`` (21 joints × 3 axis-angle).
        root_orient: Array ``[T, 3]`` root axis-angle.
        trans: Array ``[T, 3]`` root translation.
        config: Full configuration mapping (for ROM limits).
        regressor: Optional regressor; if provided and supported, may override
            mapping in the future. Currently unused (geometric path only).

    Returns:
        Tuple ``(coords, R_align)`` where ``coords`` maps OpenSim coordinate name
        → ``float32`` array of length ``T`` (after ROM limits), and ``R_align`` is
        the ``scipy.spatial.transform.Rotation`` applied to align frame 0 for reuse
        in visualization forward kinematics.
    """
    _ = regressor  # Reserved for future learned mapping.
    if body_pose.ndim != 2 or body_pose.shape[1] != 63:
        raise ValueError("body_pose must have shape [T, 63]")
    t = body_pose.shape[0]
    if root_orient.shape != (t, 3) or trans.shape != (t, 3):
        raise ValueError("root_orient and trans must match body_pose time dimension")

    coords: dict[str, np.ndarray] = {}

    r_root = R.from_rotvec(root_orient)
    r_align = R.from_rotvec(root_orient[0]).inv()
    aligned_root = r_align * r_root
    root_euler = aligned_root.as_euler("xyz")
    coords["pelvis_tilt"] = root_euler[:, 0].astype(np.float32)
    coords["pelvis_list"] = root_euler[:, 1].astype(np.float32)
    coords["pelvis_rotation"] = root_euler[:, 2].astype(np.float32)
    aligned_trans = r_align.apply(np.asarray(trans, dtype=np.float64))
    coords["pelvis_tx"] = aligned_trans[:, 0].astype(np.float32)
    coords["pelvis_ty"] = aligned_trans[:, 1].astype(np.float32)
    coords["pelvis_tz"] = aligned_trans[:, 2].astype(np.float32)

    bp = body_pose.reshape(t, 21, 3)
    for joint_idx, names in JOINT_CORRESPONDENCE.items():
        if joint_idx < 1 or joint_idx > 21:
            continue
        aa = bp[:, joint_idx - 1, :]
        if len(names) == 1:
            euler = R.from_rotvec(aa).as_euler("xyz")
            coords[names[0]] = euler[:, 0].astype(np.float32)
        else:
            euler = _axis_angle_to_xyz_euler(aa)
            for axis_i, cname in enumerate(names):
                coords[cname] = euler[:, axis_i].astype(np.float32)

    lumbar_idx = [j - 1 for j in LUMBAR_JOINTS]
    lum_rv = _weighted_average_rotvec(bp[:, lumbar_idx, :], LUMBAR_WEIGHTS)
    lum_euler = R.from_rotvec(np.asarray(lum_rv, dtype=np.float64)).as_euler("xyz")
    coords["lumbar_extension"] = lum_euler[:, 0].astype(np.float32)
    coords["lumbar_bending"] = lum_euler[:, 1].astype(np.float32)
    coords["lumbar_rotation"] = lum_euler[:, 2].astype(np.float32)

    return apply_rom_limits(coords, config), r_align

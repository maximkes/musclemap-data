"""SMPL-X body pose to OpenSim generalized coordinates (geometric mapping)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

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
    16: ["arm_flex_r", "arm_add_r", "arm_rot_r"],
    17: ["arm_flex_l", "arm_add_l", "arm_rot_l"],
    18: ["elbow_flex_r"],
    19: ["elbow_flex_l"],
    20: ["pro_sup_r"],
    21: ["pro_sup_l"],
}

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
) -> dict[str, np.ndarray]:
    """Map SMPL-X pose arrays to OpenSim coordinate time series.

    Geometric fallback: axis-angle → intrinsic XYZ Euler (body joints) and
    root orientation as ZYX Euler for pelvis tilt/list/rotation. Translations
    map to ``pelvis_tx`` / ``pelvis_ty`` / ``pelvis_tz``.

    Args:
        body_pose: Array ``[T, 63]`` (21 joints × 3 axis-angle).
        root_orient: Array ``[T, 3]`` root axis-angle.
        trans: Array ``[T, 3]`` root translation.
        config: Full configuration mapping (for ROM limits).
        regressor: Optional regressor; if provided and supported, may override
            mapping in the future. Currently unused (geometric path only).

    Returns:
        Mapping of OpenSim coordinate name → ``float32`` array of length ``T``.
    """
    _ = regressor  # Reserved for future learned mapping.
    if body_pose.ndim != 2 or body_pose.shape[1] != 63:
        raise ValueError("body_pose must have shape [T, 63]")
    t = body_pose.shape[0]
    if root_orient.shape != (t, 3) or trans.shape != (t, 3):
        raise ValueError("root_orient and trans must match body_pose time dimension")

    coords: dict[str, np.ndarray] = {}

    root_euler = R.from_rotvec(root_orient).as_euler("ZYX")
    coords["pelvis_tilt"] = root_euler[:, 0].astype(np.float32)
    coords["pelvis_list"] = root_euler[:, 1].astype(np.float32)
    coords["pelvis_rotation"] = root_euler[:, 2].astype(np.float32)
    coords["pelvis_tx"] = trans[:, 0].astype(np.float32)
    coords["pelvis_ty"] = trans[:, 1].astype(np.float32)
    coords["pelvis_tz"] = trans[:, 2].astype(np.float32)

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

    return apply_rom_limits(coords, config)

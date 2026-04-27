"""Tests for ``src.smplx_to_opensim``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.smplx_joint_regressor import get_opensim_coords
from src.smplx_to_opensim import (
    SMPLX_MOTION_DIM,
    SMPLX_SLICES,
    load_smplx_motion,
    smplx_to_mot,
)


def test_smplx_slices_motion_dim_consistent() -> None:
    keys = list(SMPLX_SLICES.keys())
    assert keys
    last = max(SMPLX_SLICES[k].stop for k in keys)
    assert last == SMPLX_MOTION_DIM == 322


def test_smplx_to_mot_writes_file(tmp_path: Path) -> None:
    motion = np.zeros((10, SMPLX_MOTION_DIM), dtype=np.float32)
    motion[:, SMPLX_SLICES["trans"]] = np.linspace(0, 1, 10)[:, None]
    cfg = {
        "paths": {"smplx_to_opensim_regressor": None},
        "dataset": {"fps": 30},
        "conversion": {
            "target_fps": 30,
            "output_fps": 30,
            "upsample_method": "none",
            "filter_order": 2,
            "filter_cutoff_hz": 10.0,
            "max_joint_velocity_rad_s": 50.0,
            "global_rom_clamp_rad": 3.14,
            "joint_rom_limits": {},
        },
    }
    out = tmp_path / "out.mot"
    mot_path, _r_align = smplx_to_mot(motion, cfg, out)
    assert mot_path == out
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "endheader" in text


def test_load_smplx_motion(tmp_path: Path) -> None:
    p = tmp_path / "m.npy"
    arr = np.zeros((5, SMPLX_MOTION_DIM), dtype=np.float32)
    np.save(p, arr)
    parts = load_smplx_motion(p)
    assert parts["trans"].shape == (5, 3)


def test_smplx_to_mot_accepts_npy_path(tmp_path: Path) -> None:
    p = tmp_path / "m.npy"
    arr = np.zeros((8, SMPLX_MOTION_DIM), dtype=np.float32)
    np.save(p, arr)
    cfg = {
        "paths": {"smplx_to_opensim_regressor": None},
        "dataset": {"fps": 30},
        "conversion": {
            "target_fps": 30,
            "output_fps": 30,
            "upsample_method": "none",
            "filter_order": 2,
            "filter_cutoff_hz": 10.0,
            "max_joint_velocity_rad_s": 50.0,
            "global_rom_clamp_rad": 3.14,
            "joint_rom_limits": {},
        },
    }
    out = tmp_path / "from_path.mot"
    _path, _r = smplx_to_mot(p, cfg, out)
    assert _path == out
    assert out.is_file()


def _minimal_regressor_cfg() -> dict:
    return {
        "paths": {"smplx_to_opensim_regressor": None},
        "conversion": {"joint_rom_limits": {}, "global_rom_clamp_rad": 3.14},
    }


def test_get_opensim_coords_t_pose_pelvis_euler_zero() -> None:
    t = 6
    body = np.zeros((t, 63), dtype=np.float32)
    root = np.zeros((t, 3), dtype=np.float32)
    trans = np.zeros((t, 3), dtype=np.float32)
    coords, _r_align = get_opensim_coords(body, root, trans, _minimal_regressor_cfg(), None)
    for name in ("pelvis_tilt", "pelvis_list", "pelvis_rotation"):
        assert np.allclose(coords[name], 0.0, atol=1e-5)
    for name in ("lumbar_extension", "lumbar_bending", "lumbar_rotation"):
        assert np.allclose(coords[name], 0.0, atol=1e-5)


def test_get_opensim_coords_alignment_removes_initial_y_rotation() -> None:
    t = 5
    body = np.zeros((t, 63), dtype=np.float32)
    root = np.zeros((t, 3), dtype=np.float32)
    root[:, :] = np.array([0.0, np.pi / 6.0, 0.0], dtype=np.float32)
    trans = np.zeros((t, 3), dtype=np.float32)
    coords, _r_align = get_opensim_coords(body, root, trans, _minimal_regressor_cfg(), None)
    assert np.isclose(float(coords["pelvis_list"][0]), 0.0, atol=1e-4)
    assert np.allclose(coords["pelvis_tilt"][0], 0.0, atol=1e-4)
    assert np.allclose(coords["pelvis_rotation"][0], 0.0, atol=1e-4)


def test_get_opensim_coords_zero_translation_at_frame_zero() -> None:
    t = 4
    body = np.zeros((t, 63), dtype=np.float32)
    root = np.zeros((t, 3), dtype=np.float32)
    trans = np.zeros((t, 3), dtype=np.float32)
    coords, _r_align = get_opensim_coords(body, root, trans, _minimal_regressor_cfg(), None)
    assert np.isclose(float(coords["pelvis_tx"][0]), 0.0, atol=1e-5)
    assert np.isclose(float(coords["pelvis_ty"][0]), 0.0, atol=1e-5)
    assert np.isclose(float(coords["pelvis_tz"][0]), 0.0, atol=1e-5)


def _cfg_with_lumbar_rom() -> dict:
    return {
        "paths": {"smplx_to_opensim_regressor": None},
        "conversion": {
            "joint_rom_limits": {
                "lumbar_extension": [-0.52, 0.87],
                "lumbar_bending": [-0.52, 0.52],
                "lumbar_rotation": [-0.35, 0.35],
            },
            "global_rom_clamp_rad": 3.14,
        },
    }


def test_get_opensim_coords_lumbar_extension_flexion_in_rom() -> None:
    """Joints 3, 6, 9 (body_pose blocks 2, 5, 8) with ~20° flexion → positive lumbar_extension."""
    t = 5
    body = np.zeros((t, 63), dtype=np.float32)
    flex = np.array([np.deg2rad(20.0), 0.0, 0.0], dtype=np.float32)
    for ji in (2, 5, 8):
        body[:, ji * 3 : ji * 3 + 3] = flex
    root = np.zeros((t, 3), dtype=np.float32)
    trans = np.zeros((t, 3), dtype=np.float32)
    coords, _r_align = get_opensim_coords(body, root, trans, _cfg_with_lumbar_rom(), None)
    ext = coords["lumbar_extension"]
    assert np.all(ext > 0.0)
    assert float(ext.min()) >= -0.52 - 1e-4
    assert float(ext.max()) <= 0.87 + 1e-4


def test_get_opensim_coords_subtalar_keys_present() -> None:
    t = 3
    body = np.zeros((t, 63), dtype=np.float32)
    root = np.zeros((t, 3), dtype=np.float32)
    trans = np.zeros((t, 3), dtype=np.float32)
    coords, _r_align = get_opensim_coords(body, root, trans, _minimal_regressor_cfg(), None)
    assert "subtalar_angle_r" in coords
    assert "subtalar_angle_l" in coords

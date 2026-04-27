"""Tests for ``src.smplx_to_opensim``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

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
    smplx_to_mot(motion, cfg, out)
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
    smplx_to_mot(p, cfg, out)
    assert out.is_file()

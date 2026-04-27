"""Tests for ``src.opensim_pipeline`` dry-run path."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.notebook_helpers import summarize_solver_metrics
from src.smplx_to_opensim import SMPLX_MOTION_DIM
from src.opensim_pipeline import _parse_so_metrics_from_lines, run_full_pipeline


def test_run_full_pipeline_dry_run(tmp_path: Path) -> None:
    mot = tmp_path / "seq.npy"
    np.save(mot, np.zeros((6, SMPLX_MOTION_DIM), dtype=np.float32))
    cfg = {
        "paths": {
            "opensim_model": "models/does_not_exist.osim",
            "temp_dir": str(tmp_path / "t"),
        },
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
    acts, names = run_full_pipeline(mot, cfg, tmp_path / "work", dry_run=True)
    assert acts.dtype == np.float32
    assert acts.shape == (6, len(names))


def test_parse_so_metrics_from_lines() -> None:
    lines = [
        "time = 3.06667 Performance = 4.94614 Constraint violation = 1396.37",
        "noise line",
        "time = 0.1 Performance = 7.2 Constraint violation = 58.4",
    ]
    metrics = _parse_so_metrics_from_lines(lines)
    assert len(metrics) == 2
    assert metrics[0]["time_s"] == 3.06667
    assert metrics[0]["performance"] == 4.94614
    assert metrics[0]["constraint_violation"] == 1396.37


def test_summarize_solver_metrics() -> None:
    metrics = [
        {"time_s": 0.0, "performance": 3.0, "constraint_violation": 10.0},
        {"time_s": 0.1, "performance": 5.0, "constraint_violation": 30.0},
    ]
    summary = summarize_solver_metrics(metrics)
    assert summary["n_frames"] == 2
    assert summary["min_constraint_violation"] == 10.0
    assert summary["max_constraint_violation"] == 30.0

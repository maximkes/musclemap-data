"""Tests for ``src.utils``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.utils import (
    get_multiprocessing_context,
    joint_velocity_clamp,
    load_config,
    resolve_against_config_dir,
    retry,
)


def _minimal_config() -> dict:
    return {
        "paths": {"motionx_root": "x", "output_root": "y", "opensim_model": "m.osim"},
        "dataset": {"motion_subdir": "m", "text_seq_subdir": "t", "fps": 30},
        "conversion": {"filter_order": 4, "output_fps": 30, "target_fps": 30},
        "ik": {},
        "rra": {},
        "static_optimization": {},
        "output": {"output_fps": 30},
        "batch": {"num_workers": 1, "max_retries": 2},
        "download": {},
        "visualization": {},
    }


def test_load_config_ok(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(_minimal_config()), encoding="utf-8")
    cfg = load_config(p)
    assert cfg["batch"]["num_workers"] == 1


def test_load_config_missing_keys(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    bad = {"paths": {}}
    p.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_config(p)
    msg = str(exc.value)
    assert "missing:batch" in msg
    assert "missing:conversion" in msg
    assert "missing:download" in msg


def test_load_config_lists_all_type_errors(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    bad = _minimal_config()
    bad["batch"]["num_workers"] = "4"
    bad["batch"]["max_retries"] = 2.5
    bad["conversion"]["filter_order"] = "4"
    p.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_config(p)
    msg = str(exc.value)
    assert "type:batch.num_workers" in msg
    assert "type:batch.max_retries" in msg
    assert "type:conversion.filter_order" in msg


def test_retry_logs_and_raises() -> None:
    calls = {"n": 0}

    @retry(max_retries=2, delay_s=0.0)
    def flaky() -> int:
        calls["n"] += 1
        raise RuntimeError("fail")

    with pytest.raises(RuntimeError):
        flaky()
    assert calls["n"] == 2


def test_joint_velocity_clamp_shapes() -> None:
    rng = np.random.default_rng(0)
    ang = np.cumsum(rng.normal(size=(50, 4)), axis=0).astype(np.float32)
    out = joint_velocity_clamp(ang, max_vel_rad_s=0.5, fps=30)
    assert out.shape == ang.shape
    assert out.dtype == ang.dtype


def test_joint_velocity_clamp_limits_velocity() -> None:
    ang = np.array([[0.0], [1.0], [3.0]], dtype=np.float32)
    out = joint_velocity_clamp(ang, max_vel_rad_s=0.5, fps=1.0)
    vel = np.diff(out[:, 0])
    assert np.all(np.abs(vel) <= 0.5 + 1e-6)
    assert out[0, 0] == pytest.approx(0.0)


def test_get_multiprocessing_context_returns_context() -> None:
    ctx = get_multiprocessing_context()
    assert hasattr(ctx, "Pool")


def test_resolve_against_config_dir(tmp_path: Path) -> None:
    cfg = tmp_path / "sub" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    out = resolve_against_config_dir(cfg, "data/motion-x")
    assert out == (tmp_path / "sub" / "data" / "motion-x").resolve()

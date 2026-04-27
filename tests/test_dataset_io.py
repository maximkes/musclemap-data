"""Tests for ``src.dataset_io``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.dataset_io import (
    MotionXSample,
    load_checkpoint,
    load_metadata,
    save_activation_sample,
    save_checkpoint,
    save_metadata,
    scan_dataset,
)
from src.smplx_to_opensim import SMPLX_MOTION_DIM


def test_save_metadata_atomic(tmp_path: Path) -> None:
    save_metadata(
        tmp_path,
        {"a": {"T": 1, "fps": 30.0, "n_muscles": 2, "source": "x", "status": "ok"}},
    )
    meta = load_metadata(tmp_path)
    assert meta["a"]["T"] == 1
    assert not (tmp_path / ".metadata.json.tmp").exists()


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "ck.json"
    save_checkpoint(p, {"x", "y"})
    assert load_checkpoint(p) == {"x", "y"}


def test_scan_dataset_warns_missing_text(tmp_path: Path) -> None:
    cfg = {
        "dataset": {
            "motion_subdir": "motion/motion_generation/smplx322",
            "text_seq_subdir": "text/semantic_label",
            "text_frame_subdir": "text/wholebody_pose_description",
        }
    }
    root = tmp_path / "mx"
    motion_dir = root / "motion/motion_generation/smplx322"
    motion_dir.mkdir(parents=True)
    np.save(motion_dir / "seq1.npy", np.zeros((3, SMPLX_MOTION_DIM), dtype=np.float32))
    samples = scan_dataset(root, cfg)
    assert len(samples) == 1
    assert isinstance(samples[0], MotionXSample)


def test_save_activation_sample_float32_assert(tmp_path: Path) -> None:
    act = np.zeros((4, 2), dtype=np.float64)
    with pytest.raises(AssertionError):
        save_activation_sample(
            tmp_path,
            "id1",
            act,
            ["m1", "m2"],
            np.zeros((4, SMPLX_MOTION_DIM), dtype=np.float32),
            {
                "semantic": "hi",
                "pose_descriptions": ["a"] * 4,
                "fps": 30.0,
                "source": "t",
                "status": "ok",
            },
        )


def test_save_activation_sample_ok(tmp_path: Path) -> None:
    act = np.zeros((2, 1), dtype=np.float32)
    smplx = np.zeros((2, SMPLX_MOTION_DIM), dtype=np.float32)
    save_activation_sample(
        tmp_path,
        "sid",
        act,
        ["m0"],
        smplx,
        {
            "semantic": "walk",
            "pose_descriptions": ["", ""],
            "fps": 30.0,
            "source": "motion-x++",
            "status": "ok",
        },
    )
    assert (tmp_path / "sid" / "activations.npy").is_file()
    meta = load_metadata(tmp_path)
    assert meta["sid"]["n_muscles"] == 1


def test_load_sample_reads_full_semantic_txt(tmp_path: Path) -> None:
    sample = MotionXSample(
        id="idea400/sample1",
        motion_path=tmp_path / "motion.npy",
        text_seq_path=tmp_path / "semantic.txt",
        text_frame_dir=None,
        source="motion-x++",
    )
    np.save(sample.motion_path, np.zeros((2, SMPLX_MOTION_DIM), dtype=np.float32))
    sample.text_seq_path.write_text(
        "The person is standing and then starts walking forward.", encoding="utf-8"
    )

    from src.dataset_io import load_sample

    loaded = load_sample(sample)
    assert loaded["semantic"] == "The person is standing and then starts walking forward."

"""End-to-end integration: one Motion-X++ clip through .mot → IK → static optimization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from src.opensim_pipeline import _read_mot_column_labels, run_ik, run_static_optimization
from src.smplx_to_opensim import SMPLX_MOTION_DIM, smplx_to_mot
from src.utils import load_config, resolve_against_config_dir
from src.visualization import get_smplx_skeleton_joints

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _REPO_ROOT / "config.yaml"
# Cap filesystem scan so huge trees stay fast; pick shortest valid clip among these.
_MAX_NPY_SCAN = 800


def _mot_numeric_body_row_count(mot_path: Path) -> int:
    """Count numeric data rows in an OpenSim .mot (after endheader and column label row)."""
    past_header = False
    past_labels = False
    n = 0
    with mot_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not past_header:
                if line.strip().lower() == "endheader":
                    past_header = True
                continue
            if not past_labels:
                past_labels = True
                continue
            if line.strip():
                n += 1
    return n


def _pick_shortest_valid_npy(search_root: Path) -> Path | None:
    """Prefer a short sequence to keep static optimization wall-clock reasonable."""
    best: tuple[int, Path] | None = None
    for i, p in enumerate(sorted(search_root.glob("**/*.npy"))):
        if i >= _MAX_NPY_SCAN:
            break
        try:
            arr = np.load(p, mmap_mode="r")
        except OSError:
            continue
        if arr.ndim != 2 or arr.shape[1] != SMPLX_MOTION_DIM:
            continue
        t = int(arr.shape[0])
        if t <= 10:
            continue
        cand = (t, Path(p))
        if best is None or t < best[0]:
            best = cand
    return best[1] if best else None


@pytest.mark.integration
def test_single_sequence_smplx_to_mot_ik_static_opt_and_aligned_skeleton(tmp_path: Path) -> None:
    pytest.importorskip("opensim")

    cfg = load_config(_CONFIG_PATH)
    paths = cfg.get("paths", {}) or {}
    ds = cfg.get("dataset", {}) or {}
    motionx = paths.get("motionx_root")
    subdir = ds.get("motion_subdir")
    if not motionx or not subdir:
        pytest.skip("config.paths.motionx_root or dataset.motion_subdir missing")

    search_root = resolve_against_config_dir(_CONFIG_PATH, motionx) / str(subdir).strip("/")
    if not search_root.is_dir():
        pytest.skip(f"Motion search directory does not exist: {search_root}")

    npy_path = _pick_shortest_valid_npy(search_root)
    if npy_path is None:
        pytest.skip(f"No valid .npy motion files (shape [T,{SMPLX_MOTION_DIM}], T>10) under {search_root}")

    motion = np.asarray(np.load(npy_path), dtype=np.float32)

    work = tmp_path / "integration_osim"
    work.mkdir(parents=True, exist_ok=True)
    coords_mot = work / "coords.mot"

    mot_path, r_align = smplx_to_mot(npy_path, cfg, coords_mot)
    assert mot_path == coords_mot
    assert mot_path.is_file()
    n_rows = _mot_numeric_body_row_count(mot_path)
    assert n_rows > 10, f"expected >10 data rows in {mot_path}, got {n_rows}"

    assert isinstance(r_align, R)
    assert r_align.as_rotvec().shape == (3,)

    model_raw = paths.get("opensim_model")
    if not model_raw:
        pytest.skip("config.paths.opensim_model missing")
    model_path = resolve_against_config_dir(_CONFIG_PATH, model_raw)
    if not model_path.is_file():
        pytest.skip(f"OpenSim model not found: {model_path}")

    ik_mot = run_ik(model_path, mot_path, work, cfg, dry_run=False)
    assert ik_mot.is_file()

    labels = _read_mot_column_labels(ik_mot)
    assert "lumbar_extension" in labels, f"IK .mot columns missing lumbar DOFs: {labels[:20]}..."

    activations, muscle_names = run_static_optimization(
        model_path, ik_mot, work, cfg, dry_run=False
    )
    assert activations.dtype == np.float32
    t, n_m = activations.shape
    assert t > 0 and n_m > 0
    assert len(muscle_names) == n_m
    assert np.all(activations >= -1e-5) and np.all(activations <= 1.0 + 1e-5)
    assert np.isfinite(activations).all()

    smplx_frame = motion[0]
    joints = get_smplx_skeleton_joints(smplx_frame, align_rotation=r_align)
    # Some Motion-X++ clips are not root-centered; compare in a pelvis-relative frame.
    joints = joints - joints[0:1]
    assert float(np.linalg.norm(joints[0])) < 1e-6
    assert float(joints[1, 1]) < float(joints[12, 1])

"""Notebook-only helper functions for exploratory sections."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from src.opensim_pipeline import run_full_pipeline
from src.smplx_mesh_preview import show_smplx_mesh_preview
from src.smplx_to_opensim import SMPLX_MOTION_DIM, smplx_to_mot
from src.visualization import animate_motion_interactive


def build_smplx_mesh_preview(smplx_motion: np.ndarray, config: dict[str, object]) -> Any:
    """Show a shaded SMPL-X mesh in the notebook (pyrender off-screen).

    Requires ``paths.smplx_model_folder`` and optional mesh dependencies
    (``poetry install --with mesh``). Delegates to ``show_smplx_mesh_preview`` in
    ``src.smplx_mesh_preview``.

    Args:
        smplx_motion: Motion array with shape ``[T, D]`` (``D`` matches ``SMPLX_SLICES``).
        config: Project configuration dictionary.

    Returns:
        ``ipywidgets.VBox`` after displaying the preview UI.
    """
    if smplx_motion.ndim != 2 or smplx_motion.shape[1] != SMPLX_MOTION_DIM:
        raise ValueError(f"smplx_motion must have shape [T, {SMPLX_MOTION_DIM}]")
    if smplx_motion.shape[0] < 1:
        raise ValueError("smplx_motion must have at least one frame")
    return show_smplx_mesh_preview(smplx_motion, config)


def build_original_motion_doll_animation(
    smplx_motion: np.ndarray, config: dict[str, object]
) -> Any:
    """Build a motion-only doll animation from raw SMPL-X motion.

    Args:
        smplx_motion: Motion array with shape ``[T, D]`` (``D`` matches ``SMPLX_SLICES``).
        config: Project configuration dictionary.

    Returns:
        ``matplotlib.animation.FuncAnimation``. With the inline backend the figure is
        closed after the JS embed so only one player appears in the output.
    """
    if smplx_motion.ndim != 2 or smplx_motion.shape[1] != SMPLX_MOTION_DIM:
        raise ValueError(f"smplx_motion must have shape [T, {SMPLX_MOTION_DIM}]")
    if smplx_motion.shape[0] < 1:
        raise ValueError("smplx_motion must have at least one frame")
    tmp_dir = Path(tempfile.mkdtemp(prefix="motion_doll_"))
    mot_path = tmp_dir / "motion_only_coords.mot"
    smplx_to_mot(smplx_motion, config, mot_path)
    dummy_acts = np.zeros((smplx_motion.shape[0], 1), dtype=np.float32)
    return animate_motion_interactive(
        mot_path,
        dummy_acts,
        ["motion_only"],
        config,
    )


def format_sequence_description(sample_id: str, semantic_text: str) -> str:
    """Format sequence-level description into a printable report.

    Args:
        sample_id: Sequence identifier.
        semantic_text: Sequence-level semantic label.

    Returns:
        Printable multi-line string report.
    """
    semantic = semantic_text.strip() if semantic_text.strip() else "<empty>"
    lines = [f"Sequence ID: {sample_id}", "", "Sequence-level description:", semantic]
    return "\n".join(lines)


def run_pipeline_with_progress(
    smplx_npy: Path,
    config: dict[str, Any],
    tmp_dir: Path,
    dry_run: bool = False,
    save_solver_log: bool = False,
    save_solver_metrics: bool = False,
    expected_frames: int | None = None,
) -> tuple[np.ndarray, list[str], list[dict[str, float]]]:
    """Run one sequence with suppressed solver spam and notebook progress updates.

    Args:
        smplx_npy: Input motion file path.
        config: Full project config.
        tmp_dir: Temporary sequence directory.
        dry_run: Skip OpenSim when true.
        save_solver_log: Keep raw solver log file in sequence temp directory.
        save_solver_metrics: Persist parsed SO metrics to JSON/CSV in temp directory.
        expected_frames: Optional expected frame count used as progress-bar upper bound.

    Returns:
        Tuple of activations, muscle names, and parsed solver metrics.
    """
    cfg = copy.deepcopy(config)
    so = cfg.setdefault("static_optimization", {})
    so["log_capture_mode"] = "progress_only"
    so["save_solver_log"] = bool(save_solver_log)
    so["save_solver_metrics"] = bool(save_solver_metrics)
    metrics: list[dict[str, float]] = []
    total = int(expected_frames) if expected_frames is not None and expected_frames > 0 else None
    pbar = tqdm(total=total, desc="SO frames", unit="frame")

    def _on_metric(_: dict[str, float]) -> None:
        pbar.update(1)

    try:
        activations, muscle_names = run_full_pipeline(
            smplx_npy,
            cfg,
            tmp_dir,
            dry_run=dry_run,
            progress_callback=_on_metric,
            so_metrics_out=metrics,
        )
    finally:
        pbar.close()
    return activations, muscle_names, metrics


def summarize_solver_metrics(metrics: list[dict[str, float]]) -> dict[str, float | int]:
    """Summarize parsed static-optimization frame metrics."""
    if not metrics:
        return {
            "n_frames": 0,
            "min_constraint_violation": float("nan"),
            "median_constraint_violation": float("nan"),
            "max_constraint_violation": float("nan"),
            "mean_performance": float("nan"),
        }
    constraints = np.array([m["constraint_violation"] for m in metrics], dtype=np.float64)
    performance = np.array([m["performance"] for m in metrics], dtype=np.float64)
    return {
        "n_frames": int(len(metrics)),
        "min_constraint_violation": float(np.min(constraints)),
        "median_constraint_violation": float(np.median(constraints)),
        "max_constraint_violation": float(np.max(constraints)),
        "mean_performance": float(np.mean(performance)),
    }


def write_solver_metrics_artifact(path: Path, metrics: list[dict[str, float]]) -> None:
    """Persist solver metrics as JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

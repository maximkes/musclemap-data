"""OpenSim IK → optional RRA → static optimization pipeline."""

from __future__ import annotations

import logging
import multiprocessing
import os
import re
import shutil
import traceback
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.smplx_joint_regressor import get_opensim_coords
from src.smplx_to_opensim import SMPLX_MOTION_DIM, smplx_to_mot
from src.utils import get_multiprocessing_context

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SO_METRIC_RE = re.compile(
    r"time\s*=\s*([\-+0-9.eE]+)\s+Performance\s*=\s*([\-+0-9.eE]+)\s+Constraint violation\s*=\s*([\-+0-9.eE]+)"
)


def _parse_so_metrics_from_lines(lines: list[str]) -> list[dict[str, float]]:
    """Parse Static Optimization progress rows from OpenSim text output."""
    metrics: list[dict[str, float]] = []
    for line in lines:
        m = _SO_METRIC_RE.search(line)
        if not m:
            continue
        metrics.append(
            {
                "time_s": float(m.group(1)),
                "performance": float(m.group(2)),
                "constraint_violation": float(m.group(3)),
            }
        )
    return metrics


@contextmanager
def _capture_fd_output(log_path: Path, enabled: bool):
    """Capture C-level stdout/stderr to a file (for OpenSim spam suppression)."""
    if not enabled:
        yield
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as _:
        pass
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    with open(log_path, "a", encoding="utf-8") as log_file:
        try:
            os.dup2(log_file.fileno(), 1)
            os.dup2(log_file.fileno(), 2)
            yield
        finally:
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)


def _run_static_opt_worker(
    xml_path_str: str, error_path_str: str, log_path_str: str, capture_output: bool
) -> None:
    """Run OpenSim AnalyzeTool in a child process and persist traceback on failure."""
    error_path = Path(error_path_str)
    log_path = Path(log_path_str)
    try:
        opensim = _import_opensim()
        with _capture_fd_output(log_path, capture_output):
            tool = opensim.AnalyzeTool(xml_path_str)
            tool.run()
    except Exception:  # noqa: BLE001
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise


def _resolve_model_path(config: dict[str, Any]) -> Path:
    paths = config.get("paths", {}) or {}
    raw_cfg = paths.get("opensim_model")
    if not raw_cfg:
        raise ValueError("Missing required config path: paths.opensim_model")
    raw = Path(str(raw_cfg))
    if raw.is_file():
        return raw.resolve()
    candidate = (_REPO_ROOT / raw).resolve()
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"OpenSim model not found: {raw}")


def _import_opensim():  # pragma: no cover - optional dependency
    try:
        import opensim  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenSim not found. Run python scripts/setup_check.py") from exc
    return opensim


def _log_opensim_errors(opensim: Any, stage: str) -> None:
    logger.error("OpenSim stage '%s' failed; dumping OpenSim error reporter.", stage)
    try:
        rep = opensim.IO.getErrorReporter()
        if rep and hasattr(rep, "dump"):
            rep.dump()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not dump OpenSim error reporter (%s): %s", stage, exc)


def _parse_template(name: str) -> ET.ElementTree:
    path = _TEMPLATES_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing template: {path}")
    return ET.parse(path)


def _write_xml(tree: ET.ElementTree, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def get_muscle_names(
    model_path: Path, dry_run: bool = False, config: dict[str, Any] | None = None
) -> list[str]:
    """Return muscle names for the given OpenSim model."""
    if dry_run:
        cfg = config or {}
        try:
            return _load_muscle_names_from_output_root(cfg)
        except Exception:
            zero_motion = np.zeros((1, SMPLX_MOTION_DIM), dtype=np.float32)
            coord_names = sorted(
                get_opensim_coords(
                    zero_motion[:, 3:66], zero_motion[:, 0:3], zero_motion[:, 309:312], cfg, None
                ).keys()
            )
            return [f"synthetic_muscle_{name}" for name in coord_names]
    opensim = _import_opensim()
    resolved = model_path if model_path.is_file() else (_REPO_ROOT / model_path).resolve()
    model = opensim.Model(str(resolved))
    model.initSystem()
    muscle_set = model.getMuscles()
    return [muscle_set.get(i).getName() for i in range(muscle_set.getSize())]


def _load_muscle_names_from_output_root(config: dict[str, Any]) -> list[str]:
    """Load muscle names from output dataset if available."""
    paths = config.get("paths", {}) or {}
    output_root_raw = paths.get("output_root")
    if not output_root_raw:
        raise RuntimeError("Cannot infer muscle names: config.paths.output_root is not set.")
    output_root = Path(str(output_root_raw))
    if not output_root.is_absolute():
        output_root = (_REPO_ROOT / output_root).resolve()
    names_path = output_root / "muscle_names.json"
    if not names_path.is_file():
        raise RuntimeError(
            "Cannot infer muscle names for dry_run. Expected file at "
            f"{names_path}. Generate one sample first or provide a model with OpenSim."
        )
    import json

    names = json.loads(names_path.read_text(encoding="utf-8"))
    if not isinstance(names, list) or not names:
        raise RuntimeError(f"Invalid muscle_names.json at {names_path}")
    return [str(x) for x in names]


def _mot_time_range(mot_path: Path, opensim: Any) -> tuple[float, float]:
    storage = opensim.Storage(str(mot_path))
    return float(storage.getFirstTime()), float(storage.getLastTime())


def _read_mot_column_labels(mot_path: Path) -> list[str]:
    with mot_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip().lower() == "endheader":
                header = f.readline().strip().split("\t")
                return header
    raise ValueError(f"Could not read column labels from {mot_path}")


def _build_ik_xml(
    model_path: Path,
    mot_path: Path,
    out_mot: Path,
    xml_path: Path,
    config: dict[str, Any],
    opensim: Any,
) -> None:
    tree = _parse_template("ik_setup.xml")
    root = tree.getroot()
    ik = root.find(".//InverseKinematicsTool")
    if ik is None:
        raise ValueError("ik_setup.xml must contain InverseKinematicsTool")

    t0, t1 = _mot_time_range(mot_path, opensim)

    def set_text(tag: str, text: str) -> None:
        el = ik.find(tag)
        if el is None:
            el = ET.SubElement(ik, tag)
        el.text = text

    set_text("model_file", str(model_path))
    # Keep both tags populated for compatibility with template/OSim variants.
    # An empty `coordinates_file` tag can cause IK to parse an empty filename.
    set_text("coordinate_file", str(mot_path))
    set_text("coordinates_file", str(mot_path))
    set_text("output_motion_file", str(out_mot))
    set_text("time_range", f"{t0} {t1}")

    ik_cfg = config.get("ik", {}) or {}
    weight = float(ik_cfg.get("coordinate_weight", 20.0))
    accuracy = float(ik_cfg.get("accuracy", 1.0e-5))
    max_iter = int(ik_cfg.get("max_iterations", 500))

    set_text("accuracy", str(accuracy))
    set_text("max_iterations", str(max_iter))

    task_set = ik.find("IKTaskSet")
    if task_set is None:
        task_set = ET.SubElement(ik, "IKTaskSet")
    objects = task_set.find("objects")
    if objects is None:
        objects = ET.SubElement(task_set, "objects")
    for child in list(objects):
        objects.remove(child)

    try:
        storage = opensim.Storage(str(mot_path))
        labels = []
        for i in range(storage.getColumnLabels().getSize()):
            labels.append(storage.getColumnLabels().get(i))
    except Exception:
        labels = _read_mot_column_labels(mot_path)

    model = opensim.Model(str(model_path))
    model.initSystem()
    coord_set = model.getCoordinateSet()
    model_coords = {coord_set.get(i).getName() for i in range(coord_set.getSize())}

    for label in labels:
        if label == "time":
            continue
        if label not in model_coords:
            continue
        task = ET.SubElement(objects, "IKCoordinateTask")
        task.set("name", label)
        ET.SubElement(task, "apply").text = "true"
        ET.SubElement(task, "weight").text = str(weight)

    _write_xml(tree, xml_path)


def run_ik(
    model_path: Path,
    mot_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    dry_run: bool = False,
) -> Path:
    """Run inverse kinematics; return path to resulting ``.mot``."""
    if dry_run:
        return mot_path
    if not (config.get("ik", {}) or {}).get("enabled", True):
        return mot_path
    opensim = _import_opensim()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_mot = output_dir / "ik_results.mot"
    xml_path = output_dir / "ik_setup.xml"
    _build_ik_xml(model_path, mot_path, out_mot, xml_path, config, opensim)
    try:
        tool = opensim.InverseKinematicsTool(str(xml_path))
        tool.run()
    except Exception as exc:
        _log_opensim_errors(opensim, "ik")
        raise RuntimeError(
            f"OpenSim stage 'ik' failed for sequence '{output_dir.name}': {exc}"
        )
    if not out_mot.is_file():
        raise RuntimeError(
            f"OpenSim stage 'ik' failed for sequence '{output_dir.name}': "
            f"missing output motion file {out_mot}"
        )
    return out_mot


def _build_rra_xml(
    model_path: Path,
    kin_mot: Path,
    out_mot: Path,
    xml_path: Path,
    config: dict[str, Any],
    opensim: Any,
) -> None:
    tree = _parse_template("rra_setup.xml")
    root = tree.getroot()
    rra = root.find(".//RRATool")
    if rra is None:
        raise ValueError("rra_setup.xml must contain RRATool")
    t0, t1 = _mot_time_range(kin_mot, opensim)

    def set_text(parent: ET.Element, tag: str, text: str) -> None:
        el = parent.find(tag)
        if el is None:
            el = ET.SubElement(parent, tag)
        el.text = text

    set_text(rra, "model_file", str(model_path))
    set_text(rra, "coordinates_file", str(kin_mot))
    set_text(rra, "output_motion_file", str(out_mot))
    set_text(rra, "time_range", f"{t0} {t1}")
    rra_cfg = config.get("rra", {}) or {}
    set_text(
        rra,
        "desired_kinematics_file",
        str(kin_mot),
    )
    set_text(rra, "lowpass_cutoff_frequency", str(float(rra_cfg.get("filter_cutoff_hz", 6.0))))
    _write_xml(tree, xml_path)


def run_rra(
    model_path: Path,
    ik_mot: Path,
    output_dir: Path,
    config: dict[str, Any],
    dry_run: bool = False,
) -> Path:
    """Run RRA if enabled; otherwise return ``ik_mot`` unchanged."""
    if dry_run:
        return ik_mot
    if not (config.get("rra", {}) or {}).get("enabled", False):
        return ik_mot
    opensim = _import_opensim()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_mot = output_dir / "rra_adjusted.mot"
    xml_path = output_dir / "rra_setup.xml"
    _build_rra_xml(model_path, ik_mot, out_mot, xml_path, config, opensim)
    try:
        tool = opensim.RRATool(str(xml_path))
        tool.run()
    except Exception as exc:
        _log_opensim_errors(opensim, "rra")
        raise RuntimeError(
            f"OpenSim stage 'rra' failed for sequence '{output_dir.name}': {exc}"
        )
    if not out_mot.is_file():
        raise RuntimeError(
            f"OpenSim stage 'rra' failed for sequence '{output_dir.name}': "
            f"missing output motion file {out_mot}"
        )
    return out_mot


def _build_static_opt_xml(
    model_path: Path,
    kin_mot: Path,
    results_dir: Path,
    xml_path: Path,
    config: dict[str, Any],
    opensim: Any,
) -> None:
    tree = _parse_template("static_opt_setup.xml")
    root = tree.getroot()
    tool_el = root.find(".//AnalyzeTool")
    if tool_el is None:
        raise ValueError("static_opt_setup.xml must contain AnalyzeTool")
    t0, t1 = _mot_time_range(kin_mot, opensim)

    def set_text(parent: ET.Element, tag: str, text: str) -> None:
        el = parent.find(tag)
        if el is None:
            el = ET.SubElement(parent, tag)
        el.text = text

    set_text(tool_el, "model_file", str(model_path))
    set_text(tool_el, "coordinates_file", str(kin_mot))
    set_text(tool_el, "initial_time", str(t0))
    set_text(tool_el, "final_time", str(t1))
    set_text(tool_el, "results_directory", str(results_dir))

    so = config.get("static_optimization", {}) or {}
    # Configure StaticOptimization as an explicit analysis in AnalyzeTool.
    # Without this AnalysisSet entry, AnalyzeTool errors with:
    # "ERROR- no analyses have been set."
    analysis_set = tool_el.find("AnalysisSet")
    if analysis_set is None:
        analysis_set = ET.SubElement(tool_el, "AnalysisSet")
    objects = analysis_set.find("objects")
    if objects is None:
        objects = ET.SubElement(analysis_set, "objects")
    for child in list(objects):
        objects.remove(child)

    static_opt = ET.SubElement(objects, "StaticOptimization")
    static_opt.set("name", "static_optimization")
    ET.SubElement(static_opt, "on").text = "true"
    ET.SubElement(static_opt, "in_degrees").text = "false"
    ET.SubElement(static_opt, "start_time").text = str(t0)
    ET.SubElement(static_opt, "end_time").text = str(t1)
    ET.SubElement(static_opt, "step_interval").text = "1"
    ET.SubElement(static_opt, "use_model_force_set").text = "true"
    ET.SubElement(static_opt, "activation_exponent").text = str(
        int(so.get("activation_exponent", 2))
    )
    ET.SubElement(static_opt, "use_muscle_physiology").text = str(
        bool(so.get("use_muscle_physiology", True))
    ).lower()

    _write_xml(tree, xml_path)


def _parse_activation_sto(sto_path: Path, muscle_names: list[str]) -> np.ndarray:
    rows: list[list[float]] = []
    with open(sto_path, encoding="utf-8") as f:
        for line in f:
            if line.strip().lower().startswith("endheader"):
                break
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            parts = line.split()
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                # OpenSim STO commonly has a post-header column label row like:
                # "time muscleA muscleB ...". Skip such non-numeric lines.
                continue
    if not rows:
        raise ValueError(f"Empty STO file: {sto_path}")
    data = np.asarray(rows, dtype=np.float64)
    # Assume time in column 0; remaining columns align with muscle_names order if possible
    n_expected = len(muscle_names) + 1
    if data.shape[1] < 2:
        raise ValueError(f"Unexpected STO shape {data.shape} for {sto_path}")
    if data.shape[1] >= n_expected:
        act = data[:, 1 : len(muscle_names) + 1]
    else:
        act = data[:, 1:]
    return act.astype(np.float32)


def run_static_optimization(
    model_path: Path,
    kinematics_mot: Path,
    output_dir: Path,
    config: dict[str, Any],
    dry_run: bool = False,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
    metrics_out: list[dict[str, float]] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Run static optimization; return activations ``[T, N]`` and muscle names."""
    if dry_run:
        # Time length is inferred from kinematics rows (excluding header).
        rows = 0
        with open(kinematics_mot, encoding="utf-8") as f:
            in_body = False
            for line in f:
                if in_body:
                    if line.strip():
                        rows += 1
                    continue
                if line.strip().lower() == "endheader":
                    _ = f.readline()  # column labels
                    in_body = True
        muscle_names = get_muscle_names(model_path, dry_run=True, config=config)
        return np.random.rand(rows, len(muscle_names)).astype(np.float32), muscle_names
    opensim = _import_opensim()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "so_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    xml_path = output_dir / "static_opt_setup.xml"
    _build_static_opt_xml(model_path, kinematics_mot, results_dir, xml_path, config, opensim)
    muscle_names = get_muscle_names(model_path)
    so_cfg = config.get("static_optimization", {}) or {}
    timeout_s_raw = so_cfg.get("timeout_s")
    timeout_s = float(timeout_s_raw) if timeout_s_raw is not None else None
    capture_mode = str(so_cfg.get("log_capture_mode", "console")).lower()
    capture_output = capture_mode == "progress_only"
    save_solver_log = bool(so_cfg.get("save_solver_log", False))
    save_solver_metrics = bool(so_cfg.get("save_solver_metrics", False))
    solver_log_filename = str(so_cfg.get("solver_log_filename", "so_solver.log"))
    metrics_json_name = str(so_cfg.get("solver_metrics_json", "so_metrics.json"))
    metrics_csv_name = str(so_cfg.get("solver_metrics_csv", "so_metrics.csv"))
    log_path = output_dir / solver_log_filename
    parsed_metrics: list[dict[str, float]] = []
    try:
        # multiprocessing.Pool workers are daemonic and cannot spawn children.
        # In that case, run AnalyzeTool inline and skip process-based timeout.
        can_spawn_child = not multiprocessing.current_process().daemon
        use_timeout_subprocess = can_spawn_child and timeout_s is not None and timeout_s > 0
        if use_timeout_subprocess:
            err_path = output_dir / "so_worker_error.txt"
            if err_path.exists():
                err_path.unlink()
            ctx = get_multiprocessing_context()
            proc = ctx.Process(
                target=_run_static_opt_worker,
                args=(str(xml_path), str(err_path), str(log_path), capture_output),
            )
            proc.start()
            observed = 0
            while proc.is_alive():
                proc.join(timeout=0.2)
                if capture_output and log_path.is_file():
                    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                    fresh = _parse_so_metrics_from_lines(lines)
                    if len(fresh) > observed:
                        for metric in fresh[observed:]:
                            if progress_callback is not None:
                                progress_callback(metric)
                        observed = len(fresh)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5.0)
                raise TimeoutError(f"static optimization timed out after {timeout_s:.1f}s")
            if proc.exitcode != 0:
                details = ""
                if err_path.is_file():
                    details = err_path.read_text(encoding="utf-8", errors="ignore")
                msg = "Static optimization worker failed."
                if details:
                    msg = f"{msg}\n{details}"
                raise RuntimeError(msg)
            if capture_output and log_path.is_file():
                parsed_metrics = _parse_so_metrics_from_lines(
                    log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                )
        else:
            with _capture_fd_output(log_path, capture_output):
                tool = opensim.AnalyzeTool(str(xml_path))
                tool.run()
            if capture_output and log_path.is_file():
                parsed_metrics = _parse_so_metrics_from_lines(
                    log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                )
                if progress_callback is not None:
                    for metric in parsed_metrics:
                        progress_callback(metric)
    except Exception as exc:
        _log_opensim_errors(opensim, "static_optimization")
        raise RuntimeError(
            f"OpenSim stage 'static_optimization' failed for sequence '{output_dir.name}': {exc}"
        ) from exc
    finally:
        if capture_output and log_path.is_file() and not save_solver_log:
            try:
                log_path.unlink()
            except OSError:
                logger.debug("Could not remove temporary solver log %s", log_path)

    sto_candidates = sorted(results_dir.glob("*activation*.sto"))
    if not sto_candidates:
        sto_candidates = sorted(results_dir.glob("*.sto"))
    if not sto_candidates:
        raise RuntimeError("Static optimization produced no STO output.")
    sto_path = sto_candidates[0]
    activations = _parse_activation_sto(sto_path, muscle_names)
    if activations.shape[1] != len(muscle_names):
        logger.warning(
            "Activation columns (%s) != muscle count (%s); trimming/padding not applied.",
            activations.shape[1],
            len(muscle_names),
        )
    if metrics_out is not None:
        metrics_out.clear()
        metrics_out.extend(parsed_metrics)
    if parsed_metrics and save_solver_metrics:
        import json

        (output_dir / metrics_json_name).write_text(
            json.dumps(parsed_metrics, indent=2),
            encoding="utf-8",
        )
        csv_lines = ["time_s,performance,constraint_violation"]
        csv_lines.extend(
            f"{m['time_s']},{m['performance']},{m['constraint_violation']}" for m in parsed_metrics
        )
        (output_dir / metrics_csv_name).write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    return activations, muscle_names


def run_full_pipeline(
    smplx_npy: Path,
    config: dict[str, Any],
    tmp_dir: Path,
    dry_run: bool = False,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
    so_metrics_out: list[dict[str, float]] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """End-to-end OpenSim processing for one SMPL-X motion file.

    Args:
        smplx_npy: Path to ``[T, D]`` motion ``.npy`` (``D = SMPLX_MOTION_DIM``).
        config: Full configuration.
        tmp_dir: Temporary working directory for this sequence.
        dry_run: If ``True``, skip OpenSim and return random activations.

    Returns:
        Tuple ``(activations float32 [T,N], muscle_names)``.
    """
    motion = np.load(smplx_npy)
    if motion.ndim != 2 or motion.shape[1] != SMPLX_MOTION_DIM:
        raise ValueError(f"Expected [T,{SMPLX_MOTION_DIM}] at {smplx_npy}, got {motion.shape}")
    t = int(motion.shape[0])
    seq_tmp = Path(tmp_dir)
    if dry_run:
        try:
            model_path = _resolve_model_path(config)
            names = get_muscle_names(model_path)
        except Exception:
            try:
                names = _load_muscle_names_from_output_root(config)
            except Exception:
                motion_f32 = motion.astype(np.float32, copy=False)
                body_pose = motion_f32[:, 3:66]
                root_orient = motion_f32[:, 0:3]
                trans = motion_f32[:, 309:312]
                coord_names = sorted(
                    get_opensim_coords(body_pose, root_orient, trans, config, None).keys()
                )
                names = [f"synthetic_from_coords_{name}" for name in coord_names]
        return np.random.rand(t, len(names)).astype(np.float32), names

    model_path = _resolve_model_path(config)
    seq_tmp.mkdir(parents=True, exist_ok=True)
    mot_path = seq_tmp / "coords.mot"
    try:
        smplx_to_mot(motion, config, mot_path)
        ik_mot = run_ik(model_path, mot_path, seq_tmp, config, dry_run=dry_run)
        kin_mot = run_rra(model_path, ik_mot, seq_tmp, config, dry_run=dry_run)
        activations, muscle_names = run_static_optimization(
            model_path,
            kin_mot,
            seq_tmp,
            config,
            dry_run=dry_run,
            progress_callback=progress_callback,
            metrics_out=so_metrics_out,
        )
        if activations.dtype != np.float32:
            activations = activations.astype(np.float32, copy=False)
        shutil.rmtree(seq_tmp, ignore_errors=True)
        return activations, muscle_names
    except Exception as exc:
        logger.exception("OpenSim pipeline failed for %s", smplx_npy)
        tb = traceback.format_exc()
        logger.error("Traceback:\n%s", tb)
        raise RuntimeError(f"OpenSim pipeline failed for {smplx_npy.name}") from exc

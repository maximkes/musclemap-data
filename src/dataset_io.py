"""Motion-X++ dataset scanning, I/O, checkpoints, and output metadata."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.smplx_to_opensim import SMPLX_MOTION_DIM

logger = logging.getLogger(__name__)

_TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


def _read_text_file(path: Path) -> str:
    """Read a text file; tolerate Motion-X++ exports that are UTF-8, Windows-1252, or Latin-1."""
    data = path.read_bytes()
    for enc in _TEXT_ENCODINGS:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _semantic_from_npy_array(arr: np.ndarray) -> str:
    """Turn a loaded ndarray into one sequence-level string."""
    if arr.size == 0:
        return ""
    flat = np.ravel(arr)
    parts: list[str] = []
    for x in flat:
        if hasattr(x, "item"):
            x = x.item()
        parts.append(str(x))
    return " ".join(parts).strip()


@dataclass
class MotionXSample:
    """One Motion-X++ sequence with motion and optional text paths."""

    id: str
    motion_path: Path
    text_seq_path: Path
    text_frame_dir: Optional[Path]
    source: str


@contextmanager
def _dir_lock(lock_dir: Path, timeout_s: float = 30.0, poll_s: float = 0.05):
    """Acquire a simple cross-process lock via atomic directory creation."""
    start = time.monotonic()
    while True:
        try:
            lock_dir.mkdir(parents=False, exist_ok=False)
            break
        except FileExistsError:
            if (time.monotonic() - start) > timeout_s:
                raise TimeoutError(f"Timed out waiting for lock: {lock_dir}")
            time.sleep(poll_s)
    try:
        yield
    finally:
        try:
            lock_dir.rmdir()
        except OSError:
            logger.warning("Failed to remove lock dir %s", lock_dir)


def scan_dataset(root: Path, config: dict[str, Any]) -> list[MotionXSample]:
    """Discover motion ``.npy`` files under the dataset root.

    Args:
        root: Motion-X++ dataset root (``paths.motionx_root``).
        config: Full configuration (uses ``dataset`` section).

    Returns:
        List of ``MotionXSample`` records.
    """
    ds = config.get("dataset", {}) or {}
    motion_rel = ds.get("motion_subdir", "motion/motion_generation/smplx322")
    text_seq_rel = ds.get("text_seq_subdir", "text/semantic_label")
    text_frame_rel = ds.get("text_frame_subdir", "text/wholebody_pose_description")

    motion_dir = root / motion_rel
    text_seq_dir = root / text_seq_rel
    text_frame_dir = root / text_frame_rel

    samples: list[MotionXSample] = []
    if not motion_dir.is_dir():
        logger.warning("Motion directory does not exist: %s", motion_dir)
        return samples

    for npy in sorted(motion_dir.rglob("*.npy")):
        rel = npy.relative_to(motion_dir)
        stem = npy.stem
        text_seq = text_seq_dir / rel.parent / f"{stem}.npy"
        if not text_seq.is_file():
            text_seq_txt = text_seq_dir / rel.parent / f"{stem}.txt"
            text_seq = text_seq_txt if text_seq_txt.is_file() else text_seq
        if not text_seq.is_file():
            logger.warning("Missing semantic text for motion: %s", npy)

        frame_dir: Optional[Path] = None
        candidate = text_frame_dir / rel.parent / stem
        if candidate.is_dir():
            frame_dir = candidate

        sample_id = str(rel.with_suffix(""))
        samples.append(
            MotionXSample(
                id=sample_id,
                motion_path=npy,
                text_seq_path=text_seq,
                text_frame_dir=frame_dir,
                source="motion-x++",
            )
        )
    return samples


def load_sample(sample: MotionXSample) -> dict[str, Any]:
    """Load motion array and text payloads for one sample.

    Args:
        sample: Dataset sample descriptor.

    Returns:
        Dict with keys ``motion`` (``float32`` ``[T, D]`` with ``D = SMPLX_MOTION_DIM``),
        ``semantic`` (``str``), and optional ``pose_descriptions`` (``list[str]`` length ``T``).
    """
    motion = np.load(sample.motion_path)
    if motion.ndim != 2 or motion.shape[1] != SMPLX_MOTION_DIM:
        raise ValueError(f"Bad motion shape for {sample.motion_path}: {motion.shape}")
    motion = motion.astype(np.float32, copy=False)

    semantic = ""
    if sample.text_seq_path.is_file():
        suf = sample.text_seq_path.suffix.lower()
        if suf == ".txt":
            semantic = _read_text_file(sample.text_seq_path).strip()
        elif suf == ".npy":
            # Motion-X++ sequence semantics are often ``np.save``'d arrays (magic begins with 0x93).
            try:
                raw = np.load(sample.text_seq_path, allow_pickle=True)
                if isinstance(raw, np.ndarray):
                    semantic = _semantic_from_npy_array(raw)
                else:
                    semantic = str(raw).strip()
            except (OSError, ValueError) as exc:
                logger.warning(
                    "np.load failed for semantic %s (%s); trying text fallback",
                    sample.text_seq_path,
                    exc,
                )
                semantic = _read_text_file(sample.text_seq_path).strip()
        else:
            try:
                txt_body = _read_text_file(sample.text_seq_path)
                txt_arr = np.loadtxt(StringIO(txt_body), dtype=str)
                if isinstance(txt_arr, np.ndarray):
                    semantic = (
                        " ".join(str(x) for x in np.ravel(txt_arr)).strip()
                        if txt_arr.size > 0
                        else ""
                    )
                else:
                    semantic = str(txt_arr).strip()
            except (OSError, ValueError):
                semantic = _read_text_file(sample.text_seq_path).strip()

    pose_desc: list[str] = []
    t = motion.shape[0]
    if sample.text_frame_dir is not None and sample.text_frame_dir.is_dir():
        for i in range(t):
            fp = sample.text_frame_dir / f"{i}.txt"
            if fp.is_file():
                pose_desc.append(_read_text_file(fp).strip())
            else:
                pose_desc.append("")
    else:
        pose_desc = [""] * t

    return {
        "motion": motion,
        "semantic": semantic,
        "pose_descriptions": pose_desc,
    }


def load_metadata(output_root: Path) -> dict[str, Any]:
    """Load ``metadata.json`` if present."""
    path = output_root / "metadata.json"
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.warning("metadata.json is corrupted at %s; resetting to empty metadata.", path)
        return {}


def save_metadata(output_root: Path, meta: dict[str, Any]) -> None:
    """Atomically write ``metadata.json`` under ``output_root``."""
    output_root.mkdir(parents=True, exist_ok=True)
    tmp = output_root / ".metadata.json.tmp"
    final = output_root / "metadata.json"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final)


def upsert_metadata_entry(
    output_root: Path,
    sample_id: str,
    entry: dict[str, Any],
) -> None:
    """Atomically merge one sample metadata entry with a process lock."""
    lock = output_root / ".metadata.lock"
    with _dir_lock(lock):
        meta = load_metadata(output_root)
        meta[sample_id] = entry
        save_metadata(output_root, meta)


def save_activation_sample(
    output_root: Path,
    sample_id: str,
    activations: np.ndarray,
    muscle_names: list[str],
    smplx: np.ndarray,
    texts: dict[str, Any],
) -> None:
    """Write one processed sequence to the output dataset layout.

    Args:
        output_root: Dataset root (``paths.output_root``).
        sample_id: Sequence identifier (directory name).
        activations: Array ``[T, N_muscles]`` (must be ``float32``).
        muscle_names: Muscle column names (length ``N_muscles``).
        smplx: Passthrough motion ``[T, D]`` with ``D = SMPLX_MOTION_DIM`` (saved as ``float32``).
        texts: Must include ``semantic`` (``str``) and ``pose_descriptions``
            (``Sequence[str]`` length ``T``).
    """
    if activations.dtype != np.float32:
        raise AssertionError("activations must be float32 before save")
    if activations.ndim != 2 or activations.shape[1] != len(muscle_names):
        raise ValueError("activations shape must match muscle_names length")
    if smplx.dtype != np.float32:
        smplx = smplx.astype(np.float32, copy=False)
    if smplx.ndim != 2 or smplx.shape[1] != SMPLX_MOTION_DIM:
        raise ValueError(f"smplx must be [T, {SMPLX_MOTION_DIM}], got {smplx.shape}")

    seq_dir = output_root / sample_id
    seq_dir.mkdir(parents=True, exist_ok=True)
    np.save(seq_dir / "activations.npy", activations)
    np.save(seq_dir / "smplx_322.npy", smplx)

    semantic = str(texts.get("semantic", ""))
    (seq_dir / "semantic_label.txt").write_text(semantic.strip() + "\n", encoding="utf-8")

    pd = texts.get("pose_descriptions") or []
    pose_dir = seq_dir / "pose_descriptions"
    pose_dir.mkdir(parents=True, exist_ok=True)
    for i, line in enumerate(pd):
        (pose_dir / f"{i}.txt").write_text(str(line) + "\n", encoding="utf-8")

    names_path = output_root / "muscle_names.json"
    payload = list(muscle_names)
    write_names = True
    if names_path.is_file():
        try:
            existing = json.loads(names_path.read_text(encoding="utf-8"))
            if existing == payload:
                write_names = False
        except (OSError, json.JSONDecodeError):
            write_names = True
    if write_names:
        tmp = output_root / f".muscle_names.{os.getpid()}.{uuid.uuid4().hex}.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, names_path)

    fps = None  # filled by caller via config in run_batch — store in texts
    fps_val = texts.get("fps")
    if fps_val is not None:
        fps = float(fps_val)
    entry = {
        "T": int(activations.shape[0]),
        "fps": fps,
        "n_muscles": int(activations.shape[1]),
        "source": str(texts.get("source", "motion-x++")),
        "status": str(texts.get("status", "ok")),
    }
    lock = output_root / ".metadata.lock"
    with _dir_lock(lock):
        meta = load_metadata(output_root)
        meta[sample_id] = entry
        save_metadata(output_root, meta)


def load_checkpoint(path: Path) -> set[str]:
    """Load completed sample IDs from a JSON list on disk."""
    if not path.is_file():
        return set()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return set(str(x) for x in data)
    return set()


def save_checkpoint(path: Path, completed_ids: set[str]) -> None:
    """Persist completed IDs as a JSON list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sorted(completed_ids), f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

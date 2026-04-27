"""Configuration loading, logging, retries, and motion preprocessing helpers."""

from __future__ import annotations

import functools
import logging
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def resolve_against_config_dir(config_file: Path, path_value: str | Path) -> Path:
    """Resolve ``path_value`` relative to the directory containing the config file.

    Absolute paths are returned unchanged (resolved). Relative paths are joined
    with ``config_file.parent``.
    """
    p = Path(path_value)
    if p.is_absolute():
        return p.resolve()
    return (config_file.parent / p).resolve()


REQUIRED_TOP_LEVEL_KEYS = (
    "paths",
    "dataset",
    "conversion",
    "ik",
    "rra",
    "static_optimization",
    "output",
    "batch",
    "download",
    "visualization",
)

F = TypeVar("F", bound=Callable[..., Any])


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate ``config.yaml``.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        ValueError: If required keys are missing or types are invalid.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None or not isinstance(raw, dict):
        raise ValueError("Configuration must be a non-empty YAML mapping.")

    invalid_entries: list[str] = []
    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in raw]
    invalid_entries.extend([f"missing:{k}" for k in sorted(missing)])

    batch = raw.get("batch")
    if not isinstance(batch, dict):
        invalid_entries.append("type:batch")
    else:
        if "num_workers" in batch and not isinstance(batch["num_workers"], int):
            invalid_entries.append("type:batch.num_workers")
        if "max_retries" in batch and not isinstance(batch["max_retries"], int):
            invalid_entries.append("type:batch.max_retries")

    conv = raw.get("conversion")
    if not isinstance(conv, dict):
        invalid_entries.append("type:conversion")
    elif "filter_order" in conv and not isinstance(conv["filter_order"], int):
        invalid_entries.append("type:conversion.filter_order")

    out = raw.get("output")
    if not isinstance(out, dict):
        invalid_entries.append("type:output")

    if invalid_entries:
        raise ValueError("Invalid configuration entries: " + ", ".join(invalid_entries))
    return raw


def retry(max_retries: int, delay_s: float) -> Callable[[F], F]:
    """Return a decorator that retries a function on failure.

    Args:
        max_retries: Maximum number of attempts (>= 1).
        delay_s: Seconds to wait between attempts after a failure.

    Returns:
        Decorator that logs warnings and re-raises the last exception.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        "Attempt %s/%s failed in %s: %s",
                        attempt,
                        max_retries,
                        func.__name__,
                        exc,
                        exc_info=attempt == max_retries,
                    )
                    if attempt < max_retries:
                        time.sleep(delay_s)
            assert last_exc is not None
            raise last_exc

        wrapper.__wrapped__ = func  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


def get_multiprocessing_context() -> multiprocessing.context.BaseContext:
    """Return a multiprocessing context with a safe start method.

    Returns:
        ``fork`` on Linux only; ``spawn`` on macOS and Windows.
    """
    method = "fork" if sys.platform.startswith("linux") else "spawn"
    return multiprocessing.get_context(method)


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Configure root logging with stream and optional file handlers.

    Args:
        level: Log level name (e.g. ``INFO``, ``DEBUG``).
        log_file: Optional path to append logs to.
    """
    fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(fmt))
    root.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt))
        root.addHandler(fh)


def joint_velocity_clamp(
    angles: np.ndarray, max_vel_rad_s: float, fps: float
) -> np.ndarray:
    """Clip per-frame angular velocity and reconstruct angles from frame 0.

    Args:
        angles: Array of shape ``[T, D]`` with joint angles in radians.
        max_vel_rad_s: Maximum allowed angular velocity magnitude.
        fps: Sampling rate in Hz (used as ``1/dt`` for differencing).

    Returns:
        Clamped angles with the same shape and dtype as ``angles``.
    """
    if angles.ndim != 2:
        raise ValueError("angles must have shape [T, D]")
    if fps <= 0:
        raise ValueError("fps must be positive")
    dt = 1.0 / float(fps)
    out = angles.astype(np.float64, copy=True)
    vel = np.diff(out, axis=0) / dt
    clip = max_vel_rad_s
    vel = np.clip(vel, -clip, clip)
    recon = np.zeros_like(out)
    recon[0] = out[0]
    recon[1:] = out[0] + np.cumsum(vel * dt, axis=0)
    return recon.astype(angles.dtype, copy=False)

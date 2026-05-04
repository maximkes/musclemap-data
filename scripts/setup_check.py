"""
setup_check.py — Environment validator and OpenSim path auto-detector.

Run ONCE after cloning the repository and setting up your conda environment.

What it does:
  1. Searches all conda environments for one that has OpenSim installed.
  2. Verifies the OpenSim Python bindings are importable.
  3. Checks all required Python packages in the current env.
  4. Writes the detected opensim_python_exe and opensim_conda_env into config.yaml.
  5. Prints a clear status report with fix instructions for anything missing.

Usage:
    # Activate your opensim conda environment first, then:
    python scripts/setup_check.py

    # With a custom config path:
    python scripts/setup_check.py --config path/to/config.yaml

    # Detect and report only — do not write to config:
    python scripts/setup_check.py --dry-run
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("tqdm", "tqdm"),
    ("pyyaml", "yaml"),
    ("gdown", "gdown"),
    ("pyrender", "pyrender"),
    ("trimesh", "trimesh"),
    ("ipywidgets", "ipywidgets"),
    ("plotly", "plotly"),
    ("dash", "dash"),
    ("pandas", "pandas"),
    ("rerun-sdk", "rerun"),
    ("torch", "torch"),
    ("smplx", "smplx"),
]


# ─── OpenSim Detection ────────────────────────────────────────────────────────

def detect_opensim_python_path() -> Optional[Path]:
    """Search for a Python executable that has opensim installed.

    Tries three strategies in order:
    1. Current Python interpreter (already in the right env).
    2. All conda environments on the system.
    3. Common filesystem locations for site-packages.

    Returns:
        Path to the Python executable with opensim, or None if not found.
    """
    if _can_import_opensim(sys.executable):
        logger.info("OpenSim found in current Python environment.")
        return Path(sys.executable)

    logger.info("OpenSim not in current env. Scanning conda environments...")
    conda_result = _search_conda_envs()
    if conda_result:
        return conda_result

    logger.info("Searching common filesystem locations...")
    return _search_library_paths()


def _can_import_opensim(python_exe: str) -> bool:
    """Return True if the given Python executable can import opensim."""
    try:
        result = subprocess.run(
            [python_exe, "-c", "import opensim,sys;sys.stdout.write(str(opensim.__version__))"],
            capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _search_conda_envs() -> Optional[Path]:
    """Search all conda environments for opensim."""
    conda_exe = shutil.which("conda")
    if not conda_exe:
        logger.debug("conda not found on PATH.")
        return None

    try:
        result = subprocess.run(
            [conda_exe, "env", "list", "--json"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return None

        envs = json.loads(result.stdout).get("envs", [])
        system = platform.system()

        for env_path_str in envs:
            env_path = Path(env_path_str)
            python_exe = _python_in_env(env_path, system)
            if python_exe and _can_import_opensim(str(python_exe)):
                logger.info(f"Found OpenSim in conda env: {env_path}")
                return python_exe

    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
        logger.debug(f"Conda search error: {exc}")

    return None


def _python_in_env(env_path: Path, system: str) -> Optional[Path]:
    """Return the Python executable inside a conda environment directory."""
    candidate = (
        env_path / "python.exe" if system == "Windows"
        else env_path / "bin" / "python"
    )
    return candidate if candidate.exists() else None


def _search_library_paths() -> Optional[Path]:
    """Search common install directories for opensim site-packages."""
    system = platform.system()

    if system == "Darwin":
        bases = [
            Path.home() / "opt" / "anaconda3",
            Path.home() / "anaconda3",
            Path.home() / "miniconda3",
            Path("/opt/homebrew"),
            Path("/usr/local"),
        ]
    elif system == "Linux":
        bases = [
            Path.home() / "anaconda3",
            Path.home() / "miniconda3",
            Path("/opt/conda"),
            Path("/usr/local"),
        ]
    else:
        bases = [
            Path.home() / "anaconda3",
            Path.home() / "Anaconda3",
            Path("C:/ProgramData/Anaconda3"),
            Path("C:/tools/miniconda3"),
        ]

    for base in bases:
        if not base.exists():
            continue
        for init_file in base.rglob("opensim/__init__.py"):
            parts = init_file.parts
            try:
                sp_idx = next(i for i, p in enumerate(parts) if p == "site-packages")
                env_root = Path(*parts[: sp_idx - 2])
                python_exe = _python_in_env(env_root, system)
                if python_exe and python_exe.exists():
                    return python_exe
            except StopIteration:
                continue

    return None


def get_opensim_version(python_exe: Path) -> Optional[str]:
    """Return the OpenSim version string from the given Python executable."""
    try:
        result = subprocess.run(
            [str(python_exe), "-c", "import opensim,sys;sys.stdout.write(str(opensim.__version__))"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


# ─── Package Check ────────────────────────────────────────────────────────────

def check_required_packages() -> dict[str, bool]:
    """Check availability of all required Python packages in the current env.

    Returns:
        Dict mapping package display name to availability bool.
    """
    results = {}
    for display_name, import_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
            results[display_name] = True
        except ImportError:
            results[display_name] = False
    return results


# ─── Config Update ────────────────────────────────────────────────────────────

def update_config_with_opensim_path(config_path: Path, python_exe: Path) -> None:
    """Write opensim Python exe and conda env root into config.yaml.

    Args:
        config_path: Path to config.yaml.
        python_exe: Path to the Python executable that has opensim.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    config.setdefault("paths", {})
    config["paths"]["opensim_python_exe"] = str(python_exe)

    # Derive env root: .../envs/opensim-env/bin/python -> .../envs/opensim-env
    env_root = (
        python_exe.parent  # Windows: env/python.exe -> env/
        if platform.system() == "Windows"
        else python_exe.parent.parent  # Unix: env/bin/python -> env/
    )
    config["paths"]["opensim_conda_env"] = str(env_root)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info(f"Wrote OpenSim paths to {config_path}")


# ─── Report ───────────────────────────────────────────────────────────────────

def print_report(
    opensim_python: Optional[Path],
    opensim_version: Optional[str],
    pkg_status: dict[str, bool],
    config_path: Path,
    dry_run: bool,
) -> bool:
    """Print human-readable status report.

    Args:
        opensim_python: Detected Python executable with opensim, or None.
        opensim_version: OpenSim version string, or None.
        pkg_status: Package availability dict.
        config_path: Path to config.yaml.
        dry_run: If True, report only — do not write config.

    Returns:
        True if all checks passed.
    """
    sep = "═" * 62
    logger.info("%s", sep)
    logger.info("  musclemap-data — Environment Check")
    logger.info("  Python : %s  (%s)", sys.version.split()[0], sys.executable)
    logger.info("  Platform: %s %s", platform.system(), platform.machine())
    logger.info("%s", sep)

    all_ok = True

    # ── OpenSim ──────────────────────────────────────────────────────────────
    logger.info("[ OpenSim ]")
    if opensim_python:
        logger.info("  ✅  Executable : %s", opensim_python)
        logger.info("  ✅  Version    : %s", opensim_version or "unknown")
        if not dry_run:
            logger.info("  ✅  Paths written to config.yaml")
        else:
            logger.info("  ℹ️   dry-run: would write paths to config.yaml")
    else:
        all_ok = False
        logger.error("  ❌  OpenSim NOT FOUND")
        system = platform.system()
        machine = platform.machine()
        if system == "Darwin" and machine == "arm64":
            logger.info("  Install (macOS Apple Silicon — native arm64 since OpenSim 4.6):")
            logger.info("    conda create -n opensim-env python=3.11 -c conda-forge")
            logger.info("    conda activate opensim-env")
            logger.info("    conda install -c opensim-org opensim")
        elif system == "Darwin":
            logger.info("  Install (macOS Intel x86_64):")
            logger.info("    conda create -n opensim-env python=3.11")
            logger.info("    conda activate opensim-env")
            logger.info("    conda install -c opensim-org opensim")
        elif system == "Linux":
            logger.info("  Install (Linux x86_64):")
            logger.info("    conda create -n opensim-env python=3.11")
            logger.info("    conda activate opensim-env")
            logger.info("    conda install -c opensim-org opensim")
        else:
            logger.info("  Install (Windows x86_64):")
            logger.info("    conda create -n opensim-env python=3.11")
            logger.info("    conda activate opensim-env")
            logger.info("    conda install -c opensim-org opensim")
        logger.info("  Then re-run from inside that environment:")
        logger.info("    conda activate opensim-env")
        logger.info("    python scripts/setup_check.py")

    # ── Python Packages ──────────────────────────────────────────────────────
    logger.info("[ Python Packages (current env) ]")
    missing = []
    for pkg, ok in pkg_status.items():
        mark = "✅" if ok else "❌"
        logger.info("  %s  %s", mark, pkg)
        if not ok:
            missing.append(pkg)

    if missing:
        all_ok = False
        logger.info("  Install missing packages:")
        logger.info("    conda env update -f environment.yml --prune")
        logger.info("    conda activate musclemap-data")
        logger.info("  Or: pip install -e .  (after conda deps are satisfied)")

    # ── Config ───────────────────────────────────────────────────────────────
    logger.info("[ Config ]")
    if config_path.exists():
        logger.info("  ✅  %s", config_path)
    else:
        all_ok = False
        logger.error("  ❌  Not found: %s", config_path)
        logger.error("      Copy config.yaml.example to config.yaml and edit paths.")

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("%s", sep)
    if all_ok:
        logger.info("  ✅  All checks passed. Ready to run musclemap-data.")
    else:
        logger.error("  ❌  Some checks failed. Fix the issues above and re-run.")
    logger.info("%s", sep)

    return all_ok


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main() -> None:
    """Main entry point for the environment checker."""
    parser = argparse.ArgumentParser(
        description="Detect OpenSim and validate the musclemap-data environment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml (default: project root config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect and report only. Do not modify config.yaml.",
    )
    args = parser.parse_args()

    opensim_python = detect_opensim_python_path()
    opensim_version = get_opensim_version(opensim_python) if opensim_python else None
    pkg_status = check_required_packages()

    if opensim_python and not args.dry_run and args.config.exists():
        update_config_with_opensim_path(args.config, opensim_python)

    all_ok = print_report(
        opensim_python=opensim_python,
        opensim_version=opensim_version,
        pkg_status=pkg_status,
        config_path=args.config,
        dry_run=args.dry_run,
    )

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

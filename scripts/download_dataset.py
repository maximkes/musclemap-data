"""Download Motion-X++ subsets from Google Drive with resumable cache."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path

from tqdm import tqdm

try:
    from src.utils import load_config, resolve_against_config_dir, setup_logging
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.utils import load_config, resolve_against_config_dir, setup_logging

logger = logging.getLogger(__name__)

_MODALITY_SUBDIRS: dict[str, str] = {
    "motion": "motion/motion_generation/smplx322",
    "text_seq": "text/semantic_label",
    "text_frame": "text/wholebody_pose_description",
    "video": "video",
    "audio": "audio",
    "keypoints": "keypoints",
    "mesh_recovery": "mesh_recovery",
}


def _job_key(subset: str, modality: str) -> str:
    """Return a stable key for one download/extract job."""
    return f"{subset}::{modality}"


def _load_resume_state(path: Path) -> set[str]:
    """Load completed job keys from disk."""
    if not path.is_file():
        return set()
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(payload, list):
        return set()
    return {str(item) for item in payload}


def _save_resume_state(path: Path, completed: set[str]) -> None:
    """Persist completed job keys as an atomic JSON list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sorted(completed), f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _gdown_download(url: str, output: Path, _timeout_s: int) -> bool:
    try:
        import gdown
    except ImportError:
        logger.warning("gdown not installed; cannot auto-download.")
        return False
    try:
        try:
            gdown.download(url, str(output), quiet=False, fuzzy=True)
        except TypeError:
            # Older gdown versions do not support ``fuzzy=``.
            gdown.download(url, str(output), quiet=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("gdown failed: %s", exc)
        return False
    return output.is_file()


def _download_zip_from_drive_folder(
    folder_id: str,
    subset: str,
    modality: str,
    modality_path: str,
    zip_path: Path,
    staging_root: Path,
) -> bool:
    """Download a subset zip from a Drive folder by scanning folder contents.

    gdown.download() with folder URLs can fetch HTML pages instead of files.
    This helper lists the Drive folder with ``skip_download=True``, downloads
    each file behind a per-file tqdm bar, then picks the best ``{subset}.zip``
    path matching the modality layout.
    """
    try:
        import gdown
    except ImportError:
        logger.warning("gdown not installed; cannot auto-download.")
        return False

    subset_stage = staging_root / subset
    subset_stage.mkdir(parents=True, exist_ok=True)
    wanted_name = f"{subset}.zip"
    candidates = sorted(subset_stage.rglob(wanted_name))
    if not candidates:
        try:
            pending = gdown.download_folder(
                id=folder_id,
                output=str(subset_stage),
                quiet=True,
                skip_download=True,
            )
        except RecursionError:
            logger.warning("gdown folder listing hit recursion depth for folder id=%s", folder_id)
            return False
        except Exception as exc:  # noqa: BLE001
            logger.warning("gdown folder listing failed: %s", exc)
            return False

        if not pending:
            logger.warning("Drive folder lists zero downloadable files (id=%s).", folder_id)
            return False

        for item in tqdm(
            pending,
            desc=f"gdown files ({subset}/{modality})",
            unit="file",
            leave=False,
        ):
            local_path = Path(item.local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if local_path.suffix:
                download_output = str(local_path)
            else:
                download_output = str(local_path.parent) + os.sep
            try:
                gdown.download(
                    url=f"https://drive.google.com/uc?id={item.id}",
                    output=download_output,
                    quiet=True,
                    resume=True,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed downloading %s (%s): %s", item.path, item.id, exc)
                return False

        candidates = sorted(subset_stage.rglob(wanted_name))
    if not candidates:
        logger.warning("No %s found inside downloaded Drive folder snapshot.", wanted_name)
        return False

    modality_tokens = [p for p in Path(modality_path).parts if p]
    scored: list[tuple[int, Path]] = []
    for cand in candidates:
        hay = "/".join(cand.parts).lower()
        score = sum(1 for t in modality_tokens if t.lower() in hay)
        scored.append((score, cand))
    scored.sort(key=lambda item: item[0], reverse=True)
    best = scored[0][1]
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, zip_path)
    logger.info("Selected zip candidate %s -> %s", best, zip_path)
    return True


def _extract_zip(zip_path: Path, dest: Path) -> None:
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error(
            "Cannot create extract directory %s (%s). "
            "Set paths.motionx_root in config to a writable path.",
            dest,
            exc,
        )
        raise
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        for m in tqdm(
            members,
            desc=f"Extract {zip_path.name}",
            unit="file",
            leave=False,
        ):
            zf.extract(m, path=dest)


def _manual_fallback(
    subset: str, modality: str, folder_id: str, modality_path: str, zip_path: Path
) -> None:
    logger.error("══════════════════════════════════════════════════════")
    logger.error("Auto-download failed: %s / %s", subset, modality)
    logger.error("  1. Open: https://drive.google.com/drive/folders/%s", folder_id)
    logger.error("  2. Navigate to: %s/", modality_path)
    logger.error("  3. Download: %s.zip", subset)
    logger.error("  4. Place at: %s", zip_path)
    logger.error("══════════════════════════════════════════════════════")
    input("Press Enter when ready...")


def _print_summary_table(summary: list[tuple[str, str, str]]) -> None:
    """Log a simple summary table."""
    h1, h2, h3 = "Subset", "Modality", "Status"
    w1 = max(len(h1), *(len(s) for s, _, _ in summary)) if summary else len(h1)
    w2 = max(len(h2), *(len(m) for _, m, _ in summary)) if summary else len(h2)
    w3 = max(len(h3), *(len(st) for _, _, st in summary)) if summary else len(h3)
    line = f"{h1:<{w1}} | {h2:<{w2}} | {h3:<{w3}}"
    logger.info("%s", line)
    logger.info("%s", "-" * len(line))
    for subset, modality, status in summary:
        logger.info("%s", f"{subset:<{w1}} | {modality:<{w2}} | {status:<{w3}}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Motion-X++ dataset archives.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--subset", action="append", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument(
        "--reset-progress",
        action="store_true",
        help="Reset saved download progress state before running.",
    )
    args = parser.parse_args()

    log_path = Path("download.log")
    setup_logging("INFO", log_path)
    cfg_path = args.config.resolve()
    cfg = load_config(args.config)
    dl = cfg.get("download", {}) or {}
    paths = cfg.get("paths", {}) or {}
    motionx_root = resolve_against_config_dir(cfg_path, paths["motionx_root"])
    folder_id = str(dl.get("gdrive_folder_id", ""))
    subsets = args.subset if args.subset else list(dl.get("subsets", []))
    modalities = dl.get("modalities", {}) or {}
    cache_dir = resolve_against_config_dir(cfg_path, dl.get("zip_cache_dir", ".download_cache"))
    staging_dir = resolve_against_config_dir(cfg_path, ".download_stage")
    progress_path = resolve_against_config_dir(cfg_path, ".download_progress.json")
    keep_zips = bool(dl.get("keep_zips", False))
    max_retries = int(dl.get("max_download_retries", 3))
    timeout = int(dl.get("download_timeout_s", 300))

    if args.reset and cache_dir.exists():
        shutil.rmtree(cache_dir)
        logger.info("Removed cache dir %s", cache_dir)
    if args.reset_progress and progress_path.exists():
        progress_path.unlink()
        logger.info("Removed progress file %s", progress_path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    try:
        motionx_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error(
            "Cannot create paths.motionx_root=%s (%s). "
            "Use a writable directory (e.g. under your project: data/motion-x).",
            motionx_root,
            exc,
        )
        raise SystemExit(1) from exc

    summary: list[tuple[str, str, str]] = []
    completed_jobs = _load_resume_state(progress_path)

    jobs: list[tuple[str, str, Path]] = []
    for subset in subsets:
        for modality, enabled in modalities.items():
            if not enabled:
                continue
            if modality not in _MODALITY_SUBDIRS:
                logger.warning("Unknown modality key %s; skipping.", modality)
                continue
            jobs.append((subset, modality, Path(_MODALITY_SUBDIRS[modality])))

    with tqdm(total=len(jobs), desc="Download jobs", unit="job") as progress:
        for subset, modality, rel in jobs:
            target = motionx_root / rel / subset
            zip_path = cache_dir / f"{subset}__{modality}.zip"
            status = "skipped"
            key = _job_key(subset, modality)

            if target.exists() and any(target.iterdir()):
                logger.info("Already extracted: %s", target)
                status = "extracted"
                summary.append((subset, modality, status))
                completed_jobs.add(key)
                _save_resume_state(progress_path, completed_jobs)
                progress.update(1)
                continue

            if key in completed_jobs:
                logger.info("Already completed in progress state: %s / %s", subset, modality)
                status = "completed"
                summary.append((subset, modality, status))
                progress.update(1)
                continue

            if args.dry_run:
                logger.info("dry-run: would download %s %s -> %s", subset, modality, zip_path)
                status = "dry-run"
                summary.append((subset, modality, status))
                progress.update(1)
                continue

            try:
                import gdown  # noqa: F401
            except ImportError:
                logger.warning(
                    "gdown is not installed; switching to manual fallback for %s / %s.",
                    subset,
                    modality,
                )
                _manual_fallback(subset, modality, folder_id, str(rel), zip_path)
                ok = zip_path.is_file() and zipfile.is_zipfile(zip_path)
            else:
                ok = False
                for attempt in range(1, max_retries + 1):
                    logger.info(
                        "Download attempt %s/%s for %s / %s", attempt, max_retries, subset, modality
                    )
                    ok = _download_zip_from_drive_folder(
                        folder_id=folder_id,
                        subset=subset,
                        modality=modality,
                        modality_path=str(rel),
                        zip_path=zip_path,
                        staging_root=staging_dir,
                    )
                    if ok:
                        break
                if not ok:
                    _manual_fallback(subset, modality, folder_id, str(rel), zip_path)
                    ok = zip_path.is_file() and zipfile.is_zipfile(zip_path)

            if ok and zip_path.is_file() and zipfile.is_zipfile(zip_path):
                try:
                    _extract_zip(zip_path, target.parent)
                except OSError:
                    status = "failed"
                    logger.exception("Extract failed for %s %s", subset, modality)
                else:
                    status = "ok"
                    completed_jobs.add(key)
                    _save_resume_state(progress_path, completed_jobs)
                    if not keep_zips:
                        zip_path.unlink(missing_ok=True)
            else:
                status = "failed"
                logger.error(
                    "Download/extract failed for %s %s (missing or invalid zip at %s)",
                    subset,
                    modality,
                    zip_path,
                )

            summary.append((subset, modality, status))
            progress.set_postfix(last=f"{subset}/{modality}:{status}")
            progress.update(1)

    logger.info("Subset | Modality | Status")
    for subset, modality, status in summary:
        logger.info("%s | %s | %s", subset, modality, status)
    _print_summary_table(summary)


if __name__ == "__main__":
    main()

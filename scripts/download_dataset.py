"""Download Motion-X++ subsets from Google Drive with resumable cache."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import urllib.parse
import zipfile
from pathlib import Path

from tqdm import tqdm

try:
    from src.utils import load_config, resolve_against_config_dir, setup_logging
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.utils import load_config, resolve_against_config_dir, setup_logging

logger = logging.getLogger(__name__)

_GDRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"
# Same UA as gdown.download_folder (embedded folder view expects a browser-like client).
_GDOWN_FOLDER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36"
)


def _list_gdrive_embedded_folder(
    sess: object,
    folder_id: str,
    *,
    verify: bool | str,
    timeout_s: int,
) -> tuple[str, list[tuple[str, str, str]]]:
    """List one Drive folder via embeddedfolderview (same protocol as gdown).

    Returns (folder_title, [(id, name, type_or_mime), ...]).
    """
    try:
        import bs4
        from gdown.exceptions import DownloadError
    except ImportError as e:
        raise RuntimeError(
            "gdown (and its dependencies) are required for Drive folder listing."
        ) from e

    params = urllib.parse.urlencode({"id": folder_id})
    url = f"https://drive.google.com/embeddedfolderview?{params}"
    res = sess.get(url, verify=verify, timeout=timeout_s)
    if res.status_code != 200:
        raise DownloadError(
            f"Failed to retrieve folder contents for folder ID: {folder_id} "
            f"(status code {res.status_code}). "
            "You may need to change the permission to 'Anyone with the link'. "
            "See https://github.com/wkentaro/gdown#faq.",
        )

    soup = bs4.BeautifulSoup(res.text, features="html.parser")
    if soup.title is None or soup.title.string is None:
        raise DownloadError(
            f"Failed to parse folder contents for folder ID: {folder_id}. "
            "The page structure may have changed.",
        )
    folder_name = soup.title.string

    children: list[tuple[str, str, str]] = []
    for a_tag in soup.find_all(name="a"):
        href = a_tag.get("href", "")
        if not isinstance(href, str):
            continue

        file_match = re.match(
            r"https://drive\.google\.com/file/d/([-\w]{25,})/view",
            href,
        )
        if file_match:
            file_id = file_match.group(1)
            file_name = a_tag.get_text(strip=True)
            children.append((file_id, file_name, "application/octet-stream"))
            continue

        docs_match = re.match(
            r"https://docs\.google\.com/\w+/d/([-\w]{25,})/",
            href,
        )
        if docs_match:
            file_id = docs_match.group(1)
            file_name = a_tag.get_text(strip=True)
            children.append((file_id, file_name, "application/octet-stream"))
            continue

        folder_match = re.match(
            r"https://drive\.google\.com/drive/folders/([-\w]{25,})",
            href,
        )
        if folder_match:
            child_folder_id = folder_match.group(1)
            child_name = a_tag.get_text(strip=True)
            children.append((child_folder_id, child_name, _GDRIVE_FOLDER_MIME))
            continue

    return (folder_name, children)


def _resolve_gdrive_path_to_folder_id(
    root_folder_id: str,
    relative_posix_path: str,
    *,
    timeout_s: int,
    verify: bool | str = True,
) -> str | None:
    """Walk root → … → leaf by path segments only (no full-tree recursion)."""
    try:
        from gdown.download import _get_session
    except ImportError:
        logger.warning("gdown not installed; cannot resolve Drive subfolder.")
        return None

    parts = [p for p in Path(relative_posix_path.replace("\\", "/")).parts if p not in ("", ".")]
    if not parts:
        return root_folder_id

    sess, _ = _get_session(proxy=None, use_cookies=True, user_agent=_GDOWN_FOLDER_UA)
    current_id = root_folder_id
    for i, part in enumerate(parts):
        try:
            _title, children = _list_gdrive_embedded_folder(
                sess,
                current_id,
                verify=verify,
                timeout_s=timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Drive folder listing failed at step %s/%s (%r): %s",
                i + 1,
                len(parts),
                part,
                exc,
            )
            return None

        match_id: str | None = None
        for child_id, child_name, child_type in children:
            if child_type != _GDRIVE_FOLDER_MIME:
                continue
            if child_name == part or child_name.casefold() == part.casefold():
                match_id = child_id
                break

        if match_id is None:
            folder_names = [n for _, n, t in children if t == _GDRIVE_FOLDER_MIME]
            logger.warning(
                "Drive subfolder %r not found under id=%s (folders here: %s)",
                part,
                current_id,
                folder_names,
            )
            return None

        logger.info(
            "Drive path segment %s/%s → %r (folder id %s…)",
            i + 1,
            len(parts),
            part,
            match_id[:8],
        )
        current_id = match_id

    return current_id


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


def _filter_pending_by_subset(
    pending: list[object],
    subset: str,
) -> list[object]:
    """Keep Drive listing entries for this subset only (Motion-X++ uses e.g. ``idea400.zip``).

    Avoids fetching unrelated archives in the same folder (same quota surface).
    """
    sub_cf = subset.casefold()
    matches: list[object] = []
    for item in pending:
        rel = getattr(item, "path", "") or ""
        name = Path(rel).name
        if not name.lower().endswith(".zip"):
            continue
        stem = Path(name).stem
        if stem.casefold() == sub_cf:
            matches.append(item)
    return matches


def _find_subset_zip_paths(subset_root: Path, subset: str) -> list[Path]:
    """Find ``*.zip`` files for ``subset`` under ``subset_root`` (case-insensitive stem)."""
    sub_cf = subset.casefold()
    found: list[Path] = []
    for p in subset_root.rglob("*.zip"):
        if p.stem.casefold() == sub_cf:
            found.append(p)
    return sorted(found)


def _resolve_subset_leaf_folder_id(
    root_folder_id: str,
    modality_path: str,
    subset: str,
    *,
    timeout_s: int,
) -> tuple[str | None, str]:
    """Prefer ``…/modality_path/subset`` on Drive; else ``modality_path`` only.

    Returns (folder_id_or_none, description_for_logs).
    """
    mp = modality_path.replace("\\", "/").strip("/")
    combined = f"{mp}/{subset}" if mp else subset
    fid = _resolve_gdrive_path_to_folder_id(
        root_folder_id,
        combined,
        timeout_s=timeout_s,
    )
    if fid is not None:
        return fid, combined
    fid = _resolve_gdrive_path_to_folder_id(
        root_folder_id,
        modality_path,
        timeout_s=timeout_s,
    )
    if fid is not None:
        return fid, modality_path
    return None, ""


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
    timeout_s: int,
) -> bool:
    """Download a subset zip from a Drive folder by scanning folder contents.

    gdown.download() with folder URLs can fetch HTML pages instead of files.
    This helper lists the Drive folder with ``skip_download=True``, downloads
    each file behind a per-file tqdm bar, then picks the best ``{subset}.zip``
    path matching the modality layout.

    The shared Motion-X++ Drive root is huge; ``gdown.download_folder`` would
    recurse through every branch. We first resolve ``modality_path`` (e.g.
    ``motion/motion_generation/smplx322``) to a leaf folder id and list only
    that subtree.
    """
    try:
        import gdown
    except ImportError:
        logger.warning("gdown not installed; cannot auto-download.")
        return False

    logger.info(
        "Resolving Google Drive folder for subset %s under %r (timeout %ss per request)",
        subset,
        modality_path,
        timeout_s,
    )
    leaf_folder_id, resolved_as = _resolve_subset_leaf_folder_id(
        folder_id,
        modality_path,
        subset,
        timeout_s=timeout_s,
    )
    if leaf_folder_id is None:
        logger.warning(
            "Could not resolve Drive subfolder for modality path %r (root id=%s).",
            modality_path,
            folder_id,
        )
        return False
    logger.info("Drive folder scope: %r", resolved_as)

    subset_stage = staging_root / subset
    subset_stage.mkdir(parents=True, exist_ok=True)
    candidates = _find_subset_zip_paths(subset_stage, subset)
    if not candidates:
        try:
            pending = gdown.download_folder(
                id=leaf_folder_id,
                output=str(subset_stage),
                quiet=True,
                skip_download=True,
            )
        except RecursionError:
            logger.warning(
                "gdown folder listing hit recursion depth for folder id=%s",
                leaf_folder_id,
            )
            return False
        except Exception as exc:  # noqa: BLE001
            logger.warning("gdown folder listing failed: %s", exc)
            return False

        if not pending:
            logger.warning(
                "Drive folder lists zero downloadable files (id=%s).",
                leaf_folder_id,
            )
            return False

        to_fetch = _filter_pending_by_subset(list(pending), subset)
        if not to_fetch:
            zip_names = sorted(
                {Path(getattr(x, "path", "") or "").name for x in pending}
                - {""}
            )
            logger.warning(
                "No archive matching subset %r in Drive listing (have %s). "
                "Filenames on Drive use lowercase in many subsets (e.g. idea400.zip).",
                subset,
                zip_names[:20],
            )
            return False

        logger.info(
            "Downloading %s zip(s) for subset %s (skipping %s other entries in folder)",
            len(to_fetch),
            subset,
            len(pending) - len(to_fetch),
        )

        for item in tqdm(
            to_fetch,
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
                kwargs = {
                    "id": item.id,
                    "output": download_output,
                    "quiet": True,
                    "resume": True,
                    "use_cookies": True,
                }
                try:
                    kwargs["fuzzy"] = True
                    gdown.download(**kwargs)
                except TypeError:
                    kwargs.pop("fuzzy", None)
                    gdown.download(**kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed downloading %s (%s): %s", item.path, item.id, exc)
                logger.warning(
                    "If this repeats for every file, Google Drive may be blocking "
                    "anonymous downloads (quota). Put cookies in ~/.cache/gdown/cookies.txt "
                    "or download in a browser; see https://github.com/wkentaro/gdown#faq",
                )
                return False

        candidates = _find_subset_zip_paths(subset_stage, subset)
    if not candidates:
        logger.warning(
            "No zip archive matching subset %r (case-insensitive stem) under %s after fetch.",
            subset,
            subset_stage,
        )
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
                        timeout_s=timeout,
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

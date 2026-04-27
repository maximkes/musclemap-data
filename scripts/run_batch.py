"""Parallel batch driver: SMPL-X motions → OpenSim → output_dataset."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.dataset_io import (
    MotionXSample,
    load_checkpoint,
    load_sample,
    save_activation_sample,
    save_checkpoint,
    scan_dataset,
)
from src.opensim_pipeline import run_full_pipeline
from src.utils import (
    get_multiprocessing_context,
    load_config,
    resolve_against_config_dir,
    retry,
    setup_logging,
)

logger = logging.getLogger(__name__)


def _checkpoint_lock_path(ck_path: Path) -> Path:
    return ck_path.with_name(ck_path.name + ".lock")


def _acquire_lock(lock_dir: Path, timeout_s: float = 30.0, poll_s: float = 0.05) -> None:
    start = time.monotonic()
    while True:
        try:
            lock_dir.mkdir(parents=False, exist_ok=False)
            return
        except FileExistsError:
            if (time.monotonic() - start) > timeout_s:
                raise TimeoutError(f"Timed out waiting for lock: {lock_dir}")
            time.sleep(poll_s)


def _release_lock(lock_dir: Path) -> None:
    try:
        lock_dir.rmdir()
    except OSError:
        logger.warning("Could not remove checkpoint lock %s", lock_dir)


def _update_checkpoint_for_ok(ck_path: Path, sample_id: str) -> None:
    lock = _checkpoint_lock_path(ck_path)
    _acquire_lock(lock)
    try:
        done = load_checkpoint(ck_path)
        done.add(sample_id)
        save_checkpoint(ck_path, done)
    finally:
        _release_lock(lock)


def _process_one(args: tuple[Any, ...]) -> tuple[str, str, str | None]:
    """Process a single sample (module-level for multiprocessing pickling)."""
    sample_dict, config_dict, output_root_str, dry_run, ck_path_str = args
    sample = (
        sample_dict
        if isinstance(sample_dict, MotionXSample)
        else MotionXSample(
            id=str(sample_dict["id"]),
            motion_path=Path(sample_dict["motion_path"]),
            text_seq_path=Path(sample_dict["text_seq_path"]),
            text_frame_dir=(
                Path(sample_dict["text_frame_dir"])
                if sample_dict.get("text_frame_dir")
                else None
            ),
            source=str(sample_dict["source"]),
        )
    )
    output_root = Path(output_root_str)
    ck_path = Path(ck_path_str)
    config = config_dict
    paths = config.get("paths", {}) or {}
    tmp_dir = Path(paths["temp_dir"]) / sample.id
    failed_log = output_root / "failed_files.log"

    max_retries = int(config.get("batch", {}).get("max_retries", 3))
    delay_s = float(config.get("batch", {}).get("retry_delay_s", 2.0))

    @retry(max_retries=max_retries, delay_s=delay_s)
    def _run_pipeline() -> tuple[Any, list[str]]:
        return run_full_pipeline(sample.motion_path, config, tmp_dir, dry_run=dry_run)

    try:
        activations, muscle_names = _run_pipeline()
        data = load_sample(sample)
        motion = data["motion"]
        if activations.shape[0] != motion.shape[0]:
            raise RuntimeError(
                f"Activation length {activations.shape[0]} != motion length {motion.shape[0]}"
            )
        fps = float(config.get("output", {}).get("output_fps", config["conversion"]["output_fps"]))
        texts = {
            "semantic": data.get("semantic", ""),
            "pose_descriptions": data.get("pose_descriptions", []),
            "fps": fps,
            "source": sample.source,
            "status": "ok",
        }
        save_activation_sample(
            output_root,
            sample.id,
            activations,
            muscle_names,
            motion,
            texts,
        )
        _update_checkpoint_for_ok(ck_path, sample.id)
        return sample.id, "ok", None
    except Exception:
        import traceback as tb_mod

        err = tb_mod.format_exc()
        logger.exception("Failed processing %s", sample.id)
        try:
            with open(failed_log, "a", encoding="utf-8") as fh:
                fh.write(f"\n=== {sample.id} ===\n{err}\n")
        except OSError as exc:
            logger.error("Could not append failed_files.log: %s", exc)
        return sample.id, "failed", err


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch SMPL-X → muscle activations.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset-checkpoint", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg_path = args.config.resolve()
    cfg = load_config(cfg_path)
    paths = cfg.setdefault("paths", {})
    for key in ("motionx_root", "output_root", "temp_dir"):
        if paths.get(key):
            paths[key] = str(resolve_against_config_dir(cfg_path, paths[key]))
    output_root = Path(paths["output_root"])
    motionx_root = Path(paths["motionx_root"])
    ck_name = str(cfg.get("batch", {}).get("checkpoint_file", ".progress_checkpoint.json"))
    ck_path = (cfg_path.parent / ck_name).resolve()

    output_root.mkdir(parents=True, exist_ok=True)
    setup_logging("INFO", output_root / "batch.log")

    if args.reset_checkpoint and ck_path.is_file():
        ck_path.unlink()
        logger.info("Removed checkpoint %s", ck_path)

    done = load_checkpoint(ck_path)
    samples = [s for s in scan_dataset(motionx_root, cfg) if s.id not in done]
    if args.limit is not None:
        samples = samples[: max(0, args.limit)]
    logger.info("Resuming: %s completed, %s remaining.", len(done), len(samples))

    workers = args.workers if args.workers is not None else int(cfg.get("batch", {}).get("num_workers", 1))
    ctx = get_multiprocessing_context()

    work_args: list[tuple[Any, ...]] = []
    for s in samples:
        work_args.append(
            (
                {
                    "id": s.id,
                    "motion_path": str(s.motion_path),
                    "text_seq_path": str(s.text_seq_path),
                    "text_frame_dir": str(s.text_frame_dir) if s.text_frame_dir else None,
                    "source": s.source,
                },
                cfg,
                str(output_root),
                args.dry_run,
                str(ck_path),
            )
        )

    results: list[tuple[str, str, str | None]] = []
    # Use tqdm's preformatted rate token to avoid None formatting errors on startup.
    bar_fmt = "[{elapsed}<{remaining}] {n}/{total} | {rate_fmt}"

    interrupted = False
    if workers <= 1:
        try:
            for wa in tqdm(work_args, desc="batch", bar_format=bar_fmt, unit="seq"):
                res = _process_one(wa)
                results.append(res)
        except KeyboardInterrupt:
            interrupted = True
            logger.warning("Interrupted by user; saving checkpoint and exiting cleanly.")
    else:
        pool = ctx.Pool(processes=workers)
        try:
            for res in tqdm(
                pool.imap_unordered(_process_one, work_args),
                total=len(work_args),
                desc="batch",
                bar_format=bar_fmt,
                unit="seq",
            ):
                results.append(res)
        except KeyboardInterrupt:
            interrupted = True
            logger.warning("Interrupted by user; terminating workers and saving checkpoint.")
            pool.terminate()
        except Exception:
            # Ensure worker processes are not left running if progress rendering fails.
            pool.terminate()
            raise
        else:
            pool.close()
        finally:
            try:
                pool.join()
            except ValueError:
                pool.terminate()
                pool.join()

    completed = load_checkpoint(ck_path)
    summary = {
        "completed": sorted(completed),
        "last_run_utc": datetime.now(timezone.utc).isoformat(),
        "results": [{"id": r[0], "status": r[1]} for r in results],
    }
    with open(output_root / "batch_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if interrupted:
        return


if __name__ == "__main__":
    main()

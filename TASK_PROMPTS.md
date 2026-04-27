# Task Prompts — musclemap-data

Run these prompts **in order** in Cursor Agent mode, one at a time.
Wait for each task to complete and verify before running the next.

After each task, ask Cursor:
> "Review @src/<file>.py against @.cursor/rules/. List any violations."

---

## Task 1 — src/utils.py

Implement @src/utils.py fully. Replace all `raise NotImplementedError` with real implementations.

load_config(path):
  - Load YAML with yaml.safe_load
  - Validate required top-level keys: paths, dataset, conversion, ik, rra, static_optimization, output, batch, download, visualization
  - Validate types: batch.num_workers is int, batch.max_retries is int, conversion.filter_order is int
  - Raise ValueError listing ALL invalid keys at once
  - Return the dict

retry(max_retries, delay_s):
  - Decorator factory. On exception: log WARNING with attempt/total, exception, function name.
  - Wait delay_s seconds between attempts. After all retries: re-raise last exception.
  - Must be picklable (module-level, no mutable closure state). Add __wrapped__ attribute.

get_multiprocessing_context():
  - 'fork' on Linux only. 'spawn' on macOS and Windows.

setup_logging(level, log_file):
  - Root logger + StreamHandler. Optional FileHandler. Format: "%(asctime)s %(name)s %(levelname)s %(message)s"

joint_velocity_clamp(angles, max_vel_rad_s, fps):
  - Input [T, D]. Compute per-frame velocity via np.diff. Clip. Reconstruct via cumsum from frame 0.

Write tests in @tests/test_utils.py covering load_config, retry, joint_velocity_clamp.

---

## Task 2 — src/smplx_joint_regressor.py

Implement @src/smplx_joint_regressor.py fully.

JOINT_CORRESPONDENCE (module-level dict) — SMPL-X body pose joint index to Rajagopal coordinate names:
  1: ["hip_flexion_r", "hip_adduction_r", "hip_rotation_r"]
  2: ["hip_flexion_l", "hip_adduction_l", "hip_rotation_l"]
  4: ["knee_angle_r"]
  5: ["knee_angle_l"]
  7: ["ankle_angle_r"]
  8: ["ankle_angle_l"]
  16: ["arm_flex_r", "arm_add_r", "arm_rot_r"]
  17: ["arm_flex_l", "arm_add_l", "arm_rot_l"]
  18: ["elbow_flex_r"]
  19: ["elbow_flex_l"]
  20: ["pro_sup_r"]
  21: ["pro_sup_l"]

load_regressor(path): None path or missing file → return None (geometric fallback). Log INFO.

get_opensim_coords(body_pose, root_orient, trans, config, regressor):
  body_pose [T, 63], root_orient [T, 3], trans [T, 3]
  Geometric fallback: axis-angle → scipy Rotation → XYZ Euler → map to coordinate names.
  Root: ZYX Euler → pelvis_tilt, pelvis_list, pelvis_rotation. Trans → pelvis_tx/ty/tz.
  Apply ROM clamping from config['conversion']['joint_rom_limits'].
  Return dict[str, np.ndarray[T]].

---

## Task 3 — src/smplx_to_opensim.py

Implement @src/smplx_to_opensim.py fully.

SMPLX_SLICES (module-level constant):
  root_orient: slice(0, 3), pose_body: slice(3, 66), pose_hand: slice(66, 156),
  pose_jaw: slice(156, 159), face_expr: slice(159, 209), face_shape: slice(209, 309),
  trans: slice(309, 312), betas: slice(312, 322)

load_smplx_motion: load [T, 322] npy, split by SMPLX_SLICES, return dict.

butterworth_filter: scipy.signal.butter + filtfilt. If T < padlen: return unchanged + log WARNING.

write_mot_file: OpenSim Storage format. Header:
  Coordinates
  nRows=<T>
  nColumns=<N+1>
  inDegrees=no
  endheader
  time\t<col1>...
  Tab-separated, 8 decimal places. Time column: 0 to (T-1)/fps.

smplx_to_mot: load → get_opensim_coords → butterworth_filter → joint_velocity_clamp →
  ROM clamp → upsample (cubic/linear spline via scipy.interpolate) → write_mot_file.

---

## Task 4 — src/opensim_pipeline.py

Implement @src/opensim_pipeline.py fully.

All opensim imports inside functions with this guard:
  try:
      import opensim
  except ImportError as e:
      raise RuntimeError("OpenSim not found. Run python scripts/setup_check.py") from e

XML setup files: generate from src/templates/ using xml.etree.ElementTree. Never inline XML strings.

run_ik: generate IK XML → opensim.InverseKinematicsTool(xml).run() → return output .mot path.
run_rra: skip if config['rra']['enabled'] is False. RRATool(xml).run() → return .mot path.
run_static_optimization: AnalyzeTool(xml).run() → parse .sto → return (float32 ndarray [T,N], muscle_names).
get_muscle_names: opensim.Model(path).initSystem() → enumerate getMuscles() → return list[str].
run_full_pipeline: smplx_to_mot → run_ik → run_rra → run_static_optimization.
  dry_run=True: return np.random.rand(T, N_muscles).astype(np.float32), synthetic muscle_names.
  Success: delete tmp subdir. Failure: keep tmp subdir, log traceback, raise RuntimeError.
  Wrap every .run() call in try/except. Log opensim error reporter output at ERROR level.

---

## Task 5 — src/dataset_io.py

Implement @src/dataset_io.py fully.

scan_dataset: walk motion_subdir, match text files, build MotionXSample list.
  Log WARNING for .npy files missing text counterparts (do not skip them).

load_sample: np.load for motion. For text: try np.loadtxt first, fall back to open().read().
  (Motion-X++ uses .npy extension with text content — handle both.)

save_activation_sample:
  - Save activations as float32 .npy. Assert dtype before save.
  - Save smplx_322.npy as float32.
  - Save semantic_label.txt.
  - Save pose_descriptions/<frame_idx>.txt per frame.
  - Save muscle_names.json at output_root only if not present or changed.
  - Call save_metadata() to update metadata.json.

save_metadata: atomic write — write to .metadata.json.tmp then os.replace(). CRITICAL.

load/save_checkpoint: JSON list on disk, set[str] in memory for O(1) lookup.

Write tests in @tests/test_dataset_io.py: atomic write, checkpoint round-trip, scan_dataset mock.

---

## Task 6 — src/visualization.py

Implement @src/visualization.py fully.

plot_activation_topk(activations, muscle_names, k):
  - Top subplot: bar chart of variance for top-k muscles, sorted descending.
  - Bottom subplot: time-series lines for top-k muscles. Colormap: tab10.
  - Colorbar labeled "Muscle Activation [0-1]". Return Figure (no plt.show()).

animate_motion_interactive(smplx_motion, activations, muscle_names, config):
  - Try pyrender backend first (import inside function with try/except fallback).
  - Matplotlib stick figure fallback: implement fully.
  - Start from T-pose. Blend T-pose → frame 0 over config['visualization']['tpose_blend_frames'].
  - Color segments by mean activation of primary muscles. Colormap: coolwarm.
  - ipywidgets controls: Play button, Pause button, Restart button, IntSlider for frame.
  - Return FuncAnimation.

get_smplx_skeleton_joints: FK pass from SMPL-X frame [322] → [24, 3] joint positions.

---

## Task 7 — scripts/download_dataset.py

Implement @scripts/download_dataset.py fully.

CLI: --config, --subset (repeatable, overrides config list), --dry-run, --reset.

For each enabled (subset × modality):
  1. Check extracted → skip.
  2. Try gdown (import inside function, ImportError → manual fallback immediately).
  3. On failure after max_retries: print manual fallback block, wait input(), verify zip, extract.
  4. tqdm progress during zipfile extraction.

Manual fallback print block (exact format):
  print("══════════════════════════════════════════════════════")
  print(f"Auto-download failed: {subset} / {modality}")
  print(f"\n  1. Open: https://drive.google.com/drive/folders/{folder_id}")
  print(f"  2. Navigate to: {modality_path}/")
  print(f"  3. Download: {subset}.zip")
  print(f"  4. Place at: {zip_path}\n")
  input("Press Enter when ready...")

Final summary: print table (Subset | Modality | Status). Log all actions to download.log.

---

## Task 8 — scripts/run_batch.py

Implement @scripts/run_batch.py fully.

CLI: --config, --workers (overrides config), --dry-run, --reset-checkpoint, --limit N.

Flow:
  1. load_config → setup_logging.
  2. scan_dataset. load_checkpoint. Log resume status.
  3. multiprocessing Pool with context from get_multiprocessing_context().
  4. Map _process_one with tqdm. Format: "[{elapsed}<{remaining}] {n}/{total} | {rate:.1f} seq/s".
  5. Write output_root/batch_summary.json on completion.

_process_one(args) — MODULE-LEVEL function, not closure:
  Unpack (sample, config, output_root, dry_run).
  run_full_pipeline (wrapped with retry) → save_activation_sample → update checkpoint.
  Return (sample_id, "ok") or (sample_id, "failed", traceback_string).
  Log failed files with full traceback to output_root/failed_files.log.

---

## Task 9 — notebooks/01_explore_and_tune.ipynb

Fill all stub cells. See SCAFFOLD_PROMPT.md for cell structure.
Every cell must be independently re-runnable after setup cells.
Use DRY_RUN = False with comment "# TUNE THIS — set True to skip OpenSim".
Use SAMPLE_IDX = 0 with comment "# TUNE THIS".
Use K = 10 with comment "# TUNE THIS".

---

## Final Review Prompt

Run this after all tasks:

Review @src/ and @scripts/ against @.cursor/rules/. For each file check:
1. No print() — must use logging
2. No hardcoded paths
3. No top-level opensim imports
4. All public functions have type hints and Google docstrings
5. No float64 activations (must be float32)
6. No hardcoded N_muscles or fps
7. metadata.json only written via save_metadata() with atomic write
8. All multiprocessing worker functions are module-level

List every violation. Fix all of them.

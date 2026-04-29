# musclemap-data

`musclemap-data` is the data-generation stage of the MuscleMap thesis pipeline:

1. Load Motion-X++ SMPL-X motion (`[T, 322]`)
2. Convert to OpenSim coordinates and run static optimization
3. Save muscle activations and aligned text metadata for model training

## Project pipeline context

This repo is **Project 1 of 3**:

- `musclemap-data` (this project): motion -> muscle activation dataset
- `musclemap-model` (future): text -> muscle activations (fine-tuned MotionGPT)
- `musclemap-eval` (future): benchmarking and comparisons

Project 2 is expected to train from this project's `output_dataset/` artifact layout (described below).

## Requirements

- Python `>=3.10` (Poetry-managed dependencies)
- Conda installation for OpenSim runtime
- OpenSim Python bindings available in a conda environment
- Platform support:
  - macOS arm64 / x86_64
  - Linux x86_64
  - Windows x86_64

## Installation

### 1) Clone and enter the repository

```bash
cd /path/to/musclemap-data
```

### 2) Create/activate OpenSim conda environment

Install OpenSim via conda (`opensim-org` channel) in your preferred conda env, then activate it.

### 3) Install Poetry dependencies

```bash
poetry install
```

Optional mesh-preview dependencies (for SMPL-X shaded mesh notebook preview):

```bash
poetry install --with mesh
```

Optional Rerun dependencies (interactive timeline viewer):

```bash
poetry install --with rerun
```

### 4) Download required model files

Place OpenSim model files under `models/` (default points to `models/Rajagopal2016.osim`).

For optional notebook mesh preview, download official SMPL-X model files and set:

- `paths.smplx_model_folder` to the parent directory containing `smplx/`
  - expected layout example: `<folder>/smplx/SMPLX_NEUTRAL.npz`

### 5) Download Motion-X++ subsets

```bash
python scripts/download_dataset.py --config config.yaml
```

Useful options:

- `--subset IDEA400` (repeatable)
- `--modality motion --modality text_seq --modality text_frame` (repeatable)
- `--yes` (non-interactive where applicable)

### 6) Run environment check and auto-detect OpenSim interpreter

```bash
python scripts/setup_check.py --config config.yaml
```

This script validates imports and writes:

- `paths.opensim_python_exe`
- `paths.opensim_conda_env`

Use `--dry-run` to inspect without writing.

## Quick usage

### Notebook workflow (exploration/tuning)

Use `notebooks/01_explore_and_tune.ipynb` to:

- inspect motion/text samples
- preview original and reconstructed motion with multiple backends:
  - Rerun (`build_rerun_smplx_animation`, `animate_motion_interactive`)
  - Dash/Plotly (`show_dash_smplx_motion`, `show_dash_app`)
- run one-sequence OpenSim pipeline with progress bars
- inspect solver metrics and activation plots

Recommended comparison workflow in notebook:

```python
# 1) Original source motion (raw SMPL-X)
from src.visualization import show_dash_smplx_motion, show_dash_app

show_dash_smplx_motion(
    smplx_motion=motion,  # from load_sample(sample)["motion"]
    config=cfg,
    inline=True,
)

# 2) Reconstructed motion from OpenSim coordinates + activations
mot_path = (tmp_root / sample.id / "viz_coords.mot").resolve()
show_dash_app(
    mot_path=mot_path,
    activations=acts,
    muscle_names=muscle_names,
    config=cfg,
    smplx_motion=motion,  # optional overlay of raw SMPL-X joints
    inline=True,
)
```

### Batch processing

Run full (or limited) dataset processing:

```bash
python scripts/run_batch.py --config config.yaml
```

Common flags:

- `--workers N` override `batch.num_workers`
- `--limit N` process first N remaining samples
- `--dry-run` skip OpenSim and generate synthetic activations
- `--reset-checkpoint` clear progress checkpoint before run

Batch artifacts:

- `output_root/batch.log`
- `output_root/batch_summary.json`
- `output_root/failed_files.log` (on failures)

## Output dataset contract

Generated dataset layout:

```text
output_dataset/
  <sequence_id>/
    activations.npy          # float32, [T, N_muscles]
    smplx_322.npy            # float32, [T, 322] passthrough
    semantic_label.txt       # sequence-level text description
    pose_descriptions/
      <frame_idx>.txt        # per-frame text
  muscle_names.json          # list[str], len == N_muscles
  metadata.json              # dict[id -> {T, fps, source, n_muscles, status}]
```

Important invariants:

- `activations.npy` is always `float32`
- output FPS comes from config (`output.output_fps`)
- muscle count is inferred from names (`muscle_names.json`)

## Configuration reference (`config.yaml`)

Main sections:

- `paths`:
  - dataset root (`motionx_root`)
  - output root (`output_root`)
  - OpenSim model path (`opensim_model`)
  - OpenSim python executable (`opensim_python_exe`)
  - temporary dir (`temp_dir`)
  - optional regressor path (`smplx_to_opensim_regressor`)
  - optional SMPL-X mesh model folder (`smplx_model_folder`)
- `dataset`:
  - subdirectories for motion / sequence text / frame text
  - source fps
- `download`:
  - Google Drive folder id, subset/modality toggles, cache controls
- `conversion`:
  - target/output FPS, filtering, velocity clamp, ROM limits
- `ik`, `rra`, `static_optimization`:
  - OpenSim stage settings and solver controls
  - optional solver log/metric persistence
- `visualization`:
  - Rerun settings (`rerun_app_id`, `rerun_recording_id`, `rerun_spawn`)
  - Dash settings (`dash_port`, `dash_inline`, `dash_scene_bg`, etc.)
  - stick figure camera and 3D settings (for helper transforms/bounds)
  - mesh preview parameters
- `output`:
  - output fps and output file toggles
- `batch`:
  - retries, worker count, checkpoint filename

## Testing

Run full test suite:

```bash
poetry run pytest tests/ -q
```

## Troubleshooting

### `ModuleNotFoundError: opensim`

- Ensure OpenSim is installed in a conda env
- Run `python scripts/setup_check.py --config config.yaml`
- Confirm notebook/CLI is using the intended Python executable

### Download failures from Drive

- Re-run `scripts/download_dataset.py`
- Use cached zips under `.download_cache/`
- Follow manual fallback instructions printed by the script

### Batch run interrupted

- Re-run `scripts/run_batch.py` (checkpoint resumes automatically)
- Use `--reset-checkpoint` only if you want to restart from scratch

### Rerun opens separate window instead of notebook

- Set `cfg["visualization"]["rerun_spawn"] = False` to render in-cell via `rr.notebook_show()`.
- Ensure the active notebook kernel has `rerun-sdk` installed.

### Dash opens in browser instead of inline

- Pass `inline=True` to `show_dash_app(...)` / `show_dash_smplx_motion(...)`.
- For robust inline support, install `jupyter-dash`; otherwise the code falls back to a browser tab.

### Mesh preview errors

- Install optional deps: `poetry install --with mesh`
- Set `paths.smplx_model_folder` correctly (parent of `smplx/`)

# Scaffold Prompt — musclemap-data

> **Usage**: Open Cursor in Agent mode on the empty project directory.
> Paste this entire document. Do NOT implement logic — only create structure.

---

## Role

You are scaffolding a Python research project. Create all files with:
- Correct imports
- Type-hinted function signatures
- Google-style docstrings
- `raise NotImplementedError` as every function body

Do not implement any logic. Do not overwrite files that already exist
(config.yaml, scripts/setup_check.py, .cursor/rules/).

---

## Broader Context — Read but Do Not Implement

This is **Project 1 of 3** in a thesis pipeline:

- **Project 1** (this): Motion-X++ SMPL-X → muscle activations via OpenSim
- **Project 2** (`musclemap-model`): Fine-tune MotionGPT to generate muscle activations from text
- **Project 3** (`musclemap-eval`): Benchmark Project 2 vs Kinesis and original MotionGPT

Project 2 loads `output_dataset/` from this project as training data.
Output contract (never break):
```
output_dataset/
  <id>/
    activations.npy     # float32 [T, N_muscles]
    smplx_322.npy       # float32 [T, 322] passthrough
    semantic_label.txt
    pose_descriptions/<frame>.txt
  muscle_names.json     # list[str]
  metadata.json         # dict[id -> {T, fps, source, n_muscles, status}]
```
N_muscles and fps are NEVER hardcoded — always derived from files/config.

---

## Directory Tree

Create every file listed. Skip files that already exist.

```
musclemap-data/
├── pyproject.toml
├── requirements-opensim-env.txt
├── Makefile
├── README.md
├── config.yaml                          ← EXISTS, skip
├── .cursor/rules/                       ← EXISTS, skip
├── models/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── smplx_to_opensim.py
│   ├── smplx_joint_regressor.py
│   ├── opensim_pipeline.py
│   ├── dataset_io.py
│   ├── visualization.py
│   └── templates/
│       ├── ik_setup.xml
│       ├── rra_setup.xml
│       └── static_opt_setup.xml
├── notebooks/
│   └── 01_explore_and_tune.ipynb
├── scripts/
│   ├── setup_check.py                   ← EXISTS, skip
│   ├── download_dataset.py
│   └── run_batch.py
└── tests/
    ├── __init__.py
    ├── test_utils.py
    ├── test_smplx_to_opensim.py
    └── test_dataset_io.py
```

---

## pyproject.toml

```toml
[tool.poetry]
name = "musclemap-data"
version = "0.1.0"
description = "Motion-X++ SMPL-X motion to muscle activations via OpenSim"
authors = []
readme = "README.md"
packages = [{include = "src"}]

[tool.poetry.dependencies]
python = "^3.10"
numpy = ">=1.26"
scipy = ">=1.11"
matplotlib = ">=3.8"
tqdm = ">=4.66"
pyyaml = ">=6.0"
gdown = ">=5.1"
pyrender = ">=0.1.45"
trimesh = ">=4.0"
jupyter = ">=1.0"
ipywidgets = ">=8.0"
# Note: OpenSim is NOT here. Install: conda install -c opensim-org opensim
# Then run: python scripts/setup_check.py

[tool.poetry.group.dev.dependencies]
pytest = ">=8.0"
pytest-cov = ">=5.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

## requirements-opensim-env.txt

```
# Install inside the opensim conda env:
# pip install -r requirements-opensim-env.txt
numpy>=1.26
scipy>=1.11
pyyaml>=6.0
tqdm>=4.66
```

## Makefile

```makefile
.PHONY: setup check download notebook batch test clean

setup:
	poetry install

check:
	python scripts/setup_check.py

download:
	python scripts/download_dataset.py --config config.yaml

notebook:
	jupyter lab notebooks/01_explore_and_tune.ipynb

batch:
	python scripts/run_batch.py --config config.yaml

test:
	pytest tests/ -v --cov=src

clean:
	rm -rf .tmp_opensim .download_cache
```

## README.md — Section headers only

```markdown
# musclemap-data

## Overview
## Project Pipeline Context
## Requirements
## Installation
### 1. Install conda
### 2. Install OpenSim
### 3. Install Python dependencies
### 4. Download models
### 5. Download dataset
### 6. Run environment check
## Usage
### Notebook (parameter tuning)
### Batch processing
## Output Format
## Configuration Reference
## Troubleshooting
```

---

## src/utils.py — Stubs

```python
import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np

logger = logging.getLogger(__name__)

def load_config(path: Path) -> dict: ...
def retry(max_retries: int, delay_s: float) -> Callable: ...
def get_multiprocessing_context() -> str: ...
def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None: ...
def joint_velocity_clamp(
    angles: np.ndarray, max_vel_rad_s: float, fps: float
) -> np.ndarray: ...
```

## src/smplx_to_opensim.py — Stubs

```python
from pathlib import Path
from typing import Optional
import numpy as np

SMPLX_SLICES: dict = {}   # fill with actual slices

def load_smplx_motion(path: Path) -> dict[str, np.ndarray]: ...
def axis_angle_to_euler(aa: np.ndarray) -> np.ndarray: ...
def butterworth_filter(
    data: np.ndarray, order: int, cutoff_hz: float, fps: float
) -> np.ndarray: ...
def write_mot_file(
    coords: dict[str, np.ndarray], fps: float, output_path: Path
) -> None: ...
def smplx_to_mot(
    motion_npy: np.ndarray, config: dict, output_path: Path
) -> Path: ...
```

## src/smplx_joint_regressor.py — Stubs

```python
from pathlib import Path
from typing import Optional
import numpy as np

JOINT_CORRESPONDENCE: dict = {}

def load_regressor(path: Optional[Path]) -> Optional[object]: ...
def get_opensim_coords(
    body_pose: np.ndarray,
    root_orient: np.ndarray,
    trans: np.ndarray,
    config: dict,
    regressor: Optional[object] = None,
) -> dict[str, np.ndarray]: ...
```

## src/opensim_pipeline.py — Stubs

```python
from pathlib import Path
import numpy as np

def run_ik(
    model_path: Path, mot_path: Path, output_dir: Path, config: dict
) -> Path: ...
def run_rra(
    model_path: Path, ik_mot: Path, output_dir: Path, config: dict
) -> Path: ...
def run_static_optimization(
    model_path: Path, kinematics_mot: Path, output_dir: Path, config: dict
) -> tuple[np.ndarray, list[str]]: ...
def get_muscle_names(model_path: Path) -> list[str]: ...
def run_full_pipeline(
    smplx_npy: Path, config: dict, tmp_dir: Path, dry_run: bool = False
) -> tuple[np.ndarray, list[str]]: ...
```

## src/dataset_io.py — Stubs

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

@dataclass
class MotionXSample:
    id: str
    motion_path: Path
    text_seq_path: Path
    text_frame_dir: Optional[Path]
    source: str

def scan_dataset(root: Path, config: dict) -> list[MotionXSample]: ...
def load_sample(sample: MotionXSample) -> dict: ...
def save_activation_sample(
    output_root: Path,
    sample_id: str,
    activations: np.ndarray,
    muscle_names: list[str],
    smplx: np.ndarray,
    texts: dict,
) -> None: ...
def load_metadata(output_root: Path) -> dict: ...
def save_metadata(output_root: Path, meta: dict) -> None: ...
def load_checkpoint(path: Path) -> set[str]: ...
def save_checkpoint(path: Path, completed_ids: set[str]) -> None: ...
```

## src/visualization.py — Stubs

```python
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.figure
import matplotlib.animation

def plot_activation_topk(
    activations: np.ndarray, muscle_names: list[str], k: int
) -> matplotlib.figure.Figure: ...
def animate_motion_interactive(
    smplx_motion: np.ndarray,
    activations: np.ndarray,
    muscle_names: list[str],
    config: dict,
) -> matplotlib.animation.FuncAnimation: ...
def get_smplx_skeleton_joints(smplx_frame: np.ndarray) -> np.ndarray: ...
```

## XML Templates — src/templates/

ik_setup.xml:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<OpenSimDocument Version="40000">
  <InverseKinematicsTool name="ik">
    <!-- To be filled programmatically by opensim_pipeline.py -->
  </InverseKinematicsTool>
</OpenSimDocument>
```

rra_setup.xml and static_opt_setup.xml: same structure with RRATool and AnalyzeTool respectively.

## notebooks/01_explore_and_tune.ipynb

Create a notebook with these cells (markdown + empty code cells):
1. [markdown] # Setup
2. [code] (empty)
3. [markdown] # Select Sequence
4. [code] (empty)
5. [markdown] # Raw Motion — Joint Angles Before/After Filter
6. [code] (empty)
7. [markdown] # Run OpenSim Pipeline
8. [code] (empty)
9. [markdown] # Muscle Activations
10. [code] (empty)
11. [markdown] # Top-K Activation Variance — `K = 10  # TUNE THIS`
12. [code] (empty)
13. [markdown] # Interactive Animation
14. [code] (empty)
15. [markdown] # Parameter Tuning Notes\n\n**IK accuracy:**\n\n**RRA residuals:**\n\n**Filter cutoff:**\n\n**ROM clamp:**

## test stubs

test_utils.py: stub test functions for load_config, retry, joint_velocity_clamp
test_smplx_to_opensim.py: stub for SMPLX_SLICES and smplx_to_mot
test_dataset_io.py: stub for atomic write, checkpoint, scan_dataset

---

## After Creating Files

Print a checklist: filename | ✅ created / ⏭ skipped (existed) / ❌ failed

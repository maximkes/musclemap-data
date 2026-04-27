# Cursor Agent Prompt — musclemap-data (Project 1)

> Paste this in Cursor Agent mode **before anything else**.
> This file defines the full context. Never re-paste it — Cursor holds it.

---

## What You Are Building

You are implementing `musclemap-data`, Project 1 of a 3-project thesis pipeline:

| # | Project | Purpose |
|---|---------|---------|
| 1 | **musclemap-data** (HERE) | Motion-X++ SMPL-X → muscle activations via OpenSim |
| 2 | musclemap-model | Fine-tune MotionGPT: text → muscle activations |
| 3 | musclemap-eval | Benchmark model vs Kinesis and MotionGPT |

Project 2 depends on the exact file layout this project writes.

---

## Unbreakable Output Contract

Every successfully processed sequence must produce exactly:
```
output_root/<sequence_id>/
  activations.npy              # np.float32, shape [T, N_muscles]
  smplx_322.npy                # np.float32, shape [T, 322]  — unchanged passthrough
  semantic_label.txt           # one-line text description
  pose_descriptions/<n>.txt    # one file per frame
output_root/
  muscle_names.json            # list[str], len == N_muscles (columns of activations.npy)
  metadata.json                # dict[sequence_id → {T, fps, n_muscles, source, status}]
```
**N_muscles and fps are NEVER hardcoded.** Derive from muscle_names.json and config.

---

## Project Rules (abridged — full rules in .cursor/rules/)

1. **No print() in src/ or scripts/** — use logging module only.
2. **No top-level `import opensim`** — always inside functions, always with import guard.
3. **No hardcoded paths** — all from config.yaml.
4. **No threading** — multiprocessing only (OpenSim is not thread-safe).
5. **Worker functions must be module-level** — picklable for multiprocessing.
6. **metadata.json only via `dataset_io.save_metadata()`** — atomic write (tmp → rename).
7. **activations always float32** — assert dtype before saving.
8. **No plt.show() in src/** — return Figure objects.

---

## OpenSim — Critical Notes

- Installed via **conda** (`conda install -c opensim-org opensim`), not pip.
- Python executable path lives in `config.paths.opensim_python_exe` (auto-filled by setup_check.py).
- Use Python API only — **no subprocess calls to the OpenSim CLI**.
- Use verified API names only (see `.cursor/rules/04-opensim-api.mdc`).
- Support `dry_run=True` in all pipeline functions (returns random float32 arrays).

---

## Platform Support

| Platform | Notes |
|---|---|
| macOS arm64 (Apple Silicon) | OpenSim 4.6 has native arm64 — no Rosetta needed |
| macOS x86_64 | fully supported |
| Linux x86_64 | fully supported |
| Windows x86_64 | fully supported |

Use `utils.get_multiprocessing_context()` for start method ('fork' on Linux, 'spawn' elsewhere).

---

## Pipeline Stages

```
Motion-X++ .npy [T, 322]
    │
    ▼  smplx_to_opensim.py
 .mot file (OpenSim coordinate storage)
    │
    ▼  opensim_pipeline.py
    ├─ Inverse Kinematics (IK)
    ├─ Residual Reduction Algorithm (RRA)  ← optional, config.rra.enabled
    └─ Static Optimization (SO)
    │
    ▼  dataset_io.py
 output_dataset/  (Project 2 training data)
```

---

## Running Order

**Step 1 — Verify environment:**
```bash
python scripts/setup_check.py
```
This auto-detects OpenSim, writes paths to config.yaml, and prints a pass/fail checklist.
**Always start here. Especially on a new machine.**

**Step 2 — Download dataset:**
```bash
python scripts/download_dataset.py --config config.yaml
```

**Step 3 — Tune parameters (notebook):**
```bash
jupyter lab notebooks/01_explore_and_tune.ipynb
```
Adjust config.yaml until the notebook produces clean, anatomically plausible activations.

**Step 4 — Batch processing:**
```bash
python scripts/run_batch.py --config config.yaml
```
Resumable. On restart it reads the checkpoint and continues from the last unprocessed file.

---

## File Map

```
musclemap-data/
├── .cursor/rules/              ← Cursor rule files (auto-loaded, do not delete)
│   ├── 00-project.mdc
│   ├── 01-platform.mdc
│   ├── 02-code-conventions.mdc
│   ├── 03-notebook.mdc
│   ├── 04-opensim-api.mdc
│   └── 05-no-go.mdc
├── config.yaml                 ← Edit paths; run setup_check.py for opensim entries
├── pyproject.toml
├── Makefile
├── models/                     ← Place Rajagopal2016.osim + geometry/ here
├── src/
│   ├── utils.py                ← config loader, retry, logging, velocity clamp
│   ├── smplx_to_opensim.py     ← SMPL-X 322 → .mot file
│   ├── smplx_joint_regressor.py← SMPL-X body pose → OpenSim coordinate names
│   ├── opensim_pipeline.py     ← IK → RRA → StaticOpt wrappers
│   ├── dataset_io.py           ← scan, load, save, checkpoint, metadata
│   ├── visualization.py        ← top-k plot + interactive animation
│   └── templates/              ← XML setup file templates
├── notebooks/
│   └── 01_explore_and_tune.ipynb
├── scripts/
│   ├── setup_check.py          ← auto-detects OpenSim, fills config
│   ├── download_dataset.py     ← gdown + manual fallback
│   └── run_batch.py            ← parallel batch with tqdm + checkpoint
└── tests/
```

---

## How to Work with This Codebase in Cursor

1. **Always read the relevant `.cursor/rules/` before touching a file.**
2. **After implementing any file, run the review prompt from TASK_PROMPTS.md.**
3. **Run `make test` after each task.**
4. **Do not skip the dry_run path** — it must work without OpenSim installed.

---

## First Action

Run the Scaffold Prompt from `SCAFFOLD_PROMPT.md` to create all empty stubs.
Then work through tasks 1–9 in `TASK_PROMPTS.md` in order.

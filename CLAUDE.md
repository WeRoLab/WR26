# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Teaching materials for a graduate **Wearable Robotics** course: musculoskeletal
modeling and simulation with [OpenSim](https://opensim.stanford.edu/), inverse
kinematics from motion capture, and the data-processing that feeds a
wearable-robot design. See [`README.md`](README.md) for the full layout table.

## Environment setup

Everything runs in one Python virtual environment (the class calls it `WR26`):

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install -r Examples/requirements.txt
pip install -r Assignments/A2/requirements.txt
```

`opensim>=4.6` (includes Moco) installs from PyPI on Windows/macOS/Linux for
Python 3.11–3.13 — no conda needed. On a platform with no wheel, use
`conda install opensim-org::opensim` in the same env instead.

## Fetching data

Large binaries are **not** committed. Each module fetches its own data on demand:

- `python Assignments/A2/get_data.py` — ~5 MB motion capture + model, from Google Drive → gitignored `data/`
- `python Examples/examples/get_example_data.py` — Moco example data, from the pinned OpenSim source

`Examples/` tutorials need no fetch step; their inputs are bundled under `resources/`.

There is no lint or test tooling configured in this repo (no pytest, flake8,
pyproject.toml). Verification means running a module's pipeline end-to-end and
checking the resulting comparison plot/output values, not a test suite.

## Architecture

### `Examples/` — reference material, treated as ground truth

Eight guided Jupyter tutorials (`Tutorial 1–8 ....ipynb`, in
[`Examples/`](Examples/)) covering modeling → simulation → scaling/IK/ID →
static optimization → Moco optimal control, plus `examples/` — ~35 standalone
API example scripts ported from OpenSim's own examples, tagged by category in
[`examples/README.md`](Examples/examples/README.md).

- Each tutorial notebook starts with `from tutorial_setup import prepare;
  prepare("Tutorial N")`, which copies `resources/Tutorial N/` into a gitignored
  `_work/Tutorial N/`, `chdir`s there, and fixes a Windows PATH issue so Moco's
  Ipopt plugin loads.
- Run any script under `examples/` through the wrapper, not directly:
  `python run.py <script.py>` (from `Examples/examples/`) — it applies the same
  chdir/PATH fix.
- Because this module is known-correct, working OpenSim API usage, treat it as
  the reference pattern when writing or checking OpenSim code elsewhere in the
  repo (e.g. in `Assignments/`).

### `Assignments/A2/` — graded assignment: knee angle during a walking stride

Computes the knee angle two ways — anatomical reference frames vs. OpenSim's
`ScaleTool` + `InverseKinematicsTool` — on the Alizadeh et al. (2025) whole-body
dataset. Given infrastructure vs. student-edited files:

- `c3d_io.py`, `frames_viz.py` — given: C3D read/lab→OpenSim rotation/TRC write,
  and matplotlib 3D frame plotting (`--show-frames`).
- `get_data.py` — fetches the trial data into gitignored `data/`.
- `opensim_ik_reference.py` — student edits `build_task_set()` marker weights
  and the IK time window; runs the constrained-IK reference and writes
  `_work/stride_1_ik.mot`.
- `calculate_knee_angle.py` — student fills 4 TODOs (femur/tibia anatomical
  frames → rotation matrices → relative rotation → `atan2` knee angle); reads
  `_work/stride_1_ik.mot` and writes `knee_angle_stride.png` overlaying both
  methods.

Run order: `get_data.py` → `opensim_ik_reference.py` → `calculate_knee_angle.py`.

### `Assignments/A3/` — graded assignment: knee torque from first principles

Companion assignment to `Data/OpenSim_Tutorial3/` (below): compute the right
knee torque during a student-chosen single-limb-support window (left leg in
swing), from first principles, and compare it against OpenSim's own inverse
dynamics answer and against Winter's textbook data.

- `calculate_knee_torque.py` — given: loads everything needed as NumPy
  arrays — joint angles and scaled-model mass/COM/inertia from
  `Data/OpenSim_Tutorial3/Output/`, plus ground reaction forces/CoP and the
  `R.Heel`/`R.Toe.Tip` markers from `Data/OpenSim_Tutorial3/Data/` (all
  linearly resampled onto the IK time base). Also computes and hands over
  the two-segment (foot + shank) model's own dimensions/mass properties
  (`segments["foot"|"shank"]["l"|"r"|"m"|"I"]`, plus `PELVIS_TO_HIP`/
  `L_THIGH`) and joint-angle value/velocity/acceleration
  (`ik_kinematics`, via `osim.GCVSplineSet`) — so students don't have to
  extract dimensions from the OpenSim API or hand-differentiate joint
  angles themselves, and can focus on the equations of motion. Stops
  there — it does not perform the calculation; that's the student's code,
  written below the loading section.

Run order: `Data/OpenSim_Tutorial3/Python/scale_ik_id.py` →
`Assignments/A3/calculate_knee_torque.py`.

### `Data/OpenSim_Tutorial3/`

A standalone reproduction of OpenSim's official Tutorial 3 (Scale/IK/ID),
restructured into `Setup/` (XML configs, already filled in), `Data/`
(subject marker/GRF/model files), `Geometry/` (a handful of gait2354's
`.vtp` body-shape meshes, bundled so the API doesn't warn about missing
display geometry), and `Output/` (gitignored, regenerated by the script
below). Distinct from `Examples/resources/Tutorial 3`.

- `Python/scale_ik_id.py` — runs Scaling -> Inverse Kinematics -> Inverse
  Dynamics via the OpenSim Python API, each step built straight from its
  `Setup/*.xml` file, followed by an "Analysis" section that reads back the
  results and writes two verification plots into `Output/`. Feeds
  `Assignments/A3/calculate_knee_torque.py` above.
- `Data/` and `Geometry/` are tracked in git but marked `skip-worktree`
  (`git update-index --skip-worktree <path>`): they're inputs/reference
  outputs that a local pipeline run overwrites in place, and git is told to
  ignore those local changes rather than flag them as modifications.

### `.gitignore` layering

The root `.gitignore` covers repo-wide junk (bytecode, venvs, `.ipynb_checkpoints`,
`*.log`, generic run-output dirs like `_work/` and `results/`); each module's own
`.gitignore` additionally covers its fetched data and output paths.

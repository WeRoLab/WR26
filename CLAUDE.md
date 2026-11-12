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
pip install -r Lectures/CVX_Examples/requirements.txt
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

### `Assignments/A5/` — graded assignment: DC motor selection for an elbow prosthesis

- `trajectory_generator.py` — given: Python port of the former MATLAB
  `TrajectoryGenerator.m`. `gen_trajectory(x, t, n)` solves the 7x7 system for
  a 6th-order polynomial meeting `[x0, xf, xm, xd0, xdf, xdd0, xddf]` (mid
  point at `(t0+tf)/2`) and returns `time` and a `(3, n)` position/velocity/
  acceleration array. Run directly for a demo plot + constraint check. Students
  write the inverse dynamics, motor feasible region, voltages, and energy.
  Only needs numpy + matplotlib.

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

### `Lectures/CVX_Examples/` — convex-optimization demos, MATLAB + Python

Standalone lecture scripts: `Fourier_Transform_Truncated/` (least-squares
Fourier fit of the Winter 1983 ankle angle; the outlier variant compares
L2/L1), `Signal_Smoothing/` (minimum-jerk smoothing, two solvers compared),
and `MPC/` (DC-motor LQR in `StateFeedback_LQR`; `cvxgen/` holds the MPC QP).

- Every `.m` has a Python twin beside it (`main_fourier.py`, `Smoothing.py`,
  `StateFeedback_LQR.py`, `cvxgen/example_use.py`) with matching variable
  names, comments and plots. Python uses cvxpy for every optimization,
  `control.lqr` for LQR, and Clarabel for the MPC (OSQP returns
  "inaccurate" on its random unstable systems). Keep the two versions in
  sync when changing either; they agree with MATLAB to ~1e-14.
- MATLAB needs the Control System Toolbox and CVX (not installed on every
  machine; the scripts that need it fail or warn without it). Run each `.m`
  from its own folder. Everything in `MPC/cvxgen/` except `ExampleUse.m` and
  `example_use.py` is CVXGEN-generated: don't edit it; regenerate from
  `description.cvxgen` on cvxgen.com instead.
- Verify with `MPLBACKEND=Agg python <script>.py` (runs from any folder) and
  `matlab -batch` from each `.m` script's folder.
- `Inverse_Dynamics_Kuo98/` is the exception: Python + OpenSim only, no `.m`.
  It is an in-class activity reproducing Kuo (1998), least-squares inverse
  dynamics, against `Kuo1998.tex`, a condensed copy with corrected equations.
  - `four_segment_model.py`: the Appendix A equations.
  - `four_segment_opensim.py`: the same model in OpenSim, which generates
    the truth data. Run directly, it must print PASS on all of its
    equation-vs-OpenSim checks.
  - `least_squares_id.py`: student file, shipped with its cvxpy TODOs
    unfilled. The filled-in solution lives outside the repo in
    `WR26/Lectures/L18 Kuo98 activity solution/`; copy it into this folder
    to verify. With `SEED = 0` it expects NE 7.91 vs. LS 4.23 N·m overall.

### `Lectures/L19 IMU Drift/` — IMU orientation drift, MATLAB + Python

Plots azimuth/pitch/roll of a tablet left still on a table for ~16.5 min
(MATLAB Mobile, ~10 Hz); the angles should be constant, so any trend is drift.

- `DataCollectionDrift.mat` is the raw MATLAB Mobile recording: a timetable,
  which `scipy.io.loadmat` can't read. `convert_drift_data.m` re-saves it as
  plain arrays (`time` [s], `X`/`Y`/`Z` [deg], `start_time` text) in
  `DataCollectionDrift_arrays.mat` (`-v7`). Rerun it only for new recordings.
- Both `.mat` files are committed, static inputs (`.gitattributes`:
  `*.mat binary`). MATLAB stamps the save date into every MAT-file, so
  regenerating makes a "modified" binary even with identical data; the
  convert script therefore refuses to overwrite an existing arrays file.
  If a `.mat` shows up modified by accident, `git restore` it.
- `main_IMU_Drift.m` and its twin `main_IMU_Drift.py` both read the arrays
  file and draw the same 3-subplot figure. Keep them in sync.
- Verify with `MPLBACKEND=Agg python main_IMU_Drift.py` (from any folder) and
  `matlab -batch main_IMU_Drift` from the folder.

### `.gitignore` layering

The root `.gitignore` covers repo-wide junk (bytecode, venvs, `.ipynb_checkpoints`,
`*.log`, generic run-output dirs like `_work/` and `results/`); each module's own
`.gitignore` additionally covers its fetched data and output paths.

# WR26 -- Wearable Robotics course materials

Teaching materials for a graduate **Wearable Robotics** course: musculoskeletal
modeling and simulation with [OpenSim](https://opensim.stanford.edu/), inverse
kinematics / inverse dynamics from motion capture, and the data-processing that
feeds a wearable-robot design. Meant to be handed to students.

## Layout

| folder | what it is | start at |
|--------|------------|----------|
| [`Examples/`](Examples/) | **Reference material.** Eight guided OpenSim 4.6 Jupyter tutorials (modeling &rarr; simulation &rarr; scaling/IK/ID &rarr; static optimization &rarr; Moco optimal control) plus ~35 standalone API example scripts. Ported from the OpenSim project's conda/Colab notebooks to run locally against a pip-installed `opensim`. Also used as "ground truth" for LLM-assisted work. | [`README.md`](Examples/README.md) |
| [`Lectures/`](Lectures/) | **In-class activities.** `IK/` -- compute sit-to-stand joint angles from marker data two ways (the OpenSim `InverseKinematicsTool` and a hand-built 4-DOF planar model solved by nonlinear least squares), then compare and connect to a wearable-robot design. `solution/` = complete instructor pipeline; `activity/` = student version with blanks. | [`IK/README.md`](Lectures/IK/README.md), then [`IK/activity/handout.md`](Lectures/IK/activity/handout.md) |
| [`Assignments/`](Assignments/) | **Graded assignments.** `A2/` -- knee angle during a walking stride, computed two ways (anatomical reference frames vs. OpenSim `ScaleTool` + `InverseKinematicsTool`), on the Alizadeh et al. (2025) whole-body dataset. Student Python with `TODO` blanks; data fetched by `get_data.py`. | [`A2/README.md`](Assignments/A2/README.md) |

## Environment

Everything Python runs in one virtual environment (the class calls it `WR26`):

```bash
python -m venv .venv           # or wherever you keep venvs
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install -r Examples/requirements.txt
pip install -r Lectures/IK/requirements.txt
pip install -r Assignments/A2/requirements.txt
```

- **OpenSim** installs from PyPI (`pip install opensim`, 4.6+, includes Moco;
  wheels for Windows / macOS / Linux, Python 3.11-3.13). No conda needed. On a
  platform with no wheel, `conda install opensim-org::opensim` into the same env.
- Each module pins its own `requirements.txt`; the union is
  `opensim numpy scipy matplotlib pandas scikit-learn ipywidgets jupyterlab gdown`.

## Data

Large binaries are **not** committed. Each module fetches its own:

- `Lectures/IK/` -> `python Lectures/IK/get_data.py` (~190 MB motion capture + models, from Google Drive)
- `Assignments/A2/` -> `python Assignments/A2/get_data.py` (~5 MB motion capture + model, from Google Drive)
- `Examples/examples/` -> `python examples/get_example_data.py` (Moco example data, from the pinned OpenSim source)
- `Examples/` tutorials -> nothing to fetch; inputs are bundled in `resources/`

The root `.gitignore` covers repo-wide junk (bytecode, venvs, `.ipynb_checkpoints`,
`*.log`, run outputs); each module's own `.gitignore` adds its data and output
paths.

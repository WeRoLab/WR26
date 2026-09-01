# WR26 -- Wearable Robotics course materials

Teaching materials for a graduate **Wearable Robotics** course: musculoskeletal
modeling and simulation with [OpenSim](https://opensim.stanford.edu/), inverse
kinematics / inverse dynamics from motion capture, and the data-processing that
feeds a wearable-robot design. Meant to be handed to students.

## Layout

| folder | what it is | start at |
|--------|------------|----------|
| [`OpenSim_Python_Examples_Tutorials/`](OpenSim_Python_Examples_Tutorials/) | **Reference material.** Eight guided OpenSim 4.6 Jupyter tutorials (modeling &rarr; simulation &rarr; scaling/IK/ID &rarr; static optimization &rarr; Moco optimal control) plus ~35 standalone API example scripts. Ported from the OpenSim project's conda/Colab notebooks to run locally against a pip-installed `opensim`. Also used as "ground truth" for LLM-assisted work. | [`README.md`](OpenSim_Python_Examples_Tutorials/README.md) |
| [`IK/`](IK/) | **In-class activity.** Compute sit-to-stand joint angles from marker data two ways -- the OpenSim `InverseKinematicsTool` and a hand-built 4-DOF planar model solved by nonlinear least squares -- then compare and connect to a wearable-robot design. `solution/` = complete instructor pipeline; `activity/` = student version with blanks. | [`README.md`](IK/README.md), then [`activity/handout.md`](IK/activity/handout.md) |

`Assignments/` holds course assignments; it is a scaffold for now.

## Environment

Everything Python runs in one virtual environment (the class calls it `WR26`):

```bash
python -m venv .venv           # or wherever you keep venvs
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install -r OpenSim_Python_Examples_Tutorials/requirements.txt
pip install -r IK/requirements.txt
```

- **OpenSim** installs from PyPI (`pip install opensim`, 4.6+, includes Moco;
  wheels for Windows / macOS / Linux, Python 3.11-3.13). No conda needed. On a
  platform with no wheel, `conda install opensim-org::opensim` into the same env.
- `IK/` and the OpenSim tutorials each pin their own `requirements.txt`; the union
  is `opensim numpy scipy matplotlib pandas scikit-learn ipywidgets jupyterlab gdown`.

## Data

Large binaries are **not** committed. Each module fetches its own:

- `IK/` -> `python IK/get_data.py` (~190 MB motion capture + models, from Google Drive)
- `OpenSim_Python_Examples_Tutorials/examples/` -> `python examples/get_example_data.py` (Moco example data, from the pinned OpenSim source)
- `OpenSim_Python_Examples_Tutorials/` tutorials -> nothing to fetch; inputs are bundled in `resources/`

The root `.gitignore` covers repo-wide junk (bytecode, venvs, `.ipynb_checkpoints`,
`*.log`, run outputs); each module's own `.gitignore` adds its data and output
paths.

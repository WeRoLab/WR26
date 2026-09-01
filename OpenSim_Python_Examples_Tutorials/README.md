# OpenSim Python Examples & Tutorials

Curated, **runnable** OpenSim 4.6 Python material for the **Wearable Robotics
(WR26)** graduate course. Two parts:

| folder / files            | what it is |
|---------------------------|------------|
| `Tutorial 1-8 ....ipynb`  | Eight guided Jupyter notebooks (modeling &rarr; simulation &rarr; scaling/IK/ID &rarr; static optimization &rarr; Moco optimal control). |
| `examples/`               | ~35 short standalone scripts from the OpenSim source tree, as an API reference. See [`examples/README.md`](examples/README.md). |
| `resources/`              | Every input file the notebooks need (models, motion, marker, GRF, setup XML). Bundled - **nothing is downloaded**. |
| `tutorial_setup.py`       | One helper the notebooks call to stage `resources/` into a scratch dir. |

These files are also meant to be **reference material ("ground truth") for
agentic / LLM-assisted work**: they show correct, current OpenSim 4.6 Python API
usage that has been executed end to end against the pinned environment below.

## Provenance

Adapted from **[opensim-org/opensim-core](https://github.com/opensim-org/opensim-core)**
(`Bindings/Python/tutorials` and `Bindings/Python/examples`), Apache-2.0. The
tutorials were originally written for the **conda + Google Colab** workflow
([OpenSimColab](https://simtk.org/projects/opencolab),
[Mokhtarzadeh et al. 2022](https://doi.org/10.1080/10255842.2022.2104607)). This
copy is changed only to:

- assume a **pip-installed `opensim`** (no `conda`, no `condacolab`, no Colab);
- load inputs from the bundled **`resources/`** instead of `gdown` / Google Drive;
- drop Colab-only calls (`google.colab`, the "Open in Colab" badges).

The OpenSim API code itself is unchanged.

## Setup

OpenSim publishes official PyPI wheels from **4.6** onward - Windows, macOS, and
Linux, Python **3.11-3.13**, **Moco included**. No conda required.

```bash
python -m venv .venv
# Windows:            .venv\Scripts\activate
# macOS / Linux:      source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

If your platform has no wheel, `conda install opensim-org::opensim` into the same
environment instead; everything else here is unchanged.

`requirements.txt` pins the packages this material was validated with
(`opensim==4.6`, plus `numpy`, `scipy`, `matplotlib`, and - for Tutorial 8 -
`pandas`, `scikit-learn`, `ipywidgets`).

## How the notebooks find their data

Section `X.2` of each notebook calls:

```python
from tutorial_setup import prepare
prepare("Tutorial 5")
```

`prepare()` copies `resources/Tutorial 5/` into `_work/Tutorial 5/` and changes
the working directory there. The notebook then opens files by their bare names
(as the original Colab notebooks did), and everything it writes - solutions,
reports, logs - stays inside `_work/` (git-ignored). Delete `_work/` anytime to
reset. Run notebooks from **this folder** so `tutorial_setup` and `resources/`
resolve.

`prepare()` also puts the `opensim` package directory on the Windows DLL search
path - the pip `opensim` 4.6 wheel omits this, and without it Tutorials 7-8 fail
with `Plugin 'ipopt' is not found`. Every notebook calls `prepare()`, so this is
handled for you. (For the scripts in `examples/`, use `examples/run.py`.)

## The tutorials

| # | Topic | OpenSim tools / classes | Inputs (`resources/`) | Runtime* |
|---|-------|-------------------------|-----------------------|----------|
| 1 | Intro: build & simulate a pendulum | `ModelFactory`, `Manager`, `TimeSeriesTable` | - | seconds |
| 2 | Build & simulate a 2-body arm model | `Body`, `PinJoint`, `Millard2012EquilibriumMuscle`, `PrescribedController` | - | seconds |
| 3 | Load & modify a model | `Model` I/O, `BodySet`/`JointSet`/`CoordinateSet`, `printToXML` | `double_pendulum.osim` | seconds |
| 4 | Musculoskeletal models, motion files, `MuscleAnalysis` | `gait2392`, `AnalyzeTool`, `MuscleAnalysis`, `Storage` | `gait2392.osim`, `*_gait.mot` | ~1 min |
| 5 | Scaling, Inverse Kinematics, Inverse Dynamics | `ScaleTool`, `InverseKinematicsTool`, `InverseDynamicsTool` | gait2354 model + marker/GRF/setup files | ~2 min |
| 6 | Static Optimization (4 studies) | `AnalyzeTool` + `StaticOptimization`, reserve/residual actuators | gait10dof18musc model + IK/GRF | ~3-5 min |
| 7 | Intro to Moco: the "sliding mass" problem | `MocoStudy`, `MocoProblem`, `MocoCasADiSolver`, `MocoFinalTimeGoal` | - | ~1 min |
| 8 | Muscle-driven Moco: squat-to-stand, assistive device, synergies | `PolynomialPathFitter`, `MocoStudy`, `MocoParameter`, `SynergyController`, `osim.report` | `squatToStand_3dof9musc.osim` | ~10-20 min |

\* Order-of-magnitude, on a laptop CPU. Moco problems (7, 8) dominate.

## Known limitations

- **No 3D visualizer.** The Simbody visualizer needs a display and is not part of
  this setup; `study.visualize(...)` calls are left commented. Analysis and
  plotting (matplotlib) work fully.
- **No bone geometry meshes.** The pip `opensim` wheel ships no `Geometry/`
  folder, so loading `gait2392` / `squatToStand` prints
  `Couldn't find geometry file ...` warnings. These are cosmetic (meshes are
  visual only) and do not affect results.
- Notebooks are committed **without outputs**. Run them yourself; the table above
  and each notebook's text state what to expect.

## Environment validated

`opensim 4.6`, `numpy 2.5`, `scipy 1.18`, `matplotlib 3.11`, `pandas 2.x`,
`scikit-learn 1.x`, Python 3.13, Windows. All 8 notebooks execute end to end via
`jupyter nbconvert --to notebook --execute`.

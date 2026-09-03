# `examples/` - standalone OpenSim / Moco Python scripts

Short, focused scripts that each demonstrate one part of the OpenSim 4.6 Python
API. Copied verbatim (Apache-2.0) from
**[opensim-org/opensim-core](https://github.com/opensim-org/opensim-core)**
(`Bindings/Python/examples`, tag `4.6`) - the API code is unchanged; this folder
just adds this README and `get_example_data.py`.

Use them as an API reference alongside the notebooks in the parent folder.

## Running

```bash
pip install -r ../requirements.txt        # opensim 4.6 + numpy/scipy/matplotlib
python get_example_data.py                # once - data for the DATA-tagged scripts
python run.py Moco/exampleOptimizeMass.py # <- use the wrapper
```

`run.py <script>` `chdir`s into the script's folder (the scripts read/write
files by bare name) and puts the `opensim` package directory on `PATH`. **The
second part matters on Windows:** the pip `opensim` 4.6 wheel does not add itself
to `PATH`, so Moco's CasADi/Ipopt solver plugin fails to load and every Moco
example dies with `Plugin 'ipopt' is not found`. `run.py` fixes that.

Running a script directly also works if you handle both yourself:

```bash
cd Moco
# PowerShell:  $env:PATH = (python -c "import os,opensim;print(os.path.dirname(opensim.__file__))") + ';' + $env:PATH
# bash:        export PATH="$(python -c 'import os,opensim;print(os.path.dirname(opensim.__file__))'):$PATH"
python exampleOptimizeMass.py
```

Outputs (`*_solution.sto`, `*.omoco`, PDF reports) are git-ignored.

## Legend

| tag | meaning |
|-----|---------|
| **SC**   | self-contained - runs with only `pip install opensim` |
| **DATA** | needs `python get_example_data.py` first |
| **VIZ**  | calls `study.visualize(...)` / `setUseVisualizer(True)` - needs a display; the solve/analysis still runs, the 3D window step does not (comment it out to run headless) |
| **SKEL** | fill-in-the-blank teaching script - **read `*_answers.py` for working code** |
| **LIB**  | imported by another script, not run directly |
| **SLOW** | trajectory optimization; minutes, not seconds |

## Top level

| script | tag | what it shows |
|--------|-----|---------------|
| `build_simple_arm_model.py` | SC | Build a 2-body arm (`Body`, `PinJoint`, `Millard2012EquilibriumMuscle`, `PrescribedController`), simulate with `Manager`, write `SimpleArm.osim`. |
| `dynamic_walker_example_optimization.py` | SC | Load the bundled `dynamic_walker_example_model.osim`, sweep spring stiffness/damping, keep the model that walks furthest. |
| `wiring_inputs_and_outputs_with_TableReporter.py` | SC | Connect component `Output`s to a `TableReporter` and dump a `TimeSeriesTable` to `.sto`. |
| `posthoc_StatesTrajectory_example.py` | SC | Rebuild a `StatesTrajectory` from a saved states file and query it after the fact. |
| `numpy_conversions.py` | SC | Convert between OpenSim `Vector`/`Matrix`/`RowVector` and NumPy arrays. |
| `extend_OpenSim_Vec3_class.py` | SC | Add Python dunder methods (`__iter__`, `__getitem__`, ...) to `osim.Vec3` at run time. |

## `ExponentialContactForce/`

| script | tag | what it shows |
|--------|-----|---------------|
| `exampleExponentialContactForce.py` | SC | A block dropped onto a plane using `ExponentialContactForce`. |

## `Wrapping/`

| script | tag | what it shows |
|--------|-----|---------------|
| `exampleScholz2015GeometryPath.py` | SC, VIZ | A `PathSpring` on a `Scholz2015GeometryPath` wrapping a cylinder, on a double pendulum. |

## `PolynomialPathFitter/`

| script | tag | what it shows |
|--------|-----|---------------|
| `examplePolynomialPathFitter.py` | DATA, SLOW | Fit `FunctionBasedPath`s to `GeometryPath` muscle geometry (`subject_walk_scaled.osim`, `coordinates.sto`). |
| `examplePolynomialPathFitter_plotting.py` | LIB | Plotting helpers for the above. |

## `Moco/` - trajectory optimization

| script | tag | what it shows |
|--------|-----|---------------|
| `exampleSlidingMass.py` | SC, VIZ | Minimal `MocoStudy`: a point mass, minimum-time translation. |
| `exampleOptimizeMass.py` | SC | Optimize a body mass so a spring-mass system hits a target frequency (`MocoParameter`). |
| `exampleHangingMuscle.py` | SC, SLOW | A single hanging `DeGrooteFregly2016Muscle`; solve, then cross-check with `AnalyzeTool`. |
| `exampleKinematicConstraints.py` | SC | A point mass on a circular track - a kinematic (holonomic) constraint in Moco. |
| `examplePredictAndTrack.py` | SC, VIZ, SLOW | Double pendulum: predictive solve, then track the prediction's states / markers. |
| `plot_trajectory.py` | LIB | CLI/utility to plot a `MocoTrajectory` `.sto`. |
| `plot_casadi_sparsity.py` | LIB | Plot the NLP Jacobian/Hessian sparsity CasADi writes out. |

### `Moco/example2DWalking/`

| script | tag | what it shows |
|--------|-----|---------------|
| `example2DWalking.py` | DATA, VIZ, SLOW | `MocoTrack` gait tracking + a predictive gait problem with periodicity and contact-force goals. |
| `example2DWalkingMetabolics.py` | DATA, VIZ, SLOW | Adds a `Bhargava2004...` metabolic-cost goal. Also needs `gait10dof18musc.osim` (see `get_example_data.py` output). |
| `example2DWalkingStepAsymmetry.py` | DATA, VIZ, SLOW | Step-time and step-length asymmetry goals. Run `example2DWalking.py` first (uses its tracking solution). |

### `Moco/example3DWalking/`

| script | tag | what it shows |
|--------|-----|---------------|
| `exampleMocoInverse.py` | DATA, SLOW | `MocoInverse`: muscle activations from prescribed 3D kinematics + GRF; variant that tracks EMG. |
| `exampleMocoTrack.py` | DATA, VIZ, SLOW | `MocoTrack`: torque- then muscle-driven tracking of a 3D walking trial. |
| `example3DWalking.py` | DATA, VIZ, SLOW | Full 3D predictive/tracking gait with `FunctionBasedPath`s and foot-ground contact. |

### `Moco/exampleEMGTracking/`

| script | tag | what it shows |
|--------|-----|---------------|
| `exampleEMGTracking_answers.py` | DATA, SLOW | Track experimental EMG while tracking walking kinematics (`MocoStudy` + `MocoControlTrackingGoal`). |
| `exampleEMGTracking.py` | SKEL | Same, with blanks to fill in. |
| `exampleEMGTracking_helpers.py` | LIB | Builds the 18-muscle walking model and plotting helpers. |

### `Moco/exampleSquatToStand/`

| script | tag | what it shows |
|--------|-----|---------------|
| `exampleSquatToStand_answers.py` | DATA, VIZ, SLOW | Torque- vs muscle-driven squat-to-stand: predict, track, `MocoInverse`, add a knee `SpringGeneralizedForce`. |
| `exampleSquatToStand.py` | SKEL | Same, with blanks. |
| `exampleSquatToStand_helpers.py` | LIB | `getTorqueDrivenModel()` / `getMuscleDrivenModel()`. |
| `mocoPlotTrajectory.py` | LIB | Trajectory comparison plot. |
| `exampleIMUTracking/exampleIMUTracking_answers.py` | DATA, VIZ, SLOW | Synthesize IMU signals from a squat-to-stand, then track accelerometer/gyroscope data. |
| `exampleIMUTracking/exampleIMUTracking.py` | SKEL | Same, with blanks. |
| `exampleIMUTracking/exampleIMUTracking_helpers.py` | LIB | Model + IMU-frame helpers. |

## This folder's own files

| file | purpose |
|------|---------|
| `run.py` | wrapper: `chdir` + Windows Moco/`PATH` fix (see **Running**). |
| `get_example_data.py` | downloads data for the **DATA** scripts from the pinned `4.6` source. |
| `dynamic_walker_example_model.osim` | the one bundled model (input to `dynamic_walker_example_optimization.py`). |

## Notes

- Scripts assume their own folder is the working directory (`run.py` handles it).
  The **DATA** scripts also need `get_example_data.py` to have run.
- `study.visualize(...)` / `setUseVisualizer(True)` opens a Simbody 3D window and
  needs a display; headless it raises. Comment the line out, or run from a
  desktop session. The optimization/analysis before it still runs.
- The `# @title` / `# @markdown` comments in a few files are inert Google Colab
  form annotations - harmless in plain Python/Jupyter.
- `.osim` / `.sto` / `.mot` / `.trc` / `.xml` under `examples/` are git-ignored
  (they are fetched or generated); only the `.py` and this README are tracked.

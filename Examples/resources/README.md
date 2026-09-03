# `resources/` - bundled tutorial inputs

Every input file the eight tutorial notebooks need, one folder per tutorial. The
notebooks copy the relevant folder into `_work/<tutorial>/` at run time (via
`tutorial_setup.prepare`); these originals are never modified.

Originally these files were downloaded from per-tutorial Google Drive folders by
the OpenSim Colab notebooks. They come from the OpenSim project
([opensim-org/opensim-core](https://github.com/opensim-org/opensim-core),
`Bindings/Python/tutorials`) and the standard OpenSim example datasets
(`gait2354` / `gait2392` / `gait10dof18musc`, subject01 walking trial).

| folder | files | used for |
|--------|-------|----------|
| `Tutorial 3/` | `double_pendulum.osim` | load / inspect / modify a model |
| `Tutorial 4/` | `gait2392.osim`; `normal_gait.mot`, `crouch_gait.mot` | lower-limb model, motion files, `MuscleAnalysis` of hamstring length |
| `Tutorial 5/` | `gait2354_simbody.osim`; `gait2354_Setup_Scale.xml`, `gait2354_Scale_MarkerSet.xml`; `subject01_static.trc`, `subject01_walk1.trc`; `subject01_Setup_IK.xml`; `subject01_Setup_InverseDynamics.xml`, `subject01_walk1_grf.mot`, `subject01_walk1_grf.xml` | Scale &rarr; Inverse Kinematics &rarr; Inverse Dynamics |
| `Tutorial 6/` | `gait10dof18musc_simbody.osim`, `gait10dof18musc_Strong_actuators.xml`; `subject01_walk_IK.mot`, `subject01_walk_grf.mot`, `subject01_walk_grf.xml`; `subject_adjusted_Kinematics_q.sto` | Static Optimization (4 studies) |
| `Tutorial 7/` | `sliding_mass.png` | figure only (the model is built in code) |
| `Tutorial 8/` | `squatToStand_3dof9musc.osim`; `ipopt.opt`; `images/*.png` | `PolynomialPathFitter` + Moco squat-to-stand |

Tutorials 1 and 2 build their models entirely in code and need no resources.

**Geometry meshes are not included.** The gait models reference `.vtp` bone
meshes for visualization; loading them without the meshes prints
`Couldn't find geometry file ...` warnings, which are harmless here (the
tutorials do not use the 3D visualizer).

# Instructor solution -- marker-based inverse kinematics, two ways

Complete, validated solution for the WR26 sit-to-stand inverse-kinematics
activity.  Build/verify this first; the student-facing version lives in
`../activity/`.

## Run

```
cd Lectures/IK
pip install -r requirements.txt
python get_data.py            # one-time: fetch Data/ OpenSim/ "Python Documentation/"

cd solution
python s0_prepare_data.py     # raw C3D  -> TRC + GRF + events           (GIVEN)
python s1_prepare_model.py    # scaled model -> IK model                 (GIVEN)
python s2_opensim_ik.py       # OpenSim InverseKinematicsTool            (student)
python s3_planar_ik.py        # barehand 4-DOF nonlinear-least-squares   (student)
python s4_compare.py          # compare the two                         (student)
```

`python` = the `WR26` venv (`C:\Users\edgar\.venvs\WR26`).  End-to-end ~3 min.
All outputs land in `results/<trial>/`.

## What each step does

| step | in | out | key idea |
|---|---|---|---|
| **s0** | `Data/AB03/C3D/*.c3d` | `<trial>.trc`, `<trial>_grf.mot`, `<trial>_extloads.xml`, `<trial>_events.*` | read C3D with `opensim`; rotate lab frame (Z-up) -> OpenSim (Y-up) with an exact -90 deg turn about X; yaw-align so the subject faces +X; low-pass filter |
| **s1** | `Generated_Model.osim` | `results/Model_ik.osim` | strip muscles; lock arms / neck / subtalar / toes |
| **s2** | `.trc` + `Model_ik.osim` | `<trial>_ik.mot`, `<trial>_ik_marker_errors.sto` | `InverseKinematicsTool`: weighted least-squares marker tracking in 3-D |
| **s3** | `.trc` + `static.trc` | `<trial>_planar.mot`, `planar_calibration.json` | 4-DOF planar linkage (ankle-knee-hip-lumbar), foot planted; per-frame Levenberg-Marquardt fit of the sagittal markers; analytic Jacobian |
| **s4** | `_ik.mot` + `_planar.mot` | `<trial>_compare.png`, `s4_*` | align to the standing pose, tabulate joint-angle RMSE, repeatability |

## Verification (checked automatically)

1. **s0 rotation** reproduces the lab's own `static.trc` to `< 1e-6 m`.
2. **s0 GRF** total vertical settles to ~590 N standing (subject weight 60.0 kg * g).
3. **s2 OpenSim IK** marker RMS ~1.5-2.1 cm mean, ~5 cm worst single marker
   (typical mocap; the 2023 AddBiomechanics run of a similar trial got 3.0 cm).
4. **s3 analytic Jacobian** matches central differences to `< 1e-8`; the
   hand-rolled LM solution matches `scipy.optimize.least_squares` to `< 1e-6 deg`.
5. **s3 marker RMS** 0.3-0.5 cm standing, rising to ~5 cm at the deepest seated
   flexion -- the rigid planar model's error is concentrated where soft-tissue
   artefact and the fixed-foot / joint-centre assumptions are worst.
6. **s4 barehand vs OpenSim** joint-angle RMSE (mean over 3 reps):

   | motion | ankle | knee | hip | lumbar |
   |---|---|---|---|---|
   | sit_to_stand  | 4.7 | 5.9 | 6.4 | 3.1 |
   | stand_to_sit  | 4.9 | 4.8 | 6.2 | 3.9 |
   | squat         | 1.3 | 2.3 | 4.2 | 2.3 |

   The two methods agree within a few degrees through most of the motion and
   diverge ~10 deg at peak flexion (see any `*_compare.png`).

## Data notes

* Movement trials have **49 markers**; the static trial has **57** -- the medial
  markers (`RTAM`, `RFME`, `RFM2`, ...) are removed before the movement trials, so
  the barehand model calibrates joint centres from the static trial and, during
  movement, reconstructs them from the lateral markers plus the Harrington hip
  regression.
* Force plates: only plates **1 (right foot)** and **2 (left foot)** are active.
* The **squat** trials are ~17 s of quiet standing with a single squat; s2/s3
  crop them to a +-1.8 s window around the lowest point.
* GRF / ExternalLoads files are produced now but not used until the **inverse
  dynamics** follow-on (`s5`, next week).

# IK activity -- student files

Read **`handout.md`** first.

```
activity/
  handout.md            the assignment
  s0_prepare_data.py    GIVEN -- raw C3D -> results/<trial>/<trial>.trc  (+ GRF, events)
  s1_prepare_model.py   GIVEN -- scaled model -> results/Model_ik.osim
  s2_opensim_ik.py      YOU  -- fill build_task_set() + the tool config
  s3_planar_ik.py       driver (calls lib/planar_ik.py -- that is what you edit)
  s4_compare.py         driver (calls lib/compare.py -- that is what you edit)
  lib/
    io.py  c3d_prep.py  events.py  model_tools.py  calibration.py  paths.py  GIVEN
    planar_ik.py   YOU -- forward_kinematics, the Jacobian, solve_frame
    compare.py     YOU -- the angle signs and rmse_table
  solutions/            the four completed files (peek only after you try)
```

The full runnable instructor pipeline, with all `results/` and figures, is one
level up in `../solution/`.

## First-time setup

```
cd "IK Learning Module"
pip install -r requirements.txt
python get_data.py            # downloads Data/ OpenSim/ "Python Documentation/"  (~190 MB, once)
```

## Run order

```
cd "IK Learning Module/activity"
python s0_prepare_data.py     # given, just run it
python s1_prepare_model.py    # given, just run it
python s2_opensim_ik.py       # after you finish s2
python s3_planar_ik.py        # after you finish lib/planar_ik.py
python s4_compare.py          # after you finish lib/compare.py
```

`python` is the `WR26` virtual environment (`C:\Users\edgar\.venvs\WR26`).

## Checking yourself

- `s3` prints `analytic vs finite-difference Jacobian : <number>` -- it must be
  below ~1e-6, otherwise your Jacobian is wrong.
- `s3` also compares your Levenberg-Marquardt result to `scipy.optimize.least_squares`.
- barehand marker RMS: ~0.5 cm standing, up to ~5 cm at the deepest seated pose.
- `s4` barehand-vs-OpenSim joint RMSE: a few degrees for most of the motion.

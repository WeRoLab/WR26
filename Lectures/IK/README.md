# IK Learning Module — sit-to-stand inverse kinematics

An in-class activity for **Wearable Robotics (WR26)**: compute the joint angles of
a sit-to-stand from marker data using two methods: 1) with the OpenSim
`InverseKinematicsTool` and 2) with a hand-built 4-DOF planar model solved by
nonlinear least squares. Then, compare them and connect the result to
a wearable robot design.

```
Lectures/IK/
├── environment.md         venv + package versions + how to run
├── requirements.txt       pip dependencies
├── get_data.py            downloads the data / model / tutorials (see Setup)
├── solution/              COMPLETE instructor solution  (build/verify first)
│   ├── s0_prepare_data.py  ... s4_compare.py
│   ├── lib/                io, c3d_prep, events, model_tools, calibration,
│   │                       planar_ik, compare, paths
│   └── README.md           what each step does + expected numbers
├── activity/              STUDENT version
│   ├── handout.md          the assignment (start here)
│   ├── s2 / lib/planar_ik.py / lib/compare.py   have TODO blanks
│   └── solutions/          reference copy
│
│   -- fetched by get_data.py, NOT in the repo: --
├── Data/AB03/             raw motion capture (C3D) + a processed static trial
├── OpenSim/Model/         Rajagopal full-body model + geometry
└── Python Documentation/  OpenSim tutorials + Moco examples (reference)
```

## Setup

```
pip install -r requirements.txt      # numpy, scipy, matplotlib, gdown, opensim
python get_data.py                   # ~190 MB -> Data/, OpenSim/, Python Documentation/
```

`get_data.py` pulls one zip from Google Drive.  If it cannot reach Drive, it
prints a link to download the zip by hand.  (`opensim` may need conda on some
platforms — see `requirements.txt`.)

## Quick start (instructor)

```
cd solution
python s0_prepare_data.py && python s1_prepare_model.py && \
python s2_opensim_ik.py  && python s3_planar_ik.py     && python s4_compare.py
```

~3 minutes with the `WR26` venv.  Outputs and figures land in `solution/results/`.

## Status

- **This week:** inverse kinematics (`s0`–`s4`) — done and validated.
- **Next week:** inverse dynamics — joint torques from these angles plus the
  portable force plates (`solution/s5_inverse_dynamics.py`, placeholder).  The
  GRF / ExternalLoads files are already produced by `s0`.

## Headline results (barehand vs OpenSim, joint-angle RMSE, mean of 3 reps)

| motion | ankle | knee | hip | lumbar |
|---|---|---|---|---|
| sit_to_stand  | 4.7° | 5.9° | 6.4° | 3.1° |
| stand_to_sit  | 4.9° | 4.8° | 6.2° | 3.9° |
| squat         | 1.3° | 2.3° | 4.2° | 2.3° |

The two methods agree within a few degrees for most of the movement and diverge
~10° at the deepest flexion — where soft-tissue artefact, the fixed-foot
assumption and 2-D projection all bite. That divergence is the point of the
discussion.

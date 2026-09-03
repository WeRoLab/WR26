# Assignment 2 — Inverse kinematics from marker data

Compute the sagittal **right-knee angle** during one walking stride **two ways**
and compare them. Full instructions are in the assignment statement (Canvas /
Gradescope); this folder is the code.

| file | method | you write |
|---|---|---|
| `calculate_knee_angle.py` | **unconstrained** IK — anatomical reference frames + rotation matrices | the 4 `TODO` blocks (steps 1–4) |
| `opensim_ik_reference.py` | **constrained** IK — OpenSim `ScaleTool` + `InverseKinematicsTool` | 2 `TODO` blocks (marker weights, IK time window) |

Given, complete — read them, the OpenSim calls and the frame math are the point:

| file | what it does |
|---|---|
| `c3d_io.py` | read a C3D, rotate lab→OpenSim, write TRC, read MOT/STO — all via the OpenSim API |
| `frames_viz.py` | draw 3-D vectors and coordinate frames with matplotlib |
| `get_data.py` | download + unpack the data (below) |

## Setup

Use the course Python environment (`WR26`), then from this folder:

```
pip install -r requirements.txt
python get_data.py
```

`get_data.py` downloads `A2_Data.zip` (~5 MB) from Google Drive and unpacks it to
`data/` (git-ignored):

```
data/
  static.c3d               A-pose calibration trial   (57 markers, 200 Hz)
  stride_1.c3d             one walking stride          (49 markers, 200 Hz)
  Model/Biomech57.osim     generic full-body model (Rajagopal 2016)
  Model/marker_set.xml     the 57 OptiTrack Biomech-57 markers
  Model/Geometry/          bone meshes (3-D visualisation only)
  Model/save_data_info.m   subject AB03: mass 60.01 kg, height 1.70 m
```

Dataset: Alizadeh Noghani, Green & Bolívar-Nieto, *Scientific Data* **12**, 1971
(2025), subject **AB03**, "Stride" activity, repeat 1. Motion capture is in
metres, lab frame **Z up**. The medial knee/ankle markers (`RFME`, `RTAM`) are
"calibration" markers removed after the static trial, so `stride_1.c3d` has only
the lateral ones — `calculate_knee_angle.py` reconstructs them from `static.c3d`.

## Run

```
python opensim_ik_reference.py     # scale + IK  -> _work/stride_1_ik.mot  (the reference curve)
python calculate_knee_angle.py     # your anatomical-frame angle, overlaid on the reference
```

`python calculate_knee_angle.py --show-frames` also opens a 3-D view of your
femur/tibia frames at the start, middle and end of the stride.

### What a correct solution looks like

- `opensim_ik_reference.py`: marker error RMS ≈ **1.1 cm**; `knee_angle_r` a smooth
  0 → ~60° → 0 flexion curve over the ~3.6 s stride.
- `calculate_knee_angle.py`: the static-trial reconstruction residual prints as
  **≈ 0.2 mm**; the anatomical-frame curve tracks the OpenSim one to within a few
  degrees after a constant ~8° landmark offset (`knee_angle_stride.png`).

## Deliverable

`knee_angle_stride.png` plus written answers to the statement's questions,
including the David Winter (1983) reading.

# Assignment 3 — Knee torque from first principles

Compute the **right knee torque** during single-limb support (left leg in
swing) two ways and compare them: OpenSim's own `InverseDynamicsTool`, and a
bottom-up 2D (sagittal-plane) Newton-Euler calculation you derive and code
yourself, using a 2-segment (foot + shank) link-segment model. Full
instructions are in the assignment statement (Canvas / Gradescope); this
folder is the code.

| file | what it does | you write |
|---|---|---|
| `calculate_knee_torque.py` | given: loads joint angles, joint velocities, joint accelerations, ground reaction forces, and foot markers, together with the 2-segment model's own dimensions ($l_i$, $r_i$, $m_i$, $I_i$), all as NumPy arrays | the knee-torque derivation and calculation, below the loading section |

## Setup

Use the course Python environment (`WR26`). This assignment reads its inputs
from `Data/OpenSim_Tutorial3/Output/`, so run that pipeline first:

```
python ../../Data/OpenSim_Tutorial3/Python/scale_ik_id.py   # Scale + IK + ID -> Data/OpenSim_Tutorial3/Output/
python calculate_knee_torque.py                             # loads the results as NumPy arrays
```

See [`Data/OpenSim_Tutorial3/README.md`](../../Data/OpenSim_Tutorial3/README.md)
for what that pipeline does.

## What you get

Running `calculate_knee_torque.py` loads (and prints a summary of) each of
the following:

- `ik_time`, `ik_angles` — every IK joint angle over time (degrees)
- `bodies` — mass / center of mass / inertia for every body in the scaled model
- `grf` — ground reaction forces and centers of pressure, for both feet, resampled onto `ik_time`
- `markers` — `R.Heel`/`R.Toe.Tip` positions (meters, lab frame), resampled onto `ik_time`
- `segments` — for the "foot" and "shank" segments of a 2-segment model: $l_i$ (segment length), $r_i$ (distance to the center of mass), $m_i$ (segment mass), $I_i$ (segment mass moment of inertia about the center of mass), plus `PELVIS_TO_HIP`/`L_THIGH`
- `ik_kinematics` — joint angle value/velocity/acceleration (radians, rad/s, rad/s²) for the pelvis/hip/knee/ankle coordinates relevant to that model

The last two exist so you don't have to extract dimensions from the OpenSim
API or hand-differentiate joint angles yourself — your own work is the
bottom-up Newton-Euler derivation and its implementation, which starts after
the loading section.

## Deliverable

Written derivation of the equations of motion, your code, and the comparison
plot against OpenSim's own Inverse Dynamics result — see the assignment
statement for the full list, plus the David Winter (1983) reading.

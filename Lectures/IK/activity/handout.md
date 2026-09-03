# Inverse Kinematics from Marker Data — Two Ways

**Wearable Robotics (WR26) — in-class activity**

## Why this activity

An exoskeleton or exosuit that assists sit-to-stand needs a *reference*: how do the
ankle, knee, hip and trunk angles change as an unimpaired person stands up? That
reference comes from **inverse kinematics (IK)** — turning measured marker
positions into joint angles.

You will compute the sit-to-stand joint angles **two ways**:

- **Track A — OpenSim.** Drive the `InverseKinematicsTool` on a scaled
  musculoskeletal model. This is what most labs do.
- **Track B — "barehand."** Build a 4-link planar model yourself, write the
  forward-kinematics equations, and solve IK as a **nonlinear least-squares**
  problem with your own Gauss–Newton / Levenberg–Marquardt solver.

Then you compare them and explain the differences. The punchline: **OpenSim's IK
is the barehand method** — weighted nonlinear least squares on marker
positions — just in 3-D with a full skeleton.

## Learning objectives

By the end you can:

1. explain IK as an over-determined optimization, not a closed-form inversion;
2. write the forward-kinematics map from joint angles to marker positions for a
   serial chain, and differentiate it (the Jacobian);
3. implement Levenberg–Marquardt for weighted nonlinear least squares, with
   warm-starting across frames;
4. read and judge an IK result — marker residuals, what "good" looks like
   (~1–2 cm), where a model breaks down;
5. connect the joint-angle trajectories to exoskeleton design (reference
   trajectories, range of motion, seat-off timing).

## The data — `Data/AB03/`

Subject AB03 (60 kg), motion capture at 200 Hz + two portable force plates at
1000 Hz, sagittal-plane tasks:

| task | reps | notes |
|---|---|---|
| `sit_to_stand` | 3 | trimmed to the transition (~1 s) |
| `stand_to_sit` | 3 | the reverse |
| `squat` | 3 | long trials; one squat each |

* `OpenSim/Model/Biomech57.osim` — full-body model (Rajagopal 2016).
* `Data/AB03/Static/1/` — a standing calibration trial, **already** scaled
  (`generated/Generated_Model.osim`) and converted (`Data/static.trc`).

Get the data first (one time, from the repo root):

```
pip install -r requirements.txt
python get_data.py
```

You **start from `results/<trial>/<trial>.trc`** (markers, in the OpenSim frame)
produced by the provided `s0_prepare_data.py`. Read `s0` — do not treat the
C3D → TRC step as magic — but you do not have to write it.

### The one coordinate-frame fact you need

The capture frame is **Z-up**; the model frame is **Y-up, X-forward**. `s0`
rotates markers by an exact −90° about X, `[x, y, z] → [x, z, −y]`, then spins
each trial about vertical so the subject faces **+X**. After that the **sagittal
plane is the X–Y plane**: X = anterior, Y = up.

---

## Track A — OpenSim inverse kinematics  (`s2_opensim_ik.py`)

### A1. Model prep — given (`s1_prepare_model.py`)
`Generated_Model.osim` → `results/Model_ik.osim`: muscles removed, arms / neck /
subtalar / toes locked. The pelvis, hip, knee, ankle and lumbar coordinates stay
free in 3-D.

### A2. Build the IK task set — **you**
Every marker gets a weight in the least-squares cost. Bony landmarks (malleoli,
epicondyles, ASIS, heel) are trusted more than skin-mounted wands. Fill in
`build_task_set()` and pick weights. Run on all trials.

### A3. Read the result — **you**
For every trial report the per-frame **marker RMS** and **max** error from
`<trial>_ik_marker_errors.sto`. Plot `hip_flexion_r`, `knee_angle_r`,
`ankle_angle_r`, `lumbar_extension`. Is every frame under ~2 cm RMS? Which marker
is worst, and where in the movement?

### Questions
- What happens to the joint angles and the marker error if you set **all** weights
  equal? If you up-weight only the foot markers ×100?
- OpenSim solves each frame independently. How could using the previous frame
  help or hurt?

---

## Track B — barehand planar IK  (`s3_planar_ik.py`, `lib/planar_ik.py`)

### B1. The model

Sagittal plane, right side (assume left/right symmetry). Four rigid segments in a
chain **grounded at the foot**:

```
   ankle JC ──shank──▶ knee JC ──thigh──▶ hip JC ──pelvis──▶ lumbar JC ──trunk──▶
      (θ_ankle)          (θ_knee)          (θ_hip)            (θ_lumbar)
```

**Unknowns:** `q = [θ_ankle, θ_knee, θ_hip, θ_lumbar]`, measured as the change
from the standing pose (so `q = 0` when standing).

**Known every frame, straight from markers:**
- ankle joint centre `AJC` — from the lateral malleolus `RFAL` plus a small
  offset measured in the static trial (the medial malleolus marker is gone in
  the movement trials);
- foot orientation `φ_foot` — heel `RFCC` to forefoot `RFM1/RFM5`.

### B2. Forward kinematics

Absolute segment angles accumulate down the chain:

```
φ_shank  = φ_foot  + (φ_shank0  − φ_foot0)   + θ_ankle
φ_thigh  = φ_shank + (φ_thigh0  − φ_shank0)  + θ_knee
φ_pelvis = φ_thigh + (φ_pelvis0 − φ_thigh0)  + θ_hip
φ_trunk  = φ_pelvis+ (φ_trunk0  − φ_pelvis0) + θ_lumbar
```

(the `φ·0` are the static-pose reference angles from calibration). Joint centres:

```
KJC = AJC + L_shank · [cos φ_shank, sin φ_shank]
HJC = KJC + L_thigh · [cos φ_thigh, sin φ_thigh]
LJC = HJC + R(φ_pelvis) · d_pelvis→lumbar
```

Each tracked marker `i` sits at a **fixed local position** `(a_i, b_i)` in its
segment's frame (measured once, in the static trial):

```
mᵢ(q) = O_seg + R(φ_seg) · [a_i, b_i]ᵀ ,   R(φ) = [[cos φ, −sin φ],[sin φ, cos φ]]
```

Implement `forward_kinematics()` and `predict_markers()`.

### B3. The least-squares problem

```
r(q) = [ √wᵢ ( mᵢ(q) − mᵢ_measured ) ]  stacked over all tracked markers
f(q) = ½ ‖r(q)‖²
```

~16 markers → ~32 equations, 4 unknowns: heavily over-determined, and no `q` fits
all markers exactly (soft tissue, 2-D projection). That's why it's least squares.

### B4. The Jacobian

`J = ∂r/∂q` (2N × 4). In 2-D the derivative of a rotation is clean:

```
d/dφ  R(φ) = R(φ + π/2) ,     d/dφ [cos φ, sin φ] = [−sin φ, cos φ]
```

Because `φ_seg` depends on **all** upstream joints, a marker on the trunk has
non-zero derivatives w.r.t. all four unknowns; a marker on the shank, only w.r.t.
`θ_ankle`. Derive `∂mᵢ/∂θⱼ` and fill in the Jacobian in
`residual_and_jacobian()`. Check it against `jacobian_fd()` (finite differences)
— they must agree to ~1e-8.

You also need the 2×4 derivatives of the joint centres themselves,
`∂KJC/∂q`, `∂HJC/∂q`, `∂LJC/∂q` (`∂AJC/∂q = 0`). The provided code already uses
these (`dO[...]`) both for the marker Jacobian and for two optional **soft
anchors** — extra residuals that pull the chain's `HJC` toward the Harrington
estimate and `KJC` toward the lateral knee marker. Without them the foot-anchored
3-link chain lets ~13° of trunk lean leak into the hip; with them the hip error
roughly halves. You get the anchors for free once `dO` is correct.

### B5. The solver — Levenberg–Marquardt

```
repeat:
    r, J  = residual_and_jacobian(q)
    solve (JᵀJ + λ·diag(JᵀJ)) Δq = −Jᵀr
    if ‖r(q+Δq)‖ < ‖r(q)‖ :  accept, decrease λ
    else                  :  reject, increase λ
until ‖Jᵀr‖ small
```

`λ → 0` is Gauss–Newton (fast, needs a good guess); large `λ` is gradient descent
(slow, safe). Implement `solve_frame()`. For frame 1, initialise from the marker
geometry (`geometric_init()`); after that, **warm-start from the previous frame**.

### B6. Run and report — **you**
Solve every trial. Save `<trial>_planar.mot`. Plot the four angles and, next to
them, the marker RMS and iteration count vs time. Where is the fit worst? Why?

### Questions
- Your marker RMS should be ~5 mm standing and rise toward ~5 cm at the deepest
  seated posture. List three reasons the rigid planar model cannot do better
  there.
- The foot is *planted* (fixed `AJC`). What does that assume, and when is it
  violated in sit-to-stand?
- Move the assumed lumbar-joint height by ±5 cm and re-run. Which angles change?
  Why does the hip/trunk split depend on it?

---

## Track comparison  (`s4_compare.py`)

Express both tracks as **deviation from the standing pose** and match signs to the
OpenSim convention. Report per-joint **RMSE** and **peak error** in degrees; plot
the two on the same axes; check repeatability across the three reps.

Expected: agreement within a few degrees for most of the motion, ~10° divergence
at peak flexion, hip the worst joint.

### Discussion — hand in

1. Where do the two methods agree, where do they disagree, and **why** — name the
   assumption responsible for each disagreement (planarity, fixed foot, rigid
   segments, joint-centre estimates, soft-tissue artefact, 3-D vs 2-D).
2. Which method would you trust for an exoskeleton **reference trajectory**?
   Which for a quick field estimate with three markers and a laptop?
3. From the joint angles alone, when is **seat-off**? How would you detect it
   online for an exo controller?
4. Peak hip and knee flexion here vs. what an exo needs to allow — does a design
   that hard-stops the knee at 90° work for this subject?

## Deliverables

- Your completed `s2`, `s3` (with `lib/planar_ik.py`), `s4`.
- The `*_compare.png` for one `sit_to_stand`, one `stand_to_sit`, one `squat`.
- ≤ 1 page answering the discussion questions.

*(Next week: inverse dynamics — joint torques from these angles plus the force
plates — and matching them to real actuator specs.)*

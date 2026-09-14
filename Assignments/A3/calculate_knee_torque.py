"""Load the inputs for a "by hand" (first-principles) knee torque calculation.

Assignment 3 asks you to compute the right knee torque, from first
principles, during a time window when the left leg is in swing (not
touching the ground) -- then compare it against OpenSim's own inverse
dynamics answer and against the torques reported in Winter's textbook.

Before this file will work, you must have already run
`Data/OpenSim_Tutorial3/Python/scale_ik_id.py`, which performs the Scaling,
Inverse Kinematics, and Inverse Dynamics steps in OpenSim and writes their
results into `Data/OpenSim_Tutorial3/Output/`. This file only *loads* those
results (plus the raw ground reaction force data) into plain NumPy arrays --
it does not compute anything for you. Your knee-torque calculation is the
next thing to write, below the loading code.

----------------------------------------------------------------------------
New to Python? Things you will meet in this file:
  * A dictionary (`dict`) maps a name to a value, e.g. `ik_angles["knee_angle_r"]`
    is the NumPy array of the right knee angle over time. `{}` starts an
    empty dict; `some_dict[key] = value` adds an entry.
  * `numpy.interp(new_x, old_x, old_y)` linearly interpolates a signal
    (`old_y`, sampled at `old_x`) onto a new set of sample points (`new_x`).
    It is used below to resample the ground reaction forces.
  * `Path(__file__).resolve().parents[2]` walks up from this file's location
    two folders at a time -- see the comment where it's defined.
----------------------------------------------------------------------------
"""

import sys
from pathlib import Path

import numpy as np
import opensim as osim

# --------------------------------------------------------------------------- #
# Where everything lives
# --------------------------------------------------------------------------- #

# This file is Root/Assignments/A3/calculate_knee_torque.py.
# parents[0] = Root/Assignments/A3, parents[1] = Root/Assignments, parents[2] = Root.
REPO_DIR = Path(__file__).resolve().parents[2]
TUTORIAL_DIR = REPO_DIR / "Data" / "OpenSim_Tutorial3"
OUTPUT_DIR = TUTORIAL_DIR / "Output"
DATA_DIR = TUTORIAL_DIR / "Data"

# Joint angles solved by Inverse Kinematics -- written by scale_ik_id.py's
# IK step (Step 2), one row per time sample, one column per joint angle.
ik_mot_file = OUTPUT_DIR / "subject01_walk1_ik.mot"

# The gait2354 model after Scaling (Step 1): resized to subject01's segment
# lengths, with markers placed -- this is where the segment mass, center of
# mass, and inertia below come from.
scaled_model_file = OUTPUT_DIR / "subject01_simbody.osim"

# Raw ground reaction force + center-of-pressure data recorded by the force
# plates during the walking trial. This is an INPUT to the pipeline (it
# already existed before scale_ik_id.py ran; it lives in Data/, not
# Output/), and is also the same file ID's external_loads_file setup uses.
grf_mot_file = DATA_DIR / "subject01_walk1_grf.mot"

if not ik_mot_file.exists() or not scaled_model_file.exists():
    sys.exit(
        "Missing OpenSim results in "
        f"{OUTPUT_DIR}\n"
        "Run Data/OpenSim_Tutorial3/Python/scale_ik_id.py first."
    )

print("Loading inverse dynamics inputs ...")

# --------------------------------------------------------------------------- #
# 1. Joint angles from Inverse Kinematics
# --------------------------------------------------------------------------- #

ik_table = osim.TimeSeriesTable(str(ik_mot_file))
ik_time = np.array(ik_table.getIndependentColumn())

# One NumPy array per coordinate, e.g. ik_angles["knee_angle_r"], keyed by
# the same column names OpenSim uses (hip_flexion_r, knee_angle_r,
# ankle_angle_r, and their _l counterparts, plus the pelvis and lumbar
# coordinates). All are in degrees, matching the .mot file.
ik_angles = {}
for label in ik_table.getColumnLabels():
    ik_angles[label] = ik_table.getDependentColumn(label).to_numpy()

print(f"  IK: {len(ik_time)} time points from {ik_time[0]:.3f} to {ik_time[-1]:.3f} s, "
      f"{len(ik_angles)} coordinates")

# --------------------------------------------------------------------------- #
# 2. Anthropometric data (mass, center of mass, inertia) from the scaled model
# --------------------------------------------------------------------------- #

# Load the scaled model (the one scale_ik_id.py wrote) fresh, just for this.
# (The .vtp files below are body-shape meshes used only for 3D display --
# not needed for this loading step -- so this just quiets OpenSim's warnings
# about not finding them.)
osim.ModelVisualizer.addDirToGeometrySearchPaths(str(TUTORIAL_DIR / "Geometry"))
scaled_model = osim.Model(str(scaled_model_file))
scaled_state = scaled_model.initSystem()
body_set = scaled_model.getBodySet()

# One entry per body, e.g. bodies["tibia_r"]["mass"], bodies["tibia_r"]["mass_center"].
bodies = {}
# OpenSim's BodySet is indexed like a list by number, not looped directly.
for i in range(body_set.getSize()):
    body = body_set.get(i)
    mass_center = body.getMassCenter()
    moments = body.getInertia().getMoments()     # Ixx, Iyy, Izz
    products = body.getInertia().getProducts()   # Ixy, Ixz, Iyz
    bodies[body.getName()] = {
        "mass": body.getMass(),
        "mass_center": np.array([mass_center[0], mass_center[1], mass_center[2]]),
        "inertia_moments": np.array([moments[0], moments[1], moments[2]]),
        "inertia_products": np.array([products[0], products[1], products[2]]),
    }

print(f"  Anthropometrics: {len(bodies)} bodies -- {', '.join(bodies.keys())}")

# --------------------------------------------------------------------------- #
# 3. Ground reaction forces, resampled onto the IK time base
# --------------------------------------------------------------------------- #

# The force plate data (subject01_walk1_grf.mot) was NOT recorded at the same
# sample rate, or over the same time span, as the IK solution above: the
# force plate covers the whole trial at ~600 Hz, while IK only solved the
# trimmed 0.4-1.6 s window at ~60 Hz. To let you combine a joint angle and
# the ground force acting at that same instant, we linearly interpolate every
# GRF column onto `ik_time` -- IK's window sits entirely inside the force
# plate's recorded range, so this is interpolation, never extrapolation.
grf_table = osim.TimeSeriesTable(str(grf_mot_file))
grf_time = np.array(grf_table.getIndependentColumn())

# One NumPy array per column, resampled onto ik_time. Columns without a "1_"
# prefix are the RIGHT foot; columns starting with "1_" are the LEFT foot:
#   ground_force_v{x,y,z}    right-foot ground reaction force vector (N)
#   ground_force_p{x,y,z}    right-foot center of pressure (m)
#   1_ground_force_v{x,y,z}  left-foot ground reaction force vector (N)
#   1_ground_force_p{x,y,z}  left-foot center of pressure (m)
#   ground_torque_{x,y,z}    right-foot free moment
#   1_ground_torque_{x,y,z}  left-foot free moment
#
# IMPORTANT -- reference frame: these forces and centers of pressure are
# expressed in the lab ("ground") frame, the same global frame the scaled
# model and marker data use here (see the `point_expressed_in_body = ground`
# setting in Setup/subject01_walk1_grf.xml). That makes them consistent with
# *this* skeleton. If you build your own simplified segment model for the
# knee free-body diagram, relating its frames back to this same ground frame
# is part of your own modeling work -- this loader can't decide that for you.
#
# IMPORTANT -- CoP validity: a center of pressure (ground_force_p*) is only
# physically meaningful while the paired vertical force (ground_force_vy) is
# well above zero. Near foot-strike/foot-off, and throughout swing, the
# vertical force is close to zero and the reported CoP location is noisy or
# meaningless -- this is not filtered out below, so take it into account
# when you pick your analysis window.
grf = {}
for label in grf_table.getColumnLabels():
    grf_values = grf_table.getDependentColumn(label).to_numpy()
    grf[label] = np.interp(ik_time, grf_time, grf_values)

print(f"  Ground reaction forces: {len(grf)} columns, resampled to {len(ik_time)} points")

# --------------------------------------------------------------------------- #
# 4. Right-foot markers (R.Heel, R.Toe.Tip), resampled onto the IK time base
# --------------------------------------------------------------------------- #

# The ground reaction force's center of pressure (grf["ground_force_p*"]) is
# given in the lab/ground frame -- it does not, by itself, say WHERE on the
# foot the force is being applied. These two experimental markers (from the
# same motion-capture trial used for IK) can help with that: the vector from
# R.Heel to R.Toe.Tip approximates the foot's long (heel-to-toe) axis, in the
# lab frame, at each instant. Projecting the CoP onto that direction (and the
# perpendicular one) is one way to turn "CoP in the lab frame" into "how far
# along the foot, from the heel, the force is acting" -- useful if your
# free-body diagram needs the CoP relative to the foot rather than the lab.
marker_table = osim.TimeSeriesTableVec3(str(DATA_DIR / "subject01_walk1.trc"))
marker_time = np.array(marker_table.getIndependentColumn())

# One (N, 3) array per marker, e.g. markers["R.Heel"][:, 0] is its x
# coordinate over time.
markers = {}
for marker_name in ("R.Heel", "R.Toe.Tip"):
    marker_column = marker_table.getDependentColumn(marker_name)

    # Each row of marker_column is a Vec3 (x, y, z), in millimeters (see the
    # "Units" field in subject01_walk1.trc's header) -- build a plain
    # (N, 3) array and convert to meters, to match the model/GRF units used
    # everywhere else in this file.
    xyz_mm = np.zeros((marker_table.getNumRows(), 3))
    for i in range(marker_table.getNumRows()):
        point = marker_column[i]
        xyz_mm[i, 0] = point[0]
        xyz_mm[i, 1] = point[1]
        xyz_mm[i, 2] = point[2]
    xyz_m = xyz_mm / 1000.0

    # Marker data covers the whole trial at 60 Hz; IK only covers the
    # trimmed 0.4-1.6 s window -- resample onto ik_time exactly like the
    # ground reaction forces above, one axis (column) at a time.
    resampled = np.zeros((len(ik_time), 3))
    for axis in range(3):
        resampled[:, axis] = np.interp(ik_time, marker_time, xyz_m[:, axis])
    markers[marker_name] = resampled

print(f"  Foot markers: {', '.join(markers.keys())}, resampled to {len(ik_time)} points")

# --------------------------------------------------------------------------- #
# 5. Two-segment (foot + shank) model dimensions, from the scaled model's
#    default pose
# --------------------------------------------------------------------------- #

# A "station" is a point fixed to a body/frame; this returns where that
# point sits in the ground frame, at the model's default pose, x and y only
# (this is a sagittal-plane model, so the z coordinate is dropped).
def station_xy(frame, local_point=(0.0, 0.0, 0.0)):
    local_vec3 = osim.Vec3(local_point[0], local_point[1], local_point[2])
    p = frame.findStationLocationInGround(scaled_state, local_vec3)
    return np.array([p[0], p[1]])


joint_set = scaled_model.getJointSet()

pelvis_xy = station_xy(body_set.get("pelvis"))
hip_xy = station_xy(joint_set.get("hip_r").getChildFrame())
knee_xy = station_xy(joint_set.get("knee_r").getChildFrame())
ankle_xy = station_xy(joint_set.get("ankle_r").getChildFrame())
mtp_xy = station_xy(joint_set.get("mtp_r").getChildFrame())

# Two more fixed quantities needed to chain position from the pelvis down
# to the knee (see PELVIS_TO_HIP/L_THIGH below); not needed for the knee
# free-body equations themselves.
PELVIS_TO_HIP = hip_xy - pelvis_xy            # fixed 2D offset, ground frame, default pose
L_THIGH = np.linalg.norm(hip_xy - knee_xy)    # hip-to-knee distance

# Shank (i=2): OpenSim already models this as one rigid body (tibia_r) --
# reuse the mass/center-of-mass/inertia already loaded into `bodies` above.
# r_2 is measured from the PROXIMAL joint (the knee) to the shank's COM.
shank_l = np.linalg.norm(knee_xy - ankle_xy)  # knee-to-ankle distance
shank_r = np.linalg.norm(bodies["tibia_r"]["mass_center"])
shank_m = bodies["tibia_r"]["mass"]
shank_I = bodies["tibia_r"]["inertia_moments"][2]  # about the out-of-plane (Z) axis

# Foot (i=1): the model splits the foot into three separate bodies
# (talus_r, calcn_r, toes_r). The subtalar and MTP joint angles stay close
# to their default values for this whole trial (they are heavily weighted
# toward `default_value` in the Scale/IK task sets), so treating the three
# as one rigid segment -- combined once, here, at the default pose -- is a
# reasonable simplification. Combining them takes two standard steps:
# a mass-weighted average for the combined center of mass, then the
# parallel-axis theorem to shift each body's own inertia onto that combined
# center of mass before adding the three together.
foot_body_names = ("talus_r", "calcn_r", "toes_r")
foot_mass = 0.0
foot_com_weighted = np.zeros(2)
foot_com_by_body = {}
for name in foot_body_names:
    body = body_set.get(name)
    mass = body.getMass()
    com_xy = station_xy(body, body.getMassCenter())
    foot_com_by_body[name] = (mass, com_xy)
    foot_mass += mass
    foot_com_weighted += mass * com_xy
foot_com_xy = foot_com_weighted / foot_mass

foot_I = 0.0
for name in foot_body_names:
    mass, com_xy = foot_com_by_body[name]
    izz_own = body_set.get(name).getInertia().getMoments()[2]
    distance_to_combined_com = np.linalg.norm(com_xy - foot_com_xy)
    foot_I += izz_own + mass * distance_to_combined_com**2

foot_l = np.linalg.norm(ankle_xy - mtp_xy)          # ankle-to-mtp distance
foot_r = np.linalg.norm(foot_com_xy - ankle_xy)     # ankle (proximal) -> foot COM
foot_m = foot_mass

# l_i, r_i, m_i, I_i for each segment of the 2-segment model, e.g.
# segments["shank"]["I"] is the shank's moment of inertia about its own COM.
segments = {
    "foot": {"l": foot_l, "r": foot_r, "m": foot_m, "I": foot_I},
    "shank": {"l": shank_l, "r": shank_r, "m": shank_m, "I": shank_I},
}

print("  Segment dimensions:")
for seg_name, seg in segments.items():
    print(f"    {seg_name}: l={seg['l']:.4f} m, r={seg['r']:.4f} m, "
          f"m={seg['m']:.4f} kg, I={seg['I']:.6f} kg*m^2")

# --------------------------------------------------------------------------- #
# 6. Joint angle kinematics (value, velocity, acceleration)
# --------------------------------------------------------------------------- #

# InverseDynamicsTool itself does not differentiate the raw IK angles
# directly -- it first fits a smoothing spline to each coordinate (see its
# lowpass_cutoff_frequency_for_coordinates setting) and differentiates
# THAT, rather than taking noisy finite differences of the raw samples.
# osim.GCVSplineSet does the same kind of fit; splines.evaluate(index,
# order, t) then gives the value (order=0), velocity (order=1), or
# acceleration (order=2) of a coordinate at any time t.
ik_storage = osim.Storage(str(ik_mot_file))
splines = osim.GCVSplineSet(5, ik_storage, -1)

# The coordinates needed to build a sagittal-plane, 2-segment (foot+shank)
# model: the pelvis's own planar position/tilt, plus the relative hip,
# knee, and ankle angles that chain down through the shank to the foot.
KINEMATIC_COORDS = [
    "pelvis_tx", "pelvis_ty", "pelvis_tilt",
    "hip_flexion_r", "knee_angle_r", "ankle_angle_r",
]

# One (value, velocity, acceleration) tuple of (N,) arrays per coordinate,
# e.g. ik_kinematics["knee_angle_r"][1] is knee angular velocity (rad/s).
# Angles are converted from degrees (the .mot file's units) to radians;
# pelvis_tx/pelvis_ty are already in meters, so they pass through unchanged.
ik_kinematics = {}
for name in KINEMATIC_COORDS:
    idx = splines.getIndex(name)
    value = np.array([splines.evaluate(idx, 0, t) for t in ik_time])
    velocity = np.array([splines.evaluate(idx, 1, t) for t in ik_time])
    acceleration = np.array([splines.evaluate(idx, 2, t) for t in ik_time])
    if name not in ("pelvis_tx", "pelvis_ty"):
        value = np.radians(value)
        velocity = np.radians(velocity)
        acceleration = np.radians(acceleration)
    ik_kinematics[name] = (value, velocity, acceleration)

print(f"  Joint kinematics: value/velocity/acceleration for {len(ik_kinematics)} coordinates")

# --------------------------------------------------------------------------- #
# Your knee-torque calculation starts here.
#
# You now have:
#   ik_time      -- (N,) array of time points, in seconds
#   ik_angles    -- dict of (N,) arrays, one per joint angle, in degrees
#   bodies       -- dict of per-body mass / center of mass / inertia
#   grf          -- dict of (N,) arrays, ground reaction forces and centers
#                   of pressure for both feet, resampled onto ik_time
#   markers      -- dict of (N, 3) arrays, R.Heel and R.Toe.Tip positions
#                   (meters, lab frame), resampled onto ik_time
#   segments     -- dict of l_i, r_i, m_i, I_i for "foot" and "shank"
#   PELVIS_TO_HIP, L_THIGH -- fixed dimensions to reach the knee from the
#                   pelvis (segments doesn't include the thigh -- it never
#                   appears in the knee-moment equations, only in the
#                   position chain used to get there)
#   ik_kinematics -- dict of (value, velocity, acceleration) tuples, one per
#                   pelvis/hip/knee/ankle coordinate (radians, rad/s, rad/s^2)
#
# Next: pick a time window where the left foot is in swing (look at
# grf["1_ground_force_vy"] -- it should be near zero there, since there is
# no left-foot ground contact). Then, using the bottom-up approach from
# class: assemble each segment's absolute orientation from the relative IK
# angles above, build the position/velocity/acceleration of each joint and
# segment center of mass, and write the Newton-Euler equations of motion
# for the foot and shank to solve for the right knee torque.
# --------------------------------------------------------------------------- #

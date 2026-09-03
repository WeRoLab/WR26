"""Prepare the provided scaled model for inverse kinematics.

Scaling was already done by the lab -- ``Data/AB03/Static/1/generated/
Generated_Model.osim`` (subject mass 60.01 kg).  Here we only:

* strip all muscles / torque actuators (IK does not use forces), and
* lock the upper body, neck and foot-detail joints (their markers are not part
  of this sagittal activity), so IK runs fast and those segments do not drift.

The lower-limb + trunk coordinates are left **free** in 3-D.  We deliberately do
*not* weld the model down to a strict 2-D linkage: locking the small out-of-plane
rotations (``pelvis_list`` ~ 6 deg, ``hip_rotation`` ~ 10 deg here) actually makes
the model a *worse* fit to the markers (RMS 1.5 cm -> 2.5 cm) and shifts the
sagittal angles by up to 3 deg.  Instead we run the honest 3-D solve and pull the
sagittal coordinates out for the comparison in ``s4`` -- the residual difference
against the barehand model then measures how planar the motion really is.

----------------------------------------------------------------------------
OpenSim vocabulary used below:
  * Model        -- the whole skeleton: bodies, joints, markers, muscles.
  * coordinate   -- one joint angle / degree of freedom, e.g. "knee_angle_r".
  * ForceSet     -- the muscles and torque actuators attached to the model.
  * "locking" a coordinate holds it at a fixed value during the IK solve.
  * OpenSim methods come in two flavours:  get...()  returns a read-only view,
    upd...()  ("update") returns an editable one.
  * OpenSim collections are indexed like C++ / Java, not iterated like Python:
        for i in range(cs.getSize()):  c = cs.get(i)
----------------------------------------------------------------------------
"""

# Makes the ": type" hints lazy; harmless, ignore it.
from __future__ import annotations

import pathlib                       # file paths
import numpy as np                   # (kept for helpers; not heavily used here)
import opensim as osim               # the OpenSim Python library

# --- tell OpenSim where the bone mesh files live --------------------------- #
# The model .osim references files like "r_femur.vtp".  Walk up the folder tree
# from this file; the first folder that has "OpenSim/Model/Geometry" inside it
# is the one we register, so OpenSim can find (and stop warning about) the meshes.
for _p in pathlib.Path(__file__).resolve().parents:
    _g = _p / "OpenSim" / "Model" / "Geometry"
    if _g.is_dir():
        osim.ModelVisualizer.addDirToGeometrySearchPaths(str(_g))
        break                        # found it -> stop searching


# --- configuration lists ------------------------------------------------- #

# Coordinates (joint angles) we hold fixed: both arms, both wrists, the neck,
# and the small foot joints.  Their markers are not used in this activity, so
# freezing them keeps the IK solve fast and stops those segments wandering.
# A Python list is written [a, b, c]; it can span several lines.
LOCK_COORDS = [
    "subtalar_angle_r", "mtp_angle_r", "subtalar_angle_l", "mtp_angle_l",
    "arm_flex_r", "arm_add_r", "arm_rot_r", "elbow_flex_r", "pro_sup_r",
    "wrist_flex_r", "wrist_dev_r",
    "arm_flex_l", "arm_add_l", "arm_rot_l", "elbow_flex_l", "pro_sup_l",
    "wrist_flex_l", "wrist_dev_l",
    "neck_bending", "neck_rotation", "neck_extension",
]

# The four coordinates s4 compares, one-to-one, against the barehand model.
COMPARE_COORDS = ["ankle_angle_r", "knee_angle_r", "hip_flexion_r", "lumbar_extension"]

# Marker names the IK solve should track: the right lower limb + trunk (the part
# we care about) plus the left leg (so the 3-D solve is still well constrained).
# The same list is used by the barehand model, so both methods see the same data.
IK_MARKERS = [
    "RFCC", "RFM1", "RFM2", "RFM5", "RDP1",
    "RFAL", "RTAM", "RSK", "RTTC", "RFAX",
    "RFLE", "RFME", "RTH", "RFTC",
    "RIAS", "LIAS", "RIPS", "LIPS",
    "CV7", "SJN", "SXS", "TV2", "TV7",
    "LFCC", "LFM1", "LFM5", "LFAL", "LTAM", "LSK", "LTTC", "LFLE", "LFME", "LTH", "LFTC",
]


def make_ik_model(generated_osim, out_path):
    """Load the lab-scaled model, trim it for IK, and save it to `out_path`.

    `generated_osim` and `out_path` are file paths.  Returns the in-memory model
    object as well (handy for the print-out at the end).
    """
    model = osim.Model(str(generated_osim))       # load the .osim file
    model.setName("AB03_ik")                      # rename it (cosmetic)

    # updForceSet() = the editable list of muscles + actuators.
    # clearAndDestroy() removes them all -- inverse kinematics never uses forces.
    model.updForceSet().clearAndDestroy()

    # initSystem() tells OpenSim "I'm done editing for now, build the internal
    # machinery".  It must be called before we can query coordinates.
    model.initSystem()

    # updCoordinateSet() = the editable collection of all joint coordinates.
    cs = model.updCoordinateSet()
    locked = []                                   # names we end up locking (for the print-out)
    for i in range(cs.getSize()):                 # OpenSim collections: loop by index
        c = cs.get(i)                             # coordinate number i
        if c.getName() in LOCK_COORDS:            # is this one on our lock list?
            c.set_locked(True)                    # yes -> freeze it
            locked.append(c.getName())            # remember that we did

    model.finalizeConnections()                   # re-wire the model after the edits
    model.initSystem()                            # rebuild the internal machinery
    model.printToXML(str(out_path))               # save the trimmed model to disk

    # f-strings with {expr}.  ", ".join(list_of_strings) glues them with commas.
    print(f"[model] wrote {out_path}")
    print(f"[model]   locked: {', '.join(locked)}")
    print(f"[model]   free  : {', '.join(free_coordinates(model))}")
    return model


def free_coordinates(model):
    """List the coordinates that are actually solved for: not locked, and not
    driven by another coordinate (OpenSim calls that 'constrained' -- e.g. the
    knee cap follows the knee angle).
    """
    model.initSystem()
    st = model.getWorkingState()                  # a snapshot of the model's state
    cs = model.getCoordinateSet()                 # read-only view is fine here
    # This is a "list comprehension": build a list by looping and filtering in
    # one line.  Read it as:
    #   [ name(i)  for each i  if  (not locked)  and  (not constrained) ]
    return [cs.get(i).getName() for i in range(cs.getSize())
            if not cs.get(i).get_locked() and not cs.get(i).isConstrained(st)]

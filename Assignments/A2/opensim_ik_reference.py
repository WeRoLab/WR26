"""Constrained inverse kinematics with the OpenSim API.

This produces the "reference" knee angle that `calculate_knee_angle.py` compares
against.  It walks the standard OpenSim marker-IK pipeline (the same one as
Tutorial 5 of the course):

Fill in the two `TODO` blocks: `build_task_set()` (marker weights) and the time
window in `run_ik()`.  Everything else is given -- read it, this is the point of
the exercise.

    1.  C3D  ->  .trc          convert both trials into the OpenSim frame
    2.  Scale               fit the generic Biomech-57 model to subject AB03
    3.  Trim               drop muscles, lock the upper body (IK needs neither)
    4.  Inverse Kinematics   solve joint angles that best match the stride markers
    5.  Read the result     pull `knee_angle_r` and the marker errors back out

Run order for the assignment:

    python get_data.py                 (in Assignments/A2/)
    python opensim_ik_reference.py     <-- this file, writes _work/stride_1_ik.mot
    python calculate_knee_angle.py

----------------------------------------------------------------------------
OpenSim API vocabulary used below
  * Model              the skeleton: bodies, joints, coordinates, markers, muscles.
  * coordinate         one joint angle / DOF, e.g. "knee_angle_r".
  * ScaleTool          resizes a generic model to a subject from a static trial.
      GenericModelMaker    which .osim + marker set to start from
      ModelScaler          how to resize each body (here: marker-pair "measurements")
      MarkerPlacer         nudge the model markers onto the subject's static pose
  * InverseKinematicsTool  per-frame weighted least-squares marker tracking.
  * IKTaskSet / IKMarkerTask   which markers to track and how much to trust each.
  * OpenSim collections are indexed like C++/Java, not iterated like Python:
        for i in range(s.getSize()):  item = s.get(i)
  * get...() returns a read-only view;  upd...() ("update") returns an editable one.
  * OpenSim wants file paths as plain strings, so we wrap Path objects: str(path).

----------------------------------------------------------------------------
New to Python?  Things you will meet in this file:
  * `def name(args):` defines a function.  `return a, b` returns two values at
    once; the caller writes `x, y = name(...)` to unpack them.
  * an argument written `name=value` in the `def` line is a DEFAULT: callers may
    leave it out.  `*values` collects "any number of extra arguments" into a tuple.
  * a name in ALL_CAPS at the top of the file is a module-level constant.
  * `[a, b, c]` is a list (ordered).  `{a, b, c}` is a set (no order, but very
    fast "is x in here?" checks -- that is why BONY and LOCK_COORDS are sets).
  * f-string: `f"knee is {angle:.1f} deg"` drops the value of `angle` into the
    text, formatted to 1 decimal.
  * `{k: f(v) for k, v in d.items()}` is a "dict comprehension": build a new dict
    by transforming every entry of `d`.  `[f(x) for x in xs if cond]` is the list
    version.
  * `pathlib.Path` represents a file path.  `p / "sub" / "file.txt"` joins paths
    with `/` (on any OS).  `p.parent`, `p.glob("*.sto")`, `p.mkdir()` etc.
  * a leading underscore (`_measurement`, `_model`) means "internal / not
    important to the caller".  A bare `_` means "a value I must catch but will
    not use".
  * `if __name__ == "__main__":` at the bottom = "run main() only when this file
    is executed directly, not when it is imported".
----------------------------------------------------------------------------
"""

# Makes the ": type" hints below lazy; harmless, you can ignore it.
from __future__ import annotations

import pathlib                       # work with file paths as objects

import numpy as np                   # arrays and math
import opensim as osim               # the OpenSim Python library

# Pull a few helpers out of our own c3d_io.py (in this same folder).
from c3d_io import find_data_dir, read_c3d_markers, read_storage, to_opensim_frame, write_trc

# --- subject AB03, from data/Model/save_data_info.m --------------------------- #
SUBJECT_MASS_KG = 60.01
SUBJECT_HEIGHT_M = 1.70

# __file__ is the path to THIS script.  .resolve() makes it absolute; .parent is
# the folder it lives in.  So HERE = ".../Assignments/A2".
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE / "_work"                       # scratch folder: .trc, scaled model, IK output
DATA = find_data_dir()                      # the data/ folder made by get_data.py

# Keep OpenSim's console chatter down to real warnings.
osim.Logger.setLevelString("warn")

# Markers we ask inverse kinematics to track.  The right lower limb + trunk (what
# we care about) plus the left leg so the 3-D solve stays well constrained.  Bony
# landmarks are trusted more than skin-mounted wands (RTH, RSK) -- see build_task_set.
# This is a list: order does not matter here, we just loop over it.
IK_MARKERS = [
    "RIAS", "LIAS", "RIPS", "LIPS",
    "RFTC", "RTH", "RFLE", "RTTC", "RSK", "RFAX", "RFAL", "RFCC", "RFM1", "RFM5", "RDP1",
    "LFTC", "LTH", "LFLE", "LTTC", "LSK", "LFAX", "LFAL", "LFCC", "LFM1", "LFM5", "LDP1",
    "CV7", "SJN", "SXS", "TV2", "TV7",
]
# A set (curly braces): we only ever ask "is this marker name in BONY?", and sets
# answer that instantly.
BONY = {"RIAS", "LIAS", "RIPS", "LIPS", "CV7",
        "RFTC", "RFLE", "RTTC", "RFAX", "RFAL", "RFCC",
        "LFTC", "LFLE", "LTTC", "LFAX", "LFAL", "LFCC"}
HIGH_WEIGHT, LOW_WEIGHT = 5.0, 1.0          # assign two names on one line

# Coordinates we freeze during IK: their markers are not tracked here, so leaving
# them free would let those segments drift.  Also a set (fast membership test).
LOCK_COORDS = {
    "subtalar_angle_r", "mtp_angle_r", "subtalar_angle_l", "mtp_angle_l",
    "arm_flex_r", "arm_add_r", "arm_rot_r", "elbow_flex_r", "pro_sup_r", "wrist_flex_r", "wrist_dev_r",
    "arm_flex_l", "arm_add_l", "arm_rot_l", "elbow_flex_l", "pro_sup_l", "wrist_flex_l", "wrist_dev_l",
    "neck_bending", "neck_rotation", "neck_extension",
}


# --------------------------------------------------------------------------- #
# small helpers for the typed OpenSim "Array" containers
#
# OpenSim methods do not accept a plain Python list -- they want their own array
# types (osim.ArrayStr, osim.ArrayDouble).  These two helpers build one from
# whatever arguments you pass:  _array_str("a", "b")  ->  an ArrayStr of {a, b}.
# --------------------------------------------------------------------------- #

def _array_str(*values) -> osim.ArrayStr:
    a = osim.ArrayStr()                     # start empty
    for v in values:                       # `values` is a tuple of everything passed in
        a.append(v)                        # add them one at a time
    return a


def _array_dbl(*values) -> osim.ArrayDouble:
    a = osim.ArrayDouble()
    for v in values:
        a.append(float(v))                 # force each value to a floating-point number
    return a


def _fill_task_set(task_set, marker_names, weight=1.0):
    """Append one IKMarkerTask per name into an existing IKTaskSet.

    `weight` has a default of 1.0, so callers can leave it out.
    """
    for name in marker_names:              # loop over the list of marker names
        task = osim.IKMarkerTask()         # make a new "track this marker" object
        task.setName(name)                 # which marker
        task.setApply(True)                # yes, use it
        task.setWeight(weight)             # how much to trust it
        task_set.cloneAndAppend(task)      # add a copy into the set
    return task_set


# --------------------------------------------------------------------------- #
# 1.  C3D  ->  .trc  (markers only, rotated into the OpenSim frame)
# --------------------------------------------------------------------------- #

def c3d_to_trc(c3d_path, trc_path):
    """Read a .c3d, rotate every marker lab -> OpenSim, write a .trc.

    Returns two things (a tuple): the time-stamp array, and the list of marker
    names actually present in the file.
    """
    markers, time = read_c3d_markers(c3d_path)     # markers is a dict {name: (N,3) array}
    # dict comprehension: rebuild the dict with every trajectory rotated.
    # `.items()` gives (name, array) pairs to loop over.
    markers = {name: to_opensim_frame(xyz) for name, xyz in markers.items()}
    write_trc(trc_path, time, markers)
    return time, list(markers)                     # list(a_dict) -> its keys, i.e. the names


# --------------------------------------------------------------------------- #
# 2.  Scale the generic model to subject AB03
# --------------------------------------------------------------------------- #

def _measurement(name, marker_pairs, bodies, axes=("X", "Y", "Z")):
    """One ScaleTool 'measurement': segment scale = (mean subject marker-pair
    distance) / (mean model marker-pair distance), applied to `bodies`.

    `marker_pairs` is a list of 2-tuples like [("RFTC", "RFLE"), ...];
    `bodies` is a list of body names; `axes` says which directions to scale.
    """
    m = osim.Measurement()
    m.setName(name)
    m.setApply(True)
    for a, b in marker_pairs:              # unpack each pair into two names a, b
        m.getMarkerPairSet().cloneAndAppend(osim.MarkerPair(a, b))
    for body in bodies:
        bs = osim.BodyScale()
        bs.setName(body)
        bs.setAxisNames(_array_str(*axes))   # `*axes` spreads the tuple as separate args
        m.getBodyScaleSet().cloneAndAppend(bs)
    return m


# Which marker pair sizes which segment.  Lower-limb + trunk are what matter for a
# knee angle; the arms/head are scaled loosely just so the model is not grotesque.
# This is a list built by calling _measurement(...) eight times.
MEASUREMENTS = [
    _measurement("pelvis", [("RIAS", "LIAS"), ("RFTC", "LFTC")], ["pelvis"]),
    _measurement("torso", [("RIPS", "CV7"), ("LIPS", "CV7")], ["torso"]),
    _measurement("femur", [("RFTC", "RFLE"), ("LFTC", "LFLE")], ["femur_r", "femur_l"]),
    _measurement("tibia", [("RFAX", "RFAL"), ("LFAX", "LFAL")], ["tibia_r", "tibia_l"]),
    _measurement("foot", [("RFCC", "RDP1"), ("LFCC", "LDP1")],
                 ["talus_r", "calcn_r", "toes_r", "talus_l", "calcn_l", "toes_l"]),
    _measurement("humerus", [("RCAJ", "RHLE"), ("LCAJ", "LHLE")], ["humerus_r", "humerus_l"]),
    _measurement("forearm", [("RHLE", "RRSP"), ("LHLE", "LRSP")],
                 ["ulna_r", "radius_r", "hand_r", "ulna_l", "radius_l", "hand_l"]),
    _measurement("head", [("RAH", "LPH"), ("LAH", "RPH")], ["head"]),
]


def scale_model(static_trc, static_time, out_osim):
    """Fit data/Model/Biomech57.osim to subject AB03 using the static trial.

    A ScaleTool has three parts; we configure each in turn, then call .run().
    """
    # A short, still window in the middle of the A-pose recording:
    # np.quantile(x, 0.3) is the value 30% of the way through the sorted times.
    t0, t1 = float(np.quantile(static_time, 0.3)), float(np.quantile(static_time, 0.7))

    tool = osim.ScaleTool()
    tool.setName("AB03")
    tool.setSubjectMass(SUBJECT_MASS_KG)   # the scaled model will end up with this mass

    # --- part 1: where the generic model + marker set come from ---
    gmm = tool.getGenericModelMaker()
    gmm.setModelFileName(str(DATA / "Model" / "Biomech57.osim"))   # str(): Path -> string
    gmm.setMarkerSetFileName(str(DATA / "Model" / "marker_set.xml"))

    # --- part 2: how to resize each body (measurement-based) ---
    scaler = tool.getModelScaler()
    scaler.setApply(True)
    scaler.setScalingOrder(_array_str("measurements"))
    mset = osim.MeasurementSet()
    for m in MEASUREMENTS:                 # copy our eight measurements into the set
        mset.cloneAndAppend(m)
    scaler.setMeasurementSet(mset)
    scaler.setMarkerFileName(str(static_trc))
    scaler.setTimeRange(_array_dbl(t0, t1))
    scaler.setPreserveMassDist(True)       # keep each segment's fraction of the total mass
    scaler.setOutputModelFileName(str(out_osim))

    # --- part 3: nudge the model markers onto the subject's static pose ---
    placer = tool.getMarkerPlacer()
    placer.setApply(True)
    placer.setStaticPoseFileName(str(static_trc))
    placer.setTimeRange(_array_dbl(t0, t1))
    # getIKTaskSet() hands back the (empty) task set to fill in place.
    _fill_task_set(placer.getIKTaskSet(), IK_MARKERS, weight=1.0)
    placer.setMoveModelMarkers(True)
    placer.setOutputModelFileName(str(out_osim))

    tool.run()                            # <-- writes the scaled .osim to disk
    return out_osim


# --------------------------------------------------------------------------- #
# 3.  Trim the scaled model for inverse kinematics
# --------------------------------------------------------------------------- #

def trim_for_ik(scaled_osim, out_osim):
    """Drop muscles/actuators (IK never uses forces) and lock the upper body."""
    model = osim.Model(str(scaled_osim))          # load the .osim file
    model.setName("AB03_ik")
    model.updForceSet().clearAndDestroy()         # upd... = editable; delete every muscle/actuator
    model.initSystem()                            # "I'm done editing for now, rebuild internals"

    coords = model.updCoordinateSet()
    # OpenSim collections are looped by index, not with `for c in coords`:
    for i in range(coords.getSize()):
        if coords.get(i).getName() in LOCK_COORDS:   # fast set lookup
            coords.get(i).set_locked(True)           # hold this joint angle fixed during IK

    model.finalizeConnections()                   # re-wire after the edits
    model.initSystem()
    model.printToXML(str(out_osim))               # save the trimmed model
    return out_osim


# --------------------------------------------------------------------------- #
# 4.  Run InverseKinematicsTool on the stride
# --------------------------------------------------------------------------- #

def build_task_set(marker_names):
    """Turn a list of marker names into an IKTaskSet (name + tracking weight each).

    An IKMarkerTask says "track this marker, with this weight in the least-squares
    cost".  Higher weight = the solver tries harder to sit on that marker.
    """
    task_set = osim.IKTaskSet()
    for name in marker_names:
        task = osim.IKMarkerTask()
        task.setName(name)
        task.setApply(True)
        # TODO: give bony landmarks (the set `BONY`, e.g. malleoli, epicondyles,
        #       ASIS/PSIS, heel) the weight HIGH_WEIGHT and every other marker
        #       (the skin-mounted wands RTH/RSK/...) the weight LOW_WEIGHT.
        #       A one-line choice in Python:  A if condition else B
        task.setWeight(1.0)                       # <-- replace
        task_set.cloneAndAppend(task)             # add a copy into the set
    return task_set


def run_ik(ik_osim, stride_trc, stride_markers, out_mot):
    """Solve IK over the whole stride; return the parsed marker-error table."""
    # list comprehension: keep only the markers we want AND that exist in the trial
    # (the medial knee/ankle markers are missing from movement trials).
    present = [m for m in IK_MARKERS if m in stride_markers]

    table = osim.TimeSeriesTableVec3(str(stride_trc))
    t = np.array(table.getIndependentColumn())    # the trial's time stamps as a NumPy array

    model = osim.Model(str(ik_osim))
    model.initSystem()

    tool = osim.InverseKinematicsTool()
    tool.setModel(model)
    tool.set_marker_file(str(stride_trc))
    tool.set_IKTaskSet(build_task_set(present))
    # TODO: solve the whole trial -- set the start and end time from `t`
    #       (the .trc time-stamp array).  The first stamp is t[0], the last is
    #       t[-1] (negative index counts from the end).  Methods:
    #       tool.setStartTime(float(...)),  tool.setEndTime(float(...)).
    tool.setStartTime(0.0)                        # <-- replace
    tool.setEndTime(0.0)                          # <-- replace
    tool.set_report_errors(True)                  # also write a marker-error file
    tool.set_report_marker_locations(False)
    tool.setResultsDir(str(WORK))
    tool.set_output_motion_file(str(out_mot))
    # delete any error file left over from a previous run
    for stale in WORK.glob("*_marker_errors.sto"):
        stale.unlink()
    tool.run()                                    # <-- does the actual IK solve

    # glob() returns an iterator over matching files; next(...) takes the first one.
    err_file = next(WORK.glob("*_marker_errors.sto"))
    err_file = err_file.rename(WORK / "stride_1_ik_marker_errors.sto")   # tidy name
    return read_storage(err_file), model          # returns a tuple (table, model)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    WORK.mkdir(exist_ok=True)                     # make the _work/ folder (ok if it exists)
    # tell OpenSim where the bone-mesh files are, so it stops warning about them
    osim.ModelVisualizer.addDirToGeometrySearchPaths(str(DATA / "Model" / "Geometry"))

    print("=" * 70)                               # "=" * 70 -> a line of 70 '=' characters
    print("OpenSim constrained inverse kinematics  (subject AB03, stride_1)")
    print(f"  subject: {SUBJECT_MASS_KG} kg, {SUBJECT_HEIGHT_M} m")
    print("=" * 70)

    print("1. C3D -> TRC ...")
    # the `_` catches the marker-name list we do not need for the static trial
    static_time, _ = c3d_to_trc(DATA / "static.c3d", WORK / "static.trc")
    stride_time, stride_markers = c3d_to_trc(DATA / "stride_1.c3d", WORK / "stride_1.trc")

    print("2. scaling the generic model to AB03 ...")
    scaled = scale_model(WORK / "static.trc", static_time, WORK / "AB03_scaled.osim")

    print("3. trimming the scaled model for IK ...")
    ik_osim = trim_for_ik(scaled, WORK / "AB03_ik.osim")

    print("4. running InverseKinematicsTool over the stride ...")
    # `_model` is returned but not used here; the leading _ marks that on purpose.
    err, _model = run_ik(ik_osim, WORK / "stride_1.trc", stride_markers,
                         WORK / "stride_1_ik.mot")
    rms = err["marker_error_RMS"] * 100.0         # a whole column, metres -> centimetres
    mx = err["marker_error_max"] * 100.0
    # .mean() / .std() / .max() collapse the array to one number
    print(f"   marker error: RMS {rms.mean():.2f} +- {rms.std():.2f} cm "
          f"(worst frame {rms.max():.2f} cm, worst marker {mx.max():.2f} cm)")

    ik = read_storage(WORK / "stride_1_ik.mot")
    kr = ik["knee_angle_r"]                       # the right-knee-flexion column
    print(f"5. knee_angle_r: {kr.min():.1f}..{kr.max():.1f} deg over {ik.time[-1]-ik.time[0]:.2f} s")
    print(f"\n   wrote {WORK/'stride_1_ik.mot'}")
    print("   next:  python calculate_knee_angle.py")


# Run main() only when this file is executed directly (`python opensim_ik_reference.py`),
# not if it is imported from another script.
if __name__ == "__main__":
    main()

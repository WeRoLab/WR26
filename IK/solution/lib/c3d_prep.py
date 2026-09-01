"""Raw C3D  ->  OpenSim-ready inputs.

Steps performed here (this is the "given" data-prep layer -- students start from
its output):

1. Read markers and ground-reaction data from the Motive C3D with
   ``opensim.C3DFileAdapter``.
2. Rotate everything from the **lab frame** (Motive: X = medio-lateral,
   Y = anterior-posterior, Z = up) into the **OpenSim frame**
   (X = anterior, Y = up, Z = right).  The rotation was reverse-engineered from
   the lab's own ``static.trc`` and is an exact -90 deg turn about X::

       [x, y, z]_opensim  =  [x_lab,  z_lab,  -y_lab]

3. Low-pass filter (zero-lag Butterworth).
4. Write ``<trial>.trc`` (markers), ``<trial>_grf.mot`` (ground reactions in the
   OpenSim column convention) and ``<trial>_extloads.xml`` (ExternalLoads).

Force plate 1 is under the **right** foot, plate 2 under the **left** foot
(plates 3-6 are unused).  Confirmed from the data and the lab's
``process_force_data.m``.

----------------------------------------------------------------------------
New to Python / NumPy?  Things you will see a lot below:
  * a NumPy array is a grid of numbers.  A marker trajectory is shape (N, 3):
    N rows (time frames), 3 columns (x, y, z).
  * `A @ B`     -- matrix multiplication (used to rotate points).
  * `arr[:, 0]` -- "every row, column 0"  (here: the x value at every frame).
  * `arr[mask]` where `mask` is a True/False array -- selects/edits only the
    rows where mask is True.
  * `{k: f(v) for k, v in d.items()}` -- a "dict comprehension": build a new
    dict by transforming every value of `d`.
  * a name starting with `_` (e.g. `_yaw_about_y`) is a private helper: only
    meant to be used inside this file.
----------------------------------------------------------------------------
"""

# `from __future__ import annotations` just makes the ": type" hints lazy; ignore it.
from __future__ import annotations

import numpy as np                            # arrays and math
import opensim as osim                        # the OpenSim Python library (reads C3D)
from scipy.signal import butter, filtfilt     # tools to build + apply a low-pass filter

# ------------------------------------------------------------------ constants --

# The fixed 3x3 rotation from the lab frame to the OpenSim frame.  Applied as
# `v_opensim = v_lab @ LAB_TO_OSIM`.  Multiplying [x, y, z] by this matrix gives
# [x, z, -y]  (a -90 deg turn about the X axis).
LAB_TO_OSIM = np.array([[1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0],
                        [0.0, 1.0, 0.0]])

PLATE_BODY = {1: "calcn_r", 2: "calcn_l"}   # force plate number -> model body it loads
FORCE_ACTIVE_THRESHOLD_N = 20.0             # a plate counts as "used" if |force| ever exceeds this (newtons)


# --------------------------------------------------------------------------- io


def _table_to_dict(table):
    """Convert an OpenSim results table into plain NumPy.

    OpenSim hands data back as a `TimeSeriesTableVec3` object.  We pull the
    numbers into a dict  {"RFCC": array_shape_(n_frames, 3), ...}  and a 1-D
    array of time stamps -- much easier to work with.
    """
    labels = list(table.getColumnLabels())        # column names, e.g. ["RFCC", ...] or ["f1","p1",...]
    n = table.getNumRows()                         # how many time frames
    time = np.array(table.getIndependentColumn(), float)   # the time-stamp column

    out = {}                                       # empty dict to fill in
    for j, lab in enumerate(labels):               # enumerate -> (0, "RFCC"), (1, "RIAS"), ...
        arr = np.empty((n, 3))                     # uninitialised space for n rows of (x, y, z)
        for i in range(n):                         # i = 0, 1, ..., n-1  (each frame)
            # getRowAtIndex(i).getElt(0, j) is the 3-number vector (Vec3) at
            # frame i, column j.  .get(0/1/2) reads its x / y / z part.
            v = table.getRowAtIndex(i).getElt(0, j)
            arr[i] = (v.get(0), v.get(1), v.get(2))
        out[lab] = arr                             # save this column under its name
    return time, out                               # return two values (a tuple)


def read_c3d(path):
    """Open one .c3d file; return its markers and force data, still in the raw
    lab coordinate frame.  `path` is a file path (str or pathlib.Path).
    """
    # C3DFileAdapter is OpenSim's C3D reader.  We ask it to report each plate's
    # load at the "centre of pressure" (where the foot presses down).
    adapter = osim.C3DFileAdapter()
    adapter.setLocationForForceExpression(
        osim.C3DFileAdapter.ForceLocation_CenterOfPressure)
    tables = adapter.read(str(path))               # actually read the file

    # Split the result into markers and forces, each as (time, {label: array}).
    m_time, markers = _table_to_dict(adapter.getMarkersTable(tables))
    f_time, forces = _table_to_dict(adapter.getForcesTable(tables))

    # This dataset stores markers in metres, but check the file's "Units" tag and
    # convert from millimetres if that is what it says.
    scale = 1.0
    mt = adapter.getMarkersTable(tables)
    if mt.hasTableMetaDataKey("Units") and mt.getTableMetaDataAsString("Units").lower() in ("mm", "millimeter"):
        scale = 1e-3                                # 1 mm = 0.001 m
    if scale != 1.0:
        # rebuild the dict with every array multiplied by `scale`
        markers = {k: v * scale for k, v in markers.items()}

    # np.diff(m_time) = gaps between consecutive time stamps;
    # np.median(...)  = the typical gap;   1 / gap = samples per second (Hz).
    return dict(marker_time=m_time, markers=markers,
                force_time=f_time, forces=forces,
                marker_rate=1.0 / np.median(np.diff(m_time)),
                force_rate=1.0 / np.median(np.diff(f_time)))


# ---------------------------------------------------------------------- rotate


def rotate(v):
    """Rotate points/vectors from the lab frame into the OpenSim frame.

    `v` may be a single (3,) point or a whole (N, 3) array of points.  `@` is
    matrix multiplication: `v @ LAB_TO_OSIM` transforms every row of `v`, giving
    [x, z, -y].  `np.asarray(v, float)` makes sure `v` is a float array first.
    """
    return np.asarray(v, float) @ LAB_TO_OSIM


def _yaw_about_y(angle_rad):
    """Build the 3x3 matrix that rotates by `angle_rad` about the vertical (+Y)
    axis.  Used to spin a whole trial so the subject ends up facing +X.
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)    # assign two variables at once
    # rotation about +Y (up); the middle row [0, 1, 0] leaves the Y value alone
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])


def subject_yaw(markers_osim):
    """Which horizontal direction is the subject facing?  (radians)

    The capture volume is not aligned with the subject, so `s0` uses this to spin
    every trial until the subject faces +X.  Then the sagittal plane is just the
    OpenSim X-Y plane.

    The pelvis "forward" direction = (mid-point of the two FRONT hip markers)
    minus (mid-point of the two BACK hip markers), averaged over the trial.
    """
    mid_asis = 0.5 * (markers_osim["RIAS"] + markers_osim["LIAS"])   # front hip, midline
    mid_psis = 0.5 * (markers_osim["RIPS"] + markers_osim["LIPS"])   # back hip, midline
    # (mid_asis - mid_psis) is (N, 3): one forward vector per frame.
    # np.nanmean(..., axis=0) averages down the frames -> a single (3,) vector,
    # skipping frames where a marker was missing (NaN = "not a number").
    ant = np.nanmean(mid_asis - mid_psis, axis=0)
    # np.arctan2(y, x) = angle of the 2-D vector (x, y).  We use the horizontal
    # components: x = ant[0] (forward), z = ant[2] (right).
    return float(np.arctan2(ant[2], ant[0]))


# ---------------------------------------------------------------------- filter


def lowpass(x, cutoff_hz, fs, order=4):
    """Smooth a signal with a zero-lag Butterworth low-pass filter.

    x        : array; filtered along axis 0 (time).  Shape (N,) or (N, 3) or ...
    cutoff_hz: frequencies above this (Hz) are removed.  `None` -> do nothing.
    fs       : the sampling rate of `x`, in Hz.
    order    : filter steepness (default 4).

    Missing samples (NaN) are filled by straight-line interpolation first.
    """
    x = np.asarray(x, float)
    # Skip filtering when: no cutoff was given; the cutoff is too high to mean
    # anything (fs/2, the "Nyquist frequency", is the highest one can represent);
    # or there are too few frames for the filter to run.
    if cutoff_hz is None or cutoff_hz >= 0.5 * fs or x.shape[0] < 3 * (order + 1):
        return x

    # butter(...) designs the filter and returns its coefficients b, a.
    b, a = butter(order, cutoff_hz / (fs / 2.0), btype="low")

    # Flatten every extra dimension into columns so we can filter one track at a
    # time.  reshape(N, -1): keep the time axis, merge the rest; -1 = "figure out
    # this size".  .copy() so we do not modify the caller's array.
    flat = x.reshape(x.shape[0], -1).copy()
    t = np.arange(flat.shape[0])                   # 0, 1, 2, ... frame numbers
    for c in range(flat.shape[1]):                 # c = each column
        col = flat[:, c]                           # this column (editable view)
        bad = np.isnan(col)                        # True where the value is missing
        if bad.all():                              # whole column missing:
            continue                               #   skip it
        if bad.any():                              # some values missing:
            # np.interp(want_at, known_x, known_y): fill gaps by drawing a
            # straight line between the good samples on either side.
            # `~bad` means "not bad", i.e. the good frames.
            col[bad] = np.interp(t[bad], t[~bad], col[~bad])
        # filtfilt runs the filter forwards then backwards -> no time lag.
        flat[:, c] = filtfilt(b, a, col)
    return flat.reshape(x.shape)                   # restore the original shape


# ------------------------------------------------------------------- assemble


def active_plates(forces):
    """Which force plates actually carried load in this trial?  e.g. returns [1, 2].

    `forces` is the dict from read_c3d: {"f1": (N,3), "p1": ..., "m1": ..., "f2": ...}.
    """
    out = []
    for p in (1, 2, 3, 4, 5, 6):                   # check every possible plate number
        f = forces.get(f"f{p}")                    # dict.get returns None if "f3" is absent
        # np.linalg.norm(f, axis=1) = the force magnitude |(Fx,Fy,Fz)| at each
        # frame;  np.nanmax(...) = the biggest such magnitude over the whole trial.
        if f is not None and np.nanmax(np.linalg.norm(f, axis=1)) > FORCE_ACTIVE_THRESHOLD_N:
            out.append(p)                          # this plate was loaded -> keep it
    return out


def prepare_trial(c3d_path, marker_cutoff_hz=6.0, force_cutoff_hz=15.0, align_yaw=True):
    """Main entry point: read one C3D, rotate it into the OpenSim frame,
    optionally spin it so the subject faces +X, low-pass filter, and pack
    everything the writer functions need into one dict.

    The `= 6.0` etc. are DEFAULT argument values; a caller can override them by
    name, e.g. `prepare_trial(path, align_yaw=False)`.
    """
    raw = read_c3d(c3d_path)                       # raw lab-frame data

    # --- markers: apply the fixed lab -> OpenSim rotation to every marker -----
    # dict comprehension: for each (name, array) pair, store rotate(array).
    markers = {name: rotate(xyz) for name, xyz in raw["markers"].items()}

    # --- yaw: spin the whole trial about vertical so the subject faces +X ----
    yaw = 0.0
    # only if this trial actually contains all four pelvis markers
    if align_yaw and all(m in markers for m in ("RIAS", "LIAS", "RIPS", "LIPS")):
        yaw = subject_yaw(markers)                 # the current heading
        Ry = _yaw_about_y(yaw)                     # matrix that removes it
        # `xyz @ Ry.T` rotates each row of xyz by Ry.  `.T` is transpose: Ry is
        # written for column vectors, but our points are stored as rows.
        markers = {name: xyz @ Ry.T for name, xyz in markers.items()}

    # shift time so the first frame is t = 0
    m_time = raw["marker_time"] - raw["marker_time"][0]
    # smooth every marker track (replace each value in the dict with its filtered version)
    for name in markers:
        markers[name] = lowpass(markers[name], marker_cutoff_hz, raw["marker_rate"])

    # --- ground reactions: same rotations, on force / point / moment --------
    f_time = raw["force_time"] - raw["force_time"][0]
    Ry = _yaw_about_y(yaw)
    plates = {}                                    # will become {1: {...}, 2: {...}}
    for p in active_plates(raw["forces"]):
        # rotate(...) does lab -> OpenSim;  `@ Ry.T` then applies the yaw spin.
        F = rotate(lowpass(raw["forces"][f"f{p}"], force_cutoff_hz, raw["force_rate"])) @ Ry.T
        P = rotate(raw["forces"][f"p{p}"]) @ Ry.T          # centre of pressure: a point -> rotate only (do not filter)
        M = rotate(lowpass(raw["forces"][f"m{p}"], force_cutoff_hz, raw["force_rate"])) @ Ry.T

        # When the foot is off the plate the readings are just noise -- zero them.
        # `unloaded` is a True/False array (one per frame); `P[unloaded] = 0.0`
        # sets only those rows to zero.
        unloaded = np.linalg.norm(F, axis=1) < FORCE_ACTIVE_THRESHOLD_N
        P[unloaded] = 0.0
        M[unloaded] = 0.0
        F[unloaded] = 0.0
        # PLATE_BODY.get(p, f"plate{p}") = the body name for plate p, or a fallback.
        plates[p] = dict(force=F, point=P, moment=M, body=PLATE_BODY.get(p, f"plate{p}"))

    # Package the results.  The `trial=` line turns a path like
    #   "C:\...\sit_to_stand_1.c3d"  into  "sit_to_stand_1":
    #   replace "\" with "/", split on "/" and take the last piece ([-1]),
    #   then drop the ".c3d" extension with rsplit(".", 1)[0].
    return dict(
        trial=str(c3d_path).replace("\\", "/").split("/")[-1].rsplit(".", 1)[0],
        marker_time=m_time, markers=markers, marker_rate=raw["marker_rate"],
        force_time=f_time, plates=plates, force_rate=raw["force_rate"],
    )


# --------------------------------------------------------------------- writers


def write_trc(prep, path):
    """Save the markers to a .trc file (the format OpenSim's IK tool reads)."""
    # A LOCAL import: only loaded when this function runs.  `as _w` renames it so
    # it does not collide with THIS function (also called write_trc).
    from .io import write_trc as _w
    _w(path, prep["marker_time"], prep["markers"], units="m")


def write_grf_mot(prep, path):
    """Save the ground-reaction forces to a .mot file, using the exact column
    names OpenSim expects (ground_force_1_vx, ..._px, ground_moment_1_mx, ...).
    """
    from .io import write_mot as _w
    cols = {}                                      # {column_name: 1-D array}
    for p, d in prep["plates"].items():            # p = plate number, d = its data dict
        F, P, M = d["force"], d["point"], d["moment"]   # unpack three (N, 3) arrays
        # F[:, 0] is the x-component at every frame; [:, 1] = y; [:, 2] = z.
        cols[f"ground_force_{p}_vx"] = F[:, 0]
        cols[f"ground_force_{p}_vy"] = F[:, 1]
        cols[f"ground_force_{p}_vz"] = F[:, 2]
        cols[f"ground_force_{p}_px"] = P[:, 0]
        cols[f"ground_force_{p}_py"] = P[:, 1]
        cols[f"ground_force_{p}_pz"] = P[:, 2]
        cols[f"ground_moment_{p}_mx"] = M[:, 0]
        cols[f"ground_moment_{p}_my"] = M[:, 1]
        cols[f"ground_moment_{p}_mz"] = M[:, 2]
    if not cols:                                   # no active plates -> nothing to write
        return
    _w(path, prep["force_time"], cols, name=prep["trial"] + " ground reactions")


def write_external_loads_xml(prep, grf_mot_path, path):
    """Write the small XML that tells OpenSim which .mot columns act on which
    body (plate 1 -> right heel `calcn_r`, plate 2 -> left heel `calcn_l`).
    Used by inverse dynamics next week, not by inverse kinematics.
    """
    if not prep["plates"]:
        return
    loads = osim.ExternalLoads()                   # an OpenSim container object
    # store just the file name (no folder): the .mot is expected next to the .xml
    loads.setDataFileName(str(grf_mot_path).replace("\\", "/").split("/")[-1])
    for p, d in prep["plates"].items():
        ef = osim.ExternalForce()                  # one "external force" per plate
        ef.setName(f"plate{p}")
        ef.set_applied_to_body(d["body"])                       # e.g. "calcn_r"
        ef.set_force_expressed_in_body("ground")
        ef.set_point_expressed_in_body("ground")
        # the column-name PREFIX that holds this plate's force / point / torque
        ef.set_force_identifier(f"ground_force_{p}_v")
        ef.set_point_identifier(f"ground_force_{p}_p")
        ef.set_torque_identifier(f"ground_moment_{p}_m")
        loads.cloneAndAppend(ef)                   # add a copy to the container
    loads.printToXML(str(path))                    # save the XML file

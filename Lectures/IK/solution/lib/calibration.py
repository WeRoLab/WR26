"""Calibrate the barehand 4-segment sagittal model from the static trial.

Everything here works in the **sagittal plane** = the OpenSim X-Y plane after the
yaw alignment done in ``s0`` (X = anterior, Y = up).  A 3-D marker ``(x, y, z)``
is projected to ``(x, y)``.

Segment chain, grounded at the foot::

    AJC --shank--> KJC --thigh--> HJC --pelvis--> LJC --trunk-->
        (theta_ankle)   (theta_knee)   (theta_hip)   (theta_lumbar)

Per frame we *measure* the ankle joint centre and the foot angle; the four joint
angles are the unknowns solved in ``planar_ik``.

Joint centres
    ankle  = midpoint(RFAL, RTAM)         (lateral + medial malleolus)
    knee   = midpoint(RFLE, RFME)         (used only for calibration / init)
    hip    = Harrington (2007) regression from the pelvis markers
    lumbar = mid-PSIS + a small superior offset  (L5/S1 proxy)

What "calibration" produces (a dict), used by planar_ik:
    L_shank, L_thigh        segment lengths (m)
    phi_*0                  each segment's angle in the standing pose (rad)
    segments[seg]["markers"][name]  = each marker's fixed (x, y) position within
                                      its segment, measured in the static trial
    d_pelvis_lumbar        vector from the hip centre to the lumbar joint
    ankle_from_RFAL, knee_from_RFLE  small offsets, see the note near the bottom

----------------------------------------------------------------------------
Python / NumPy notes:
  * a 2-D "point" here is a length-2 NumPy array  np.array([x, y]).
  * `A @ B` is matrix multiplication;  `rot(phi) @ v` rotates the vector v.
  * `np.arctan2(y, x)` returns the angle of the vector (x, y).
  * `*thing` "splats" a sequence into separate arguments: f(*[a, b]) == f(a, b).
  * `{**d, "k": v}` makes a copy of dict d with key "k" set/added.
----------------------------------------------------------------------------
"""

# `from __future__ import annotations` makes ": type" hints lazy; ignore it.
from __future__ import annotations

import numpy as np

from .io import read_trc          # our own .trc reader, from lib/io.py

# Which markers "belong to" each segment (used as targets by the solver).
# Medial markers (RTAM, RFME, RFM2) exist only in the static trial -- they are
# removed before the movement trials, so they help calibrate but are never
# movement-trial targets.  This is a dict of lists.
SEGMENT_MARKERS = {
    "shank": ["RSK", "RTTC", "RFAX"],
    "thigh": ["RFLE", "RTH", "RFTC"],
    "pelvis": ["RIAS", "LIAS", "RIPS", "LIPS"],
    "trunk": ["CV7", "SJN", "SXS", "TV2", "TV7"],
}
# Lumbar joint (L5/S1) proxy: centre of the four pelvis markers, raised ~5 cm.
# Its height decides how trunk lean is split between the hip and the lumbar
# joint; 5 cm best matches the OpenSim model's `back` joint for this subject.
LUMBAR_SUPERIOR_OFFSET_M = 0.05


def _xy(a):
    """Keep only the first two components (x, y) -- i.e. project 3-D -> sagittal.

    `a[..., :2]` means "for any leading dimensions, take columns 0 and 1".
    Works on a single point (3,) -> (2,) or a whole trajectory (N, 3) -> (N, 2).
    """
    return np.asarray(a)[..., :2]


def rot(phi):
    """2-D rotation matrix for angle `phi` (radians).  `rot(phi) @ v` rotates v."""
    c, s = np.cos(phi), np.sin(phi)             # cosine and sine, assigned together
    return np.array([[c, -s],
                     [s,  c]])


def hip_centre_harrington(r_asis, l_asis, r_psis, l_psis):
    """Estimate the RIGHT hip-joint centre from the four pelvis markers, using the
    Harrington et al. (2007) regression, returned as a sagittal (x, y) point.

    Regression (in the pelvis's own frame: x forward, y up), in metres::

        x_hjc = -0.24 * depth - 0.0099        (behind mid-ASIS)
        y_hjc = -0.30 * width - 0.0109        (below mid-ASIS)

    width = distance between the two ASIS markers;
    depth = distance from mid-ASIS to mid-PSIS.
    """
    # `map(f, seq)` applies f to each item; here it force-converts all four to
    # NumPy arrays in one line.
    r_asis, l_asis, r_psis, l_psis = map(np.asarray, (r_asis, l_asis, r_psis, l_psis))
    mid_asis = 0.5 * (r_asis + l_asis)          # midpoint of the front markers
    mid_psis = 0.5 * (r_psis + l_psis)          # midpoint of the back markers
    width = np.linalg.norm(r_asis - l_asis)     # np.linalg.norm = length of a vector
    depth = np.linalg.norm(mid_asis - mid_psis)

    x_hjc = -0.24 * depth - 0.0099              # metres
    y_hjc = -0.30 * width - 0.0109

    # Build the pelvis's forward ("ex") and up ("ey") directions in the sagittal
    # plane, then place the hip centre relative to mid-ASIS.
    ex = mid_asis - mid_psis                    # points forward
    ex = _xy(ex) / np.linalg.norm(_xy(ex))      # divide by its length -> unit vector
    ey = np.array([-ex[1], ex[0]])             # rotate ex by 90 deg -> points up
    return _xy(mid_asis) + x_hjc * ex + y_hjc * ey


def joint_centres(m):
    """Given a dict `m` of sagittal marker points, return the four joint centres.

    Only used with the STATIC trial (it needs the medial markers RTAM, RFME).
    """
    ajc = 0.5 * (m["RFAL"] + m["RTAM"])                       # ankle  = midpoint of the malleoli
    kjc = 0.5 * (m["RFLE"] + m["RFME"])                       # knee   = midpoint of the epicondyles
    hjc = hip_centre_harrington(m["RIAS"], m["LIAS"], m["RIPS"], m["LIPS"])
    pelvis_centre = 0.25 * (m["RIAS"] + m["LIAS"] + m["RIPS"] + m["LIPS"])   # average of 4 markers
    # np.array([0.0, offset]) adds `offset` to the y (vertical) component only.
    ljc = pelvis_centre + np.array([0.0, LUMBAR_SUPERIOR_OFFSET_M])
    return dict(AJC=ajc, KJC=kjc, HJC=hjc, LJC=ljc)


def foot_angle(m):
    """Sagittal foot orientation: the angle of the heel -> forefoot direction (rad)."""
    # average of whichever forefoot markers are present this frame
    forefoot = np.mean([m[k] for k in ("RFM1", "RFM2", "RFM5") if k in m], axis=0)
    v = forefoot - m["RFCC"]                    # vector from heel marker to forefoot
    return float(np.arctan2(v[1], v[0]))       # arctan2(y, x) -> the vector's angle


def _mean_frames(trc, names, sl):
    """Average each named marker's (x, y) over the frames in slice `sl`.

    Dict comprehension: {name: average_position for each name}.
    np.nanmean ignores frames where the marker was missing (NaN).
    """
    return {n: np.nanmean(_xy(trc[n])[sl], axis=0) for n in names}


def calibrate(static_trc_path, frames=(50, 550)):
    """Build the calibration dict from a window of frames of the static trial.

    `frames=(50, 550)` -> use frames 50..549 (a couple of seconds of quiet stance).
    """
    trc = read_trc(static_trc_path)
    sl = slice(*frames)                         # slice(50, 550) -> the range 50:550

    # Collect every marker name we will need.
    #   SEGMENT_MARKERS.values() -> the four lists;  sum(lists, []) flattens them
    #   into one list;  set(...) removes duplicates.
    need = set(sum(SEGMENT_MARKERS.values(), []))
    need |= {"RFAL", "RTAM", "RFLE", "RFME", "RIAS", "LIAS", "RIPS", "LIPS",
             "RFCC", "RFM1", "RFM2", "RFM5"}    # `|=` adds these to the set
    # average position of each marker that is actually in the file
    m = _mean_frames(trc, [n for n in need if n in trc.data], sl)

    jc = joint_centres(m)                       # the four static joint centres

    # ---- each segment's absolute angle in the standing pose --------------- #
    # `(vec)[::-1]` reverses [dx, dy] to [dy, dx];  `*` splats it into
    # np.arctan2(dy, dx)  ->  the angle of the vector (dx, dy).
    phi_foot0 = foot_angle(m)
    phi_shank0 = np.arctan2(*(jc["KJC"] - jc["AJC"])[::-1])       # ankle -> knee direction
    phi_thigh0 = np.arctan2(*(jc["HJC"] - jc["KJC"])[::-1])       # knee -> hip direction
    mid_asis = 0.5 * (m["RIAS"] + m["LIAS"])
    mid_psis = 0.5 * (m["RIPS"] + m["LIPS"])
    phi_pelvis0 = np.arctan2(*(mid_asis - mid_psis)[::-1])        # pelvis forward direction
    phi_trunk0 = np.arctan2(*(m["CV7"] - jc["LJC"])[::-1])        # lumbar joint -> C7 direction

    # where each segment's local frame sits (origin) and how it is oriented (phi0)
    origins = {"shank": jc["AJC"], "thigh": jc["KJC"],
               "pelvis": jc["HJC"], "trunk": jc["LJC"]}
    phis = {"shank": phi_shank0, "thigh": phi_thigh0,
            "pelvis": phi_pelvis0, "trunk": phi_trunk0}

    # ---- record each marker's fixed position WITHIN its segment ----------- #
    segments = {}
    for seg, names in SEGMENT_MARKERS.items():          # seg = "shank", names = [...]
        Rinv = rot(-phis[seg])                          # matrix that "un-rotates" the segment
        local = {}
        for n in names:
            if n in m:
                # (marker - segment origin) is the marker in world axes;
                # Rinv @ (...) rotates it into the segment's own axes.
                local[n] = Rinv @ (m[n] - origins[seg])
        segments[seg] = dict(origin=seg, phi0=phis[seg], markers=local)

    # vector from the hip centre to the lumbar joint, in the pelvis's own axes
    d_pelvis_lumbar = rot(-phi_pelvis0) @ (jc["LJC"] - jc["HJC"])

    # Package everything into one dict (dict(key=value, ...) syntax).
    return dict(
        L_shank=float(np.linalg.norm(jc["KJC"] - jc["AJC"])),   # ankle-to-knee length
        L_thigh=float(np.linalg.norm(jc["HJC"] - jc["KJC"])),   # knee-to-hip length
        d_pelvis_lumbar=d_pelvis_lumbar,
        phi_foot0=phi_foot0, phi_shank0=float(phi_shank0), phi_thigh0=float(phi_thigh0),
        phi_pelvis0=float(phi_pelvis0), phi_trunk0=float(phi_trunk0),
        segments=segments,
        # .tolist() turns arrays into plain lists so this can be saved as json
        static_joint_centres={k: v.tolist() for k, v in jc.items()},
        # Small offset from the LATERAL marker to the true joint centre.  The
        # movement trials have no medial markers, so during movement we do
        # AJC = RFAL + ankle_from_RFAL.  It is small because the malleoli /
        # epicondyles differ mostly side-to-side, which is out of the sagittal plane.
        ankle_from_RFAL=(jc["AJC"] - m["RFAL"]),
        knee_from_RFLE=(jc["KJC"] - m["RFLE"]),
    )


def restrict_to(calib, available):
    """Return a copy of `calib` with any calibrated marker that a movement trial
    does not contain removed from every segment's marker list.

    `{**calib, "segments": {}}` copies calib but replaces "segments" with a fresh
    empty dict, which the loop then fills in.
    """
    avail = set(available)
    out = {**calib, "segments": {}}
    for seg, spec in calib["segments"].items():
        out["segments"][seg] = {**spec,
                                "markers": {n: p for n, p in spec["markers"].items()
                                            if n in avail}}
    return out

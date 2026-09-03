"""Unconstrained inverse kinematics: knee angle from anatomical frames.

Alizadeh et al. (2025) dataset, subject AB03, trial `stride_1`.

Fill in the four `TODO` blocks inside the frame loop.  The idea (same as the
MATLAB activity):

  1.  Build an anatomical reference frame for the femur and one for the tibia
      from three markers each.
  2.  Stack the frame's basis vectors as columns  ->  rotation matrix segment -> global.
  3.  R_tibia<-femur = R_global<-tibia^T @ R_global<-femur .
  4.  The sagittal knee angle is the rotation of that matrix about the frame's
      medio-lateral (3rd) axis:  atan2(R[1, 0], R[0, 0]) .

One thing is new, forced by the marker set:  the Biomech-57 set drops the medial
knee (`RFME`) and medial ankle (`RTAM`) markers after the static trial, so the
`stride_1` movement trial does not contain them.  Step 0 below reconstructs them
from the static trial with a rigid 3-marker cluster (a "technical frame") -- the
same rotation-matrix idea as the rest of the activity.

Run `opensim_ik_reference.py` first: this script overlays its `knee_angle_r`
(constrained IK) on the anatomical-frame result (unconstrained IK).

    python get_data.py                 (in Assignments/A2/)
    python opensim_ik_reference.py
    python calculate_knee_angle.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

from c3d_io import find_data_dir, read_c3d_markers, read_storage

HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE / "_work"
DATA = find_data_dir()

# Right-leg markers used to build the anatomical frames.
KNEE_LATERAL = "RFLE"     # lateral femoral epicondyle   (in every trial)
KNEE_MEDIAL = "RFME"      # medial femoral epicondyle     (static trial only -> reconstructed)
ANKLE_LATERAL = "RFAL"    # lateral malleolus             (in every trial)
ANKLE_MEDIAL = "RTAM"     # medial malleolus              (static trial only -> reconstructed)
TROCHANTER = "RFTC"       # greater trochanter            (in every trial)

# 3-marker clusters (all present in BOTH trials) used to carry the medial markers
# from the static pose into the moving trial.
THIGH_CLUSTER = ("RFTC", "RFLE", "RTH")     # (origin, toward, third)
SHANK_CLUSTER = ("RFAX", "RFAL", "RTTC")


# --------------------------------------------------------------------------- #
# Coordinate-frame helpers  (GIVEN -- read them, they are the same idea you use
# for the femur/tibia frames below)
# --------------------------------------------------------------------------- #

def technical_frame(origin, toward, third):
    """Build a right-handed orthonormal frame from three points.

    e1 points from `origin` toward `toward`;
    e3 is perpendicular to the plane of the three points;
    e2 completes the right-handed set.

    Each argument is either a single (3,) point or an (N, 3) array of points; the
    return is (origin, R) with R of shape (3, 3) or (N, 3, 3), the basis vectors
    stacked as COLUMNS.
    """
    origin = np.asarray(origin, float)
    e1 = np.asarray(toward, float) - origin
    e1 = e1 / np.linalg.norm(e1, axis=-1, keepdims=True)
    v = np.asarray(third, float) - origin
    e3 = np.cross(e1, v)
    e3 = e3 / np.linalg.norm(e3, axis=-1, keepdims=True)
    e2 = np.cross(e3, e1)
    R = np.stack([e1, e2, e3], axis=-1)        # columns = e1, e2, e3
    return origin, R


def reconstruct_marker(missing_name, cluster, static_markers, moving_markers,
                       static_window):
    """Carry a marker that exists only in the static trial into the moving trial.

    1.  In the static trial, build the cluster's technical frame and record where
        the missing marker sits *in that frame* (a constant local vector).
    2.  In every frame of the moving trial, rebuild the cluster frame and place
        the marker back at that same local position.
    """
    o_name, t_name, third_name = cluster

    # --- static: local position of the missing marker in the cluster frame ---
    sl = static_window
    o0, R0 = technical_frame(static_markers[o_name][sl].mean(0),
                             static_markers[t_name][sl].mean(0),
                             static_markers[third_name][sl].mean(0))
    p_local = R0.T @ (np.nanmean(static_markers[missing_name][sl], axis=0) - o0)

    # --- moving: rebuild the frame every sample and restore the marker ---
    o, R = technical_frame(moving_markers[o_name],
                           moving_markers[t_name],
                           moving_markers[third_name])
    return o + np.einsum("nij,j->ni", R, p_local), p_local


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(show_frames=False):
    # ------------------------------------------------------------------ data --
    static, static_time = read_c3d_markers(DATA / "static.c3d")
    mv, mv_time = read_c3d_markers(DATA / "stride_1.c3d")
    mv_time = mv_time - mv_time[0]
    n = len(mv_time)

    # A still window in the middle of the A-pose recording (30%-70% of frames).
    static_window = slice(int(0.3 * len(static_time)), int(0.7 * len(static_time)))

    # -------------------------------------------- step 0: reconstruct medials --
    rfme, _ = reconstruct_marker(KNEE_MEDIAL, THIGH_CLUSTER, static, mv, static_window)
    rtam, _ = reconstruct_marker(ANKLE_MEDIAL, SHANK_CLUSTER, static, mv, static_window)

    # sanity check: how well does the same recipe rebuild the markers *in the
    # static trial*, where the true positions are known?
    for name, cluster in ((KNEE_MEDIAL, THIGH_CLUSTER), (ANKLE_MEDIAL, SHANK_CLUSTER)):
        check, _ = reconstruct_marker(name, cluster, static, static, static_window)
        resid = np.linalg.norm(check - static[name], axis=1)
        print(f"  static reconstruction residual for {name}: {np.nanmean(resid)*1e3:.1f} mm")

    lat_knee = mv[KNEE_LATERAL]
    lat_ankle = mv[ANKLE_LATERAL]
    troch = mv[TROCHANTER]

    # ----------------------------------------------- steps 1-4, frame by frame --
    # (a plain loop, like the MATLAB version, to keep every step visible)
    knee_angle = np.zeros(n)
    frames_femur, frames_tibia = [], []
    for i in range(n):
        # markers this frame (each is a length-3 array):
        #   lat_knee[i], rfme[i], troch[i]        (femur: lateral knee, medial knee, trochanter)
        #   lat_ankle[i], rtam[i]                 (tibia: lateral malleolus, medial malleolus)

        # ================================================================== #
        # TODO step 1a -- femur anatomical frame.
        #   Z: medio-lateral, from the medial knee (rfme[i]) to the lateral
        #      knee (lat_knee[i]).  Normalise it.
        #   Take a second vector down the thigh (rfme[i] - troch[i]); use cross
        #   products (np.cross) to get X (normalise it) then Y.
        x_femur = np.array([1.0, 0.0, 0.0])   # <-- replace
        y_femur = np.array([0.0, 1.0, 0.0])   # <-- replace
        z_femur = np.array([0.0, 0.0, 1.0])   # <-- replace

        # ================================================================== #
        # TODO step 1b -- tibia anatomical frame, same recipe, with
        #   lat_ankle[i], rtam[i], and a vector up the shank (rtam[i] - rfme[i]).
        x_tibia = np.array([1.0, 0.0, 0.0])   # <-- replace
        y_tibia = np.array([0.0, 1.0, 0.0])   # <-- replace
        z_tibia = np.array([0.0, 0.0, 1.0])   # <-- replace

        # ================================================================== #
        # TODO step 2 -- rotation matrices segment -> global.
        #   Stack each frame's basis vectors as the COLUMNS of a 3x3 matrix.
        #   Hint: np.column_stack([x, y, z])
        R_glo_fem = np.eye(3)   # <-- replace
        R_glo_tib = np.eye(3)   # <-- replace

        # ================================================================== #
        # TODO step 3 -- express the femur frame in the tibia frame.
        #   R_tib_fem = R_glo_tib^T @ R_glo_fem      (^T = .T,  matmul = @)
        R_tib_fem = np.eye(3)   # <-- replace

        # ================================================================== #
        # TODO step 4 -- sagittal knee angle from R_tib_fem.
        #   angle about the 3rd axis:  np.arctan2(R_tib_fem[1, 0], R_tib_fem[0, 0])
        knee_angle[i] = 0.0     # <-- replace

        if show_frames and i in (0, n // 2, n - 1):
            frames_femur.append((lat_knee[i].copy(), R_glo_fem.copy()))
            frames_tibia.append((lat_ankle[i].copy(), R_glo_tib.copy()))

    knee_deg = np.degrees(knee_angle)

    # ------------------------------------------------------ the OpenSim result --
    ik_mot = WORK / "stride_1_ik.mot"
    if not ik_mot.exists():
        sys.exit(f"{ik_mot} not found -- run  python opensim_ik_reference.py  first.")
    ik = read_storage(ik_mot)
    knee_osim = ik["knee_angle_r"]
    ik_time = ik.time - ik.time[0]

    # constant offset: the two methods put "zero flexion" a few degrees apart
    # (different landmark definitions).  Report the fit with and without it.
    resampled = np.interp(ik_time, mv_time, knee_deg)
    rmse_raw = float(np.sqrt(np.mean((resampled - knee_osim) ** 2)))
    offset = float(np.mean(knee_osim - resampled))
    rmse_off = float(np.sqrt(np.mean((resampled + offset - knee_osim) ** 2)))
    print(f"\n  anatomical-frame knee angle : {knee_deg.min():.1f} .. {knee_deg.max():.1f} deg")
    print(f"  OpenSim knee_angle_r        : {knee_osim.min():.1f} .. {knee_osim.max():.1f} deg")
    print(f"  RMSE vs OpenSim             : {rmse_raw:.1f} deg  "
          f"({rmse_off:.1f} deg after removing a {offset:+.1f} deg offset)")

    # --------------------------------------------------------------- the plot --
    import matplotlib
    if not show_frames:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(mv_time, knee_deg, "--o", ms=3, label="anatomical frames (unconstrained)")
    ax.plot(ik_time, knee_osim, "-", lw=2, label="OpenSim IK (constrained)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("right knee flexion [deg]")
    ax.set_title("stride_1 -- knee angle, two ways")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_png = HERE / "knee_angle_stride.png"
    fig.savefig(out_png, dpi=130)
    print(f"  wrote {out_png}")

    if show_frames:
        from frames_viz import draw_frame, new_axes
        ax3 = new_axes("anatomical frames at start / mid / end of the stride")
        for (o, R) in frames_femur:
            draw_frame(ax3, o, R, scale=0.08)
        for (o, R) in frames_tibia:
            draw_frame(ax3, o, R, scale=0.08)
        plt.show()


if __name__ == "__main__":
    main(show_frames="--show-frames" in sys.argv)

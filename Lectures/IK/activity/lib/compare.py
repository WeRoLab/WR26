"""Line up the barehand joint angles with the OpenSim joint angles and score them.

>>> TWO things to fill in -- search for "TODO". <<<
    1. the sign of each barehand angle relative to the OpenSim convention
    2. rmse_table()  -- the RMSE / peak-error numbers

Both tracks are expressed as **deviation from the standing pose** (so offsets from
different anatomical zero definitions drop out).  You still have to match the
*sign*: e.g. is barehand `theta_hip` positive or negative when the subject is
seated (deep hip flexion)?  Compare that with OpenSim `hip_flexion_r` at the same
instant and set the sign so both increase together.

OpenSim conventions:  ankle_angle_r  + = dorsiflexion
                      knee_angle_r   + = flexion
                      hip_flexion_r  + = flexion
                      lumbar_extension + = extension  (so - = forward lean)

----------------------------------------------------------------------------
Python / NumPy notes:
  * a list of tuples:  [("a", 1), ...] -- unpack in a loop:  for name, n in list:
  * np.degrees(x): radians -> degrees.   np.interp: resample onto new time points.
  * `arr - arr[k]` subtracts one value from the whole array (re-zeroing it).
----------------------------------------------------------------------------
"""

# `from __future__ import annotations` makes ": type" hints lazy; ignore it.
from __future__ import annotations

import numpy as np

# For each barehand angle: matching OpenSim coordinate, the SIGN needed to make
# them agree, and a label.  This is a list of 4 tuples.
# TODO: replace each `None` with +1.0 or -1.0  (see the docstring for how).
ANGLE_MAP = [
    ("theta_ankle", "ankle_angle_r", None, "ankle dorsiflexion"),
    ("theta_knee", "knee_angle_r", None, "knee flexion"),
    ("theta_hip", "hip_flexion_r", None, "hip flexion"),
    ("theta_lumbar", "lumbar_extension", None, "lumbar extension"),
]


def standing_index(time, motion):
    """Which frame counts as 'standing' for this motion (the zero point).

    `time` is the array of time stamps; the return value is a frame NUMBER.
    """
    if motion == "sit_to_stand":
        return len(time) - 1                        # ends standing -> last frame
    if motion == "stand_to_sit":
        return 0                                    # starts standing -> first frame
    return int(np.argmax(time >= time[0]))          # squat: first (upright) frame


def aligned_series(bh, ik, motion):
    """Put both tracks on the same footing: {label: (time, barehand_deg, opensim_deg)},
    every curve in degrees and measured relative to the standing pose.
    """
    tb, ti = bh["time"], ik.time
    ib = standing_index(tb, motion)                 # standing frame in the barehand data
    ii = standing_index(ti, motion)                 # standing frame in the OpenSim data
    out = {}
    for bkey, ikey, sign, label in ANGLE_MAP:       # unpack each 4-tuple
        if sign is None:
            raise NotImplementedError(f"set the sign for {bkey} in ANGLE_MAP")
        b = np.degrees(bh[bkey]) * sign             # radians -> deg, flip sign if needed
        b = b - b[ib]                               # re-zero at the standing frame
        o = ik[ikey] - ik[ikey][ii]                 # OpenSim angle (already deg), re-zeroed
        # OpenSim and barehand may use different time stamps; np.interp resamples
        # the OpenSim curve onto the barehand time stamps.
        out[label] = (tb, b, np.interp(tb, ti, o))
    return out


def rmse_table(bh, ik, motion):
    """dict label -> {rmse_deg, peak_err_deg, bh_rom_deg, os_rom_deg}.

    For each joint, with e = b - o over the trial:
        rmse_deg     = sqrt(mean(e**2))
        peak_err_deg = max(|e|)
        bh_rom_deg   = b.max() - b.min()      (range of motion, barehand)
        os_rom_deg   = o.max() - o.min()      (range of motion, OpenSim)
    """
    rows = {}
    for label, (t, b, o) in aligned_series(bh, ik, motion).items():
        # TODO: compute the four numbers above and store them, e.g.
        #   rows[label] = dict(rmse_deg=float(np.sqrt(np.mean((b - o) ** 2))),
        #                      peak_err_deg=...,  bh_rom_deg=...,  os_rom_deg=...)
        raise NotImplementedError("rmse_table")
    return rows


def plot_compare(bh, ik, motion, title, path):
    """Draw the four joints, barehand vs OpenSim, and save a PNG to `path`."""
    import matplotlib
    matplotlib.use("Agg")                           # render to a file, no window
    import matplotlib.pyplot as plt

    series = aligned_series(bh, ik, motion)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharex=True)   # 1 row of 4 plots
    for ax, (label, (t, b, o)) in zip(axes, series.items()):
        ax.plot(t, o, label="OpenSim", lw=2)
        ax.plot(t, b, label="barehand NLLS", lw=1.6, ls="--")
        ax.set_title(label)
        ax.set_xlabel("time (s)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("angle re. standing (deg)")
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)

"""Line up the barehand joint angles with the OpenSim joint angles and score them.

Both tracks are expressed as **deviation from the standing pose** (so offsets from
different anatomical zero definitions drop out), and the barehand sign is flipped
to the OpenSim convention where needed:

    barehand theta_ankle   ->  -1 * ankle_angle_r     (OpenSim: + dorsiflexion)
    barehand theta_knee    ->  +1 * knee_angle_r      (OpenSim: + flexion)
    barehand theta_hip     ->  -1 * hip_flexion_r     (OpenSim: + flexion)
    barehand theta_lumbar  ->  +1 * lumbar_extension  (OpenSim: + extension)

----------------------------------------------------------------------------
Python / NumPy notes:
  * a list of tuples:  [("a", 1), ("b", 2)]  -- unpack in a loop with
        for name, number in the_list:
  * np.degrees(x)  radians -> degrees;   np.interp  resamples onto new time points
  * `arr - arr[k]`  subtracts one value from the whole array (re-zeroing it)
----------------------------------------------------------------------------
"""

# `from __future__ import annotations` makes ": type" hints lazy; ignore it.
from __future__ import annotations

import numpy as np

# For each barehand angle: which OpenSim coordinate it matches, the sign needed
# to make them agree, and a human-readable label.  This is a list of 4 tuples.
ANGLE_MAP = [
    ("theta_ankle", "ankle_angle_r", -1.0, "ankle dorsiflexion"),
    ("theta_knee", "knee_angle_r", +1.0, "knee flexion"),
    ("theta_hip", "hip_flexion_r", -1.0, "hip flexion"),
    ("theta_lumbar", "lumbar_extension", +1.0, "lumbar extension"),
]


def standing_index(time, motion):
    """Which frame counts as 'standing' for this motion (used as the zero point).

    `time` is the array of time stamps; the return value is a frame NUMBER.
    """
    if motion == "sit_to_stand":
        return len(time) - 1                        # ends standing -> last frame
    if motion == "stand_to_sit":
        return 0                                    # starts standing -> first frame
    return int(np.argmax(time >= time[0]))          # squat: first (upright) frame


def aligned_series(bh, ik, motion):
    """Put both tracks on the same footing.

    Returns {label: (time, barehand_deg, opensim_deg)} where every curve is in
    degrees and measured relative to the standing pose.
    `bh` is the barehand result dict; `ik` is a Storage object (read from _ik.mot).
    """
    tb = bh["time"]
    ti = ik.time
    ib = standing_index(tb, motion)                 # standing frame in the barehand data
    ii = standing_index(ti, motion)                 # standing frame in the OpenSim data
    out = {}
    for bkey, ikey, sign, label in ANGLE_MAP:       # unpack each 4-tuple
        b = np.degrees(bh[bkey]) * sign             # radians -> deg, flip sign if needed
        b = b - b[ib]                               # re-zero at the standing frame
        o = ik[ikey] - ik[ikey][ii]                 # OpenSim angle (already deg), re-zeroed
        # OpenSim and barehand may be sampled at different times; np.interp
        # resamples the OpenSim curve onto the barehand time stamps.
        o_on_b = np.interp(tb, ti, o)
        out[label] = (tb, b, o_on_b)
    return out


def rmse_table(bh, ik, motion):
    """For each joint, how far apart are the two methods?

    rmse_deg     = root-mean-square of (barehand - opensim)
    peak_err_deg = largest absolute difference
    bh_rom_deg / os_rom_deg = range of motion (max - min) for each method
    """
    rows = {}
    for label, (t, b, o) in aligned_series(bh, ik, motion).items():
        err = b - o
        rows[label] = dict(rmse_deg=float(np.sqrt(np.mean(err ** 2))),
                           peak_err_deg=float(np.max(np.abs(err))),
                           bh_rom_deg=float(b.max() - b.min()),
                           os_rom_deg=float(o.max() - o.min()))
    return rows


def plot_compare(bh, ik, motion, title, path):
    """Draw the four joints, barehand vs OpenSim, and save a PNG to `path`."""
    import matplotlib
    matplotlib.use("Agg")                           # render to a file, no window
    import matplotlib.pyplot as plt

    series = aligned_series(bh, ik, motion)
    # one row of 4 sub-plots that share the x-axis
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharex=True)
    # zip pairs each sub-plot with one joint's (label, (t, b, o))
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

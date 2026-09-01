"""Movement-event detection for sit-to-stand / stand-to-sit / squat trials.

The trials are pre-trimmed to the transition, so we mostly need a robust
**seat-off** (or, for a squat, the lowest point) plus the start/end of motion.
Two independent cues are used and reported so students can compare them:

* kinematic  -- vertical position / velocity of the mid-pelvis marker cluster
* kinetic    -- total vertical ground reaction under the feet vs body weight

----------------------------------------------------------------------------
New to Python / NumPy?
  * a NumPy array acts like a whole column of numbers at once:
    `np.abs(vy)` takes the absolute value of every element.
  * `arr[:, 1]` = "every row, column 1" (here: the vertical value each frame).
  * `np.argmax(x)` / `np.argmin(x)` = the INDEX of the largest / smallest value.
  * functions named with a leading `_` are private helpers for this file only.
  * `events` is a dict (a labelled bag of values); `events["t_event"]` reads one.
----------------------------------------------------------------------------
"""

# `from __future__ import annotations` makes ": type" hints lazy; ignore it.
from __future__ import annotations

import numpy as np

# The four pelvis markers, averaged to get a "mid-pelvis" point.  A tuple (round
# brackets) is like a list but cannot be changed.
PELVIS_MARKERS = ("RIAS", "LIAS", "RIPS", "LIPS")


def _midpelvis_vertical(prep):
    """Vertical (OpenSim Y) height of the mid-pelvis, one value per frame."""
    # For each pelvis marker that exists, take column 1 (the vertical coordinate).
    ys = [prep["markers"][m][:, 1] for m in PELVIS_MARKERS if m in prep["markers"]]
    # np.mean(..., axis=0) averages the list of arrays element-by-element.
    return np.mean(ys, axis=0)


def _deriv(x, dt):
    """Time-derivative of a signal.  np.gradient uses centred differences."""
    return np.gradient(x, dt)


def detect_events(prep, mass_kg, motion=None):
    """Find seat-off / start / end times for one trial.

    Returns a dict of event times (seconds), plus (under the key "_signals")
    the curves that were used, so plot_events() can draw them.
    """
    t = prep["marker_time"]
    dt = float(np.median(np.diff(t)))            # time step between frames
    y = _midpelvis_vertical(prep)                # pelvis height vs time
    vy = _deriv(y, dt)                           # pelvis vertical velocity vs time

    # `motion or _guess_motion(...)`:  use `motion` if the caller gave one,
    # otherwise guess it from the trial name.
    motion = motion or _guess_motion(prep["trial"])

    # ---- kinematic start / end: pelvis moving faster than 5 % of its peak ----
    speed = np.abs(vy)
    thr = 0.05 * speed.max()
    moving = speed > thr                         # True/False array, one per frame
    idx = np.where(moving)[0]                    # the frame numbers where it is True
    # `A if cond else B`: first moving frame if there is one, else the very first.
    t_start = t[idx[0]] if idx.size else t[0]
    t_end = t[idx[-1]] if idx.size else t[-1]

    events = {"motion": motion, "t_start": float(t_start), "t_end": float(t_end)}

    # ---- the main event: seat-off (or the bottom of a squat) ----------------
    if motion == "squat":
        events["t_lowpoint"] = float(t[np.argmin(y)])       # frame of lowest pelvis
        events["t_event"] = events["t_lowpoint"]
    elif motion == "stand_to_sit":
        events["t_seatoff_kin"] = float(t[np.argmin(vy)])   # fastest downward motion
        events["t_event"] = events["t_seatoff_kin"]
    else:  # sit_to_stand
        events["t_seatoff_kin"] = float(t[np.argmax(vy)])   # fastest upward motion
        events["t_event"] = events["t_seatoff_kin"]

    # ---- a second, independent seat-off estimate from the force plates ------
    bw = mass_kg * 9.80665                       # body weight in newtons (m * g)
    if prep["plates"]:                           # only if this trial has force data
        tf = prep["force_time"]
        # sum(...) adds the vertical force from every active plate, frame by frame.
        fy = sum(d["force"][:, 1] for d in prep["plates"].values())
        # np.sign(fy - bw) is +1 above body weight, -1 below.  np.diff of that is
        # non-zero exactly where the signal crosses body weight.
        crossings = np.where(np.diff(np.sign(fy - bw)) != 0)[0]
        if crossings.size:
            k = crossings[0] if motion == "sit_to_stand" else crossings[-1]
            events["t_seatoff_kinetic"] = float(tf[k])
        events["bodyweight_N"] = float(bw)

    # keys starting with "_" hold bulky arrays; s0 drops them before saving json.
    events["_signals"] = dict(t=t, pelvis_y=y, pelvis_vy=vy)
    return events


def motion_window(events, prep, pad_s=0.3):
    """[t0, t1] bracketing the actual movement, clipped to the trial.

    sit/stand trials are short and returned almost whole; the long squat trials
    are cropped to the single squat rep plus padding.
    """
    t = prep["marker_time"]
    if events["motion"] == "squat":
        c = events.get("t_lowpoint", 0.5 * (t[0] + t[-1]))
        # max(t[0], ...) / min(t[-1], ...) keep the window inside the trial.
        return [max(t[0], c - 1.5 - pad_s), min(t[-1], c + 1.5 + pad_s)]
    return [max(t[0], events["t_start"] - pad_s), min(t[-1], events["t_end"] + pad_s)]


def _guess_motion(trial_name):
    """Work out the motion type from the file name, e.g. 'sit_to_stand_1'."""
    n = trial_name.lower()                       # lower-case so matching is safe
    if "stand_to_sit" in n:                      # `"x" in "text"` = substring test
        return "stand_to_sit"
    if "sit_to_stand" in n:
        return "sit_to_stand"
    if "squat" in n:
        return "squat"
    return "sit_to_stand"                        # fallback


def plot_events(prep, events, path=None):
    """Draw the pelvis-height / velocity / force curves with the events marked.

    If `path` is given, save a PNG there; otherwise just return the figure.
    """
    # Import matplotlib here (not at the top) so the rest of the module loads even
    # if plotting is unavailable.  "Agg" = draw to a file, no on-screen window.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = events["_signals"]
    # a figure with 2 stacked plots (2 rows, 1 column) that share the x-axis
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(s["t"], s["pelvis_y"], label="mid-pelvis height")
    ax[0].set_ylabel("height (m)")
    ax[1].plot(s["t"], s["pelvis_vy"], label="mid-pelvis vertical velocity")
    ax[1].axhline(0, color="k", lw=0.5)         # horizontal line at y = 0
    ax[1].set_ylabel("velocity (m/s)")
    ax[1].set_xlabel("time (s)")

    if prep["plates"]:
        tf = prep["force_time"]
        fy = sum(d["force"][:, 1] for d in prep["plates"].values())
        axf = ax[1].twinx()                     # a second y-axis on the same plot
        axf.plot(tf, fy, color="tab:green", alpha=0.5, label="total foot $F_y$")
        if "bodyweight_N" in events:
            axf.axhline(events["bodyweight_N"], color="tab:green", ls=":", lw=1)
        axf.set_ylabel("force (N)", color="tab:green")

    # draw a dashed vertical line for each event time that exists
    for key, color in [("t_event", "r"), ("t_seatoff_kinetic", "g"),
                       ("t_start", "0.6"), ("t_end", "0.6")]:
        if key in events:
            for a in ax:                        # on both sub-plots
                a.axvline(events[key], color=color, ls="--", lw=1)
    ax[0].set_title(f"{prep['trial']}  ({events['motion']})  "
                    f"event={events.get('t_event', float('nan')):.3f}s")
    for a in ax:
        a.legend(loc="best", fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
        plt.close(fig)                          # free the memory
    return fig

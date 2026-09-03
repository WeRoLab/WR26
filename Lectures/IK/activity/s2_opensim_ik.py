"""s2 -- inverse kinematics with the OpenSim tool   [STUDENT]

>>> TWO things to fill in -- search for "TODO". <<<
    1. build_task_set()  -- pick which markers to track and their weights
    2. the InverseKinematicsTool configuration in run_trial()

Part 1  build an IK marker task set (which markers, what weights)
Part 2  run InverseKinematicsTool on every trial over its movement window
Part 3  read back the per-frame marker errors and the joint angles, and plot

Inputs :  results/Model_ik.osim                (from s1)
          results/<trial>/<trial>.trc          (from s0)
Outputs:  results/<trial>/<trial>_ik.mot
          results/<trial>/<trial>_ik_marker_errors.sto
          results/<trial>/<trial>_ik.png

----------------------------------------------------------------------------
New to Python?
  * `pathlib.Path`: `p.name` (final piece), `p.exists()`, `p.glob("*.x")`,
    `p.iterdir()` (list a folder), `p.unlink()` (delete).
  * `json.loads(text)` turns JSON text into a Python dict.
  * `A if cond else B` is a one-line choice between two values.
----------------------------------------------------------------------------
"""

# `from __future__ import annotations` makes ": type" hints lazy; ignore it.
from __future__ import annotations

import json
import pathlib

import numpy as np
import opensim as osim               # the OpenSim Python library

from lib import model_tools          # our helper (has the IK_MARKERS list)
from lib.io import read_mot          # our .mot reader

# --- paths (see s0_prepare_data.py for the pattern) ---
HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
MODEL = RESULTS / "Model_ik.osim"

# Markers in HIGH_WEIGHT get weight `HIGH`; all others get `LOW`.  Bony landmarks
# (malleoli, epicondyles, ASIS/PSIS, heel, C7) are steadier than skin-mounted
# wands.  Experiment with these.
HIGH_WEIGHT = {"RFCC", "RFAL", "RFLE", "RIAS", "LIAS", "RIPS", "LIPS",
               "LFCC", "LFAL", "LFLE", "CV7"}
HIGH, LOW = 5.0, 1.0                  # assign two names at once

# which model coordinates to draw in the plot
PLOT_COORDS = ["hip_flexion_r", "knee_angle_r", "ankle_angle_r", "lumbar_extension"]


def build_task_set(markers):
    """Return an osim.IKTaskSet: one IKMarkerTask per name, with a weight.

    Skeleton:
        ts = osim.IKTaskSet()
        for m in markers:
            task = osim.IKMarkerTask()
            task.setName(m)                # which marker
            task.setApply(True)            # include it
            task.setWeight(<HIGH or LOW>)  # how much to trust it
            ts.cloneAndAppend(task)        # add a copy to the set
        return ts
    """
    # TODO
    raise NotImplementedError("build_task_set")


def run_trial(trial_dir):
    """Run OpenSim IK on one trial folder; print a summary; return the numbers."""
    trial = trial_dir.name                                # e.g. "sit_to_stand_1"
    trc = trial_dir / f"{trial}.trc"
    if not trc.exists():                                  # no marker file -> skip
        return None
    events = json.loads((trial_dir / f"{trial}_events.json").read_text())

    # ---- pick the time window to solve --------------------------------- #
    table = osim.TimeSeriesTableVec3(str(trc))
    tcol = np.array(table.getIndependentColumn())         # the trial's time stamps
    if events["motion"] == "squat":                       # long trial -> crop to the squat
        c = events.get("t_lowpoint", 0.5 * (tcol[0] + tcol[-1]))
        t0, t1 = max(tcol[0], c - 1.8), min(tcol[-1], c + 1.8)
    else:                                                 # short trial -> use all of it
        t0, t1 = tcol[0], tcol[-1]

    model = osim.Model(str(MODEL))
    out_mot = trial_dir / f"{trial}_ik.mot"

    ikt = osim.InverseKinematicsTool()
    # TODO: configure the tool, then it is run below.
    #   ikt.setModel(model)
    #   ikt.set_marker_file(str(trc))
    #   ikt.set_IKTaskSet(build_task_set(model_tools.IK_MARKERS))
    #   ikt.setStartTime(float(t0)); ikt.setEndTime(float(t1))
    #   ikt.set_report_errors(True)                       # also writes a marker-error file
    #   ikt.setResultsDir(str(trial_dir))
    #   ikt.set_output_motion_file(str(out_mot))
    raise NotImplementedError("run_trial: configure InverseKinematicsTool")

    # ---- provided from here down -------------------------------------- #
    for stale in trial_dir.glob("*_marker_errors.sto"):  # delete any old error file
        stale.unlink()
    ikt.run()                                             # <-- does the actual IK solve

    err_file = next(trial_dir.glob("*_marker_errors.sto"))
    err_file = err_file.rename(trial_dir / f"{trial}_ik_marker_errors.sto")   # tidy name
    err = read_mot(err_file)
    rms = err["marker_error_RMS"] * 100.0                 # metres -> centimetres
    mx = err["marker_error_max"] * 100.0
    print(f"  {trial:18s}  frames {len(err.time):4d}  "
          f"RMS {rms.mean():.2f}+-{rms.std():.2f} cm (max {rms.max():.2f})  "
          f"worst marker {mx.max():.2f} cm")
    plot_trial(trial, trial_dir, out_mot, err)
    return dict(trial=trial, rms_mean_cm=float(rms.mean()),
               rms_max_cm=float(rms.max()), max_marker_cm=float(mx.max()))


def plot_trial(trial, trial_dir, mot_path, err):
    """Two panels: the joint angles, and the marker error over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ik = read_mot(mot_path)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))        # 1 row, 2 columns of plots
    for c in PLOT_COORDS:
        ax[0].plot(ik.time, ik[c], label=c)
    ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("angle (deg)")
    ax[0].set_title(f"{trial}: OpenSim IK joint angles"); ax[0].legend(fontsize=8)

    ax[1].plot(err.time, err["marker_error_RMS"] * 100, label="RMS")
    ax[1].plot(err.time, err["marker_error_max"] * 100, label="max", alpha=0.6)
    ax[1].axhline(2.0, color="k", ls=":", lw=1, label="2 cm guideline")
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("marker error (cm)")
    ax[1].set_title("marker tracking error"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(trial_dir / f"{trial}_ik.png", dpi=110)
    plt.close(fig)


def main():
    print("=" * 70)
    print("s2  OpenSim InverseKinematicsTool")
    print("=" * 70)
    if not MODEL.exists():
        raise SystemExit("run s1_prepare_model.py first")   # stop with a message
    rows = []
    # every sub-folder of results/ except "static", in alphabetical order
    for trial_dir in sorted(p for p in RESULTS.iterdir() if p.is_dir() and p.name != "static"):
        r = run_trial(trial_dir)
        if r:                                             # skip trials that returned None
            rows.append(r)
    (RESULTS / "s2_ik_summary.json").write_text(json.dumps(rows, indent=2))
    worst = max(r["rms_mean_cm"] for r in rows)           # biggest mean RMS across trials
    print(f"\nworst trial mean marker RMS: {worst:.2f} cm")


if __name__ == "__main__":
    main()

"""s4 -- compare the two inverse-kinematics tracks   [STUDENT]

Part 1  per trial: line up barehand vs OpenSim joint angles, tabulate RMSE, plot
Part 2  repeatability across the three repetitions of each motion
Part 3  overall summary

Inputs :  results/<trial>/<trial>_ik.mot        (s2, OpenSim)
          results/<trial>/<trial>_planar.mot    (s3, barehand)
Outputs:  results/<trial>/<trial>_compare.png
          results/s4_compare_summary.json
          results/s4_repeatability.png

The alignment / scoring maths lives in lib/compare.py; this file loops over the
trials and draws the summary figures.
"""

# `from __future__ import annotations` makes ": type" hints lazy; ignore it.
from __future__ import annotations

import json
import pathlib

import numpy as np

from lib import compare
from lib.io import read_mot

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
MOTIONS = ["sit_to_stand", "stand_to_sit", "squat"]


def _load(trial_dir):
    """Read one trial's OpenSim and barehand results back in.

    Returns (trial_name, motion_type, barehand_dict, opensim_storage).
    The barehand .mot stores degrees; we convert to radians so it matches the
    shape lib/compare.py expects (same as s3's in-memory result).
    """
    trial = trial_dir.name
    ik = read_mot(trial_dir / f"{trial}_ik.mot")
    pl = read_mot(trial_dir / f"{trial}_planar.mot")
    bh = dict(time=pl.time,
              theta_ankle=np.radians(pl["ankle_angle"]),
              theta_knee=np.radians(pl["knee_angle"]),
              theta_hip=np.radians(pl["hip_angle"]),
              theta_lumbar=np.radians(pl["lumbar_angle"]))
    # read the "motion" field out of the events JSON
    motion = json.loads((trial_dir / f"{trial}_events.json").read_text())["motion"]
    return trial, motion, bh, ik


def per_trial():
    """For every trial: compute the RMSE table and save the comparison plot.

    Returns a dict  {trial_name: rmse_table}.
    """
    rows = {}
    for trial_dir in sorted(p for p in RESULTS.iterdir() if p.is_dir() and p.name != "static"):
        t = trial_dir.name
        # need both result files present
        if not ((trial_dir / f"{t}_planar.mot").exists()
                and (trial_dir / f"{t}_ik.mot").exists()):
            print(f"  {t:18s}  skipped (run s2 and s3 first)")
            continue
        trial, motion, bh, ik = _load(trial_dir)
        table = compare.rmse_table(bh, ik, motion)
        compare.plot_compare(bh, ik, motion, f"{trial}: barehand vs OpenSim",
                             trial_dir / f"{trial}_compare.png")
        rows[trial] = table
        # build a short "ankle 4.1, knee 5.2, ..." string.
        # lab.split()[0] takes the first word of "ankle dorsiflexion" -> "ankle".
        rmses = ", ".join(f"{lab.split()[0]} {v['rmse_deg']:.1f}" for lab, v in table.items())
        print(f"  {trial:18s}  RMSE(deg): {rmses}")
    return rows


def repeatability(rows):
    """Bar chart per motion: mean +/- SD of the RMSE across the 3 repetitions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    summary = {}
    for ax, motion in zip(axes, MOTIONS):          # one sub-plot per motion type
        trials = [t for t in rows if t.startswith(motion)]     # e.g. the 3 "sit_to_stand" reps
        # the joint labels, taken from any trial's table
        labels = list(next(iter(rows.values())).keys())
        x = np.arange(len(labels))                 # 0, 1, 2, 3 -> bar positions
        # nested list comprehension -> a (n_trials, n_joints) array of RMSE values
        vals = np.array([[rows[t][l]["rmse_deg"] for l in labels] for t in trials])
        # vals.mean(0) averages DOWN the trials -> one value per joint; .std(0) likewise
        ax.bar(x, vals.mean(0), yerr=vals.std(0), capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([l.split()[0] for l in labels], rotation=20)
        ax.set_title(motion)
        ax.set_ylabel("barehand vs OpenSim RMSE (deg)")
        # store mean and SD for each joint (dict comprehension over enumerate)
        summary[motion] = {l: dict(rmse_deg_mean=float(vals.mean(0)[i]),
                                   rmse_deg_sd=float(vals.std(0)[i]))
                           for i, l in enumerate(labels)}
    fig.tight_layout()
    fig.savefig(RESULTS / "s4_repeatability.png", dpi=110)
    plt.close(fig)
    return summary


def main():
    print("=" * 70)
    print("s4  barehand vs OpenSim comparison")
    print("=" * 70)
    rows = per_trial()
    summary = repeatability(rows)
    (RESULTS / "s4_compare_summary.json").write_text(
        json.dumps(dict(per_trial=rows, per_motion=summary), indent=2))

    print("\nper-motion RMSE (deg, mean over 3 reps):")
    for motion, tab in summary.items():
        print(f"  {motion}")
        for lab, v in tab.items():
            print(f"     {lab:20s} {v['rmse_deg_mean']:5.1f} +- {v['rmse_deg_sd']:.1f}")


if __name__ == "__main__":
    main()

"""s3 -- barehand inverse kinematics: 4-DOF planar model, nonlinear least squares
                                                                        [STUDENT]

Part 1  calibrate the 4-segment sagittal model from the static trial
Part 2  check the analytic Jacobian against finite differences (and SciPy)
Part 3  solve every movement trial, frame by frame, with Levenberg-Marquardt
Part 4  save joint angles + solver diagnostics, and plot

Inputs :  results/static/static.trc              (from s0)
          results/<trial>/<trial>.trc            (from s0)
Outputs:  results/<trial>/<trial>_planar.mot     joint angles (deg)
          results/<trial>/<trial>_planar.png
          results/planar_calibration.json

This script is mostly plumbing -- the real work is in lib/calibration.py and
lib/planar_ik.py.
"""

# `from __future__ import annotations` makes ": type" hints lazy; ignore it.
from __future__ import annotations

import json
import pathlib

import numpy as np

# `import X as Y` gives the module a short name.
from lib import calibration as cal
from lib import planar_ik as pik
from lib.io import read_trc, write_mot
from lib.calibration import foot_angle, _xy       # pull two functions out directly

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
STATIC_TRC = RESULTS / "static" / "static.trc"


# ---------------------------------------------------------------- Part 2 checks
def check_jacobian_and_scipy(calib):
    """Sanity-check the solver on one frame:
      (a) analytic Jacobian vs a finite-difference one  -> should match to ~1e-8
      (b) our Levenberg-Marquardt vs scipy.optimize.least_squares -> ~same answer
    """
    trc = read_trc(RESULTS / "sit_to_stand_1" / "sit_to_stand_1.trc")
    c = cal.restrict_to(calib, list(trc.data))
    use = [n for n in trc.data]
    i = trc.n_frames // 3                          # `//` is integer division
    # this frame's marker positions, dropping any that are NaN (missing)
    meas = {n: _xy(trc[n])[i] for n in use if not np.any(np.isnan(_xy(trc[n])[i]))}
    fa = foot_angle(meas)
    ajc = pik.ankle_centre(meas, c)
    q = pik.geometric_init(meas, c, fa)

    # `_, J = f(...)` throws away the first return value, keeps the second.
    _, J = pik.residual_and_jacobian(q, c, fa, ajc, meas, pik.DEFAULT_WEIGHTS)
    Jfd = pik.jacobian_fd(q, c, fa, ajc, meas, pik.DEFAULT_WEIGHTS)
    jac_err = float(np.max(np.abs(J - Jfd)))       # biggest disagreement anywhere

    scipy_err = None
    try:                                          # try this; if anything fails, skip it
        from scipy.optimize import least_squares
        # a lambda: the residual as a function of x only (other args fixed)
        fun = lambda x: pik.residual_and_jacobian(x, c, fa, ajc, meas,
                                                  pik.DEFAULT_WEIGHTS, want_jac=False)
        sp = least_squares(fun, q, method="lm", xtol=1e-12, ftol=1e-12)
        mine, _ = pik.solve_frame(q, c, fa, ajc, meas, pik.DEFAULT_WEIGHTS)
        scipy_err = float(np.max(np.abs(np.degrees(sp.x - mine))))   # difference in degrees
    except Exception as e:                         # `e` is the error object
        scipy_err = f"skipped ({e})"

    print(f"[check] analytic vs finite-difference Jacobian : {jac_err:.2e}")
    print(f"[check] our LM vs scipy.least_squares (deg)    : {scipy_err}")
    return jac_err


# ---------------------------------------------------------------- Part 3 solve
def run_trial(trial_dir, calib):
    """Solve one trial with the barehand model; save the angles and a plot."""
    trial = trial_dir.name
    trc_path = trial_dir / f"{trial}.trc"
    if not trc_path.exists():
        return None
    trc = read_trc(trc_path)
    events = json.loads((trial_dir / f"{trial}_events.json").read_text())

    # crop the long squat trials to a window around the lowest point
    if events["motion"] == "squat":
        c = events.get("t_lowpoint", 0.5 * (trc.time[0] + trc.time[-1]))
        t_range = (max(trc.time[0], c - 1.8), min(trc.time[-1], c + 1.8))
    else:
        t_range = None                            # None -> use every frame

    res = pik.solve_trial(trc, calib, t_range=t_range)   # <-- the actual solve

    # write the four angle time series to a .mot file (converted to degrees)
    write_mot(trial_dir / f"{trial}_planar.mot", res["time"], {
        "ankle_angle": np.degrees(res["theta_ankle"]),
        "knee_angle": np.degrees(res["theta_knee"]),
        "hip_angle": np.degrees(res["theta_hip"]),
        "lumbar_angle": np.degrees(res["theta_lumbar"]),
    }, name=f"{trial} barehand planar IK", in_degrees=True)

    print(f"  {trial:18s}  {len(res['time']):4d} frames  "
          f"RMS {res['rms_cm'].mean():.2f}+-{res['rms_cm'].std():.2f} cm "
          f"(max {res['rms_cm'].max():.2f})  iters {res['iters'].mean():.1f}")
    plot_trial(trial, trial_dir, res)
    return dict(trial=trial, rms_mean_cm=float(res["rms_cm"].mean()),
               rms_max_cm=float(res["rms_cm"].max()),
               iters_mean=float(res["iters"].mean()),
               grad_inf_max=float(res["grad_inf"].max()))


def plot_trial(trial, trial_dir, res):
    """Left panel: the four joint angles.  Right panel: solver diagnostics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = res["time"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    # loop over (dict key, legend label) pairs
    for k, lab in [("theta_ankle", "ankle"), ("theta_knee", "knee"),
                   ("theta_hip", "hip"), ("theta_lumbar", "lumbar")]:
        ax[0].plot(t, np.degrees(res[k]), label=lab)
    ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("angle re. standing (deg)")
    ax[0].set_title(f"{trial}: barehand joint angles"); ax[0].legend(fontsize=8)

    ax[1].plot(t, res["rms_cm"], label="marker RMS")
    ax[1].axhline(2.0, color="k", ls=":", lw=1, label="2 cm")
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("marker RMS (cm)")
    axi = ax[1].twinx()                            # second y-axis for the iteration count
    axi.plot(t, res["iters"], color="tab:red", alpha=0.4)
    axi.set_ylabel("LM iterations", color="tab:red")
    ax[1].set_title("solver diagnostics"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(trial_dir / f"{trial}_planar.png", dpi=110)
    plt.close(fig)


def main():
    print("=" * 70)
    print("s3  barehand 4-DOF planar nonlinear-least-squares IK")
    print("=" * 70)
    if not STATIC_TRC.exists():
        raise SystemExit("run s0_prepare_data.py first (need results/static/static.trc)")

    # ---- Part 1: calibrate from the standing trial ---------------------- #
    calib = cal.calibrate(STATIC_TRC)
    print(f"[calib] L_shank {calib['L_shank']*100:.1f} cm  L_thigh {calib['L_thigh']*100:.1f} cm  "
          f"d(HJC->LJC) {np.round(calib['d_pelvis_lumbar']*100, 1)} cm")
    # save the calibration numbers as JSON.  `v.tolist() if isinstance(v, np.ndarray)
    # else v` converts arrays to plain lists (JSON cannot store NumPy arrays).
    (RESULTS / "planar_calibration.json").write_text(json.dumps(
        {k: (v.tolist() if isinstance(v, np.ndarray) else v)
         for k, v in calib.items() if k != "segments"}, indent=2))

    # ---- Part 2: check the solver ------------------------------------- #
    check_jacobian_and_scipy(calib)

    # ---- Part 3: solve every movement trial -------------------------- #
    print("\nsolving trials:")
    rows = []
    for trial_dir in sorted(p for p in RESULTS.iterdir() if p.is_dir() and p.name != "static"):
        r = run_trial(trial_dir, calib)
        if r:
            rows.append(r)
    (RESULTS / "s3_planar_summary.json").write_text(json.dumps(rows, indent=2))
    print(f"\nworst mean marker RMS: {max(r['rms_mean_cm'] for r in rows):.2f} cm")


if __name__ == "__main__":
    main()

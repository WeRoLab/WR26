"""Truncated Fourier series fit of the ankle angle (least squares, cvxpy)
Python version of main_fourier.m.

Fits one gait cycle of the ankle angle (Winter, 1983) with a truncated
Fourier series of order j:

    ankle(t) ~ sum_{k=1..j} a_k*sin(k*w*t) + b_k*cos(k*w*t),  w = 2*pi/T

The model is linear in the coefficients p = [a_1 b_1 ... a_j b_j], so
stacking one row per sample gives ankle ~ R @ p, and the best fit is the
least-squares problem  min_p ||ankle - R @ p||_2. It is solved with cvxpy,
and checked against np.linalg.lstsq (the equivalent of MATLAB's "\").

Requires: numpy, scipy, matplotlib, cvxpy (see ../requirements.txt).
Output:   a figure overlaying the data and the Fourier fit.
Try:      change j. Low orders smooth the curve; high orders chase every
          sample (overfitting). Keep 2*j <= n so the problem stays
          well posed (more samples than unknowns).
Note:     there is no constant (cos(0*w*t)) column, so the fit cannot
          represent the mean ankle angle. Adding a column of ones is a
          good exercise.
"""

from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent      # folder of this script (finds the .mat)


def fourier_regressor(time, w, j):
    """Regressor matrix: row i = [sin(w t_i) cos(w t_i) ... sin(j w t_i) cos(j w t_i)]."""
    wkt = np.outer(time, w * np.arange(1, j + 1))   # (n, j): k*w*t_i
    R = np.empty((len(time), 2 * j))
    R[:, 0::2] = np.sin(wkt)                         # columns 1, 3, 5, ... in MATLAB
    R[:, 1::2] = np.cos(wkt)                         # columns 2, 4, 6, ...
    return R


def main():
    data = loadmat(HERE / "Winter1983_Ankle.mat")
    ankle = data["ankle"].ravel()           # one gait cycle
    time = data["time"].ravel()

    # Parameters of the fit
    w = 2 * np.pi / time[-1]                # Fundamental frequency (one cycle = one period)
    j = 8                                   # Order of the fit - Change it!
    R = fourier_regressor(time, w, j)       # Regressor matrix (R) -- ankle ~ R @ p

    # Get the coefficients that multiply the sinusoids to create the fit
    p = cp.Variable(2 * j)
    cp.Problem(cp.Minimize(cp.norm(ankle - R @ p, 2))).solve()
    ankleFour = R @ p.value

    # Same problem in closed form (MATLAB: g = R\ankle)
    g, *_ = np.linalg.lstsq(R, ankle)
    print(f"Max |cvxpy - lstsq| coefficient difference: {np.max(np.abs(p.value - g)):.2e}")

    # Compare results with a plot
    plt.figure()
    plt.title("Ankle Angle")
    plt.xlabel("Time [seconds]")
    plt.ylabel("Ankle Angle [deg] -- Positive Dorsiflexion")
    plt.plot(time, ankle, "--o", markerfacecolor="none", label="Dataset")
    plt.plot(time, ankleFour, label="Fourier")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

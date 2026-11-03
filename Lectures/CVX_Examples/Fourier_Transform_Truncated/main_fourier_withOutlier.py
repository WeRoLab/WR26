"""Truncated Fourier series fit with an outlier (cvxpy)
Python version of main_fourier_withOutlier.m.

Same fit as main_fourier.py, but one sample is corrupted by +90 deg (a
marker swap, say):

    minimize ||ankle - M @ x||_p

One bad sample among 1001 moves the L2 (least-squares) fit only ~1.4 deg.
A burst of bad samples is different: squared residuals punish large
errors heavily, so the L2 fit bends toward them.

Requires: numpy, scipy, matplotlib, cvxpy (see ../requirements.txt).
Output:   a figure overlaying the corrupted data and the fit.
Try:      corrupt a burst, ankle[99:150] += 90, and compare NORM = 2 with
          NORM = 1. The L2 fit shifts by up to ~66 deg; the L1 fit lets those
          samples keep large residuals and stays within ~5 deg of its
          clean-data fit. (NORM = "inf" is the opposite extreme.)
"""

from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from main_fourier import fourier_regressor   # same R as main_fourier.py

HERE = Path(__file__).resolve().parent
NORM = 2                                      # change to 1 for a robust (L1) fit


def main():
    data = loadmat(HERE / "Winter1983_Ankle.mat")
    ankle = data["ankle"].ravel()
    time = data["time"].ravel()

    # Introduce outlier (sample 100 in MATLAB's 1-based indexing)
    ankle[99] = ankle[99] + 90

    # Create regressor matrix
    w = 2 * np.pi / time[-1]
    j = 8                                     # Order of the fit
    M = fourier_regressor(time, w, j)

    # Fit the coefficients x
    x = cp.Variable(2 * j)
    cp.Problem(cp.Minimize(cp.norm(ankle - M @ x, NORM))).solve()

    plt.figure()
    plt.title(f"Ankle Angle (L{NORM} fit)")
    plt.xlabel("Time [seconds]")
    plt.ylabel("Ankle Angle [deg] - Pos. Dorsiflexion")
    plt.plot(time, ankle, "--o", markerfacecolor="none", label="Dataset")
    plt.plot(time, M @ x.value, label="Fourier")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

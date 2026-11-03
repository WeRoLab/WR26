"""Smoothing of signal to minimize jerk (cvxpy)
Python version of Smoothing.m.

Removes noise from a sampled signal by finding the curve x that stays
close to the measurement y while having small jerk (3rd derivative):

    minimize ||x - y||^2 + (alpha*dt^3)^2 * ||D3 @ x||^2

D3 is a finite-difference matrix, so this is a linear least-squares
problem: stack [I; alpha*dt^3*D3] @ x ~ [y; 0]. It is solved with a sparse
least-squares routine (the equivalent of MATLAB's "\") and with cvxpy, to
compare run times and check that both solvers agree.

Requires: numpy, scipy, matplotlib, cvxpy (see ../requirements.txt).
Output:   two figures (position, and position/velocity/acceleration of
          the original, noisy and smoothed signals) and printed timings.
Try:      change alpha (third argument of smooth_cvx). Too small keeps the
          noise; too large flattens the signal. The default 1e-4 is in the
          "too small" range (the result is almost the noisy signal);
          alpha = 1 to 10 visibly smooths it. Note how the noise is
          amplified in the velocity and acceleration of the noisy signal.
"""

import time as timer

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr


def central_difference(stencil, n, h_power):
    """n-4 x n matrix applying a 5-point stencil on samples k-2..k+2.

    Rows whose stencil would run off the ends are dropped (MATLAB: D(3:end-2, :)).
    """
    D = sp.diags(stencil, offsets=[-2, -1, 0, 1, 2], shape=(n, n), format="csr")
    return D[2:-2, :] / h_power


def smooth_cvx(sig, dt, alp):
    """Find a set of points that are close to sig but minimize jerk.

    Returns the smoothed signal and the 1st/2nd derivative matrices (for plotting).
    """
    n = len(sig)

    # First derivative - 4th order accuracy:  [1 -8 0 8 -1]/(12 dt)
    D1 = central_difference([1/12, -2/3, 0, 2/3, -1/12], n, dt)
    # Second derivative - 4th order accuracy: [-1 16 -30 16 -1]/(12 dt^2)
    D2 = central_difference([-1/12, 4/3, -5/2, 4/3, -1/12], n, dt**2)
    # Third derivative - 2nd order accuracy:  [-1 2 0 -2 1]/(2 dt^3)
    D3 = central_difference([-1/2, 1, 0, -1, 1/2], n, dt**3)

    # Minimize the norm of signal fit and jerk: || [I; alp*dt^3*D3] @ x - [sig; 0] ||
    m = D3.shape[0]
    A = sp.vstack([sp.eye(n), (alp * dt**3) * D3], format="csr")
    y = np.concatenate([sig, np.zeros(m)])

    # Solve with SciPy's sparse least squares (MATLAB: fil = A\y)
    tic = timer.perf_counter()
    fil = lsqr(A, y, atol=1e-12, btol=1e-12)[0]
    tSciPy = timer.perf_counter() - tic

    # Solve the same problem with cvxpy
    tic = timer.perf_counter()
    x = cp.Variable(n)
    cp.Problem(cp.Minimize(cp.sum_squares(A @ x - y))).solve()
    fil2 = x.value
    tCVX = timer.perf_counter() - tic

    print(f"Elapsed time SciPy:\t{tSciPy:3.4f} [s]")
    print(f"Elapsed time cvxpy:\t{tCVX:3.4f} [s]")
    # Both solvers should find the same minimizer
    print(f"Max |SciPy - cvxpy|:\t{np.max(np.abs(fil - fil2)):3.2e}")

    return fil, D1, D2


def main():
    time = np.linspace(0, 2, 1000)
    dt = time[1] - time[0]
    rng = np.random.default_rng(0)
    sig = np.sin(2 * np.pi * time)                              # Signal
    noi = sig + (rng.random(len(sig)) - 0.5) * 0.1              # Noisy signal (uniform, +/-0.05)

    # Smooth the signal (calculate alpha (third parameter) by trial and error)
    smo, D1, D2 = smooth_cvx(noi, dt, 10e-5)                   # 10e-5 = 1e-4

    # Plot and compare
    plt.figure()
    plt.title("Reference signals")
    plt.xlabel("Time"), plt.ylabel("Position")
    plt.plot(time, sig, label="Original signal")
    plt.plot(time, noi, "--", label="Noisy signal")
    plt.plot(time, smo, "--o", markerfacecolor="none", label="Smoothed signal")
    plt.grid(True)
    plt.legend()

    # D1/D2 drop the first and last two samples, hence time[2:-2]
    fig, axs = plt.subplots(3, 1, sharex=True)
    fig.suptitle("Signal smoothing")
    for ax, D, label in zip(axs, [None, D1, D2], ["Position", "Velocity", "Acceleration"]):
        t = time if D is None else time[2:-2]
        f = (lambda s: s) if D is None else (lambda s, D=D: D @ s)
        ax.plot(t, f(sig), label="Original signal")
        ax.plot(t, f(noi), "--", label="Noisy signal")
        ax.plot(t, f(smo), "--o", markerfacecolor="none", label="Smoothed signal")
        ax.set_ylabel(label)
        ax.grid(True)
    axs[0].legend()
    axs[-1].set_xlabel("Time")
    plt.show()


if __name__ == "__main__":
    main()

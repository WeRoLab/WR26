"""Generate a smooth 6th-order polynomial trajectory (position, velocity, acceleration).

Assignment 5 asks you to select a DC motor for an elbow prosthesis that lifts
the forearm from 0 deg to 90 deg and back to 0 deg in 3 s. To size the motor
you need the elbow position, velocity, and acceleration over time, which then
go into the inverse dynamics (your code). This file gives you that
trajectory; it does not compute any torques.

The polynomial q(t) = a0 + a1 t + ... + a6 t^6 has 7 coefficients, fixed by 7
constraints: initial/final/mid-time position, initial/final velocity, and
initial/final acceleration. See the appendix of the assignment statement.

Run it directly to see two example trajectories:

    python trajectory_generator.py

or import it from your own script (in the same folder):

    from trajectory_generator import gen_trajectory
    time, traj = gen_trajectory([0, 0, 90, 0, 0, 0, 0], [0, 3], n=300)
    q, qd, qdd = traj            # position, velocity, acceleration

----------------------------------------------------------------------------
New to Python? Things you will meet in this file:
  * `np.linalg.solve(A, x)` solves the linear system A a = x for `a`
    (MATLAB's `A\\x`).
  * NumPy operations act on whole arrays at once, e.g. `time**2` squares
    every sample -- no `for` loop needed.
  * `traj[0]` is the first row of a 2D array (here, the position).
----------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np


def gen_trajectory(x, t, n=100, demo_plot=False):
    """Return a 6th-order polynomial trajectory.

    x : 7 desired values [x0, xf, xm, xd0, xdf, xdd0, xddf] -- initial, final,
        and mid-time position; initial and final velocity; initial and final
        acceleration. Units are whatever you pass in (e.g. deg or rad).
    t : [t0, tf], initial and final time in seconds. The mid-time position
        xm is reached at tm = (t0 + tf) / 2.
    n : number of samples.
    demo_plot : if True, plot position, velocity, and acceleration.

    Returns (time, traj): `time` has shape (n,); `traj` has shape (3, n) with
    rows position, velocity, acceleration.
    """
    t0, tf = t
    tm = (t0 + tf) / 2  # Midpoint in time

    # Each row applies one constraint to the coefficients a = [a0, ..., a6]
    A = np.array([
        [1, t0, t0**2, t0**3, t0**4, t0**5, t0**6],                  # q(t0)
        [1, tf, tf**2, tf**3, tf**4, tf**5, tf**6],                  # q(tf)
        [1, tm, tm**2, tm**3, tm**4, tm**5, tm**6],                  # q(tm)
        [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4, 6*t0**5],            # qd(t0)
        [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4, 6*tf**5],            # qd(tf)
        [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3, 30*t0**4],               # qdd(t0)
        [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3, 30*tf**4],               # qdd(tf)
    ], dtype=float)
    # Polynomial coefficients
    a = np.linalg.solve(A, np.asarray(x, dtype=float))

    # Evaluate position, velocity, and acceleration at every sample
    time = np.linspace(t0, tf, n)
    one, zero = np.ones_like(time), np.zeros_like(time)
    basis = np.array([
        [one, time, time**2, time**3, time**4, time**5, time**6],
        [zero, one, 2*time, 3*time**2, 4*time**3, 5*time**4, 6*time**5],
        [zero, zero, 2*one, 6*time, 12*time**2, 20*time**3, 30*time**4],
    ])                                   # shape (3, 7, n)
    traj = np.einsum("dkn,k->dn", basis, a)  # sum_k basis[d, k, :] * a[k]

    if demo_plot:
        fig, axes = plt.subplots(3, 1, sharex=True)
        fig.suptitle("Polynomial Trajectory")
        for ax, row, label in zip(axes, traj, ["Position", "Velocity", "Acceleration"]):
            ax.plot(time, row)
            ax.set_ylabel(label)
            ax.grid(True)
        axes[-1].set_xlabel("Time [s]")

    return time, traj


if __name__ == "__main__":
    # Two trajectory examples: [x0, xf, xm, xd0, xdf, xdd0, xddf]
    x1 = [0, 0, 90, 0, 0, 0, 0]          # A5 elbow lift cycle (deg): 0 -> 90 -> 0
    x2 = [10, 20, 90, 30, 40, 50, 60]    # arbitrary boundary conditions
    t = [0, 3]                           # Initial and final time [s]
    n = 100                              # Number of samples

    time1, traj1 = gen_trajectory(x1, t, n, demo_plot=True)
    time2, traj2 = gen_trajectory(x2, t, n, demo_plot=True)

    # Sanity check: the trajectory meets its boundary conditions
    for x in [x1, x2]:
        _, s = gen_trajectory(x, t, 3)  # 3 samples: at t0, tm, tf
        got = [s[0, 0], s[0, 2], s[0, 1], s[1, 0], s[1, 2], s[2, 0], s[2, 2]]
        print(f"x = {x} -> constraints met: {np.allclose(got, x)}")

    # Remember: convert degrees to radians before using SI units in the dynamics,
    # e.g. q, qd, qdd = np.deg2rad(traj1)
    plt.show()

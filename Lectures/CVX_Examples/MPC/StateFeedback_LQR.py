"""State-space position control of a DC motor - LQR state feedback
Python version of StateFeedback_LQR.m.

Builds the state-space model of a brushed DC motor from its datasheet,
checks controllability, designs a state-feedback gain with LQR, and
simulates the closed loop tracking a constant position reference r.

    State  x = [theta, omega, i]  (angle [rad], speed [rad/s], current [A])
    Input  u = winding voltage [V]
    Output y = theta

    LQR picks K in u = -K @ (x - x_ref) to minimize
    J = integral( e'Qe + u'Ru ) dt,  e = x - x_ref.

Requires: numpy, scipy, matplotlib, control (see ../requirements.txt).
          control.lqr is the equivalent of MATLAB's lqr.
Output:   "Controller Response" figure (position vs. reference, voltage).
Try:      raise Q[0, 0] for a faster response, raise R to use less voltage.
          Compare the peak voltage to what the motor driver can supply.
          This is the gap that MPC (see cvxgen/) closes by adding |u| <= u_max.
"""

import control
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Define system parameters from datasheet. For example, PN 310005 - Maxon
Rw = 0.196                      # Ohm            winding resistance
Lw = 0.0345e-3                  # Henry          winding inductance
Kt = 13.9e-3                    # N.m/amp        torque constant
Kv = Kt                         # V/(rad/s)      back-EMF constant (= Kt in SI)
Jm = 33.7 * (1 / 1e3) * (1 / 100**2)   # Kg m^2  rotor inertia (33.7 g cm^2)
bm = 3.61e-3                    # N.m/rad*sec    viscous friction


def simulate_and_plot_ABCD(A, B, C, K, r):
    """Simulate x' = A e + B u with u = -K e, e = x - [r, 0, 0].

    Writing A e instead of A x is valid because A's first column is zero:
    any constant angle with zero speed and current is an equilibrium.
    """
    x_ref = np.array([r, 0, 0])
    x0 = [10, 2, 3]                 # Initial conditions for the state
    tspan = [0, 40]                 # Time window for simulation

    def closed_loop(t, x):
        e = x - x_ref
        return A @ e + B @ (-K @ e)

    # The electrical pole (-Rw/Lw ~ -5700 1/s) makes this ODE stiff, so use
    # LSODA (switches to a stiff method) instead of RK45, MATLAB's ode45 analog.
    sol = solve_ivp(closed_loop, tspan, x0, method="LSODA",
                    t_eval=np.linspace(*tspan, 4001), rtol=1e-6, atol=1e-9)
    t, x = sol.t, sol.y             # x is 3 x len(t), like MATLAB's x.'

    # Compute output and control input from the simulated state
    y = C @ x
    u = -K @ (x - x_ref[:, None])

    # Plot the analysis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.suptitle("Controller Response")
    ax1.set_title("Output")
    ax1.plot(t, y.ravel(), label="Output")
    ax1.plot(t, np.full_like(t, r), "--r", label="Reference")
    ax1.set_ylabel("Motor Angular Position [rad]")
    ax1.grid(True)
    ax1.legend()
    ax2.set_title("Control Input")
    ax2.plot(t, u.ravel())
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Voltage Input [Volts]")
    ax2.grid(True)
    plt.show()


def main():
    # Create state space model
    #   theta' = omega
    #   Jm*omega' = -bm*omega + Kt*i          (mechanical)
    #   Lw*i'     = -Kv*omega - Rw*i + u      (electrical)
    A = np.array([[0, 1, 0],
                  [0, -bm / Jm, Kt / Jm],
                  [0, -Kv / Lw, -Rw / Lw]])
    B = np.array([[0], [0], [1 / Lw]])
    C = np.array([[1, 0, 0]])
    D = 0

    # Reference position [rad]
    r = 20

    # --- Properties of the system
    # Controllability matrix [B AB A^2B]
    W = control.ctrb(A, B)
    if np.linalg.matrix_rank(W) == 3:
        print("The system is controllable")
    else:
        print("The system is not controllable")

    # --- State feedback (LQR) controller design
    Q = np.diag([10, 1, 1])     # weights on position error, speed, current
    R = 100                     # weight on voltage

    K, S, P = control.lqr(A, B, Q, R)   # K: gain, S: Riccati solution, P: closed-loop poles
    print("LQR gain K =", K)

    # --- Evaluate performance through simulation
    simulate_and_plot_ABCD(A, B, C, K, r)


if __name__ == "__main__":
    main()

"""Four-segment sagittal-plane body model of Kuo (1998), Appendix A.

Segments, numbered from the ground up: 1 foot (motionless), 2 shank,
3 thigh, 4 head-arms-trunk (HAT). Joints: 1 ankle, 2 knee, 3 hip.

Conventions (Kuo1998.tex, Fig. 1):
- phi_i is the absolute angle of segment i, counterclockwise from the
  horizontal (+x forward, +y up). Phi = [phi_1..phi_4]; phi = Phi[1:].
- tau = [tau_1, tau_2, tau_3] = ankle, knee, hip torque. tau_i acts on
  segment i+1 and its reaction -tau_i acts on segment i.
- Ground reaction on the foot, taken about joint 0 (the foot's lower end,
  at the origin): tau_0, f_0 = [f_x, f_y].

The OpenSim model (four_segment_opensim.py) uses relative joint angles
q_i = phi_{i+1} - phi_i. By virtual work, its generalized force for q_i is
exactly Kuo's tau_i, so the two models report the same torques.

Equation numbers refer to Kuo1998.tex in this folder.
"""

import numpy as np

# Benchmark parameters (Appendix A). Index 0..3 = segment 1..4.
L = np.array([0.177, 0.405, 0.440, 0.795])    # segment length [m]
LC = np.array([0.086, 0.235, 0.268, 0.302])   # lower end to COM [m]
MASS = np.array([1.78, 7.30, 15.5, 44.6])     # [kg]
INERTIA = np.array([0.0080, 0.097, 0.253, 1.51])  # about the COM [kg m^2]
G = 9.81                                      # [m/s^2]
# Kuo does not give the foot angle. With phi_1 = 90 deg the foot link is
# vertical, so standing upright means all OpenSim joint angles are zero.
PHI1 = np.pi / 2

N_SEG = 4
I2 = np.eye(2)


def _kinematics(Phi, dPhi):
    """Jacobian J = d x_c / d Phi and its time derivative (after A1)."""
    s, c = np.sin(Phi), np.cos(Phi)
    # Lever[i, j]: length of segment j that reaches the COM of segment i
    Lever = np.tril(np.tile(L, (N_SEG, 1)), -1) + np.diag(LC)
    J = np.zeros((2 * N_SEG, N_SEG))
    Jdot = np.zeros((2 * N_SEG, N_SEG))
    J[0::2], J[1::2] = -Lever * s, Lever * c
    Jdot[0::2], Jdot[1::2] = -Lever * c * dPhi, -Lever * s * dPhi
    return J, Jdot


def _moment_arms(Phi):
    """[P0 | P | ...] with columns for f_0..f_4 (A5 and A12).

    Row i holds r_i x f_{i-1} - s_i x f_i for segment i. Kuo's P is
    columns f_0..f_3, P0 is f_0 only, and P-hat is f_1..f_4.
    """
    s, c = np.sin(Phi), np.cos(Phi)
    P = np.zeros((N_SEG, 2 * (N_SEG + 1)))
    for i in range(N_SEG):
        P[i, 2 * i:2 * i + 2] = LC[i] * np.array([s[i], -c[i]])
        P[i, 2 * i + 2:2 * i + 4] = (L[i] - LC[i]) * np.array([s[i], -c[i]])
    return P


# Constant matrices of Appendix A
S1 = np.eye(N_SEG) - np.eye(N_SEG, k=1)          # tau_{i-1} - tau_i
S2 = np.kron(S1, I2)                             # f_{i-1} - f_i
S1_HAT = -np.eye(N_SEG) + np.eye(N_SEG, k=-1)
S2_HAT = np.kron(S1_HAT, I2)
S1p = np.ones(N_SEG)                             # first row of S1^-1
S2p = np.kron(np.ones(N_SEG), I2)                # first block row of S2^-1
M_D = np.diag(np.repeat(MASS, 2))
g_V = np.kron(MASS, [0.0, 1.0]) * G


def matrices(Phi, dPhi):
    """Equations of motion (3) and ground reactions (4).

    Returns M, Q, g, v such that  M phi_dd = Q tau + g + v      (3)
    and C, c such that            C phi_dd = [tau_0; f_0] + c   (4)
    """
    J, Jdot = _kinematics(Phi, dPhi)
    P = _moment_arms(Phi)[:, :2 * N_SEG]                 # f_0..f_3
    PS = P @ np.linalg.inv(S2)
    Mcal = np.diag(INERTIA) - PS @ M_D @ J               # (A7)
    Vcal = PS @ M_D @ Jdot @ dPhi
    Gcal = PS @ g_V
    # Lower blocks of (A8): segments 2..4
    M, Q, g, v = Mcal[1:, 1:], S1[1:, 1:], Gcal[1:], Vcal[1:]
    # (A9)-(A10); drop column 1 because phi_1_dd = 0
    C = np.vstack([S1p @ Mcal, S2p @ M_D @ J])[:, 1:]
    c = np.concatenate([[S1p @ (Vcal + Gcal)],
                        -S2p @ M_D @ Jdot @ dPhi - S2p @ g_V])
    return M, Q, g, v, C, c


def ls_system(Phi, dPhi, ddphi, tau0, f0):
    """A and b of the overdetermined system A tau = b, eq. (6).

    Rows: tau_0, f_x, f_y, phi_2_dd, phi_3_dd, phi_4_dd.
    """
    M, Q, g, v, C, c = matrices(Phi, dPhi)
    Minv = np.linalg.inv(M)
    A = np.vstack([C @ Minv @ Q, Minv @ Q])
    b = (np.concatenate([[tau0], f0, ddphi])
         + np.concatenate([c, np.zeros(3)])
         - np.vstack([C, np.eye(3)]) @ Minv @ (g + v))
    return A, b


def newton_euler(Phi, dPhi, ddphi, tau0, f0):
    """Conventional Newton-Euler, from the force plate upward (A11-A12).

    Returns [tau_1, tau_2, tau_3, tau_4]; tau_4 is the residual torque on
    the top segment, which would be zero with perfect data.
    """
    J, Jdot = _kinematics(Phi, dPhi)
    P_all = _moment_arms(Phi)
    P0, P_hat = P_all[:, :2], P_all[:, 2:]
    ddPhi = np.concatenate([[0.0], ddphi])
    xc_dd = J @ ddPhi + Jdot @ dPhi                                  # (A3)
    f_hat = np.linalg.solve(S2_HAT, M_D @ xc_dd + g_V) + S2p.T @ f0  # (A11)
    rhs = np.diag(INERTIA) @ ddPhi - P_hat @ f_hat - P0 @ f0
    return np.linalg.solve(S1_HAT, rhs) + S1p * tau0                # (A12)


def joint_positions(Phi):
    """(x, y) of joint centers 1..4 (ankle, knee, hip, top of HAT).

    Markers at these points give the segment angles 2..4 by eq. (B3).
    """
    seg = L[:, None] * np.array([np.cos(Phi), np.sin(Phi)]).T
    return np.cumsum(seg, axis=0)

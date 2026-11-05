"""Least-squares inverse dynamics (Kuo, 1998) with cvxpy

Problem.  Conventional Newton-Euler (NE) inverse dynamics starts from the
force plate and works up the body, one segment at a time. With noisy data
it cannot use every measurement consistently: the leftover error ends up as
a residual torque on the top segment, and the hip torque is noisy.

Kuo (1998) instead stacks every measurement at time step k, the ground
reaction [tau_0, f_x, f_y] and the angular accelerations of segments 2-4,
into one overdetermined linear system in the three joint torques,

    A[k] tau[k] = b[k]                                        eq. (6)

and finds the torques that need the smallest (weighted) adjustment of the
measurements to make them consistent:

    minimize  sum_k || W^(-1/2) (A[k] tau[k] - b[k]) ||^2      eqs. (7)-(8)

W holds the variance of each measurement, so precise measurements (the
force plate) count more than noisy ones (twice-differentiated markers).

The four-segment model is the benchmark of Kuo's Appendix A
(four_segment_model.py); the "true" motion, torques and ground reactions
come from the equivalent OpenSim model (four_segment_opensim.py).

Your tasks (section 4).  TODO 1: write the least-squares problem in cvxpy.
TODO 2 (only when ESTIMATE_BIAS = True): add an unknown constant bias.

Requires: numpy, scipy, matplotlib, cvxpy, opensim (see ../requirements.txt).
Output:   printed RMSE table; figure of the joint torques (Kuo's Fig. 3a)
          and a bar chart of the errors (Fig. 3b).
Try:      1. Remove "fx" (or "tau0") from GRF_CHANNELS: least squares still
             works with an incomplete force plate (Kuo's Fig. 6).
          2. Set TAU0_BIAS = 0.01 * TOTAL_WEIGHT (the force plate's origin
             1 cm off): NE gets worse. Then set ESTIMATE_BIAS = True and do
             TODO 2: least squares finds the bias (Kuo's Fig. 7).
          3. Change MARKER_NOISE or FORCE_NOISE, or SEED.
"""

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.signal import butter, filtfilt

import four_segment_model as kuo
from four_segment_opensim import generate_truth

# ---- Settings ------------------------------------------------------------
FS = 60.0                     # sampling rate [Hz]
MARKER_NOISE = 0.01           # marker position noise, std [m]
FORCE_NOISE = 0.1             # force plate noise, std [N] and [N m]
CUTOFF = 5.0                  # low-pass filter cutoff [Hz]
SEED = 0                      # random seed for the noise
GRF_CHANNELS = ["tau0", "fx", "fy"]   # force plate channels to use
TOTAL_WEIGHT = kuo.MASS.sum() * kuo.G
TAU0_BIAS = 0.0               # constant error added to tau_0 [N m]
ESTIMATE_BIAS = False         # estimate that bias with least squares

JOINT_NAMES = ["Ankle", "Knee", "Hip"]
ROW_NAMES = ["tau0", "fx", "fy", "ddphi2", "ddphi3", "ddphi4"]  # rows of A, b

# ---- 1. True motion, torques and ground reactions (OpenSim) ----------------
truth = generate_truth(fs=FS)
t, N = truth["t"], truth["t"].size

# ---- 2. Simulated measurements (Kuo's protocol) ----------------------------
# Markers at the joint centers and the force plate, each with white noise,
# then a 3rd-order Butterworth low-pass filter run forward and backward.
rng = np.random.default_rng(SEED)
markers = np.array([kuo.joint_positions(Phi) for Phi in truth["Phi"]])  # (N, 4, 2)
markers = markers + MARKER_NOISE * rng.standard_normal(markers.shape)
grf = truth["grf"] + FORCE_NOISE * rng.standard_normal(truth["grf"].shape)
grf[:, 0] += TAU0_BIAS

bf, af = butter(3, CUTOFF / (FS / 2))
markers = filtfilt(bf, af, markers, axis=0)
grf = filtfilt(bf, af, grf, axis=0)

# Segment angles from neighboring markers, eq. (B3), then differentiate twice
d = np.diff(markers, axis=1)                                   # (N, 3, 2)
phi = np.unwrap(np.arctan2(d[..., 1], d[..., 0]), axis=0)      # segments 2-4
dphi = np.gradient(phi, 1 / FS, axis=0)
ddphi = np.gradient(dphi, 1 / FS, axis=0)
Phi = np.hstack([np.full((N, 1), kuo.PHI1), phi])
dPhi = np.hstack([np.zeros((N, 1)), dphi])

# Measurement covariance W (diagonal): the variance of each measurement's
# error. In a simulation we know the error; with real data, Kuo's Appendix B
# predicts it from the noise level, the filter and the differentiation.
meas_error = np.hstack([grf - truth["grf"], ddphi - truth["ddPhi"][:, 1:]])
sigma = meas_error.std(axis=0)

# ---- 3. Conventional Newton-Euler (A12), from the force plate upward -------
tau_ne = np.array([kuo.newton_euler(Phi[k], dPhi[k], ddphi[k], grf[k, 0], grf[k, 1:])
                   for k in range(N)])      # (N, 4); column 4 is the residual

# ---- 4. Least squares, all time steps at once ------------------------------
# Keep only the rows of the available measurements (all accelerations, plus
# the force plate channels listed in GRF_CHANNELS).
keep = np.array([r in GRF_CHANNELS or r.startswith("ddphi") for r in ROW_NAMES])
A_list, b_list = [], []
for k in range(N):
    A, b = kuo.ls_system(Phi[k], dPhi[k], ddphi[k], grf[k, 0], grf[k, 1:])
    A_list.append(sp.csr_array(A[keep]))
    b_list.append(b[keep])
# Block-diagonal system for all N steps: A_all @ [tau[1]; ...; tau[N]] = b_all
A_all = sp.block_diag(A_list, format="csr")
b_all = np.concatenate(b_list)
w_all = np.tile(1 / sigma[keep], N)          # row weights W^(-1/2)
B_all = np.tile(np.array(ROW_NAMES)[keep] == "tau0", N).astype(float)  # eq. (9)

tau = cp.Variable(3 * N)
# TODO 1a: the residual of eq. (6b) at every time step, A_all tau - b_all.
residual = None
if ESTIMATE_BIAS:
    beta = cp.Variable()
    # TODO 2: add an unknown constant bias on the tau_0 measurement,
    #   A tau = b + B beta, eq. (9), so beta should come out as -TAU0_BIAS.
    #   B_all marks the tau_0 rows; subtract B_all * beta from the residual.
    residual = residual
# TODO 1b: the objective of eqs. (7)-(8): minimize the sum of squares of
#   the residual, each entry weighted by w_all (use cp.multiply).
objective = None

if objective is None:
    raise NotImplementedError("Fill in TODO 1 (section 4) first.")
cp.Problem(objective).solve()
if ESTIMATE_BIAS and beta.value is None:
    raise NotImplementedError("Fill in TODO 2 (section 4), or set ESTIMATE_BIAS = False.")
tau_ls = tau.value.reshape(N, 3)

# Check: the same problem in closed form, the weighted pseudoinverse of eq. (8)
G = A_all.toarray()
if ESTIMATE_BIAS:
    G = np.hstack([G, -B_all[:, None]])
x_ref = np.linalg.lstsq(w_all[:, None] * G, w_all * b_all, rcond=None)[0]
print(f"Max |cvxpy - closed form|: {np.abs(tau.value - x_ref[:3 * N]).max():.1e} N m")
if ESTIMATE_BIAS:
    print(f"Estimated beta: {beta.value:.2f} N m (expected {-TAU0_BIAS:.2f})")

# ---- 5. Results ------------------------------------------------------------
def rmse(est):
    """Root time-averaged squared error per joint, eq. (14), and overall, eq. (15)."""
    err = est - truth["tau"]
    return np.sqrt(np.mean(err**2, axis=0)), np.sqrt(np.mean(np.sum(err**2, axis=1)))

rmse_ne, total_ne = rmse(tau_ne[:, :3])
rmse_ls, total_ls = rmse(tau_ls)
print(f"\nJoint torque RMSE [N m]   {'NE':>7} {'LS':>7}")
for name, e_ne, e_ls in zip(JOINT_NAMES, rmse_ne, rmse_ls):
    print(f"  {name:<23} {e_ne:7.2f} {e_ls:7.2f}")
print(f"  {'Overall':<23} {total_ne:7.2f} {total_ls:7.2f}"
      f"   ({100 * (1 - total_ls / total_ne):.0f}% lower with LS)")
print(f"  {'Residual tau_4 (NE)':<23} {np.sqrt(np.mean(tau_ne[:, 3]**2)):7.2f}    none")

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(7, 7))
fig.suptitle("Joint torques (Kuo 1998, Fig. 3a)")
for i, ax in enumerate(axs):
    ax.plot(t, truth["tau"][:, i], "k", lw=2, label="True")
    ax.plot(t, tau_ne[:, i], "--", label="Newton-Euler")
    ax.plot(t, tau_ls[:, i], label="Least squares")
    ax.set_ylabel(f"{JOINT_NAMES[i]} [N m]")
    ax.grid(True)
fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center",
           ncol=3, bbox_to_anchor=(0.5, 0.95))
axs[-1].set_xlabel("Time [s]")

plt.figure()
x = np.arange(4)
plt.bar(x - 0.2, np.append(rmse_ne, total_ne), 0.4, label="Newton-Euler")
plt.bar(x + 0.2, np.append(rmse_ls, total_ls), 0.4, label="Least squares")
plt.xticks(x, JOINT_NAMES + ["Overall"])
plt.ylabel("RMSE [N m]")
plt.title("Joint torque error (Kuo 1998, Fig. 3b)")
plt.legend()
plt.grid(True, axis="y")
plt.show()

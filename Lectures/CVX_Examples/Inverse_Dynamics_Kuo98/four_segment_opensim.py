"""Equivalent OpenSim model of Kuo's (1998) four-segment body.

Builds the four-segment benchmark of Appendix A with the OpenSim API, and
uses it to:
  1. generate the "true" (noise-free) data for least_squares_id.py:
     joint torques from OpenSim's inverse dynamics, and ground reactions
     from the reaction force in the joint that welds the foot to the floor;
  2. check that the equations in four_segment_model.py give the same
     numbers as OpenSim (they should agree to round-off, ~1e-10);
  3. write _work/four_segment.osim and _work/sway.sto, so you can watch the
     motion in the OpenSim GUI (File > Open Model, then File > Load Motion).

Kuo drove his simulation with sinusoidal joint torques plus a feedback
controller. Here the joint angles are prescribed instead (sinusoids from
0.5 to 2.2 Hz, Kuo's frequency band), which gives the same kind of motion
without having to stabilize a simulation. The torques and ground reactions
then follow exactly from inverse dynamics.

Run directly to validate and write the visualization files:
    python four_segment_opensim.py
"""

from pathlib import Path

import numpy as np
import opensim as osim

import four_segment_model as kuo

HERE = Path(__file__).resolve().parent
WORK_DIR = HERE / "_work"
SEGMENTS = ["foot", "shank", "thigh", "hat"]
JOINTS = ["ankle", "knee", "hip"]


# --------------------------------------------------------------------------- #
# Model: same segments and parameters as four_segment_model.py
# --------------------------------------------------------------------------- #
def build_model():
    """Planar 4-segment model. Each body frame sits at the segment's lower
    joint, with the segment along the body's +x axis and the COM at
    (lc, 0, 0). The foot is welded to the ground at angle phi_1; ankle, knee
    and hip are pin joints about z with relative angles q_i = phi_{i+1} -
    phi_i (positive = counterclockwise, i.e. the upper segment tilts back).
    """
    model = osim.Model()
    model.setName("kuo1998_four_segment")
    model.set_gravity(osim.Vec3(0, -kuo.G, 0))

    bodies = []
    for name, m, lc, I, l in zip(SEGMENTS, kuo.MASS, kuo.LC, kuo.INERTIA, kuo.L):
        # Only the z (out-of-plane) inertia matters for planar motion
        body = osim.Body(name, m, osim.Vec3(lc, 0, 0), osim.Inertia(I))
        # Display geometry: an ellipsoid along the segment
        mid = osim.PhysicalOffsetFrame(f"{name}_mid", body,
                                       osim.Transform(osim.Vec3(l / 2, 0, 0)))
        body.addComponent(mid)
        mid.attachGeometry(osim.Ellipsoid(l / 2, 0.04, 0.04))
        model.addBody(body)
        bodies.append(body)

    # Foot welded to the floor; its frame origin is Kuo's joint 0, so the
    # weld's reaction moment is the ground reaction torque tau_0.
    weld = osim.WeldJoint("ground_foot", model.getGround(), osim.Vec3(0),
                          osim.Vec3(0, 0, kuo.PHI1), bodies[0], osim.Vec3(0),
                          osim.Vec3(0))
    model.addJoint(weld)

    for i, name in enumerate(JOINTS):
        parent, child = bodies[i], bodies[i + 1]
        joint = osim.PinJoint(name, parent, osim.Vec3(kuo.L[i], 0, 0),
                              osim.Vec3(0), child, osim.Vec3(0), osim.Vec3(0))
        joint.updCoordinate().setName(f"{name}_angle")
        model.addJoint(joint)
        # A torque actuator per joint, used to apply the inverse-dynamics
        # torques back to the model as a check
        act = osim.CoordinateActuator(f"{name}_angle")
        act.setName(f"tau_{name}")
        act.setOptimalForce(1.0)
        model.addForce(act)

    model.finalizeConnections()
    return model


# --------------------------------------------------------------------------- #
# Truth data: prescribed joint angles -> torques and ground reactions
# --------------------------------------------------------------------------- #
def joint_angle_trajectory(t, seed=0):
    """Joint angles q (N x 3) and their exact 1st and 2nd derivatives.

    Each angle is an offset (a slightly flexed stance) plus five sinusoids
    from 0.5 to 2.2 Hz with random phases. Amplitudes fall as 1/f^2, so each
    sinusoid adds the same angular acceleration, ACC [rad/s^2].
    """
    ACC = np.array([0.2, 0.3, 0.4])[:, None]    # ankle, knee, hip
    rng = np.random.default_rng(seed)
    offset = np.radians([-5.0, 10.0, -5.0])
    w = 2 * np.pi * np.linspace(0.5, 2.2, 5)    # [rad/s]
    amp = ACC / w**2                            # (3, 5) [rad]
    phase = rng.uniform(0, 2 * np.pi, (3, w.size))
    arg = w * t[:, None, None] + phase          # (N, 3, 5)
    q = offset + (amp * np.sin(arg)).sum(axis=2)
    dq = (amp * w * np.cos(arg)).sum(axis=2)
    ddq = -(amp * w**2 * np.sin(arg)).sum(axis=2)
    return q, dq, ddq


def generate_truth(duration=4.0, fs=60.0, seed=0):
    """Sample the prescribed motion at fs and compute, with OpenSim, the
    joint torques (inverse dynamics) and the ground reaction on the foot.

    Returns a dict with t, q (N x 3), Phi, dPhi, ddPhi (N x 4 absolute
    segment angles), tau (N x 3: ankle, knee, hip) and grf (N x 3: tau_0,
    f_x, f_y).
    """
    model = build_model()
    state = model.initSystem()
    coords = [model.getCoordinateSet().get(f"{j}_angle") for j in JOINTS]
    actuators = [osim.ScalarActuator.safeDownCast(
        model.getForceSet().get(f"tau_{j}")) for j in JOINTS]
    weld = model.getJointSet().get("ground_foot")
    id_solver = osim.InverseDynamicsSolver(model)

    t = np.arange(0, duration, 1 / fs)
    q, dq, ddq = joint_angle_trajectory(t, seed)
    tau, grf = np.zeros((t.size, 3)), np.zeros((t.size, 3))
    udot_err = 0.0
    for k in range(t.size):
        for i, coord in enumerate(coords):
            coord.setValue(state, q[k, i], False)
            coord.setSpeedValue(state, dq[k, i])
        # 1) Inverse dynamics with the actuators switched off
        for act in actuators:
            act.overrideActuation(state, True)
            act.setOverrideActuation(state, 0.0)
        tau[k] = id_solver.solve(state, osim.Vector(ddq[k].tolist())).to_numpy()
        # 2) Apply those torques and run the forward dynamics: the model
        #    must reproduce the prescribed accelerations
        for act, torque in zip(actuators, tau[k]):
            act.setOverrideActuation(state, torque)
        model.realizeAcceleration(state)
        udot_err = max(udot_err, np.abs(state.getUDot().to_numpy() - ddq[k]).max())
        # 3) Ground reaction = force and moment of the floor on the foot
        reaction = weld.calcReactionOnChildExpressedInGround(state)
        moment, force = reaction.get(0), reaction.get(1)
        grf[k] = [moment.get(2), force.get(0), force.get(1)]

    # Absolute segment angles: phi_1 fixed, phi_{i+1} = phi_i + q_i
    Phi = kuo.PHI1 + np.hstack([np.zeros((t.size, 1)), np.cumsum(q, axis=1)])
    dPhi = np.hstack([np.zeros((t.size, 1)), np.cumsum(dq, axis=1)])
    ddPhi = np.hstack([np.zeros((t.size, 1)), np.cumsum(ddq, axis=1)])
    return dict(t=t, q=q, Phi=Phi, dPhi=dPhi, ddPhi=ddPhi, tau=tau, grf=grf,
                udot_err=udot_err)


# --------------------------------------------------------------------------- #
# Validation: Appendix A equations vs. OpenSim
# --------------------------------------------------------------------------- #
def validate(truth, tol=1e-8):
    """Compare the equations of four_segment_model.py with OpenSim at every
    sample. Prints the largest difference for each check."""
    err = dict(tau=0.0, grf=0.0, ne=0.0, residual=0.0, ls=0.0)
    for k in range(truth["t"].size):
        Phi, dPhi = truth["Phi"][k], truth["dPhi"][k]
        ddphi = truth["ddPhi"][k, 1:]
        tau, grf = truth["tau"][k], truth["grf"][k]
        M, Q, g, v, C, c = kuo.matrices(Phi, dPhi)
        tau_eq = np.linalg.solve(Q, M @ ddphi - g - v)             # eq. (3)
        grf_eq = C @ ddphi - c                                     # eq. (4)
        ne = kuo.newton_euler(Phi, dPhi, ddphi, grf[0], grf[1:])   # (A12)
        A, b = kuo.ls_system(Phi, dPhi, ddphi, grf[0], grf[1:])    # eq. (6)
        err["tau"] = max(err["tau"], np.abs(tau_eq - tau).max())
        err["grf"] = max(err["grf"], np.abs(grf_eq - grf).max())
        err["ne"] = max(err["ne"], np.abs(ne[:3] - tau).max())
        err["residual"] = max(err["residual"], abs(ne[3]))
        err["ls"] = max(err["ls"], np.abs(A @ tau - b).max())

    checks = [
        ("OpenSim forward dynamics reproduces the accelerations", truth["udot_err"]),
        ("Joint torques: eq. (3) vs. OpenSim inverse dynamics", err["tau"]),
        ("Ground reactions: eq. (4) vs. OpenSim weld reaction", err["grf"]),
        ("Newton-Euler (A12) torques vs. OpenSim", err["ne"]),
        ("Newton-Euler residual tau_4 (should be 0)", err["residual"]),
        ("A tau - b with exact data, eq. (6) (should be 0)", err["ls"]),
    ]
    for label, e in checks:
        print(f"  {'PASS' if e < tol else 'FAIL'}  max error {e:8.1e}  {label}")
    return all(e < tol for _, e in checks)


def write_visualization(truth):
    """Write the model and its motion for the OpenSim GUI."""
    WORK_DIR.mkdir(exist_ok=True)
    build_model().printToXML(str(WORK_DIR / "four_segment.osim"))
    labels = [f"/jointset/{j}/{j}_angle/value" for j in JOINTS]
    table = osim.TimeSeriesTable(truth["t"].tolist(),
                                 osim.Matrix.createFromMat(truth["q"]), labels)
    table.addTableMetaDataString("inDegrees", "no")
    osim.STOFileAdapter.write(table, str(WORK_DIR / "sway.sto"))
    print(f"Wrote {WORK_DIR / 'four_segment.osim'} and {WORK_DIR / 'sway.sto'}")


if __name__ == "__main__":
    truth = generate_truth()
    print("Validating the Appendix A equations against OpenSim:")
    validate(truth)
    write_visualization(truth)

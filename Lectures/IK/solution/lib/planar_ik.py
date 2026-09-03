"""Barehand inverse kinematics: a 4-DOF planar linkage solved per frame by
weighted nonlinear least squares (Levenberg-Marquardt).

Unknowns   q = [theta_ankle, theta_knee, theta_hip, theta_lumbar]   (rad,
           measured as the change from the static standing pose)

Given, each frame
    ajc        ankle joint centre, sagittal (x, y)
    foot_ang   foot orientation (heel -> forefoot), rad
    meas       measured sagittal positions of the segment markers

Forward kinematics
    phi_shank  = foot_ang + (phi_shank0 - phi_foot0) + theta_ankle
    phi_thigh  = phi_shank + (phi_thigh0 - phi_shank0) + theta_knee
    phi_pelvis = phi_thigh + (phi_pelvis0 - phi_thigh0) + theta_hip
    phi_trunk  = phi_pelvis + (phi_trunk0 - phi_pelvis0) + theta_lumbar
    KJC = ajc + L_shank * [cos phi_shank, sin phi_shank]
    HJC = KJC + L_thigh * [cos phi_thigh, sin phi_thigh]
    LJC = HJC + R(phi_pelvis) @ d_pelvis_lumbar
    marker_i = O_seg + R(phi_seg) @ local_i

Cost   f(q) = 1/2 * sum_i w_i * || marker_i(q) - meas_i ||^2
We drive the cost to a minimum with Gauss-Newton / Levenberg-Marquardt.

----------------------------------------------------------------------------
Maths <-> code cheat sheet:
    R(phi)              rot(phi)        2-D rotation matrix
    d/dphi [cos, sin]   _e_prime = _ep(phi)
    d/dphi R(phi)       _Rp(phi)  ( == R(phi + 90 deg) )
    A @ B               matrix / matrix-vector multiply
    x @ x               dot product of a vector with itself ( = ||x||^2 )
    J.T                 transpose of J
Python notes:
    lambda a, b: expr   a tiny inline function
    r[2*i : 2*i+2]      rows 2i and 2i+1 (this marker's x- and y-error)
    try: ... except E:  run the first block, jump to the second if it errors
----------------------------------------------------------------------------
"""

# `from __future__ import annotations` makes ": type" hints lazy; ignore it.
from __future__ import annotations

import numpy as np

from .calibration import SEGMENT_MARKERS, rot     # rot(phi) = 2-D rotation matrix


def _e(phi):
    """Unit vector pointing along angle `phi`:  [cos phi, sin phi]."""
    return np.array([np.cos(phi), np.sin(phi)])


def _ep(phi):
    """Derivative of _e with respect to phi:  [-sin phi, cos phi]."""
    return np.array([-np.sin(phi), np.cos(phi)])


def _Rp(phi):
    """Derivative of the rotation matrix rot(phi) with respect to phi.
    (It equals rot(phi + 90 deg).)
    """
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[-s, -c],
                     [c, -s]])


# The four segments, from the ground up.  Their index (0..3) matches `q`.
SEG_ORDER = ["shank", "thigh", "pelvis", "trunk"]

# How much to trust each marker (w_i in the cost).  Bony landmarks (ASIS, PSIS,
# knee, C7) move less relative to the bone than skin-mounted wands, so they get
# bigger weights.
DEFAULT_WEIGHTS = {
    "RIAS": 10.0, "LIAS": 10.0, "RIPS": 10.0, "LIPS": 10.0,
    "RFLE": 6.0, "CV7": 6.0, "SJN": 4.0,
    "RTH": 1.0, "RFTC": 2.0, "RSK": 1.0, "RTTC": 1.0, "RFAX": 1.0,
    "SXS": 3.0, "TV2": 3.0, "TV7": 3.0,
}


def forward_kinematics(q, calib, foot_ang, ajc):
    """Given the four joint angles `q`, work out where every segment is.

    Returns a dict:
        "phi"    -> array [phi_shank, phi_thigh, phi_pelvis, phi_trunk]  (world angles)
        "origin" -> {"shank": AJC, "thigh": KJC, "pelvis": HJC, "trunk": LJC}
    """
    # The "d_*" values are the fixed angle between neighbouring segments in the
    # standing pose (from calibration).  Adding q on top gives the current angle.
    d_sh = calib["phi_shank0"] - calib["phi_foot0"]
    d_th = calib["phi_thigh0"] - calib["phi_shank0"]
    d_pe = calib["phi_pelvis0"] - calib["phi_thigh0"]
    d_tr = calib["phi_trunk0"] - calib["phi_pelvis0"]

    # walk the chain: each segment's world angle = previous + fixed offset + its joint angle
    phi_s = foot_ang + d_sh + q[0]
    phi_t = phi_s + d_th + q[1]
    phi_p = phi_t + d_pe + q[2]
    phi_r = phi_p + d_tr + q[3]

    ajc = np.asarray(ajc, float)
    kjc = ajc + calib["L_shank"] * _e(phi_s)         # step L_shank along the shank direction
    hjc = kjc + calib["L_thigh"] * _e(phi_t)         # step L_thigh along the thigh direction
    ljc = hjc + rot(phi_p) @ calib["d_pelvis_lumbar"]  # offset (fixed in pelvis) rotated into the world
    return dict(phi=np.array([phi_s, phi_t, phi_p, phi_r]),
                origin=dict(shank=ajc, thigh=kjc, pelvis=hjc, trunk=ljc))


def predict_markers(q, calib, foot_ang, ajc):
    """Where the model says each tracked marker should be, for state `q`.

    marker = segment origin + (rotation of the segment) applied to the marker's
    fixed position within that segment.
    """
    fk = forward_kinematics(q, calib, foot_ang, ajc)
    out = {}
    for si, seg in enumerate(SEG_ORDER):            # si = 0,1,2,3   seg = name
        O, phi = fk["origin"][seg], fk["phi"][si]
        Rm = rot(phi)
        for name, p in calib["segments"][seg]["markers"].items():   # p = local (x, y)
            out[name] = O + Rm @ p
    return out


def _marker_list(calib):
    """Flatten the per-segment marker dicts into three parallel lists:
        names   -- marker name
        segs    -- which segment index (0..3) it is on
        locs    -- its fixed (x, y) position within that segment
    """
    names, segs, locals_ = [], [], []
    for si, seg in enumerate(SEG_ORDER):
        for name, p in calib["segments"][seg]["markers"].items():
            names.append(name)                     # list.append adds one item
            segs.append(si)
            locals_.append(np.asarray(p, float))
    return names, np.array(segs), np.array(locals_)


def residual_and_jacobian(q, calib, foot_ang, ajc, meas, weights, want_jac=True,
                          anchors=None):
    """The heart of the solver.

    residual r : for each marker, sqrt(weight) * (model position - measured position),
                 x and y stacked -> length 2N.
    Jacobian J : d r / d q, shape (2N, 4) -- how each residual changes when each
                 of the four joint angles changes.  Only built if want_jac is True.

    `anchors` (optional) adds a few extra residual rows that gently pull a chain
    joint centre toward an independent estimate, e.g.
        {"HJC": (harrington_point, weight)}
    """
    names, segs, locs = _marker_list(calib)
    fk = forward_kinematics(q, calib, foot_ang, ajc)
    phi = fk["phi"]                                 # the 4 world angles
    O = fk["origin"]                                # the 4 joint centres

    # ---- how each joint CENTRE moves when q changes (2x4 matrices) ---------- #
    Ls, Lt = calib["L_shank"], calib["L_thigh"]
    dpl = calib["d_pelvis_lumbar"]
    dKJC = np.zeros((2, 4)); dKJC[:, 0] = Ls * _ep(phi[0])          # KJC depends on q[0] only
    dHJC = dKJC.copy()
    dHJC[:, 1] += Lt * _ep(phi[1]); dHJC[:, 0] += Lt * _ep(phi[1])  # HJC also depends on q[1]
    dLJC = dHJC.copy()
    rp_p = _Rp(phi[2]) @ dpl                                        # how LJC's offset rotates
    dLJC[:, 0] += rp_p; dLJC[:, 1] += rp_p; dLJC[:, 2] += rp_p      # LJC also depends on q[2]
    dO = {0: np.zeros((2, 4)), 1: dKJC, 2: dHJC, 3: dLJC}          # AJC (seg 0) never moves with q

    # d(segment angle) / d q.  Segment 0 depends on q0; segment 1 on q0,q1; etc.
    # -> a lower-triangular matrix of ones.
    dphi = np.array([[1, 0, 0, 0],
                     [1, 1, 0, 0],
                     [1, 1, 1, 0],
                     [1, 1, 1, 1]], float)

    anchors = anchors or {}
    n_rows = 2 * (len(names) + len(anchors))        # 2 rows (x, y) per marker + per anchor
    r = np.zeros(n_rows)
    J = np.zeros((n_rows, 4)) if want_jac else None

    # ---- one marker at a time --------------------------------------------- #
    # zip(...) walks the three lists together; enumerate adds the counter i.
    for i, (name, si, p) in enumerate(zip(names, segs, locs)):
        w = np.sqrt(weights.get(name, 1.0))        # sqrt so that r.T @ r == sum of w * error^2
        pred = O[SEG_ORDER[si]] + rot(phi[si]) @ p  # model position of this marker
        r[2 * i:2 * i + 2] = w * (pred - meas[name])   # its x- and y-error, into rows 2i, 2i+1
        if want_jac:
            # marker moves because (a) its segment's origin moves: dO[si]
            #                  and (b) its segment rotates: _Rp @ p, scaled by how
            #                      much that segment's angle depends on each q (dphi[si]).
            # np.outer(a, b)[k, m] = a[k] * b[m]  -> a (2, 4) block.
            col = dO[si] + np.outer(_Rp(phi[si]) @ p, dphi[si])
            J[2 * i:2 * i + 2, :] = w * col

    # ---- optional anchor rows (soft joint-centre constraints) ------------- #
    seg_idx = {"KJC": 1, "HJC": 2, "LJC": 3}
    for k, (name, (target, w)) in enumerate(anchors.items()):
        row = 2 * (len(names) + k)                  # anchors go after all the markers
        si = seg_idx[name]
        sw = np.sqrt(w)
        r[row:row + 2] = sw * (O[SEG_ORDER[si]] - np.asarray(target))
        if want_jac:
            J[row:row + 2, :] = sw * dO[si]
    # return both if a Jacobian was asked for, otherwise just the residual
    return (r, J) if want_jac else r


def jacobian_fd(q, calib, foot_ang, ajc, meas, weights, eps=1e-7):
    """A second Jacobian, from finite differences -- ONLY for checking the
    analytic one above.  For each unknown k, nudge q[k] by +/- eps and see how
    the residual changes:  dr/dq_k ~= (r(q+eps) - r(q-eps)) / (2 eps).
    """
    J = np.zeros((2 * len(_marker_list(calib)[0]), 4))
    for k in range(4):
        dq = np.zeros(4); dq[k] = eps
        rp = residual_and_jacobian(q + dq, calib, foot_ang, ajc, meas, weights, want_jac=False)
        rm = residual_and_jacobian(q - dq, calib, foot_ang, ajc, meas, weights, want_jac=False)
        J[:, k] = (rp - rm) / (2 * eps)
    return J


def solve_frame(q0, calib, foot_ang, ajc, meas, weights,
                max_iter=60, tol=1e-9, lam0=1e-3, anchors=None):
    """Find the `q` that best fits this frame's markers, starting from guess `q0`.

    Levenberg-Marquardt: repeatedly solve  (J'J + lam*diag(J'J)) dq = -J'r  for a
    step dq.  Small `lam` -> fast Gauss-Newton step; large `lam` -> small, safe,
    downhill step.  We shrink `lam` after a step that lowers the cost and grow it
    after one that does not.

    Returns (q, info) where info has iters / cost / rms_m / grad_inf.
    """
    # `rj` is a shorthand: rj(x, True) -> (residual, Jacobian);  rj(x, False) -> residual.
    rj = lambda x, jac: residual_and_jacobian(x, calib, foot_ang, ajc, meas,
                                              weights, want_jac=jac, anchors=anchors)
    q = np.array(q0, float)
    r, J = rj(q, True)
    cost = 0.5 * r @ r                              # r @ r is the dot product = ||r||^2
    lam = lam0

    for it in range(max_iter):
        H = J.T @ J                                 # (4, 4) approximate Hessian
        g = J.T @ r                                 # (4,) gradient direction
        # np.linalg.norm(g, np.inf) = the largest absolute entry of g.
        if np.linalg.norm(g, np.inf) < tol:
            break                                   # already at a minimum

        step_ok = False
        for _ in range(12):                         # try increasing lam up to 12 times
            try:
                # np.diag(np.diag(H)): keep only H's diagonal (the LM damping term).
                # np.linalg.solve(A, b) solves  A x = b  for x.
                dq = np.linalg.solve(H + lam * np.diag(np.diag(H)), -g)
            except np.linalg.LinAlgError:           # matrix not solvable
                lam *= 10.0
                continue                            # try again with more damping
            q_new = q + dq
            r_new = rj(q_new, False)                # residual only (cheaper)
            cost_new = 0.5 * r_new @ r_new
            if cost_new < cost:                     # the step helped -> accept it
                q, cost = q_new, cost_new
                r, J = rj(q, True)
                lam = max(lam * 0.3, 1e-9)          # be a bit bolder next time
                step_ok = True
                break
            lam *= 10.0                             # step made things worse -> damp more
        if not step_ok or np.linalg.norm(dq) < tol:
            break                                   # cannot improve, or step is tiny

    # ---- report an honest (unweighted) marker fit error ------------------- #
    pred = predict_markers(q, calib, foot_ang, ajc)
    d = np.array([pred[n] - meas[n] for n in pred if n in meas])   # (n_markers, 2) errors
    # per marker: sqrt(dx^2 + dy^2); then average, then sqrt again -> RMS in metres
    rms = float(np.sqrt(np.mean((d ** 2).sum(axis=1)))) if len(d) else np.nan
    return q, dict(iters=it + 1, cost=cost, rms_m=rms,
                   grad_inf=float(np.linalg.norm(J.T @ r, np.inf)),
                   n_markers=len(d))


def ankle_centre(meas, calib):
    """Ankle joint centre in a movement frame.  The medial malleolus marker is
    gone, so we take the lateral one (RFAL) plus the small offset from calibration.
    """
    return meas["RFAL"] + calib["ankle_from_RFAL"]


def geometric_init(meas, calib, foot_ang):
    """A quick starting guess for `q`, read straight off the markers.

    We estimate each segment's world angle from the markers, then subtract the
    standing-pose reference so the result is 'change from standing', matching how
    `q` is defined.
    """
    from .calibration import hip_centre_harrington

    ajc = ankle_centre(meas, calib)
    kjc = meas["RFLE"] + calib["knee_from_RFLE"]
    hjc = hip_centre_harrington(meas["RIAS"], meas["LIAS"], meas["RIPS"], meas["LIPS"])
    mid_asis = 0.5 * (meas["RIAS"] + meas["LIAS"])
    mid_psis = 0.5 * (meas["RIPS"] + meas["LIPS"])

    def ang(v):                                     # a small helper, local to this function
        """Angle of the 2-D vector v."""
        return float(np.arctan2(v[1], v[0]))

    phi_s = ang(kjc - ajc)                          # ankle -> knee
    phi_t = ang(hjc - kjc)                          # knee -> hip
    phi_p = ang(mid_asis - mid_psis)               # pelvis forward
    phi_r = ang(meas["CV7"] - mid_psis) if "CV7" in meas else phi_p

    q = np.zeros(4)                                 # start with [0, 0, 0, 0]
    q[0] = (phi_s - foot_ang) - (calib["phi_shank0"] - calib["phi_foot0"])
    q[1] = (phi_t - phi_s) - (calib["phi_thigh0"] - calib["phi_shank0"])
    q[2] = (phi_p - phi_t) - (calib["phi_pelvis0"] - calib["phi_thigh0"])
    q[3] = (phi_r - phi_p) - (calib["phi_trunk0"] - calib["phi_pelvis0"])
    return q


# How hard the soft anchors pull.  Bigger -> the chain hip/knee centre is held
# closer to the pelvis-marker (Harrington) / knee-marker estimate.
ANCHOR_HJC_W = 300.0
ANCHOR_KJC_W = 60.0


def solve_trial(trc, calib, weights=None, t_range=None,
                anchor_hjc=ANCHOR_HJC_W, anchor_kjc=ANCHOR_KJC_W):
    """Solve every frame of a whole trial.

    Each frame is solved twice -- warm-started from the previous frame AND from a
    fresh geometric guess -- and the lower-cost answer is kept.  That keeps the
    hardest (deep-flexion) frames from settling into a wrong local minimum.

    Returns a dict of time series: theta_ankle/knee/hip/lumbar (rad), plus
    per-frame diagnostics (rms_cm, iters, grad_inf).
    """
    from .calibration import foot_angle, _xy, restrict_to, hip_centre_harrington

    # `A if cond else B`: use the caller's weights, or the defaults.
    weights = DEFAULT_WEIGHTS if weights is None else weights

    t = trc.time
    if t_range is not None:
        # boolean array: True for frames inside [t0, t1].  `&` = elementwise AND.
        keep = (t >= t_range[0]) & (t <= t_range[1])
    else:
        keep = np.ones(len(t), bool)                # all True
    idx = np.where(keep)[0]                         # the frame numbers to solve

    present = list(trc.data)                        # marker names available this trial
    calib = restrict_to(calib, present)            # drop calibrated markers we do not have
    need = {"RFAL", "RFLE", "RIAS", "LIAS", "RIPS", "LIPS", "RFCC", "RFM1", "RFM5", "CV7"}
    # markers we will read each frame: the "need" list plus every segment target
    use = [n for n in present if n in need or any(
        n in calib["segments"][s]["markers"] for s in SEG_ORDER)]

    Q = np.full((len(idx), 4), np.nan)              # solved angles, one row per kept frame
    diag = {k: np.zeros(len(idx)) for k in ("iters", "rms_m", "grad_inf")}
    q_prev = None
    for j, i in enumerate(idx):                     # j = output row, i = frame number
        # this frame's marker positions (x, y), dropping any that are missing
        meas = {n: _xy(trc[n])[i] for n in use}
        meas = {n: v for n, v in meas.items() if not np.any(np.isnan(v))}
        fa = foot_angle(meas)
        ajc = ankle_centre(meas, calib)

        # build the soft anchors for this frame
        anchors = {}
        if anchor_hjc and all(k in meas for k in ("RIAS", "LIAS", "RIPS", "LIPS")):
            hjc = hip_centre_harrington(meas["RIAS"], meas["LIAS"], meas["RIPS"], meas["LIPS"])
            anchors["HJC"] = (hjc, anchor_hjc)
        if anchor_kjc and "RFLE" in meas:
            anchors["KJC"] = (meas["RFLE"] + calib["knee_from_RFLE"], anchor_kjc)

        # solve from the fresh guess, and (if we have one) from the previous frame
        cand = [solve_frame(geometric_init(meas, calib, fa), calib, fa, ajc, meas,
                            weights, anchors=anchors)]
        if q_prev is not None:
            cand.append(solve_frame(q_prev, calib, fa, ajc, meas, weights, anchors=anchors))
        # min(list, key=f) returns the item with the smallest f(item).
        # ci is (q, info);  ci[1]["cost"] is that solution's cost.
        q, info = min(cand, key=lambda ci: ci[1]["cost"])

        q_prev = q
        Q[j] = q
        for k in diag:
            diag[k][j] = info[k]

    return dict(
        time=t[idx],
        theta_ankle=Q[:, 0], theta_knee=Q[:, 1],   # column 0, 1, ... of the solved angles
        theta_hip=Q[:, 2], theta_lumbar=Q[:, 3],
        rms_cm=diag["rms_m"] * 100.0,               # metres -> centimetres
        iters=diag["iters"], grad_inf=diag["grad_inf"],
    )

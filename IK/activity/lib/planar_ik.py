"""Barehand inverse kinematics: a 4-DOF planar linkage solved per frame by
weighted nonlinear least squares (Levenberg-Marquardt).

>>> THREE things are left for you to implement -- search for "TODO". <<<
    1. forward_kinematics()          -- the FK chain
    2. residual_and_jacobian()       -- the analytic Jacobian dr/dq
    3. solve_frame()                 -- the Levenberg-Marquardt loop
Everything else (helpers, initial guess, per-trial driver) is provided and
commented.

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

----------------------------------------------------------------------------
Maths <-> code cheat sheet:
    R(phi)              rot(phi)        2-D rotation matrix
    d/dphi [cos, sin]   _ep(phi)        == [-sin phi, cos phi]
    d/dphi R(phi)       _Rp(phi)        == R(phi + 90 deg)
    A @ B               matrix / matrix-vector multiply
    x @ x               dot product ( = ||x||^2 )
    J.T                 transpose
    d phi_seg / d q  is lower-triangular ones (see `dphi` below)
Python notes:
    lambda a, b: expr   a tiny inline function
    r[2*i : 2*i+2]      rows 2i and 2i+1 (this marker's x- and y-error)
    try: ... except E:  run the first block, jump to the second on error
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
    """Derivative of rot(phi) with respect to phi (equals rot(phi + 90 deg))."""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[-s, -c],
                     [c, -s]])


# The four segments, ground up.  Their index (0..3) matches `q`.
SEG_ORDER = ["shank", "thigh", "pelvis", "trunk"]

# How much to trust each marker (w_i in the cost).  Bony landmarks move less
# relative to the bone than skin-mounted wands, so they get bigger weights.
DEFAULT_WEIGHTS = {
    "RIAS": 10.0, "LIAS": 10.0, "RIPS": 10.0, "LIPS": 10.0,
    "RFLE": 6.0, "CV7": 6.0, "SJN": 4.0,
    "RTH": 1.0, "RFTC": 2.0, "RSK": 1.0, "RTTC": 1.0, "RFAX": 1.0,
    "SXS": 3.0, "TV2": 3.0, "TV7": 3.0,
}


def forward_kinematics(q, calib, foot_ang, ajc):
    """Given the four joint angles `q`, work out where every segment is.

    Return a dict:
        {"phi":    np.array([phi_shank, phi_thigh, phi_pelvis, phi_trunk]),
         "origin": {"shank": AJC, "thigh": KJC, "pelvis": HJC, "trunk": LJC}}
    """
    # d_* = the fixed angle between neighbouring segments in the standing pose.
    d_sh = calib["phi_shank0"] - calib["phi_foot0"]
    d_th = calib["phi_thigh0"] - calib["phi_shank0"]
    d_pe = calib["phi_pelvis0"] - calib["phi_thigh0"]
    d_tr = calib["phi_trunk0"] - calib["phi_pelvis0"]
    ajc = np.asarray(ajc, float)

    # TODO: build phi_s, phi_t, phi_p, phi_r by walking the chain
    #   (each = previous world angle + the d_* offset + the matching q entry),
    # then step along each segment to KJC, HJC, LJC.
    #   KJC = ajc + calib["L_shank"] * _e(phi_s)
    #   HJC = KJC + calib["L_thigh"] * _e(phi_t)
    #   LJC = HJC + rot(phi_p) @ calib["d_pelvis_lumbar"]
    # Finally return the dict shown in the docstring.
    raise NotImplementedError("forward_kinematics")


def predict_markers(q, calib, foot_ang, ajc):
    """Where the model says each tracked marker should be, for state `q`.

    marker = segment origin + (segment rotation) applied to the marker's fixed
    position within that segment.
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
        names -- marker name
        segs  -- which segment index (0..3) it is on
        locs  -- its fixed (x, y) within that segment
    """
    names, segs, locals_ = [], [], []
    for si, seg in enumerate(SEG_ORDER):
        for name, p in calib["segments"][seg]["markers"].items():
            names.append(name)
            segs.append(si)
            locals_.append(np.asarray(p, float))
    return names, np.array(segs), np.array(locals_)


def residual_and_jacobian(q, calib, foot_ang, ajc, meas, weights, want_jac=True,
                          anchors=None):
    """residual r : for each marker, sqrt(weight) * (model position - measured),
                    x and y stacked -> length 2N.
       Jacobian J : d r / d q, shape (2N, 4).  Only built if want_jac is True.

    `anchors` (optional) adds a few extra residual rows pulling a chain joint
    centre toward an independent estimate, e.g. {"HJC": (harrington_point, weight)}.
    """
    names, segs, locs = _marker_list(calib)
    fk = forward_kinematics(q, calib, foot_ang, ajc)
    phi = fk["phi"]                                 # the 4 world angles
    O = fk["origin"]                                # the 4 joint centres

    Ls, Lt = calib["L_shank"], calib["L_thigh"]
    dpl = calib["d_pelvis_lumbar"]
    # d(segment angle) / d q -- segment 0 depends on q0; segment 1 on q0,q1; etc.
    dphi = np.array([[1, 0, 0, 0],
                     [1, 1, 0, 0],
                     [1, 1, 1, 0],
                     [1, 1, 1, 1]], float)

    # TODO (only needed when want_jac is True): build the 2x4 position derivatives
    # of each joint centre w.r.t. q:
    #   dKJC/dq : column 0 = Ls * _ep(phi[0]);  other columns 0
    #   dHJC/dq : dKJC + Lt * _ep(phi[1]) in columns 0 and 1
    #   dLJC/dq : dHJC + (_Rp(phi[2]) @ dpl) in columns 0, 1 and 2
    #   dO = {0: zeros(2,4), 1: dKJC, 2: dHJC, 3: dLJC}     (AJC never moves with q)
    # Then, for a marker on segment `si` with local coords `p`, its 2x4 block is
    #   dO[si] + np.outer(_Rp(phi[si]) @ p, dphi[si])
    dO = {0: np.zeros((2, 4)), 1: np.zeros((2, 4)), 2: np.zeros((2, 4)), 3: np.zeros((2, 4))}
    if want_jac:
        raise NotImplementedError("residual_and_jacobian: Jacobian")

    anchors = anchors or {}
    n_rows = 2 * (len(names) + len(anchors))        # 2 rows (x, y) per marker + per anchor
    r = np.zeros(n_rows)
    J = np.zeros((n_rows, 4)) if want_jac else None
    for i, (name, si, p) in enumerate(zip(names, segs, locs)):
        w = np.sqrt(weights.get(name, 1.0))        # sqrt so r.T @ r == sum w*error^2
        pred = O[SEG_ORDER[si]] + rot(phi[si]) @ p  # model position of this marker
        r[2 * i:2 * i + 2] = w * (pred - meas[name])   # its x- and y-error
        if want_jac:
            col = dO[si] + np.outer(_Rp(phi[si]) @ p, dphi[si])
            J[2 * i:2 * i + 2, :] = w * col

    # optional anchor rows (soft joint-centre constraints), added after the markers
    seg_idx = {"KJC": 1, "HJC": 2, "LJC": 3}
    for k, (name, (target, w)) in enumerate(anchors.items()):
        row = 2 * (len(names) + k)
        si = seg_idx[name]
        sw = np.sqrt(w)
        r[row:row + 2] = sw * (O[SEG_ORDER[si]] - np.asarray(target))
        if want_jac:
            J[row:row + 2, :] = sw * dO[si]
    return (r, J) if want_jac else r


def jacobian_fd(q, calib, foot_ang, ajc, meas, weights, eps=1e-7):
    """A finite-difference Jacobian -- use this to CHECK your analytic one.
    For each unknown k:  dr/dq_k ~= (r(q + eps) - r(q - eps)) / (2 eps).
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
    """Find the `q` that best fits this frame's markers, starting from `q0`.

    Levenberg-Marquardt loop:
        r, J = residual_and_jacobian(q)
        H = J.T @ J ;  g = J.T @ r
        solve  (H + lam * diag(diag(H))) dq = -g
        if cost(q + dq) < cost(q):  accept q,  shrink lam  (be bolder)
        else:                       reject,    grow lam    (be more careful)
        stop when ||g||_inf is tiny, or a step barely changes q
    Return (q, info) where info has iters / cost / rms_m / grad_inf.

    Handy pieces:
        np.linalg.solve(A, b)      solves A x = b
        np.diag(np.diag(H))        H with its off-diagonal set to 0
        np.linalg.norm(g, np.inf)  the largest absolute entry of g
        try: ... except np.linalg.LinAlgError: ...   (matrix not solvable)
    """
    # rj(x, True) -> (residual, Jacobian);   rj(x, False) -> residual only
    rj = lambda x, jac: residual_and_jacobian(x, calib, foot_ang, ajc, meas,
                                              weights, want_jac=jac, anchors=anchors)
    q = np.array(q0, float)
    it = 0

    # TODO: run the LM loop here -- update `q`, and count `it` iterations.
    raise NotImplementedError("solve_frame")

    # ---- provided: once `q` has converged, report an honest marker RMS ---- #
    pred = predict_markers(q, calib, foot_ang, ajc)                    # noqa: E999
    d = np.array([pred[n] - meas[n] for n in pred if n in meas])
    rms = float(np.sqrt(np.mean((d ** 2).sum(axis=1)))) if len(d) else np.nan
    r_final = rj(q, False)
    return q, dict(iters=it + 1, cost=0.5 * r_final @ r_final, rms_m=rms,
                   grad_inf=np.nan, n_markers=len(d))


def ankle_centre(meas, calib):
    """Ankle joint centre in a movement frame.  The medial malleolus marker is
    gone, so we take the lateral one (RFAL) plus the offset from calibration.
    """
    return meas["RFAL"] + calib["ankle_from_RFAL"]


def geometric_init(meas, calib, foot_ang):
    """A quick starting guess for `q`, read straight off the markers (provided).

    Estimate each segment's world angle from markers, then subtract the standing
    reference so the result is 'change from standing', matching how `q` is defined.
    """
    from .calibration import hip_centre_harrington

    ajc = ankle_centre(meas, calib)
    kjc = meas["RFLE"] + calib["knee_from_RFLE"]
    hjc = hip_centre_harrington(meas["RIAS"], meas["LIAS"], meas["RIPS"], meas["LIPS"])
    mid_asis = 0.5 * (meas["RIAS"] + meas["LIAS"])
    mid_psis = 0.5 * (meas["RIPS"] + meas["LIPS"])

    def ang(v):                                     # small helper, local to this function
        """Angle of the 2-D vector v."""
        return float(np.arctan2(v[1], v[0]))

    phi_s = ang(kjc - ajc)                          # ankle -> knee
    phi_t = ang(hjc - kjc)                          # knee -> hip
    phi_p = ang(mid_asis - mid_psis)               # pelvis forward
    phi_r = ang(meas["CV7"] - mid_psis) if "CV7" in meas else phi_p

    q = np.zeros(4)                                 # start from [0, 0, 0, 0]
    q[0] = (phi_s - foot_ang) - (calib["phi_shank0"] - calib["phi_foot0"])
    q[1] = (phi_t - phi_s) - (calib["phi_thigh0"] - calib["phi_shank0"])
    q[2] = (phi_p - phi_t) - (calib["phi_pelvis0"] - calib["phi_thigh0"])
    q[3] = (phi_r - phi_p) - (calib["phi_trunk0"] - calib["phi_pelvis0"])
    return q


# How hard the soft anchors pull.  Bigger -> chain hip/knee centre held closer to
# the pelvis-marker (Harrington) / knee-marker estimate.
ANCHOR_HJC_W = 300.0
ANCHOR_KJC_W = 60.0


def solve_trial(trc, calib, weights=None, t_range=None,
                anchor_hjc=ANCHOR_HJC_W, anchor_kjc=ANCHOR_KJC_W):
    """Solve every frame of a whole trial (provided -- it calls YOUR solve_frame).

    Each frame is solved twice -- warm-started from the previous frame AND from a
    fresh geometric guess -- and the lower-cost answer is kept.
    Returns a dict of time series: theta_ankle/knee/hip/lumbar (rad) + diagnostics.
    """
    from .calibration import foot_angle, _xy, restrict_to, hip_centre_harrington

    weights = DEFAULT_WEIGHTS if weights is None else weights
    t = trc.time
    # boolean array: True for frames inside [t0, t1] (or all True if no range given)
    keep = np.ones(len(t), bool) if t_range is None else (t >= t_range[0]) & (t <= t_range[1])
    idx = np.where(keep)[0]                         # the frame numbers to solve

    present = list(trc.data)                        # marker names available this trial
    calib = restrict_to(calib, present)            # drop calibrated markers we do not have
    need = {"RFAL", "RFLE", "RIAS", "LIAS", "RIPS", "LIPS", "RFCC", "RFM1", "RFM5", "CV7"}
    use = [n for n in present if n in need or any(
        n in calib["segments"][s]["markers"] for s in SEG_ORDER)]

    Q = np.full((len(idx), 4), np.nan)              # solved angles, one row per kept frame
    diag = {k: np.zeros(len(idx)) for k in ("iters", "rms_m", "grad_inf")}
    q_prev = None
    for j, i in enumerate(idx):                     # j = output row, i = frame number
        meas = {n: _xy(trc[n])[i] for n in use}
        meas = {n: v for n, v in meas.items() if not np.any(np.isnan(v))}   # drop missing
        fa = foot_angle(meas)
        ajc = ankle_centre(meas, calib)

        anchors = {}
        if anchor_hjc and all(k in meas for k in ("RIAS", "LIAS", "RIPS", "LIPS")):
            anchors["HJC"] = (hip_centre_harrington(meas["RIAS"], meas["LIAS"],
                                                    meas["RIPS"], meas["LIPS"]), anchor_hjc)
        if anchor_kjc and "RFLE" in meas:
            anchors["KJC"] = (meas["RFLE"] + calib["knee_from_RFLE"], anchor_kjc)

        cand = [solve_frame(geometric_init(meas, calib, fa), calib, fa, ajc, meas,
                            weights, anchors=anchors)]
        if q_prev is not None:
            cand.append(solve_frame(q_prev, calib, fa, ajc, meas, weights, anchors=anchors))
        # min(list, key=f) -> the item with the smallest f;  ci is (q, info)
        q, info = min(cand, key=lambda ci: ci[1]["cost"])

        q_prev = q
        Q[j] = q
        for k in diag:
            diag[k][j] = info[k]

    return dict(time=t[idx],
                theta_ankle=Q[:, 0], theta_knee=Q[:, 1],
                theta_hip=Q[:, 2], theta_lumbar=Q[:, 3],
                rms_cm=diag["rms_m"] * 100.0,       # metres -> centimetres
                iters=diag["iters"], grad_inf=diag["grad_inf"])

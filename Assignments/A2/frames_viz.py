"""Draw 3-D vectors and coordinate frames with matplotlib.

The Python version of the MATLAB `PlotVectors.m` helper.  Use it to sanity-check
the anatomical frames you build in `calculate_knee_angle.py` -- if an axis points
the wrong way, you will see it here.

Run this file directly for a small demo:

    python frames_viz.py
"""

from __future__ import annotations

import numpy as np


def plot_vector(ax, origin, vec, color, label=None):
    """Draw the arrow ``vec`` starting at ``origin`` on a 3-D axes ``ax``.

    ``origin`` and ``vec`` are length-3 sequences (x, y, z).  matplotlib's
    ``quiver`` in 3-D takes the tail (x, y, z) and the components (u, v, w).
    """
    o = np.asarray(origin, float)
    v = np.asarray(vec, float)
    ax.quiver(o[0], o[1], o[2], v[0], v[1], v[2], color=color, arrow_length_ratio=0.15)
    if label:
        tip = o + v
        ax.text(tip[0], tip[1], tip[2], label, color=color, fontsize=8)


def draw_frame(ax, origin, R, scale=0.1, label=""):
    """Draw a coordinate frame: its three basis vectors as red / green / blue
    arrows rooted at ``origin``.

    ``R`` is a 3x3 rotation matrix whose **columns** are the frame's x, y, z axes
    expressed in the world frame (exactly what you assemble in step 2).
    """
    origin = np.asarray(origin, float)
    for k, color in enumerate(("red", "green", "blue")):
        axis_name = "xyz"[k]
        plot_vector(ax, origin, scale * R[:, k], color,
                    label=f"{label}{axis_name}" if label else None)


def new_axes(title=""):
    """Open a fresh 3-D figure with equal aspect and return its axes."""
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_box_aspect((1, 1, 1))                    # equal-ish aspect ratio
    if title:
        ax.set_title(title)
    return ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ax = new_axes("frames_viz demo: two coordinate frames")
    a = np.array([0.2, 0.2, 0.2])                   # a sample offset
    draw_frame(ax, [0, 0, 0], np.eye(3), scale=0.1, label="world ")
    # a frame rotated 90 deg about world-x, placed at the tip of `a`
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    draw_frame(ax, a, R, scale=0.1, label="B ")
    plot_vector(ax, [0, 0, 0], a, "black")
    plt.show()

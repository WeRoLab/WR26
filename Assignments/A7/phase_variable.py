"""Phase variable from a thigh-mounted IMU -- Assignment 7 starter code.

A person with a transfemoral (above-knee) amputation walks on a treadmill
with a powered knee-ankle prosthesis (Elery et al., 2020).  The prosthesis
logs its own sensors ("Leg2Data"): a thigh IMU, knee and ankle encoders and
motor torques, a load cell, and controller parameters.

The data has already been cut into 31 strides (heel strike to heel strike)
and each stride resampled to 101 samples, so sample k is at k % of the gait
cycle.  Every signal is therefore a NumPy array of shape (31, 101):

    signal[i, :]  ->  stride i, from 0 % to 100 % of the gait cycle
    signal[:, k]  ->  all 31 strides at k % of the gait cycle

This is the transpose of the MATLAB layout (101 x 31), so per-stride
operations here work along axis=1, e.g. signal.max(axis=1).

Given:  load_leg2data, print_sensors, plot_sensor.
You:    TODO 1-3 in main() (see the assignment statement).

Requires: numpy, scipy, matplotlib.
Input:    data/TF01.mat -- run get_data.py first.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent      # folder of this script (finds data/)
MAT_PATH = HERE / "data" / "TF01.mat"

GAIT = np.linspace(0, 100, 101)             # % of the gait cycle, one per sample


def load_leg2data(path=MAT_PATH):
    """Return the prosthesis sensors as {group: {signal: array (31, 101)}}.

    Example: data["Knee"]["Angle"] is the knee angle of every stride.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} not found -- run get_data.py first")
    # squeeze_me + struct_as_record=False turn MATLAB structs into objects
    # whose fields are attributes (TF01.Leg2Data.Knee.Angle)
    tf01 = loadmat(path, squeeze_me=True, struct_as_record=False)["TF01"]
    leg2 = tf01.Leg2Data
    return {group: {name: np.asarray(getattr(getattr(leg2, group), name), dtype=float)
                    for name in getattr(leg2, group)._fieldnames}
            for group in leg2._fieldnames}


def print_sensors(data):
    """List every signal with its range (min, max), group by group."""
    width = max(len(f"{g}.{n}") for g, signals in data.items() for n in signals)
    for group, signals in data.items():
        print(f"{group}:")
        for name, values in signals.items():
            key = f"{group}.{name}"
            print(f"    {key:<{width}s}  min {values.min():10.3f}   max {values.max():10.3f}")


def plot_sensor(data, key, ylabel=None):
    """Plot one signal ("Group.Signal") for all strides vs. % gait cycle."""
    group, name = key.split(".")
    values = data[group][name]
    fig, ax = plt.subplots()
    ax.plot(GAIT, values.T, color="k", linewidth=0.5)   # .T: one line per stride
    ax.set_title(key)
    ax.set_xlabel("Gait cycle [%]")
    ax.set_ylabel(ylabel if ylabel else key)
    ax.grid(True)
    return fig, ax


def main():
    data = load_leg2data()
    n_strides = data["Time"]["Time"].shape[0]
    print(f"{n_strides} strides, each with {len(GAIT)} samples (0-100 % of the gait cycle)\n")
    print_sensors(data)

    # Example: knee angle of every stride.  Use plot_sensor the same way to
    # look at the other signals and compare them with normative gait data.
    plot_sensor(data, "Knee.Angle")

    # ------------------------------------------------------------------
    # TODO 1: pick the signals that measure the thigh angle and the thigh
    # angular velocity, e.g. thigh = data["Group"]["Signal"].
    # What are their units?
    # ------------------------------------------------------------------
    thigh = None
    thighd = None

    if thigh is None or thighd is None:
        print("\nTODO 1: set thigh and thighd, then rerun.")
        plt.show()
        return

    # ------------------------------------------------------------------
    # TODO 2: plot the phase portrait, thigh angle (x-axis) vs. thigh
    # velocity (y-axis), and compute the polar angle of each sample with
    # np.arctan2 (np.degrees converts radians to degrees).
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # TODO 3: find offsets a (angle) and b (velocity) that center each
    # stride's ellipse on the origin, and a scale z for the thigh angle that
    # makes it as close as possible to a circle.  Then compute
    #     phase = np.degrees(np.arctan2(thighd_c, -thigh_c))
    # with the centered/scaled signals, normalize it so that it starts at 0
    # and spans 1 within each stride, and plot it vs. GAIT.
    # ------------------------------------------------------------------

    plt.show()


if __name__ == "__main__":
    main()

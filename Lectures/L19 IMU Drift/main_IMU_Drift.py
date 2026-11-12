"""IMU drift: orientation of a tablet lying still on a table
Python version of main_IMU_Drift.m.

The tablet (and the IMU inside it) lay stationary on a table for the whole
trial (~16.5 min), while MATLAB Mobile logged its orientation at ~10 Hz:

    Azimuth -- rotation about the vertical (heading, like a compass)
    Pitch   -- tilt about the tablet's short axis
    Roll    -- tilt about the tablet's long axis

Axes: Z is aligned with the gravity vector (positive pointing towards the
earth); X points along the long axis of the tablet.

Because the tablet never moved, all three angles *should* be constant.
Any slow change you see in the plots is drift: the orientation is estimated
by integrating the gyroscope's angular velocity, so its small bias and noise
accumulate over time into an angle error.

Requires: numpy, scipy, matplotlib.
Input:    DataCollectionDrift_arrays.mat, made from the raw MATLAB Mobile
          recording (DataCollectionDrift.mat, a MATLAB timetable that Python
          can't read) by convert_drift_data.m.
Output:   a figure with azimuth, pitch and roll vs. time.
Try:      estimate the drift rate of each angle in deg/min (e.g. a straight-
          line fit with np.polyfit). Which angle drifts most? Pitch and roll
          can be corrected with the accelerometer (gravity always points
          down), but azimuth can only be corrected with the magnetometer
          (magnetic north), which is much less reliable indoors.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent      # folder of this script (finds the .mat)


def main():
    # Load Data -- loadmat returns a dictionary: {variable name: array}
    data = loadmat(HERE / "DataCollectionDrift_arrays.mat")

    # MATLAB stores every vector as a 2-D matrix (here 10017 x 1), so
    # .ravel() flattens each one into a 1-D NumPy array
    Azimuth = data["X"].ravel()             # [deg]
    Pitch = data["Y"].ravel()               # [deg]
    Roll = data["Z"].ravel()                # [deg]
    time = data["time"].ravel()             # [s] since the first sample
    start_time = str(data["start_time"][0]) # clock time of the first sample (text)

    print(f"{len(time)} samples over {time[-1]:.1f} s, starting at {start_time}")

    # Plot Data -- three stacked plots that share the same time axis
    fig, axes = plt.subplots(3, 1, sharex=True)
    fig.suptitle("Tablet Orientation")
    for ax, angle, name in zip(axes, [Azimuth, Pitch, Roll], ["Azimuth", "Pitch", "Roll"]):
        ax.plot(time, angle, "--o", markerfacecolor="none")  # dashed line, hollow circles (MATLAB '--o')
        ax.set_ylabel(f"{name} [deg]")
        ax.grid(True)
    axes[-1].set_xlabel(f"Time [s] (since {start_time})")
    plt.show()


if __name__ == "__main__":
    main()

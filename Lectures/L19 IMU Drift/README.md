# L19 IMU Drift: orientation of a tablet lying still (MATLAB and Python)

Lecture 19 demo for **Wearable Robotics (WR26)**. A tablet lay still on a table
for about 16.5 minutes while [MATLAB Mobile](https://www.mathworks.com/products/matlab-mobile.html)
logged the orientation estimated by its IMU (inertial measurement unit) at about
10 Hz. The tablet never moved, so all three angles *should* stay constant. Any
slow change you see in the plots is **drift**.

Drift comes from how orientation is estimated. The gyroscope measures angular
velocity, and integrating it gives the angle. Every gyroscope has a small bias
and noise, and integration adds these errors up over time, so the angle error
keeps growing. Wearable robots that use IMUs to estimate joint angles have the
same problem.

```
Lectures/L19 IMU Drift/
├── main_IMU_Drift.m / .py          plots azimuth, pitch and roll vs. time (start here)
├── DataCollectionDrift_arrays.mat  the data as plain arrays (read by both scripts)
├── convert_drift_data.m            makes the arrays file from the raw recording
└── DataCollectionDrift.mat         raw MATLAB Mobile recording (a MATLAB timetable)
```

## The data

`DataCollectionDrift_arrays.mat` holds 10017 samples:

| variable | units | meaning |
|---|---|---|
| `time` | s | time since the first sample (0 to 995.3 s) |
| `X` | deg | azimuth: rotation about the vertical, like a compass heading |
| `Y` | deg | pitch: tilt of the tablet |
| `Z` | deg | roll: tilt of the tablet |
| `start_time` | text | clock time of the first sample (2023-11-15 09:04:49.245) |

Axes: Z is aligned with the gravity vector (positive pointing towards the
earth), and X points along the long axis of the tablet.

### Why there are two `.mat` files

MATLAB Mobile saves its logs as a **timetable** with **datetime** time stamps.
These are MATLAB objects, and Python's `scipy.io.loadmat` cannot read them.
`convert_drift_data.m` loads the raw file (`DataCollectionDrift.mat`) and saves
the same numbers as plain arrays in `DataCollectionDrift_arrays.mat`, which
both MATLAB and Python can read. The arrays file is already included, so you
don't need to run the script. To run it on new data recorded with MATLAB
Mobile, change the file name in the script. The script won't overwrite an
existing `DataCollectionDrift_arrays.mat`; delete that file first if you want
to regenerate it.

## Running

**Python.** Use the class venv (`WR26`); you only need numpy, scipy and
matplotlib. The script finds the data next to itself, so it runs from any
folder:

```
python "Lectures/L19 IMU Drift/main_IMU_Drift.py"
```

**MATLAB.** `cd` into this folder first (or open the script and press
**Run**, which offers to change folder), then run `main_IMU_Drift`.

Both versions use the same variable names and draw the same figure: three
stacked plots of azimuth, pitch and roll vs. time.

## What to look at

- **Azimuth** climbs steadily, by about 1° over the trial.
- **Pitch** and **roll** only wander by a few hundredths of a degree around
  a constant value.

Why the difference? Pitch and roll can be corrected with the accelerometer:
when the tablet is still, it measures gravity, which always points down.
Azimuth is a rotation *about* gravity, so the accelerometer can't see it. It
can only be corrected with the magnetometer (magnetic north), which is much
less reliable indoors.

**Try:** estimate the drift rate of each angle in deg/min with a straight-line
fit (`polyfit` in MATLAB, `np.polyfit` in Python). At that rate, how large would
the azimuth error be after a one-hour session with a wearable robot?

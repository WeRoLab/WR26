# A7: Phase variable from a thigh-mounted IMU

A person with an above-knee (transfemoral) amputation walks on a treadmill
with a powered knee-ankle prosthesis. You use the prosthesis's own sensors to
build a *phase variable*: a single number that tells the controller where the
leg is in the gait cycle. You compute it from the thigh angle and the thigh
angular velocity. Full instructions are in the assignment statement
(Canvas / Gradescope); this folder is the code.

| file | what it does | you write |
|---|---|---|
| `get_data.py` | given: downloads `data/TF01.mat` (~5 MB) from Google Drive | -- |
| `phase_variable.py` | given: `load_leg2data()` loads the prosthesis sensors, `print_sensors()` lists them with their ranges, `plot_sensor()` plots one of them for all strides vs. % gait cycle | TODO 1: pick the thigh angle and velocity signals; TODO 2: phase portrait and polar angle; TODO 3: offsets, scale, and normalized phase |

## Setup

We recommend running the script in a Python virtual environment. A virtual
environment keeps the packages for this course separate from the rest of your
computer, so installing them cannot break other projects. If you already set up
the class environment, `WR26`, for earlier assignments, reuse it.

1. Create the environment. Skip this step if `WR26` already exists.

   ```bash
   # Windows (Command Prompt)
   python -m venv %USERPROFILE%\.venvs\WR26
   # Windows (PowerShell), macOS / Linux
   python -m venv $HOME/.venvs/WR26
   ```

2. Activate it. Do this in every new terminal before running the scripts.

   ```bash
   # Windows (Command Prompt)
   %USERPROFILE%\.venvs\WR26\Scripts\activate
   # Windows (PowerShell)
   & $HOME\.venvs\WR26\Scripts\Activate.ps1
   # macOS / Linux
   source $HOME/.venvs/WR26/bin/activate
   ```

   Your prompt should now start with `(WR26)`.

3. Install the packages. Run this from the repository root.

   ```bash
   pip install -r Assignments/A7/requirements.txt
   ```

4. Download the data (once). It is not stored in git.

   ```bash
   python Assignments/A7/get_data.py
   ```

   This creates `Assignments/A7/data/TF01.mat`. If the download fails, open
   the [Google Drive link](https://drive.google.com/file/d/1MMMY3uGZe18yfAnt3ipIgBH-Tq0_JqPR/view?usp=sharing)
   in a browser and save the file as `Assignments/A7/data/TF01.mat`.

## Run

From the repository root (the script also runs from any other folder):

```bash
python Assignments/A7/phase_variable.py
```

Unedited, the script lists every sensor with its range, plots the knee angle
of all strides as an example, and stops at TODO 1. Use `plot_sensor` to look
at other signals, then complete the TODOs in `main()`.

## Notes on the data

- The data is already cut into **31 strides** (heel strike to heel strike),
  and each stride is resampled to **101 samples**, so sample `k` is at `k` %
  of the gait cycle. `GAIT = np.linspace(0, 100, 101)` is that axis.
- Every signal is a NumPy array of shape `(31, 101)`: one row per stride.
  This is the transpose of the MATLAB layout, so per-stride operations work
  along `axis=1` (e.g. `signal.max(axis=1, keepdims=True)`).
- The signals come from the prosthesis's controller. The file does not state
  their units, so **check the units of each signal** before you use it (for
  instance, compare a signal with the time derivative of another one, using
  `data["Time"]["Time"]`, in seconds).
- `np.arctan2(y, x)` is the four-quadrant arctangent (MATLAB's `atan2`); it
  returns radians in (-pi, pi]. `np.degrees` converts them to degrees
  (MATLAB's `atan2d(y, x)` is `np.degrees(np.arctan2(y, x))`).

## Data source and license

`TF01.mat` is an extract (one subject, level-ground walking) of the dataset of

> T. Elery, S. Rezazadeh, E. Reznick, L. Gray, and R. D. Gregg, "Effects of a
> Powered Knee-Ankle Prosthesis on Amputee Hip Compensations: A Case Series,"
> IEEE DataPort, 2020.

The **complete dataset** (all subjects and conditions) is available on IEEE
DataPort:
<https://ieee-dataport.org/documents/effects-powered-knee-ankle-prosthesis-amputee-hip-compensations-case-series>.
It is distributed under the
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
license.

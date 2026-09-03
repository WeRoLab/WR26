# Environment

All scripts in this module run with the **`WR26` virtual environment**:

```
C:\Users\edgar\.venvs\WR26\Scripts\python.exe
```

That interpreter is already first on `PATH`, so plain `python` works.

## Installed packages (verified 2026-08-30)

| package     | version | used for |
|-------------|---------|----------|
| opensim     | 4.6     | C3D reading, Scale/IK/ID tools, model processing |
| numpy       | 2.5.2   | everything |
| scipy       | 1.18.1  | Butterworth filtering, `optimize.least_squares` cross-check |
| matplotlib  | 3.11.1  | figures |
| gdown       | 6.1.0   | `get_data.py` only — fetch the data zip from Google Drive |

`pip install -r requirements.txt` installs all of the above.  No `pandas`, no
Jupyter — not needed.  OpenSim reads C3D natively, so `ezc3d` / `c3d` are **not**
required.

## Get the data

`Data/`, `OpenSim/` and `Python Documentation/` are **not** in the repo
(a ~190 MB download, ~330 MB unpacked).  Fetch them once:

```
python get_data.py
```

It downloads a single zip from Google Drive and unpacks the three folders next to
`get_data.py`.  If Drive is unreachable, the script prints a link to download the
zip manually.  `s0` / `s1` will remind you to run this if the data is missing.

## Running

```
cd Lectures/IK/solution
python s0_prepare_data.py      # raw C3D  -> TRC + GRF .mot + events        (GIVEN)
python s1_prepare_model.py     # scaled model -> sagittal-locked model      (GIVEN)
python s2_opensim_ik.py        # OpenSim InverseKinematicsTool              (student)
python s3_planar_ik.py         # barehand 4-DOF nonlinear-least-squares IK  (student)
python s4_compare.py           # compare the two tracks                     (student)
```

Outputs are written under `solution/results/<trial>/`.

## Data provenance

- `Data/AB03/C3D/*.c3d` — OptiTrack Motive exports, subject AB03. Markers 200 Hz,
  force plates 1000 Hz. Lab frame is **Z-up**; force plates 1 (right foot) and 2
  (left foot) are the only active plates.
- `Data/AB03/Static/1/` — one trial already processed with the lab's MATLAB
  `trial_class` pipeline. We reuse:
  - `generated/Generated_Model.osim` — scaled AB03 model (total mass 60.01 kg).
  - `Data/static.trc`, `Data/static.mot` — reference outputs used to validate `s0`.
- `../230825 Sit to stand/` — the 2023 first version of this activity (different
  subject, AB01). Reference only.

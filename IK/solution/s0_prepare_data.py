"""s0 -- raw C3D  ->  OpenSim-ready inputs   [GIVEN to students]

For every trial in ``Data/AB03/C3D/`` this writes, under ``results/<trial>/``:

    <trial>.trc            markers, rotated to the OpenSim frame + low-pass filtered
    <trial>_grf.mot        ground reactions from the two portable force plates
    <trial>_extloads.xml   ExternalLoads config pointing at the .mot
    <trial>_events.json    seat-off / start / end times
    <trial>_events.png     quick-look plot

Students normally start from the .trc file produced here; the code is provided so
the C3D-handling and the lab-frame -> OpenSim-frame rotation are not a black box.

Run:  python s0_prepare_data.py

------------------------------------------------------------------------------
New to Python?  A few things you will see below:
  * `import X` / `from X import y`  -- load code written elsewhere.
  * `def name(args):`               -- define a function; call it later as name(...).
  * text in triple quotes just under `def`  -- the function's help string.
  * a dict  {"a": 1, "b": 2}        -- a labelled bag of values; d["a"] reads one.
  * an f-string  f"x = {value}"     -- build text with values plugged in.
  * `for item in things:`           -- do something once per item.
  * `if __name__ == "__main__":`    -- "only run this when the file is executed
                                       directly, not when it is imported".
------------------------------------------------------------------------------
"""

# --------------------------------------------------------------------------- #
# IMPORTS -- bring in code from the standard library, from NumPy, and from our #
# own `lib/` folder so we can use it here.                                     #
# --------------------------------------------------------------------------- #

# Makes the ": type" / "-> type" hints in this file lazy. Harmless; safe to ignore.
from __future__ import annotations

import json         # read and write .json text files
import pathlib      # build and inspect file paths in an OS-independent way

import numpy as np  # arrays and math;  "as np" lets us write np.something()

# Our own helper modules (they live in the sibling folder ``lib/``):
from lib import c3d_prep, events   # c3d_prep: read + rotate + filter a C3D file
                                   # events:   find seat-off / movement start / end
from lib.io import read_trc        # load a .trc marker file into memory
from lib.paths import project_root  # find the data folder (or say how to get it)


# --------------------------------------------------------------------------- #
# PATHS -- work out where the input data and the output folder live.          #
# These lines run once, as soon as the file is loaded.  ALL-CAPS names are a  #
# convention meaning "constant: set here, never changed".                     #
# --------------------------------------------------------------------------- #

# __file__ is the path of THIS script.  .resolve() makes it absolute (removes
# any "..").  .parent is the folder that holds it -> ".../IK Learning Module/solution".
HERE = pathlib.Path(__file__).resolve().parent

# The project root = the folder that has "Data/AB03" in it.  project_root() walks
# up to find it, or exits with "run get_data.py" if the data was never downloaded.
ROOT = project_root()

# The "/" between Path objects joins path pieces (like a slash in a file path).
C3D_DIR    = ROOT / "Data" / "AB03" / "C3D"                          # raw capture files
STATIC_C3D = ROOT / "Data" / "AB03" / "Static" / "1" / "Data" / "static.c3d"
STATIC_TRC = ROOT / "Data" / "AB03" / "Static" / "1" / "Data" / "static.trc"
RESULTS    = HERE / "results"                                        # where output goes


# --------------------------------------------------------------------------- #
# SETTINGS -- numbers you might want to change, kept together in one place.    #
# --------------------------------------------------------------------------- #
SUBJECT_MASS_KG  = 60.01   # subject AB03; from Data/AB03/Static/1/Setup/Setup_Scale.xml
MARKER_CUTOFF_HZ = 6.0     # low-pass filter cutoff for the marker trajectories (Hz)
FORCE_CUTOFF_HZ  = 15.0    # low-pass filter cutoff for the force-plate signals (Hz)


def validate_rotation():
    """Self-check: our lab-frame -> OpenSim-frame rotation must reproduce the
    lab's own ``static.trc`` file.  If this check fails, every later step is
    suspect, so we run it first.
    """
    # Read + rotate the static C3D ourselves.  Arguments are passed BY NAME
    # (marker_cutoff_hz=..., align_yaw=...) so the intent is readable:
    #   marker_cutoff_hz=None -> do NOT filter, so we compare the pure rotation
    #   align_yaw=False       -> do NOT spin the trial, to match the raw file
    prep = c3d_prep.prepare_trial(STATIC_C3D, marker_cutoff_hz=None,
                                  force_cutoff_hz=None, align_yaw=False)

    # Load the reference file that the lab already produced.
    ref = read_trc(STATIC_TRC)

    worst = 0.0                    # biggest disagreement seen so far (metres)
    for nm in ref.marker_names:    # loop over every marker name in the reference
        if nm in prep["markers"]:  # (skip any name missing from our version)
            # Each marker is an (N, 3) NumPy array: N time frames by (x, y, z).
            # np.nan_to_num replaces "missing value" (NaN) with 0 so we can subtract.
            a = np.nan_to_num(prep["markers"][nm])   # our result
            b = np.nan_to_num(ref[nm])               # the lab's result
            # a - b subtracts element by element; np.abs makes it positive;
            # np.nanmax picks the single largest number in the whole array;
            # max(worst, ...) keeps the running maximum.
            worst = max(worst, float(np.nanmax(np.abs(a - b))))

    # "A if condition else B" is a one-line choice between two values.
    status = "OK" if worst < 1e-6 else "FAIL"
    # In an f-string, {worst:.2e} prints worst in scientific notation, 2 decimals.
    print(f"[validate] rotation vs provided static.trc: max diff {worst:.2e} m  [{status}]")
    return worst < 1e-6           # give a True/False answer back to the caller


def process(c3d_path):
    """Turn ONE movement-trial C3D file into all of its outputs and print a
    one-line summary.  ``c3d_path`` is a Path pointing at a ``.c3d`` file.
    """
    # Read the C3D and apply the rotation + low-pass filtering.
    # `prep` is a dict: prep["trial"] (a name string), prep["markers"] (marker
    # arrays), prep["plates"] (force-plate data), prep["marker_time"], ...
    prep = c3d_prep.prepare_trial(c3d_path,
                                  marker_cutoff_hz=MARKER_CUTOFF_HZ,
                                  force_cutoff_hz=FORCE_CUTOFF_HZ)

    # Create this trial's output folder, e.g. results/sit_to_stand_1/.
    #   parents=True  -> also create any missing parent folders
    #   exist_ok=True -> it is fine if the folder already exists
    out = RESULTS / prep["trial"]
    out.mkdir(parents=True, exist_ok=True)

    # Build the output file paths.  f"{prep['trial']}.trc" becomes e.g.
    # "sit_to_stand_1.trc"; the leading `out /` puts it inside the folder.
    trc_path = out / f"{prep['trial']}.trc"
    grf_path = out / f"{prep['trial']}_grf.mot"
    xml_path = out / f"{prep['trial']}_extloads.xml"

    # Pass `prep` to each writer function; each one saves one file to disk.
    c3d_prep.write_trc(prep, trc_path)                          # marker positions
    c3d_prep.write_grf_mot(prep, grf_path)                      # ground reaction forces
    c3d_prep.write_external_loads_xml(prep, grf_path, xml_path)  # OpenSim loads config

    # Find movement events (seat-off etc.), save a plot, and save the numbers.
    ev = events.detect_events(prep, SUBJECT_MASS_KG)
    events.plot_events(prep, ev, out / f"{prep['trial']}_events.png")
    # Dict comprehension: build a new dict from `ev`, keeping every key/value pair
    # EXCEPT the private ones (keys beginning with "_", which hold bulky arrays).
    ev_json = {k: v for k, v in ev.items() if not k.startswith("_")}
    # json.dumps turns the dict into text; .write_text saves that text to a file.
    (out / f"{prep['trial']}_events.json").write_text(json.dumps(ev_json, indent=2))

    # ---- print one progress line ------------------------------------------- #
    # marker_time is a 1-D array of time stamps; [-1] is the last, [0] the first.
    dur = prep["marker_time"][-1] - prep["marker_time"][0]
    # Looping over the dict prep["plates"] yields its keys (the plate numbers).
    # " ".join(...) glues a list of strings together; `... or "-"` falls back to
    # "-" when the joined string is empty (no active plates).
    plates = ",".join(str(p) for p in prep["plates"]) or "-"
    # .shape[0] = number of rows of the RFCC marker array = number of frames.
    # Format codes inside {}:  :18s = pad text to 18 chars,  :4d = integer 4 wide,
    #                          :5.2f = float 5 wide with 2 decimals.
    print(f"  {prep['trial']:18s}  {prep['markers']['RFCC'].shape[0]:4d} frames  "
          f"{dur:5.2f}s  plates[{plates}]  "
          f"event={ev.get('t_event', float('nan')):.3f}s")
    return prep["trial"]


def process_static():
    """Like process(), but for the standing calibration trial.  Here we KEEP the
    yaw alignment (align_yaw=True) and skip force handling.  Step s3 reads this
    file to calibrate the barehand model.
    """
    prep = c3d_prep.prepare_trial(STATIC_C3D, marker_cutoff_hz=MARKER_CUTOFF_HZ,
                                  force_cutoff_hz=None, align_yaw=True)
    out = RESULTS / "static"
    out.mkdir(parents=True, exist_ok=True)
    c3d_prep.write_trc(prep, out / "static.trc")
    print(f"  {'static':18s}  {prep['markers']['RFCC'].shape[0]:4d} frames "
          f"(calibration trial)")


def main():
    """Top-level routine: do everything, in order."""
    print("=" * 70)          # "=" * 70 is a string of seventy "=" characters
    print("s0  raw C3D -> OpenSim inputs")
    print("=" * 70)

    validate_rotation()      # 1. self-check

    print("\nprocessing trials -> results/<trial>/ :")   # "\n" starts a new line
    process_static()         # 2. the calibration trial

    # C3D_DIR.glob("*.c3d") lists every file whose name ends in ".c3d";
    # sorted() puts them in alphabetical order so runs are reproducible.
    c3ds = sorted(C3D_DIR.glob("*.c3d"))
    for c3d in c3ds:         # 3. handle the movement trials one at a time
        process(c3d)

    # len(c3ds) is how many files were found.
    print(f"\ndone: {len(c3ds)} trials + static written under {RESULTS}")


# When you run  `python s0_prepare_data.py`  directly, Python sets this file's
# __name__ to "__main__", so the line below calls main().  When another script
# does  `import s0_prepare_data`,  __name__ is "s0_prepare_data" instead and
# main() is NOT called automatically.
if __name__ == "__main__":
    main()

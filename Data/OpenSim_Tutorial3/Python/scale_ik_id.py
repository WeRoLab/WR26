"""Reproduce OpenSim Tutorial 3 (Scaling, Inverse Kinematics, Inverse
Dynamics) using the OpenSim Python API.

This is exactly what you would do by hand in the OpenSim GUI, following
Tutorial 3: load a "setup" XML file for each step (Scale, then Inverse
Kinematics, then Inverse Dynamics) and run it. The setup files already live
in the `Setup/` folder next to this script, with every marker weight and
scale factor already filled in -- you do not need to edit them.

Running this script writes the scaled model, the IK motion, and the ID
moments into `Output/`, plus two verification plots (built in the
"Analysis" section below). A later assignment script
(`Assignments/A3/calculate_knee_torque.py`) reads those `Output/` files to
get the data it needs for a "by hand" (first-principles) knee torque
calculation -- run this script first.

----------------------------------------------------------------------------
New to Python? Things you will meet in this file:
  * `from pathlib import Path` gives us a `Path` object for file locations.
    `Path(__file__)` is the path to THIS script; `.resolve()` makes it an
    absolute path; `.parent` is the folder that contains it. Joining paths
    uses `/`, e.g. `SETUP_DIR / "subject01_Setup_IK.xml"`.
  * OpenSim wants file paths as plain text, so we wrap every `Path` in
    `str(...)` before handing it to an OpenSim function.
  * `os.chdir(some_folder)` changes which folder Python (and OpenSim) treat
    as "here" for the rest of the program -- see the comment below for why
    we need this.
  * A triple-quoted string like this one, at the very top of a file, is a
    "docstring" -- a comment that documents the whole file.
----------------------------------------------------------------------------
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import opensim as osim

# --------------------------------------------------------------------------- #
# Where everything lives
# --------------------------------------------------------------------------- #

# __file__ is the path to this script. HERE is the "Python" folder it lives
# in; TUTORIAL_DIR is the "OpenSim_Tutorial3" folder one level up.
HERE = Path(__file__).resolve().parent
TUTORIAL_DIR = HERE.parent
SETUP_DIR = TUTORIAL_DIR / "Setup"
DATA_DIR = TUTORIAL_DIR / "Data"
OUTPUT_DIR = TUTORIAL_DIR / "Output"

# OpenSim will not create a missing output folder for us, so make sure it
# exists before any tool tries to write into it.
OUTPUT_DIR.mkdir(exist_ok=True)

# The gait2354 model references body-shape files (.vtp) purely for 3D display
# -- they are not needed to compute scaling/IK/ID, but OpenSim prints a
# warning for each one it cannot find. This tells it where to look so those
# warnings go away.
osim.ModelVisualizer.addDirToGeometrySearchPaths(str(TUTORIAL_DIR / "Geometry"))

# The setup XML files in Setup/ use paths like "../Data/subject01_static.trc"
# and "../Output/subject01_simbody.osim". Those are relative to the Setup/
# folder itself -- the same folder you would be standing in if you ran these
# tools from the OpenSim GUI or command line. Changing into that folder here
# makes the Python API behave the same way.
os.chdir(SETUP_DIR)

# --------------------------------------------------------------------------- #
# STEP 1: Scaling -- fit the generic gait2354 model to subject01
# --------------------------------------------------------------------------- #

# Build the ScaleTool directly from its setup file. That XML file already
# lists which markers to use, how much to trust each one, and which body
# segments each measurement resizes -- we do not need to repeat any of that
# in Python. Running it writes the scaled model to Output/subject01_simbody.osim.
scale_tool = osim.ScaleTool("subject01_Setup_Scale.xml")
scale_tool.run()

# --------------------------------------------------------------------------- #
# STEP 2: Inverse Kinematics -- solve joint angles for the walking trial
# --------------------------------------------------------------------------- #

ik_tool = osim.InverseKinematicsTool("subject01_Setup_IK.xml")

# Ask the tool to also write out a marker-error report, and to save it (and
# the motion file) into Output/ rather than Setup/.
ik_tool.set_report_errors(True)
ik_tool.set_report_marker_locations(False)
ik_tool.setResultsDir(str(OUTPUT_DIR))

ik_tool.run()

# --------------------------------------------------------------------------- #
# STEP 3: Inverse Dynamics -- solve joint moments from motion + ground forces
# --------------------------------------------------------------------------- #

id_tool = osim.InverseDynamicsTool("subject01_Setup_InverseDynamics.xml")
id_tool.run()

# --------------------------------------------------------------------------- #
# ANALYSIS -- everything below is postprocessing of the three steps above:
# reading back what got written, printing a few numbers to eyeball, and
# plotting the two verification figures.
# --------------------------------------------------------------------------- #

print("=" * 70)
print("OpenSim Tutorial 3: Scaling -> Inverse Kinematics -> Inverse Dynamics")
print("=" * 70)

# --- Scaling: confirm the scaled model was written -------------------------- #

scaled_model_file = OUTPUT_DIR / "subject01_simbody.osim"
print("\nScaling")
print("  subject mass (kg):", scale_tool.getSubjectMass())
print("  subject height (mm):", scale_tool.getSubjectHeight())
print("  wrote:", scaled_model_file)

# --- Inverse Kinematics: read the marker errors and plot them --------------- #

ik_mot_file = OUTPUT_DIR / "subject01_walk1_ik.mot"
# Find the marker-error file IK just wrote. Its exact name starts with the
# tool's name (e.g. "subject01_ik_marker_errors.sto"), so we search for it
# by pattern instead of hard-coding the name.
marker_errors_file = next(OUTPUT_DIR.glob("*_marker_errors.sto"))

error_table = osim.TableProcessor(str(marker_errors_file)).process()
marker_error_rms = error_table.getDependentColumn("marker_error_RMS").to_numpy()
marker_error_max = error_table.getDependentColumn("marker_error_max").to_numpy()

print("\nInverse Kinematics")
print("  wrote:", ik_mot_file)
print(f"  marker_error_RMS: mean {marker_error_rms.mean():.4f} m, max {marker_error_rms.max():.4f} m")
print(f"  marker_error_max: mean {marker_error_max.mean():.4f} m, max {marker_error_max.max():.4f} m")

error_time = error_table.getIndependentColumn()
total_squared_error = error_table.getDependentColumn("total_squared_error").to_numpy()

fig, axs = plt.subplots(1, 1, figsize=(7, 5))
fig.suptitle("Marker Errors from Inverse Kinematics", fontsize=16)
axs.plot(error_time, total_squared_error, label="total_squared_error")
axs.plot(error_time, marker_error_rms, label="marker_error_RMS")
axs.plot(error_time, marker_error_max, label="marker_error_max")
axs.set_xlabel("Time (s)")
axs.set_ylabel("Marker Error (m)")
axs.grid()
axs.legend()
fig.savefig(OUTPUT_DIR / "marker_errors.png")
plt.close(fig)
print("  wrote:", OUTPUT_DIR / "marker_errors.png")

# --- Inverse Dynamics: read the joint moments and plot the knee moments ----- #

id_sto_file = OUTPUT_DIR / "inverse_dynamics.sto"
id_table = osim.TimeSeriesTable(str(id_sto_file))

print("\nInverse Dynamics")
print("  wrote:", id_sto_file)
print("  columns:", id_table.getColumnLabels())

id_time = id_table.getIndependentColumn()
knee_moment_r = id_table.getDependentColumn("knee_angle_r_moment").to_numpy()
knee_moment_l = id_table.getDependentColumn("knee_angle_l_moment").to_numpy()

fig, axs = plt.subplots(1, 1, figsize=(7, 5))
fig.suptitle("Inverse Dynamics: Knee Moments", fontsize=16)
axs.plot(id_time, knee_moment_r, label="knee_angle_r_moment")
axs.plot(id_time, knee_moment_l, label="knee_angle_l_moment")
axs.set_xlabel("Time (s)")
axs.set_ylabel("Moment (N-m)")
axs.grid()
axs.legend()
fig.savefig(OUTPUT_DIR / "inverse_dynamics_knee_moments.png")
plt.close(fig)
print("  wrote:", OUTPUT_DIR / "inverse_dynamics_knee_moments.png")

print("\nAll done. Files written to:", OUTPUT_DIR)
"""Small helpers for reading motion-capture files with the OpenSim API.

This module is GIVEN to you complete -- you do not need to edit it, but do read
it: three of the functions are one-line wrappers around OpenSim API calls you
will want to recognise.

    read_c3d_markers(path)   ->  (markers, time)     uses opensim.C3DFileAdapter
    to_opensim_frame(v)      ->  rotated array       the lab -> OpenSim rotation
    write_trc(path, ...)     ->  writes a .trc file  (what InverseKinematicsTool reads)
    read_storage(path)       ->  Storage object      reads a .mot / .sto results file

----------------------------------------------------------------------------
New to Python / NumPy?
  * a NumPy array is a grid of numbers.  One marker trajectory is shape (N, 3):
    N rows (time frames), 3 columns (x, y, z).
  * `A @ B`      -- matrix multiplication.
  * `arr[:, 0]`  -- "every row, column 0"  (here: the x value at every frame).
  * `{k: f(v) for k, v in d.items()}` -- build a new dict from an old one.
  * `class Foo:` defines a kind of object; `__init__` runs when you write
    `Foo(...)`; `self` is the object being built.  `obj["knee_angle_r"]` calls
    the class's `__getitem__`.
----------------------------------------------------------------------------
"""

from __future__ import annotations

import pathlib

import numpy as np
import opensim as osim


# --------------------------------------------------------------------------- #
# 0.  Where is the data?
# --------------------------------------------------------------------------- #

def find_data_dir() -> pathlib.Path:
    """Return the ``data/`` folder produced by ``get_data.py`` (it sits next to
    this file).  Exit with a helpful message if it is not there yet.
    """
    data = pathlib.Path(__file__).resolve().parent / "data"
    if (data / "stride_1.c3d").is_file():
        return data
    raise SystemExit(
        "Could not find the data.  Run  python get_data.py  in  Assignments/A2/  first."
    )


# --------------------------------------------------------------------------- #
# 1.  Reading a C3D file with OpenSim
# --------------------------------------------------------------------------- #

def _vec3_table_to_dict(table):
    """Turn an OpenSim ``TimeSeriesTableVec3`` into plain NumPy.

    OpenSim returns marker data as a table of 3-vectors (``Vec3``).  We copy it
    into a dict  {"RFCC": array of shape (n_frames, 3), ...}  plus a 1-D array of
    time stamps -- much easier to work with downstream.
    """
    labels = list(table.getColumnLabels())                 # e.g. ["RFCC", "RIAS", ...]
    n = table.getNumRows()
    time = np.array(table.getIndependentColumn(), float)    # the time-stamp column

    markers = {}
    for j, name in enumerate(labels):
        xyz = np.empty((n, 3))
        for i in range(n):
            v = table.getRowAtIndex(i).getElt(0, j)         # the Vec3 at frame i, column j
            xyz[i] = (v.get(0), v.get(1), v.get(2))         # its x, y, z parts
        markers[name] = xyz
    return markers, time


def read_c3d_markers(path):
    """Read the marker trajectories from a .c3d file.

    Returns ``(markers, time)`` where ``markers`` is a dict  name -> (n_frames, 3)
    array in **metres**, and ``time`` is a 1-D array of time stamps in seconds.
    Coordinates are left in the raw **lab frame** (this dataset: X and Y
    horizontal, Z up).  Use :func:`to_opensim_frame` when you need the OpenSim
    convention.

    A dropped marker sample is returned as ``NaN`` (not a number).
    """
    adapter = osim.C3DFileAdapter()
    # We do not use the force plates in this assignment, but the reader still
    # needs to be told how to express them; "CentreOfPressure" is the usual pick.
    adapter.setLocationForForceExpression(
        osim.C3DFileAdapter.ForceLocation_CenterOfPressure)
    tables = adapter.read(str(path))                        # <-- reads the file

    markers, time = _vec3_table_to_dict(adapter.getMarkersTable(tables))

    # The C3D "Units" metadata says mm or m.  Convert to metres if needed.
    mtable = adapter.getMarkersTable(tables)
    if (mtable.hasTableMetaDataKey("Units")
            and mtable.getTableMetaDataAsString("Units").lower() in ("mm", "millimeter")):
        markers = {name: xyz * 1e-3 for name, xyz in markers.items()}

    # OpenSim marks a missing marker sample with an exact 0.0 -- turn those into
    # NaN so they are obvious and do not corrupt any averaging.
    for name, xyz in markers.items():
        xyz[np.all(xyz == 0.0, axis=1)] = np.nan
    return markers, time


# --------------------------------------------------------------------------- #
# 2.  The lab -> OpenSim coordinate rotation
# --------------------------------------------------------------------------- #

# OpenSim models are built with Y up.  This motion-capture lab records with Z up
# (X, Y horizontal).  Turning the lab frame by -90 deg about its X axis lines the
# two up:  [x, y, z]_lab  ->  [x, z, -y]_opensim .  Applied as  v @ LAB_TO_OSIM .
LAB_TO_OSIM = np.array([[1.0, 0.0,  0.0],
                        [0.0, 0.0, -1.0],
                        [0.0, 1.0,  0.0]])


def to_opensim_frame(v):
    """Rotate a point or an (N, 3) array of points from the lab frame into the
    OpenSim frame ( [x, y, z] -> [x, z, -y] )."""
    return np.asarray(v, float) @ LAB_TO_OSIM


# --------------------------------------------------------------------------- #
# 3.  Writing a .trc marker file  (the format InverseKinematicsTool reads)
# --------------------------------------------------------------------------- #

def write_trc(path, time, markers, units="m"):
    """Write markers to a .trc text file.

    ``markers`` is a dict  name -> (n_frames, 3) array.  A NaN sample is written
    as 0.0, the convention OpenSim expects for "marker not seen this frame".
    """
    time = np.asarray(time, float)
    names = list(markers)
    n_frames = len(time)
    rate = 1.0 / np.median(np.diff(time)) if n_frames > 1 else 1.0
    fname = str(path).replace("\\", "/").split("/")[-1]

    with open(path, "w", newline="") as fh:
        fh.write(f"PathFileType\t4\t(X/Y/Z)\t{fname}\n")
        fh.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
                 "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        fh.write(f"{rate:.6f}\t{rate:.6f}\t{n_frames}\t{len(names)}\t{units}\t"
                 f"{rate:.6f}\t1\t{n_frames}\n")

        header = "Frame#\tTime" + "".join(f"\t{nm}\t\t" for nm in names)
        sub = "\t" + "".join(f"\tX{i + 1}\tY{i + 1}\tZ{i + 1}" for i in range(len(names)))
        fh.write(header + "\n" + sub + "\n\n")

        for f in range(n_frames):
            row = [str(f + 1), f"{time[f]:.6f}"]
            for nm in names:
                x, y, z = markers[nm][f]
                if np.isnan(x):
                    row += ["0.000000", "0.000000", "0.000000"]
                else:
                    row += [f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"]
            fh.write("\t".join(row) + "\n")


# --------------------------------------------------------------------------- #
# 4.  Reading a .mot / .sto results file
# --------------------------------------------------------------------------- #

class Storage:
    """A parsed .mot / .sto table: a time column plus one column per signal.

    After ``s = read_storage("stride_1_ik.mot")``:
        s.time                    -> 1-D array of time stamps
        s.column_names            -> list of names, "time" first
        s["knee_angle_r"]         -> that column as a 1-D array
        s.in_degrees              -> True if the file's angles are in degrees
    """

    def __init__(self, column_names, data, header):
        self.column_names = list(column_names)
        self.data = np.asarray(data, float)
        self.header = dict(header)

    @property
    def time(self):
        return self.data[:, 0]

    @property
    def in_degrees(self) -> bool:
        return str(self.header.get("inDegrees", "no")).lower() == "yes"

    def __getitem__(self, name):
        return self.data[:, self.column_names.index(name)]


def read_storage(path) -> Storage:
    """Read a .mot / .sto file into a :class:`Storage` object."""
    with open(path, "r") as fh:
        lines = fh.read().splitlines()

    header, i = {}, 0
    while i < len(lines) and lines[i].strip().lower() != "endheader":
        if "=" in lines[i]:
            k, v = lines[i].split("=", 1)
            header[k.strip()] = v.strip()
        i += 1
    i += 1                                          # step past "endheader"
    column_names = lines[i].split("\t")
    rows = [[float(x) for x in ln.split()] for ln in lines[i + 1:] if ln.strip()]
    return Storage(column_names, np.array(rows, float), header)

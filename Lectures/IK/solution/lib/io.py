"""Plain-text readers/writers for OpenSim marker (.trc) and storage (.mot/.sto) files.

The barehand inverse-kinematics track (``s3_planar_ik.py``) deliberately does not
import ``opensim`` -- it works only from the numbers in the .trc file.  These
helpers keep that track self-contained and make the file formats explicit for
students.

TRC layout (tab separated)::

    PathFileType   4   (X/Y/Z)   <name>.trc
    DataRate CameraRate NumFrames NumMarkers Units OrigDataRate OrigDataStartFrame OrigNumFrames
    <rate>   <rate>     <n>       <m>         m     <rate>       <start>            <n>
    Frame#  Time  M1      M2      ...            <- one marker name per XYZ triple
            ...   X1 Y1 Z1 X2 Y2 Z2 ...
    <blank line>
    1  0.000  <x> <y> <z>  <x> <y> <z> ...

MOT/STO layout::

    <optional name line>
    key=value          (nRows, nColumns, inDegrees, ...)
    ...
    endheader
    time  col1  col2 ...
    <numbers>

----------------------------------------------------------------------------
New to Python?
  * `class Foo:` defines a new kind of object.  `__init__` is its "constructor",
    run when you write `Foo(...)`.  `self` is the object being built/used.
  * `@property` lets `obj.n_frames` look like an attribute but run a function.
  * `__getitem__` is what Python calls for `obj["RFCC"]` (square brackets).
  * `with open(path) as fh:` opens a file and always closes it afterwards.
  * a "list comprehension"  `[f(x) for x in xs if cond]`  builds a list in one line.
----------------------------------------------------------------------------
"""

# `from __future__ import annotations` makes the ": type" hints lazy; ignore it.
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- TRC


class TRC:
    """One motion-capture trial held in memory (markers over time).

    After `trc = read_trc("x.trc")` you get:
        trc.time          -> 1-D array of time stamps, length n_frames
        trc.marker_names  -> list of marker names, e.g. ["RFCC", "RIAS", ...]
        trc.data["RFCC"]  -> (n_frames, 3) array of that marker's x/y/z
                             (NaN in a row means the marker was not seen then)
        trc.rate, trc.units
    """

    def __init__(self, time, marker_names, data, rate, units="m"):
        # `self.x = y` stores y on this object so other methods can use it later.
        self.time = np.asarray(time, float)      # make sure it is a float array
        self.marker_names = list(marker_names)
        self.data = data
        self.rate = float(rate)
        self.units = units

    # convenience -----------------------------------------------------------
    @property
    def n_frames(self) -> int:
        """Number of time frames.  Used as `trc.n_frames` (no parentheses)."""
        return len(self.time)

    def __getitem__(self, name) -> np.ndarray:
        """Makes `trc["RFCC"]` return that marker's (n_frames, 3) array."""
        return self.data[name]

    def stack(self, names) -> np.ndarray:
        """Return an (n_frames, len(names), 3) array for the requested markers.

        np.stack glues several same-shaped arrays into one bigger array.
        """
        return np.stack([self.data[n] for n in names], axis=1)


def read_trc(path) -> TRC:
    """Read a .trc text file into a TRC object."""
    with open(path, "r") as fh:
        lines = fh.read().splitlines()          # whole file -> list of lines (no "\n")

    # Line index 1 holds the metadata field NAMES, line 2 the VALUES (tab-separated).
    meta_keys = lines[1].split("\t")
    meta_vals = lines[2].split("\t")
    # zip pairs them up; dict(...) turns pairs into {name: value}.
    meta = dict(zip(meta_keys, meta_vals))
    rate = float(meta["DataRate"])
    units = meta.get("Units", "m")              # .get(key, default) -> "m" if missing

    # Line 3: "Frame#", "Time", then each marker name followed by two blank cells.
    name_row = lines[3].split("\t")
    # keep the entries from position 2 onward that are not blank
    marker_names = [tok for tok in name_row[2:] if tok.strip() != ""]

    # The numbers start after the "X1 Y1 Z1 ..." row and one blank line.
    start = 5
    while start < len(lines) and lines[start].strip() == "":
        start += 1                              # skip any extra blank lines

    rows = []
    for ln in lines[start:]:                    # every remaining line
        if ln.strip() == "":
            continue                            # skip blanks
        # split on tabs, drop empty pieces, turn each into a float
        rows.append([float(x) for x in ln.replace(",", ".").split("\t") if x.strip() != ""])
    arr = np.array(rows, float)                 # list of lists -> 2-D array

    time = arr[:, 1]                            # column 1 = the Time column
    data = {}
    for i, nm in enumerate(marker_names):       # i = 0, 1, 2, ...  nm = marker name
        c = 2 + 3 * i                           # this marker's x column: 2, 5, 8, ...
        xyz = arr[:, c:c + 3].copy()            # columns c, c+1, c+2  ->  (n_frames, 3)
        # OpenSim writes an exact 0 when a marker was dropped; mark those as NaN.
        xyz[xyz == 0.0] = np.nan
        data[nm] = xyz
    return TRC(time, marker_names, data, rate, units)


def write_trc(path, time, data, marker_names=None, units="m"):
    """Write a .trc file.  ``data`` is a dict  name -> (n_frames, 3) array."""
    time = np.asarray(time, float)
    if marker_names is None:                    # caller did not say -> use every key
        marker_names = list(data.keys())
    n_frames = len(time)
    n_markers = len(marker_names)
    # sampling rate = 1 / (typical gap between time stamps)
    rate = 1.0 / np.median(np.diff(time)) if n_frames > 1 else 1.0

    # take just the file name from the path (text between the last "/" and the end)
    name = str(path).replace("\\", "/").split("/")[-1]
    with open(path, "w", newline="") as fh:     # open for writing
        # fh.write(text) appends text to the file.  "\t" = tab, "\n" = newline.
        fh.write(f"PathFileType\t4\t(X/Y/Z)\t{name}\n")
        fh.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
                 "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        fh.write(f"{rate:.6f}\t{rate:.6f}\t{n_frames}\t{n_markers}\t{units}\t"
                 f"{rate:.6f}\t1\t{n_frames}\n")

        # header row: "Frame#  Time  RFCC (blank) (blank)  RIAS (blank) (blank) ..."
        header = "Frame#\tTime"
        for nm in marker_names:
            header += f"\t{nm}\t\t"             # += appends to the string
        fh.write(header + "\n")

        # sub-header row: "  X1 Y1 Z1  X2 Y2 Z2 ..."
        sub = "\t"
        for i in range(n_markers):
            sub += f"\tX{i + 1}\tY{i + 1}\tZ{i + 1}"
        fh.write(sub + "\n\n")                  # note: extra "\n" leaves a blank line

        for f in range(n_frames):               # one output line per frame
            row = [str(f + 1), f"{time[f]:.6f}"]     # frame number, then time
            for nm in marker_names:
                x, y, z = data[nm][f]               # unpack this frame's x, y, z
                if np.isnan(x):                     # marker missing this frame
                    row += ["0.000000", "0.000000", "0.000000"]
                else:
                    row += [f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"]   # 6 decimal places
            fh.write("\t".join(row) + "\n")         # join the pieces with tabs


# ----------------------------------------------------------------------- MOT/STO


class Storage:
    """A parsed .mot / .sto table (time plus one column per signal)."""

    def __init__(self, names, data, header):
        self.column_names = list(names)          # includes "time" at index 0
        self.data = np.asarray(data, float)      # 2-D array: (n_rows, n_cols)
        self.header = dict(header)               # the key=value lines above endheader

    @property
    def time(self) -> np.ndarray:
        return self.data[:, 0]                   # the first column

    @property
    def in_degrees(self) -> bool:
        """True if the file says its angle columns are in degrees."""
        return str(self.header.get("inDegrees", "no")).lower() == "yes"

    def __getitem__(self, name) -> np.ndarray:
        """`storage["hip_flexion_r"]` -> that column as a 1-D array."""
        # .index(name) finds which column number has that name.
        return self.data[:, self.column_names.index(name)]

    def columns(self, names) -> np.ndarray:
        """Several columns at once, as an (n_rows, len(names)) array."""
        idx = [self.column_names.index(n) for n in names]
        return self.data[:, idx]


def read_mot(path) -> Storage:
    """Read a .mot / .sto text file into a Storage object."""
    with open(path, "r") as fh:
        lines = fh.read().splitlines()

    header = {}
    i = 0
    # walk down the lines until we reach the word "endheader"
    while i < len(lines) and lines[i].strip().lower() != "endheader":
        if "=" in lines[i]:                      # a "key=value" line
            k, v = lines[i].split("=", 1)        # split once, at the first "="
            header[k.strip()] = v.strip()
        i += 1
    i += 1                                       # step past the "endheader" line
    col_names = lines[i].split("\t")             # the column-name row
    i += 1

    rows = []
    for ln in lines[i:]:                         # the numeric rows
        if ln.strip() == "":
            continue
        rows.append([float(x) for x in ln.split()])   # split on any whitespace
    return Storage(col_names, np.array(rows, float), header)


# .sto and .mot have the same layout, so re-use the same reader under a 2nd name.
read_sto = read_mot


def write_mot(path, time, columns, name="data", in_degrees=None):
    """Write a .mot file.  ``columns`` is a dict  col_name -> (n_rows,) array."""
    time = np.asarray(time, float)
    names = list(columns.keys())
    # np.column_stack puts 1-D arrays side by side as columns of a 2-D array.
    # [time] + [ ... ] joins two lists: the time column first, then the rest.
    mat = np.column_stack([time] + [np.asarray(columns[n], float) for n in names])
    n_rows, n_cols = mat.shape                   # .shape is (rows, columns)

    with open(path, "w", newline="") as fh:
        fh.write(f"{name}\n")
        fh.write(f"version=1\n")
        fh.write(f"nRows={n_rows}\n")
        fh.write(f"nColumns={n_cols}\n")
        if in_degrees is not None:
            # inline if/else:  ("yes" if X else "no")
            fh.write(f"inDegrees={'yes' if in_degrees else 'no'}\n")
        fh.write("endheader\n")
        fh.write("\t".join(["time"] + names) + "\n")     # the column-name row
        for r in range(n_rows):
            # format every number in row r to 8 decimals, join with tabs
            fh.write("\t".join(f"{v:.8f}" for v in mat[r]) + "\n")

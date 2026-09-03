"""Fetch the data files that some scripts in this folder need to run.

Most scripts here build their models in code and need nothing extra (see
`README.md`).  A handful - the 2D / 3D walking examples, EMG tracking, the
polynomial path fitter - open data files by name that are **not** committed to
this repo.  This script downloads them from the pinned OpenSim source release so
the versions match the installed `opensim` wheel.

    pip install -r ../requirements.txt      # (for `opensim`; this script needs only the stdlib)
    python get_example_data.py              # ~one source tarball, then copies data into place
    python get_example_data.py --force      # re-download / overwrite

If GitHub is unreachable, download the tarball named below by hand and re-run
with  --archive <path-to.tar.gz>.
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

TAG = "4.6"  # matches `pip install opensim==4.6`
TARBALL_URL = f"https://github.com/opensim-org/opensim-core/archive/refs/tags/{TAG}.tar.gz"

HERE = Path(__file__).resolve().parent
RESOURCES = HERE.parent / "resources"

DATA_EXT = {".osim", ".sto", ".mot", ".trc", ".xml", ".c3d", ".csv"}

# Copy every data file from <archive source dir>  ->  <local dir under examples/>.
# Paths are relative to the repo root inside the tarball.
DIR_MAP = {
    "OpenSim/Examples/Moco/example2DWalking": ["Moco/example2DWalking"],
    "OpenSim/Examples/Moco/example3DWalking": ["Moco/example3DWalking"],
}

# Copy only the named files from <archive source dir> -> <local dir>.
FILE_MAP = {
    "OpenSim/Examples/Moco/example3DWalking": (
        ["subject_walk_scaled.osim", "coordinates.sto"], "PolynomialPathFitter"),
}

# Files to locate anywhere in the archive by basename -> local dir(s).
BY_NAME = {
    "gait10dof18musc.osim": ["Moco/example2DWalking"],
    "subject_walk_armless_18musc.osim": ["Moco/exampleEMGTracking"],
    "emg.sto": ["Moco/exampleEMGTracking"],
    "coordinates.mot": ["Moco/exampleEMGTracking"],
    "external_loads.xml": ["Moco/exampleEMGTracking"],
}

# Files we can satisfy from this repo's own bundled resources.
FROM_RESOURCES = {
    RESOURCES / "Tutorial 8" / "squatToStand_3dof9musc.osim": [
        "Moco/exampleSquatToStand",
        "Moco/exampleSquatToStand/exampleIMUTracking",
    ],
}


def _copy(data: bytes, name: str, rel_dirs: list[str], done: set) -> None:
    for rel in rel_dirs:
        dest_dir = HERE / rel
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / name).write_bytes(data)
        done.add(f"{rel}/{name}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    ap.add_argument("--archive", type=Path, help="use a local opensim-core-4.6.tar.gz")
    args = ap.parse_args(argv)

    if args.archive:
        if not args.archive.is_file():
            print(f"no such file: {args.archive}", file=sys.stderr)
            return 1
        raw = args.archive.read_bytes()
    else:
        print(f"downloading {TARBALL_URL}")
        try:
            with urllib.request.urlopen(TARBALL_URL) as r:  # noqa: S310 (trusted host)
                raw = r.read()
        except Exception as e:  # pragma: no cover
            print(f"\ncould not download: {e}\n"
                  f"Download it by hand and re-run:  "
                  f"python get_example_data.py --archive <file.tar.gz>", file=sys.stderr)
            return 1

    wanted_names = set(BY_NAME)
    done: set[str] = set()

    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        members = {m.name: m for m in tf.getmembers() if m.isfile()}
        root = next(iter(members)).split("/", 1)[0]  # e.g. opensim-core-4.6

        for src_rel, dests in DIR_MAP.items():
            prefix = f"{root}/{src_rel}/"
            hits = [n for n in members if n.startswith(prefix)
                    and Path(n).suffix.lower() in DATA_EXT]
            for n in hits:
                _copy(tf.extractfile(members[n]).read(), Path(n).name, dests, done)

        for src_rel, (names, dest) in FILE_MAP.items():
            for base in names:
                n = f"{root}/{src_rel}/{base}"
                if n in members:
                    _copy(tf.extractfile(members[n]).read(), base, [dest], done)

        for n, m in members.items():
            base = Path(n).name
            if base in wanted_names and Path(n).suffix.lower() in DATA_EXT:
                _copy(tf.extractfile(m).read(), base, BY_NAME[base], done)
                wanted_names.discard(base)

    for src_path, dests in FROM_RESOURCES.items():
        if src_path.is_file():
            for rel in dests:
                (HERE / rel).mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, HERE / rel / src_path.name)
                done.add(f"{rel}/{src_path.name}")

    print(f"\ncopied {len(done)} file(s):")
    for f in sorted(done):
        print(f"  {f}")

    if wanted_names:
        print(f"\nnot found in the opensim-core {TAG} source: {sorted(wanted_names)}")
        if "gait10dof18musc.osim" in wanted_names:
            print("  gait10dof18musc.osim ships with the OpenSim GUI application "
                  "(Applications/OpenSim/*/Models/Gait10dof18musc/). Copy it into "
                  "Moco/example2DWalking/ if you want to run example2DWalkingMetabolics.py.")

    print("\nSee README.md for which script needs what.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

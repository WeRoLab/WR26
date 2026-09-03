"""Download the data for Assignment 2 (motion capture + OpenSim model).

The data is NOT stored in git (~5 MB zip, mostly binary C3D + bone meshes).
Everything lives in a single zip on Google Drive.  Run this once, from this
folder:

    pip install -r requirements.txt        # installs `gdown`
    python get_data.py

It creates, next to this file:

    data/
        Model/
            Biomech57.osim      generic full-body model (Rajagopal 2016 + head)
            marker_set.xml      the 57 OptiTrack "Biomech-57" markers
            Geometry/           ~90 bone-mesh .vtp files (for 3-D visualisation)
            save_data_info.m    per-subject mass + height (subject AB03: 60.01 kg, 1.70 m)
        static.c3d              A-pose calibration trial  (57 markers, 200 Hz)
        stride_1.c3d            one walking stride         (49 markers, 200 Hz)

Options:
    python get_data.py --force     re-download even if data/ already exists

Manual fallback (if the script cannot reach Google Drive): open the link below
in a browser, download `A2_Data.zip`, and unzip it here so that
"data/stride_1.c3d" exists (rename the unzipped "A2_Data" folder to "data").

    https://drive.google.com/file/d/1-Vnjx9NQ_fsy6oQXMHKkgA4e-crwm6yI/view?usp=sharing
"""

from __future__ import annotations

import pathlib
import shutil
import sys
import zipfile

# The shared Google Drive FILE (not a folder).  Downloaded by its id, which is
# stable across `gdown` versions.  ZIP_URL is only used in the printed messages.
FILE_ID = "1-Vnjx9NQ_fsy6oQXMHKkgA4e-crwm6yI"
ZIP_URL = f"https://drive.google.com/file/d/{FILE_ID}/view?usp=sharing"

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
ZIP_PATH = HERE / "_A2_data.zip"

# Files the unpacked zip must produce (relative to DATA).  We check a few deep
# entries so a half-finished download is not mistaken for success.
EXPECT = [
    "stride_1.c3d",
    "static.c3d",
    "Model/Biomech57.osim",
    "Model/marker_set.xml",
    "Model/Geometry",
]


def have_everything() -> bool:
    """True if every expected file/folder is already in place under data/."""
    return all((DATA / rel).exists() for rel in EXPECT)


def _unpack(zip_path: pathlib.Path) -> None:
    """Extract `zip_path` and end up with the payload directly under data/.

    The archive wraps everything in a single top-level folder (``A2_Data/``);
    we extract to a scratch folder and then move that folder to ``data/``.
    """
    scratch = HERE / "_A2_data_unzip"
    if scratch.exists():
        shutil.rmtree(scratch)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(scratch)

    # Find the folder that actually holds the payload (the wrapper, or scratch
    # itself if the archive had no wrapper).
    roots = [scratch] + [p for p in scratch.iterdir() if p.is_dir()]
    payload = next((r for r in roots if (r / "stride_1.c3d").is_file()), None)
    if payload is None:
        raise RuntimeError("could not find stride_1.c3d inside the downloaded zip")

    if DATA.exists():
        shutil.rmtree(DATA)
    shutil.move(str(payload), str(DATA))
    shutil.rmtree(scratch, ignore_errors=True)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    force = "--force" in argv

    if have_everything() and not force:
        print("data already present -- nothing to do (use --force to re-download)")
        return 0

    try:
        import gdown
    except ImportError:
        print("The `gdown` package is not installed.  Run:\n"
              "    pip install -r requirements.txt\n", file=sys.stderr)
        return 1

    print(f"downloading the Assignment 2 data (~5 MB) from:\n  {ZIP_URL}\n")
    out = gdown.download(id=FILE_ID, output=str(ZIP_PATH), quiet=False)
    if not out or not ZIP_PATH.is_file():
        print("\nDownload failed.  Download `A2_Data.zip` by hand from the link "
              f"above and unzip it into:\n  {HERE}\n"
              "(rename the unzipped 'A2_Data' folder to 'data').", file=sys.stderr)
        return 1

    print("\nunzipping ...")
    _unpack(ZIP_PATH)
    ZIP_PATH.unlink()

    missing = [rel for rel in EXPECT if not (DATA / rel).exists()]
    if missing:
        print(f"\nUnzipped, but these are still missing: {missing}\n"
              "The zip layout may have changed -- unzip it by hand into "
              f"{DATA}\n", file=sys.stderr)
        return 1

    print("\ndone.  Ready:")
    for rel in EXPECT:
        print(f"  data/{rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

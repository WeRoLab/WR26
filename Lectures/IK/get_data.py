"""Download the data / model / tutorials for this activity.

They are NOT stored in the git repo (a ~190 MB download, ~330 MB unpacked,
mostly binary).  Everything lives in a single zip on Google Drive.  Run this
once, from the repo root:

    pip install -r requirements.txt        # installs `gdown`
    python get_data.py

It creates, next to this file:

    Data/                    raw motion capture + the scaled subject model
    OpenSim/                 the generic musculoskeletal model + bone meshes
    Python Documentation/    OpenSim's own tutorial notebooks (reference)

Options:
    python get_data.py --force     re-download even if the folders already exist

Manual fallback (if the script cannot reach Google Drive): open the link below in
a browser, download the zip, and unzip it here so that "Data/AB03/C3D/" exists.

    https://drive.google.com/file/d/10WfyC948GXu7VsfAAvlNFpmtR7p5P9uD/view?usp=sharing
"""

from __future__ import annotations

import pathlib
import shutil
import sys
import zipfile

# The shared Google Drive FILE (not folder).  We download it by its id, which is
# stable across `gdown` versions.  ZIP_URL is only for the printed messages.
FILE_ID = "10WfyC948GXu7VsfAAvlNFpmtR7p5P9uD"
ZIP_URL = f"https://drive.google.com/file/d/{FILE_ID}/view?usp=sharing"

HERE = pathlib.Path(__file__).resolve().parent
ZIP_PATH = HERE / "_activity_data.zip"

# Folders the zip must produce (relative to HERE).  We check one deep sub-folder
# of each so a half-finished download is not mistaken for success.
EXPECT = ["Data/AB03/C3D", "OpenSim/Model", "Python Documentation"]


def have_everything() -> bool:
    """True if all the expected folders are already in place."""
    return all((HERE / rel).is_dir() for rel in EXPECT)


def _flatten_single_wrapper():
    """If the zip unpacked into one wrapper folder (e.g. 'IK Data/Data/...'),
    move its contents up so we end with Data/, OpenSim/, ... directly here.
    """
    if have_everything():
        return
    for child in HERE.iterdir():
        if not child.is_dir() or child.name in {".git", "solution", "activity"}:
            continue
        # does this wrapper hold the folders we want?
        if all((child / rel).is_dir() for rel in EXPECT):
            for item in child.iterdir():
                dest = HERE / item.name
                if dest.exists():
                    shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
                shutil.move(str(item), str(dest))
            child.rmdir()
            return


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    force = "--force" in argv

    if have_everything() and not force:
        print("data already present -- nothing to do "
              "(use --force to re-download)")
        return 0

    try:
        import gdown
    except ImportError:
        print("The `gdown` package is not installed.  Run:\n"
              "    pip install -r requirements.txt\n", file=sys.stderr)
        return 1

    print(f"downloading the activity data (~190 MB) from:\n  {ZIP_URL}\n")
    out = gdown.download(id=FILE_ID, output=str(ZIP_PATH), quiet=False)
    if not out or not ZIP_PATH.is_file():
        print("\nDownload failed.  Download the zip by hand from the link above "
              f"and unzip it into:\n  {HERE}\n", file=sys.stderr)
        return 1

    print("\nunzipping ...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(HERE)
    ZIP_PATH.unlink()
    _flatten_single_wrapper()

    missing = [rel for rel in EXPECT if not (HERE / rel).is_dir()]
    if missing:
        print(f"\nUnzipped, but these folders are still missing: {missing}\n"
              "The zip layout may differ from what was expected -- unzip it by "
              f"hand into:\n  {HERE}\n", file=sys.stderr)
        return 1

    print("\ndone.  Ready:")
    for rel in EXPECT:
        print(f"  {(HERE / rel).relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

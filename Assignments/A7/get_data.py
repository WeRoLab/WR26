"""Download the data for Assignment 7 (powered knee-ankle prosthesis, TF01).

The data is NOT stored in git (~5 MB MATLAB file).  It lives on Google Drive.
Run this once, from any folder:

    pip install -r requirements.txt        # installs `gdown`
    python get_data.py

It creates, next to this file:

    data/
        TF01.mat    one transfemoral amputee walking on a treadmill with a
                    powered knee-ankle prosthesis: 31 strides, each one
                    normalized to 101 samples (0-100 % of the gait cycle)

TF01.mat is an extract of the complete dataset of Elery et al. (2020),
"Effects of a Powered Knee-Ankle Prosthesis on Amputee Hip Compensations:
A Case Series", available on IEEE DataPort (CC BY 4.0):

    https://ieee-dataport.org/documents/effects-powered-knee-ankle-prosthesis-amputee-hip-compensations-case-series

Options:
    python get_data.py --force     re-download even if data/TF01.mat exists

Manual fallback (if the script cannot reach Google Drive): open the link below
in a browser, download `TF01.mat`, and save it here as "data/TF01.mat".

    https://drive.google.com/file/d/1MMMY3uGZe18yfAnt3ipIgBH-Tq0_JqPR/view?usp=sharing
"""

from __future__ import annotations

import pathlib
import sys

# The shared Google Drive FILE.  Downloaded by its id, which is stable across
# `gdown` versions.  FILE_URL is only used in the printed messages.
FILE_ID = "1MMMY3uGZe18yfAnt3ipIgBH-Tq0_JqPR"
FILE_URL = f"https://drive.google.com/file/d/{FILE_ID}/view?usp=sharing"

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
MAT_PATH = DATA / "TF01.mat"


def is_valid(path: pathlib.Path) -> bool:
    """True if `path` is a MAT-file that holds TF01.Leg2Data.

    Catches a half-finished download, or an HTML error page that Google Drive
    sometimes serves instead of the file.
    """
    if not path.is_file():
        return False
    try:
        from scipy.io import loadmat
        mat = loadmat(path, squeeze_me=True, struct_as_record=False,
                      variable_names=["TF01"])
        return hasattr(mat["TF01"], "Leg2Data")
    except Exception:
        return False


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    force = "--force" in argv

    if is_valid(MAT_PATH) and not force:
        print("data already present -- nothing to do (use --force to re-download)")
        return 0

    try:
        import gdown
    except ImportError:
        print("The `gdown` package is not installed.  Run:\n"
              "    pip install -r requirements.txt\n", file=sys.stderr)
        return 1

    DATA.mkdir(exist_ok=True)
    print(f"downloading the Assignment 7 data (~5 MB) from:\n  {FILE_URL}\n")
    out = gdown.download(id=FILE_ID, output=str(MAT_PATH), quiet=False)

    if not out or not is_valid(MAT_PATH):
        print("\nDownload failed.  Download `TF01.mat` by hand from the link "
              f"above and save it as:\n  {MAT_PATH}", file=sys.stderr)
        return 1

    print(f"\ndone.  Ready:\n  data/{MAT_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

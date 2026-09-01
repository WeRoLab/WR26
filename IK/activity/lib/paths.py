"""Find the project root -- the folder that holds ``Data/``, ``OpenSim/`` etc.

The data is not stored in git, so before anything can run it has to be fetched
with ``get_data.py``.  ``project_root()`` locates the folder that contains the
data and, if it is not there yet, exits with a clear instruction instead of a
confusing traceback.
"""

from __future__ import annotations

import pathlib
import sys


def project_root() -> pathlib.Path:
    """Return the folder that contains ``Data/AB03/``.

    Walk up from this file through its parent folders; the first one that has a
    ``Data/AB03`` folder inside it is the project root.  If none do, the data has
    not been downloaded yet -- print how to get it and stop.
    """
    here = pathlib.Path(__file__).resolve()
    for p in here.parents:
        if (p / "Data" / "AB03").is_dir():
            return p

    # Data is missing.  Find where get_data.py lives so the message is exact.
    root = next((p for p in here.parents if (p / "get_data.py").is_file()),
                here.parents[2])
    sys.exit(
        "\n  The activity data is not in the repo (a ~190 MB download, fetched "
        "separately).\n  Download it first:\n\n"
        f'      cd "{root}"\n'
        "      pip install -r requirements.txt\n"
        "      python get_data.py\n"
    )

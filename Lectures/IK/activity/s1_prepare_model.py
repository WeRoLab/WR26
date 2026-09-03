"""s1 -- prepare the scaled model for inverse kinematics   [GIVEN to students]

Scaling was already done by the lab (``Data/AB03/Static/1/generated/
Generated_Model.osim``, subject mass 60.01 kg).  This script only strips the
muscles and immobilises the upper body / neck / foot-detail joints so the IK
solve is fast and focused on the sagittal lower-limb + trunk motion.

Run:  python s1_prepare_model.py
Output: results/Model_ik.osim

The real work is one function call (``model_tools.make_ik_model``); this file
just sorts out the file paths and calls it.
"""

# Makes the ": type" hints lazy; harmless, safe to ignore.
from __future__ import annotations

import pathlib                      # build/inspect file paths, OS-independently

from lib import model_tools         # our helper module in the sibling lib/ folder
from lib.paths import project_root  # find the data folder (or say how to get it)


# --- work out where things are (see s0_prepare_data.py for the same pattern) ---

# __file__ = this script's path;  .resolve() -> absolute;  .parent -> its folder.
HERE = pathlib.Path(__file__).resolve().parent

# The project root = the folder that has "Data/AB03" in it.  project_root() finds
# it, or exits with "run get_data.py" if the data was never downloaded.
ROOT = project_root()

# "/" joins path pieces.  Input: the model the lab already scaled to this subject.
GENERATED = ROOT / "Data" / "AB03" / "Static" / "1" / "generated" / "Generated_Model.osim"

# Output: where we will save the trimmed model for the IK steps.
OUT = HERE / "results" / "Model_ik.osim"


def main():
    """Run everything: print a banner, make the output folder, build the model."""
    print("=" * 70)                              # a line of 70 "=" characters
    print("s1  prepare scaled model for IK")
    print("=" * 70)

    # OUT.parent is the "results" folder.  Create it if it does not exist yet
    # (parents=True: also make missing parents; exist_ok=True: no error if present).
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Load GENERATED, remove its muscles, lock the arms / neck / subtalar / toes,
    # and save the result to OUT.  All of that lives in lib/model_tools.py.
    model_tools.make_ik_model(GENERATED, OUT)


# Only call main() when this file is run directly ( python s1_prepare_model.py ),
# not when another script imports it.
if __name__ == "__main__":
    main()

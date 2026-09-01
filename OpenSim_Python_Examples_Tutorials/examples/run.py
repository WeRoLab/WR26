"""Run one example script with the environment it expects.

    python run.py Moco/exampleOptimizeMass.py
    python run.py build_simple_arm_model.py

Does two things the bare `python <script>.py` does not:

1. Prepends the installed ``opensim`` package directory to ``PATH`` so Moco's
   CasADi/Ipopt solver plugin loads on Windows (the pip ``opensim`` 4.6 wheel
   does not do this itself; without it the Moco examples fail with
   ``Plugin 'ipopt' is not found``).
2. ``chdir``s into the script's own folder, since the scripts read and write
   files by bare name.

You can still run scripts directly (``cd`` into the folder first); this wrapper
is just the reliable path, especially on Windows.
"""

from __future__ import annotations

import os
import runpy
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    script = os.path.abspath(sys.argv[1])
    if not os.path.isfile(script):
        print(f"no such script: {sys.argv[1]}", file=sys.stderr)
        return 1

    try:
        import opensim
        pkg = os.path.dirname(opensim.__file__)
        os.environ["PATH"] = pkg + os.pathsep + os.environ.get("PATH", "")
        add = getattr(os, "add_dll_directory", None)
        if add is not None:
            try:
                add(pkg)
            except OSError:
                pass
    except Exception as e:  # pragma: no cover
        print(f"warning: could not import opensim ({e})", file=sys.stderr)

    os.chdir(os.path.dirname(script))
    sys.argv = [script, *sys.argv[2:]]
    runpy.run_path(script, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

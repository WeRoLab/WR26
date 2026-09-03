"""Local-run helpers for the OpenSim tutorial notebooks.

These notebooks were adapted from the OpenSim project's Google Colab tutorials to
run **locally** against a pip-installed ``opensim`` (>= 4.6).  They no longer
install anything or download data from the internet: every input file they need
ships in this folder under ``resources/``.

Typical use, once per notebook (in the "Set up OpenSim" section)::

    from tutorial_setup import prepare
    prepare("Tutorial 5")      # -> copies resources/Tutorial 5/* into _work/Tutorial 5/
                               #    and changes the working directory to it

After ``prepare(...)`` the notebook can open its input files by their bare names
(``osim.Model("gait2354_simbody.osim")``), exactly as the original Colab
notebooks did, and every file it writes lands in ``_work/<tutorial>/`` which is
git-ignored.  Nothing is written back into ``resources/`` or the repo folder.

Importing this module also runs :func:`enable_moco_dlls` (see below).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

#: Absolute path to this folder (``Examples``).  Resolved once, at import time,
#: so it stays valid after ``prepare`` calls ``os.chdir``.
ROOT = Path(__file__).resolve().parent

RESOURCES = ROOT / "resources"
WORK = ROOT / "_work"


def enable_moco_dlls() -> bool:
    """Put the installed ``opensim`` package directory on the Windows DLL path.

    Moco's ``MocoCasADiSolver`` loads its optimizer (Ipopt) as a CasADi *plugin*
    at run time.  CasADi's plugin loader searches ``PATH`` for the plugin DLL and
    its dependencies, but the pip ``opensim`` wheel installs those DLLs inside the
    package directory, which is not on ``PATH``.  Without this, Tutorials 7 and 8
    fail with ``Plugin 'ipopt' is not found``.

    Safe and idempotent; a no-op off Windows or if ``opensim`` is not importable.
    Returns ``True`` if the fix was applied.
    """
    try:
        import opensim  # noqa: F401
    except Exception:
        return False
    pkg = os.path.dirname(opensim.__file__)
    if os.name == "nt":
        if pkg not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = pkg + os.pathsep + os.environ.get("PATH", "")
        add = getattr(os, "add_dll_directory", None)
        if add is not None:
            try:
                add(pkg)
            except OSError:
                pass
    return True


def prepare(tutorial: str) -> Path:
    """Set up an isolated working directory for one tutorial and ``cd`` into it.

    Copies ``resources/<tutorial>/`` (if present) into ``_work/<tutorial>/`` and
    makes that the current working directory.  Safe to call more than once - it
    refreshes the input files and leaves anything else in place.

    Parameters
    ----------
    tutorial:
        Folder name under ``resources/``, e.g. ``"Tutorial 5"``.  A tutorial with
        no bundled resources (1, 2, 7) still calls this to get a clean scratch
        directory (and the Moco DLL fix).

    Returns
    -------
    pathlib.Path
        Absolute path to the working directory (now also ``os.getcwd()``).
    """
    enable_moco_dlls()

    work = WORK / tutorial
    work.mkdir(parents=True, exist_ok=True)

    src = RESOURCES / tutorial
    if src.is_dir():
        shutil.copytree(src, work, dirs_exist_ok=True)

    os.chdir(work)
    print(f"working directory: {work}")
    if src.is_dir():
        files = sorted(p.name for p in src.iterdir() if p.is_file())
        if files:
            print("input files copied from resources/{}: {}".format(
                tutorial, ", ".join(files)))
    return work


def resource(tutorial: str, name: str) -> Path:
    """Return the absolute path to a single bundled resource file.

    Useful when you would rather read a file straight from ``resources/`` than
    copy the whole folder with :func:`prepare`.
    """
    path = RESOURCES / tutorial / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Bundled resources live in "
            f"{RESOURCES}; see resources/README.md."
        )
    return path


enable_moco_dlls()

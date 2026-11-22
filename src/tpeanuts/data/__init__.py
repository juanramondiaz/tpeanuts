#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
#
# TPeanuts — Fast Simulation of Neutrino Oscillations in Matter.
#
# Developed as part of the Master's Thesis for the MSc in Advanced Physics,
# specialization in Theoretical Physics, at the Universitat de València.
#
# Author
# ------
# Juan Ramón Díaz Santos
# diazjuan@alumni.uv.es
#
# Supervisors
# -----------
# Roberto Ruiz de Austri Bazán
# rruiz@ific.uv.es
#
# Michele Lucente
# michele.lucente@unibo.it
#
# Universitat de València, Spain
# September 2026
# =============================================================================

"""Where every piece of data tpeanuts reads or writes lives.

Data fall in three groups, each with its own root, so that nothing ever has to
be located relative to a source checkout (which does not exist for an installed
package) and nothing is ever written into the installed package:

``packaged``
    Small, redistributable tables shipped inside the wheel, in this very
    directory (``earth/``, ``solar/``, ``experiments/...``). Read-only. Their origin
    and terms are in ``DATA_LICENSES.md`` at the repository root.
``datasets``
    Large third-party data that are downloaded on request (the IceCube Monte
    Carlo, the raw Honda tables). Root: ``$TPEANUTS_DATA_DIR`` if set, otherwise
    the per-user cache directory.
``generated``
    Products computed on the user's machine (MCEq and Honda flux files, the
    canonical Honda surface-flux table). Writable. Root:
    ``$TPEANUTS_USER_DATA_DIR`` if set, otherwise the per-user data directory.

The submodules ``registry`` (what can be downloaded, with checksums), ``fetch`` (verified downloads)
and ``cli`` (the ``tpeanuts-data`` command) build on the functions below.

Path resolution never touches the disk, so it is safe at import time; the
loaders call ``require`` when they actually need a file and get an error that
says what is missing and where it was looked for.

Module contents:
    DATA_DIR_ENV, USER_DATA_DIR_ENV
        Names of the two environment variables that relocate the roots.
    DataNotAvailableError
        ``FileNotFoundError`` subclass raised for a missing dataset or file.
    package_data_dir(...)
        Directory of the data shipped inside the package.
    package_data(...)
        Path of one shipped file or directory, checked to exist.
    datasets_dir(...), dataset_dir(...)
        Root of the downloaded datasets, and the directory of one of them.
    generated_dir(...)
        Root of the writable, user-generated products.
    require(...)
        Return a path if it exists, otherwise raise ``DataNotAvailableError``.
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path
from typing import Optional

import platformdirs

DATA_DIR_ENV = "TPEANUTS_DATA_DIR"
USER_DATA_DIR_ENV = "TPEANUTS_USER_DATA_DIR"
_APP_NAME = "tpeanuts"


class DataNotAvailableError(FileNotFoundError):
    """A data file or dataset tpeanuts needs is not where it is looked for.

    A ``FileNotFoundError`` subclass, so code (and tests) that already handle a
    missing file keep working; the message states what is missing, where it was
    looked for and, when known, how to obtain it.
    """


def package_data_dir() -> Path:
    """Return the directory of the data shipped inside the installed package (read-only).

    Returns:
        The ``tpeanuts/data`` directory, in the source tree for an editable
        install and under ``site-packages`` for a normal one.
    """
    return Path(str(resources.files(__package__)))


def package_data(*parts: str) -> Path:
    """Return the path of a file or directory shipped with the package.

    Args:
        *parts: Path components below the package data directory, e.g.
            ``("earth", "density", "prem_density.csv")``.

    Returns:
        The absolute path, guaranteed to exist.

    Raises:
        DataNotAvailableError: If it does not exist. For a packaged file this
            means the installation is broken, not that the user forgot a download.
    """
    path = package_data_dir().joinpath(*parts)
    if not path.exists():
        raise DataNotAvailableError(
            f"Packaged data '{Path(*parts).as_posix()}' is missing from {package_data_dir()}; "
            "the tpeanuts installation is incomplete (reinstall the package)."
        )
    return path


def datasets_dir() -> Path:
    """Return the root directory of the downloaded datasets (not created here).

    Returns:
        ``$TPEANUTS_DATA_DIR`` if set, otherwise ``<user cache dir>/datasets``.
    """
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path(platformdirs.user_cache_dir(_APP_NAME, appauthor=False)) / "datasets"


def dataset_dir(name: str) -> Path:
    """Return the directory where the dataset ``name`` is (or will be) stored.

    Args:
        name: Dataset name, e.g. ``"icecube"`` or ``"honda"``.

    Returns:
        ``<datasets root>/<name>``. It may not exist yet.
    """
    return datasets_dir() / name


def generated_dir() -> Path:
    """Return the root directory of the writable, user-generated products (not created here).

    Returns:
        ``$TPEANUTS_USER_DATA_DIR`` if set, otherwise ``<user data dir>/generated``.
    """
    override = os.environ.get(USER_DATA_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path(platformdirs.user_data_dir(_APP_NAME, appauthor=False)) / "generated"


def require(path: Path, *, what: str, how_to_obtain: Optional[str] = None) -> Path:
    """Return ``path`` if it exists, otherwise raise an error that says how to get it.

    Args:
        path: File or directory the caller is about to read.
        what: Human description of the data, e.g. ``"IceCube Monte Carlo"``.
        how_to_obtain: Optional instruction appended to the message.

    Returns:
        ``path`` itself.

    Raises:
        DataNotAvailableError: If ``path`` does not exist.
    """
    if path.exists():
        return path
    message = f"{what} not found at {path}."
    if how_to_obtain:
        message += f" {how_to_obtain}"
    raise DataNotAvailableError(message)

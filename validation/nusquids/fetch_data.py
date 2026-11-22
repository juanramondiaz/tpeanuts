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

"""Fetch the nuSQuIDS ``astro/*.dat`` model files that the PyPI wheel does not ship.

The ``nusquids`` wheel compiles in an absolute resource path
(``/project/share/nuSQuIDS/``) that does not exist on a normal machine, and
carries none of the Earth/Sun model files, so ``Earth()``/``EarthAtm()``/
``SunASnu()`` fail to open their default model. The source distribution does
contain them (``data/astro``). This script downloads that sdist from PyPI,
extracts only ``data/astro`` and prints the directory to export as
``NUSQUIDS_DATA_DIR`` (see ``validation.nusquids.adapter.data_file_args``).

Usage::

    python -m validation.nusquids.fetch_data [TARGET_DIR]
    export NUSQUIDS_DATA_DIR=<TARGET_DIR>

Module contents:
    fetch_data(...)
        Download the sdist and extract ``data/astro`` into a directory.
    main(...)
        Command-line entry point.
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

_PYPI_JSON_URL = "https://pypi.org/pypi/nusquids/json"
_ASTRO_MEMBER = "data/astro/"


def fetch_data(target_dir: Path) -> Path:
    """Download the nuSQuIDS sdist and extract its ``data/astro`` files.

    Args:
        target_dir: Directory that will hold an ``astro/`` subdirectory.

    Returns:
        ``target_dir`` itself, ready to be used as ``NUSQUIDS_DATA_DIR``.

    Raises:
        RuntimeError: If PyPI lists no sdist for ``nusquids`` or the
            archive holds no ``data/astro`` files.
    """
    with urllib.request.urlopen(_PYPI_JSON_URL, timeout=60) as response:
        metadata = json.load(response)
    sdist_urls = [f["url"] for f in metadata["urls"] if f["packagetype"] == "sdist"]
    if not sdist_urls:
        raise RuntimeError("PyPI lists no source distribution for 'nusquids'.")

    with urllib.request.urlopen(sdist_urls[0], timeout=300) as response:
        payload = response.read()

    astro_dir = Path(target_dir) / "astro"
    astro_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            parts = member.name.split("/", 1)  # drop the top-level "nusquids-X.Y.Z/"
            if len(parts) != 2 or not member.isfile() or not parts[1].startswith(_ASTRO_MEMBER):
                continue
            destination = astro_dir / Path(parts[1]).name
            destination.write_bytes(archive.extractfile(member).read())
            extracted += 1
    if extracted == 0:
        raise RuntimeError("The nusquids sdist contains no data/astro files.")
    return Path(target_dir)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point: fetch the files into ``argv[0]`` (default ``./nusquids_data``)."""
    args = list(sys.argv[1:] if argv is None else argv)
    target = Path(args[0] if args else "nusquids_data").resolve()
    fetch_data(target)
    print(f"nuSQuIDS data written to {target / 'astro'}")
    print(f"export NUSQUIDS_DATA_DIR={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

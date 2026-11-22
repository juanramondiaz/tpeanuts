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

"""Loaders for the real IceCube DeepCore data release, cached under ``data/experiments/icecube/raw/``.

Ported from v1_0's ``detector/icecube/io.py``, replacing pandas with a
plain ``csv.reader`` (this project's own convention -- no pandas
anywhere in v2) and returning ``{column: numpy array}`` dicts instead
of a ``DataFrame``. Files are real tab-separated ``.tab`` text despite
the extension (confirmed against the release's own ``readme.md``).

**Only the columns this project's own forward model actually reads are
parsed** -- e.g. the MC files' own ``Q2``/``W``/``x``/``y``/
``MaCCQE_*``/``MaCCRES_*``/``type``/``interaction`` columns (interaction
-uncertainty/kinematic bookkeeping, not used anywhere in
``tpeanuts.experiments.icecube.event_rate``) and the hypersurface files'
own ``*_sigma`` columns are never even read, not silently dropped after
loading -- a deliberate scope choice for a release with files up to
285163 rows / 44.6 MB (``mc_numu_cc.tab``), confirmed to matter for
load time at this scale. Every function stays a thin, literal read of
its own file; the numpy arrays it returns are handed to
``tpeanuts.experiments.icecube.event_rate`` unchanged.

Module contents:
    load_observed_counts(...)
        Real observed event counts per analysis bin.
    load_mc_events(...)
        Real event-by-event Monte Carlo for one reaction channel.
    load_muon_background(...)
        Real pre-binned atmospheric-muon background counts.
    load_hypersurfaces(...)
        Real per-bin, per-DeltamSq31-slice detector-systematics
        correction coefficients for one reaction channel.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

import tpeanuts.config.default as default
from tpeanuts.data import require

_MC_CHANNEL_FILES = {
    "nc": "mc_nu_nc.tab",
    "nue_cc": "mc_nue_cc.tab",
    "numu_cc": "mc_numu_cc.tab",
    "nutau_cc": "mc_nutau_cc.tab",
}
_HYPERSURFACE_CHANNEL_FILES = {
    "nc": "hs_nu_nc_nue_cc.tab",
    "nue_cc": "hs_nu_nc_nue_cc.tab",
    "numu_cc": "hs_numu_cc.tab",
    "nutau_cc": "hs_nutau_cc.tab",
}

_SYSTEMATICS_NAMES = ("dom_eff", "hole_ice_p0", "hole_ice_p1", "bulk_ice_abs", "bulk_ice_scatter")


@lru_cache(maxsize=16)
def _parse_tab(
    path: str, mtime_ns: int, size: int, columns: Tuple[str, ...], int_columns: Tuple[str, ...],
) -> Dict[str, np.ndarray]:
    """Parse the requested columns of one ``.tab`` file; cached per (file, its mtime and size, columns).

    ``mtime_ns`` and ``size`` are part of the key only so that a modified file is read again. The arrays
    returned here are shared by the cache and must not be handed out: ``_read_tab_columns`` copies them.
    """
    with open(path, newline="") as handle:
        header = handle.readline().rstrip("\r\n").split("\t")
    missing = set(columns) - set(header)
    if missing:
        raise ValueError(f"{path}: missing required column(s) {sorted(missing)}.")
    indices = [header.index(name) for name in columns]
    table = np.loadtxt(path, delimiter="\t", skiprows=1, usecols=indices, dtype=np.float64, ndmin=2)
    int_set = set(int_columns)
    return {
        name: (table[:, k].astype(np.int64) if name in int_set else table[:, k].copy())
        for k, name in enumerate(columns)
    }


def _read_tab_columns(
    path: Union[str, Path], columns: Sequence[str], *, int_columns: Sequence[str] = (),
) -> Dict[str, np.ndarray]:
    """Read only the requested columns of a real ``.tab`` (TSV) file into numpy arrays.

    The parse is cached (the larger files are 44 MB and several modules and tests read them repeatedly);
    every call still returns its own arrays, so callers may modify them freely.

    Args:
        path: Path to the ``.tab`` file.
        columns: Column names to extract, in the file's own header.
        int_columns: Subset of ``columns`` to cast to ``np.int64``
            instead of the default ``np.float64``.

    Returns:
        ``{column: numpy array}``, one entry per requested column, each
        shaped ``(n_rows,)``.

    Raises:
        DataNotAvailableError: If ``path`` does not exist (the IceCube release is
            not packaged: it is a separate download, see ``DATA_LICENSES.md``).
        ValueError: If a requested column is not in the file's header.
    """
    path = require(
        Path(path),
        what="IceCube data-release file",
        how_to_obtain=(
            "The IceCube 9-year release (CC0, doi:10.7910/DVN/B4RITM) is not distributed with tpeanuts: "
            "run `tpeanuts-data fetch icecube` (64 MB; files go to <TPEANUTS_DATA_DIR>/icecube, by default the "
            "per-user cache directory)."
        ),
    )
    status = path.stat()
    cached = _parse_tab(str(path.resolve()), status.st_mtime_ns, status.st_size, tuple(columns), tuple(int_columns))
    return {name: array.copy() for name, array in cached.items()}


def load_observed_counts(path: Optional[Union[str, Path]] = None) -> Dict[str, np.ndarray]:
    """Real observed event counts per analysis bin.

    Args:
        path: Optional override path. None loads the bundled
            ``data.tab`` (200 rows).

    Returns:
        ``{"count", "pid", "reco_coszen", "reco_energy"}``, each shaped
        ``(200,)``.
    """
    if path is None:
        path = default.ICECUBE_DATA_DIR / "data.tab"
    return _read_tab_columns(path, ("count", "pid", "reco_coszen", "reco_energy"))


def load_mc_events(channel: str, path: Optional[Union[str, Path]] = None) -> Dict[str, np.ndarray]:
    """Real event-by-event Monte Carlo for one reaction channel.

    Args:
        channel: One of ``"nc"``, ``"nue_cc"``, ``"numu_cc"``,
            ``"nutau_cc"``.
        path: Optional override path. None loads the bundled MC file
            for ``channel``.

    Returns:
        ``{"reco_energy", "reco_coszen", "pid", "pdg", "true_energy",
        "true_coszen", "weight"}``, one entry per real simulated event
        (``"pdg"`` as ``np.int64``, the rest ``np.float64``).

    Raises:
        KeyError: If ``channel`` is not one of the four real channels.
    """
    if path is None:
        path = default.ICECUBE_DATA_DIR / _MC_CHANNEL_FILES[channel]
    return _read_tab_columns(
        path, ("reco_energy", "reco_coszen", "pid", "pdg", "true_energy", "true_coszen", "weight"),
        int_columns=("pdg",),
    )


def load_muon_background(path: Optional[Union[str, Path]] = None) -> Dict[str, np.ndarray]:
    """Real pre-binned atmospheric-muon background counts.

    Args:
        path: Optional override path. None loads the bundled
            ``mc_mu.tab`` (200 rows).

    Returns:
        ``{"count", "pid", "reco_coszen", "reco_energy"}``, each shaped
        ``(200,)``.
    """
    if path is None:
        path = default.ICECUBE_DATA_DIR / "mc_mu.tab"
    return _read_tab_columns(path, ("count", "pid", "reco_coszen", "reco_energy"))


def load_hypersurfaces(channel: str, path: Optional[Union[str, Path]] = None) -> Dict[str, np.ndarray]:
    """Real per-bin, per-DeltamSq31-slice detector-systematics hypersurface coefficients.

    Args:
        channel: One of ``"nc"``, ``"nue_cc"``, ``"numu_cc"``,
            ``"nutau_cc"`` (``"nc"``/``"nue_cc"`` share the same real
            hypersurface file, matching the release's own grouping).
        path: Optional override path. None loads the bundled
            hypersurface file for ``channel``.

    Returns:
        ``{"intercept", "dom_eff", "hole_ice_p0", "hole_ice_p1",
        "bulk_ice_abs", "bulk_ice_scatter", "deltam31", "pid",
        "reco_coszen", "reco_energy"}``, each shaped ``(4000,)`` (200
        bins x 20 DeltamSq31 slices).

    Raises:
        KeyError: If ``channel`` is not one of the four real channels.
    """
    if path is None:
        path = default.ICECUBE_DATA_DIR / _HYPERSURFACE_CHANNEL_FILES[channel]
    columns = ("intercept",) + _SYSTEMATICS_NAMES + ("deltam31", "pid", "reco_coszen", "reco_energy")
    return _read_tab_columns(path, columns)

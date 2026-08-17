#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
#  This module is part of the Master's Thesis (MSc Dissertation):
#  - Fast Simulation of Neutrino Oscillations in Matter
#
#  Author:
#      Juan Ramon Diaz Santos <diazjuan@alumni.uv.es>
#
#  Supervisors:
#      Roberto Ruiz de Austri Bazan <rruiz@ific.uv.es>
#      Michele Lucente <michele.lucente@unibo.it>
#
#  Date:
#      June 2026
# =============================================================================

"""
Loaders for the real IceCube DeepCore data release, cached under ``data/detector/icecube/raw/``.

Every function here reads a file fetched verbatim by
``notebooks/external/icecube/IceCube1_generator.ipynb`` from the official
Harvard Dataverse release (DOI 10.7910/DVN/B4RITM) -- see that notebook and
this package's own module docstring for provenance and scope. Files are
real tab-separated ``.tab`` text despite the extension (confirmed against
the release's own ``readme.md``), loaded here with pandas and returned as
plain torch tensors.

Module contents:
    load_observed_counts(...)
        Real observed event counts per analysis bin.
    load_mc_events(...)
        Real event-by-event Monte Carlo for one reaction channel.
    load_muon_background(...)
        Real pre-binned atmospheric-muon background (count + uncertainty).
    load_hypersurfaces(...)
        Real per-bin, per-DeltamSq31-slice detector-systematics correction
        coefficients for one reaction channel.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import torch

from tpeanuts.util.io import package_dir

_ICECUBE_RAW_DIR = package_dir() / "data" / "detector" / "icecube" / "raw"

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


def load_observed_counts(
    *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64,
) -> pd.DataFrame:
    """Real observed event counts per analysis bin (``count``/``pid``/``reco_coszen``/``reco_energy``).

    Returns:
        The raw pandas DataFrame (200 rows) -- kept as a DataFrame rather
        than tensors since callers need to match its ``(pid, reco_coszen,
        reco_energy)`` bin-center columns against
        ``detector.icecube.parameters``'s bin edges to build a bin index.
    """
    return pd.read_csv(_ICECUBE_RAW_DIR / "data.tab", sep="\t")


def load_mc_events(channel: str) -> pd.DataFrame:
    """Real event-by-event Monte Carlo for one reaction channel.

    Args:
        channel: One of ``"nc"``, ``"nue_cc"``, ``"numu_cc"``, ``"nutau_cc"``.

    Returns:
        The raw pandas DataFrame, one row per simulated event.

    Raises:
        KeyError: If ``channel`` is not one of the four real channels.
    """
    return pd.read_csv(_ICECUBE_RAW_DIR / _MC_CHANNEL_FILES[channel], sep="\t")


def load_muon_background() -> pd.DataFrame:
    """Real pre-binned atmospheric-muon background (``count``/``abs_uncertainty`` per bin)."""
    return pd.read_csv(_ICECUBE_RAW_DIR / "mc_mu.tab", sep="\t")


def load_hypersurfaces(channel: str) -> pd.DataFrame:
    """Real per-bin, per-DeltamSq31-slice detector-systematics hypersurface coefficients.

    Args:
        channel: One of ``"nc"``, ``"nue_cc"``, ``"numu_cc"``, ``"nutau_cc"``
            (``"nc"``/``"nue_cc"`` share the same real hypersurface file,
            matching the release's own grouping).

    Returns:
        The raw pandas DataFrame (4000 rows: 200 bins x 20 DeltamSq31
        slices).
    """
    return pd.read_csv(_ICECUBE_RAW_DIR / _HYPERSURFACE_CHANNEL_FILES[channel], sep="\t")

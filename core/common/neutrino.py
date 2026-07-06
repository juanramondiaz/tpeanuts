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
Canonical neutrino flavour names and indices.

This module centralizes the flavour convention shared by probabilities, flux
propagation, and user-facing selection APIs:

    electron -> 0
    muon     -> 1
    tau      -> 2

Module contents:
    FLAVOUR_TO_INDEX
        Maps accepted flavour aliases to their canonical integer index.

    flavour_index(...)
        Validates a flavour name or index and returns its canonical index.
"""

from __future__ import annotations

from typing import Union


FLAVOUR_TO_INDEX = {
    "e": 0,
    "electron": 0,
    "nue": 0,
    "nu_e": 0,
    "mu": 1,
    "muon": 1,
    "numu": 1,
    "nu_mu": 1,
    "tau": 2,
    "nutau": 2,
    "nu_tau": 2,
}


def flavour_index(flavour: Union[str, int]) -> int:
    """Return the canonical index of a neutrino flavour.

    Args:
        flavour: Flavour index or accepted electron, muon, or tau alias.

    Returns:
        Canonical index in the order electron=0, muon=1, tau=2.

    Raises:
        ValueError: If the integer or flavour alias is not recognized.
    """
    if isinstance(flavour, int):
        if flavour not in (0, 1, 2):
            raise ValueError("Flavour index must be 0, 1, or 2.")
        return flavour

    key = flavour.lower()
    if key not in FLAVOUR_TO_INDEX:
        raise ValueError(f"Unknown flavour label: {flavour}")

    return FLAVOUR_TO_INDEX[key]

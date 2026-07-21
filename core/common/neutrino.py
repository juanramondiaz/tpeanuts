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
    sterile  -> 3   (3+1 sterile extension only, see core.BSM.bsm_sterile)

The sterile flavour has no charged-lepton counterpart, but the 3+1 extension
(``tpeanuts.core.BSM.bsm_sterile.PMNS_sterile``) fixes index 3 as its
canonical position (see that module's docstring), so it is included here
under a small set of aliases for callers that want to build or select it by
name instead of hand-indexing a length-4 tensor.

Module contents:
    FLAVOUR_TO_INDEX
        Maps accepted flavour aliases to their canonical integer index.

    flavour_index(...)
        Validates a flavour name or index and returns its canonical index.

    flavour_state(...)
        Returns the canonical unit flavour state, sized to 3 (Standard
        Model) or 4 (3+1 sterile) flavours.
"""

from __future__ import annotations

from typing import Union

import torch

from tpeanuts.util.type import cdtype_from_real


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
    "s": 3,
    "sterile": 3,
    "nus": 3,
    "nu_s": 3,
}

FLAVOUR_ORDER = ("nue", "numu", "nutau")
FLAVOUR_ORDER_STERILE = ("nue", "numu", "nutau", "nus")


def flavour_index(flavour: Union[str, int]) -> int:
    """Return the canonical index of a neutrino flavour.

    Args:
        flavour: Flavour index or accepted electron, muon, tau, or sterile
            alias.

    Returns:
        Canonical index in the order electron=0, muon=1, tau=2, sterile=3.

    Raises:
        ValueError: If the integer or flavour alias is not recognized.
    """
    if isinstance(flavour, int):
        if flavour not in (0, 1, 2, 3):
            raise ValueError("Flavour index must be 0, 1, 2, or 3.")
        return flavour

    key = flavour.lower()
    if key not in FLAVOUR_TO_INDEX:
        raise ValueError(f"Unknown flavour label: {flavour}")

    return FLAVOUR_TO_INDEX[key]


def flavour_state(
    flavour: Union[str, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
    n_flavours: int = 3,
) -> torch.Tensor:
    """Return the canonical unit flavour state using a real base dtype.

    Args:
        flavour: Flavour index or accepted alias (see ``flavour_index``).
            Selecting the sterile flavour (index 3) requires
            ``n_flavours=4``.
        device: Target device for the returned tensor.
        dtype: Real base dtype; the returned state uses its complex
            counterpart (see ``cdtype_from_real``).
        n_flavours: Length of the returned state vector, 3 (Standard Model)
            or 4 (3+1 sterile extension).

    Returns:
        Complex unit vector shaped ``(n_flavours,)`` with a 1 at the
        selected flavour's index.

    Raises:
        ValueError: If ``n_flavours`` is not 3 or 4, or if the selected
            flavour's index is out of range for ``n_flavours``.
    """
    if n_flavours not in (3, 4):
        raise ValueError("n_flavours must be 3 or 4.")

    index = flavour_index(flavour)
    if index >= n_flavours:
        raise ValueError(
            f"Flavour index {index} is out of range for n_flavours={n_flavours}."
        )

    state = torch.zeros(n_flavours, device=device, dtype=cdtype_from_real(dtype))
    state[index] = 1.0
    return state

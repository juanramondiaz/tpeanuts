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
PMNS dressing utilities for earth propagation in peanuts-torch.

The core peanuts segment evolution is computed in a reduced flavour basis.
This module converts reduced evolution operators into the full flavour basis.

The dressing convention is:

    U_full =
        R23 Delta U_red Delta^dagger R23^T

where Delta = diag(1, 1, exp(i delta)).

Module functions:
    
    earth_dressing_matrices(...)
        Extracts R23 and Delta phase matrices from a PMNS object and
        conjugates them for antineutrino propagation when needed.
        
    dress_reduced_evolutor(...)
        Transforms a reduced-basis evolution operator to the full flavour
        basis with R23 Delta U_red Delta^dagger R23^T.
"""



from __future__ import annotations

from typing import Union

import torch


def earth_dressing_matrices(
    pmns: object,
    *,
    antinu: Union[bool, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract the R23 rotation and CP-phase matrices required to dress reduced evolutors into the full flavour basis.
    
    Args:
        pmns: PMNS object exposing full and reduced mixing matrices plus R23 and Delta builders.
        antinu: Bool or boolean tensor; True selects antineutrino sign or complex-conjugated mixing.
        device: Optional torch device for newly created tensors.
        dtype: Real or complex torch dtype for newly created tensors.
    
    Returns:
        Tuple (R23, Delta) of complex matrices broadcastable with reduced evolutors.
    """
    r23 = pmns.R23().to(device=device, dtype=dtype)
    delta = pmns.Delta().to(device=device, dtype=dtype)

    if isinstance(antinu, bool):
        if antinu:
            r23 = torch.conj(r23)
            delta = torch.conj(delta)
    else:
        antinu = antinu.to(device=device, dtype=torch.bool)
        r23 = torch.where(
            antinu[..., None, None],
            torch.conj(r23),
            r23,
        )
        delta = torch.where(
            antinu[..., None, None],
            torch.conj(delta),
            delta,
        )

    return r23, delta


def dress_reduced_evolutor(
    U_red: torch.Tensor,
    r23: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """
    Transform a reduced-basis evolutor into the full flavour-basis evolutor using R23 and the CP phase.
    
    Formula: Uses S = R23 Delta U_red Delta^dagger R23^T.
    
    Args:
        U_red: Reduced-basis evolutor tensor shaped (..., 3, 3).
        r23: R23 rotation matrix tensor shaped (..., 3, 3) or (3, 3).
        delta: CP-phase matrix Delta shaped (..., 3, 3) or (3, 3).
    
    Returns:
        Full flavour-basis evolutor tensor shaped (..., 3, 3).
    """
    return (
        r23
        @ delta
        @ U_red
        @ torch.conj(delta).transpose(-1, -2)
        @ r23.transpose(-1, -2)
    )

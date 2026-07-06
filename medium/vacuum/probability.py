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
Vacuum neutrino oscillation probabilities in pure PyTorch.

This module converts vacuum evolution operators into flavour probabilities.

Functions
---------
vacuum_probability(...)
    Return transition probabilities |S_ab|^2 for all flavour pairs.
pvacuum(...)
    Return final flavour probabilities for either flavour-basis amplitudes or
    mass-basis incoherent weights.
"""



from __future__ import annotations

from typing import Optional

import torch

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.common.probability import (
    probability_from_evolutor,
    probability_transition,
)
from tpeanuts.medium.vacuum.evolutor import _resolve_vacuum_context, vacuum_evolutor
from tpeanuts.util.constant import R_E
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import (
    TensorLike,
    cdtype_from_real,
    state_tensor,
    broadcast_last3,
)


@torch.no_grad()
def vacuum_probability(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    L_km: TensorLike,
    *,
    context: Optional[RuntimeContext] = None,
    evolution_scale_m: TensorLike = R_E,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """
    Compute all vacuum flavour-transition probabilities.

    Args:
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy in MeV.
        L_km: Propagation baseline in km.
        context: Optional runtime device/dtype.
        evolution_scale_m: Positive scale in metres used for kinetic phases.
        legacy_precision: Accepted for API consistency with matter
            propagation. It does not alter vacuum probabilities.

    Returns:
        Real tensor |S_ab|^2 with shape (..., 3, 3). The final two dimensions
        are final flavour and initial flavour.
    """
    S = vacuum_evolutor(
        oscillation,
        E_MeV,
        L_km,
        context=context,
        evolution_scale_m=evolution_scale_m,
        legacy_precision=legacy_precision,
    )

    return probability_transition(S, real_dtype=context.dtype if context is not None else None)


@torch.no_grad()
def pvacuum(
    nustate: TensorLike,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    L_km: TensorLike,
    *,
    massbasis: bool = True,
    context: Optional[RuntimeContext] = None,
    evolution_scale_m: TensorLike = R_E,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """
    Compute final vacuum flavour probabilities for an initial state.

    The function builds the vacuum evolution operator ``S`` and then applies
    one of two constructions, depending on the basis assigned to ``nustate``:

        1. ``massbasis=False`` treats ``nustate`` as coherent flavour-basis
           amplitudes. The state is evolved as ``psi_f = S psi_i`` and the
           returned probabilities are ``P_alpha = |psi_f,alpha|^2``.
        2. ``massbasis=True`` treats ``nustate`` as incoherent mass-basis
           weights ``w_i``. The code builds the mass-to-flavour probability
           matrix from ``S @ U`` and returns
           ``P_alpha = sum_i P(alpha | i) w_i``.

    In both cases the output is a final flavour-probability vector, not a flux.
    The ``massbasis`` flag selects the interpretation of the input state, not
    the basis of the returned probabilities.

    Args:
        nustate: Initial state. When massbasis=False, this is a flavour-basis
            amplitude vector with final dimension 3. When massbasis=True, this
            is interpreted as incoherent mass-basis weights.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy in MeV.
        L_km: Propagation baseline in km.
        massbasis: Selects the interpretation of nustate. True returns the
            incoherent mass-basis probability mixture; False evolves flavour
            amplitudes coherently and squares them.
        context: Optional runtime device/dtype.
        evolution_scale_m: Positive scale in metres used for kinetic phases.
        legacy_precision: Accepted for API consistency with matter
            propagation. It does not alter vacuum probabilities.

    Returns:
        Real tensor of final flavour probabilities with final dimension 3.
    """
    context = _resolve_vacuum_context(context, E_MeV, L_km)
    device, dtype = context.device, context.dtype

    S = vacuum_evolutor(
        oscillation,
        E_MeV,
        L_km,
        context=context,
        evolution_scale_m=evolution_scale_m,
        legacy_precision=legacy_precision,
    )

    if not massbasis:
        state = state_tensor(
            nustate,
            device=device,
            dtype=cdtype_from_real(dtype),
        )
        state = broadcast_last3(state, S.shape[:-2])
    else:
        state = state_tensor(nustate, device=device, dtype=dtype)
        state = broadcast_last3(state, S.shape[:-2])

    return probability_from_evolutor(
        S,
        state,
        pmns=oscillation.pmns,
        massbasis=massbasis,
        antinu=oscillation.antinu,
        real_dtype=dtype,
    )

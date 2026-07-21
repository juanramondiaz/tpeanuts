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
vacuum_probability_transition(...)
    Return transition probabilities |S_ab|^2 for all flavour pairs.
vacuum_probability_state(...)
    Return final flavour probabilities for either flavour-basis amplitudes or
    mass-basis incoherent weights.
vacuum_probability_integrated(...)
    Average final flavour probabilities over energy, weighted by an explicit
    production spectrum.
"""



from __future__ import annotations

from typing import Optional

import torch

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.common.probability import (
    probability_integrated,
    probability_state,
    probability_transition,
)
from tpeanuts.medium.vacuum.evolutor import _resolve_vacuum_context, vacuum_evolutor
from tpeanuts.util.constant import R_E
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import (
    TensorLike,
    cdtype_from_real,
    state_tensor,
    broadcast_flavour_vector,
)


@torch.no_grad()
def vacuum_probability_transition(
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
        Real tensor |S_ab|^2 with shape (..., N, N), N in {3, 4}. The final
        two dimensions are final flavour and initial flavour.
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
def vacuum_probability_state(
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
            amplitude vector with final dimension matching
            ``oscillation.pmns.n_flavours`` (3, or 4 for the 3+1 sterile
            extension). When massbasis=True, this is interpreted as
            incoherent mass-basis weights of the same dimension.
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
        Real tensor of final flavour probabilities with final dimension
        matching ``oscillation.pmns.n_flavours``.
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
        state = broadcast_flavour_vector(state, S.shape[:-2])
    else:
        state = state_tensor(nustate, device=device, dtype=dtype)
        state = broadcast_flavour_vector(state, S.shape[:-2])

    return probability_state(
        S,
        state,
        pmns=oscillation.pmns,
        massbasis=massbasis,
        antinu=oscillation.antinu,
        real_dtype=dtype,
    )


@torch.no_grad()
def vacuum_probability_integrated(
    nustate: TensorLike,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    L_km: TensorLike,
    spectrum: TensorLike,
    *,
    massbasis: bool = True,
    context: Optional[RuntimeContext] = None,
    evolution_scale_m: TensorLike = R_E,
    legacy_precision: bool = False,
    energy_dim: int = -2,
) -> torch.Tensor:
    """Average final vacuum flavour probabilities over energy.

    Builds the energy-resolved probabilities with ``vacuum_probability_state``
    and averages them with ``core.common.probability.probability_integrated``,
    weighted by an explicit production ``spectrum``.

    Args:
        nustate: Initial state passed to ``vacuum_probability_state``.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy grid in MeV, one-dimensional, matching
            ``E_grid_MeV`` of ``probability_integrated``.
        L_km: Propagation baseline in km.
        spectrum: Spectral weight w(E), required (no default).
        massbasis: Selects the interpretation of ``nustate``.
        context: Optional runtime device/dtype.
        evolution_scale_m: Positive scale in metres used for kinetic phases.
        legacy_precision: Accepted for API consistency with matter
            propagation. It does not alter vacuum probabilities.
        energy_dim: Axis of the resulting probability tensor holding the
            energy grid. Must not be the final (flavour) axis.

    Returns:
        Spectrum-weighted average probability, with the energy axis removed.
    """
    probabilities = vacuum_probability_state(
        nustate,
        oscillation,
        E_MeV,
        L_km,
        massbasis=massbasis,
        context=context,
        evolution_scale_m=evolution_scale_m,
        legacy_precision=legacy_precision,
    )

    return probability_integrated(
        probabilities,
        E_MeV,
        spectrum,
        energy_dim=energy_dim,
    )

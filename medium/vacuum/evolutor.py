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
Vacuum neutrino evolution operators in pure PyTorch.

This module implements coherent three-flavour propagation in the vacuum limit.
No matter potential is included: mass eigenstates acquire only kinetic phases
and are then rotated back to the flavour basis with the PMNS matrix.

Functions
---------
vacuum_evolutor(...)
    Build S(L, E) in vacuum using a configurable evolution length scale.
vacuum_evolved_state(...)
    Apply the vacuum evolution operator to an initial flavour state.
"""

from __future__ import annotations

from typing import Optional

import torch

from tpeanuts.core.common.evolutor import apply_evolutor_to_state
from tpeanuts.core.common.hamiltonian import kinetic_eigenvalue_vector
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.util.constant import R_E
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.torch_util import default_device, resolve_dtype
from tpeanuts.util.type import (
    TensorLike,
    as_tensor,
    cdtype_from_real,
    state_tensor,
    broadcast_flavour_vector,
)


def _resolve_vacuum_context(
    context: Optional[RuntimeContext],
    *values: TensorLike,
) -> RuntimeContext:
    if context is not None:
        return context
    return RuntimeContext(
        device=default_device(None),
        dtype=resolve_dtype(None, *values),
    )


@torch.no_grad()
def vacuum_evolutor(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    L_km: TensorLike,
    *,
    context: Optional[RuntimeContext] = None,
    evolution_scale_m: TensorLike = R_E,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """
    Build the vacuum flavour-basis evolution operator.

    The operator is constructed as
    U diag(exp(-i k_i L/evolution_scale_m)) U^\dagger, where k_i are the
    kinetic mass-basis terms. E and L may be scalar or tensor-valued and are
    broadcast with the kinetic vector.

    Args:
        oscillation: Built pmns object plus mass splittings and antinu
            selection. ``oscillation.pmns.n_flavours`` (3 for the Standard
            Model, 4 for the 3+1 sterile extension) sizes the returned
            operator; ``kinetic_eigenvalue_vector`` already extends the
            kinetic vector with a fourth eigenvalue derived from
            ``DeltamSq41`` for a 4-flavour ``pmns``.
        E_MeV: Neutrino energy in MeV.
        L_km: Propagation baseline in km.
        context: Optional runtime device/dtype. If omitted, the device
            defaults to CUDA when available (else CPU) and the dtype is
            inferred from E_MeV or L_km tensors, otherwise float64.
        evolution_scale_m: Positive scale in metres used for both k_i and the
            dimensionless baseline.
        legacy_precision: Accepted for API consistency with matter
            propagation. It does not alter vacuum kinetic phases.

    Returns:
        Complex tensor with shape (..., N, N), N in {3, 4}, where leading
        dimensions follow the broadcast shape of energy, baseline, and mass
        splittings.
    """
    context = _resolve_vacuum_context(context, E_MeV, L_km)
    device, dtype = context.device, context.dtype
    cdtype = cdtype_from_real(dtype)

    E = as_tensor(E_MeV, device=device, dtype=dtype)
    L = as_tensor(L_km, device=device, dtype=dtype)
    scale = as_tensor(evolution_scale_m, device=device, dtype=dtype)
    if torch.any(scale <= 0):
        raise ValueError("evolution_scale_m must be positive.")
    x = L * 1.0e3 / scale

    ki = kinetic_eigenvalue_vector(
        oscillation=oscillation,
        E_MeV=E,
        context=context,
        evolution_scale_m=evolution_scale_m,
        legacy_precision=legacy_precision,
    )

    n_flavours = ki.shape[-1]
    batch_shape = torch.broadcast_shapes(ki.shape[:-1], x.shape)
    ki = torch.broadcast_to(ki, (*batch_shape, n_flavours))
    x = torch.broadcast_to(x, batch_shape)

    phase = torch.exp(-1j * ki.to(dtype=cdtype) * x.unsqueeze(-1).to(dtype=cdtype))
    U = oscillation.pmns.pmns_matrix(antinu=oscillation.antinu).to(device=device, dtype=cdtype)
    Udag = U.conj().transpose(-2, -1)

    return (U * phase[..., None, :]) @ Udag


@torch.no_grad()
def vacuum_evolved_state(
    nustate: TensorLike,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    L_km: TensorLike,
    *,
    context: Optional[RuntimeContext] = None,
    evolution_scale_m: TensorLike = R_E,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """
    Evolve one initial flavour-basis state through vacuum.

    Args:
        nustate: Initial flavour amplitudes with final dimension matching
            ``oscillation.pmns.n_flavours`` (3, or 4 for the 3+1 sterile
            extension). Leading dimensions may be broadcast against the
            evolution operator.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy in MeV.
        L_km: Propagation baseline in km.
        context: Optional runtime device/dtype.
        evolution_scale_m: Positive scale in metres used for kinetic phases.
        legacy_precision: Accepted for API consistency with matter
            propagation. It does not alter vacuum kinetic phases.

    Returns:
        Complex evolved flavour amplitudes with final dimension matching
        ``oscillation.pmns.n_flavours``.
    """
    context = _resolve_vacuum_context(context, E_MeV, L_km)
    device, dtype = context.device, context.dtype
    cdtype = cdtype_from_real(dtype)

    S = vacuum_evolutor(
        oscillation,
        E_MeV,
        L_km,
        context=context,
        evolution_scale_m=evolution_scale_m,
        legacy_precision=legacy_precision,
    )

    state = state_tensor(nustate, device=device, dtype=cdtype)
    state = broadcast_flavour_vector(state, S.shape[:-2])

    return apply_evolutor_to_state(S, state)

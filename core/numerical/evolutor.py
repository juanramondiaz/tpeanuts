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
"""Medium-independent numerical propagation.

Module functions:
    evolutor_numerical_segment(...)
        Build local matrix-exponential evolutors from sampled electron
        densities and dimensionless path increments.
    evolutor_numerical(...)
        Compose numerical segment evolutors into a total operator, optionally
        returning the accumulated history.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Union
import torch

import tpeanuts.util.default as default
from tpeanuts.util.constant import R_E

from tpeanuts.core.BSM.hamiltonian import hamiltonian_flavour_bsm
from tpeanuts.core.common.hamiltonian import hamiltonian_flavour
from tpeanuts.core.common.evolutor import compose_segment_evolutors
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import TensorLike, as_tensor


@torch.no_grad()
def evolutor_numerical_segment(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    n_e_mol_cm3: TensorLike,
    dx_evolution: TensorLike,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = default.dtype,
    evolution_scale_m: TensorLike = R_E,
    epsilon: Optional[torch.Tensor] = None,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """Build one matrix-exponential evolutor per numerical segment.

    Args:
        oscillation: Built pmns object plus mass splittings
            (``DeltamSq21``/``DeltamSq3l``/``DeltamSq41``) and antinu
            selection. ``DeltamSq41`` is required when ``oscillation.pmns``
            is a 3+1 ``PMNS_sterile`` object; ignored otherwise.
        E_MeV: Neutrino energy in MeV. It must be broadcastable with the
            sampled profile without the final segment dimension.
        n_e_mol_cm3: Electron density samples in mol/cm^3. The last dimension
            enumerates the path segments.
        dx_evolution: Dimensionless segment lengths, broadcastable with
            ``n_e_mol_cm3``.
        device: Optional torch device.
        dtype: Real dtype used by Hamiltonian inputs.
        evolution_scale_m: Positive scale in metres used to normalize the
            Hamiltonian.
        epsilon: Optional NSI matrix passed to ``hamiltonian_flavour``.
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor in the Hamiltonian builder.

    Returns:
        Segment evolutors shaped ``(..., n_segments, n_flavours, n_flavours)``.
    """
    pmns = oscillation.pmns

    if device is None:
        device = (
            E_MeV.device
            if isinstance(E_MeV, torch.Tensor)
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(device)
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

    n_e = as_tensor(n_e_mol_cm3, device=device, dtype=dtype)
    dx = as_tensor(dx_evolution, device=device, dtype=dtype)
    if n_e.ndim == 0:
        raise ValueError("n_e_mol_cm3 must include a segment dimension.")
    if dx.shape != n_e.shape:
        dx = torch.broadcast_to(dx, n_e.shape)

    E = as_tensor(E_MeV, device=device, dtype=dtype)
    while E.ndim < n_e.ndim:
        E = E.unsqueeze(-1)

    antinu_steps = oscillation.antinu
    if torch.is_tensor(antinu_steps):
        antinu_steps = antinu_steps.to(device=device, dtype=torch.bool)
        while antinu_steps.ndim < n_e.ndim:
            antinu_steps = antinu_steps.unsqueeze(-1)

    hamiltonian_builder = (
        hamiltonian_flavour_bsm
        if epsilon is not None or int(pmns.n_flavours) != 3
        else hamiltonian_flavour
    )
    hamiltonian_kwargs = {}
    if hamiltonian_builder is hamiltonian_flavour_bsm:
        hamiltonian_kwargs["epsilon"] = epsilon

    oscillation_segment = dataclasses.replace(
        oscillation,
        DeltamSq21=as_tensor(oscillation.DeltamSq21, device=device, dtype=dtype),
        DeltamSq3l=as_tensor(oscillation.DeltamSq3l, device=device, dtype=dtype),
        DeltamSq41=(
            None if oscillation.DeltamSq41 is None
            else as_tensor(oscillation.DeltamSq41, device=device, dtype=dtype)
        ),
        antinu=antinu_steps,
    )

    H = hamiltonian_builder(
        oscillation_segment,
        E,
        n_e,
        context=RuntimeContext(device=device, dtype=dtype),
        evolution_scale_m=evolution_scale_m,
        legacy_precision=legacy_precision,
        **hamiltonian_kwargs,
    )

    U_steps = torch.linalg.matrix_exp(
        -1j * H * dx[..., None, None].to(cdtype)
    )

    return U_steps


@torch.no_grad()
def evolutor_numerical(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    n_e_mol_cm3: TensorLike,
    dx_evolution: TensorLike,
    *,
    return_history: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = default.dtype,
    evolution_scale_m: TensorLike = R_E,
    epsilon: Optional[torch.Tensor] = None,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """Compose numerical segment evolutors over sampled densities.

    Args:
        oscillation: Built pmns object plus mass splittings
            (``DeltamSq21``/``DeltamSq3l``/``DeltamSq41``) and antinu
            selection. ``DeltamSq41`` is required when ``oscillation.pmns``
            is a 3+1 ``PMNS_sterile`` object; ignored otherwise.
        E_MeV: Neutrino energy in MeV. It must be broadcastable with the
            sampled profile without the final segment dimension.
        n_e_mol_cm3: Electron density samples in mol/cm^3. The last dimension
            enumerates the path segments.
        dx_evolution: Dimensionless segment lengths, broadcastable with
            ``n_e_mol_cm3``.
        return_history: If True, return the accumulated operator after each
            segment, with an inserted segment-history dimension. Otherwise
            return only the final operator.
        device: Optional torch device.
        dtype: Real dtype used by Hamiltonian inputs.
        evolution_scale_m: Positive scale in metres used to normalize the
            Hamiltonian.
        epsilon: Optional NSI matrix passed to ``hamiltonian_flavour``.
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor in segment Hamiltonians.

    Returns:
        Complex tensor containing either the final operator with shape
        ``(..., n_flavours, n_flavours)`` or the full history with shape
        ``(..., n+1, n_flavours, n_flavours)``.
    """
    U_steps = evolutor_numerical_segment(
        oscillation,
        E_MeV=E_MeV,
        n_e_mol_cm3=n_e_mol_cm3,
        dx_evolution=dx_evolution,
        device=device,
        dtype=dtype,
        evolution_scale_m=evolution_scale_m,
        epsilon=epsilon,
        legacy_precision=legacy_precision,
    )

    if not return_history:
        return compose_segment_evolutors(
            U_steps,
            segment_dim=-3,
            multiply="left",
        )

    batch_shape = U_steps.shape[:-3]
    n_flavours = U_steps.shape[-1]
    identity = torch.eye(n_flavours, device=U_steps.device, dtype=U_steps.dtype)
    S = identity.expand(*batch_shape, n_flavours, n_flavours).clone()

    S_list = [S.clone()]

    for j in range(U_steps.shape[-3]):
        S = U_steps[..., j, :, :] @ S
        S_list.append(S.clone())

    return torch.stack(S_list, dim=-3)

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
Atmosphere evolution utilities for atmosphere neutrinos.

This module implements the atmosphere part of the propagation:

    production height h  ->  earth surface

The evolution operator can be computed either as vacuum propagation or with an
atmosphere matter density profile.

Module functions:
    
    atmosphere_evolutor(...)
        Segments the atmosphere trajectory, evaluates the density profile,
        exponentiates local Hamiltonians using a configurable evolution scale,
        and composes the evolution operator.
"""




from __future__ import annotations

from typing import Optional
import torch

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import TensorLike, cdtype_from_real
from tpeanuts.core.numerical.evolutor import evolutor_numerical
from tpeanuts.util.torch_util import infer_device_dtype

from tpeanuts.medium.atmosphere.profile import AtmosphereParameters, AtmosphereProfile



# ============================================================
# Atmosphere evolution
# ============================================================

@torch.no_grad()
def atmosphere_evolutor(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    h_km: TensorLike,
    theta_deg: TensorLike,
    depth_km: TensorLike = 0.0,
    *,
    atmosphere: Optional[AtmosphereParameters] = None,
    context: Optional[RuntimeContext] = None,
    epsilon: torch.Tensor | None = None,
    legacy_precision: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the atmosphere evolution operator over a segmented trajectory.

    Args:
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy in MeV. Scalar or tensor.
        h_km: Production altitude in km. Scalar or tensor broadcastable with
            E_MeV and theta_deg.
        theta_deg: Atmosphere zenith angle in degrees.
        depth_km: Detector depth below surface in km.
        atmosphere: Atmosphere density profile construction settings. None
            uses ``AtmosphereParameters()`` defaults.
        context: Optional runtime device/dtype. If omitted, both are inferred
            from the tensor inputs.
        epsilon: Optional NSI matrix passed to the numerical Hamiltonian.
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor in atmosphere segment Hamiltonians.

    Returns:
        Pair (S, x_grid), where S has shape (..., 3, 3) and is the complex
        atmosphere evolution operator, and x_grid is the dimensionless path
        grid L/evolution_scale_m with final dimension atmosphere.nsteps + 1.
    """
    atmosphere = atmosphere or AtmosphereParameters()
    if context is not None:
        dev, dtype = context.device, context.dtype
    else:
        dev, dtype = infer_device_dtype(E_MeV, h_km, theta_deg, depth_km)
    cdtype = cdtype_from_real(dtype)
    resolved_context = RuntimeContext(device=dev, dtype=dtype)

    if atmosphere.nsteps < 1:
        raise ValueError("atmosphere.nsteps must be at least one segment.")

    profile_atmosphere = AtmosphereProfile(
        h_km=h_km,
        theta_deg=theta_deg,
        depth_km=depth_km,
        params=atmosphere,
        context=resolved_context,
    )

    S = evolutor_numerical(
        oscillation,
        E_MeV=E_MeV,
        n_e_mol_cm3=profile_atmosphere.n_e_molcm3,
        dx_evolution=profile_atmosphere.dx_evolution,
        return_history=False,
        device=dev,
        dtype=dtype,
        evolution_scale_m=atmosphere.evolution_scale_m,
        epsilon=epsilon,
        legacy_precision=legacy_precision,
    )

    n_flavours = S.shape[-1]
    identity = torch.eye(n_flavours, device=dev, dtype=cdtype)
    S = torch.where(
        (profile_atmosphere.trajectory.meta["L_atm_km"] <= 0.0)[..., None, None],
        identity.expand(*S.shape[:-2], n_flavours, n_flavours),
        S,
    )

    return S, profile_atmosphere.x


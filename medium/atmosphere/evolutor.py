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
    atmosphere_evolutor_analytical(...)
        Fit a piecewise polynomial atmosphere profile and propagate it with
        the first-order perturbative evolutor.
    atmosphere_evolutor_numerical(...)
        Propagate a sampled atmosphere profile with matrix exponentials.
    atmosphere_evolutor(...)
        Dispatch to the analytical or numerical atmosphere path.
"""




from __future__ import annotations

import dataclasses
from typing import Literal, Optional
import torch

from tpeanuts.core.common.oscillation import OscillationParameters, resolve_include_matter_nc
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import TensorLike, cdtype_from_real
from tpeanuts.core.numerical.evolutor import evolutor_numerical
from tpeanuts.core.common.evolutor import compose_segment_evolutors
from tpeanuts.core.perturbative.evolutor import evolutor_perturbative_segment
from tpeanuts.core.perturbative.models.atmosphere import AtmospherePolynomialProfile
from tpeanuts.medium.atmosphere.density import atmosphere_density
from tpeanuts.medium.atmosphere.geometry import (
    altitude_along_detector_path,
    atmosphere_path_length,
    underground_path_length,
)
from tpeanuts.util.constant import R_E
from tpeanuts.util.type import as_tensor
from tpeanuts.util.torch_util import infer_device_dtype

from tpeanuts.medium.atmosphere.profile import AtmosphereParameters, AtmosphereProfile



# ============================================================
# Atmosphere evolution
# ============================================================

@torch.no_grad()
def atmosphere_evolutor_numerical(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    h_km: TensorLike,
    theta_deg: TensorLike,
    depth_km: TensorLike = 0.0,
    *,
    atmosphere: Optional[AtmosphereParameters] = None,
    context: Optional[RuntimeContext] = None,
    legacy_precision: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the atmosphere evolution operator over a segmented trajectory.

    Args:
        oscillation: Built pmns object plus mass splittings, antinu
            selection, and the optional ``nsi`` (NSIConfig) attribute.
        E_MeV: Neutrino energy in MeV. Scalar or tensor.
        h_km: Production altitude in km. Scalar or tensor broadcastable with
            E_MeV and theta_deg.
        theta_deg: Atmosphere zenith angle in degrees.
        depth_km: Detector depth below surface in km.
        atmosphere: Atmosphere density profile construction settings. None
            uses ``AtmosphereParameters()`` defaults. When
            ``atmosphere.include_matter_nc`` is True, neutron density is also
            sampled and forwarded as ``n_n_mol_cm3``, enabling the 3+1
            sterile extension's neutral-current matter term (only meaningful
            when ``oscillation.pmns`` is 4-flavour).
        context: Optional runtime device/dtype. If omitted, both are inferred
            from the tensor inputs.
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor in atmosphere segment Hamiltonians.

    Returns:
        Pair (S, x_grid), where S has shape (..., N, N), N in {3, 4}, and is
        the complex atmosphere evolution operator, and x_grid is the
        dimensionless path grid L/evolution_scale_m with final dimension
        atmosphere.nsteps + 1.
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

    include_matter_nc = resolve_include_matter_nc(
        atmosphere.include_matter_nc, oscillation,
        has_neutron_data=True,
        context_name="atmosphere_evolutor_numerical",
    )
    atmosphere = dataclasses.replace(atmosphere, include_matter_nc=include_matter_nc)

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
        n_n_mol_cm3=profile_atmosphere.n_n_molcm3,
        return_history=False,
        device=dev,
        dtype=dtype,
        evolution_scale_m=atmosphere.evolution_scale_m,
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


@torch.no_grad()
def atmosphere_evolutor_analytical(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    h_km: TensorLike,
    theta_deg: TensorLike,
    depth_km: TensorLike = 0.0,
    *,
    atmosphere: Optional[AtmosphereParameters] = None,
    context: Optional[RuntimeContext] = None,
    legacy_precision: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute atmosphere evolution with automatically fitted polynomials.

    Args:
        oscillation: PMNS parameters, mass splittings, antineutrino flag,
            and the optional ``nsi`` (NSIConfig) attribute.
        E_MeV: Neutrino energy in MeV.
        h_km: Production altitude in km.
        theta_deg: Detector zenith angle in degrees.
        depth_km: Detector depth below the surface in km.
        atmosphere: Density source and perturbative fit configuration.
            When ``atmosphere.include_matter_nc`` is True, a second
            polynomial is fitted to a neutron-density sample at the same
            nodes and added to each segment model, enabling the 3+1 sterile
            extension's neutral-current matter term (only meaningful when
            ``oscillation.pmns`` is 4-flavour).
        context: Optional runtime device and real dtype.
        legacy_precision: Use the legacy matter-potential prefactor.

    Returns:
        Pair ``(S, x)`` containing the full flavour-basis evolutor and the
        fitted segment-boundary grid in evolution coordinates.
    """
    atmosphere = atmosphere or AtmosphereParameters()
    if context is not None:
        dev, dtype = context.device, context.dtype
    else:
        dev, dtype = infer_device_dtype(E_MeV, h_km, theta_deg, depth_km)
    resolved_context = RuntimeContext(device=dev, dtype=dtype)
    cdtype = cdtype_from_real(dtype)
    n_segments, degree = atmosphere.perturbative_segments, atmosphere.perturbative_degree
    if n_segments < 1 or degree < 0:
        raise ValueError("perturbative_segments must be positive and perturbative_degree non-negative.")

    include_matter_nc = resolve_include_matter_nc(
        atmosphere.include_matter_nc, oscillation,
        has_neutron_data=True,
        context_name="atmosphere_evolutor_analytical",
    )

    h = as_tensor(h_km, device=dev, dtype=dtype)
    theta = as_tensor(theta_deg, device=dev, dtype=dtype)
    depth = as_tensor(depth_km, device=dev, dtype=dtype)
    L_atm = atmosphere_path_length(h, theta, depth, device=dev, dtype=dtype, check_geometry=False)
    L_und = underground_path_length(theta, depth, device=dev, dtype=dtype, check_geometry=False)
    scale_km = as_tensor(atmosphere.evolution_scale_m, device=dev, dtype=dtype) / 1.0e3
    u = torch.linspace(0.0, 1.0, n_segments + 1, device=dev, dtype=dtype)
    boundaries = (L_atm / scale_km)[..., None] * u

    q = (
        torch.zeros(1, device=dev, dtype=dtype)
        if degree == 0
        else torch.linspace(-1.0, 1.0, degree + 1, device=dev, dtype=dtype)
    )
    centres = 0.5 * (boundaries[..., :-1] + boundaries[..., 1:])
    half = 0.5 * (boundaries[..., 1:] - boundaries[..., :-1])
    x_nodes = centres[..., None] + half[..., None] * q
    s_detector_km = L_und[..., None, None] + x_nodes * scale_km
    altitude = altitude_along_detector_path(
        s_detector_km, theta[..., None, None], depth[..., None, None],
        device=dev, dtype=dtype,
    )
    if atmosphere.matter:
        density = atmosphere_density(
            altitude,
            source=atmosphere.atmosphere_density_source,
            density_type="electron_density",
            context=resolved_context,
            **dict(atmosphere.atmosphere_density_kwargs or {}),
        )
        density_n = (
            atmosphere_density(
                altitude,
                source=atmosphere.atmosphere_density_source,
                density_type="neutron_density",
                context=resolved_context,
                **dict(atmosphere.atmosphere_density_kwargs or {}),
            )
            if include_matter_nc
            else None
        )
    else:
        density = torch.zeros_like(altitude)
        density_n = torch.zeros_like(altitude) if include_matter_nc else None

    fitted = AtmospherePolynomialProfile(boundaries, density)
    coefficients_n = (
        AtmospherePolynomialProfile(boundaries, density_n).coefficients
        if density_n is not None
        else None
    )
    model = fitted.segment_model(
        coefficients_n=coefficients_n,
        antinu=oscillation.antinu,
        profile_scale_m=atmosphere.evolution_scale_m,
        evolution_scale_m=atmosphere.evolution_scale_m,
        device=dev,
        dtype=dtype,
        legacy_precision=legacy_precision,
    )
    U_segments = evolutor_perturbative_segment(
        oscillation,
        E_MeV=as_tensor(E_MeV, device=dev, dtype=dtype).unsqueeze(-1),
        profile_model=model,
        evolution_scale_m=atmosphere.evolution_scale_m,
        legacy_precision=legacy_precision,
        include_matter_nc=include_matter_nc,
    )
    U_red = compose_segment_evolutors(U_segments, segment_dim=-3, multiply="left")
    S = oscillation.pmns.flavour_basis(
        U_red,
        antinu=oscillation.antinu,
        device=dev,
        dtype=cdtype,
    )
    n_flavours = S.shape[-1]
    identity = torch.eye(n_flavours, device=dev, dtype=cdtype)
    S = torch.where((L_atm <= 0)[..., None, None], identity.expand(*S.shape[:-2], n_flavours, n_flavours), S)
    return S, boundaries


@torch.no_grad()
def atmosphere_evolutor(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    h_km: TensorLike,
    theta_deg: TensorLike,
    depth_km: TensorLike = 0.0,
    *,
    method: Literal["analytical", "numerical"] = "numerical",
    atmosphere: Optional[AtmosphereParameters] = None,
    context: Optional[RuntimeContext] = None,
    legacy_precision: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch atmosphere propagation to the selected evolution method."""
    if method == "analytical":
        return atmosphere_evolutor_analytical(
            oscillation, E_MeV, h_km, theta_deg, depth_km,
            atmosphere=atmosphere, context=context, legacy_precision=legacy_precision,
        )
    if method == "numerical":
        return atmosphere_evolutor_numerical(
            oscillation, E_MeV, h_km, theta_deg, depth_km,
            atmosphere=atmosphere, context=context,
            legacy_precision=legacy_precision,
        )
    raise ValueError("method must be 'analytical' or 'numerical'.")


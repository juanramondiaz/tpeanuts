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

from tpeanuts.core.common.oscillation import (
    OscillationParameters,
    oscillation_needs_neutron_composition,
    resolve_include_matter_nc,
)
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.math import project_to_unitary
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
    validate_theta_range,
)
from tpeanuts.util.constant import R_E
from tpeanuts.util.type import as_tensor
from tpeanuts.util.torch_util import infer_device_dtype

from tpeanuts.medium.atmosphere.profile import AtmosphereParameters, AtmosphereProfile



# ============================================================
# Atmosphere evolution
# ============================================================

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
            uses ``AtmosphereParameters()`` defaults. Neutron density is
            sampled and forwarded as ``n_n_mol_cm3`` whenever
            ``atmosphere.include_matter_nc`` is True (enabling the 3+1
            sterile extension's neutral-current matter term, only meaningful
            when ``oscillation.pmns`` is 4-flavour) and/or
            ``oscillation.nsi.has_neutron_coupling`` is True (enabling the
            NSI composition term, any flavour count -- see
            ``core.BSM.bsm_nsi``'s "Composition dependence" section); the
            two are independent and both are auto-detected, so no separate
            flag is needed for the second one.
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
    validate_theta_range(theta_deg, device=dev, dtype=dtype)

    if atmosphere.nsteps < 1:
        raise ValueError("atmosphere.nsteps must be at least one segment.")

    include_matter_nc = resolve_include_matter_nc(
        atmosphere.include_matter_nc, oscillation,
        has_neutron_data=True,
        context_name="atmosphere_evolutor_numerical",
    )
    # AtmosphereProfile only reads params.include_matter_nc to decide whether
    # to sample n_n_molcm3 at all -- the sterile NC term and the NSI
    # composition term (core.BSM.bsm_nsi's "Composition dependence" section)
    # both need that same raw neutron-density sample, so both must be able
    # to request it, independent of one another (evolutor_numerical/
    # hamiltonian_reduced apply each term on its own once n_n_mol_cm3 is
    # supplied -- see oscillation_needs_neutron_composition).
    fetch_neutron_density = include_matter_nc or oscillation_needs_neutron_composition(oscillation)
    atmosphere = dataclasses.replace(atmosphere, include_matter_nc=fetch_neutron_density)

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
    analytic_eigenvalues: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute atmosphere evolution with automatically fitted polynomials.

    Args:
        oscillation: PMNS parameters, mass splittings, antineutrino flag,
            and the optional ``nsi`` (NSIConfig) attribute.
        E_MeV: Neutrino energy in MeV.
        h_km: Production altitude in km.
        theta_deg: Detector zenith angle in degrees.
        depth_km: Detector depth below the surface in km.
        atmosphere: Density source and perturbative fit configuration. A
            second polynomial is fitted to a neutron-density sample at the
            same nodes and added to each segment model whenever
            ``atmosphere.include_matter_nc`` is True (3+1 sterile
            neutral-current term, only meaningful when ``oscillation.pmns``
            is 4-flavour) and/or ``oscillation.nsi.has_neutron_coupling`` is
            True (NSI composition term, any flavour count -- see
            ``core.BSM.bsm_nsi``'s "Composition dependence" section); both
            are independent and auto-detected.
        context: Optional runtime device and real dtype.
        legacy_precision: Use the legacy matter-potential prefactor.
        analytic_eigenvalues: If True, compute each segment Hamiltonian's
            eigenvalues with the closed-form Cardano (3-flavour SM/NSI) or
            Ferrari (3+1 sterile extension) solution instead of
            ``torch.linalg.eigvalsh``, forwarded to
            ``core.perturbative.evolutor.evolutor_perturbative_segment``.

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
    validate_theta_range(theta_deg, device=dev, dtype=dtype)
    n_segments, degree = atmosphere.perturbative_segments, atmosphere.perturbative_degree
    if n_segments < 1 or degree < 0:
        raise ValueError("perturbative_segments must be positive and perturbative_degree non-negative.")

    include_matter_nc = resolve_include_matter_nc(
        atmosphere.include_matter_nc, oscillation,
        has_neutron_data=True,
        context_name="atmosphere_evolutor_analytical",
    )
    # Fitting a neutron-density polynomial is needed for the sterile NC term
    # and/or the NSI composition term (see the analogous comment in
    # atmosphere_evolutor_numerical); evolutor_perturbative_segment applies
    # each independently once density_n/coefficients_n is available.
    include_matter_nc = include_matter_nc or oscillation_needs_neutron_composition(oscillation)

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
        analytic_eigenvalues=analytic_eigenvalues,
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
    analytic_eigenvalues: bool = False,
    reunitarize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch atmosphere propagation to the selected evolution method.

    ``analytic_eigenvalues`` (Cardano/Ferrari eigenvalues instead of
    ``eigvalsh``, see ``atmosphere_evolutor_analytical``) only affects
    ``method="analytical"``; it is not applicable to ``method="numerical"``,
    which propagates with ``torch.linalg.matrix_exp`` and never
    diagonalizes.

    ``reunitarize`` applies to both methods: unlike ``analytic_eigenvalues``,
    unitary-projection is a generic post-processing step on the returned
    evolutor S, independent of how S was built (mirrors
    ``medium.earth.evolutor.earth_evolutor``'s ``reunitarize``).

    Args:
        reunitarize: If True, project the returned evolutor onto the nearest
            unitary matrix (``util.math.project_to_unitary``) to absorb
            small numerical drift.

    Raises:
        ValueError: If ``method`` is not ``"analytical"``/``"numerical"``,
            or if ``analytic_eigenvalues=True`` is requested with
            ``method="numerical"`` -- that combination would otherwise
            silently drop the flag instead of raising or applying it, since
            ``atmosphere_evolutor_numerical`` has no such parameter.
    """
    if method == "numerical" and analytic_eigenvalues:
        raise ValueError(
            "analytic_eigenvalues=True has no effect for method='numerical': "
            "only the analytical perturbative evolutor supports closed-form "
            "eigenvalues (method='numerical' propagates via "
            "torch.linalg.matrix_exp and never diagonalizes). Pass "
            "analytic_eigenvalues=False (the default), or use "
            "method='analytical'."
        )
    if method == "analytical":
        S, x = atmosphere_evolutor_analytical(
            oscillation, E_MeV, h_km, theta_deg, depth_km,
            atmosphere=atmosphere, context=context, legacy_precision=legacy_precision,
            analytic_eigenvalues=analytic_eigenvalues,
        )
    elif method == "numerical":
        S, x = atmosphere_evolutor_numerical(
            oscillation, E_MeV, h_km, theta_deg, depth_km,
            atmosphere=atmosphere, context=context,
            legacy_precision=legacy_precision,
        )
    else:
        raise ValueError("method must be 'analytical' or 'numerical'.")

    if reunitarize:
        S = project_to_unitary(S)

    return S, x


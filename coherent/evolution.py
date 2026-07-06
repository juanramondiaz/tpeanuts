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
coherent solar propagation utilities.

This module implements the first coherent solar-to-earth block:

1. Radial coherent propagation from a solar production radius to the solar
   surface using the tabulated solar density profile.
2. Optional coherent vacuum propagation from the solar surface to earth.

Solar profile radii and Hamiltonian evolution may use independent length
scales. Radii are converted from rho = r/R_sun to the configured profile
coordinate; the core then converts them to x_e = r/evolution_scale_m.

Module functions:
    solar_radius_fraction_to_core_x(...)
        Convert solar radius fractions to a configurable evolution coordinate.
    solar_surface_evolutor(...)
        Propagate from a solar production radius to the surface using separate
        profile and evolution scales.
    solar_surface_state(...)
        Apply the solar-surface evolutor to an initial state.
    solar_to_earth_state(...)
        Add configurable-scale coherent vacuum propagation to Earth.
    solar_to_earth_probabilities(...)
        Square the amplitudes returned by ``solar_to_earth_state`` into final
        flavour probabilities.
"""



from __future__ import annotations

from typing import Literal, Union

import torch

from tpeanuts.coherent.coordinates import solar_path_grid
from tpeanuts.core.common.evolutor import compose_segment_evolutors
from tpeanuts.core.common.hamiltonian import (
    hamiltonian_kinetic_reduced,
    hamiltonian_matter_reduced,
)
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.common.potential import matter_potential
from tpeanuts.core.perturbative.evolutor import evolutor_zero_order
from tpeanuts.medium.solar.profile import SolarProfile, build_solar_profile
from tpeanuts.medium.vacuum.evolutor import vacuum_evolutor
from tpeanuts.util.constant import R_E, R_SUN, R_SUN_KM
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import as_tensor, cdtype_from_real, state_tensor

TensorLike = Union[float, int, torch.Tensor]
StateLike = Union[list[complex], tuple[complex, ...], torch.Tensor]
solarMethod = Literal["constant"]


def solar_radius_fraction_to_core_x(
    rho: TensorLike,
    *,
    context: RuntimeContext = RuntimeContext.resolve(None, torch.float64),
    check_bounds: bool = True,
    evolution_scale_m: TensorLike = R_E,
) -> torch.Tensor:
    """Convert rho = r/R_sun to the dimensionless evolution coordinate.

    The Hamiltonian evolution operators in ``core.common`` and
    ``core.perturbative`` are parametrized by a dimensionless path coordinate
    x_e = r / evolution_scale_m (by default normalized to the Earth radius
    R_E, for consistency with the rest of the pipeline), while solar profile
    tables are tabulated in rho = r / R_sun. This helper performs that
    rescaling: x_e = rho * (R_sun / evolution_scale_m).

    Args:
        rho: Dimensionless solar radius fraction(s) in [0, 1].
        context: Runtime device/dtype used to build the result tensor.
        check_bounds: If True, raise ``ValueError`` when any ``rho`` value
            falls outside [0, 1] (within a small numerical tolerance).
        evolution_scale_m: Positive evolution length scale in metres that
            the returned coordinate is normalized by.

    Returns:
        Dimensionless evolution coordinate x_e = rho * (R_sun /
        evolution_scale_m).
    """
    rho_t = as_tensor(rho, device=context.device, dtype=context.dtype)

    if check_bounds:
        if torch.any(rho_t < -1.0e-12).item() or torch.any(rho_t > 1.0 + 1.0e-12).item():
            raise ValueError("rho must be a solar radius fraction in [0, 1].")

    scale = as_tensor(evolution_scale_m, device=context.device, dtype=context.dtype)
    if torch.any(scale <= 0):
        raise ValueError("evolution_scale_m must be positive.")
    return rho_t * (R_SUN / scale)


def _flavour_state(
    flavour: str,
    *,
    context: RuntimeContext,
) -> torch.Tensor:
    labels = {
        "e": 0,
        "nue": 0,
        "electron": 0,
        "mu": 1,
        "numu": 1,
        "muon": 1,
        "tau": 2,
        "nutau": 2,
    }

    key = flavour.lower()
    if key not in labels:
        raise ValueError("flavour must be one of e/nue, mu/numu, tau/nutau.")

    state = torch.zeros(3, device=context.device, dtype=cdtype_from_real(context.dtype))
    state[labels[key]] = 1.0 + 0.0j
    return state


def _squeeze_scalar_batch(
    value: torch.Tensor,
    batch_shape: tuple[int, ...],
) -> torch.Tensor:
    if batch_shape:
        return value.reshape(*batch_shape, 3, 3)
    return value.reshape(3, 3)


def _apply_operator_to_state(
    operator: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    if state.ndim == 1:
        return torch.einsum("...ij,j->...i", operator, state)
    return torch.einsum("...ij,...j->...i", operator, state)


@torch.no_grad()
def solar_surface_evolutor(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    rho0: TensorLike,
    *,
    profile: SolarProfile | None = None,
    method: solarMethod = "constant",
    context: RuntimeContext = RuntimeContext.resolve(None, torch.float64),
    profile_scale_m: TensorLike = R_SUN,
    evolution_scale_m: TensorLike = R_E,
) -> torch.Tensor:
    """Build the production-to-surface solar evolution operator.

    Integrates the coherent flavour-basis Schrodinger equation radially
    outward from a production point at solar radius fraction ``rho0`` to the
    solar surface (rho = 1), using the tabulated B16 electron-density
    profile to build the matter potential at each radial shell. The path
    rho0 -> 1 is discretized with :func:`tpeanuts.coherent.coordinates.solar_path_grid`
    (aligned to the tabulated profile grid), and the per-shell zero-order
    (constant-density) evolutor is composed shell by shell into a single
    production-to-surface evolution operator U_sun, such that
    psi(surface) = U_sun psi(rho0).

    ``profile_scale_m`` defines the coordinate supplied to the core segment
    evolutor, while ``evolution_scale_m`` normalizes its Hamiltonian and phase.
    The physical solar profile remains tabulated in rho = r/R_sun.

    Args:
        oscillation: Built pmns object plus mass splittings (DeltamSq21,
            DeltamSq3l in eV^2) and antinu selection.
        E_MeV: Neutrino energy in MeV. May be scalar or batched.
        rho0: Production-point solar radius fraction(s) in [0, 1]
            (rho0 = 0 is the solar centre, rho0 = 1 is the surface). May be
            scalar or batched.
        profile: Optional SolarProfile providing the tabulated radius and
            electron-density grid. None loads the default B16 profile.
        method: Per-shell integration method. Only "constant" (piecewise
            constant electron density per shell, zero-order evolutor) is
            currently implemented.
        context: Runtime device/dtype used for the calculation.
        profile_scale_m: Positive length scale in metres used to convert the
            solar profile's rho coordinate into the coordinate passed to the
            core segment evolutor. Defaults to the solar radius R_sun.
        evolution_scale_m: Positive evolution length scale in metres used to
            normalize the Hamiltonian and the dimensionless path length of
            each shell (kinetic and matter potential terms). Defaults to the
            Earth radius R_E for consistency with the rest of the pipeline.

    Returns:
        Complex production-to-surface evolution operator with shape
        ``(..., 3, 3)``, where leading dimensions follow the broadcast shape
        of ``E_MeV`` and ``rho0``.
    """
    if method != "constant":
        raise ValueError("Only method='constant' is implemented for now.")

    pmns = oscillation.pmns
    antinu = oscillation.antinu
    dev, dtype = context.device, context.dtype
    E = as_tensor(E_MeV, device=dev, dtype=dtype)
    rho0_t = as_tensor(rho0, device=dev, dtype=dtype)

    solar_profile = build_solar_profile(profile, context=context)
    cdtype = cdtype_from_real(dtype)

    E_shape = tuple(E.shape)
    rho_shape = tuple(rho0_t.shape)
    E_flat = E.reshape(-1)
    rho_flat = rho0_t.reshape(-1)

    if E_flat.numel() == 0 or rho_flat.numel() == 0:
        raise ValueError("E_MeV and rho0 must not be empty.")

    U_by_rho = []
    identity_E = torch.eye(3, device=dev, dtype=cdtype).expand(E_flat.numel(), 3, 3)

    for rho_value in rho_flat:
        grid = solar_path_grid(
            rho_value,
            profile_radius=solar_profile.radius,
            context=context,
        )

        if grid.numel() < 2:
            U_by_rho.append(identity_E)
            continue

        profile_scale = as_tensor(profile_scale_m, device=dev, dtype=dtype)
        if torch.any(profile_scale <= 0):
            raise ValueError("profile_scale_m must be positive.")
        x_grid = grid * (R_SUN / profile_scale)
        x1 = x_grid[:-1]
        x2 = x_grid[1:]
        rho_mid = 0.5 * (grid[:-1] + grid[1:])
        ne_mid = solar_profile.electron_density(rho_mid)

        E_segments = E_flat[:, None].expand(-1, ne_mid.numel())
        Ured = pmns.reduced(antinu=antinu)
        Hkin, ki = hamiltonian_kinetic_reduced(
            DeltamSq21=oscillation.DeltamSq21,
            DeltamSq3l=oscillation.DeltamSq3l,
            E_MeV=E_segments,
            Ured=Ured,
            evolution_scale_m=evolution_scale_m,
            return_ki=True,
        )
        V = matter_potential(
            ne_mid.reshape(1, -1),
            antinu=antinu,
            evolution_scale_m=evolution_scale_m,
        )
        H = Hkin + hamiltonian_matter_reduced(
            V,
            context=RuntimeContext(device=Hkin.device, dtype=Hkin.real.dtype),
        )
        trace_H = (ki.sum(dim=-1) + V).to(dtype=H.dtype)
        L = (x2 - x1).reshape(1, -1) * (
            as_tensor(profile_scale_m, device=dev, dtype=dtype)
            / as_tensor(evolution_scale_m, device=dev, dtype=dtype)
        )
        U_segments = evolutor_zero_order(
            H,
            L,
            trace_H=trace_H,
        )

        U_by_rho.append(compose_segment_evolutors(U_segments, segment_dim=1))

    U = torch.stack(U_by_rho, dim=1)
    batch_shape = E_shape + rho_shape
    return _squeeze_scalar_batch(U, batch_shape)


@torch.no_grad()
def solar_surface_state(
    initial_state: StateLike | str,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    rho0: TensorLike,
    *,
    profile: SolarProfile | None = None,
    context: RuntimeContext = RuntimeContext.resolve(None, torch.float64),
    profile_scale_m: TensorLike = R_SUN,
    evolution_scale_m: TensorLike = R_E,
) -> torch.Tensor:
    """Propagate an initial coherent flavour state to the solar surface.

    Builds the production-to-surface evolution operator with
    :func:`solar_surface_evolutor` and applies it to ``initial_state``,
    returning the coherent flavour-amplitude vector at the solar surface,
    psi(surface) = U_sun psi(rho0).

    Args:
        initial_state: Initial coherent flavour-basis amplitude vector with
            final dimension 3, or one of the flavour-label strings accepted
            by the internal flavour lookup ("e"/"nue"/"electron",
            "mu"/"numu"/"muon", "tau"/"nutau"), which is mapped to the
            corresponding unit amplitude vector.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy in MeV. May be scalar or batched.
        rho0: Production-point solar radius fraction(s) in [0, 1]. May be
            scalar or batched.
        profile: Optional SolarProfile providing the tabulated electron
            density. None loads the default B16 profile.
        context: Runtime device/dtype used for the calculation.
        profile_scale_m: Positive length scale in metres used for the solar
            profile coordinate (see :func:`solar_surface_evolutor`).
        evolution_scale_m: Positive evolution length scale in metres used to
            normalize the Hamiltonian (see :func:`solar_surface_evolutor`).

    Returns:
        Complex coherent flavour-amplitude vector at the solar surface, with
        final dimension 3 and leading dimensions following the broadcast
        shape of ``E_MeV`` and ``rho0``.
    """
    cdtype = cdtype_from_real(context.dtype)

    if isinstance(initial_state, str):
        state0 = _flavour_state(initial_state, context=context)
    else:
        state0 = state_tensor(initial_state, device=context.device, dtype=cdtype)

    U_sun = solar_surface_evolutor(
        oscillation,
        E_MeV,
        rho0,
        profile=profile,
        context=context,
        profile_scale_m=profile_scale_m,
        evolution_scale_m=evolution_scale_m,
    )

    return _apply_operator_to_state(U_sun, state0)


@torch.no_grad()
def solar_to_earth_state(
    initial_state: StateLike | str,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    rho0: TensorLike,
    earth_distance_km: TensorLike,
    *,
    profile: SolarProfile | None = None,
    subtract_solar_radius: bool = True,
    context: RuntimeContext = RuntimeContext.resolve(None, torch.float64),
    profile_scale_m: TensorLike = R_SUN,
    evolution_scale_m: TensorLike = R_E,
) -> torch.Tensor:
    """Propagate a coherent flavour state from the Sun to Earth.

    Chains two coherent propagation stages:

        1. ``solar_surface_state`` integrates from the production point
           ``rho0`` to the solar surface through the tabulated solar
           density, producing psi(surface).
        2. A vacuum evolution operator (:func:`tpeanuts.medium.vacuum.evolutor.vacuum_evolutor`)
           propagates psi(surface) over the remaining vacuum baseline from
           the solar surface to the detector, ``L_vac = earth_distance_km -
           R_sun`` (when ``subtract_solar_radius=True``), producing the
           final coherent flavour-amplitude vector psi(Earth) = U_vac
           psi(surface).

    No Earth matter regeneration is applied here; this function only
    accounts for the Sun-to-Earth vacuum leg following matter-affected
    propagation inside the Sun.

    Args:
        initial_state: Initial coherent flavour-basis amplitude vector with
            final dimension 3, or a flavour-label string (see
            :func:`solar_surface_state`).
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy in MeV. May be scalar or batched.
        rho0: Production-point solar radius fraction(s) in [0, 1]. May be
            scalar or batched.
        earth_distance_km: Sun-to-Earth (centre-to-detector) distance in km.
            Must be at least one solar radius when
            ``subtract_solar_radius=True``.
        profile: Optional SolarProfile providing the tabulated electron
            density. None loads the default B16 profile.
        subtract_solar_radius: If True, the vacuum baseline used for the
            second stage is ``earth_distance_km - R_sun`` (i.e.
            ``earth_distance_km`` is measured from the solar centre). If
            False, ``earth_distance_km`` is used directly as the vacuum
            baseline (i.e. it is already measured from the solar surface).
        context: Runtime device/dtype used for the calculation.
        profile_scale_m: Positive length scale in metres used for the solar
            profile coordinate in the production-to-surface stage.
        evolution_scale_m: Positive evolution length scale in metres used to
            normalize both the production-to-surface and the vacuum-leg
            Hamiltonians.

    Returns:
        Complex coherent flavour-amplitude vector at the detector, with
        final dimension 3 and leading dimensions following the broadcast
        shape of ``E_MeV`` and ``rho0``.

    Raises:
        ValueError: If the resulting vacuum baseline ``L_vac_km`` is
            negative, i.e. ``earth_distance_km`` is smaller than one solar
            radius while ``subtract_solar_radius=True``.
    """
    dev, dtype = context.device, context.dtype
    cdtype = cdtype_from_real(dtype)
    E = as_tensor(E_MeV, device=dev, dtype=dtype)
    rho0_t = as_tensor(rho0, device=dev, dtype=dtype)
    distance = as_tensor(earth_distance_km, device=dev, dtype=dtype)

    psi_surface = solar_surface_state(
        initial_state,
        oscillation,
        E_MeV,
        rho0_t,
        profile=profile,
        context=context,
        profile_scale_m=profile_scale_m,
        evolution_scale_m=evolution_scale_m,
    )

    surface_radius_km = R_SUN_KM * (1.0 - rho0_t * 0.0)
    L_vac_km = distance - surface_radius_km if subtract_solar_radius else distance
    if torch.any(L_vac_km < 0.0).item():
        raise ValueError("earth_distance_km must be at least one solar radius when subtract_solar_radius=True.")

    E_vac = E
    L_vac_input = L_vac_km
    if E.ndim > 0 and rho0_t.ndim > 0:
        E_vac = E.reshape(*E.shape, *((1,) * rho0_t.ndim))
        L_vac_input = L_vac_km.reshape(*((1,) * E.ndim), *rho0_t.shape)

    U_vac = vacuum_evolutor(
        oscillation,
        E_vac,
        L_vac_input,
        context=context,
        evolution_scale_m=evolution_scale_m,
    )

    if rho0_t.ndim > 0 and U_vac.ndim == 2:
        U_vac = U_vac.reshape(*((1,) * rho0_t.ndim), 3, 3)

    return _apply_operator_to_state(U_vac.to(dtype=cdtype), psi_surface)


@torch.no_grad()
def solar_to_earth_probabilities(*args, **kwargs) -> torch.Tensor:
    """Compute final flavour probabilities for coherent Sun-to-Earth propagation.

    Thin wrapper around :func:`solar_to_earth_state` that squares the
    returned coherent flavour amplitudes, P_alpha = |psi(Earth)_alpha|^2.
    Accepts the same positional and keyword arguments as
    :func:`solar_to_earth_state`.

    Returns:
        Real flavour-probability tensor with final dimension 3, summing to 1
        over that dimension (up to numerical precision), and leading
        dimensions following the broadcast shape of the underlying ``E_MeV``
        and ``rho0`` arguments.
    """
    psi = solar_to_earth_state(*args, **kwargs)
    return torch.abs(psi) ** 2

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

The peanuts core Hamiltonian uses dimensionless distances x = L / R_E.  solar
model radii are therefore converted from rho = r / R_sun into x before calling
the core segment evolutors.
"""



from __future__ import annotations

from typing import Literal, Optional, Union

import torch

from tpeanuts.coherent.coordinates import solar_path_grid
from tpeanuts.core.segment_evolution import compose_segment_evolutors, constant_density_segment_evolutor
from tpeanuts.solar.profiles import SolarProfile, load_default_solar_profile
from tpeanuts.vacuum.probabilities import vacuum_evolutor
from tpeanuts.util.constant import R_E, R_SUN_KM
from tpeanuts.util.torch_util import _default_device
from tpeanuts.util.type import _as_tensor, _cdtype_from_real, _state_tensor

TensorLike = Union[float, int, torch.Tensor]
StateLike = Union[list[complex], tuple[complex, ...], torch.Tensor]
solarMethod = Literal["constant"]


def solar_radius_fraction_to_core_x(
    rho: TensorLike,
    *,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float64,
    check_bounds: bool = True,
) -> torch.Tensor:
    dev = _default_device(device)
    rho_t = _as_tensor(rho, device=dev, dtype=dtype)

    if check_bounds:
        if torch.any(rho_t < -1.0e-12).item() or torch.any(rho_t > 1.0 + 1.0e-12).item():
            raise ValueError("rho must be a solar radius fraction in [0, 1].")

    return rho_t * (R_SUN_KM * 1.0e3 / R_E)


def _flavour_state(
    flavour: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
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

    state = torch.zeros(3, device=device, dtype=_cdtype_from_real(dtype))
    state[labels[key]] = 1.0 + 0.0j
    return state


def _prepare_profile(
    profile: SolarProfile | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> SolarProfile:
    if profile is None:
        return load_default_solar_profile(device=device, dtype=dtype)

    if profile.radius.device == device and profile.radius.dtype == dtype:
        return profile

    return SolarProfile(
        radius=profile.radius.to(device=device, dtype=dtype),
        density=profile.density.to(device=device, dtype=dtype),
        fractions={key: value.to(device=device, dtype=dtype) for key, value in profile.fractions.items()},
        fluxes={key: value.to(device=device, dtype=dtype) for key, value in profile.fluxes.items()},
    )


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
    pmns: object,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    E_MeV: TensorLike,
    rho0: TensorLike,
    *,
    profile: SolarProfile | None = None,
    antinu: Union[bool, torch.Tensor] = False,
    method: solarMethod = "constant",
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    if method != "constant":
        raise ValueError("Only method='constant' is implemented for now.")

    dev = _default_device(device)
    E = _as_tensor(E_MeV, device=dev, dtype=dtype)
    rho0_t = _as_tensor(rho0, device=dev, dtype=dtype)

    solar_profile = _prepare_profile(profile, device=dev, dtype=dtype)
    cdtype = _cdtype_from_real(dtype)

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
            device=dev,
            dtype=dtype,
        )

        if grid.numel() < 2:
            U_by_rho.append(identity_E)
            continue

        x_grid = solar_radius_fraction_to_core_x(grid, device=dev, dtype=dtype, check_bounds=False)
        x1 = x_grid[:-1]
        x2 = x_grid[1:]
        rho_mid = 0.5 * (grid[:-1] + grid[1:])
        ne_mid = solar_profile.electron_density(rho_mid)
        zeros = torch.zeros_like(ne_mid)

        E_segments = E_flat[:, None].expand(-1, ne_mid.numel())
        U_segments = constant_density_segment_evolutor(
            DeltamSq21=DeltamSq21,
            DeltamSq3l=DeltamSq3l,
            pmns=pmns,
            E_MeV=E_segments,
            x1=x1.reshape(1, -1),
            x2=x2.reshape(1, -1),
            a=ne_mid.reshape(1, -1),
            b=zeros.reshape(1, -1),
            c=zeros.reshape(1, -1),
            antinu=antinu,
        )

        U_by_rho.append(compose_segment_evolutors(U_segments, segment_dim=1))

    U = torch.stack(U_by_rho, dim=1)
    batch_shape = E_shape + rho_shape
    return _squeeze_scalar_batch(U, batch_shape)


@torch.no_grad()
def solar_surface_state(
    initial_state: StateLike | str,
    pmns: object,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    E_MeV: TensorLike,
    rho0: TensorLike,
    *,
    profile: SolarProfile | None = None,
    antinu: Union[bool, torch.Tensor] = False,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    dev = _default_device(device)
    cdtype = _cdtype_from_real(dtype)

    if isinstance(initial_state, str):
        state0 = _flavour_state(initial_state, device=dev, dtype=dtype)
    else:
        state0 = _state_tensor(initial_state, device=dev, dtype=cdtype)

    U_sun = solar_surface_evolutor(
        pmns,
        DeltamSq21,
        DeltamSq3l,
        E_MeV,
        rho0,
        profile=profile,
        antinu=antinu,
        device=dev,
        dtype=dtype,
    )

    return _apply_operator_to_state(U_sun, state0)


@torch.no_grad()
def solar_to_earth_state(
    initial_state: StateLike | str,
    pmns: object,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    E_MeV: TensorLike,
    rho0: TensorLike,
    earth_distance_km: TensorLike,
    *,
    profile: SolarProfile | None = None,
    antinu: Union[bool, torch.Tensor] = False,
    subtract_solar_radius: bool = True,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    dev = _default_device(device)
    cdtype = _cdtype_from_real(dtype)
    E = _as_tensor(E_MeV, device=dev, dtype=dtype)
    rho0_t = _as_tensor(rho0, device=dev, dtype=dtype)
    distance = _as_tensor(earth_distance_km, device=dev, dtype=dtype)

    psi_surface = solar_surface_state(
        initial_state,
        pmns,
        DeltamSq21,
        DeltamSq3l,
        E_MeV,
        rho0_t,
        profile=profile,
        antinu=antinu,
        device=dev,
        dtype=dtype,
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
        pmns,
        DeltamSq21,
        DeltamSq3l,
        E_vac,
        L_vac_input,
        antinu=antinu,
        device=dev,
        dtype=dtype,
    )

    if rho0_t.ndim > 0 and U_vac.ndim == 2:
        U_vac = U_vac.reshape(*((1,) * rho0_t.ndim), 3, 3)

    return _apply_operator_to_state(U_vac.to(dtype=cdtype), psi_surface)


@torch.no_grad()
def solar_to_earth_probabilities(*args, **kwargs) -> torch.Tensor:
    psi = solar_to_earth_state(*args, **kwargs)
    return torch.abs(psi) ** 2

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
solar coordinate conversion helpers for coherent propagation.

The coherent solar evolutor will integrate from a production point inside the
Sun to the solar surface.  solar model files use the natural coordinate

    rho = r / R_sun

while Hamiltonian evolution needs physical path lengths.  This module keeps
that conversion explicit and torch-native.
"""



from __future__ import annotations

from typing import Literal, Optional, Union

import torch

from tpeanuts.util.constant import R_SUN, R_SUN_KM
from tpeanuts.util.torch_util import _default_device
from tpeanuts.util.type import _as_tensor

TensorLike = Union[float, int, torch.Tensor]
DistanceUnit = Literal["m", "km"]


def _solar_radius_value(unit: DistanceUnit, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if unit == "m":
        return torch.tensor(R_SUN, device=device, dtype=dtype)
    if unit == "km":
        return torch.tensor(R_SUN_KM, device=device, dtype=dtype)
    raise ValueError("unit must be either 'm' or 'km'.")


def _check_radius_fraction(rho: torch.Tensor, *, name: str = "rho") -> None:
    if torch.any(rho < -1.0e-12).item() or torch.any(rho > 1.0 + 1.0e-12).item():
        raise ValueError(f"{name} must be a solar radius fraction in [0, 1].")


def solar_radius_fraction_to_distance(
    rho: TensorLike,
    *,
    unit: DistanceUnit = "m",
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float64,
    check_bounds: bool = True,
) -> torch.Tensor:
    dev = _default_device(device)
    rho_t = _as_tensor(rho, device=dev, dtype=dtype)

    if check_bounds:
        _check_radius_fraction(rho_t)

    return rho_t * _solar_radius_value(unit, device=dev, dtype=dtype)


def distance_to_solar_radius_fraction(
    distance: TensorLike,
    *,
    unit: DistanceUnit = "m",
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float64,
    check_bounds: bool = True,
) -> torch.Tensor:
    dev = _default_device(device)
    distance_t = _as_tensor(distance, device=dev, dtype=dtype)
    rho = distance_t / _solar_radius_value(unit, device=dev, dtype=dtype)

    if check_bounds:
        _check_radius_fraction(rho, name="distance / R_sun")

    return rho


def production_to_surface_path_length(
    rho0: TensorLike,
    *,
    unit: DistanceUnit = "m",
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float64,
    check_bounds: bool = True,
) -> torch.Tensor:
    dev = _default_device(device)
    rho0_t = _as_tensor(rho0, device=dev, dtype=dtype)

    if check_bounds:
        _check_radius_fraction(rho0_t, name="rho0")

    return (1.0 - rho0_t) * _solar_radius_value(unit, device=dev, dtype=dtype)


def solar_shell_widths(
    rho_grid: TensorLike,
    *,
    unit: DistanceUnit = "m",
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float64,
    check_bounds: bool = True,
) -> torch.Tensor:
    dev = _default_device(device)
    rho = _as_tensor(rho_grid, device=dev, dtype=dtype)

    if rho.ndim != 1:
        raise ValueError("rho_grid must be one-dimensional.")
    if rho.numel() < 2:
        raise ValueError("rho_grid must contain at least two points.")
    if torch.any(torch.diff(rho) < 0.0).item():
        raise ValueError("rho_grid must be monotonically increasing.")
    if check_bounds:
        _check_radius_fraction(rho, name="rho_grid")

    return torch.diff(rho) * _solar_radius_value(unit, device=dev, dtype=dtype)


def solar_path_grid(
    rho0: TensorLike,
    *,
    nsteps: Optional[int] = None,
    profile_radius: Optional[TensorLike] = None,
    include_surface: bool = True,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float64,
    check_bounds: bool = True,
) -> torch.Tensor:
    dev = _default_device(device)
    rho0_t = _as_tensor(rho0, device=dev, dtype=dtype)

    if rho0_t.ndim != 0:
        raise ValueError("rho0 must be a scalar for solar_path_grid.")
    if check_bounds:
        _check_radius_fraction(rho0_t, name="rho0")

    if profile_radius is not None:
        profile = _as_tensor(profile_radius, device=dev, dtype=dtype)
        if profile.ndim != 1:
            raise ValueError("profile_radius must be one-dimensional.")
        if profile.numel() < 2:
            raise ValueError("profile_radius must contain at least two points.")
        if torch.any(torch.diff(profile) < 0.0).item():
            raise ValueError("profile_radius must be monotonically increasing.")
        if check_bounds:
            _check_radius_fraction(profile, name="profile_radius")

        selected = profile[profile > rho0_t]
        grid = torch.cat([rho0_t.reshape(1), selected])
    else:
        if nsteps is None:
            nsteps = 256
        if nsteps < 1:
            raise ValueError("nsteps must be at least 1.")
        grid = torch.linspace(rho0_t, 1.0, nsteps + 1, device=dev, dtype=dtype)

    if include_surface and not torch.isclose(grid[-1], torch.tensor(1.0, device=dev, dtype=dtype)):
        grid = torch.cat([grid, torch.ones(1, device=dev, dtype=dtype)])

    return torch.clamp(grid, 0.0, 1.0)

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

Module functions:
    solar_radius_fraction_to_distance(...)
        Convert a solar radius fraction rho = r/R_sun to a physical distance.
    distance_to_solar_radius_fraction(...)
        Convert a physical distance from Sun centre to a radius fraction rho.
    production_to_surface_path_length(...)
        Compute the radial path length from a production point at rho0 to the
        solar surface (rho = 1).
    solar_shell_widths(...)
        Convert consecutive differences of a radius-fraction grid into
        physical shell widths.
    solar_path_grid(...)
        Build a monotonically increasing radius-fraction grid from a
        production point rho0 to the solar surface, either evenly spaced or
        aligned with a tabulated solar profile grid.
"""



from __future__ import annotations

from typing import Literal, Optional, Union

import torch

from tpeanuts.util.constant import R_SUN, R_SUN_KM
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import as_tensor

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
    context: RuntimeContext = RuntimeContext.resolve(None, torch.float64),
    check_bounds: bool = True,
) -> torch.Tensor:
    """Convert a solar radius fraction rho = r/R_sun to a physical distance.

    Args:
        rho: Dimensionless solar radius fraction(s) in [0, 1], where rho = 0
            is the solar centre and rho = 1 is the solar surface.
        unit: Output distance unit, either "m" (metres) or "km" (kilometres).
        context: Runtime device/dtype used to build the result tensor.
        check_bounds: If True, raise ``ValueError`` when any ``rho`` value
            falls outside [0, 1] (within a small numerical tolerance).

    Returns:
        Physical distance from the solar centre, ``r = rho * R_sun``, in the
        requested unit.
    """
    rho_t = as_tensor(rho, device=context.device, dtype=context.dtype)

    if check_bounds:
        _check_radius_fraction(rho_t)

    return rho_t * _solar_radius_value(unit, device=context.device, dtype=context.dtype)


def distance_to_solar_radius_fraction(
    distance: TensorLike,
    *,
    unit: DistanceUnit = "m",
    context: RuntimeContext = RuntimeContext.resolve(None, torch.float64),
    check_bounds: bool = True,
) -> torch.Tensor:
    """Convert a physical distance from the solar centre to rho = r/R_sun.

    Inverse of :func:`solar_radius_fraction_to_distance`.

    Args:
        distance: Physical distance(s) from the solar centre, in the unit
            given by ``unit``.
        unit: Unit of ``distance``, either "m" (metres) or "km" (kilometres).
        context: Runtime device/dtype used to build the result tensor.
        check_bounds: If True, raise ``ValueError`` when the resulting
            fraction falls outside [0, 1] (within a small numerical
            tolerance), i.e. when ``distance`` is not between the solar
            centre and the solar surface.

    Returns:
        Dimensionless solar radius fraction rho = distance / R_sun.
    """
    distance_t = as_tensor(distance, device=context.device, dtype=context.dtype)
    rho = distance_t / _solar_radius_value(unit, device=context.device, dtype=context.dtype)

    if check_bounds:
        _check_radius_fraction(rho, name="distance / R_sun")

    return rho


def production_to_surface_path_length(
    rho0: TensorLike,
    *,
    unit: DistanceUnit = "m",
    context: RuntimeContext = RuntimeContext.resolve(None, torch.float64),
    check_bounds: bool = True,
) -> torch.Tensor:
    """Compute the radial path length from a production point to the surface.

    Assumes purely radial propagation, so the path length from a production
    point at radius fraction ``rho0`` to the solar surface (rho = 1) is
    simply ``(1 - rho0) * R_sun``.

    Args:
        rho0: Production-point solar radius fraction(s) in [0, 1].
        unit: Output distance unit, either "m" (metres) or "km" (kilometres).
        context: Runtime device/dtype used to build the result tensor.
        check_bounds: If True, raise ``ValueError`` when any ``rho0`` value
            falls outside [0, 1] (within a small numerical tolerance).

    Returns:
        Radial physical path length from ``rho0`` to the solar surface, in
        the requested unit.
    """
    rho0_t = as_tensor(rho0, device=context.device, dtype=context.dtype)

    if check_bounds:
        _check_radius_fraction(rho0_t, name="rho0")

    return (1.0 - rho0_t) * _solar_radius_value(unit, device=context.device, dtype=context.dtype)


def solar_shell_widths(
    rho_grid: TensorLike,
    *,
    unit: DistanceUnit = "m",
    context: RuntimeContext = RuntimeContext.resolve(None, torch.float64),
    check_bounds: bool = True,
) -> torch.Tensor:
    """Convert a radius-fraction grid into physical radial shell widths.

    Args:
        rho_grid: One-dimensional, monotonically increasing grid of solar
            radius fractions in [0, 1], with at least two points.
        unit: Output distance unit, either "m" (metres) or "km" (kilometres).
        context: Runtime device/dtype used to build the result tensor.
        check_bounds: If True, raise ``ValueError`` when any grid value falls
            outside [0, 1] (within a small numerical tolerance).

    Returns:
        Physical widths ``diff(rho_grid) * R_sun`` of the ``n - 1`` shells
        bounded by the ``n`` grid points, in the requested unit.
    """
    rho = as_tensor(rho_grid, device=context.device, dtype=context.dtype)

    if rho.ndim != 1:
        raise ValueError("rho_grid must be one-dimensional.")
    if rho.numel() < 2:
        raise ValueError("rho_grid must contain at least two points.")
    if torch.any(torch.diff(rho) < 0.0).item():
        raise ValueError("rho_grid must be monotonically increasing.")
    if check_bounds:
        _check_radius_fraction(rho, name="rho_grid")

    return torch.diff(rho) * _solar_radius_value(unit, device=context.device, dtype=context.dtype)


def solar_path_grid(
    rho0: TensorLike,
    *,
    nsteps: Optional[int] = None,
    profile_radius: Optional[TensorLike] = None,
    include_surface: bool = True,
    context: RuntimeContext = RuntimeContext.resolve(None, torch.float64),
    check_bounds: bool = True,
) -> torch.Tensor:
    """Build a radius-fraction grid from a production point to the surface.

    Two construction modes are supported:

        ``profile_radius`` given
            The grid is the subset of ``profile_radius`` strictly greater
            than ``rho0``, prefixed by ``rho0`` itself. This aligns the
            integration grid with the tabulated solar density profile so
            each shell uses an electron density sample directly from the
            model.
        ``profile_radius`` omitted
            The grid is ``nsteps + 1`` evenly spaced points between ``rho0``
            and 1 (default ``nsteps = 256``).

    Args:
        rho0: Scalar production-point solar radius fraction in [0, 1].
        nsteps: Number of evenly spaced steps to use when ``profile_radius``
            is not given. Ignored otherwise. Must be at least 1.
        profile_radius: Optional one-dimensional, monotonically increasing
            radius-fraction grid (e.g. the tabulated solar profile grid) used
            to align the returned grid with existing density samples.
        include_surface: If True, append rho = 1 to the grid when it is not
            already (approximately) present, guaranteeing the path reaches
            the solar surface.
        context: Runtime device/dtype used to build the result tensor.
        check_bounds: If True, raise ``ValueError`` when ``rho0`` or
            ``profile_radius`` values fall outside [0, 1] (within a small
            numerical tolerance).

    Returns:
        One-dimensional, monotonically increasing radius-fraction grid from
        ``rho0`` to 1 (inclusive), clamped to [0, 1].
    """
    dev, dtype = context.device, context.dtype
    rho0_t = as_tensor(rho0, device=dev, dtype=dtype)

    if rho0_t.ndim != 0:
        raise ValueError("rho0 must be a scalar for solar_path_grid.")
    if check_bounds:
        _check_radius_fraction(rho0_t, name="rho0")

    if profile_radius is not None:
        profile = as_tensor(profile_radius, device=dev, dtype=dtype)
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

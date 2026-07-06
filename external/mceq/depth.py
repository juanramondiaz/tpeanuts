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
Atmosphere-depth utilities.

This module contains all transformations related to:

    h <-> X(h, theta)

and numerical derivatives such as:

    dX/dh

The Atmosphere depth (also called slant depth when theta != 0) is
defined as

    X(h) = \int_h^\infty rho(h') dh'

with units:

    X : g/cm^2
    rho : g/cm^3

Physically, X(h) is the column mass of air ("how much atmosphere") that
a particle travelling along its trajectory has yet to cross before
reaching altitude h. It is the natural depth coordinate used by MCEq's
cascade-equation solver: the transport/interaction probability for a
particle in an air shower scales with the amount of matter traversed,
not with geometric altitude. For an inclined trajectory at zenith angle
theta, geometric (vertical) depth must be divided by cos(theta) to
obtain the slant depth actually crossed.

Most functions below are thin tpeanuts-native tensor algebra (trapezoid
integration, finite differences, the cos(theta) projection). Only
compute_vertical_depth_from_mceq, compute_slant_depth_from_mceq and
height_to_slant_depth call into an MCEq object (via
tpeanuts.external.mceq.density), either using MCEq's own native
get_mass_overburden method directly, or falling back to integrating
MCEq's mass-density profile with compute_vertical_depth_from_density
when the native method is unavailable.

Module functions:
    theta_deg_to_cos:
        Convert a zenith angle in degrees to cos(theta), validating
        that the trajectory is not exactly horizontal or upward-going.
    compute_vertical_depth_from_density:
        Integrate a tabulated mass-density profile rho(h) to obtain the
        vertical Atmosphere depth X(h) via the trapezoidal rule.
    compute_vertical_depth_from_mceq:
        Obtain the vertical Atmosphere depth X(h) at theta=0 from an
        MCEq density model, preferring MCEq's native overburden method
        and falling back to numerical integration of its density.
    compute_slant_depth_from_vertical_depth:
        Project a vertical depth profile onto an inclined trajectory by
        dividing by cos(theta).
    compute_slant_depth_from_mceq:
        Combine compute_vertical_depth_from_mceq and the cos(theta)
        projection to obtain the slant depth X(h, theta) from MCEq.
    compute_dXdh:
        Compute the numerical derivative dX/dh of a depth profile with
        respect to altitude.
    height_to_slant_depth:
        Convenience wrapper around compute_slant_depth_from_mceq.
"""



from __future__ import annotations

from typing import Optional, Union

import torch

from tpeanuts.util.type import as_tensor
from tpeanuts.util.torch_util import default_device

from tpeanuts.external.mceq.config import (
    MCEqModelConfig,
)

from tpeanuts.external.mceq.core import init_mceq

from tpeanuts.external.mceq.density import (
    atmosphere_mass_density_profile_from_mceq,
    atmosphere_mass_overburden_profile_from_mceq,
)


TensorLike = Union[float, int, torch.Tensor]


# ============================================================
# Utilities
# ============================================================

def theta_deg_to_cos(theta_deg: TensorLike) -> float:
    """
    Convert a zenith angle in degrees to cos(theta).

    theta is the standard zenith angle convention used throughout this
    package: theta=0 is a vertically downward-going trajectory and
    theta increases towards the horizon. cos(theta) is required to be
    strictly positive because a slant depth X(h, theta) = X(h) /
    cos(theta) is only physically defined (finite) for trajectories that
    are not horizontal or upward-going.

    Args:
        theta_deg: Scalar or tensor-like zenith angle(s) in degrees.

    Returns:
        cos(theta) as a Python float if theta_deg has a single element,
        otherwise as a torch.Tensor with the same shape as theta_deg.

    Raises:
        ValueError: If any cos(theta) value is non-positive (theta >= 90
            degrees).
    """
    theta_t = as_tensor(
        theta_deg,
        device="cpu",
        dtype=torch.float64,
    )

    theta_rad = torch.deg2rad(theta_t)

    cos_theta = torch.cos(theta_rad)

    if torch.any(cos_theta <= 0.0):
        raise ValueError(
            "theta_deg must satisfy cos(theta) > 0."
        )

    if cos_theta.numel() == 1:
        return float(cos_theta.reshape(-1)[0].item())

    return cos_theta


# ============================================================
# Vertical depth
# ============================================================

@torch.no_grad()
def compute_vertical_depth_from_density(
    h_km: TensorLike,
    rho_gcm3: TensorLike,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Integrate a tabulated Atmosphere mass-density profile to obtain the
    vertical Atmosphere depth X(h) = integral_h^infinity rho(h') dh'.

    This is a tpeanuts-native trapezoidal-rule integration; it does not
    call MCEq. It evaluates the reversed cumulative integral of rho(h)
    from the top of the supplied altitude grid down to each grid point,
    which is the vertical (theta=0) column mass of air above altitude
    h. The result for a trajectory with theta != 0 still requires
    dividing by cos(theta) (see compute_slant_depth_from_vertical_depth).

    Args:
        h_km: Strictly increasing 1-D altitude grid in kilometres.
        rho_gcm3: Atmosphere mass density in g/cm^3 evaluated at each
            point of h_km; same shape as h_km.
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the computation and output.

    Returns:
        Tensor of vertical Atmosphere depth values in g/cm^2, same shape
        as h_km. The last grid point (top of the supplied range) is
        defined as zero depth remaining above it.
    """
    dev = default_device(device)

    h_t = as_tensor(h_km, device=dev, dtype=dtype)
    rho_t = as_tensor(rho_gcm3, device=dev, dtype=dtype)

    if h_t.ndim != 1:
        raise ValueError("h_km must be one-dimensional.")

    if rho_t.ndim != 1:
        raise ValueError("rho_gcm3 must be one-dimensional.")

    if h_t.shape != rho_t.shape:
        raise ValueError(
            "h_km and rho_gcm3 must have the same shape."
        )

    if torch.any(torch.diff(h_t) <= 0.0):
        raise ValueError("h_km must be strictly increasing.")

    h_cm = h_t * 1.0e5

    segment_X = 0.5 * (rho_t[:-1] + rho_t[1:]) * torch.diff(h_cm)
    X_t = torch.zeros_like(h_t)

    if segment_X.numel() > 0:
        X_t[:-1] = torch.flip(
            torch.cumsum(torch.flip(segment_X, dims=(0,)), dim=0),
            dims=(0,),
        )

    return X_t


@torch.no_grad()
def compute_vertical_depth_from_mceq(
    h_km: TensorLike,
    mceq=None,
    theta_deg: TensorLike = 0.0,
    config: Optional[MCEqModelConfig] = None,
    interaction_model: Optional[str] = None,
    primary_model: Optional[Union[str, tuple]] = None,
    density_model: Optional[str] = None,
    *,
    prefer_native_mceq: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Obtain the vertical Atmosphere depth X(h) at a set of altitudes
    using an MCEq Atmosphere density model.

    This wraps the external MCEq density model two ways: by default
    (prefer_native_mceq=True) it calls MCEq's own
    get_mass_overburden(h) directly via
    atmosphere_mass_overburden_profile_from_mceq, which is MCEq's native
    column-mass calculation. If that is unavailable for the selected
    density model, it falls back to evaluating MCEq's mass-density
    profile rho(h) and numerically integrating it with
    compute_vertical_depth_from_density. theta_deg only affects which
    MCEqRun is constructed when mceq is not supplied (zenith angle used
    to instantiate MCEqRun); the returned depth itself is the vertical
    (theta=0) overburden, not yet projected onto the slant path.

    Args:
        h_km: Scalar or tensor-like altitude(s) in kilometres at which
            to evaluate X(h).
        mceq: Optional initialized MCEq object. When omitted, a new
            instance is created via init_mceq using theta_deg and the
            supplied configuration arguments.
        theta_deg: Zenith angle in degrees passed to init_mceq when
            mceq is omitted.
        config: Optional MCEq model configuration object.
        interaction_model: Optional interaction-model override for
            init_mceq.
        primary_model: Optional primary cosmic-ray model override for
            init_mceq.
        density_model: Optional MCEq density-model override for
            init_mceq.
        prefer_native_mceq: If True, first try MCEq's native
            get_mass_overburden method before falling back to numerical
            integration of the density profile.
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the computation and output.

    Returns:
        Tensor of vertical Atmosphere depth values in g/cm^2, same shape
        as h_km.
    """
    dev = default_device(device)

    h_t = as_tensor(
        h_km,
        device=dev,
        dtype=dtype,
    )

    if mceq is None:
        mceq = init_mceq(
            theta_deg=theta_deg,
            config=config,
            interaction_model=interaction_model,
            primary_model=primary_model,
            density_model=density_model,
            info=False,
        )

    if prefer_native_mceq:

        try:
            return atmosphere_mass_overburden_profile_from_mceq(
                h_km=h_t,
                mceq=mceq,
                device=dev,
                dtype=dtype,
            )

        except Exception:
            pass

    rho_t = atmosphere_mass_density_profile_from_mceq(
        h_km=h_t,
        mceq=mceq,
        device=dev,
        dtype=dtype,
    )

    return compute_vertical_depth_from_density(
        h_km=h_t,
        rho_gcm3=rho_t,
        device=dev,
        dtype=dtype,
    )


# ============================================================
# Slant depth
# ============================================================

@torch.no_grad()
def compute_slant_depth_from_vertical_depth(
    X_vertical_gcm2: TensorLike,
    theta_deg: TensorLike,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Project a vertical Atmosphere depth profile onto an inclined
    trajectory: X(h, theta) = X_vertical(h) / cos(theta).

    A flat-Atmosphere (plane-parallel) approximation: the column of air
    crossed along a trajectory at zenith angle theta is 1/cos(theta)
    times the vertical column, since the path length through each
    infinitesimal density layer scales with 1/cos(theta).

    Args:
        X_vertical_gcm2: Vertical Atmosphere depth in g/cm^2 (as
            returned by compute_vertical_depth_from_density or
            compute_vertical_depth_from_mceq).
        theta_deg: Zenith angle in degrees of the trajectory; must
            satisfy cos(theta) > 0 (theta < 90 degrees).
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the computation and output.

    Returns:
        Tensor of slant Atmosphere depth values in g/cm^2, broadcast
        from X_vertical_gcm2 and cos(theta_deg).
    """
    dev = default_device(device)

    X_t = as_tensor(
        X_vertical_gcm2,
        device=dev,
        dtype=dtype,
    )

    cos_theta = theta_deg_to_cos(theta_deg)
    cos_t = as_tensor(
        cos_theta,
        device=dev,
        dtype=dtype,
    )

    if cos_t.ndim > 0 and X_t.ndim > 0:
        cos_t = cos_t.unsqueeze(-1)

    return X_t / cos_t


@torch.no_grad()
def compute_slant_depth_from_mceq(
    h_km: TensorLike,
    theta_deg: TensorLike,
    mceq=None,
    config: Optional[MCEqModelConfig] = None,
    interaction_model: Optional[str] = None,
    primary_model: Optional[Union[str, tuple]] = None,
    density_model: Optional[str] = None,
    *,
    prefer_native_mceq: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Compute the slant Atmosphere depth X(h, theta) at a set of
    altitudes for a given trajectory, using an MCEq density model.

    Combines compute_vertical_depth_from_mceq (which queries the MCEq
    Atmosphere density model, possibly via its native overburden
    method) with the cos(theta) projection of
    compute_slant_depth_from_vertical_depth. This is the slant depth
    coordinate actually used by MCEq's cascade-equation solve for an
    inclined trajectory.

    Args:
        h_km: Scalar or tensor-like altitude(s) in kilometres.
        theta_deg: Zenith angle in degrees of the trajectory; must
            satisfy cos(theta) > 0.
        mceq: Optional initialized MCEq object. When omitted, a new
            instance is created via init_mceq using theta_deg and the
            supplied configuration arguments.
        config: Optional MCEq model configuration object.
        interaction_model: Optional interaction-model override for
            init_mceq.
        primary_model: Optional primary cosmic-ray model override for
            init_mceq.
        density_model: Optional MCEq density-model override for
            init_mceq.
        prefer_native_mceq: If True, prefer MCEq's native
            get_mass_overburden method over numerical integration of
            its density profile when computing the vertical depth.
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the computation and output.

    Returns:
        Tensor of slant Atmosphere depth values in g/cm^2, same shape
        as h_km.
    """
    dev = default_device(device)

    X_vertical = compute_vertical_depth_from_mceq(
        h_km=h_km,
        mceq=mceq,
        theta_deg=theta_deg,
        config=config,
        interaction_model=interaction_model,
        primary_model=primary_model,
        density_model=density_model,
        prefer_native_mceq=prefer_native_mceq,
        device=dev,
        dtype=dtype,
    )

    return compute_slant_depth_from_vertical_depth(
        X_vertical_gcm2=X_vertical,
        theta_deg=theta_deg,
        device=dev,
        dtype=dtype,
    )


# ============================================================
# Derivatives
# ============================================================

@torch.no_grad()
def compute_dXdh(
    X_gcm2: TensorLike,
    h_km: TensorLike,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Compute the numerical derivative dX/dh of a depth profile X(h) with
    respect to altitude h.

    This is a tpeanuts-native finite-difference derivative (no MCEq
    call); it is used downstream to convert a depth-differential source
    term Q(E, X, theta) into a height-differential one via the
    Jacobian |dX/dh| (see
    tpeanuts.external.mceq.profiles.convert_depth_source_to_height_source).
    Since X(h) is monotonically decreasing with increasing altitude h
    (more atmosphere remains below lower altitudes), dX/dh is negative;
    callers that need a positive Jacobian factor take its absolute
    value.

    Args:
        X_gcm2: Atmosphere depth values in g/cm^2 evaluated on h_km.
        h_km: Altitude grid in kilometres at which X_gcm2 is sampled.
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the computation and output.

    Returns:
        Tensor of dX/dh values in g/cm^2/km, same shape as X_gcm2,
        computed with torch.gradient using h_km as the spacing.
    """
    dev = default_device(device)

    X_t = as_tensor(
        X_gcm2,
        device=dev,
        dtype=dtype,
    )

    h_t = as_tensor(
        h_km,
        device=dev,
        dtype=dtype,
    )

    return torch.gradient(
        X_t,
        spacing=(h_t,),
    )[0]


# ============================================================
# Mapping utilities
# ============================================================

@torch.no_grad()
def height_to_slant_depth(
    h_km: TensorLike,
    theta_deg: TensorLike,
    mceq=None,
    config: Optional[MCEqModelConfig] = None,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Convenience alias for compute_slant_depth_from_mceq with the
    default prefer_native_mceq behaviour.

    Maps altitude h (km) to slant Atmosphere depth X(h, theta) (g/cm^2)
    for a given trajectory zenith angle, using an MCEq Atmosphere
    density model. See compute_slant_depth_from_mceq for full details.

    Args:
        h_km: Scalar or tensor-like altitude(s) in kilometres.
        theta_deg: Zenith angle in degrees of the trajectory; must
            satisfy cos(theta) > 0.
        mceq: Optional initialized MCEq object; a new one is created via
            init_mceq when omitted.
        config: Optional MCEq model configuration object.
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the computation and output.

    Returns:
        Tensor of slant Atmosphere depth values in g/cm^2, same shape
        as h_km.
    """
    return compute_slant_depth_from_mceq(
        h_km=h_km,
        theta_deg=theta_deg,
        mceq=mceq,
        config=config,
        device=device,
        dtype=dtype,
    )

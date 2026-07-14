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
Geometry utilities for atmosphere-neutrino propagation.

This module defines the geometrical relations needed to propagate
atmosphere neutrinos produced at a height h above the earth's surface
and arriving at a detector with a given zenith angle theta.

The convention used for theta is the mceq convention:

    theta = 0 deg   -> vertically downward
    theta = 90 deg  -> horizontal

peanuts uses instead the nadir angle eta:

    eta = pi        -> vertically downward
    eta = pi/2      -> horizontal
    eta = 0         -> vertically upward

Therefore, the basic conversion is:

    eta = pi - theta

All distances are expressed in km unless otherwise stated.

Module functions:
    
    theta_to_eta(...)
        Converts atmosphere zenith angles in degrees to peanuts nadir
        angles in radians using eta = pi - theta.
    
    eta_to_theta(...)
        Converts peanuts nadir angles back to atmosphere zenith degrees.
    
    alpha_surface_to_theta_detector(...), theta_detector_to_alpha_surface(...)
        Convert between detector and surface zenith angles.
        
    alpha_max_for_detector_depth(...)
        Compute the limiting surface angle that maps to a detector ray
    
    atmosphere_path_length(...), underground_path_length(...),
        total_path_length(...)
        Compute spherical path lengths for atmosphere production, detector
        depth, and Earth-surface intersections.
    
    altitude_along_detector_path(...), atmosphere_path_grid(...)
        Build altitude and distance grids along the detector-to-production
        ray used by density and propagation modules.
    
    
"""



from __future__ import annotations

from typing import Union, Optional
import torch

from tpeanuts.util.constant import R_E_KM
from tpeanuts.util.type import TensorLike, as_tensor
from tpeanuts.util.torch_util import resolve_device


# ============================================================
# Angle conversions
# ============================================================

@torch.no_grad()
def theta_to_eta(
    theta_deg: TensorLike,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Convert an atmosphere zenith angle to the peanuts nadir angle.

    Args:
        theta_deg: Atmosphere zenith angle in degrees. Scalar or broadcastable
            tensor; theta=0 is vertically downward and theta=90 is horizontal.
        device: Optional torch device for the returned tensor.
        dtype: Real dtype used for the computation.

    Returns:
        Tensor with the broadcast shape of theta_deg containing eta in radians,
        using eta = pi - theta.
    """
    theta_deg = as_tensor(theta_deg, device=device, dtype=dtype)
    theta_rad = torch.deg2rad(theta_deg)
    return torch.pi - theta_rad


@torch.no_grad()
def eta_to_theta(
    eta_rad: TensorLike,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Convert a peanuts nadir angle to the atmosphere zenith convention.

    Args:
        eta_rad: Nadir angle in radians. Scalar or tensor.
        device: Optional torch device for the returned tensor.
        dtype: Real dtype used in the conversion.

    Returns:
        Tensor of atmosphere zenith angles in degrees, theta = pi - eta.
    """
    eta_rad = as_tensor(eta_rad, device=device, dtype=dtype)
    theta_rad = torch.pi - eta_rad
    return torch.rad2deg(theta_rad)


@torch.no_grad()
def alpha_surface_to_theta_detector(
    alpha_deg: TensorLike,
    h_d_km: TensorLike,
    R_km: TensorLike = R_E_KM,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Convert a surface angle to its corresponding detector angle.

    Args:
        alpha_deg: Trajectory angle at the Earth surface in degrees, measured
            relative to the local radial direction. Scalar or tensor.
        h_d_km: Detector depth below the Earth surface in km. Must be smaller
            than R_km and broadcastable with alpha_deg.
        R_km: Earth radius in km.
        device: Optional torch device used for inputs and output.
        dtype: Real torch dtype used for the calculation.

    Returns:
        Tensor of trajectory angles at the detector in degrees, with the
        broadcast shape of alpha_deg and h_d_km.

    Raises:
        ValueError: If the detector radius R_km - h_d_km is not positive.
    """
    device = resolve_device(device)
    alpha = as_tensor(alpha_deg, device=device, dtype=dtype)
    h_d_km = as_tensor(h_d_km, device=alpha.device, dtype=dtype)
    R_km = as_tensor(R_km, device=alpha.device, dtype=dtype)

    r_d = R_km - h_d_km
    if bool(torch.any(r_d <= 0.0).detach().cpu()):
        raise ValueError("Detector radius R_km - h_d_km must be positive.")

    alpha_rad = torch.deg2rad(alpha)
    # Impact parameter conservation: r_d sin(theta) = R sin(alpha).
    arg = torch.clamp((R_km / r_d) * torch.sin(alpha_rad), -1.0, 1.0)

    return torch.rad2deg(torch.arcsin(arg))


@torch.no_grad()
def theta_detector_to_alpha_surface(
    theta_deg: TensorLike,
    h_d_km: TensorLike,
    R_km: TensorLike = R_E_KM,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Convert a detector angle to its corresponding surface angle.

    Args:
        theta_deg: Trajectory angle at the detector in degrees, measured
            relative to the local radial direction. Scalar or tensor.
        h_d_km: Detector depth below the Earth surface in km. Must be smaller
            than R_km and broadcastable with theta_deg.
        R_km: Earth radius in km.
        device: Optional torch device used for inputs and output.
        dtype: Real torch dtype used for the calculation.

    Returns:
        Tensor of trajectory angles at the Earth surface in degrees, with the
        broadcast shape of theta_deg and h_d_km.

    Raises:
        ValueError: If the detector radius R_km - h_d_km is not positive.
    """
    device = resolve_device(device)
    theta = as_tensor(theta_deg, device=device, dtype=dtype)
    h_d_km = as_tensor(h_d_km, device=theta.device, dtype=dtype)
    R_km = as_tensor(R_km, device=theta.device, dtype=dtype)

    r_d = R_km - h_d_km
    if bool(torch.any(r_d <= 0.0).detach().cpu()):
        raise ValueError("Detector radius R_km - h_d_km must be positive.")

    theta_rad = torch.deg2rad(theta)
    # Impact parameter conservation: R sin(alpha) = r_d sin(theta).
    arg = torch.clamp((r_d / R_km) * torch.sin(theta_rad), -1.0, 1.0)

    return torch.rad2deg(torch.arcsin(arg))


@torch.no_grad()
def alpha_max_for_detector_depth(
    h_d_km: TensorLike,
    R_km: TensorLike = R_E_KM,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Compute the limiting surface angle that maps to a detector ray.

    Args:
        h_d_km: Detector depth below the Earth surface in km. Scalar or tensor
            with values smaller than R_km.
        R_km: Earth radius in km.
        device: Optional torch device used for inputs and output.
        dtype: Real torch dtype used for the calculation.

    Returns:
        Tensor containing ``asin((R_km - h_d_km) / R_km)`` in degrees. This
        is the largest surface angle whose inverse detector-angle argument
        remains within the arcsine domain.

    Raises:
        ValueError: If the detector radius R_km - h_d_km is not positive.
    """
    device = resolve_device(device)
    h_d_km = as_tensor(h_d_km, device=device, dtype=dtype)
    R_km = as_tensor(R_km, device=h_d_km.device, dtype=dtype)

    r_d = R_km - h_d_km
    if bool(torch.any(r_d <= 0.0).detach().cpu()):
        raise ValueError("Detector radius R_km - h_d_km must be positive.")

    # Above this surface angle, (R / r_d) sin(alpha) exceeds the arcsine domain.
    return torch.rad2deg(torch.arcsin(torch.clamp(r_d / R_km, -1.0, 1.0)))


# ============================================================
# Atmosphere trajectory
# ============================================================

@torch.no_grad()
def atmosphere_path_length(
    h_km: TensorLike,
    theta_deg: TensorLike,
    depth_km: TensorLike = 0.0,
    R_km: TensorLike = R_E_KM,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
    check_geometry: bool = True,
) -> torch.Tensor:
    """
    Compute the atmosphere part of a production-to-detector trajectory.

    Args:
        h_km: Production altitude above the Earth surface in km. Scalar or
            broadcastable tensor.
        theta_deg: Atmosphere zenith angle at the detector in degrees.
        depth_km: Detector depth below the surface in km. Scalar or tensor.
        R_km: Earth radius in km.
        device: Optional torch device for tensor conversion.
        dtype: Real dtype for geometry calculations.
        check_geometry: If True, raise ValueError for impossible intersections.

    Returns:
        Tensor with atmosphere path length in km, excluding the underground
        detector-to-surface segment.
    """
    h_km = as_tensor(h_km, device=device, dtype=dtype)
    theta_deg = as_tensor(theta_deg, device=h_km.device, dtype=dtype)
    depth_km = as_tensor(depth_km, device=h_km.device, dtype=dtype)

    L_total = total_path_length(
        h_km=h_km,
        theta_deg=theta_deg,
        depth_km=depth_km,
        R_km=R_km,
        device=h_km.device,
        dtype=dtype,
        check_geometry=check_geometry,
    )

    L_und = underground_path_length(
        theta_deg=theta_deg,
        depth_km=depth_km,
        R_km=R_km,
        device=h_km.device,
        dtype=dtype,
        check_geometry=check_geometry,
    )

    return L_total - L_und


# ============================================================
# Underground trajectory
# ============================================================

@torch.no_grad()
def underground_path_length(
    theta_deg: TensorLike,
    depth_km: TensorLike,
    R_km: TensorLike = R_E_KM,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
    check_geometry: bool = True,
) -> torch.Tensor:
    """
    Compute the spherical detector-depth path to the surface.

    Args:
        theta_deg: Atmospheric zenith angle at the detector in degrees.
        depth_km: Detector depth below the surface in km.
        R_km: Earth radius in km.
        device: Optional torch device.
        dtype: Real dtype for returned tensor.
        check_geometry: If True, reject trajectories with no surface crossing.

    Returns:
        Tensor of underground path lengths in km from detector to surface.
    """
    theta_deg = as_tensor(theta_deg, device=device, dtype=dtype)
    depth_km = as_tensor(depth_km, device=theta_deg.device, dtype=dtype)
    R_km = as_tensor(R_km, device=theta_deg.device, dtype=dtype)
    theta_rad = torch.deg2rad(theta_deg)

    r_d = R_km - depth_km
    if check_geometry and bool(torch.any(r_d <= 0.0).detach().cpu()):
        raise ValueError("Detector radius R_km - depth_km must be positive.")

    sin_theta = torch.sin(theta_rad)
    cos_theta = torch.cos(theta_rad)

    discriminant = R_km**2 - r_d**2 * sin_theta**2

    if check_geometry and bool(torch.any(discriminant < 0.0).detach().cpu()):
        raise ValueError(
            "Invalid underground geometry. Check theta_deg and depth_km."
        )

    discriminant = torch.clamp(discriminant, min=0.0)

    L_und_km = -r_d * cos_theta + torch.sqrt(discriminant)

    return L_und_km
    

# ============================================================
# Complete trajectory
# ============================================================

@torch.no_grad()
def total_path_length(
    h_km: TensorLike,
    theta_deg: TensorLike,
    depth_km: TensorLike = 0.0,
    R_km: TensorLike = R_E_KM,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
    check_geometry: bool = True,
) -> torch.Tensor:
    """
    Compute the full straight-line distance from detector to production shell.

    Args:
        h_km: Production altitude in km above the Earth surface.
        theta_deg: Atmospheric zenith angle in degrees.
        depth_km: Detector depth below surface in km.
        R_km: Earth radius in km.
        device: Optional torch device.
        dtype: Real dtype for geometry calculations.
        check_geometry: If True, reject non-intersecting trajectories.

    Returns:
        Tensor of total detector-to-production distance in km, including
        underground and atmosphere portions.
    """
    h_km = as_tensor(h_km, device=device, dtype=dtype)
    theta_deg = as_tensor(theta_deg, device=h_km.device, dtype=dtype)
    depth_km = as_tensor(depth_km, device=h_km.device, dtype=dtype)
    R_km = as_tensor(R_km, device=h_km.device, dtype=dtype)

    theta_rad = torch.deg2rad(theta_deg)

    r_d = R_km - depth_km
    r_h = R_km + h_km

    sin_theta = torch.sin(theta_rad)
    cos_theta = torch.cos(theta_rad)

    discriminant = r_h**2 - r_d**2 * sin_theta**2

    if check_geometry and bool(torch.any(discriminant < 0.0).detach().cpu()):
        raise ValueError(
            "The trajectory does not intersect the production shell. "
            "Check h_km, theta_deg and depth_km."
        )

    discriminant = torch.clamp(discriminant, min=0.0)

    L_total_km = -r_d * cos_theta + torch.sqrt(discriminant)

    return L_total_km


@torch.no_grad()
def altitude_along_detector_path(
    s_km: TensorLike,
    theta_deg: TensorLike,
    depth_km: TensorLike = 0.0,
    R_km: TensorLike = R_E_KM,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Evaluate altitude along a ray measured from the detector.

    Args:
        s_km: Distance from the detector along the back-traced trajectory in
            km. Scalar or tensor; often shaped (..., n_steps).
        theta_deg: Detector atmosphere zenith angle in degrees, broadcastable
            to s_km.
        depth_km: Detector depth below surface in km.
        R_km: Earth radius in km.
        device: Optional torch device.
        dtype: Real dtype for calculations.

    Returns:
        Tensor of altitudes in km with the broadcast shape of s_km and angles.
    """
    s_km = as_tensor(s_km, device=device, dtype=dtype)
    theta_deg = as_tensor(theta_deg, device=s_km.device, dtype=dtype)
    depth_km = as_tensor(depth_km, device=s_km.device, dtype=dtype)
    R_km = as_tensor(R_km, device=s_km.device, dtype=dtype)

    theta_rad = torch.deg2rad(theta_deg)

    r_d = R_km - depth_km

    r_s_sq = (
        r_d**2
        + s_km**2
        + 2.0 * r_d * s_km * torch.cos(theta_rad)
    )

    r_s = torch.sqrt(torch.clamp(r_s_sq, min=0.0))

    altitude_km = r_s - R_km

    return altitude_km


@torch.no_grad()
def atmosphere_path_grid(
    h_km: TensorLike,
    theta_deg: TensorLike,
    depth_km: TensorLike = 0.0,
    n_steps: int = 500,
    R_km: TensorLike = R_E_KM,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
    check_geometry: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a path grid through the atmosphere segment.

    Args:
        h_km: Production altitude in km. Scalar or tensor.
        theta_deg: Detector atmosphere zenith angle in degrees.
        depth_km: Detector depth in km.
        n_steps: Number of grid points along the atmosphere path.
        R_km: Earth radius in km.
        device: Optional torch device.
        dtype: Real dtype for returned tensors.
        check_geometry: If True, validate trajectory intersections.

    Returns:
        Pair (s_atm_grid_km, h_grid_km). Both tensors have a final dimension
        of length n_steps; s is atmosphere distance from surface crossing and
        h is altitude along the same points.
    """
    
    h_km_t = as_tensor(h_km, device=device, dtype=dtype)
    theta_deg_t = as_tensor(theta_deg, device=h_km_t.device, dtype=dtype)
    depth_km_t = as_tensor(depth_km, device=h_km_t.device, dtype=dtype)

    L_und_km = underground_path_length(
        theta_deg=theta_deg_t,
        depth_km=depth_km_t,
        R_km=R_km,
        device=h_km_t.device,
        dtype=dtype,
        check_geometry=check_geometry,
    )

    L_atm_km = atmosphere_path_length(
        h_km=h_km_t,
        theta_deg=theta_deg_t,
        depth_km=depth_km_t,
        R_km=R_km,
        device=h_km_t.device,
        dtype=dtype,
        check_geometry=check_geometry,
    )

    u = torch.linspace(
        0.0,
        1.0,
        int(n_steps),
        device=h_km_t.device,
        dtype=dtype,
    )

    s_atm_grid_km = L_atm_km[..., None] * u

    s_detector_grid_km = L_und_km[..., None] + s_atm_grid_km

    h_grid_km = altitude_along_detector_path(
        s_km=s_detector_grid_km,
        theta_deg=theta_deg_t[..., None],
        depth_km=depth_km_t[..., None],
        R_km=R_km,
        device=h_km_t.device,
        dtype=dtype,
    )

    return s_atm_grid_km, h_grid_km

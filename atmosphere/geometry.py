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
Geometry utilities for atmospheric neutrino propagation.

This module defines the geometrical relations needed to propagate
atmospheric neutrinos produced at a height h above the earth's surface
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
        Converts atmospheric zenith angles in degrees to peanuts nadir
        angles in radians using eta = pi - theta.
    
    theta_deg_to_rad(...), eta_to_theta(...)
        Provide explicit angle-conversion helpers for user-facing grids and
        internal torch calculations.
    
    atmospheric_path_length(...), underground_path_length(...),
        underground_path_length_planar(...), total_path_length(...)
        Compute spherical or planar path lengths for atmospheric production,
        detector depth, and Earth-surface intersections.
    
    altitude_along_detector_path(...), atmospheric_path_grid(...)
        Build altitude and distance grids along the detector-to-production
        ray used by density and propagation modules.
    
    detector_alpha_to_surface_theta(...), surface_intersection_angle(...),
    validate_downward_trajectory(...),
    
    theta_h_to_eta_and_baseline(...)
        Provide geometry validation and peanuts-compatible coordinates for
        atmospheric-neutrino workflows.
"""



from __future__ import annotations

from typing import Union, Optional
import torch


TensorLike = Union[float, int, torch.Tensor]

from tpeanuts.util.constant import R_E_KM
from tpeanuts.util.type import _as_tensor


def _resolve_geometry_device(device):
    if callable(device):
        return device()
    return device


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
    Convert an atmospheric zenith angle to the peanuts nadir angle.

    Args:
        theta_deg: Atmospheric zenith angle in degrees. Scalar or broadcastable
            tensor; theta=0 is vertically downward and theta=90 is horizontal.
        device: Optional torch device for the returned tensor.
        dtype: Real dtype used for the computation.

    Returns:
        Tensor with the broadcast shape of theta_deg containing eta in radians,
        using eta = pi - theta.
    """
    theta_rad = theta_deg_to_rad(theta_deg, device=device, dtype=dtype)
    return torch.pi - theta_rad


@torch.no_grad()
def theta_deg_to_rad(
    theta_deg: TensorLike,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Convert degrees to radians on the requested torch device.

    Args:
        theta_deg: Scalar, array-like, or tensor angle in degrees.
        device: Optional torch device for conversion.
        dtype: Real dtype for the returned tensor.

    Returns:
        Tensor with the same broadcast shape as theta_deg and values in radians.
    """
    theta_deg = _as_tensor(theta_deg, device=device, dtype=dtype)
    return torch.deg2rad(theta_deg)


@torch.no_grad()
def eta_to_theta(
    eta_rad: TensorLike,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Convert a peanuts nadir angle to the atmospheric zenith convention.

    Args:
        eta_rad: Nadir angle in radians. Scalar or tensor.
        device: Optional torch device for the returned tensor.
        dtype: Real dtype used in the conversion.

    Returns:
        Tensor of atmospheric zenith angles in degrees, theta = pi - eta.
    """
    eta_rad = _as_tensor(eta_rad, device=device, dtype=dtype)
    theta_rad = torch.pi - eta_rad
    return torch.rad2deg(theta_rad)


# ============================================================
# Atmospheric trajectory
# ============================================================

@torch.no_grad()
def atmospheric_path_length(
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
    Compute the atmospheric part of a production-to-detector trajectory.

    Args:
        h_km: Production altitude above the Earth surface in km. Scalar or
            broadcastable tensor.
        theta_deg: Atmospheric zenith angle at the detector in degrees.
        depth_km: Detector depth below the surface in km. Scalar or tensor.
        R_km: Earth radius in km.
        device: Optional torch device for tensor conversion.
        dtype: Real dtype for geometry calculations.
        check_geometry: If True, raise ValueError for impossible intersections.

    Returns:
        Tensor with atmospheric path length in km, excluding the underground
        detector-to-surface segment.
    """
    h_km = _as_tensor(h_km, device=device, dtype=dtype)
    theta_deg = _as_tensor(theta_deg, device=h_km.device, dtype=dtype)
    depth_km = _as_tensor(depth_km, device=h_km.device, dtype=dtype)

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
        theta_deg: Detector atmospheric zenith angle in degrees, broadcastable
            to s_km.
        depth_km: Detector depth below surface in km.
        R_km: Earth radius in km.
        device: Optional torch device.
        dtype: Real dtype for calculations.

    Returns:
        Tensor of altitudes in km with the broadcast shape of s_km and angles.
    """
    s_km = _as_tensor(s_km, device=device, dtype=dtype)
    theta_deg = _as_tensor(theta_deg, device=s_km.device, dtype=dtype)
    depth_km = _as_tensor(depth_km, device=s_km.device, dtype=dtype)
    R_km = _as_tensor(R_km, device=s_km.device, dtype=dtype)

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
def atmospheric_path_grid(
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
    Build a path grid through the atmospheric segment.

    Args:
        h_km: Production altitude in km. Scalar or tensor.
        theta_deg: Detector atmospheric zenith angle in degrees.
        depth_km: Detector depth in km.
        n_steps: Number of grid points along the atmospheric path.
        R_km: Earth radius in km.
        device: Optional torch device.
        dtype: Real dtype for returned tensors.
        check_geometry: If True, validate trajectory intersections.

    Returns:
        Pair (s_atm_grid_km, h_grid_km). Both tensors have a final dimension
        of length n_steps; s is atmospheric distance from surface crossing and
        h is altitude along the same points.
    """
    
    h_km_t = _as_tensor(h_km, device=device, dtype=dtype)
    theta_deg_t = _as_tensor(theta_deg, device=h_km_t.device, dtype=dtype)
    depth_km_t = _as_tensor(depth_km, device=h_km_t.device, dtype=dtype)

    L_und_km = underground_path_length(
        theta_deg=theta_deg_t,
        depth_km=depth_km_t,
        R_km=R_km,
        device=h_km_t.device,
        dtype=dtype,
        check_geometry=check_geometry,
    )

    L_atm_km = atmospheric_path_length(
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
    theta_deg = _as_tensor(theta_deg, device=device, dtype=dtype)
    depth_km = _as_tensor(depth_km, device=theta_deg.device, dtype=dtype)
    R_km = _as_tensor(R_km, device=theta_deg.device, dtype=dtype)
    theta_rad = torch.deg2rad(theta_deg)

    r_d = R_km - depth_km

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
    

@torch.no_grad()
def underground_path_length_planar(
    theta_deg: TensorLike,
    depth_km: TensorLike,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
    check_geometry: bool = True,
) -> torch.Tensor:
    """
    Compute the planar approximation to the underground path length.

    Args:
        theta_deg: Downward detector zenith angle in degrees. Requires
            theta_deg < 90 when check_geometry is True.
        depth_km: Detector depth in km.
        device: Optional torch device.
        dtype: Real dtype for returned tensor.
        check_geometry: If True, reject horizontal/upward planar trajectories.

    Returns:
        Tensor depth_km / cos(theta) in km.
    """
    theta_deg = _as_tensor(theta_deg, device=device, dtype=dtype)
    depth_km = _as_tensor(depth_km, device=theta_deg.device, dtype=dtype)

    theta_rad = torch.deg2rad(theta_deg)
    cos_theta = torch.cos(theta_rad)

    if check_geometry and bool(torch.any(cos_theta <= 0.0).detach().cpu()):
        raise ValueError(
            "Planar underground path length requires theta_deg < 90 deg."
        )

    return depth_km / cos_theta


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
        underground and atmospheric portions.
    """
    h_km = _as_tensor(h_km, device=device, dtype=dtype)
    theta_deg = _as_tensor(theta_deg, device=h_km.device, dtype=dtype)
    depth_km = _as_tensor(depth_km, device=h_km.device, dtype=dtype)
    R_km = _as_tensor(R_km, device=h_km.device, dtype=dtype)

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
def alpha_surface_to_theta_detector(
    alpha_deg: TensorLike,
    h_d_km: TensorLike,
    R_km: TensorLike = R_E_KM,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    device = _resolve_geometry_device(device)
    alpha = _as_tensor(alpha_deg, device=device, dtype=dtype)
    h_d_km = _as_tensor(h_d_km, device=alpha.device, dtype=dtype)
    R_km = _as_tensor(R_km, device=alpha.device, dtype=dtype)

    r_d = R_km - h_d_km
    if bool(torch.any(r_d <= 0.0).detach().cpu()):
        raise ValueError("Detector radius R_km - h_d_km must be positive.")

    alpha_rad = torch.deg2rad(alpha)
    arg = torch.clamp((r_d / R_km) * torch.sin(alpha_rad), -1.0, 1.0)

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
    device = _resolve_geometry_device(device)
    theta = _as_tensor(theta_deg, device=device, dtype=dtype)
    h_d_km = _as_tensor(h_d_km, device=theta.device, dtype=dtype)
    R_km = _as_tensor(R_km, device=theta.device, dtype=dtype)

    r_d = R_km - h_d_km
    if bool(torch.any(r_d <= 0.0).detach().cpu()):
        raise ValueError("Detector radius R_km - h_d_km must be positive.")

    theta_rad = torch.deg2rad(theta)
    arg = torch.clamp((R_km / r_d) * torch.sin(theta_rad), -1.0, 1.0)

    return torch.rad2deg(torch.arcsin(arg))


@torch.no_grad()
def alpha_max_for_detector_depth(
    h_d_km: TensorLike,
    R_km: TensorLike = R_E_KM,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    device = _resolve_geometry_device(device)
    h_d_km = _as_tensor(h_d_km, device=device, dtype=dtype)
    R_km = _as_tensor(R_km, device=h_d_km.device, dtype=dtype)

    r_d = R_km - h_d_km
    if bool(torch.any(r_d <= 0.0).detach().cpu()):
        raise ValueError("Detector radius R_km - h_d_km must be positive.")

    return torch.rad2deg(torch.arcsin(torch.clamp(r_d / R_km, -1.0, 1.0)))


@torch.no_grad()
def surface_intersection_angle(
    zeta_detector_deg: TensorLike,
    depth_km: TensorLike = 0.0,
    R_km: TensorLike = R_E_KM,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
    return_distances: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a detector ray angle to the corresponding surface angle.

    Args:
        zeta_detector_deg: Detector angle in degrees relative to the local
            vertical convention used by this helper.
        depth_km: Detector depth in km.
        R_km: Earth radius in km.
        device: Optional torch device.
        dtype: Real dtype for returned tensors.
        return_distances: If True, also return detector-to-surface distance.

    Returns:
        Surface angle in degrees. If return_distances=True, returns
        (zeta_surface_deg, s_surface_km).
    """
    zeta_detector_deg = _as_tensor(
        zeta_detector_deg,
        device=device,
        dtype=dtype,
    )

    depth_km = _as_tensor(
        depth_km,
        device=zeta_detector_deg.device,
        dtype=dtype,
    )

    R_km = _as_tensor(
        R_km,
        device=zeta_detector_deg.device,
        dtype=dtype,
    )

    zeta_d = torch.deg2rad(zeta_detector_deg)

    r_d = R_km - depth_km

    sin_zeta_s = (r_d / R_km) * torch.sin(zeta_d)
    sin_zeta_s = torch.clamp(sin_zeta_s, -1.0, 1.0)

    zeta_surface_rad = torch.arcsin(torch.abs(sin_zeta_s))
    zeta_surface_deg = torch.rad2deg(zeta_surface_rad)

    if not return_distances:
        return zeta_surface_deg

    # Distance from detector to surface crossing along the back-traced ray
    discriminant = R_km**2 - r_d**2 * torch.sin(zeta_d)**2
    discriminant = torch.clamp(discriminant, min=0.0)

    s_surface_km = (
        -r_d * torch.cos(zeta_d)
        + torch.sqrt(discriminant)
    )

    return zeta_surface_deg, s_surface_km


@torch.no_grad()
def detector_alpha_to_surface_theta(
    alpha_deg: TensorLike,
    *,
    detector_depth_m: float = 0.0,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
    return_distance: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    device = _resolve_geometry_device(device)
    depth_km = _as_tensor(
        detector_depth_m / 1.0e3,
        device=device,
        dtype=dtype,
    )

    return surface_intersection_angle(
        zeta_detector_deg=alpha_deg,
        depth_km=depth_km,
        device=depth_km.device,
        dtype=dtype,
        return_distances=return_distance,
    )

@torch.no_grad()
def validate_downward_trajectory(
    alpha_deg: TensorLike,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> None:
    """
    Validate the downward-going atmospheric-angle range.

    Args:
        alpha_deg: Detector angle in degrees. Scalar or tensor.
        device: Optional torch device.
        dtype: Real dtype for validation.

    Returns:
        None. Raises ValueError if any value is outside 0 <= alpha_deg < 90.
    """
    alpha_deg = _as_tensor(alpha_deg, device=device, dtype=dtype)

    invalid = torch.any(alpha_deg < 0.0) or torch.any(alpha_deg >= 90.0)

    if bool(invalid.detach().cpu()):
        raise ValueError(
            "This atmospheric geometry module currently assumes "
            "0 <= alpha_deg < 90 degrees."
        )
        
# ============================================================
# helper functions for peanuts Torch propagation
# ============================================================

@torch.no_grad()
def theta_h_to_eta_and_baseline(
    h_km: TensorLike,
    theta_deg: TensorLike,
    depth_km: TensorLike = 0.0,
    R_km: TensorLike = R_E_KM,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
    check_geometry: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert atmospheric production coordinates to peanuts coordinates.

    Args:
        h_km: Production altitude in km.
        theta_deg: Atmospheric zenith angle in degrees.
        depth_km: Detector depth in km.
        R_km: Earth radius in km.
        device: Optional torch device.
        dtype: Real dtype for returned tensors.
        check_geometry: If True, validate trajectory intersections.

    Returns:
        Pair (eta, L_total), where eta is the peanuts nadir angle in radians
        and L_total is the detector-to-production baseline in km.
    """
    h_km = _as_tensor(h_km, device=device, dtype=dtype)
    theta_deg = _as_tensor(theta_deg, device=h_km.device, dtype=dtype)
    depth_km = _as_tensor(depth_km, device=h_km.device, dtype=dtype)

    eta = theta_to_eta(
        theta_deg,
        device=h_km.device,
        dtype=dtype,
    )

    L_total = total_path_length(
        h_km=h_km,
        theta_deg=theta_deg,
        depth_km=depth_km,
        R_km=R_km,
        device=h_km.device,
        dtype=dtype,
        check_geometry=check_geometry,
    )

    return eta, L_total
        
        
        
        

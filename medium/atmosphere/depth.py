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
Generic atmosphere-depth utilities.

This module contains tensor-only transformations between altitude,
vertical atmospheric depth, and slant atmospheric depth. It does not call
MCEq or any other external atmosphere backend.

Definitions:
    h:
        Altitude above sea level, in km.
    rho(h):
        Atmospheric mass density, in g/cm^3.
    X_vertical(h):
        Vertical atmospheric depth, in g/cm^2:

            X_vertical(h) = integral_h^infinity rho(h') dh'

        with the altitude integration converted from km to cm.
    X_slant(h, alpha):
        Slant atmospheric depth along a plane-parallel inclined path:

            X_slant(h, alpha) = X_vertical(h) / cos(alpha)

    alpha:
        Surface zenith angle in degrees. alpha=0 is vertical downward,
        and alpha must satisfy cos(alpha) > 0 for the plane-parallel
        slant-depth projection.

Module functions:
    alpha_deg_to_cos:
        Convert surface zenith angle alpha in degrees to cos(alpha),
        validating that the trajectory is not horizontal/upward-going.
    atmosphere_vertical_depth:
        Integrate a tabulated mass-density profile rho(h) into the
        vertical atmospheric depth X_vertical(h).
    atmosphere_slant_depth:
        Project a vertical atmospheric-depth profile into slant depth
        using X_slant = X_vertical / cos(alpha).
    compute_dXdh:
        Compute the numerical derivative dX/dh on an altitude grid.
    interpolate_flux_at_Xobs:
        Interpolate a depth-tabulated flux Phi(E, X, alpha) to one or
        more observation depths X_obs.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from tpeanuts.util.type import as_tensor
from tpeanuts.util.torch_util import default_device


TensorLike = Union[float, int, torch.Tensor]


def alpha_deg_to_cos(alpha_deg: TensorLike) -> float:
    """
    Convert a surface zenith angle in degrees to cos(alpha).

    The returned cosine is intended for slant-depth projections, so all
    values must be strictly positive. Horizontal and upward-going
    trajectories are rejected because X_vertical / cos(alpha) would be
    undefined or negative in the plane-parallel approximation.

    Args:
        alpha_deg: Scalar or tensor-like surface zenith angle(s), in degrees.

    Returns:
        cos(alpha) as a Python float for scalar input, otherwise a tensor with
        the same shape as alpha_deg.

    Raises:
        ValueError: If any cos(alpha) value is non-positive.
    """
    alpha_t = as_tensor(
        alpha_deg,
        device="cpu",
        dtype=torch.float64,
    )

    cos_alpha = torch.cos(torch.deg2rad(alpha_t))

    if torch.any(cos_alpha <= 10.0 * torch.finfo(cos_alpha.dtype).eps):
        raise ValueError("alpha_deg must satisfy cos(alpha) > 0.")

    if cos_alpha.numel() == 1:
        return float(cos_alpha.reshape(-1)[0].item())

    return cos_alpha


@torch.no_grad()
def atmosphere_vertical_depth(
    h_km: TensorLike,
    rho_gcm3: TensorLike,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Integrate a tabulated mass-density profile to obtain vertical depth.

    Computes X(h) = integral_h^infinity rho(h') dh' with h converted from
    kilometres to centimetres, returning X in g/cm^2.

    The input altitude grid must be strictly increasing. The returned
    tensor has the same shape as h_km, and the last altitude point is
    assigned zero remaining overburden above the supplied grid.

    Args:
        h_km: One-dimensional altitude grid in km, strictly increasing.
        rho_gcm3: Atmospheric mass density in g/cm^3 evaluated at each
            altitude in h_km.
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the computation and output.

    Returns:
        Vertical atmospheric depth X_vertical(h), in g/cm^2.

    Raises:
        ValueError: If h_km or rho_gcm3 is not one-dimensional, if their
            shapes differ, or if h_km is not strictly increasing.
    """
    dev = default_device(device)

    h_t = as_tensor(h_km, device=dev, dtype=dtype)
    rho_t = as_tensor(rho_gcm3, device=dev, dtype=dtype)

    if h_t.ndim != 1:
        raise ValueError("h_km must be one-dimensional.")

    if rho_t.ndim != 1:
        raise ValueError("rho_gcm3 must be one-dimensional.")

    if h_t.shape != rho_t.shape:
        raise ValueError("h_km and rho_gcm3 must have the same shape.")

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
def atmosphere_slant_depth(
    X_vertical_gcm2: TensorLike,
    alpha_deg: TensorLike,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Project vertical atmospheric depth onto an inclined trajectory.

    Uses the plane-parallel approximation:

        X_slant(h, alpha) = X_vertical(h) / cos(alpha)

    Args:
        X_vertical_gcm2: Vertical atmospheric depth in g/cm^2.
        alpha_deg: Surface zenith angle in degrees. All values must
            satisfy cos(alpha) > 0.
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the computation and output.

    Returns:
        Slant atmospheric depth in g/cm^2, broadcast from
        X_vertical_gcm2 and alpha_deg.

    Raises:
        ValueError: If any alpha_deg value has cos(alpha) <= 0.
    """
    dev = default_device(device)

    X_t = as_tensor(
        X_vertical_gcm2,
        device=dev,
        dtype=dtype,
    )

    cos_alpha = alpha_deg_to_cos(alpha_deg)
    cos_t = as_tensor(
        cos_alpha,
        device=dev,
        dtype=dtype,
    )

    if cos_t.ndim > 0 and X_t.ndim > 0:
        cos_t = cos_t.unsqueeze(-1)

    return X_t / cos_t


@torch.no_grad()
def compute_dXdh(
    X_gcm2: TensorLike,
    h_km: TensorLike,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Compute the numerical derivative dX/dh of a depth profile.

    Args:
        X_gcm2: Atmospheric depth values in g/cm^2 sampled on h_km.
        h_km: Altitude grid in km.
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the computation and output.

    Returns:
        Numerical derivative dX/dh in g/cm^2/km, with the same shape as
        X_gcm2.
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


@torch.no_grad()
def interpolate_flux_at_Xobs(
    X_grid_gcm2: TensorLike,
    flux_XE: TensorLike,
    X_obs_gcm2: TensorLike,
    *,
    log_interp: bool = True,
    eps: float = 1.0e-300,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Interpolate a depth-tabulated flux to an observation depth.

    The interpolation is performed along the second-to-last dimension of
    flux_XE, which is interpreted as the atmospheric-depth axis. The
    final dimension is interpreted as energy. This is backend-neutral:
    flux_XE may come from MCEq, Honda, cached files, or any other source
    tabulated as Phi(E, X, alpha).

    Args:
        X_grid_gcm2: Strictly increasing one-dimensional atmospheric
            depth grid in g/cm^2.
        flux_XE: Flux tensor with shape (..., n_X, n_E), where n_X
            matches X_grid_gcm2.
        X_obs_gcm2: Observation depth(s) in g/cm^2. Values must lie
            within the X_grid_gcm2 range and are broadcast against the
            leading dimensions of flux_XE.
        log_interp: If True, interpolate log(flux) linearly in X and
            exponentiate the result. This is useful for positive fluxes
            spanning many orders of magnitude.
        eps: Positive floor applied before log interpolation.
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the computation and output.

    Returns:
        Interpolated flux at X_obs_gcm2 with shape (..., n_E), in the
        same units as flux_XE.

    Raises:
        ValueError: If flux_XE has an incompatible shape, if X_grid_gcm2
            is not strictly increasing, or if X_obs_gcm2 lies outside
            the tabulated depth range.
    """
    dev = default_device(device)

    X_t = as_tensor(
        X_grid_gcm2,
        device=dev,
        dtype=dtype,
    ).reshape(-1)

    flux_t = as_tensor(
        flux_XE,
        device=dev,
        dtype=dtype,
    )

    X_obs_t = as_tensor(
        X_obs_gcm2,
        device=dev,
        dtype=dtype,
    )

    if flux_t.ndim < 2:
        raise ValueError("flux_XE must have shape (..., n_X, n_E).")

    if flux_t.shape[-2] != X_t.numel():
        raise ValueError(
            "flux_XE.shape[-2] must match len(X_grid_gcm2)."
        )

    if torch.any(torch.diff(X_t) <= 0.0):
        raise ValueError("X_grid_gcm2 must be strictly increasing.")

    if torch.any((X_obs_t < X_t[0]) | (X_obs_t > X_t[-1])):
        raise ValueError(
            f"X_obs_gcm2 is outside X_grid range "
            f"[{float(X_t.min().item())}, {float(X_t.max().item())}]."
        )

    batch_shape = torch.broadcast_shapes(
        flux_t.shape[:-2],
        X_obs_t.shape,
    )

    n_X = X_t.numel()
    n_E = flux_t.shape[-1]

    flux_b = torch.broadcast_to(
        flux_t,
        (*batch_shape, n_X, n_E),
    )
    X_obs_b = torch.broadcast_to(X_obs_t, batch_shape)

    idx = torch.searchsorted(
        X_t,
        X_obs_b,
        right=False,
    )

    idx = torch.clamp(idx, min=1, max=n_X - 1)

    x0 = X_t[idx - 1]
    x1 = X_t[idx]

    w = (X_obs_b - x0) / (x1 - x0)

    gather_shape = (*batch_shape, 1, n_E)
    idx0 = (idx - 1).unsqueeze(-1).unsqueeze(-1).expand(gather_shape)
    idx1 = idx.unsqueeze(-1).unsqueeze(-1).expand(gather_shape)

    y0 = torch.gather(flux_b, dim=-2, index=idx0).squeeze(-2)
    y1 = torch.gather(flux_b, dim=-2, index=idx1).squeeze(-2)
    w = w.unsqueeze(-1)

    if log_interp:
        log_y0 = torch.log(torch.clamp(y0, min=eps))
        log_y1 = torch.log(torch.clamp(y1, min=eps))

        return torch.exp((1.0 - w) * log_y0 + w * log_y1)

    return (1.0 - w) * y0 + w * y1

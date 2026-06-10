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
Atmospheric-depth utilities.

This module contains all transformations related to:

    h <-> X(h, theta)

and numerical derivatives such as:

    dX/dh

The atmospheric depth is defined as

    X(h) = \int_h^\infty rho(h') dh'

with units:

    X : g/cm^2
    rho : g/cm^3
"""



from __future__ import annotations

from typing import Optional, Union

import torch

from tpeanuts.util.type import _as_tensor
from tpeanuts.util.torch_util import _default_device

from tpeanuts.external.mceq.config import (
    MCEqModelConfig,
)

from tpeanuts.external.mceq.core import init_mceq

from tpeanuts.external.mceq.density import (
    atmospheric_mass_density_profile_from_mceq,
    atmospheric_mass_overburden_profile_from_mceq,
)


TensorLike = Union[float, int, torch.Tensor]


# ============================================================
# Utilities
# ============================================================

def theta_deg_to_cos(theta_deg: TensorLike) -> float:
    theta_t = _as_tensor(
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
    dev = _default_device(device)

    h_t = _as_tensor(h_km, device=dev, dtype=dtype)
    rho_t = _as_tensor(rho_gcm3, device=dev, dtype=dtype)

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
    dev = _default_device(device)

    h_t = _as_tensor(
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
            return atmospheric_mass_overburden_profile_from_mceq(
                h_km=h_t,
                mceq=mceq,
                device=dev,
                dtype=dtype,
            )

        except Exception:
            pass

    rho_t = atmospheric_mass_density_profile_from_mceq(
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
    dev = _default_device(device)

    X_t = _as_tensor(
        X_vertical_gcm2,
        device=dev,
        dtype=dtype,
    )

    cos_theta = theta_deg_to_cos(theta_deg)
    cos_t = _as_tensor(
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
    dev = _default_device(device)

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
    dev = _default_device(device)

    X_t = _as_tensor(
        X_gcm2,
        device=dev,
        dtype=dtype,
    )

    h_t = _as_tensor(
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
    return compute_slant_depth_from_mceq(
        h_km=h_km,
        theta_deg=theta_deg,
        mceq=mceq,
        config=config,
        device=device,
        dtype=dtype,
    )

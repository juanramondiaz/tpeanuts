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
Smoothing and derivative utilities for mceq fluxes.

This module operates on tensors with shape:

    flux_XE : (n_X, n_E)

where:

    X -> atmospheric depth
    E -> energy grid
"""



from __future__ import annotations

from typing import Optional, Union

import torch

from tpeanuts.util.type import _as_tensor
from tpeanuts.util.torch_util import _default_device

from tpeanuts.external.mceq.config import SmoothingConfig


TensorLike = Union[float, int, torch.Tensor]


# ============================================================
# Validation
# ============================================================

def validate_flux_depth_inputs(
    X_grid_gcm2: TensorLike,
    flux_XE: TensorLike,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
):
    dev = _default_device(device)

    X_t = _as_tensor(
        X_grid_gcm2,
        device=dev,
        dtype=dtype,
    ).reshape(-1)

    flux_t = _as_tensor(
        flux_XE,
        device=dev,
        dtype=dtype,
    )

    if X_t.ndim != 1:
        raise ValueError("X_grid_gcm2 must be one-dimensional.")

    if flux_t.ndim < 2:
        raise ValueError("flux_XE must have shape (..., n_X, n_E).")

    if flux_t.shape[-2] != X_t.numel():
        raise ValueError(
            "flux_XE.shape[-2] must match len(X_grid_gcm2)."
        )

    if torch.any(torch.diff(X_t) <= 0.0):
        raise ValueError("X_grid_gcm2 must be strictly increasing.")

    return X_t, flux_t


# ============================================================
# Gaussian smoothing
# ============================================================

@torch.no_grad()
def gaussian_kernel1d(
    sigma: float,
    *,
    truncate: float = 4.0,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    dev = _default_device(device)

    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")

    radius = int(truncate * sigma + 0.5)

    x = torch.arange(
        -radius,
        radius + 1,
        device=dev,
        dtype=dtype,
    )

    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()

    return kernel


@torch.no_grad()
def smooth_flux_gaussian(
    flux_XE: TensorLike,
    sigma: float = 2.0,
    *,
    truncate: float = 4.0,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    dev = _default_device(device)

    flux_t = _as_tensor(
        flux_XE,
        device=dev,
        dtype=dtype,
    )

    if flux_t.ndim < 2:
        raise ValueError("flux_XE must have shape (..., n_X, n_E).")

    if sigma <= 0.0:
        return flux_t.clone()

    kernel = gaussian_kernel1d(
        sigma=sigma,
        truncate=truncate,
        device=dev,
        dtype=dtype,
    )

    pad = kernel.numel() // 2

    original_shape = flux_t.shape
    n_X = flux_t.shape[-2]
    n_E = flux_t.shape[-1]

    # Conv1d expects (batch, channels, length).
    x = flux_t.reshape(-1, n_X, n_E).permute(0, 2, 1)

    x_pad = torch.nn.functional.pad(
        x,
        pad=(pad, pad),
        mode="replicate",
    )

    weight = kernel.reshape(1, 1, -1).repeat(n_E, 1, 1)

    y = torch.nn.functional.conv1d(
        x_pad,
        weight,
        groups=n_E,
    )

    return y.permute(0, 2, 1).reshape(original_shape)


# ============================================================
# Spline-like smoothing
# ============================================================

@torch.no_grad()
def smooth_flux_log_moving_average(
    flux_XE: TensorLike,
    window: int = 7,
    eps: float = 1.0e-300,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    dev = _default_device(device)

    flux_t = _as_tensor(
        flux_XE,
        device=dev,
        dtype=dtype,
    )

    if flux_t.ndim < 2:
        raise ValueError("flux_XE must have shape (..., n_X, n_E).")

    if window <= 1:
        return flux_t.clone()

    if window % 2 == 0:
        window += 1

    log_flux = torch.log(torch.clamp(flux_t, min=eps))

    kernel = torch.ones(
        window,
        device=dev,
        dtype=dtype,
    ) / float(window)

    pad = window // 2

    original_shape = flux_t.shape
    n_X = flux_t.shape[-2]
    n_E = flux_t.shape[-1]

    x = log_flux.reshape(-1, n_X, n_E).permute(0, 2, 1)

    x_pad = torch.nn.functional.pad(
        x,
        pad=(pad, pad),
        mode="replicate",
    )

    weight = kernel.reshape(1, 1, -1).repeat(n_E, 1, 1)

    y = torch.nn.functional.conv1d(
        x_pad,
        weight,
        groups=n_E,
    )

    return torch.exp(y.permute(0, 2, 1).reshape(original_shape))


# ============================================================
# Main smoothing wrapper
# ============================================================

@torch.no_grad()
def smooth_flux_in_depth(
    X_grid_gcm2: TensorLike,
    flux_XE: TensorLike,
    config: Optional[SmoothingConfig] = None,
    method: Optional[str] = None,
    smoothing: Optional[float] = None,
    gaussian_sigma: Optional[float] = None,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    X_t, flux_t = validate_flux_depth_inputs(
        X_grid_gcm2=X_grid_gcm2,
        flux_XE=flux_XE,
        device=device,
        dtype=dtype,
    )

    if config is not None:
        method = config.method if method is None else method
        smoothing = config.smoothing if smoothing is None else smoothing
        gaussian_sigma = (
            config.gaussian_sigma
            if gaussian_sigma is None
            else gaussian_sigma
        )

    method = "none" if method is None else method

    if method == "none":
        return flux_t.clone()

    if method == "gaussian":
        sigma = 2.0 if gaussian_sigma is None else gaussian_sigma

        return smooth_flux_gaussian(
            flux_XE=flux_t,
            sigma=sigma,
            device=device,
            dtype=dtype,
        )

    if method in {"log_moving_average", "spline"}:
        # Practical mapping:
        # small smoothing -> small window
        # larger smoothing -> larger window
        s = 1.0e-4 if smoothing is None else float(smoothing)

        n_X = X_t.numel()

        window = max(
            3,
            int(round(s * n_X * 100.0)),
        )

        if window % 2 == 0:
            window += 1

        return smooth_flux_log_moving_average(
            flux_XE=flux_t,
            window=window,
            device=device,
            dtype=dtype,
        )

    raise ValueError(
        "method must be one of: None, 'none', 'gaussian', "
        "'log_moving_average', or 'spline'."
    )


# ============================================================
# Derivatives
# ============================================================

@torch.no_grad()
def compute_depth_derivative(
    X_grid_gcm2: TensorLike,
    flux_XE: TensorLike,
    *,
    positive_only: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    X_t, flux_t = validate_flux_depth_inputs(
        X_grid_gcm2=X_grid_gcm2,
        flux_XE=flux_XE,
        device=device,
        dtype=dtype,
    )

    dPhi_dX = torch.gradient(
        flux_t,
        spacing=(X_t,),
        dim=-2,
    )[0]

    if positive_only:
        dPhi_dX = torch.clamp(dPhi_dX, min=0.0)

    return dPhi_dX


@torch.no_grad()
def smooth_and_differentiate_flux(
    X_grid_gcm2: TensorLike,
    flux_XE: TensorLike,
    config: Optional[SmoothingConfig] = None,
    *,
    method: Optional[str] = None,
    smoothing: Optional[float] = None,
    gaussian_sigma: Optional[float] = None,
    positive_only: Optional[bool] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
):
    if config is not None:
        positive_only_eff = (
            config.positive_only
            if positive_only is None
            else positive_only
        )
    else:
        positive_only_eff = True if positive_only is None else positive_only

    flux_smooth = smooth_flux_in_depth(
        X_grid_gcm2=X_grid_gcm2,
        flux_XE=flux_XE,
        config=config,
        method=method,
        smoothing=smoothing,
        gaussian_sigma=gaussian_sigma,
        device=device,
        dtype=dtype,
    )

    dPhi_dX = compute_depth_derivative(
        X_grid_gcm2=X_grid_gcm2,
        flux_XE=flux_smooth,
        positive_only=positive_only_eff,
        device=device,
        dtype=dtype,
    )

    return flux_smooth, dPhi_dX

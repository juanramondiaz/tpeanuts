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

    X -> Atmosphere depth (g/cm^2), the slant column of air crossed
    E -> energy grid (GeV)

Everything here is tpeanuts-native tensor arithmetic (convolutions,
finite differences); no function in this module calls MCEq. It
post-processes the raw depth-tabulated flux Phi(E, X, theta) already
produced by an MCEq cascade-equation solve (see
tpeanuts.external.mceq.solver.solve_flux_vs_depth_grid) before it is
differentiated with respect to depth to obtain the depth-differential
production source term dPhi/dX(E, X, theta), used downstream to
reconstruct a height-dependent production profile. Smoothing is applied
first because MCEq's flux, especially near the production peak in X, can
be numerically noisy enough that a naive finite-difference derivative
would be dominated by noise rather than the underlying physical
production profile.

Module functions:
    validate_flux_depth_inputs:
        Validate and coerce (X_grid_gcm2, flux_XE) to consistent
        tensors.
    gaussian_kernel1d:
        Build a normalized 1-D Gaussian convolution kernel.
    smooth_flux_gaussian:
        Smooth flux_XE along the depth axis with a Gaussian kernel.
    smooth_flux_log_moving_average:
        Smooth flux_XE along the depth axis with a moving average
        applied in log-space.
    smooth_flux_in_depth:
        Dispatch to the configured smoothing method (none, gaussian, or
        spline/log_moving_average).
    compute_depth_derivative:
        Compute dPhi/dX via a finite-difference gradient along the
        depth axis, optionally clamped to non-negative values.
    smooth_and_differentiate_flux:
        Convenience wrapper combining smooth_flux_in_depth and
        compute_depth_derivative.
"""



from __future__ import annotations

from typing import Optional, Union

import torch

from tpeanuts.util.type import as_tensor
from tpeanuts.util.torch_util import default_device

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
    """
    Validate and coerce an Atmosphere depth grid and flux table to
    consistent tensors.

    Args:
        X_grid_gcm2: Atmosphere slant-depth grid in g/cm^2; coerced to a
            1-D tensor. Must be strictly increasing.
        flux_XE: Flux tensor of shape (..., n_X, n_E), in
            (cm^2 s sr GeV)^-1.
        device: Working torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the coerced tensors.

    Returns:
        Tuple (X_t, flux_t) of the coerced 1-D depth tensor and
        coerced flux tensor.

    Raises:
        ValueError: If X_grid_gcm2 is not one-dimensional, if flux_XE
            has fewer than 2 dimensions, if flux_XE.shape[-2] does not
            match len(X_grid_gcm2), or if X_grid_gcm2 is not strictly
            increasing.
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
    """
    Build a normalized 1-D Gaussian convolution kernel.

    Args:
        sigma: Standard deviation of the Gaussian, in grid-point units
            (not physical g/cm^2 units; the depth grid is treated as
            uniformly spaced index positions for this kernel). Must be
            positive.
        truncate: Kernel half-width in units of sigma; the kernel
            radius is round(truncate * sigma).
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the kernel.

    Returns:
        1-D tensor of kernel weights, summing to 1.

    Raises:
        ValueError: If sigma is not positive.
    """
    dev = default_device(device)

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
    """
    Smooth a depth-tabulated flux along the depth axis with a Gaussian
    kernel convolution.

    Reduces numerical noise in Phi(E, X, theta) along X before its
    depth-derivative is taken (see compute_depth_derivative); applied
    independently per energy channel, with edge values replicated
    (padding mode "replicate") to avoid boundary artefacts.

    Args:
        flux_XE: Flux tensor of shape (..., n_X, n_E).
        sigma: Standard deviation of the Gaussian kernel in grid-point
            units; sigma <= 0 disables smoothing (returns a copy).
        truncate: Kernel half-width in units of sigma; passed to
            gaussian_kernel1d.
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the computation and output.

    Returns:
        Tensor with the same shape as flux_XE, smoothed along the depth
        axis (second-to-last dimension).

    Raises:
        ValueError: If flux_XE has fewer than 2 dimensions.
    """
    dev = default_device(device)

    flux_t = as_tensor(
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
    """
    Smooth a depth-tabulated flux along the depth axis with a moving
    average applied in log-space.

    This is the implementation used for the "spline"/
    "log_moving_average" smoothing method (the name "spline" in
    SmoothingConfig.method is a historical/practical label; no spline
    fit is actually performed). Working in log-space respects the fact
    that flux values are strictly positive and typically vary over many
    orders of magnitude with depth, so a multiplicative (log-additive)
    smoothing is more appropriate than an additive moving average of
    the raw flux. Applied independently per energy channel, with edge
    values replicated to avoid boundary artefacts.

    Args:
        flux_XE: Flux tensor of shape (..., n_X, n_E).
        window: Moving-average window size in grid points; even values
            are incremented by 1 to keep the window symmetric/centred.
            window <= 1 disables smoothing (returns a copy).
        eps: Small positive floor used to clamp flux values before
            taking the logarithm, to avoid log(0).
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the computation and output.

    Returns:
        Tensor with the same shape as flux_XE, smoothed along the depth
        axis (second-to-last dimension).

    Raises:
        ValueError: If flux_XE has fewer than 2 dimensions.
    """
    dev = default_device(device)

    flux_t = as_tensor(
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
    """
    Dispatch to the configured flux-smoothing method along the depth
    axis.

    Resolves the smoothing method/strength either from explicit keyword
    arguments or from a SmoothingConfig (explicit keywords take
    precedence), then applies it: "none"/None leaves the flux
    unchanged, "gaussian" applies smooth_flux_gaussian, and
    "spline"/"log_moving_average" applies
    smooth_flux_log_moving_average with a window size derived from the
    smoothing strength and the number of depth grid points (larger
    smoothing -> larger averaging window).

    Args:
        X_grid_gcm2: Atmosphere slant-depth grid in g/cm^2; used to
            validate flux_XE's shape and (for the spline method) to set
            the window size relative to the grid resolution.
        flux_XE: Flux tensor of shape (..., n_X, n_E), in
            (cm^2 s sr GeV)^-1.
        config: Optional SmoothingConfig providing defaults for method,
            smoothing and gaussian_sigma.
        method: Optional override for the smoothing method; one of
            {None, "none", "gaussian", "log_moving_average", "spline"}.
        smoothing: Optional override for the smoothing strength used by
            the "spline"/"log_moving_average" method.
        gaussian_sigma: Optional override for the Gaussian kernel
            standard deviation used by the "gaussian" method.
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the computation and output.

    Returns:
        Tensor with the same shape as flux_XE, smoothed along the depth
        axis according to the resolved method.

    Raises:
        ValueError: If the resolved method is not one of the allowed
            values, or if validate_flux_depth_inputs rejects the inputs.
    """
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
    """
    Compute the depth derivative dPhi/dX of a (typically pre-smoothed)
    flux table via finite differences.

    dPhi/dX(E, X, theta) is the depth-differential particle-production
    source term: physically, the rate at which the flux Phi(E, X,
    theta) changes per unit Atmosphere depth crossed, which downstream
    (see tpeanuts.external.mceq.profiles) is reinterpreted, via the
    Jacobian |dX/dh|, as a height-dependent production source Q_eff(E,
    h, theta) used to build the production-height profile f(h | E,
    theta).

    Args:
        X_grid_gcm2: Atmosphere slant-depth grid in g/cm^2 on which
            flux_XE is tabulated; used as the spacing for the
            finite-difference gradient.
        flux_XE: Flux tensor of shape (..., n_X, n_E), in
            (cm^2 s sr GeV)^-1.
        positive_only: If True, clamp the resulting derivative to
            non-negative values, discarding unphysical negative
            "production" that can arise from residual numerical noise.
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the computation and output.

    Returns:
        Tensor with the same shape as flux_XE holding dPhi/dX, in
        (cm^2 s sr GeV g/cm^2)^-1.

    Raises:
        ValueError: If validate_flux_depth_inputs rejects the inputs.
    """
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
    """
    Smooth a depth-tabulated flux and compute its depth derivative in
    one step.

    Convenience wrapper combining smooth_flux_in_depth (to reduce
    numerical noise in Phi(E, X, theta) along the depth axis) and
    compute_depth_derivative (to obtain the depth-differential
    production source dPhi/dX(E, X, theta)) using a single,
    consistently-resolved configuration.

    Args:
        X_grid_gcm2: Atmosphere slant-depth grid in g/cm^2 on which
            flux_XE is tabulated.
        flux_XE: Flux tensor of shape (..., n_X, n_E), in
            (cm^2 s sr GeV)^-1, as returned by
            tpeanuts.external.mceq.solver.solve_flux_vs_depth_grid.
        config: Optional SmoothingConfig providing defaults for method,
            smoothing, gaussian_sigma and positive_only.
        method: Optional override for the smoothing method.
        smoothing: Optional override for the smoothing strength.
        gaussian_sigma: Optional override for the Gaussian kernel
            standard deviation.
        positive_only: Optional override for whether the derivative is
            clamped to non-negative values.
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the computation and output.

    Returns:
        Tuple (flux_smooth, dPhi_dX):
            flux_smooth: Smoothed flux, same shape as flux_XE.
            dPhi_dX: Depth derivative of the smoothed flux, same shape
                as flux_XE, in (cm^2 s sr GeV g/cm^2)^-1.
    """
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

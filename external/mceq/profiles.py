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
Production-profile reconstruction from mceq flux gradients.

This module builds the normalized production-height probability
density:

    f(h | E, theta)

from the depth-tabulated flux:

    Phi(E, X, theta)

(obtained by solving MCEq's cascade equation; see
tpeanuts.external.mceq.solver) using the depth-differential production
source:

    Q_eff(E, X, theta) = max(dPhi/dX, 0)

and the change of variables from Atmosphere slant depth X (g/cm^2) to
altitude h (km):

    Q_eff(E, h, theta)
    =
    Q_eff(E, X(h, theta), theta) |dX/dh|

f(h | E, theta) is then Q_eff(E, h, theta) normalized to integrate to 1
over h, for each energy E and zenith angle theta; multiplying it by the
observed energy-differential flux phi_E_obs = Phi(E, X_obs, theta)
yields the height-differential flux phi_Eh used elsewhere in tpeanuts.

This module is a tpeanuts-native orchestration layer: it does not call
MCEq directly itself, but its top-level function
(production_profiles_all_energies_from_flux_gradient) drives the whole
pipeline by calling init_mceq (MCEq instantiation),
solve_flux_vs_depth_grid (MCEq cascade-equation solve),
compute_slant_depth_from_mceq (MCEq Atmosphere depth query), and
several purely tensor-algebra helpers defined in this module and in
tpeanuts.external.mceq.{solver,smoothing}.

Module functions:
    interpolate_source_X_to_h:
        Interpolate a depth-tabulated quantity (e.g. dPhi/dX) onto a
        set of depths X(h, theta) corresponding to an altitude grid.
    convert_depth_source_to_height_source:
        Apply the |dX/dh| Jacobian to convert a depth-differential
        source term into a height-differential source term.
    normalize_height_profiles:
        Normalize a height-differential source term to a probability
        density f(h | E, theta) integrating to 1 over h.
    build_phi_Eh_from_profile:
        Combine phi_E_obs and f_Eh into the height-differential flux
        phi_Eh.
    production_profiles_all_energies_from_flux_gradient:
        End-to-end orchestration: solve Phi(E, X, theta) with MCEq,
        smooth and differentiate it, convert to a height-domain source,
        normalize to f(h | E, theta), and assemble phi_Eh. Returns a
        dict bundling every intermediate and final quantity.
"""



from __future__ import annotations

from typing import Optional, Union, Dict

import torch

from tpeanuts.util.type import as_tensor
from tpeanuts.util.torch_util import default_device

from tpeanuts.external.mceq.config import (
    MCEqModelConfig,
    GridConfig,
    SmoothingConfig,
)

from tpeanuts.external.mceq.core import init_mceq
from tpeanuts.external.mceq.depth import (
    compute_slant_depth_from_mceq,
    compute_dXdh,
)
from tpeanuts.external.mceq.solver import (
    solve_flux_vs_depth_grid,
    interpolate_flux_at_Xobs,
)
from tpeanuts.external.mceq.smoothing import (
    smooth_and_differentiate_flux,
)


TensorLike = Union[float, int, torch.Tensor]


# ============================================================
# Basic interpolation
# ============================================================

@torch.no_grad()
def interpolate_source_X_to_h(
    X_grid_gcm2: TensorLike,
    source_XE: TensorLike,
    X_of_h_gcm2: TensorLike,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Interpolate a depth-tabulated source term onto the depths
    corresponding to an altitude grid.

    Pure tensor interpolation (linear in X, via torch.searchsorted); no
    MCEq call. Given a quantity tabulated on the solver's Atmosphere
    depth grid X_grid_gcm2 (e.g. dPhi/dX(E, X, theta)), evaluates it at
    the depths X_of_h_gcm2 = X(h, theta) that correspond to a desired
    altitude grid h, as the first step of converting a depth-domain
    source term into a height-domain one (see
    convert_depth_source_to_height_source). Points in X_of_h_gcm2
    outside the range of X_grid_gcm2 are set to zero.

    Args:
        X_grid_gcm2: Strictly increasing 1-D Atmosphere slant-depth
            grid in g/cm^2 on which source_XE is tabulated.
        source_XE: Source tensor of shape (..., n_X, n_E), e.g.
            dPhi/dX in (cm^2 s sr GeV g/cm^2)^-1.
        X_of_h_gcm2: Depths in g/cm^2, of shape (..., n_h), at which to
            evaluate source_XE; typically X(h, theta) for an altitude
            grid h.
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the computation and output.

    Returns:
        Tensor of shape (..., n_E, n_h) holding source_XE interpolated
        onto X_of_h_gcm2, with entries outside the range of
        X_grid_gcm2 set to zero.

    Raises:
        ValueError: If source_XE has fewer than 2 dimensions, if
            source_XE.shape[-2] does not match len(X_grid_gcm2), or if
            X_grid_gcm2 is not strictly increasing.
    """
    dev = default_device(device)

    X_t = as_tensor(
        X_grid_gcm2,
        device=dev,
        dtype=dtype,
    ).reshape(-1)

    source_t = as_tensor(
        source_XE,
        device=dev,
        dtype=dtype,
    )

    Xh_t = as_tensor(
        X_of_h_gcm2,
        device=dev,
        dtype=dtype,
    )

    if source_t.ndim < 2:
        raise ValueError("source_XE must have shape (..., n_X, n_E).")

    if Xh_t.ndim < 1:
        Xh_t = Xh_t.reshape(1)

    if source_t.shape[-2] != X_t.numel():
        raise ValueError(
            "source_XE.shape[-2] must match len(X_grid_gcm2)."
        )

    if torch.any(torch.diff(X_t) <= 0.0):
        raise ValueError("X_grid_gcm2 must be strictly increasing.")

    batch_shape = torch.broadcast_shapes(
        source_t.shape[:-2],
        Xh_t.shape[:-1],
    )
    n_X = X_t.numel()
    n_E = source_t.shape[-1]
    n_h = Xh_t.shape[-1]

    source_b = torch.broadcast_to(source_t, (*batch_shape, n_X, n_E))
    Xh_b = torch.broadcast_to(Xh_t, (*batch_shape, n_h))

    valid = (Xh_b >= X_t[0]) & (Xh_b <= X_t[-1])

    if not torch.any(valid):
        return torch.zeros((*batch_shape, n_E, n_h), device=dev, dtype=dtype)

    idx = torch.searchsorted(
        X_t,
        torch.clamp(Xh_b, min=X_t[0], max=X_t[-1]),
        right=False,
    )

    idx = torch.clamp(
        idx,
        min=1,
        max=X_t.numel() - 1,
    )

    x0 = X_t[idx - 1]
    x1 = X_t[idx]

    w = (Xh_b - x0) / (x1 - x0)

    gather_shape = (*batch_shape, n_h, n_E)
    idx0 = (idx - 1).unsqueeze(-1).expand(gather_shape)
    idx1 = idx.unsqueeze(-1).expand(gather_shape)

    y0 = torch.gather(source_b, dim=-2, index=idx0)
    y1 = torch.gather(source_b, dim=-2, index=idx1)

    yq = (1.0 - w.unsqueeze(-1)) * y0 + w.unsqueeze(-1) * y1
    yq = torch.where(valid.unsqueeze(-1), yq, torch.zeros_like(yq))

    return yq.movedim(-2, -1)


# ============================================================
# Height source and normalization
# ============================================================

@torch.no_grad()
def convert_depth_source_to_height_source(
    X_grid_gcm2: TensorLike,
    source_XE: TensorLike,
    X_of_h_gcm2: TensorLike,
    dXdh_gcm2_per_km: TensorLike,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Convert a depth-differential source term into a height-differential
    source term via the |dX/dh| Jacobian.

    Implements the change of variables
    Q_eff(E, h, theta) = Q_eff(E, X(h, theta), theta) * |dX/dh|: first
    interpolates source_XE (a function of depth X) onto the depths
    X_of_h_gcm2 corresponding to the altitude grid (via
    interpolate_source_X_to_h), then multiplies by the absolute value
    of the depth-altitude Jacobian dXdh_gcm2_per_km (which is
    physically negative, since X decreases as h increases).

    Args:
        X_grid_gcm2: Strictly increasing 1-D Atmosphere slant-depth
            grid in g/cm^2 on which source_XE is tabulated.
        source_XE: Depth-differential source tensor of shape
            (..., n_X, n_E), e.g. dPhi/dX in
            (cm^2 s sr GeV g/cm^2)^-1.
        X_of_h_gcm2: Depths in g/cm^2 corresponding to the altitude
            grid, shape (..., n_h); typically X(h, theta).
        dXdh_gcm2_per_km: Depth-altitude derivative dX/dh in
            g/cm^2/km, shape (..., n_h), matching X_of_h_gcm2;
            typically negative.
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the computation and output.

    Returns:
        Height-differential source tensor of shape (..., n_E, n_h), in
        (cm^2 s sr GeV km)^-1.

    Raises:
        ValueError: If dXdh_gcm2_per_km does not have the same trailing
            length as X_of_h_gcm2, or if interpolate_source_X_to_h
            rejects the inputs.
    """
    dev = default_device(device)

    dXdh_t = as_tensor(
        dXdh_gcm2_per_km,
        device=dev,
        dtype=dtype,
    )

    if dXdh_t.ndim < 1:
        dXdh_t = dXdh_t.reshape(1)

    source_interp_Eh = interpolate_source_X_to_h(
        X_grid_gcm2=X_grid_gcm2,
        source_XE=source_XE,
        X_of_h_gcm2=X_of_h_gcm2,
        device=dev,
        dtype=dtype,
    )

    if source_interp_Eh.shape[-1] != dXdh_t.shape[-1]:
        raise ValueError(
            "dXdh_gcm2_per_km must have same length as X_of_h_gcm2."
        )

    dXdh_b = torch.broadcast_to(
        dXdh_t,
        (*source_interp_Eh.shape[:-2], source_interp_Eh.shape[-1]),
    )

    return source_interp_Eh * torch.abs(dXdh_b).unsqueeze(-2)


@torch.no_grad()
def normalize_height_profiles(
    h_grid_km: TensorLike,
    source_Eh: TensorLike,
    *,
    eps: float = 0.0,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Normalize a height-differential source term to a probability
    density integrating to 1 over altitude.

    For each energy (and any leading batch dimensions), divides
    source_Eh by its trapezoidal integral over h_grid_km, producing the
    normalized production-height probability density f(h | E, theta)
    (units 1/km). Energy channels whose integral is at or below eps
    (e.g. zero production) are returned as all-zero rows rather than
    NaN/Inf from a division by zero.

    Args:
        h_grid_km: Strictly increasing 1-D altitude grid in km.
        source_Eh: Height-differential source tensor of shape
            (..., n_E, n_h), e.g. the output of
            convert_depth_source_to_height_source, in
            (cm^2 s sr GeV km)^-1.
        eps: Threshold below which an energy channel's integral over h
            is treated as zero (avoiding division by a near-zero norm).
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the computation and output.

    Returns:
        Tensor f_Eh with the same shape as source_Eh, in units of
        1/km, satisfying integral over h of f_Eh dh = 1 for each energy
        channel whose norm exceeds eps (all-zero otherwise).

    Raises:
        ValueError: If source_Eh has fewer than 2 dimensions, or if
            source_Eh.shape[-1] does not match len(h_grid_km).
    """
    dev = default_device(device)

    h_t = as_tensor(
        h_grid_km,
        device=dev,
        dtype=dtype,
    ).reshape(-1)

    source_t = as_tensor(
        source_Eh,
        device=dev,
        dtype=dtype,
    )

    if source_t.ndim < 2:
        raise ValueError("source_Eh must have shape (..., n_E, n_h).")

    if source_t.shape[-1] != h_t.numel():
        raise ValueError(
            "source_Eh.shape[-1] must match len(h_grid_km)."
        )

    norm_E = torch.trapezoid(
        source_t,
        x=h_t,
        dim=-1,
    )

    return torch.where(
        (norm_E > eps).unsqueeze(-1),
        source_t / norm_E.clamp_min(eps).unsqueeze(-1),
        torch.zeros_like(source_t),
    )


@torch.no_grad()
def build_phi_Eh_from_profile(
    phi_E_obs: TensorLike,
    f_Eh: TensorLike,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Combine the observed energy-differential flux and the normalized
    production-height profile into the height-differential flux.

    Computes phi_Eh(E, h) = phi_E_obs(E) * f_Eh(E, h): the observed
    flux at the observation depth is distributed over altitude
    according to the normalized production-height probability density,
    so that integrating phi_Eh over h recovers phi_E_obs for each
    energy.

    Args:
        phi_E_obs: Energy-differential flux at the observation depth,
            shape (..., n_E), in (cm^2 s sr GeV)^-1.
        f_Eh: Normalized production-height density, shape
            (..., n_E, n_h), in 1/km (integrates to 1 over h per energy
            channel).
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the computation and output.

    Returns:
        Height-differential flux tensor of shape (..., n_E, n_h), in
        (cm^2 s sr GeV km)^-1.

    Raises:
        ValueError: If f_Eh has fewer than 2 dimensions, or if
            f_Eh.shape[-2] does not match phi_E_obs.shape[-1].
    """
    dev = default_device(device)

    phi_t = as_tensor(
        phi_E_obs,
        device=dev,
        dtype=dtype,
    )

    f_t = as_tensor(
        f_Eh,
        device=dev,
        dtype=dtype,
    )

    if f_t.ndim < 2:
        raise ValueError("f_Eh must have shape (..., n_E, n_h).")

    if phi_t.shape[-1] != f_t.shape[-2]:
        raise ValueError(
            "f_Eh.shape[-2] must match phi_E_obs.shape[-1]."
        )

    batch_shape = torch.broadcast_shapes(phi_t.shape[:-1], f_t.shape[:-2])
    phi_b = torch.broadcast_to(phi_t, (*batch_shape, phi_t.shape[-1]))
    f_b = torch.broadcast_to(f_t, (*batch_shape, f_t.shape[-2], f_t.shape[-1]))

    return phi_b.unsqueeze(-1) * f_b


# ============================================================
# Full profile reconstruction
# ============================================================

@torch.no_grad()
def production_profiles_all_energies_from_flux_gradient(
    theta_deg: TensorLike,
    particle: str,
    model_config: Optional[MCEqModelConfig] = None,
    grid_config: Optional[GridConfig] = None,
    smoothing_config: Optional[SmoothingConfig] = None,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> Dict[str, torch.Tensor]:
    """
    End-to-end reconstruction of the production-height profile and
    height-differential flux for one particle species and zenith angle.

    This is the top-level orchestration function of the MCEq pipeline:
    it instantiates an MCEqRun (via init_mceq), drives the external
    cascade-equation solve (via solve_flux_vs_depth_grid) to obtain
    Phi(E, X, theta) on the configured depth grid, evaluates the
    observed flux phi_E_obs = Phi(E, X_obs, theta), smooths and
    differentiates the depth-tabulated flux to obtain the production
    source dPhi/dX, maps Atmosphere depth to altitude via
    compute_slant_depth_from_mceq and compute_dXdh, converts the
    depth-domain source into a height-domain source (Jacobian
    |dX/dh|), normalizes it into the production-height density
    f(h | E, theta), and finally builds the height-differential flux
    phi_Eh = phi_E_obs * f_Eh. The numbered inline comments in the body
    mark each of these seven steps.

    Args:
        theta_deg: Zenith angle in degrees (0 <= theta_deg < 90) of the
            shower/neutrino trajectory; theta_deg=0 is vertical.
        particle: MCEq particle name (e.g. "numu", "mu+") for which to
            solve and reconstruct the production profile.
        model_config: Optional MCEqModelConfig selecting the
            interaction/primary/density models; defaults to
            MCEqModelConfig() if omitted.
        grid_config: Optional GridConfig providing the Atmosphere depth
            grid X_grid_gcm2, altitude grid h_grid_km and observation
            depth X_obs_gcm2; defaults to GridConfig() if omitted.
        smoothing_config: Optional SmoothingConfig controlling the
            depth-axis smoothing applied before differentiation;
            defaults to SmoothingConfig() if omitted.
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used throughout the computation.

    Returns:
        Dict with the following tensor entries:
            theta_deg: Scalar zenith angle in degrees.
            E_grid_GeV: Energy grid in GeV, shape (n_E,).
            X_grid_gcm2: Atmosphere slant-depth grid in g/cm^2, shape
                (n_X,).
            h_grid_km: Altitude grid in km, shape (n_h,).
            flux_XE: Raw solved flux Phi(E, X, theta), shape
                (n_X, n_E), in (cm^2 s sr GeV)^-1.
            flux_smooth_XE: Depth-smoothed flux, same shape/units as
                flux_XE.
            dPhi_dX_XE: Depth derivative of the smoothed flux, shape
                (n_X, n_E), in (cm^2 s sr GeV g/cm^2)^-1.
            X_of_h_gcm2: Slant depth X(h, theta) on the altitude grid,
                shape (n_h,), in g/cm^2.
            dXdh_gcm2_per_km: Depth-altitude derivative dX/dh on the
                altitude grid, shape (n_h,), in g/cm^2/km.
            source_Eh: Height-differential, unnormalized production
                source, shape (n_E, n_h), in (cm^2 s sr GeV km)^-1.
            f_Eh: Normalized production-height density, shape
                (n_E, n_h), in 1/km.
            phi_E_obs: Energy-differential flux at the observation
                depth, shape (n_E,), in (cm^2 s sr GeV)^-1.
            phi_Eh: Height-differential flux, shape (n_E, n_h), in
                (cm^2 s sr GeV km)^-1.

    Raises:
        ImportError: If the optional MCEq package is not installed.
        ValueError: If model_config, grid_config or smoothing_config
            fail validation, or if theta_deg is outside [0, 90) degrees.
    """
    dev = default_device(device)

    if model_config is None:
        model_config = MCEqModelConfig()

    if grid_config is None:
        grid_config = GridConfig()

    if smoothing_config is None:
        smoothing_config = SmoothingConfig()

    model_config.validate()
    grid_config.validate()
    smoothing_config.validate()

    X_grid = as_tensor(
        grid_config.X_grid_gcm2,
        device=dev,
        dtype=dtype,
    ).reshape(-1)

    h_grid = as_tensor(
        grid_config.h_grid_km,
        device=dev,
        dtype=dtype,
    ).reshape(-1)

    mceq = init_mceq(
        theta_deg=theta_deg,
        config=model_config,
    )

    # --------------------------------------------------------
    # 1. Solve Phi(E, X, theta)
    # --------------------------------------------------------
    X_grid, E_grid, flux_XE = solve_flux_vs_depth_grid(
        theta_deg=theta_deg,
        particle=particle,
        X_grid_gcm2=X_grid,
        config=model_config,
        mceq=mceq,
        device=dev,
        dtype=dtype,
    )

    # --------------------------------------------------------
    # 2. Phi(E, X_obs, theta)
    # --------------------------------------------------------
    phi_E_obs = interpolate_flux_at_Xobs(
        X_grid_gcm2=X_grid,
        flux_XE=flux_XE,
        X_obs_gcm2=grid_config.X_obs_gcm2,
        log_interp=True,
        device=dev,
        dtype=dtype,
    )

    # --------------------------------------------------------
    # 3-4. Smooth and differentiate
    # --------------------------------------------------------
    flux_smooth, dPhi_dX = smooth_and_differentiate_flux(
        X_grid_gcm2=X_grid,
        flux_XE=flux_XE,
        config=smoothing_config,
        device=dev,
        dtype=dtype,
    )

    # --------------------------------------------------------
    # 5. X(h, theta)
    # --------------------------------------------------------
    X_of_h = compute_slant_depth_from_mceq(
        h_km=h_grid,
        theta_deg=theta_deg,
        mceq=mceq,
        config=model_config,
        device=dev,
        dtype=dtype,
    )

    dXdh = compute_dXdh(
        X_gcm2=X_of_h,
        h_km=h_grid,
        device=dev,
        dtype=dtype,
    )

    # --------------------------------------------------------
    # 6. Q(E, X) -> Q(E, h)
    # --------------------------------------------------------
    source_Eh = convert_depth_source_to_height_source(
        X_grid_gcm2=X_grid,
        source_XE=dPhi_dX,
        X_of_h_gcm2=X_of_h,
        dXdh_gcm2_per_km=dXdh,
        device=dev,
        dtype=dtype,
    )

    # --------------------------------------------------------
    # 7. Normalize f(h | E, theta)
    # --------------------------------------------------------
    f_Eh = normalize_height_profiles(
        h_grid_km=h_grid,
        source_Eh=source_Eh,
        device=dev,
        dtype=dtype,
    )

    phi_Eh = build_phi_Eh_from_profile(
        phi_E_obs=phi_E_obs,
        f_Eh=f_Eh,
        device=dev,
        dtype=dtype,
    )

    return {
        "theta_deg": as_tensor(theta_deg, device=dev, dtype=dtype).reshape(()),
        "E_grid_GeV": E_grid,
        "X_grid_gcm2": X_grid,
        "h_grid_km": h_grid,
        "flux_XE": flux_XE,
        "flux_smooth_XE": flux_smooth,
        "dPhi_dX_XE": dPhi_dX,
        "X_of_h_gcm2": X_of_h,
        "dXdh_gcm2_per_km": dXdh,
        "source_Eh": source_Eh,
        "f_Eh": f_Eh,
        "phi_E_obs": phi_E_obs,
        "phi_Eh": phi_Eh,
    }

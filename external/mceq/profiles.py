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

This module builds:

    f(h | E, theta)

from:

    Phi(E, X, theta)

using:

    Q_eff(E, X, theta) = max(dPhi/dX, 0)

and the change of variables:

    Q_eff(E, h, theta)
    =
    Q_eff(E, X(h, theta), theta) |dX/dh|
"""



from __future__ import annotations

from typing import Optional, Union, Dict

import torch

from tpeanuts.util.type import _as_tensor
from tpeanuts.util.torch_util import _default_device

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
    dev = _default_device(device)

    X_t = _as_tensor(
        X_grid_gcm2,
        device=dev,
        dtype=dtype,
    ).reshape(-1)

    source_t = _as_tensor(
        source_XE,
        device=dev,
        dtype=dtype,
    )

    Xh_t = _as_tensor(
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
    dev = _default_device(device)

    dXdh_t = _as_tensor(
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
    dev = _default_device(device)

    h_t = _as_tensor(
        h_grid_km,
        device=dev,
        dtype=dtype,
    ).reshape(-1)

    source_t = _as_tensor(
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
    dev = _default_device(device)

    phi_t = _as_tensor(
        phi_E_obs,
        device=dev,
        dtype=dtype,
    )

    f_t = _as_tensor(
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
    dev = _default_device(device)

    if model_config is None:
        model_config = MCEqModelConfig()

    if grid_config is None:
        grid_config = GridConfig()

    if smoothing_config is None:
        smoothing_config = SmoothingConfig()

    model_config.validate()
    grid_config.validate()
    smoothing_config.validate()

    X_grid = _as_tensor(
        grid_config.X_grid_gcm2,
        device=dev,
        dtype=dtype,
    ).reshape(-1)

    h_grid = _as_tensor(
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
        "theta_deg": _as_tensor(theta_deg, device=dev, dtype=dtype).reshape(()),
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

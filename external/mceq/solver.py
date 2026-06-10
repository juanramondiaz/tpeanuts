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
mceq flux solvers.

This module contains direct calls to mceq for obtaining:

    Phi(E; X_obs, theta)

and

    Phi(E, X, theta)

It should not contain smoothing, profile reconstruction, I/O, or plotting.
"""



from __future__ import annotations

from typing import Optional, Union, Tuple

import torch

from tpeanuts.util.type import _as_tensor
from tpeanuts.util.torch_util import _default_device

from tpeanuts.external.mceq.config import (
    MCEqModelConfig,
    GridConfig,
)

from tpeanuts.external.mceq.core import init_mceq


TensorLike = Union[float, int, torch.Tensor]


# ============================================================
# Internal helpers
# ============================================================

def _to_1d_cpu_double(x: TensorLike) -> torch.Tensor:
    x_t = _as_tensor(
        x,
        device="cpu",
        dtype=torch.float64,
    ).reshape(-1)

    return x_t


def _get_energy_grid(mceq) -> torch.Tensor:
    return torch.as_tensor(
        mceq.e_grid,
        dtype=torch.float64,
        device="cpu",
    )


def _get_solution_tensor(
    mceq,
    particle: str,
    *,
    grid_idx: Optional[int] = None,
) -> torch.Tensor:
    if grid_idx is None:
        phi = mceq.get_solution(particle)
    else:
        phi = mceq.get_solution(particle, grid_idx=grid_idx)

    return torch.as_tensor(
        phi,
        dtype=torch.float64,
        device="cpu",
    ).clamp_min(0.0)


def _solve_mceq_at_depth_grid(
    mceq,
    X_grid_gcm2: torch.Tensor,
    particle: str,
) -> torch.Tensor:
    X_numpy = X_grid_gcm2.detach().cpu().numpy()

    mceq.solve(int_grid=X_numpy)

    flux_list = [
        _get_solution_tensor(
            mceq,
            particle,
            grid_idx=i,
        )
        for i in range(len(X_grid_gcm2))
    ]

    return torch.stack(flux_list, dim=0)


def _solve_mceq_depth_by_depth(
    theta_deg: TensorLike,
    X_grid_gcm2: torch.Tensor,
    particle: str,
    config: Optional[MCEqModelConfig] = None,
    interaction_model: Optional[str] = None,
    primary_model: Optional[Union[str, tuple]] = None,
    density_model: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    flux_list = []
    E_grid = None

    for X in X_grid_gcm2:

        mceq = init_mceq(
            theta_deg=theta_deg,
            config=config,
            interaction_model=interaction_model,
            primary_model=primary_model,
            density_model=density_model,
            info=False,
        )

        try:
            mceq.solve(max_depth=float(X.item()))
            phi = _get_solution_tensor(mceq, particle)

        except Exception:
            mceq.solve()
            phi = _get_solution_tensor(mceq, particle)

        if E_grid is None:
            E_grid = _get_energy_grid(mceq)

        flux_list.append(phi)

    flux_XE = torch.stack(flux_list, dim=0)

    return E_grid, flux_XE


# ============================================================
# Public solvers
# ============================================================

@torch.no_grad()
def solve_flux_vs_depth_grid(
    theta_deg: TensorLike,
    particle: str,
    X_grid_gcm2: Optional[TensorLike] = None,
    config: Optional[MCEqModelConfig] = None,
    grid_config: Optional[GridConfig] = None,
    interaction_model: Optional[str] = None,
    primary_model: Optional[Union[str, tuple]] = None,
    density_model: Optional[str] = None,
    mceq=None,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
    fallback: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = _default_device(device)

    if X_grid_gcm2 is None:
        if grid_config is not None:
            X_grid_gcm2 = grid_config.X_grid_gcm2
        else:
            X_grid_gcm2 = torch.linspace(
                1.0,
                1030.0,
                220,
                device="cpu",
                dtype=torch.float64,
            )

    X_cpu = _to_1d_cpu_double(X_grid_gcm2)

    if torch.any(torch.diff(X_cpu) <= 0.0):
        raise ValueError("X_grid_gcm2 must be strictly increasing.")

    if mceq is None:
        mceq = init_mceq(
            theta_deg=theta_deg,
            config=config,
            interaction_model=interaction_model,
            primary_model=primary_model,
            density_model=density_model,
            info=False,
        )

    try:
        flux_XE_cpu = _solve_mceq_at_depth_grid(
            mceq=mceq,
            X_grid_gcm2=X_cpu,
            particle=particle,
        )

        E_cpu = _get_energy_grid(mceq)

    except Exception:
        if not fallback:
            raise

        E_cpu, flux_XE_cpu = _solve_mceq_depth_by_depth(
            theta_deg=theta_deg,
            X_grid_gcm2=X_cpu,
            particle=particle,
            config=config,
            interaction_model=interaction_model,
            primary_model=primary_model,
            density_model=density_model,
        )

    return (
        X_cpu.to(device=dev, dtype=dtype),
        E_cpu.to(device=dev, dtype=dtype),
        flux_XE_cpu.to(device=dev, dtype=dtype),
    )


@torch.no_grad()
def get_mceq_flux_at_Xobs(
    theta_deg: TensorLike,
    particle: str,
    X_obs_gcm2: Optional[TensorLike] = None,
    config: Optional[MCEqModelConfig] = None,
    grid_config: Optional[GridConfig] = None,
    interaction_model: Optional[str] = None,
    primary_model: Optional[Union[str, tuple]] = None,
    density_model: Optional[str] = None,
    mceq=None,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dev = _default_device(device)

    if X_obs_gcm2 is None:
        if grid_config is not None:
            X_obs_gcm2 = grid_config.X_obs_gcm2
        else:
            X_obs_gcm2 = 1030.0

    X_obs_t = _as_tensor(
        X_obs_gcm2,
        device="cpu",
        dtype=torch.float64,
    ).reshape(-1)[0]

    if mceq is None:
        mceq = init_mceq(
            theta_deg=theta_deg,
            config=config,
            interaction_model=interaction_model,
            primary_model=primary_model,
            density_model=density_model,
            info=False,
        )

    try:
        mceq.solve(int_grid=X_obs_t.reshape(1).numpy())
        phi_E_cpu = _get_solution_tensor(
            mceq,
            particle,
            grid_idx=0,
        )

    except Exception:
        try:
            mceq.solve(max_depth=float(X_obs_t.item()))
            phi_E_cpu = _get_solution_tensor(mceq, particle)

        except Exception:
            mceq.solve()
            phi_E_cpu = _get_solution_tensor(mceq, particle)

    E_cpu = _get_energy_grid(mceq)

    return (
        E_cpu.to(device=dev, dtype=dtype),
        phi_E_cpu.to(device=dev, dtype=dtype),
    )


# ============================================================
# Interpolation
# ============================================================

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

    X_obs_t = _as_tensor(
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

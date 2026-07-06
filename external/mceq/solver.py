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

i.e. the energy-differential particle flux (in units of
(cm^2 s sr GeV)^-1) produced by MCEq's cascade-equation solve, either at
a single Atmosphere slant depth X_obs (g/cm^2) or tabulated over a full
depth grid X (g/cm^2), for a fixed zenith angle theta and particle
species. Every "solve" call in this module (mceq.solve(...)) directly
invokes the external MCEq package's numerical cascade-equation
integrator; this module's own code is limited to driving those calls
(constructing/reusing an MCEqRun via init_mceq, handling MCEq API
fallbacks) and converting MCEq's numpy outputs into torch tensors.

It should not contain smoothing, profile reconstruction, I/O, or plotting.

Module functions:
    solve_flux_vs_depth_grid:
        Solve MCEq's cascade equations and tabulate the
        energy-differential flux Phi(E, X, theta) for one particle
        species over a grid of Atmosphere slant depths X.
    get_mceq_flux_at_Xobs:
        Solve MCEq's cascade equations and extract the
        energy-differential flux Phi(E, X_obs, theta) at a single
        observation depth.
    interpolate_flux_at_Xobs:
        Tensor-only (no MCEq call) interpolation of a depth-tabulated
        flux Phi(E, X, theta) to a specific observation depth X_obs,
        used as a cheaper alternative to re-solving MCEq at X_obs once
        solve_flux_vs_depth_grid has already produced a full depth
        table.
"""



from __future__ import annotations

from typing import Optional, Union, Tuple

import torch

from tpeanuts.util.type import as_tensor
from tpeanuts.util.torch_util import default_device

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
    """Coerce a tensor-like value to a 1-D CPU float64 tensor."""
    x_t = as_tensor(
        x,
        device="cpu",
        dtype=torch.float64,
    ).reshape(-1)

    return x_t


def _get_energy_grid(mceq) -> torch.Tensor:
    """
    Read the energy grid (GeV) directly from an MCEqRun instance
    (mceq.e_grid, set internally by MCEq's interaction-model tables) and
    return it as a CPU float64 tensor.
    """
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
    """
    Call MCEq's mceq.get_solution(particle, ...) to extract the
    energy-differential flux for one particle species from the most
    recent mceq.solve(...) call, clamped to non-negative values.

    Args:
        mceq: Initialized, already-solved MCEqRun instance.
        particle: MCEq particle name (e.g. "numu", "antinumu").
        grid_idx: Optional index into the depth grid supplied to
            mceq.solve(int_grid=...); None retrieves the solution at the
            final (deepest) integration point.

    Returns:
        1-D CPU float64 tensor of the energy-differential flux in
        (cm^2 s sr GeV)^-1 at the requested depth grid index.
    """
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
    """
    Solve MCEq's cascade equations once over an explicit Atmosphere
    depth grid (mceq.solve(int_grid=X_grid_gcm2)) and collect the
    resulting flux at every depth grid point.

    This is the preferred, single-solve strategy: MCEq's int_grid
    feature integrates the cascade equations once while recording the
    solution at every requested intermediate depth, rather than
    re-solving from scratch at each depth (see
    _solve_mceq_depth_by_depth for the fallback that does the latter
    when int_grid is not supported for the active configuration).

    Args:
        mceq: Initialized MCEqRun instance.
        X_grid_gcm2: 1-D CPU tensor of Atmosphere slant depths in
            g/cm^2, strictly increasing.
        particle: MCEq particle name whose flux is extracted.

    Returns:
        Tensor of shape (n_X, n_E) with the energy-differential flux at
        each depth in X_grid_gcm2, in (cm^2 s sr GeV)^-1.
    """
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
    """
    Fallback solver: re-initializes a fresh MCEqRun and re-solves MCEq's
    cascade equations independently for every requested depth.

    Used by solve_flux_vs_depth_grid when the single-solve
    int_grid strategy of _solve_mceq_at_depth_grid fails (e.g. due to
    MCEq API differences for a given interaction/density model). For
    each depth X, this constructs a new MCEqRun (via init_mceq) and
    calls mceq.solve(max_depth=X), or mceq.solve() with no constraint
    as a last resort, which is markedly slower since the cascade
    equations are re-integrated from scratch at every depth.

    Args:
        theta_deg: Zenith angle in degrees passed to init_mceq for each
            re-initialized MCEqRun.
        X_grid_gcm2: 1-D CPU tensor of Atmosphere slant depths in
            g/cm^2 at which to solve.
        particle: MCEq particle name whose flux is extracted.
        config: Optional MCEq model configuration object.
        interaction_model: Optional interaction-model override for
            init_mceq.
        primary_model: Optional primary cosmic-ray model override for
            init_mceq.
        density_model: Optional MCEq density-model override for
            init_mceq.

    Returns:
        Tuple (E_grid, flux_XE) where E_grid is the 1-D CPU float64
        energy grid in GeV, and flux_XE has shape (n_X, n_E) with the
        energy-differential flux at each requested depth, in
        (cm^2 s sr GeV)^-1.
    """
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
    """
    Solve MCEq's cascade equations and tabulate the energy-differential
    flux Phi(E, X, theta) over an Atmosphere slant-depth grid.

    This is a direct driver of the external MCEq solver: it builds (or
    reuses) an MCEqRun instance via init_mceq for the given zenith
    angle and physics-model selection, then solves the cascade
    equations either in a single pass over the full depth grid
    (preferred, via _solve_mceq_at_depth_grid and MCEq's int_grid
    option) or, if that raises an exception and fallback=True,
    depth-by-depth by re-solving MCEq independently at each requested X
    (via _solve_mceq_depth_by_depth). The physical result is the
    same either way: the secondary-particle flux of the requested
    species as a function of energy E and how much atmosphere (slant
    depth X) the shower/neutrino has crossed, for a fixed zenith angle.

    Args:
        theta_deg: Zenith angle in degrees (0 <= theta_deg < 90) of the
            trajectory; only used to build a new MCEqRun when mceq is
            not supplied.
        particle: MCEq particle name whose flux is extracted (e.g.
            "numu", "antinumu", "mu+").
        X_grid_gcm2: Atmosphere slant-depth grid in g/cm^2 at which to
            tabulate the flux; must be strictly increasing. Defaults to
            grid_config.X_grid_gcm2 if grid_config is given, otherwise
            220 points linearly spaced in [1, 1030].
        config: Optional MCEq model configuration object used to build a
            new MCEqRun when mceq is not supplied.
        grid_config: Optional GridConfig providing the default
            X_grid_gcm2 when X_grid_gcm2 is not given explicitly.
        interaction_model: Optional interaction-model override for
            init_mceq.
        primary_model: Optional primary cosmic-ray model override for
            init_mceq.
        density_model: Optional MCEq density-model override for
            init_mceq.
        mceq: Optional pre-initialized MCEqRun instance to reuse instead
            of constructing a new one.
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the returned tensors.
        fallback: If True, fall back to the slower depth-by-depth solve
            strategy when the single-pass int_grid solve fails; if
            False, propagate the original exception.

    Returns:
        Tuple (X_grid_gcm2, E_grid_GeV, flux_XE):
            X_grid_gcm2: 1-D tensor of the Atmosphere slant depths used,
                in g/cm^2.
            E_grid_GeV: 1-D tensor of the MCEq energy grid, in GeV.
            flux_XE: Tensor of shape (n_X, n_E) with the
                energy-differential flux Phi(E, X, theta) in
                (cm^2 s sr GeV)^-1, clamped to non-negative values.
    """
    dev = default_device(device)

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
    """
    Solve MCEq's cascade equations and extract the energy-differential
    flux Phi(E, X_obs, theta) at a single Atmosphere observation depth.

    Direct driver of the external MCEq solver for the common case of
    needing the flux at one specific depth (e.g. the ground or a
    detector's slant depth) rather than over a full depth grid. It
    first tries MCEq's int_grid mechanism restricted to the single
    requested depth; if that fails it falls back to
    mceq.solve(max_depth=X_obs_gcm2), and as a last resort to an
    unconstrained mceq.solve() (which solves to the maximum
    configured depth, not necessarily X_obs_gcm2).

    Args:
        theta_deg: Zenith angle in degrees (0 <= theta_deg < 90) of the
            trajectory; only used to build a new MCEqRun when mceq is
            not supplied.
        particle: MCEq particle name whose flux is extracted.
        X_obs_gcm2: Atmosphere slant depth in g/cm^2 at which to
            evaluate the flux. Defaults to grid_config.X_obs_gcm2 if
            grid_config is given, otherwise 1030.0 (approximately sea
            level, vertical equivalent).
        config: Optional MCEq model configuration object used to build a
            new MCEqRun when mceq is not supplied.
        grid_config: Optional GridConfig providing the default
            X_obs_gcm2 when X_obs_gcm2 is not given explicitly.
        interaction_model: Optional interaction-model override for
            init_mceq.
        primary_model: Optional primary cosmic-ray model override for
            init_mceq.
        density_model: Optional MCEq density-model override for
            init_mceq.
        mceq: Optional pre-initialized MCEqRun instance to reuse instead
            of constructing a new one.
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the returned tensors.

    Returns:
        Tuple (E_grid_GeV, phi_E_obs):
            E_grid_GeV: 1-D tensor of the MCEq energy grid, in GeV.
            phi_E_obs: 1-D tensor of the energy-differential flux
                Phi(E, X_obs, theta) in (cm^2 s sr GeV)^-1, clamped to
                non-negative values.
    """
    dev = default_device(device)

    if X_obs_gcm2 is None:
        if grid_config is not None:
            X_obs_gcm2 = grid_config.X_obs_gcm2
        else:
            X_obs_gcm2 = 1030.0

    X_obs_t = as_tensor(
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
    """
    Interpolate a depth-tabulated flux Phi(E, X, theta) to a specific
    observation depth X_obs, without calling MCEq.

    This is a tpeanuts-native, piecewise-linear interpolation along the
    depth axis of a flux table already produced by
    solve_flux_vs_depth_grid; it is a cheaper alternative to calling
    get_mceq_flux_at_Xobs (which re-solves or re-queries MCEq) once a
    full depth grid has already been solved. Particle fluxes typically
    vary over many orders of magnitude with depth, so by default the
    interpolation is performed in log-space (log_interp=True), which is
    generally more accurate for steeply falling/rising flux curves than
    linear interpolation of the flux values directly.

    Args:
        X_grid_gcm2: Strictly increasing 1-D Atmosphere slant-depth grid
            in g/cm^2 on which flux_XE is tabulated.
        flux_XE: Flux tensor of shape (..., n_X, n_E) as returned by
            solve_flux_vs_depth_grid, in (cm^2 s sr GeV)^-1.
        X_obs_gcm2: Observation depth(s) in g/cm^2 at which to
            interpolate; must lie within the range of X_grid_gcm2.
        log_interp: If True, interpolate log(flux) linearly in X and
            exponentiate the result (recommended for flux quantities
            spanning many decades); if False, interpolate flux values
            linearly in X.
        eps: Small positive floor used to clamp flux values before
            taking the logarithm when log_interp is True, to avoid
            log(0).
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the computation and output.

    Returns:
        Tensor of the energy-differential flux at X_obs_gcm2, in
        (cm^2 s sr GeV)^-1, with shape (..., n_E) broadcast over the
        batch dimensions of flux_XE and X_obs_gcm2.

    Raises:
        ValueError: If flux_XE does not have shape (..., n_X, n_E)
            matching X_grid_gcm2, if X_grid_gcm2 is not strictly
            increasing, or if X_obs_gcm2 falls outside the range of
            X_grid_gcm2.
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

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

    Phi(E, X, alpha)

the energy-differential particle flux (in units of
(cm^2 s sr GeV)^-1) produced by MCEq's cascade-equation solve,
tabulated over a full depth grid X (g/cm^2), for a fixed surface zenith
angle alpha and particle species. Every "solve" call in this module
(mceq.solve(...)) directly
invokes the external MCEq package's numerical cascade-equation
integrator; this module's own code is limited to driving those calls
(constructing/reusing an MCEqRun via init_mceq) and converting MCEq's
numpy outputs into torch tensors.

It should not contain smoothing, profile reconstruction, I/O, or plotting.

Module functions:
    solve_flux_vs_depth_grid:
        Solve MCEq's cascade equations and tabulate the
        energy-differential flux Phi(E, X, alpha) for one particle
        species over a grid of Atmosphere slant depths X.
"""



from __future__ import annotations

from typing import Optional, Union, Tuple

import torch

from tpeanuts.util.type import as_tensor
from tpeanuts.util.torch_util import as_1d_tensor, default_device

from tpeanuts.external.mceq.config import (
    MCEqModelConfig,
    GridConfig,
)

from tpeanuts.external.mceq.core import init_mceq


TensorLike = Union[float, int, torch.Tensor]


# ============================================================
# Internal helpers
# ============================================================

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

    MCEq's int_grid feature integrates the cascade equations once while
    recording the solution at every requested intermediate depth.

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


# ============================================================
# Public solvers
# ============================================================

@torch.no_grad()
def solve_flux_vs_depth_grid(
    alpha_deg: TensorLike,
    particle: str,
    X_grid_gcm2: Optional[TensorLike] = None,
    config: Optional[MCEqModelConfig] = None,
    grid_config: Optional[GridConfig] = None,
    mceq=None,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Solve MCEq's cascade equations and tabulate the energy-differential
    flux Phi(E, X, alpha) over an Atmosphere slant-depth grid.

    This is a direct driver of the external MCEq solver: it builds (or
    reuses) an MCEqRun instance via init_mceq for the given surface zenith
    angle and physics-model selection, then solves the cascade
    equations in a single pass over the full depth grid via
    _solve_mceq_at_depth_grid and MCEq's int_grid option. The result is
    the secondary-particle flux of the requested species as a function
    of energy E and slant depth X for a fixed surface zenith angle.

    Args:
        alpha_deg: Surface/MCEq zenith angle in degrees (0 <= alpha_deg < 90) of the
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
        mceq: Optional pre-initialized MCEqRun instance to reuse instead
            of constructing a new one.
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the returned tensors.
    Returns:
        Tuple (X_grid_gcm2, E_grid_GeV, flux_XE):
            X_grid_gcm2: 1-D tensor of the Atmosphere slant depths used,
                in g/cm^2.
            E_grid_GeV: 1-D tensor of the MCEq energy grid, in GeV.
            flux_XE: Tensor of shape (n_X, n_E) with the
                energy-differential flux Phi(E, X, alpha) in
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

    X_cpu = as_1d_tensor(
        X_grid_gcm2,
        name="X_grid_gcm2",
        device="cpu",
        dtype=torch.float64,
    )

    if torch.any(torch.diff(X_cpu) <= 0.0):
        raise ValueError("X_grid_gcm2 must be strictly increasing.")

    if mceq is None:
        mceq = init_mceq(
            alpha_deg=alpha_deg,
            config=config,
            info=False,
        )

    flux_XE_cpu = _solve_mceq_at_depth_grid(
        mceq=mceq,
        X_grid_gcm2=X_cpu,
        particle=particle,
    )

    E_cpu = _get_energy_grid(mceq)

    return (
        X_cpu.to(device=dev, dtype=dtype),
        E_cpu.to(device=dev, dtype=dtype),
        flux_XE_cpu.to(device=dev, dtype=dtype),
    )


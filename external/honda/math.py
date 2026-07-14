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
Honda/HKKM numerical helpers.

This module contains small Honda-specific interpolation and reconstruction
routines used by the height-differential flux generator.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tpeanuts.util.math import interp_logx, numpy_trapezoid

M2_TO_CM2_FLUX = 1.0e-4


def interpolate_flux_cm2(
    flux_table: dict[str, Any],
    *,
    honda_flavour: str,
    cosz: float,
    energy_grid: np.ndarray,
) -> np.ndarray:
    """
    Interpolate a Honda flux table onto a requested energy grid and cos(zenith).

    Interpolation is log-linear in energy (matching the Honda grid spacing)
    and linear in cos(zenith) between the 20 Honda zenith bin centers.

    Args:
        flux_table: Parsed Honda flux table, as returned by
            external.honda.tables.read_flux_table.
        honda_flavour: Honda flux column name (e.g. "NuMu", "NuEbar").
        cosz: cos(zenith) at the detector; cosz=1 is vertically downward.
        energy_grid: 1D array of neutrino energies in GeV at which to
            evaluate the flux.

    Returns:
        1D array of differential flux in (cm^2 s sr GeV)^-1 (converted from
        the Honda (m^2 s sr GeV)^-1 units via M2_TO_CM2_FLUX), matching
        energy_grid.
    """
    source_energy = flux_table["energy_GeV"]
    source_cosz = flux_table["cosz_center"]
    values = flux_table["flux_m2"][honda_flavour]

    by_cosz = np.empty((source_cosz.size, energy_grid.size), dtype=float)
    for iz in range(source_cosz.size):
        by_cosz[iz] = interp_logx(energy_grid, source_energy, values[iz])

    flux_m2 = np.empty(energy_grid.size, dtype=float)
    for ie in range(energy_grid.size):
        flux_m2[ie] = np.interp(cosz, source_cosz, by_cosz[:, ie])

    return flux_m2 * M2_TO_CM2_FLUX


def interpolate_quantiles(
    height_table: dict[str, Any],
    *,
    cosz: float,
    energy_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate Honda production-height quantiles onto an energy grid and cos(zenith).

    Each quantile (e.g. the 10%, 20%, ... height of the production-height
    cumulative distribution) is interpolated independently: log-linear in
    energy, then linear in cos(zenith) between Honda zenith bin centers.

    Args:
        height_table: Parsed Honda production-height table, as returned by
            external.honda.tables.read_height_table.
        cosz: cos(zenith) at the detector.
        energy_grid: 1D array of neutrino energies in GeV.

    Returns:
        Pair (probabilities, q) where probabilities is the 1D array of
        quantile probability levels in [0, 1] and q is a 2D array of shape
        (n_energy, n_probabilities) with the corresponding production
        heights in km.
    """
    source_energy = height_table["energy_GeV"]
    source_cosz = height_table["cosz_center"]
    probabilities = height_table["probabilities"]
    quantiles = height_table["height_quantiles_km"]

    q_energy = np.empty((source_cosz.size, energy_grid.size, probabilities.size), dtype=float)
    for iz in range(source_cosz.size):
        for ip in range(probabilities.size):
            q_energy[iz, :, ip] = interp_logx(
                energy_grid,
                source_energy,
                quantiles[iz, :, ip],
            )

    # Guarantee ascending xp for np.interp (Honda height tables are keyed by
    # zenith_bin, which yields decreasing cosz if not re-sorted at read time).
    sort_idx = np.argsort(source_cosz)
    source_cosz_asc = source_cosz[sort_idx]
    q_energy_asc = q_energy[sort_idx]

    q = np.empty((energy_grid.size, probabilities.size), dtype=float)
    for ie in range(energy_grid.size):
        for ip in range(probabilities.size):
            q[ie, ip] = np.interp(cosz, source_cosz_asc, q_energy_asc[:, ie, ip])

    return probabilities, q


def density_from_quantiles(
    probabilities: np.ndarray,
    quantile_heights_km: np.ndarray,
    h_grid_km: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct a normalized production-height density f(h|E) from quantiles.

    For each energy row, the (probability, height) quantile pairs define a
    piecewise-linear cumulative distribution function (CDF) of production
    height, anchored at h=0 (CDF=0) and at the top of the grid (CDF=1). The
    density is obtained as the numerical derivative dCDF/dh on h_grid_km,
    clipped to be non-negative and renormalized so that its integral over
    h_grid_km equals 1. If the quantile data is degenerate (zero or
    non-finite normalization), a narrow delta-like density at h=0 is used
    instead.

    Args:
        probabilities: 1D array of quantile probability levels in [0, 1].
        quantile_heights_km: 2D array of shape (n_energy, n_probabilities)
            with production-height quantiles in km, one row per energy.
        h_grid_km: 1D altitude grid in km on which to evaluate the density.

    Returns:
        2D array of shape (n_energy, h_grid_km.size) with the
        height-density f(h|E) in 1/km, normalized so that
        integral f(h|E) dh = 1 for each energy row.
    """
    f = np.zeros((quantile_heights_km.shape[0], h_grid_km.size), dtype=float)

    for ie, heights in enumerate(quantile_heights_km):
        order = np.argsort(heights)
        h_sorted = heights[order]
        p_sorted = probabilities[order]

        h_nodes = np.concatenate(([0.0], h_sorted, [max(float(h_grid_km[-1]), float(h_sorted[-1]))]))
        p_nodes = np.concatenate(([0.0], p_sorted, [1.0]))

        unique_h, unique_indices = np.unique(h_nodes, return_index=True)
        unique_p = p_nodes[unique_indices]
        cdf = np.interp(h_grid_km, unique_h, unique_p, left=0.0, right=1.0)
        density = np.gradient(cdf, h_grid_km, edge_order=1)
        density = np.clip(density, 0.0, None)
        norm = numpy_trapezoid(density, x=h_grid_km)

        if norm > 0.0 and np.isfinite(norm):
            density = density / norm
        else:
            density = np.zeros_like(h_grid_km)
            density[0] = 1.0 / max(float(h_grid_km[1] - h_grid_km[0]), 1.0)
            density = density / numpy_trapezoid(density, x=h_grid_km)

        f[ie] = density

    return f


def flat_height_density(energy_grid: np.ndarray, h_grid_km: np.ndarray) -> np.ndarray:
    """
    Build a uniform (energy-independent) production-height density.

    Used as a fallback when no Honda production-height table is available
    for a particle (e.g. no flavour-specific table and no tau-neutrino
    fallback). Approximates ignorance of the true height distribution by
    spreading the flux uniformly over the requested height grid.

    Args:
        energy_grid: 1D array of neutrino energies in GeV (only its size is
            used).
        h_grid_km: 1D altitude grid in km.

    Returns:
        2D array of shape (energy_grid.size, h_grid_km.size), uniform along
        the height axis and normalized so that integral f(h) dh = 1 for
        each energy row.
    """
    density = np.ones((energy_grid.size, h_grid_km.size), dtype=float)
    norm = numpy_trapezoid(density, x=h_grid_km, axis=1)
    return density / norm[:, None]

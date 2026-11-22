#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
#
# TPeanuts — Fast Simulation of Neutrino Oscillations in Matter.
#
# Developed as part of the Master's Thesis for the MSc in Advanced Physics,
# specialization in Theoretical Physics, at the Universitat de València.
#
# Author
# ------
# Juan Ramón Díaz Santos
# diazjuan@alumni.uv.es
#
# Supervisors
# -----------
# Roberto Ruiz de Austri Bazán
# rruiz@ific.uv.es
#
# Michele Lucente
# michele.lucente@unibo.it
#
# Universitat de València, Spain
# September 2026
# =============================================================================

"""Daya Bay detector response: IAV redistribution, LSNL nonlinearity, Gaussian resolution.

Applied in this order to the true prompt-energy spectrum, before it
reaches the analysis binning -- **IAV, then LSNL, then Gaussian
resolution**, matching the official Daya Bay analysis pipeline's own
stage order:

1. **IAV** (Inner Acrylic Vessel): a 240x240 energy-redistribution
   matrix (``DayaBayParameters.iav_matrix``, columns summing to 1),
   capturing the vertex-position-dependent light-collection
   non-uniformity near the acrylic vessel wall -- applied before the
   light-yield nonlinearity below, since energy lost into the inert
   acrylic never produces scintillation light.
2. **LSNL** (Liquid Scintillator Non-Linearity): a multiplicative
   energy-scale correction ``f(E) = E_reconstructed / E_true_deposited``
   (``DayaBayParameters.lsnl_curve_E_MeV/F``), warping the
   IAV-redistributed energy ``T`` to a nonlinear scale
   ``T_nl = T * f(T)``.
3. **Gaussian resolution**: the 3-term formula
   ``sigma(E)/E = sqrt(a^2 + b^2/E + c^2/E^2)``
   (``DayaBayParameters.eres_a/b/c`` -- spatial/temporal,
   photon-statistics and dark-noise terms respectively).

The three steps are independent, official response ingredients,
composed here as sequential linear operators -- a standard
simplification when a full joint response covariance is not reproduced
(this project uses their nominal/central values only, not the
LSNL/IAV pull-curve uncertainties).

Notes:
    - Since ``tpeanuts.detector.common.event_rate.apply_response``
      integrates R(T'|T) against T by trapezoidal rule, each step above
      is built as a discrete, mass-conserving transfer matrix on the
      shared ``T_grid_MeV`` grid, and only the final composed matrix is
      converted from probability mass to a density (dividing by the
      grid spacing).

Module contents:
    energy_resolution_MeV(...)
        sigma(T) on a given true-energy grid, from the resolution
        formula.
    response_matrix(...)
        The full R(T'|T): IAV redistribution -> LSNL warp -> Gaussian
        resolution, composed and returned as a density (see above).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from tpeanuts.detector.common.response import gaussian_response_matrix
from tpeanuts.experiments.dayabay.parameters import DayaBayParameters
from tpeanuts.util.math import interp1d_linear, searchsorted_contiguous

Tensor = torch.Tensor


def energy_resolution_MeV(
    params: DayaBayParameters,
    T_grid_MeV: Tensor,
    *,
    a: Optional[float] = None,
    b: Optional[float] = None,
    c: Optional[float] = None,
) -> Tensor:
    """Energy resolution ``sigma(T) = T * sqrt(a^2 + b^2/T + c^2/T^2)``, MeV.

    Args:
        params: Loaded Daya Bay parameters (default ``a``/``b``/``c``
            source, when the respective override is None).
        T_grid_MeV: True-observable grid, shape ``(n_T,)``.
        a, b, c: Resolution-formula coefficients (see module docstring).
            None (default) uses ``params.eres_a``/``eres_b``/``eres_c``.

    Returns:
        Tensor shaped ``(n_T,)``, sigma(T) in MeV.
    """
    a = params.eres_a if a is None else a
    b = params.eres_b if b is None else b
    c = params.eres_c if c is None else c
    T = T_grid_MeV.clamp_min(torch.finfo(T_grid_MeV.dtype).tiny)
    return T * torch.sqrt(a ** 2 + b ** 2 / T + c ** 2 / T ** 2)


def _lsnl_curves_on_grid(params: DayaBayParameters, T_grid_MeV: Tensor) -> Tuple[Tensor, Tensor]:
    """LSNL nominal + pull curves, linearly interpolated onto ``T_grid_MeV``.

    Args:
        params: Loaded Daya Bay parameters.
        T_grid_MeV: True-observable grid, shape ``(n_T,)``.

    Returns:
        ``(f0_grid, fk_grid)``: ``f0_grid`` shaped ``(n_T,)`` (nominal
        curve) and ``fk_grid`` shaped ``(4, n_T)`` (the 4 pull curves),
        both constant-extrapolated outside the curves' own energy range.
    """
    f0_grid = interp1d_linear(
        T_grid_MeV, params.lsnl_curve_E_MeV, params.lsnl_curve_F,
        left=params.lsnl_curve_F[0], right=params.lsnl_curve_F[-1],
    )
    fk_rows = [
        interp1d_linear(T_grid_MeV, pull_E, pull_f, left=pull_f[0], right=pull_f[-1])
        for pull_E, pull_f in params.lsnl_curve_pulls
    ]
    fk_grid = torch.stack(fk_rows, dim=0)
    return f0_grid, fk_grid


def _lsnl_warp_matrix(
    params: DayaBayParameters, T_grid_MeV: Tensor, lsnl_pulls: Optional[Tensor] = None,
) -> Tensor:
    """LSNL mass matrix: redistribute each grid point's mass onto its image ``T_nl = T*f(T)``.

    Column ``j`` (true energy ``T_grid_MeV[j]``) is split, by linear
    interpolation, between the two ``T_grid_MeV`` points bracketing
    ``T_nl = T_grid_MeV[j] * f(T_grid_MeV[j])``, built as an explicit
    ``(n_T, n_T)`` matrix so it can be composed with the IAV and
    Gaussian matrices below. ``f`` is the official linear LSNL
    systematic model,

        ``f(E) = f0(E) + sum_k lsnl_pulls[k] * (f_k(E) - f0(E))``,

    with ``f0``/``f_k`` the nominal/pull curves (see
    ``_lsnl_curves_on_grid``). Differentiable w.r.t. ``lsnl_pulls`` (the
    curve interpolation itself is fixed data; only the linear
    combination and the resulting warp-matrix weights carry gradient).

    Args:
        params: Loaded Daya Bay parameters.
        T_grid_MeV: True-observable grid, shape ``(n_T,)``, strictly
            increasing.
        lsnl_pulls: LSNL pull-curve nuisances, shape ``(4,)``, nominal
            (official prior mean) all-zero -- None uses the nominal
            curve unchanged.

    Returns:
        Tensor shaped ``(n_T, n_T)``; each column sums to 1.
    """
    f0_grid, fk_grid = _lsnl_curves_on_grid(params, T_grid_MeV)
    if lsnl_pulls is None:
        f_grid = f0_grid
    else:
        f_grid = f0_grid + torch.einsum("k,kt->t", lsnl_pulls, fk_grid - f0_grid)
    T_nl = T_grid_MeV * f_grid

    n = T_grid_MeV.shape[0]
    idx_hi = searchsorted_contiguous(T_grid_MeV, T_nl.detach()).clamp(min=1, max=n - 1)
    idx_lo = idx_hi - 1
    x_lo, x_hi = T_grid_MeV[idx_lo], T_grid_MeV[idx_hi]
    weight_hi = ((T_nl - x_lo) / (x_hi - x_lo)).clamp(0.0, 1.0)
    weight_lo = 1.0 - weight_hi

    cols = torch.arange(n, device=T_grid_MeV.device)
    warp = torch.zeros((n, n), dtype=T_grid_MeV.dtype, device=T_grid_MeV.device)
    warp.index_put_((idx_lo, cols), weight_lo, accumulate=True)
    warp.index_put_((idx_hi, cols), weight_hi, accumulate=True)
    return warp


def _iav_mass_matrix(params: DayaBayParameters, T_grid_MeV: Tensor) -> Tensor:
    """Embed the 240x240 IAV matrix into ``T_grid_MeV``'s ``(n_T, n_T)`` grid.

    ``T_grid_MeV`` must be the IAV matrix's own 0.05 MeV, 0-12 MeV
    binning (``params.T_grid_MeV``, 241 points = 240 IAV bins' left
    edges plus the final right edge at 12.0 MeV); the single extra grid
    point (index 240, T=12.0) is given an identity (no-redistribution)
    response, since the IBD prompt-energy spectrum is negligible that
    far above threshold.

    Args:
        params: Loaded Daya Bay parameters.
        T_grid_MeV: True-observable grid, shape ``(n_T,)``, ``n_T >= 240``.

    Returns:
        Tensor shaped ``(n_T, n_T)``.

    Raises:
        ValueError: If ``T_grid_MeV`` has fewer points than the IAV
            matrix's own binning.
    """
    n = T_grid_MeV.shape[0]
    n_iav = params.iav_matrix.shape[0]
    if n < n_iav:
        raise ValueError(f"T_grid_MeV must have at least {n_iav} points (the IAV binning), got {n}.")
    out = torch.zeros((n, n), dtype=T_grid_MeV.dtype, device=T_grid_MeV.device)
    out[:n_iav, :n_iav] = params.iav_matrix.to(dtype=T_grid_MeV.dtype, device=T_grid_MeV.device)
    if n > n_iav:
        out[n_iav:, n_iav:] = torch.eye(n - n_iav, dtype=T_grid_MeV.dtype, device=T_grid_MeV.device)
    return out


def response_matrix(
    params: DayaBayParameters,
    T_grid_MeV: Optional[Tensor] = None,
    Tprime_grid_MeV: Optional[Tensor] = None,
    *,
    a: Optional[float] = None,
    b: Optional[float] = None,
    c: Optional[float] = None,
    lsnl_pulls: Optional[Tensor] = None,
) -> Tensor:
    """Daya Bay's response matrix R(T'|T): IAV redistribution -> LSNL warp -> Gaussian resolution.

    Each step is built as a discrete, mass-conserving transfer matrix on
    ``T_grid_MeV`` and composed by matrix multiplication; only the
    final matrix is converted from probability mass to the density
    ``tpeanuts.detector.common.event_rate.apply_response`` expects
    (dividing by the grid spacing).

    Args:
        params: Loaded Daya Bay parameters.
        T_grid_MeV: True-observable grid, shape ``(n_T,)``; None (the
            default) uses ``params.T_grid_MeV`` (the IAV matrix's own
            binning).
        Tprime_grid_MeV: Reconstructed-observable grid, shape
            ``(n_Tp,)``; None (the default) uses ``params.T_grid_MeV``.
        a, b, c: Gaussian resolution-formula coefficients (see
            ``energy_resolution_MeV``).
        lsnl_pulls: LSNL pull-curve nuisances, shape ``(4,)``; None
            (default) uses the nominal LSNL curve unchanged.

    Returns:
        Tensor shaped ``(n_Tp, n_T)``, a probability density in T'
        (trapezoidal-integrating a column over ``Tprime_grid_MeV`` gives
        approximately 1).
    """
    if T_grid_MeV is None:
        T_grid_MeV = params.T_grid_MeV
    if Tprime_grid_MeV is None:
        Tprime_grid_MeV = params.T_grid_MeV

    dT = T_grid_MeV[1] - T_grid_MeV[0]
    gaussian_mass = gaussian_response_matrix(
        T_grid_MeV, Tprime_grid_MeV, energy_resolution_MeV(params, T_grid_MeV, a=a, b=b, c=c),
    ) * dT

    lsnl_mass = _lsnl_warp_matrix(params, T_grid_MeV, lsnl_pulls)
    iav_mass = _iav_mass_matrix(params, T_grid_MeV)

    total_mass = gaussian_mass @ lsnl_mass @ iav_mass
    return total_mass / dT

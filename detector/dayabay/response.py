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
Real Daya Bay detector response: IAV redistribution, LSNL nonlinearity, Gaussian resolution.

The full real chain, applied in this order to the true prompt-energy
spectrum before it reaches the analysis binning -- **IAV, then LSNL, then
Gaussian resolution**, matching the official Daya Bay Collaboration
analysis pipeline's own stage order (its dagflow/GNA model literally names
its cumulative-correction nodes ``stages.iav`` -> ``stages.evis`` (IAV +
LSNL applied) -> ``stages.erec`` (+ resolution); confirmed from the
Collaboration's own CHEP 2026 presentation on that framework -- an earlier
version of this module had IAV and LSNL swapped, since corrected):

1. **IAV** (Inner Acrylic Vessel): a real 240x240 energy-redistribution
   matrix (``detector.dayabay.parameters.IAV_MATRIX``, columns summing to
   1), capturing the vertex-position-dependent light-collection non-
   uniformity near the acrylic vessel wall -- applied to the true deposited
   energy, before the light-yield nonlinearity below (energy lost into the
   inert acrylic never produces scintillation light, so it cannot be
   subject to a light-yield nonlinearity curve).
2. **LSNL** (Liquid Scintillator Non-Linearity): a real multiplicative
   energy-scale correction ``f(E) = E_reconstructed / E_true_deposited``
   (``detector.dayabay.parameters.LSNL_CURVE_E_MEV/F``, the official
   nominal curve), warping the IAV-redistributed energy ``T`` to a
   nonlinear scale ``T_nl = T * f(T)``.
3. **Gaussian resolution**: the real 3-term formula
   sigma(E)/E = sqrt(a^2 + b^2/E + c^2/E^2)
   (``detector.dayabay.parameters.ERES_A/B/C`` -- "spatial/temporal",
   "photon statistics", and "dark noise" terms respectively), the same
   photostatistics smearing used before this module was extended.

All three steps are real, independent, official detector-response
ingredients (the official data release ships them as separate files
precisely because they are physically distinct effects), composed here as
sequential, independent linear operators -- a standard simplifying
assumption when a full joint response covariance is not being reproduced
(this project does not reproduce the LSNL/IAV pull-curve systematic
uncertainties either, only their real nominal/central values -- see the
package module docstring).

Because ``detector.common.event_rate.apply_response`` integrates R(T'|T)
against T via ``torch.trapezoid``, every step above is built as a discrete,
mass-conserving "transfer" matrix on the shared ``T_GRID_MEV`` grid (0.05
MeV spacing, matching the IAV matrix's own real binning) and only the final
*composed* matrix is converted from probability mass to a density (dividing
by the grid spacing) -- see ``response_matrix``.

Module contents:
    sigma_MeV(...)
        sigma(T) on a given true-energy grid, from the real formula.
    response_matrix(...)
        The full real R(T'|T): IAV redistribution -> LSNL warp -> Gaussian
        resolution, composed and returned as a density (see above).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from tpeanuts.detector.common.response import gaussian_response_matrix
from tpeanuts.detector.dayabay.parameters import (
    ERES_A,
    ERES_B,
    ERES_C,
    IAV_MATRIX,
    LSNL_CURVE_E_MEV,
    LSNL_CURVE_F,
    LSNL_CURVE_PULLS,
    T_GRID_MEV,
    TPRIME_GRID_MEV,
)


@torch.no_grad()
def sigma_MeV(
    T_grid_MeV: torch.Tensor,
    *, a: float = ERES_A, b: float = ERES_B, c: float = ERES_C,
) -> torch.Tensor:
    """Energy resolution sigma(T) = T * sqrt(a^2 + b^2/T + c^2/T^2), MeV.

    Args:
        T_grid_MeV: True-observable grid, shape ``(n_T,)``.
        a, b, c: Resolution-formula coefficients (see module docstring).

    Returns:
        Real tensor shaped ``(n_T,)``, sigma(T) in MeV.
    """
    T = T_grid_MeV.clamp_min(torch.finfo(T_grid_MeV.dtype).tiny)
    return T * torch.sqrt(a ** 2 + b ** 2 / T + c ** 2 / T ** 2)


@torch.no_grad()
def _lsnl_curves_on_grid(T_grid_MeV: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Real LSNL nominal + pull curves, linearly interpolated onto ``T_grid_MeV`` (fixed real data).

    Args:
        T_grid_MeV: True-observable grid, shape ``(n_T,)``.

    Returns:
        ``(f0_grid, fk_grid)``: ``f0_grid`` shaped ``(n_T,)`` (nominal
        curve) and ``fk_grid`` shaped ``(4, n_T)`` (the 4 real pull
        curves), both constant-extrapolated outside the curves' own real
        energy range (``detector.dayabay.parameters.LSNL_CURVE_E_MEV/F``,
        ``LSNL_CURVE_PULLS``).
    """
    T_np = T_grid_MeV.cpu().numpy()
    curve_E_np = LSNL_CURVE_E_MEV.cpu().numpy()
    curve_f0_np = LSNL_CURVE_F.cpu().numpy()
    f0_np = np.interp(T_np, curve_E_np, curve_f0_np, left=curve_f0_np[0], right=curve_f0_np[-1])
    f0_grid = torch.as_tensor(f0_np, dtype=T_grid_MeV.dtype, device=T_grid_MeV.device)

    fk_rows = []
    for pull_E, pull_f in LSNL_CURVE_PULLS:
        pull_E_np, pull_f_np = pull_E.cpu().numpy(), pull_f.cpu().numpy()
        fk_np = np.interp(T_np, pull_E_np, pull_f_np, left=pull_f_np[0], right=pull_f_np[-1])
        fk_rows.append(torch.as_tensor(fk_np, dtype=T_grid_MeV.dtype, device=T_grid_MeV.device))
    fk_grid = torch.stack(fk_rows, dim=0)

    return f0_grid, fk_grid


def _lsnl_warp_matrix(
    T_grid_MeV: torch.Tensor,
    lsnl_pulls: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Real LSNL mass matrix: redistribute each grid point's mass onto its image T_nl = T*f(T).

    Column ``j`` (true energy ``T_grid_MeV[j]``) is split, by linear
    interpolation, between the two ``T_grid_MeV`` points bracketing
    ``T_nl = T_grid_MeV[j] * f(T_grid_MeV[j])`` -- the same mass-conserving
    "scatter" as ``detector.common.response.scatter_add_linear``, built here
    as an explicit ``(n_T, n_T)`` matrix instead (rather than applied to a
    single spectrum) so it can be composed with the IAV and Gaussian
    matrices below. ``f`` is the official linear LSNL systematic model,

        f(E) = f0(E) + sum_k lsnl_pulls[k] * (f_k(E) - f0(E)),

    with ``f0``/``f_k`` the real nominal/pull curves
    (``parameters/detector_lsnl.yaml``; see ``_lsnl_curves_on_grid``).
    Differentiable w.r.t. ``lsnl_pulls`` (the curve interpolation itself is
    fixed real data, evaluated once with ``@torch.no_grad()``; only the
    linear combination and the resulting warp-matrix weights carry
    gradient).

    Args:
        T_grid_MeV: True-observable grid, shape ``(n_T,)``, strictly
            increasing.
        lsnl_pulls: Real LSNL pull-curve nuisances, shape ``(4,)``, nominal
            (official prior mean) all-zero -- None uses the nominal curve
            unchanged (equivalent to an all-zero tensor).

    Returns:
        Real tensor shaped ``(n_T, n_T)``; each column sums to 1.
    """
    f0_grid, fk_grid = _lsnl_curves_on_grid(T_grid_MeV)
    if lsnl_pulls is None:
        f_grid = f0_grid
    else:
        f_grid = f0_grid + torch.einsum("k,kt->t", lsnl_pulls, fk_grid - f0_grid)
    T_nl = T_grid_MeV * f_grid

    n = T_grid_MeV.shape[0]
    idx_hi = torch.searchsorted(T_grid_MeV, T_nl.detach()).clamp(min=1, max=n - 1)
    idx_lo = idx_hi - 1
    x_lo, x_hi = T_grid_MeV[idx_lo], T_grid_MeV[idx_hi]
    weight_hi = ((T_nl - x_lo) / (x_hi - x_lo)).clamp(0.0, 1.0)
    weight_lo = 1.0 - weight_hi

    cols = torch.arange(n, device=T_grid_MeV.device)
    warp = torch.zeros((n, n), dtype=T_grid_MeV.dtype, device=T_grid_MeV.device)
    warp.index_put_((idx_lo, cols), weight_lo, accumulate=True)
    warp.index_put_((idx_hi, cols), weight_hi, accumulate=True)
    return warp


@torch.no_grad()
def _iav_mass_matrix(T_grid_MeV: torch.Tensor) -> torch.Tensor:
    """Embed the real 240x240 IAV matrix into ``T_grid_MeV``'s ``(n_T, n_T)`` grid.

    ``T_grid_MeV`` must be the real IAV matrix's own 0.05 MeV, 0-12 MeV
    binning (``detector.dayabay.parameters.T_GRID_MEV``, 241 points = 240
    real IAV bins' left edges plus the final right edge at 12.0 MeV); the
    single extra grid point (index 240, T=12.0) is given an identity
    (no-redistribution) response, since the real IBD prompt-energy spectrum
    is negligible that far above threshold (see module docstring).

    Args:
        T_grid_MeV: True-observable grid, shape ``(n_T,)``, ``n_T >= 240``.

    Returns:
        Real tensor shaped ``(n_T, n_T)``.
    """
    n = T_grid_MeV.shape[0]
    n_iav = IAV_MATRIX.shape[0]
    if n < n_iav:
        raise ValueError(f"T_grid_MeV must have at least {n_iav} points (the real IAV binning), got {n}.")
    out = torch.zeros((n, n), dtype=T_grid_MeV.dtype, device=T_grid_MeV.device)
    out[:n_iav, :n_iav] = IAV_MATRIX.to(dtype=T_grid_MeV.dtype, device=T_grid_MeV.device)
    if n > n_iav:
        out[n_iav:, n_iav:] = torch.eye(n - n_iav, dtype=T_grid_MeV.dtype, device=T_grid_MeV.device)
    return out


def response_matrix(
    T_grid_MeV: torch.Tensor = T_GRID_MEV,
    Tprime_grid_MeV: torch.Tensor = TPRIME_GRID_MEV,
    *,
    a: float = ERES_A, b: float = ERES_B, c: float = ERES_C,
    lsnl_pulls: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Daya Bay's real response matrix R(T'|T): IAV redistribution -> LSNL warp -> Gaussian resolution.

    Each step is built as a discrete, mass-conserving transfer matrix on
    ``T_grid_MeV`` (see module docstring) and composed by matrix
    multiplication; only the final, composed matrix is converted from
    probability mass to the density ``detector.common.event_rate
    .apply_response`` expects (dividing by the grid spacing).

    Args:
        T_grid_MeV: True-observable grid, shape ``(n_T,)``; must equal
            ``detector.dayabay.parameters.T_GRID_MEV`` (the real IAV
            matrix's own binning -- see ``_iav_mass_matrix``).
        Tprime_grid_MeV: Reconstructed-observable grid, shape ``(n_Tp,)``.
        a, b, c: Gaussian resolution-formula coefficients (see ``sigma_MeV``).
        lsnl_pulls: Real LSNL pull-curve nuisances, shape ``(4,)``; None
            (default) uses the real nominal LSNL curve unchanged -- see
            ``_lsnl_warp_matrix``.

    Returns:
        Real tensor shaped ``(n_Tp, n_T)``, a probability density in T'
        (trapezoidal-integrating a column over ``Tprime_grid_MeV`` gives
        approximately 1).
    """
    dT = T_grid_MeV[1] - T_grid_MeV[0]
    gaussian_mass = gaussian_response_matrix(
        T_grid_MeV, Tprime_grid_MeV, sigma_MeV(T_grid_MeV, a=a, b=b, c=c),
    ) * dT

    lsnl_mass = _lsnl_warp_matrix(T_grid_MeV, lsnl_pulls)
    iav_mass = _iav_mass_matrix(T_grid_MeV)

    total_mass = gaussian_mass @ lsnl_mass @ iav_mass
    return total_mass / dT

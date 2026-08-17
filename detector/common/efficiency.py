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
Detection/selection efficiency application, in reconstructed-energy space.

Efficiency (fiducial-volume cuts, trigger, analysis selection) is applied
*after* ``tpeanuts.detector.common.response.gaussian_response_matrix``
smearing, not before: analysis cuts act on the reconstructed observable T',
not the true one T.

Module contents:
    apply_efficiency(...)
        Multiply a reconstructed-energy spectrum by an efficiency curve.
    step_efficiency(...)
        A hard energy-threshold efficiency curve (eps=0 below threshold,
        1 above), the common baseline before any finer-grained cut is added.
"""

from __future__ import annotations

import torch


def apply_efficiency(
    reco_spectrum: torch.Tensor,
    efficiency: torch.Tensor,
) -> torch.Tensor:
    """Multiply a reconstructed-energy spectrum by a detection efficiency curve.

    Args:
        reco_spectrum: dR/dT' after response smearing, shape ``(..., n_Tp)``.
        efficiency: Efficiency eps(T') in [0, 1], shape ``(n_Tp,)``.

    Returns:
        ``reco_spectrum * efficiency``, shape ``(..., n_Tp)``.

    Raises:
        ValueError: If ``efficiency`` is not shaped to broadcast against
            ``reco_spectrum``'s final axis, or has an entry outside [0, 1].
    """
    if efficiency.shape[-1] != reco_spectrum.shape[-1]:
        raise ValueError(
            f"efficiency's final dimension ({efficiency.shape[-1]}) must match "
            f"reco_spectrum's ({reco_spectrum.shape[-1]})."
        )
    if torch.any(efficiency < 0) or torch.any(efficiency > 1):
        raise ValueError("efficiency entries must lie in [0, 1].")
    return reco_spectrum * efficiency


@torch.no_grad()
def step_efficiency(
    Tprime_grid_MeV: torch.Tensor,
    threshold_MeV: float,
) -> torch.Tensor:
    """Hard energy-threshold efficiency: 0 below ``threshold_MeV``, 1 above.

    A grid-only (not fit-parameter-dependent) construction, so a hard step
    is fine here -- unlike a fit parameter's likelihood, this never needs to
    be differentiated through.

    Args:
        Tprime_grid_MeV: Reconstructed-observable grid, shape ``(n_Tp,)``.
        threshold_MeV: Analysis energy threshold.

    Returns:
        Real tensor shaped ``(n_Tp,)``, values in {0, 1}.
    """
    return (Tprime_grid_MeV >= threshold_MeV).to(dtype=Tprime_grid_MeV.dtype)

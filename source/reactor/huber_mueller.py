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
Generic per-isotope Huber-Mueller reactor antineutrino spectrum.

Real per-isotope spectra (Huber, Phys. Rev. C84, 024617 (2011) for U235/
Pu239/Pu241; Mueller et al., Phys. Rev. C83, 054615 (2011) for U238),
interpolated and combined into a fission-fraction-weighted spectrum shape --
this is universal reactor-physics, independent of which experiment loaded
the tabulated curves or what its own fission fractions/thermal power are
(those stay in that experiment's own ``detector.<name>`` package, e.g.
``detector.dayabay.flux``, which calls the functions here).

Module contents:
    ISOTOPES
        The 4 fissile isotopes every standard reactor antineutrino
        spectrum decomposes into.
    huber_mueller_spectrum(...)
        Interpolate one already-loaded per-isotope tabulated curve onto a
        requested energy grid.
    weighted_spectrum_shape(...)
        Fission-fraction-weighted sum over ``ISOTOPES``.
"""

from __future__ import annotations

import numpy as np
import torch

ISOTOPES: tuple[str, ...] = ("U235", "U238", "Pu239", "Pu241")


def huber_mueller_spectrum(
    E_tab_MeV: torch.Tensor,
    N_tab_per_fission_per_MeV: torch.Tensor,
    E_grid_MeV: torch.Tensor,
) -> torch.Tensor:
    """Interpolate one isotope's tabulated spectrum onto ``E_grid_MeV``.

    Args:
        E_tab_MeV: Tabulated energy grid for this isotope, shape ``(n_tab,)``.
        N_tab_per_fission_per_MeV: Tabulated spectrum values at
            ``E_tab_MeV``, antineutrinos/fission/MeV, same shape.
        E_grid_MeV: Requested true antineutrino energy grid, shape ``(n_E,)``.

    Returns:
        Real tensor shaped ``(n_E,)``, antineutrinos/fission/MeV, 0 outside
        the tabulated range.
    """
    E_tab_np = E_tab_MeV.detach().cpu().numpy()
    N_tab_np = N_tab_per_fission_per_MeV.detach().cpu().numpy()
    E_np = E_grid_MeV.detach().cpu().numpy()
    N = np.interp(E_np, E_tab_np, N_tab_np, left=0.0, right=0.0)
    return torch.as_tensor(N, dtype=E_grid_MeV.dtype, device=E_grid_MeV.device)


def weighted_spectrum_shape(
    curves: dict[str, tuple[torch.Tensor, torch.Tensor]],
    fission_fractions: dict[str, float],
    E_grid_MeV: torch.Tensor,
) -> torch.Tensor:
    """Fission-fraction-weighted sum over ``ISOTOPES``, antineutrinos/fission/MeV.

    Args:
        curves: ``{isotope: (E_tab_MeV, N_tab_per_fission_per_MeV)}`` for
            every entry of ``ISOTOPES``.
        fission_fractions: ``{isotope: fraction}`` for every entry of
            ``ISOTOPES`` (experiment-specific, e.g. real time-averaged
            values from a data release).
        E_grid_MeV: True antineutrino energy grid, shape ``(n_E,)``.

    Returns:
        Real tensor shaped ``(n_E,)``, antineutrinos/fission/MeV.
    """
    total = torch.zeros_like(E_grid_MeV)
    for isotope in ISOTOPES:
        E_tab, N_tab = curves[isotope]
        total = total + fission_fractions[isotope] * huber_mueller_spectrum(E_tab, N_tab, E_grid_MeV)
    return total

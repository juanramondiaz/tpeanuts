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

"""PREM tabulated density-profile model for the perturbative evolutor.

This package implements a piecewise-linear-in-r² electron density model based
on the Preliminary Reference Earth Model (PREM; Dziewonski & Anderson 1981) as
tabulated at 500 radial points by IRIS SPUD (dataset ID 9785674).

Within each PREM shell the electron density satisfies

    n_e(r²) = A + B · r²,

which under the trajectory coordinate shift r² = x² + sin²η becomes

    n_e(x²) = (A + B · sin²η) + B · x².

This linear-in-x² form admits an exact analytical oscillatory residual
integral, enabling the perturbative evolutor to compute the first-order matter
correction efficiently.

Package exports:
    PremTabulatedProfile
        Layered PREM profile with piecewise-linear-in-r² electron density,
        with optional neutron-density coefficients for the 3+1 sterile
        extension's neutral-current matter term.
    PremProfileSegment
        Perturbative-evolutor-compatible segment for a single PREM shell.
    load_prem_profile
        I/O helper: read canonical PREM CSV and build (rj, coefficients) for
        electron density.
    load_prem_neutron_profile
        I/O helper: read canonical PREM CSV and build (rj, coefficients) for
        neutron density.
"""

from tpeanuts.core.perturbative.models.prem.profile_layered import PremTabulatedProfile
from tpeanuts.core.perturbative.models.prem.profile_segment import PremProfileSegment
from tpeanuts.core.perturbative.models.prem.io import (
    load_prem_profile,
    load_prem_neutron_profile,
)

__all__ = [
    "PremTabulatedProfile",
    "PremProfileSegment",
    "load_prem_profile",
    "load_prem_neutron_profile",
]

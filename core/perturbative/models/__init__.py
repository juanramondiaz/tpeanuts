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

"""Perturbative density-profile models.

This package exposes mathematical density-profile models used by the
perturbative core evolutor. Medium-specific geometry remains outside the core:
medium modules build these models from their own density and trajectory data.

Package exports:
    PerturbativeOuterSegment
        Model-neutral metadata for the outermost crossed segment.
    PerturbativeSegmentBatch
        Model-neutral ordered trajectory segment batch.
    perturbative_profile_selection
        Instantiate and return a layered profile from a public model name and kwargs.
    EvenPowerProfileLayered
        Layered profile storing even-power coefficients per shell.
    EvenPowerProfileSegment
        Profile model for ``n_e(x) = a + b*x**2 + c*x**4 + ...``.
    PremTabulatedProfile
        Layered PREM profile with piecewise-linear-in-r² electron density.
    PremProfileSegment
        Perturbative-evolutor-compatible segment for a single PREM shell.
"""

from tpeanuts.core.perturbative.models.interface import (
    PerturbativeOuterSegment,
    PerturbativeSegmentBatch,
)
from tpeanuts.core.perturbative.models.model_selection import (
    perturbative_profile_selection,
)
from tpeanuts.core.perturbative.models.even_power import (
    EvenPowerProfileLayered,
    EvenPowerProfileSegment,
)
from tpeanuts.core.perturbative.models.prem import (
    PremTabulatedProfile,
    PremProfileSegment,
)
from tpeanuts.core.perturbative.models.atmosphere import (
    AtmospherePolynomialProfile,
    AtmospherePolynomialSegment,
)

__all__ = [
    "PerturbativeOuterSegment",
    "PerturbativeSegmentBatch",
    "perturbative_profile_selection",
    "EvenPowerProfileLayered",
    "EvenPowerProfileSegment",
    "PremTabulatedProfile",
    "PremProfileSegment",
    "AtmospherePolynomialProfile",
    "AtmospherePolynomialSegment",
]

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
vacuum oscillation utilities for peanuts-torch.

This package implements coherent three-flavour neutrino oscillation with no
matter effect: mass eigenstates acquire only kinetic (vacuum) phases
proportional to Delta m^2_ij L / E, with no MSW potential term. It is used
as a standalone propagation mode, e.g. for baseline atmospheric/long-
baseline oscillation.

Submodules:
    medium.vacuum.evolutor
        Builds the vacuum flavour-basis evolution operator S(L, E) and
        applies it to an initial flavour state.
    medium.vacuum.probability
        Converts vacuum evolution operators into flavour-transition and
        final flavour probabilities, for both coherent (flavour-basis) and
        incoherent (mass-basis) initial states.
    medium.vacuum.flux
        Combines ``probability.vacuum_probability_state`` with flux
        normalization, optional spectral weighting, and energy integration.
    medium.vacuum.validation
        Helpers comparing this package's output against the legacy
        NumPy-based peanuts vacuum implementation.
"""



from tpeanuts.medium.vacuum.evolutor import (
    vacuum_evolved_state,
    vacuum_evolutor,
)
from tpeanuts.medium.vacuum.probability import (
    vacuum_probability_integrated,
    vacuum_probability_state,
    vacuum_probability_transition,
)
from tpeanuts.medium.vacuum.flux import vacuum_flux_integrated, vacuum_flux_state
from tpeanuts.medium.vacuum.validation import (
    compare_vacuum_probability_state_with_legacy,
    compare_vacuum_evolved_state_with_legacy,
)

__all__ = [
    "vacuum_probability_state",
    "vacuum_probability_integrated",
    "vacuum_flux_state",
    "vacuum_flux_integrated",
    "vacuum_evolved_state",
    "vacuum_evolutor",
    "vacuum_probability_transition",
    "compare_vacuum_probability_state_with_legacy",
    "compare_vacuum_evolved_state_with_legacy",
]

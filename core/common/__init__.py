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
Common neutrino-oscillation building blocks.

Package modules:
    pmns
        Shared PMNS infrastructure: PMNSParams (Standard Model angles) and
        the abstract PMNS base class. Concrete mixing-matrix implementations
        live in tpeanuts.core.SM (PMNS_SM) and tpeanuts.core.BSM
        (PMNS_sterile).
    oscillation
        OscillationParameters, the bundled pmns/mass-splitting/antinu state
        threaded through the rest of the project.
    potential
        Kinetic and matter-potential construction.
    hamiltonian
        Reduced and flavour-basis Hamiltonian construction.
    evolutor
        Generic evolution-operator application utilities.
    neutrino
        Neutrino flavour definitions and indexing utilities.
    probabilities
        Transition-probability and flavour-flux utilities.
    flux
        Generic conversion from probabilities to flavour-resolved fluxes.
"""

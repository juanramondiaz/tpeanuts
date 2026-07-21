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
Tests for tpeanuts.core.common.

Package modules:
    test1_potential
        Tests for matter/kinetic potential utilities and the dimension-
        agnostic mass-squared/kinetic mass vector builders.
    test2_pmns
        Tests for the PMNS mixing-matrix module.
    test3_hamiltonian
        Tests for the kinetic/matter/reduced/flavour Hamiltonian builders,
        covering the 3-flavour Standard Model, NSI, the 3+1 sterile
        extension, and combinations thereof -- all handled by the same
        functions, dispatching only on the ``oscillation`` object passed in.
"""

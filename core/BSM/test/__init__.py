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
Tests for tpeanuts.core.BSM.

Generic Hamiltonian-builder tests (kinetic/matter/reduced/flavour) now live
in ``core/common/test/test3_hamiltonian.py``, since that machinery is no
longer BSM-specific.

Package modules:
    test2_bsm_nsi
        Tests specific to the NSIConfig parameter container and its
        integration with the common Hamiltonian builders.
    test3_bsm_sterile
        Tests specific to the PMNS_sterile 3+1 mixing matrix and its
        integration with the common Hamiltonian builders and evolutor.
"""

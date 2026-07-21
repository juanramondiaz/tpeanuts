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
Beyond Standard Model (BSM) extensions for neutrino oscillations.

Submodules
----------
PMNS_sterile
    4-flavor PMNS matrix for the 3+1 sterile neutrino scenario. Named 3+1
    presets are built via
    ``tpeanuts.core.common.oscillation.oscillation_parameters_from_preset``;
    the preset data lives in ``tpeanuts.config.presets.OSCILLATION_PRESETS``.
NSIConfig
    Frozen dataclass ``NSIConfig`` for Non-Standard Interaction parameter
    sets. The preset data lives in
    ``tpeanuts.config.presets.NSI_PRESETS``.

Hamiltonian assembly (kinetic/matter/reduced/flavour builders) is not
scenario-specific and lives in ``tpeanuts.core.common.hamiltonian`` instead.
"""

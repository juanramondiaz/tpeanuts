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
Neutrino interaction cross sections, organized by physical process.

Organized by process rather than by detector, so a channel shared by two
experiments (e.g. neutrino-electron elastic scattering, used by both
Borexino and SNO's ES channel) is implemented once.

Package modules:
    neutrino_electron
        Neutrino-electron elastic scattering, Standard Model tree level.
"""

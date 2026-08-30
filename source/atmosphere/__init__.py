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
Atmospheric-neutrino source flux datasets: provider-neutral I/O.

Package contents:
    io
        Readers/writers for Atmosphere height-flux datasets (MCEq/Honda),
        and the canonical provider-neutral flux-table loader. Independent
        of atmosphere propagation physics (``medium.atmosphere``).
"""

from tpeanuts.source.atmosphere.io import (
    AtmosphericFluxTable,
    OutputConfig,
    load_atmospheric_flux,
    load_directory,
)

__all__ = [
    "AtmosphericFluxTable",
    "OutputConfig",
    "load_atmospheric_flux",
    "load_directory",
]

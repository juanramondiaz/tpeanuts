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
"""



from tpeanuts.vacuum.probabilities import (
    Pvacuum,
    pvacuum,
    vacuum_evolved_state,
    vacuum_evolutor,
    vacuum_probability_matrix,
)

__all__ = [
    "Pvacuum",
    "pvacuum",
    "vacuum_evolved_state",
    "vacuum_evolutor",
    "vacuum_probability_matrix",
]

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
Generic reactor antineutrino source physics, independent of any one experiment.

Package contents:
    huber_mueller
        Per-isotope Huber-Mueller antineutrino spectrum interpolation and
        fission-fraction-weighted combination -- real physics, reusable by
        any reactor experiment. Which isotopes, fission fractions, thermal
        power, and baselines apply is experiment-specific and stays in that
        experiment's own ``detector.<name>`` package (e.g.
        ``detector.dayabay.flux``).
"""

from tpeanuts.source.reactor.huber_mueller import (
    ISOTOPES,
    huber_mueller_spectrum,
    weighted_spectrum_shape,
)

__all__ = [
    "ISOTOPES",
    "huber_mueller_spectrum",
    "weighted_spectrum_shape",
]

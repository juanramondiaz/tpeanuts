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

"""Numerical neutrino-evolution algorithms.

Package modules:
    geometry
        Common trajectory structure and segment-sampling utilities.
    evolutor
        Medium-independent pointwise numerical integration.
"""

from tpeanuts.core.numerical.geometry import Trajectory, segment_sample_points
from tpeanuts.core.numerical.evolutor import (
    evolutor_numerical,
    evolutor_numerical_segment,
)

__all__ = [
    "Trajectory",
    "segment_sample_points",
    "evolutor_numerical",
    "evolutor_numerical_segment",
]

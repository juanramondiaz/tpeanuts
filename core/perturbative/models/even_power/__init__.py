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

"""Even-power perturbative density-profile model package."""

from tpeanuts.core.perturbative.models.even_power.profile_layered import (
    EvenPowerProfileLayered,
)
from tpeanuts.core.perturbative.models.even_power.profile_segment import (
    EvenPowerProfileSegment,
)
from tpeanuts.core.perturbative.models.even_power.io import (
    load_earth_density_from_csv,
    parse_density_table,
)

__all__ = [
    "EvenPowerProfileLayered",
    "EvenPowerProfileSegment",
    "load_earth_density_from_csv",
    "parse_density_table",
]

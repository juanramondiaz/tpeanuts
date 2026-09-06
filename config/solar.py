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

"""Composed solar source and medium construction settings."""

from dataclasses import dataclass, field

from tpeanuts.medium.solar.profile import SolarMediumParameters
from tpeanuts.source.solar import SolarSourceParameters


@dataclass(frozen=True)
class SolarParameters:
    """Keep solar-medium and solar-source configuration separate but grouped."""

    medium: SolarMediumParameters = field(default_factory=SolarMediumParameters)
    source: SolarSourceParameters = field(default_factory=SolarSourceParameters)

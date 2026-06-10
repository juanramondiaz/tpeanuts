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
Torch-native solar neutrino utilities.
"""



from tpeanuts.io.io_solar import (
    default_solar_data_dir,
    load_b16_fluxes,
    load_b16_solar_model,
    load_spectrum_csv,
)
from tpeanuts.solar.profiles import SolarProfile, load_default_solar_profile
from tpeanuts.solar.matter_mixing import (
    Vk,
    DeltamSqee,
    th13_M,
    th12_M,
)
from tpeanuts.solar.probabilities import (
    Tei,
    solar_flux_mass,
    psolar,
)

__all__ = [
    "default_solar_data_dir",
    "load_b16_fluxes",
    "load_b16_solar_model",
    "load_spectrum_csv",
    "SolarProfile",
    "load_default_solar_profile",
    "Vk",
    "DeltamSqee",
    "th13_M",
    "th12_M",
    "Tei",
    "solar_flux_mass",
    "psolar",
]

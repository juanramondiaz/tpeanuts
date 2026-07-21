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

This package implements the incoherent (adiabatic, MSW-resonance) treatment
of solar neutrino propagation: electron neutrinos are assumed to be produced
in the solar interior, project onto matter-modified mass eigenstates that
evolve adiabatically as the electron density drops from the production
point to the solar surface, and then propagate to Earth as an incoherent
mixture of vacuum mass eigenstates.

Submodules:
    medium.solar.io
        CSV loaders for the tabulated B16 solar model, source fluxes,
        production spectra, and Sun-Earth distance table.
    medium.solar.profile
        SolarProfile container and interpolation helpers built on top of
        ``medium.solar.io``. The ``use_LZ`` flag on SolarProfile enables
        Landau-Zener corrections throughout the adiabatic pipeline.
    medium.solar.matter_mixing
        Matter-modified mixing angles theta12^M, theta13^M (MSW resonance)
        and the dimensionless matter-potential ratio V_k.
    medium.solar.landau_zener
        Landau-Zener transition probability P_LZ(E) and supporting helpers
        (density gradient, resonance radius).
    medium.solar.probability
        Adiabatic mass-basis production weights and final flavour
        probabilities built from ``matter_mixing``, with optional LZ
        corrections dispatched via ``SolarProfile.use_LZ``. This medium has
        no transition function (no coherent evolutor exists in the adiabatic
        solar model).
    medium.solar.flux
        Combines ``probability.solar_probability_state`` with total source
        fluxes and optional spectra to produce flavour-resolved solar
        fluxes, and integrates them over energy.
    medium.solar.validation
        Helpers comparing this package's output against the legacy peanuts
        implementation.
"""



from tpeanuts.medium.solar.io import (
    default_solar_data_dir,
    load_b16_fluxes,
    load_b16_solar_model,
    load_spectrum_csv,
)
from tpeanuts.medium.solar.profile import SolarProfile
from tpeanuts.medium.solar.matter_mixing import (
    Vk,
    DeltamSqee,
    th13_M,
    th12_M,
)
from tpeanuts.medium.solar.landau_zener import (
    density_gradient,
    resonance_radius,
    plz,
)
from tpeanuts.medium.solar.probability import (
    Tei,
    solar_probability_mass,
    solar_probability_state,
    solar_probability_integrated,
)
from tpeanuts.medium.solar.flux import solar_flux_state, solar_flux_integrated

__all__ = [
    "default_solar_data_dir",
    "load_b16_fluxes",
    "load_b16_solar_model",
    "load_spectrum_csv",
    "SolarProfile",
    "Vk",
    "DeltamSqee",
    "th13_M",
    "th12_M",
    "density_gradient",
    "resonance_radius",
    "plz",
    "Tei",
    "solar_probability_mass",
    "solar_probability_state",
    "solar_probability_integrated",
    "solar_flux_state",
    "solar_flux_integrated",
]

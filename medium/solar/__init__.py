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
Torch-native solar neutrino propagation utilities.

This package implements the incoherent (adiabatic, MSW-resonance) treatment
of solar neutrino propagation: electron neutrinos are produced in the solar
interior (``source.solar.SolarNeutrinoSource``, production physics), project
onto matter-modified mass eigenstates that evolve adiabatically as the
electron density drops from the production point to the solar surface
(this package, propagation physics), and then propagate to Earth as an
incoherent mixture of vacuum mass eigenstates. Every probability/flux
function here takes a ``medium`` (this package's ``SolarMediumProfile``) and
a ``source`` (``source.solar.SolarNeutrinoSource``) as two separate
arguments; this package depends on ``source.solar``, never the reverse.

Submodules:
    medium.solar.io
        CSV loaders for the configured solar density/composition tables and
        the Sun-Earth distance table.
    medium.solar.profile
        SolarMediumProfile container and interpolation helpers built on top
        of ``medium.solar.io``. ``config.solar.SolarParameters`` composes
        ``SolarMediumParameters`` with ``source.solar.SolarSourceParameters``
        for pipeline-level configuration.
    medium.solar.matter_mixing
        Matter-modified mixing angles theta12^M, theta13^M (MSW resonance)
        and the dimensionless matter-potential ratio V_k.
    medium.solar.landau_zener
        Landau-Zener transition probability P_LZ(E) and supporting helpers
        (density gradient, resonance radius, spatial correction).
    medium.solar.adiabatic
        Pointwise adiabatic mass-eigenstate production weights:
        ``mass_weights_adiabatic_approximated`` (closed-form, plain SM only)
        and ``mass_weights_adiabatic_exact`` (pointwise diagonalisation,
        SM/NSI/sterile).
    medium.solar.probability
        Single entry point (``solar_probability_mass``) dispatching between
        ``method="numerical"`` (``medium.solar.evolutor``),
        ``method="adiabatic_approximated"``, and
        ``method="adiabatic_exact"`` (``medium.solar.adiabatic``), with every
        method/use_LZ/BSM-extension compatibility check centralised there.
        This medium has no transition function (no coherent evolutor exists
        in the adiabatic solar model).
    medium.solar.flux
        Combines ``probability.solar_probability_state`` with the source's
        total fluxes and optional spectra to produce flavour-resolved solar
        fluxes, and integrates them over energy.
    medium.solar.validation
        Helpers comparing this package's output against the legacy peanuts
        implementation.
"""



from tpeanuts.medium.solar.io import (
    load_solar_composition,
    load_solar_density,
    solar_provider_path,
)
from tpeanuts.medium.solar.profile import (
    SolarMediumParameters,
    SolarMediumProfile,
    build_solar_medium,
)
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
    landau_zener_spatial_correction,
)
from tpeanuts.medium.solar.adiabatic import (
    mass_weights_adiabatic_approximated,
    mass_weights_adiabatic_exact,
)
from tpeanuts.medium.solar.probability import (
    solar_probability_mass,
    solar_probability_state,
    solar_probability_integrated,
)
from tpeanuts.medium.solar.flux import solar_flux_state, solar_flux_integrated

__all__ = [
    "load_solar_composition",
    "load_solar_density",
    "solar_provider_path",
    "SolarMediumParameters",
    "SolarMediumProfile",
    "build_solar_medium",
    "Vk",
    "DeltamSqee",
    "th13_M",
    "th12_M",
    "density_gradient",
    "resonance_radius",
    "plz",
    "landau_zener_spatial_correction",
    "mass_weights_adiabatic_approximated",
    "mass_weights_adiabatic_exact",
    "solar_probability_mass",
    "solar_probability_state",
    "solar_probability_integrated",
    "solar_flux_state",
    "solar_flux_integrated",
]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
Physical constants used throughout tpeanuts.

This module is the single source of truth for the physical constants that
enter the neutrino-oscillation-in-matter calculation: the Fermi constant
(matter potential), hbar and c (unit conversions between natural and SI
units), the Avogadro constant and proton mass (mass-density to number-density
conversion for the matter potential), and the geometric constants (Earth and
Sun radii, astronomical unit) used to build propagation paths.

Module constants:
    G_F_MEV_M2: Fermi coupling constant, in MeV^-2.
    HBAR_MeV_s: Reduced Planck constant, in MeV*s.
    C_M_s: Speed of light in vacuum, in m/s.
    HBARC_MeV_m: hbar*c product, in MeV*m, used to convert between inverse
        energy and length.
    N_A: Avogadro constant, in mol^-1.
    R_E: Earth mean radius, in m.
    R_E_KM: Earth mean radius, in km.
    R_SUN: Solar radius, in m.
    R_SUN_KM: Solar radius, in km.
    AU_KM: Astronomical unit, in km.
    AU_M: Astronomical unit, in m.
    SUN_EARTH_DISTANCE_KM: Default mean Sun-Earth distance, in km (one AU).
    SUN_EARTH_DISTANCE_AU: Default mean Sun-Earth distance, in AU (always 1.0
        by definition).
    M_PROTON_KG: Proton mass, in kg, used as the nucleon mass scale for
        density conversions.
    GCM3_TO_NUCLEON_MOLCM3: Conversion factor from mass density in g/cm^3 to
        nucleon (baryon) number density in mol/cm^3, assuming the proton mass
        as the average nucleon mass.
"""

# Fermi coupling constant G_F, in MeV^-2. Sets the strength of the
# charged-current matter potential felt by electron neutrinos (MSW effect).
G_F_MEV_M2 = 1.1663787e-11

# Reduced Planck constant hbar, in MeV*s.
HBAR_MeV_s = 6.582119569e-22

# Speed of light in vacuum, in m/s (exact SI value).
C_M_s = 299_792_458.0

# hbar*c product, in MeV*m. Used to convert between inverse energy (MeV^-1)
# and length (m) in natural-unit expressions such as oscillation phases.
HBARC_MeV_m = HBAR_MeV_s * C_M_s

# Avogadro constant, in mol^-1. Used to convert mass densities to number
# densities of target particles (electrons/nucleons) for the matter potential.
N_A = 6.02214076e23

# Earth mean radius, in meters. Reference length scale for Earth-crossing
# neutrino propagation paths (PREM-like geometry).
R_E = 6.371e6  # m

# Earth mean radius, in kilometers.
R_E_KM = float(R_E) / 1.0e3

# Solar radius, in meters. Reference length scale for the neutrino production
# point inside the Sun and for solar-density profile integration.
R_SUN = 6.957e8  # m

# Solar radius, in kilometers.
R_SUN_KM = float(R_SUN) / 1.0e3

# Astronomical unit (mean Sun-Earth distance), in kilometers.
AU_KM = 149_597_870.7  # km

# Astronomical unit (mean Sun-Earth distance), in meters.
AU_M = AU_KM * 1.0e3

# Default mean Sun-Earth distance used when no ephemeris-based distance is
# supplied, expressed in kilometers (equal to one astronomical unit).
SUN_EARTH_DISTANCE_KM = AU_KM

# Default mean Sun-Earth distance in astronomical units (always 1.0 by
# definition of the AU).
SUN_EARTH_DISTANCE_AU = 1.0

# Proton mass, in kilograms. Used as the nucleon mass scale when converting
# matter mass density to nucleon number density.
M_PROTON_KG = 1.67262192369e-27

# Conversion factor from mass density [g/cm^3] to nucleon molar density
# [mol/cm^3], using the proton mass as the average nucleon mass:
# n[mol/cm^3] = rho[g/cm^3] * GCM3_TO_NUCLEON_MOLCM3.
GCM3_TO_NUCLEON_MOLCM3 = 1.0e-3 / (M_PROTON_KG * N_A)

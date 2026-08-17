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
    M_ELECTRON_MEV: Electron mass, in MeV, used by
        ``tpeanuts.detector.interaction.neutrino_electron``'s elastic
        scattering cross section.
    SIN2_THETA_W: Weak mixing angle sin^2(theta_W), effective low-energy
        value, used by the same cross section.
    Q_CC_DEUTERON_MEV: Threshold (Q-value) of nu_e + d -> p + p + e-, used by
        ``tpeanuts.detector.interaction.deuteron``.
    Q_NC_DEUTERON_MEV: Deuteron binding energy, the threshold of
        nu_x + d -> p + n + nu_x, used by the same module.
    DELTA_NP_MEV: Neutron-proton mass difference, used by
        ``tpeanuts.detector.interaction.inverse_beta_decay`` to convert
        antineutrino energy to positron energy.
    IBD_THRESHOLD_MEV: Reaction threshold of nu_e_bar + p -> e+ + n, used by
        the same module.
    SIGMA0_IBD_CM2_PER_MEV2: Leading-order normalization of the inverse
        beta decay cross section, used by the same module.
    MEV_TO_JOULE: MeV -> Joule conversion (exact, from the SI elementary
        charge), used by ``tpeanuts.detector.dayabay.flux`` to convert
        thermal power to a fission rate.
    M_NEUTRON_MEV, M_PROTON_MEV: Neutron and proton masses, in MeV (PDG2024),
        used by ``tpeanuts.detector.interaction.inverse_beta_decay``'s
        order-1/M (Vogel & Beacom, Eq. 13-15) inverse beta decay cross
        section.
    NEUTRON_LIFETIME_S: Free-neutron lifetime, in seconds (PDG2024), used by
        the same module to fix the order-1/M cross section's overall
        normalization via Vogel & Beacom, Eq. (12).
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

# Electron mass, in MeV (CODATA). Sets the kinematic endpoint T_max(E_nu)
# and the g_L*g_R interference term of neutrino-electron elastic scattering.
M_ELECTRON_MEV = 0.51099895

# Weak mixing angle sin^2(theta_W), effective low-energy value (PDG). Enters
# the neutrino-electron elastic scattering couplings g_L/g_R; sin^2(theta_W)
# has a mild scheme/scale dependence not modeled here (this is the same
# low-energy effective value conventionally used for solar/reactor-neutrino
# electron-scattering cross sections, e.g. Bahcall, Kamionkowski & Sirlin,
# Phys. Rev. D 51, 6146 (1995)).
SIN2_THETA_W = 0.23122

# Threshold (Q-value) of the SNO charged-current reaction nu_e + d -> p + p
# + e-, in MeV. A commonly quoted reaction threshold in the solar/SNO
# literature (e.g. Bahcall, Krastev & Smirnov and later SNO analyses).
Q_CC_DEUTERON_MEV = 1.4421

# Deuteron binding energy, in MeV (CODATA/nuclear-data reference value),
# equal to the threshold of the neutral-current deuteron breakup
# nu_x + d -> p + n + nu_x.
Q_NC_DEUTERON_MEV = 2.224566

# Neutron-proton mass difference, in MeV (CODATA), used to convert incident
# antineutrino energy to positron energy in inverse beta decay to leading
# (no-recoil) order: E_e ~ E_nu - DELTA_NP_MEV.
DELTA_NP_MEV = 1.29333236

# Reaction threshold of nu_e_bar + p -> e+ + n, in MeV:
# ((m_n + m_e)^2 - m_p^2) / (2 m_p), commonly quoted as 1.806 MeV
# (e.g. Vogel & Beacom, Phys. Rev. D60, 053003 (1999)).
IBD_THRESHOLD_MEV = 1.806

# Leading-order (no-recoil) normalization of the inverse beta decay cross
# section sigma(E_nu) ~ SIGMA0_IBD_CM2_PER_MEV2 * E_e * p_e, in cm^2/MeV^2
# (Vogel & Beacom, Phys. Rev. D60, 053003 (1999), Eq. 25 zeroth-order term).
SIGMA0_IBD_CM2_PER_MEV2 = 0.0952e-42

# MeV -> Joule conversion (exact, from the SI elementary charge e = 1
# eV / (1.602176634e-19 J)).
MEV_TO_JOULE = 1.602176634e-13

# Neutron mass, in MeV (PDG2024, "Review of Particle Physics", p.176). Used
# by the order-1/M inverse beta decay cross section (Vogel & Beacom, Phys.
# Rev. D60, 053003 (1999)) to form the average nucleon mass M=(M_n+M_p)/2
# that suppresses its 1/M recoil terms.
M_NEUTRON_MEV = 939.5654205

# Proton mass, in MeV (PDG2024, "Review of Particle Physics", p.174). See
# M_NEUTRON_MEV.
M_PROTON_MEV = 938.27208816

# Free-neutron lifetime, in seconds (PDG2024, "Review of Particle Physics",
# p.176). Fixes sigma0's normalization via Vogel & Beacom, Eq. (12):
# sigma0 = 2*pi^2 / (m_e^5 * f_p.s. * tau_n * (f^2+3g^2)).
NEUTRON_LIFETIME_S = 878.4

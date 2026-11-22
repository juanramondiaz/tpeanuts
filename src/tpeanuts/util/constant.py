#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
#
# TPeanuts — Fast Simulation of Neutrino Oscillations in Matter.
#
# Developed as part of the Master's Thesis for the MSc in Advanced Physics,
# specialization in Theoretical Physics, at the Universitat de València.
#
# Author
# ------
# Juan Ramón Díaz Santos
# diazjuan@alumni.uv.es
#
# Supervisors
# -----------
# Roberto Ruiz de Austri Bazán
# rruiz@ific.uv.es
#
# Michele Lucente
# michele.lucente@unibo.it
#
# Universitat de València, Spain
# September 2026
# =============================================================================

"""Physical constants used throughout tpeanuts.

The single source of truth for every physical constant the project needs --
populated incrementally as new modules need new constants, not all at once,
so nothing here is speculative or unused. Today that means exactly what
``tpeanuts.physics.common.potential`` needs to build the matter and kinetic
potentials; geometric constants (Earth/Sun radii, ...), particle masses, and
reaction thresholds join this file when the ``medium``/``source``/
``detector`` modules that need them get built.

Module constants:
    G_F_MEV_M2: Fermi coupling constant, in MeV^-2.
    HBAR_MeV_s: Reduced Planck constant, in MeV*s.
    C_M_s: Speed of light in vacuum, in m/s.
    HBARC_MeV_m: hbar*c product, in MeV*m, used to convert between inverse
        energy and length.
    N_A: Avogadro constant, in mol^-1.
    R_E: Earth mean radius, in m. Default propagation length scale.
    R_SUN: Solar radius, in m, used by
        ``tpeanuts.physics.evolution.adiabatic.landau_zener`` to convert
        the dimensionless density scale height (in units of R_sun) into
        metres.
    EARTH_INCLINATION_RAD: Earth's rotation-axis (orbital-plane) inclination,
        in radians, setting the annual swing of the Sun's apparent
        declination used by
        ``tpeanuts.medium.earth.exposure.time_average``'s nadir-angle
        exposure geometry.
    MU_B_MEV_PER_T: Bohr magneton, in MeV/T. Sets the strength of a
        transition magnetic moment (expressed in units of the Bohr magneton,
        matching the neutrino magnetic-moment literature) felt by a magnetic
        field.
    M_ELECTRON_MEV: Electron mass, in MeV, used by
        ``tpeanuts.detector.ibd_cross_section.ibd_cross_section``.
    DELTA_NP_MEV: Neutron-proton mass difference, in MeV, used by
        ``ibd_cross_section`` for the zeroth-order (no-recoil) positron energy
        ``E_e = E_nu - DELTA_NP_MEV``.
    IBD_THRESHOLD_MEV: Reaction threshold of ``nu_e_bar + p -> e+ + n``, in
        MeV, used by ``ibd_cross_section``.
    SIGMA_UNIT_CM2: Unit, in cm^2, in which every ``tpeanuts.detector`` cross
        section is expressed (see the comment at its definition).
    SIGMA0_IBD_CM2_PER_MEV2: Leading-order (zeroth-order, no-recoil)
        normalization of the inverse beta decay cross section
        (Vogel & Beacom, Phys. Rev. D60, 053003 (1999), Eq. 11), in
        cm^2/MeV^2, used by ``ibd_cross_section``.
    M_NEUTRON_MEV, M_PROTON_MEV: Neutron and proton masses, in MeV (PDG2024),
        used by ``tpeanuts.detector.ibd_cross_section.ibd_differential_cross_section_precise``
        as the order-1/M nucleon mass scale ``M = (M_NEUTRON_MEV +
        M_PROTON_MEV) / 2``.
    MEV_TO_JOULE: MeV -> Joule conversion (exact, from the SI elementary
        charge), used by ``tpeanuts.experiments.dayabay.flux`` to convert
        thermal power to a fission rate.
    M_PROTON_KG: Proton mass, in kg, used as the nucleon mass scale for
        ``GCM3_TO_NUCLEON_MOLCM3``.
    GCM3_TO_NUCLEON_MOLCM3: Conversion factor from mass density in g/cm^3
        to nucleon number density in mol/cm^3 (``n = rho *
        GCM3_TO_NUCLEON_MOLCM3``, treating every nucleon as a proton-mass
        particle), used by ``tpeanuts.medium.atmosphere.density`` to turn
        a mass-density profile into electron/neutron densities via the
        electron fraction ``Ye``/``1-Ye`` -- not needed until now since
        ``tpeanuts.medium.earth``'s own PREM table already ships
        electron/neutron density directly, precomputed offline.
    SIN2_THETA_W: Weak mixing angle sin^2(theta_W), effective low-energy
        value, used by ``tpeanuts.detector.neutrino_electron``'s elastic
        scattering couplings.
    Q_CC_DEUTERON_MEV: Threshold (Q-value) of the SNO charged-current
        reaction ``nu_e + d -> p + p + e-``, in MeV, used by
        ``tpeanuts.detector.deuteron``.
    Q_NC_DEUTERON_MEV: Deuteron binding energy, in MeV, equal to the
        threshold of the neutral-current deuteron breakup
        ``nu_x + d -> p + n + nu_x``, used by the same module.
"""

# Fermi coupling constant G_F, in MeV^-2. Sets the strength of the
# charged-current matter potential felt by electron neutrinos (MSW effect).
G_F_MEV_M2 = 1.1663787e-11

# Reduced Planck constant hbar, in MeV*s. Exact value from the SI definitions, hbar = h / (2 pi e) with h and e
# fixed by the 2019 SI (h = 6.62607015e-34 J s, e = 1.602176634e-19 C); not rounded, so that every oscillation phase
# is exact to double precision.
HBAR_MeV_s = 6.582119569509067e-22

# Speed of light in vacuum, in m/s (exact SI value).
C_M_s = 299_792_458.0

# hbar*c product, in MeV*m. Used to convert between inverse energy (MeV^-1)
# and length (m) in natural-unit expressions such as oscillation phases.
HBARC_MeV_m = HBAR_MeV_s * C_M_s

# Avogadro constant, in mol^-1. Used to convert mass densities to number
# densities of target particles (electrons/nucleons) for the matter potential.
N_A = 6.02214076e23

# Earth mean radius, in meters. Default reference length scale used to
# non-dimensionalize the propagation generator.
R_E = 6.371e6

# Solar radius, in meters (matches v1_0's own value exactly).
R_SUN = 6.957e8

# Earth's rotation-axis (orbital-plane) inclination, in radians (matches
# v1_0's own reference value, ~23.45 degrees).
EARTH_INCLINATION_RAD = 0.4091

# Bohr magneton, in MeV/T (CODATA value 5.7883818060e-5 eV/T). Sets the
# strength of a transition magnetic moment coupling to a magnetic field --
# a moment is conventionally quoted as a dimensionless multiple of this
# unit (e.g. mu_nu < 2.9e-11 mu_B), rather than directly in MeV/T.
MU_B_MEV_PER_T = 5.7883818060e-11

# Electron mass, in MeV (PDG2024).
M_ELECTRON_MEV = 0.51099895

# Neutron-proton mass difference, in MeV (PDG2024) -- sets the zeroth-order
# (no-recoil) order: E_e ~= E_nu - DELTA_NP_MEV.
DELTA_NP_MEV = 1.29333236

# Reaction threshold of nu_e_bar + p -> e+ + n, in MeV.
IBD_THRESHOLD_MEV = 1.806

# Neutron mass, in MeV (PDG2024).
M_NEUTRON_MEV = 939.5654205

# Proton mass, in MeV (PDG2024) -- distinct from M_PROTON_KG, which is in kg
# for GCM3_TO_NUCLEON_MOLCM3's own unit conversion, not MeV like
# M_NEUTRON_MEV.
M_PROTON_MEV = 938.27208816

# MeV -> Joule conversion (exact, via the SI elementary charge).
MEV_TO_JOULE = 1.602176634e-13

# Unit of every cross section in tpeanuts.detector: sigma is stored as sigma / SIGMA_UNIT_CM2,
# i.e. in units of 1e-42 cm^2 (and dsigma/dT in 1e-42 cm^2/MeV). Physical cross sections are
# ~1e-42 (IBD), ~1e-45 (nu-e elastic) and down to ~1e-49 cm^2/MeV (deuteron), i.e. subnormal or
# zero in float32 (smallest normal ~1.2e-38; CUDA also flushes subnormals to zero), whereas the
# reduced values are O(1e-7..1) everywhere. The factor is restored exactly once, where the target
# count is applied: see tpeanuts.detector.common.target.reduced_target_count.
SIGMA_UNIT_CM2 = 1.0e-42

# Leading-order (zeroth-order, no-recoil) normalization of the IBD cross
# section sigma(E_nu) ~= SIGMA0_IBD_CM2_PER_MEV2 * E_e * p_e, in cm^2/MeV^2
# (Vogel & Beacom, Phys. Rev. D60, 053003 (1999), Eq. 11).
SIGMA0_IBD_CM2_PER_MEV2 = 0.0952e-42

# Proton mass, in kg (CODATA), used as the nucleon mass scale for
# GCM3_TO_NUCLEON_MOLCM3 (matches v1_0's own value exactly).
M_PROTON_KG = 1.67262192369e-27

# n[mol/cm^3] = rho[g/cm^3] * GCM3_TO_NUCLEON_MOLCM3 -- converts a mass
# density into a nucleon number density, treating every nucleon as a
# proton-mass particle.
GCM3_TO_NUCLEON_MOLCM3 = 1.0e-3 / (M_PROTON_KG * N_A)

# Weak mixing angle sin^2(theta_W), effective low-energy value, used for
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

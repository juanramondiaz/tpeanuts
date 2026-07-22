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
Default configuration values for tpeanuts modules.

This module centralizes numerical and model defaults used across the library so
function signatures can reference a single source of truth. Defaults are
grouped by domain: I/O paths/filenames, Atmosphere density-profile models,
MCEq and pymsis backend configuration, Earth propagation, and Earth exposure
(detector live-time / nadir-angle distribution) calculations. Physical
defaults reproduce the legacy PEANUTS / nuSQuIDS / standard-atmosphere
reference values used to validate tpeanuts against those tools.
"""

# Module summary: stores reusable default arguments for density models,
# propagation helpers, and external Atmosphere backends.
from __future__ import annotations

import torch


# ============================================================
# I/O path and filename defaults
# ============================================================

# Default torch device specification (None lets callers resolve CUDA/CPU).
device = None
# Root directory for input/output data files.
data_dir = "data"
# Directory containing solar model and flux reference tables.
solar_data_dir = "data/solar"
# Directory containing legacy PEANUTS reference data, used for comparisons.
legacy_data_dir = "data/peanuts"
# Filename of the solar neutrino production-point distribution table.
# Path is relative to solar_data_dir (i.e. data/solar/).
# Default provider/model: the SF-III AGSS09 model published on Zenodo.
solar_provider = "zenodo"
solar_spectrum_provider = "legacy"
solar_density_filename = "zenodo/density/density_SF3_AGSS09.csv"
solar_production_filename = "zenodo/production/production_SF3_AGSS09.csv"
# Filename of the solar neutrino flux normalization table.
# Path is relative to solar_data_dir (i.e. data/solar/).
# Default: SF-III AGSS09 total-flux table.
solar_fluxes_filename = "zenodo/flux/fluxes_SF3_AGSS09.csv"
# Filename of the solar structure+composition table used to derive the
# neutron-density profile n_n(r) for the 3+1 sterile neutral-current term
# (see medium.solar.io.load_solar_composition). Path is relative to
# solar_data_dir. This Zenodo table remains available as an explicit fallback
# when a selected density table does not already contain neutron density.
solar_composition_filename = "zenodo/raw/struct+nu_SF3_AGSS09.dat"
# Filename of the tabulated Sun-Earth distance vs. day-of-year table.
solar_sun_earth_distance_filename = "geometry/sun_earth_distance.csv"

# Canonical provider selections for tabulated input data.
atmosphere_flux_provider = "honda"
atmosphere_flux_dir = "data/atmosphere"
atmosphere_flux_filename = "honda/flux/honda_flux.csv"
earth_density_provider = "prem"
earth_reference_data_dir = "data/earth"
earth_reference_density_filename = "prem/density/prem_density.csv"

# Perturbative fits derived from the canonical PREM provider data.
earth_density_dir = "data/earth/prem/fit"
# Filename of the default Earth density profile table.
earth_density_filename = "even_power_electron.csv"
# Filename of the neutron-density companion table for the even-power Earth
# model, read when EvenPowerProfileLayered is built with
# include_neutron=True. Same rj shells and column format (rj, alpha, beta,
# gamma) as earth_density_filename, fitted to n_n(r) instead of n_e(r).
earth_density_filename_nn = "even_power_neutron.csv"

# Canonical MCEq atmospheric mass-density table.
atmosphere_density_file = (
    "data/atmosphere/mceq/density/atmosphere_density_profile.csv"
)

# Default directory containing canonical Honda/HKKM tables.
honda_dataset = "data/atmosphere/honda"

# Default file extension used when saving torch tensors to disk.
torch_default_extension = ".pt"
# Recognized torch tensor file extensions accepted when loading.
torch_file_extensions = (".pt", ".pth")

# Default output directory for atmosphere height/energy/angle flux tensors.
atmosphere_height_flux_output_dir = "atmosphere_height_flux_outputs"
# Default filename for the saved atmospheric flux tensor
# phi(E, theta, h) (energy, zenith angle, production height).
atmosphere_height_flux_filename = "phi_E_theta_h.pt"

# Default filename for the saved detector-level flux tensor.
detector_flux_filename = "Detectorflux.pt"

# Filename prefix used when caching nadir-angle exposure distributions.
earth_exposure_cache_filename_prefix = "nadir_exposure"


# ============================================================
# Atmosphere density defaults
# ============================================================

# Atmosphere density model used by default ("exponential" isothermal
# barometric profile).
atmosphere_source_density = "exponential"
# Sea-level atmospheric mass density, in g/cm^3 (U.S. Standard Atmosphere).
atmosphere_rho0_gcm3 = 1.225e-3
# Atmospheric scale height for the exponential density profile, in km.
atmosphere_scale_height_km = 8.4
# Electron fraction Ye (electrons per nucleon) of air, dimensionless.
atmosphere_Ye = 0.5
# Whether the numerical atmosphere evolutor also samples and applies the
# neutron-density neutral-current (NC) matter term (3+1 sterile only; see
# core.common.hamiltonian.hamiltonian_matter_reduced). False (the default)
# reproduces the pre-existing CC-only behaviour exactly.
atmosphere_include_matter_nc = False
# Sea-level atmospheric mass density used by the nuSQuIDS reference
# comparison profile, in g/cm^3.
atmosphere_nusquids_rho0_gcm3 = 0.0012
# Atmospheric scale height used by the nuSQuIDS reference comparison
# profile, in km.
atmosphere_nusquids_scale_height_km = 7.594
# Electron fraction Ye used by the nuSQuIDS reference comparison profile,
# dimensionless.
atmosphere_nusquids_Ye = 0.494
# Production/reference height above sea level used by the nuSQuIDS
# comparison profile, in km.
atmosphere_nusquids_height_km = 22.0


# ============================================================
# MCEq density defaults
# ============================================================

# Default zenith angle for MCEq atmospheric flux calculations, in degrees
# (0 deg = vertical/overhead).
mceq_theta_deg = 0.0
# Default hadronic interaction model used by MCEq for cosmic-ray air showers.
mceq_interaction_model = "SIBYLL23D"
# Default atmospheric density model used by MCEq for shower development.
# "CORSIKA" (US Standard Atmosphere) is the generic mid-latitude profile
# available in MCEq/CRFlux, closest of the bundled options to a
# Frejus/JUNO-like mid-latitude site (no South-Pole bias, unlike NASA/ICECUBE).
mceq_density_model = "CORSIKA"
# Default primary cosmic-ray flux model shared between MCEq
# (mceq_interaction_model/mceq_density_model above) and the CORSIKA/CRFlux
# primary-sampling pipeline (run1/run2), so both sides inject the same
# top-of-atmosphere spectrum and their outputs remain directly comparable.
mceq_primary_model = "PolyGonato"


# ============================================================
# pymsis density defaults
# ============================================================

# Atmosphere density source flag selecting the pymsis (NRLMSIS) backend.
pymsis_source_density = "pymsis"
# Default UTC date/time used for the pymsis atmosphere query (ISO-8601).
pymsis_date = "2026-05-10T12:00"
# Default geographic longitude for the pymsis query, in degrees.
pymsis_lon_deg = -0.38
# Default geographic latitude for the pymsis query, in degrees
# (Valencia, Spain).
pymsis_lat_deg = 39.47
# Default daily F10.7 solar radio flux index, in solar flux units (sfu),
# used as a pymsis space-weather input.
pymsis_f107 = 150.0
# Default 81-day average F10.7 solar radio flux index, in solar flux units
# (sfu), used as a pymsis space-weather input.
pymsis_f107a = 150.0
# Default geomagnetic Ap index used as a pymsis space-weather input.
pymsis_ap = 7.0
# Default pymsis (NRLMSIS) model version number.
pymsis_version = 2.1
# Electron fraction Ye assumed for the pymsis atmosphere composition,
# dimensionless.
pymsis_Ye = 0.499
# Default real floating-point torch dtype used across the library.
dtype = torch.float64


# ============================================================
# Earth propagation defaults
# ============================================================

# Default Earth-crossing propagation method: "analytical" closed-form
# evolution versus numerical step-by-step integration.
earth_method = "analytical"
# Default numerical integration scheme used when earth_method requests
# numerical propagation.
earth_numerical_method = "midpoint"
# Whether the default propagation targets antineutrinos (True) or
# neutrinos (False); flips the sign of the matter potential.
earth_antinu = False
# Whether to express states/evolution in the neutrino mass basis (True) or
# flavour basis (False).
earth_massbasis = True
# Whether to compute the full coherent oscillation (True) or a reduced
# approximation (False) by default.
earth_full_oscillation = False
# Whether to re-unitarize evolution operators after numerical integration to
# correct accumulated round-off error.
earth_reunitarize = False
# Whether method="numerical" Earth propagation also samples and applies the
# neutron-density neutral-current (NC) matter term (3+1 sterile only; see
# core.common.hamiltonian.hamiltonian_matter_reduced). False (the default)
# reproduces the pre-existing CC-only behaviour exactly. Requires the Earth
# profile model to be built with include_neutron=True (both
# EvenPowerProfileLayered and PremTabulatedProfile support this); the
# default construction (include_neutron=False) does not.
earth_include_matter_nc = False
# Default torch device string requested for Earth propagation
# ("cuda" falls back to CPU when unavailable via the resolution helpers).
earth_device = "cuda"
# Default perturbative expansion scheme name used by the perturbative
# Earth-matter evolution profile.
earth_profile_perturbative_name = "even_power"
# Default number of integration steps for numerical Earth propagation.
earth_nsteps = 1000
# Default number of steps used when tabulating oscillation probabilities
# along the path.
earth_probability_nsteps = 100
# Default detector depth below the Earth's surface, in meters.
earth_depth_m = 0.0
# Whether to use a tabulated (interpolated) Earth density profile by default
# instead of an analytical shell model.
earth_tabulated_density = False


# ============================================================
# Earth exposure defaults
# ============================================================

# Default detector geographic latitude, in radians (negative = Southern
# hemisphere), used for nadir-angle exposure calculations.
earth_lam_rad = -1.0
# Default start day-of-year for the exposure time window (day 0).
earth_d1 = 0.0
# Default end day-of-year for the exposure time window (day 365, i.e. one
# full year).
earth_d2 = 365.0
# Default number of sample points used to evaluate the exposure
# distribution over the time window.
earth_exposure_ns = 1000
# Default directory used to cache computed nadir-angle exposure
# distributions.
earth_cache_dir = "data/exposure_cache"
# Default directory used for legacy-PEANUTS-compatible exposure caches.
earth_legacy_cache_dir = "cache_exposure"
# Whether the exposure distribution is normalized to unit integral by
# default.
earth_normalized_exposure = False
# Default nadir/zenith angle convention label used when reporting exposure.
earth_angle = "Nadir"
# Default day/night selection filter for exposure calculations (None means
# no day/night restriction).
earth_daynight = None
# Default chunk size (in eta) used to batch exposure integration; None
# disables chunking.
earth_chunk_eta = None
# Default detector orbital/rotational inclination angle, in radians, used in
# the exposure geometry (Super-Kamiokande-like reference value).
earth_inclination = 0.4091
# Convergence tolerance for the elliptic-integral evaluation used in the
# exposure calculation.
earth_elliptic_tol = 1.0e-12
# Small angular epsilon, in radians, used to regularize exposure
# calculations near singular nadir angles.
earth_angle_eps = 1.0e-5
# Default lower integration bound for the nadir-angle exposure integral,
# in radians.
earth_angle_a1 = 0.0
# Default upper integration bound for the nadir-angle exposure integral,
# in radians (pi, i.e. the full nadir-angle range).
earth_angle_a2 = torch.pi
# Minimum denominator used to avoid division by zero when normalizing
# exposure distributions.
earth_normalize_eps = 1.0e-30
# Whether to read/write the exposure cache by default.
earth_use_cache = True

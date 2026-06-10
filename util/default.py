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
function signatures can reference a single source of truth.
"""

# Module summary: stores reusable default arguments for density models,
# propagation helpers, and external atmospheric backends.
from __future__ import annotations

import torch


# ============================================================
# Atmospheric density defaults
# ============================================================

atmosphere_source_density = "exponential"
atmosphere_rho0_gcm3 = 1.225e-3
atmosphere_scale_height_km = 8.4
atmosphere_Ye = 0.5


# ============================================================
# MCEq density defaults
# ============================================================

mceq_theta_deg = 0.0
mceq_interaction_model = "SIBYLL23D"
mceq_density_model = "CORSIKA"


# ============================================================
# pymsis density defaults
# ============================================================

pymsis_source_density = "pymsis"
pymsis_date = "2026-05-10T12:00"
pymsis_lon_deg = -0.38
pymsis_lat_deg = 39.47
pymsis_f107 = 150.0
pymsis_f107a = 150.0
pymsis_ap = 7.0
pymsis_version = 2.1
pymsis_Ye = 0.499
dtype = torch.float64


# ============================================================
# Earth propagation defaults
# ============================================================

earth_method = "analytical"
earth_numerical_method = "midpoint"
earth_antinu = False
earth_massbasis = True
earth_full_oscillation = False
earth_reunitarize = False
earth_device = "cuda"
earth_nsteps = 1000
earth_probability_nsteps = 100
earth_rtol = 1.0e-8
earth_atol = 1.0e-8
earth_depth_m = 0.0
earth_tabulated_density = False


# ============================================================
# Earth exposure defaults
# ============================================================

earth_lam_rad = -1.0
earth_d1 = 0.0
earth_d2 = 365.0
earth_exposure_ns = 1000
earth_cache_dir = "data/exposure_cache"
earth_legacy_cache_dir = "cache_exposure"
earth_normalized_exposure = False
earth_angle = "Nadir"
earth_daynight = None
earth_chunk_eta = None
earth_inclination = 0.4091
earth_elliptic_tol = 1.0e-12
earth_angle_eps = 1.0e-5
earth_angle_a1 = 0.0
earth_angle_a2 = torch.pi
earth_normalize_eps = 1.0e-30
earth_use_cache = True

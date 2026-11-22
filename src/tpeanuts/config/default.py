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

"""Default values every configurable module's fields/parameters fall back to.

Populated incrementally as new modules need new defaults, not all at
once, exactly like ``tpeanuts.util.constant``'s own stated policy --
today that covers ``tpeanuts.inference.fit``'s optimizer/fit
hyperparameters plus the Earth/atmosphere/solar medium and source
providers ported so far; a future medium/source/detector provider
choice, file path, or numerical tolerance joins this file once that
module actually exists and needs one (mirroring v1_0's own
``config/default.py``, which grew this way across its own Earth/solar/
atmosphere modules).

Module default values:
    FIT_MAX_ITER: Default optimizer step budget for
        ``tpeanuts.inference.fit.minimize``/``fit``
        -- for LBFGS, its own internal line-search iteration cap.
        -- for any other optimizer, the number of explicit ``optimizer.step()``
          calls run.
    FIT_SCALE_MAX_RELATIVE: Upper bound of the scale that ``minimize`` gives each
        parameter, in units of that parameter's starting magnitude (or of 1 if it
        starts at exactly 0). It keeps a parameter that starts at its optimum, whose
        gradient vanishes, from being given a huge scale and thrown away by the first step.
    LBFGS_LR, LBFGS_TOLERANCE_GRAD, LBFGS_TOLERANCE_CHANGE,
    LBFGS_LINE_SEARCH_FN: Default ``tpeanuts.inference.fit.LBFGSConfig``
        field values, forwarded to ``torch.optim.LBFGS``'s own
        constructor unchanged.
    ADAM_LR, ADAM_BETAS, ADAM_EPS, ADAM_WEIGHT_DECAY: Default
        ``tpeanuts.inference.fit.AdamConfig`` field values, forwarded to
        ``torch.optim.Adam``'s own constructor unchanged.
    EARTH_NSTEPS: Default number of equal-length chord segments
        ``tpeanuts.medium.earth.earth.EarthMedium`` discretizes a
        trajectory into (matches v1_0's own ``EarthParameters.nsteps``
        default).
    EARTH_EXPOSURE_D1, EARTH_EXPOSURE_D2: Default day-of-year integration
        window (in days, ``[0, 365]``) for
        ``tpeanuts.medium.earth.exposure.time_average.integral_day`` --
        the full year by default.
    EARTH_EXPOSURE_NS: Default number of nadir-angle grid points
        ``tpeanuts.medium.earth.exposure.time_average.make_eta_grid``
        spans ``[0, pi]`` with.
    EARTH_EXPOSURE_DAYNIGHT: Default day/night restriction for
        ``make_eta_grid`` -- ``None`` keeps the full ``[0, pi]`` grid.
    EARTH_EXPOSURE_ELLIPTIC_TOL: Default convergence tolerance for the
        incomplete elliptic integral evaluated by
        ``time_average.indefinite_integral_day``
        (``tpeanuts.util.math.ellipf_incomplete``).
    EARTH_EXPOSURE_ANGLE_EPS: Default angular epsilon, in radians, used by
        ``time_average.integral_angle`` to regularize its integration
        interval near singular nadir angles.
    EARTH_EXPOSURE_NORMALIZED: Whether
        ``tpeanuts.medium.earth.exposure.table.build_nadir_exposure``
        normalizes its own exposure weights to unit trapezoidal integral
        by default (matches v1_0's own default: raw, unnormalized time
        weights).
    ATMOSPHERE_SOURCE_DENSITY: Default density backend for
        ``tpeanuts.medium.atmosphere.density.atmosphere_density``
        (matches v1_0's own ``atmosphere_source_density`` default).
    ATMOSPHERE_RHO0_GCM3, ATMOSPHERE_SCALE_HEIGHT_M: Default reference
        density and scale height for the ``"exponential"`` atmosphere
        density backend (matches v1_0's own
        ``atmosphere_rho0_gcm3``/``atmosphere_scale_height_km`` values --
        the scale height is converted from v1_0's km to this project's
        own metres convention).
    ATMOSPHERE_YE: Default electron fraction used to split an atmosphere
        mass density into electron/neutron densities (matches v1_0's own
        ``atmosphere_Ye``).
    ATMOSPHERE_PYMSIS_DATE, ATMOSPHERE_PYMSIS_LON_DEG,
    ATMOSPHERE_PYMSIS_LAT_DEG, ATMOSPHERE_PYMSIS_F107,
    ATMOSPHERE_PYMSIS_F107A, ATMOSPHERE_PYMSIS_AP,
    ATMOSPHERE_PYMSIS_VERSION: Default UTC date, geographic location,
        space-weather indices, and NRLMSIS version for the ``"pymsis"``
        atmosphere density backend (matches v1_0's own ``pymsis_date``/
        ``pymsis_lon_deg``/``pymsis_lat_deg``/``pymsis_f107``/
        ``pymsis_f107a``/``pymsis_ap``/``pymsis_version`` values
        verbatim).
    ATMOSPHERE_NSTEPS: Default number of equal-length path segments
        ``tpeanuts.medium.atmosphere.atmosphere.AtmosphereMedium``
        discretizes a trajectory into (matches v1_0's own
        ``AtmosphereParameters.nsteps`` default).
    ATMOSPHERE_PERTURBATIVE_SEGMENTS, ATMOSPHERE_PERTURBATIVE_DEGREE:
        Default number of ``PropagationMethod.ANALYTIC_PERTURBATIVE``
        segments, and the local-polynomial degree fitted within each one
        (matches v1_0's own ``AtmosphereParameters.perturbative_segments``/
        ``perturbative_degree`` defaults).
    ATMOSPHERE_MCEQ_INTERACTION_MODEL, ATMOSPHERE_MCEQ_PRIMARY_MODEL,
    ATMOSPHERE_MCEQ_DENSITY_MODEL: Default hadronic interaction/cosmic
        -ray primary/atmospheric density model names for
        ``tpeanuts.source.atmosphere.mceq.MCEqConfig`` (matches v1_0's
        own ``config/default.py`` ``mceq_interaction_model``/
        ``mceq_primary_model``/``mceq_density_model`` -- note v1_0's own
        ``external/mceq/config.py`` internal default primary model,
        ``"HillasGaisser H3a"``, disagrees with this actually-used
        value; this project follows what v1_0 actually ran with).
    ATMOSPHERE_DATA_DIR: Root directory this project's atmospheric flux
        tables (both Honda's and MCEq's -- see ``ATMOSPHERE_HONDA_FLUX_DIRNAME``/
        ``ATMOSPHERE_MCEQ_FLUX_DIRNAME`` below, plus the canonical
        ``honda/flux/honda_flux.csv`` surface table) are read from and
        generated into: ``<generated-data dir>/atmosphere``, the writable
        per-user directory of ``tpeanuts.data.generated_dir`` (relocated with
        ``$TPEANUTS_USER_DATA_DIR``), never the installed package. Shared by
        ``tpeanuts.source.atmosphere.honda``, ``.mceq``, ``.io`` and ``.model``.
    ATMOSPHERE_HONDA_RAW_DIR: Default search directory for the raw,
        externally-downloaded Honda/HKKM ``.d.gz`` table archive
        ``tpeanuts.source.atmosphere.honda.find_honda_data_dir`` looks in
        when no explicit directory or ``$HONDA_DATA_DIR`` is given:
        ``<datasets dir>/honda`` (``$TPEANUTS_DATA_DIR`` or the per-user
        cache). Raw third-party downloads, not shipped.
    ATMOSPHERE_MCEQ_INFO: Default verbosity passed to MCEq's own
        internal logger by ``MCEqConfig``/``generate_mceq_flux``.
    ATMOSPHERE_MCEQ_FLUX_DIRNAME: Default output directory name, under
        ``ATMOSPHERE_DATA_DIR``, ``generate_mceq_flux`` saves its
        per-angle CSV files into.
    ATMOSPHERE_MCEQ_X_GRID_MIN_GCM2, ATMOSPHERE_MCEQ_X_GRID_MAX_GCM2,
    ATMOSPHERE_MCEQ_X_GRID_N: Default slant-depth grid ``MCEqConfig``
        solves the cascade equations over (matches v1_0's own
        ``GridConfig`` defaults).
    ATMOSPHERE_MCEQ_X_OBS_GCM2: Default observation slant depth (sea
        level, vertical) ``MCEqConfig`` interpolates the solved flux to
        (matches v1_0's own ``GridConfig.X_obs_gcm2`` default).
    ATMOSPHERE_MCEQ_H_GRID_MIN_KM, ATMOSPHERE_MCEQ_H_GRID_MAX_KM,
    ATMOSPHERE_MCEQ_H_GRID_N: Default production-height grid
        ``MCEqConfig`` reconstructs ``f(h|E,alpha)`` over (matches
        v1_0's own ``GridConfig`` defaults).
    ATMOSPHERE_MCEQ_SMOOTHING_METHOD, ATMOSPHERE_MCEQ_SMOOTHING,
    ATMOSPHERE_MCEQ_GAUSSIAN_SIGMA: Default depth-smoothing method
        (``None``/``"none"``/``"spline"``/``"gaussian"``) and its
        strength, for ``MCEqConfig`` (matches v1_0's own
        ``SmoothingConfig`` defaults).
    ATMOSPHERE_MCEQ_POSITIVE_ONLY: Default clamp-to-nonnegative
        behaviour for ``dPhi/dX`` in ``MCEqConfig`` (matches v1_0's own
        ``SmoothingConfig.positive_only``).
    ATMOSPHERE_MCEQ_PARTICLES: Default particle-name mapping
        ``tpeanuts.source.atmosphere.mceq.generate_mceq_flux`` requests
        from MCEq per generated angle (matches v1_0's own
        ``external/mceq/config.py`` ``DEFAULT_FLAVOURS``).
    ATMOSPHERE_HONDA_SITE_CODE, ATMOSPHERE_HONDA_SEASON_CODE,
    ATMOSPHERE_HONDA_SOLAR, ATMOSPHERE_HONDA_MOUNTAIN,
    ATMOSPHERE_HONDA_ANGULAR_MODE,
    ATMOSPHERE_HONDA_AZIMUTH_AVERAGED_HEIGHT: Default Honda/HKKM table
        variant for ``tpeanuts.source.atmosphere.honda.HondaConfig``
        (matches v1_0's own ``external/honda/tables.py``
        ``HondaTableSelection()`` field defaults -- Frejus, all-year
        averaged, solar minimum, no mountain profile, azimuth-averaged
        flux and height tables; a *different* variant from the one
        already-generated ``honda_flux.csv`` this project ships, which
        was a one-off manual conversion of a specific 2006-2008,
        zenith+azimuth-binned file, not v1_0's own generator default).
    ATMOSPHERE_HONDA_FLUX_DIRNAME: Default output directory name, under
        ``ATMOSPHERE_DATA_DIR``,
        ``tpeanuts.source.atmosphere.honda.generate_honda_flux`` saves
        its per-angle CSV files into.
    ATMOSPHERE_HONDA_H_GRID_MIN_KM, ATMOSPHERE_HONDA_H_GRID_MAX_KM,
    ATMOSPHERE_HONDA_H_GRID_N: Default production-height grid
        ``HondaConfig`` reconstructs ``f(h|E,alpha)`` over (matches
        v1_0's own ``external/honda/generator.py``
        ``generate_flux_for_particle_angle`` default ``h_grid_km``).
    ATMOSPHERE_HONDA_PARTICLES: Default particle-name mapping
        ``tpeanuts.source.atmosphere.honda.generate_honda_flux`` builds
        a flux for per generated angle (same four flavours as
        ``ATMOSPHERE_MCEQ_PARTICLES``, kept as an independent constant
        since the two generators are independently configurable).
    SOLAR_PROVIDER: Default canonical provider for
        ``tpeanuts.source.solar.io.load_solar_production``/
        ``load_solar_flux`` (matches v1_0's own ``solar_provider``).
    SOLAR_SPECTRUM_PROVIDER: Default canonical provider for
        ``tpeanuts.source.solar.io.load_solar_spectrum`` -- independent of
        ``SOLAR_PROVIDER`` (matches v1_0's own ``solar_spectrum_provider``).
    SOLAR_PRODUCTION_FILENAME, SOLAR_FLUXES_FILENAME: Default production/
        flux table paths, relative to ``tpeanuts.data.solar``, used when a
        loader is given neither an explicit ``path`` nor a ``provider``
        (matches v1_0's own ``solar_production_filename``/
        ``solar_fluxes_filename``).
    SOLAR_DENSITY_FILENAME: Default solar electron/neutron density table
        path, relative to ``tpeanuts.data.solar``, used by
        ``tpeanuts.medium.solar.density.load_solar_density_table`` when
        given neither an explicit ``path`` nor a ``provider`` -- resolved
        under the same ``SOLAR_PROVIDER`` as production/flux (v1_0 uses a
        single ``solar_provider`` for all three, not a separate density
        provider).
    REACTOR_DATA_DIR: Root directory ``tpeanuts.source.reactor.io``'s loaders read the default reactor
        (Daya Bay's cores) spectra, fission fractions, thermal power and non-equilibrium correction from.
    DAYABAY_DATA_DIR: Root directory ``tpeanuts.experiments.dayabay.io``'s
        loaders read the Daya Bay data release's own cached CSVs from.
    BOREXINO_DATA_DIR: Root directory ``tpeanuts.experiments.borexino.io``'s
        loader reads the Borexino Nature 2018 observation CSV from.
    SNO_DATA_DIR: Root directory ``tpeanuts.experiments.sno.io``'s loaders
        read the real SNO Phase-I day/night observation CSVs from.
    ICECUBE_DATA_DIR: Root directory ``tpeanuts.experiments.icecube.io``'s
        loaders read the cached real IceCube DeepCore data release from.
    SNO_II_DATA_DIR: Root directory ``tpeanuts.experiments.sno_ii.io``'s
        loaders read the real SNO salt-phase (Phase II) observation/
        covariance/exposure/systematics CSVs from.
"""

from tpeanuts.data import dataset_dir, generated_dir, package_data_dir

FIT_MAX_ITER = 200
FIT_SCALE_MAX_RELATIVE = 10.0

LBFGS_LR = 1.0
LBFGS_TOLERANCE_GRAD = 1.0e-10
LBFGS_TOLERANCE_CHANGE = 1.0e-12
LBFGS_LINE_SEARCH_FN = "strong_wolfe"

ADAM_LR = 0.1
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1.0e-8
ADAM_WEIGHT_DECAY = 0.0

EARTH_NSTEPS = 100

EARTH_EXPOSURE_D1 = 0.0
EARTH_EXPOSURE_D2 = 365.0
EARTH_EXPOSURE_NS = 1000
EARTH_EXPOSURE_DAYNIGHT = None
EARTH_EXPOSURE_ELLIPTIC_TOL = 1.0e-12
EARTH_EXPOSURE_ANGLE_EPS = 1.0e-5
EARTH_EXPOSURE_NORMALIZED = False

ATMOSPHERE_SOURCE_DENSITY = "exponential"
ATMOSPHERE_RHO0_GCM3 = 1.225e-3
ATMOSPHERE_SCALE_HEIGHT_M = 8400.0
ATMOSPHERE_YE = 0.5
ATMOSPHERE_PYMSIS_DATE = "2026-05-10T12:00"
ATMOSPHERE_PYMSIS_LON_DEG = -0.38
ATMOSPHERE_PYMSIS_LAT_DEG = 39.47
ATMOSPHERE_PYMSIS_F107 = 150.0
ATMOSPHERE_PYMSIS_F107A = 150.0
ATMOSPHERE_PYMSIS_AP = 7.0
ATMOSPHERE_PYMSIS_VERSION = 2.1
ATMOSPHERE_NSTEPS = 599
ATMOSPHERE_PERTURBATIVE_SEGMENTS = 4
ATMOSPHERE_PERTURBATIVE_DEGREE = 3

# Data locations come from tpeanuts.data and are evaluated once, at import: set
# TPEANUTS_DATA_DIR / TPEANUTS_USER_DATA_DIR before importing tpeanuts to relocate them.
ATMOSPHERE_DATA_DIR = generated_dir() / "atmosphere"
ATMOSPHERE_HONDA_RAW_DIR = dataset_dir("honda")

ATMOSPHERE_MCEQ_INTERACTION_MODEL = "SIBYLL23D"
ATMOSPHERE_MCEQ_PRIMARY_MODEL = "PolyGonato"
ATMOSPHERE_MCEQ_DENSITY_MODEL = "CORSIKA"
ATMOSPHERE_MCEQ_INFO = False
ATMOSPHERE_MCEQ_FLUX_DIRNAME = "flux_mceq"
ATMOSPHERE_MCEQ_X_GRID_MIN_GCM2 = 1.0
ATMOSPHERE_MCEQ_X_GRID_MAX_GCM2 = 1030.0
ATMOSPHERE_MCEQ_X_GRID_N = 220
ATMOSPHERE_MCEQ_X_OBS_GCM2 = 1030.0
ATMOSPHERE_MCEQ_H_GRID_MIN_KM = 0.0
ATMOSPHERE_MCEQ_H_GRID_MAX_KM = 80.0
ATMOSPHERE_MCEQ_H_GRID_N = 300
ATMOSPHERE_MCEQ_SMOOTHING_METHOD = "spline"
ATMOSPHERE_MCEQ_SMOOTHING = 1.0e-4
ATMOSPHERE_MCEQ_GAUSSIAN_SIGMA = 2.0
ATMOSPHERE_MCEQ_POSITIVE_ONLY = True
ATMOSPHERE_MCEQ_PARTICLES = {"numu": "numu", "antinumu": "antinumu", "nue": "nue", "antinue": "antinue"}

ATMOSPHERE_HONDA_SITE_CODE = "frj"
ATMOSPHERE_HONDA_SEASON_CODE = "ally"
ATMOSPHERE_HONDA_SOLAR = "solmin"
ATMOSPHERE_HONDA_MOUNTAIN = False
ATMOSPHERE_HONDA_ANGULAR_MODE = "azimuth-averaged"
ATMOSPHERE_HONDA_AZIMUTH_AVERAGED_HEIGHT = True
ATMOSPHERE_HONDA_FLUX_DIRNAME = "flux_honda"
ATMOSPHERE_HONDA_H_GRID_MIN_KM = 0.0
ATMOSPHERE_HONDA_H_GRID_MAX_KM = 120.0
ATMOSPHERE_HONDA_H_GRID_N = 501
ATMOSPHERE_HONDA_PARTICLES = {"numu": "numu", "antinumu": "antinumu", "nue": "nue", "antinue": "antinue"}

SOLAR_PROVIDER = "zenodo"
SOLAR_SPECTRUM_PROVIDER = "legacy"
SOLAR_PRODUCTION_FILENAME = "zenodo/production/production_SF3_AGSS09.csv"
SOLAR_FLUXES_FILENAME = "zenodo/flux/fluxes_SF3_AGSS09.csv"
SOLAR_DENSITY_FILENAME = "zenodo/density/density_SF3_AGSS09.csv"

REACTOR_DATA_DIR = package_data_dir() / "reactor" / "dayabay"
DAYABAY_DATA_DIR = package_data_dir() / "experiments" / "dayabay"
BOREXINO_DATA_DIR = package_data_dir() / "experiments" / "borexino"
SNO_DATA_DIR = package_data_dir() / "experiments" / "sno"
ICECUBE_DATA_DIR = dataset_dir("icecube")  # downloaded, not packaged (64 MB)
SNO_II_DATA_DIR = package_data_dir() / "experiments" / "sno_ii"

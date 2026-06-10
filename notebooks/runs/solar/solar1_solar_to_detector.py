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
Run 3: solar-neutrino propagation from production to detector.

This runner dispatches one of three implementations:

    SIMULATION_MODE = "coherent"
        Fully coherent amplitude pipeline.

    SIMULATION_MODE = "incoherent"
        Torch implementation of the legacy incoherent solar workflow.

    SIMULATION_MODE = "legacy"
        Original NumPy/Numba peanuts implementation.

Each mode writes one torch file with solar, earth-arrival, detector-vs-eta,
and exposure-integrated detector probabilities.
"""



from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Literal

import torch


THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parents[3]


from tpeanuts.flux_propagation import (  # noqa: E402
    run_and_save_solar_to_detector_coherent,
    run_and_save_solar_to_detector_incoherent,
    run_and_save_solar_to_detector_legacypeanuts,
)
from tpeanuts.util.torch_util import _default_device, resolve_device  # noqa: E402


# ============================================================
# Simulation selector
# ============================================================

SimulationMode = Literal["coherent", "incoherent", "legacy", 'ALL']
SIMULATION_MODES = ["coherent", "incoherent", "legacy"]
SIMULATION_MODE: SimulationMode = "ALL"


# ============================================================
# Output
# ============================================================
DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_DATA_ROOT = Path(OUTPUT_ROOT / "data")
OUTPUT_DATA_SOLAR = Path(OUTPUT_DATA_ROOT / "solar")
OUTPUT_SOLARDETECTOR_ROOT = Path(OUTPUT_DATA_SOLAR / "detector")

OUTPUT_FILENAME = "solardetector.pt"
OVERWRITE = True
SAVE_DTYPE = torch.float32

# ============================================================
# Oscillation physics
# ============================================================

DM21_EV2 = 7.42e-5
DM3L_EV2 = 2.517e-3

THETA12 = 0.59
THETA13 = 0.15
THETA23 = 0.78
DELTA_CP = 1.20

ANTINU = False


# ============================================================
# solar production
# ============================================================

SOURCE = "8B"

# Used by the coherent pipeline. Options:
#   - "point": one production radius R/R_sun.
#   - "incoherent": coherent evolution for each production radius, followed by
#     the physical incoherent average over the SOURCE distribution.
#   - "coherent": experimental amplitude integral over SOURCE distribution.
#     This does not preserve normalization for an incoherent solar source.
coherent_PRODUCTION_MODE = "incoherent"
RHO0 = 0.08
INITIAL_STATE = "nue"

# Used by legacy peanuts. These explicit paths avoid relying on legacy default
# data locations.
LEGACY_solar_MODEL_FILE = str(PACKAGE_DIR / "data" / "peanuts" / "nudistr_b16_agss09.dat")
LEGACY_solar_flux_FILE = str(PACKAGE_DIR / "data" / "peanuts" / "fluxes_b16.dat")


# ============================================================
# Energy grid
# ============================================================

E_MIN_MEV = 1.0
E_MAX_MEV = 15.0
E_N = 21
E_GRID_MEV = torch.linspace(E_MIN_MEV, E_MAX_MEV, E_N, dtype=torch.float64)


# ============================================================
# earth and detector
# ============================================================

EARTH_DENSITY_FILE = str(PACKAGE_DIR / "data" / "density" / "earth_density.csv")
LEGACY_EARTH_DENSITY_FILE = str(PACKAGE_DIR / "data" / "peanuts" / "earth_density.csv")
TABULATED_earth_density = False

DETECTOR_DEPTH_M = 1000.0
DETECTOR_LATITUDE_RAD = 0.72
REUNITARIZE_earth = False


# ============================================================
# Sun-earth distance
# ============================================================

USE_SUN_EARTH_DISTANCE_TABLE = True
SUN_EARTH_DISTANCE_PATH = str(PACKAGE_DIR / "data" / "solar" / "sun_earth_distance.csv")
earth_DISTANCE_KM = None


# ============================================================
# Nadir exposure
# ============================================================

# For coherent/incoherent torch pipelines:
EXPOSURE_SOURCE = "math"
EXPOSURE_CSV_PATH = None
EXPOSURE_ANGLE = "Nadir"
EXPOSURE_DAYNIGHT = None
EXPOSURE_D1 = 0.0
EXPOSURE_D2 = 365.0
EXPOSURE_NS = 1000
EXPOSURE_CACHE_DIR = str(PACKAGE_DIR / "cache_exposure")
EXPOSURE_USE_CACHE = False

# For legacy peanuts:
LEGACY_EXPOSURE_NORMALIZED = True
LEGACY_EXPOSURE_FROM_FILE = None

# Optional manual eta grid. Keep None to build/integrate exposure.
ETA_GRID = None
INTEGRATE_EXPOSURE = True


# ============================================================
# Runtime
# ============================================================

DEVICE = "cuda:0"
COMPUTE_DTYPE = torch.float64
DEBUG = True


def output_dir_for_mode(mode: SimulationMode) -> str:
    return str(Path(OUTPUT_SOLARDETECTOR_ROOT) / mode)


def filename_for_mode(mode: SimulationMode) -> str:
    stem = Path(OUTPUT_FILENAME).stem
    suffix = Path(OUTPUT_FILENAME).suffix or ".pt"
    return f"{stem}_{mode}{suffix}"


def common_kwargs() -> dict:
    return {
        "E_MeV": E_GRID_MEV,
        "DeltamSq21": DM21_EV2,
        "DeltamSq3l": DM3L_EV2,
        "theta12": THETA12,
        "theta13": THETA13,
        "theta23": THETA23,
        "delta": DELTA_CP,
        "earth_distance_km": earth_DISTANCE_KM,
        "sun_earth_distance_path": SUN_EARTH_DISTANCE_PATH,
        "use_sun_earth_distance_table": USE_SUN_EARTH_DISTANCE_TABLE,
        "eta": ETA_GRID,
        "detector_depth_m": DETECTOR_DEPTH_M,
        "detector_latitude_rad": DETECTOR_LATITUDE_RAD,
        "exposure_d1": EXPOSURE_D1,
        "exposure_d2": EXPOSURE_D2,
        "exposure_ns": EXPOSURE_NS,
        "exposure_daynight": EXPOSURE_DAYNIGHT,
        "integrate_exposure": INTEGRATE_EXPOSURE,
        "antinu": ANTINU,
        "device": DEVICE,
        "dtype": COMPUTE_DTYPE,
        "debug": DEBUG,
    }


def coherent_kwargs() -> dict:
    kwargs = common_kwargs()
    kwargs.update(
        {
            "initial_state": INITIAL_STATE,
            "production_mode": coherent_PRODUCTION_MODE,
            "rho0": RHO0,
            "source": SOURCE if coherent_PRODUCTION_MODE != "point" else None,
            "earth_density_file": EARTH_DENSITY_FILE,
            "tabulated_earth_density": TABULATED_earth_density,
            "exposure_source": EXPOSURE_SOURCE,
            "exposure_csv_path": EXPOSURE_CSV_PATH,
            "exposure_angle": EXPOSURE_ANGLE,
            "exposure_cache_dir": EXPOSURE_CACHE_DIR,
            "exposure_use_cache": EXPOSURE_USE_CACHE,
            "reunitarize_earth": REUNITARIZE_earth,
        }
    )
    return kwargs


def incoherent_kwargs() -> dict:
    kwargs = common_kwargs()
    kwargs.update(
        {
            "source": SOURCE,
            "earth_density_file": EARTH_DENSITY_FILE,
            "tabulated_earth_density": TABULATED_earth_density,
            "exposure_source": EXPOSURE_SOURCE,
            "exposure_csv_path": EXPOSURE_CSV_PATH,
            "exposure_angle": EXPOSURE_ANGLE,
            "exposure_cache_dir": EXPOSURE_CACHE_DIR,
            "exposure_use_cache": EXPOSURE_USE_CACHE,
            "reunitarize_earth": REUNITARIZE_earth,
        }
    )
    return kwargs


def legacy_kwargs() -> dict:
    kwargs = common_kwargs()
    kwargs.update(
        {
            "source": SOURCE,
            "solar_model_file": LEGACY_solar_MODEL_FILE,
            "solar_flux_file": LEGACY_solar_flux_FILE,
            "earth_density_file": LEGACY_EARTH_DENSITY_FILE,
            "tabulated_earth_density": TABULATED_earth_density,
            "exposure_normalized": LEGACY_EXPOSURE_NORMALIZED,
            "exposure_from_file": LEGACY_EXPOSURE_FROM_FILE,
            "exposure_angle": EXPOSURE_ANGLE,
        }
    )
    return kwargs


def run_selected_mode(mode: SimulationMode):
    mode = mode.lower()
    if mode not in ("coherent", "incoherent", "legacy"):
        raise ValueError("SIMULATION_MODE must be 'coherent', 'incoherent' or 'legacy'.")

    output_dir = output_dir_for_mode(mode)
    filename = filename_for_mode(mode)

    print("\n" + "=" * 80)
    print(f"RUN 3 solar DETECTOR PROPAGATION | mode={mode}")
    print("=" * 80)
    print(f"E grid       : {E_GRID_MEV.numel()} points [{E_GRID_MEV[0].item():.3g}, {E_GRID_MEV[-1].item():.3g}] MeV")
    print(f"source       : {SOURCE}")
    print(f"output       : {Path(output_dir) / filename}")
    print(f"device       : {resolve_device(DEVICE)}")
    print(f"exposure ns  : {EXPOSURE_NS}")

    t0 = time.perf_counter()

    if mode == "coherent":
        result = run_and_save_solar_to_detector_coherent(
            output_dir,
            filename=filename,
            overwrite=OVERWRITE,
            save_dtype=SAVE_DTYPE,
            **coherent_kwargs(),
        )
    elif mode == "incoherent":
        result = run_and_save_solar_to_detector_incoherent(
            output_dir,
            filename=filename,
            overwrite=OVERWRITE,
            save_dtype=SAVE_DTYPE,
            **incoherent_kwargs(),
        )
    else:
        result = run_and_save_solar_to_detector_legacypeanuts(
            output_dir,
            filename=filename,
            overwrite=OVERWRITE,
            save_dtype=SAVE_DTYPE,
            **legacy_kwargs(),
        )

    elapsed = time.perf_counter() - t0

    print("\nFinished.")
    print(f"elapsed      : {elapsed:.2f} s")
    print(f"saved        : {result['output_path']}")

    p_int = result.get("detector_probabilities_integrated")
    if torch.is_tensor(p_int):
        print(f"Pdet int     : shape={tuple(p_int.shape)}")
        print(f"Pdet sum min : {torch.min(p_int.sum(dim=-1)).item():.6g}")
        print(f"Pdet sum max : {torch.max(p_int.sum(dim=-1)).item():.6g}")

    return result


if __name__ == "__main__":
    if SIMULATION_MODE == 'ALL':
        for mode in SIMULATION_MODES:
            run_selected_mode(mode)
    else:
        run_selected_mode(SIMULATION_MODE)

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
Run 1 Honda: generate height-differential atmospheric flux files.

The output contract matches the MCEq Run 1 files: one torch file per particle
and angle, with phi_Eh, phi_E_obs, f_Eh, E_grid_GeV, h_grid_km and theta_deg.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parents[3]


from tpeanuts.external.honda import HondaTableSelection  # noqa: E402
from tpeanuts.external.honda.generator import generate_flux_for_particles_angle_grid  # noqa: E402
from tpeanuts.io.io_atmosphere import load_directory, OutputConfig  # noqa: E402
from tpeanuts.util.parallel import ParallelConfig
from tpeanuts.atmosphere.geometry import detector_alpha_to_surface_theta  # noqa: E402
from tpeanuts.util.torch_util import _default_device, resolve_device  # noqa: E402
from tpeanuts.util.type import _as_tensor  # noqa: E402


# ============================================================
# Input / output
# ============================================================

DEFAULT_HONDA_DATA_DIR = Path(r"G:\Mi unidad\03.Codigo\034.TFM.UV\External\Honda")
HONDA_DATA_DIR = Path(os.environ.get("HONDA_DATA_DIR", DEFAULT_HONDA_DATA_DIR))

DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_DATA_ROOT = Path(OUTPUT_ROOT / "data")
OUTPUT_ATMOSPHERE_ROOT = OUTPUT_DATA_ROOT / "atmosphere"
OUTPUT_HONDA_ROOT = OUTPUT_ATMOSPHERE_ROOT / "honda"
OUTPUT_DIR = str(OUTPUT_HONDA_ROOT / "honda_flux_diff_002")

OUTPUT_FILENAME = "diff_flux.pt"
SAVE_DTYPE = torch.float32
OVERWRITE = False


# ============================================================
# Honda table selection
# ============================================================

HONDA_SELECTION = HondaTableSelection(
    site_code="frj",
    season_code="ally",
    solar="solmin",
    mountain=False,
    angular_mode="azimuth-averaged",
    azimuth_averaged_height=True,
)


# ============================================================
# Particles
# ============================================================

# Honda provides e/mu neutrino and antineutrino fluxes. Tau flavours are saved
# as zero source fluxes so downstream pipelines see the same six-flavour
# inventory as the MCEq production run.
PARTICLES = {
    "nue": "nue",
    "antinue": "antinue",
    "numu": "numu",
    "antinumu": "antinumu",
    "nutau": "nutau",
    "antinutau": "antinutau",
}


# ============================================================
# Angle convention
# ============================================================

# Honda/HKKM tables are binned in cosZ. For the TPeanuts atmospheric source
# files we use the MCEq-compatible surface angle:
#
#     theta_surface = arccos(cosZ)
#
# theta_surface = 0 deg is vertically downward and 90 deg is horizontal.
# If USE_DETECTOR_ALPHA_GRID is enabled, detector alpha is converted to the
# surface theta with the same geometry used by the MCEq generator.
USE_HONDA_COSZ_GRID = True
USE_DETECTOR_ALPHA_GRID = False

HONDA_COSZ_CENTERS = torch.linspace(0.95, 0.05, 10)

ALPHA_MIN = 0
ALPHA_MAX = 180
ALPHA_N = 11
ALPHA_DETECTOR_GRID_DEG = torch.linspace(ALPHA_MIN, ALPHA_MAX, ALPHA_N)

THETA_MIN = 0
THETA_MAX = 89.5
THETA_N = 37
THETA_SURFACE_GRID_DEG = torch.linspace(THETA_MIN, THETA_MAX, THETA_N)

DETECTOR_DEPTH_M = 1000.0
SURFACE_THETA_MAX_DEG = 89.999


# ============================================================
# Numerical grids
# ============================================================

# None means use the native Honda flux energy grid. The height grid is in km,
# matching TPeanuts and MCEq Run 1 output conventions.
ENERGY_GRID_GEV = None
H_GRID_MIN = 0.0
H_GRID_MAX = 120.0
H_GRID_N = 501
H_GRID_KM = torch.linspace(H_GRID_MIN, H_GRID_MAX, H_GRID_N)


# ============================================================
# Runtime
# ============================================================

GENERATION_DEVICE = "cpu"
COMPUTE_DTYPE = torch.float64

PARALLEL = False
N_JOBS = 4
PARALLEL_BACKEND = "loky"

STACK_AFTER_GENERATION = True
STACK_DEVICE = _default_device
STACK_DTYPE = torch.float64
STACK_GROUP_BY = "particle"

SAVE = True
SKIP_EXISTING = True
DEBUG = True


def prepared_angle_grids():
    if USE_HONDA_COSZ_GRID:
        cosz_grid = _as_tensor(HONDA_COSZ_CENTERS, device="cpu", dtype=torch.float64).reshape(-1)
        valid = (cosz_grid > 0.0) & (cosz_grid <= 1.0)
        if not torch.all(valid):
            dropped = int((~valid).sum().item())
            print(f"Dropping {dropped} Honda cosZ centres outside (0, 1].")
            cosz_grid = cosz_grid[valid]

        theta_grid = torch.rad2deg(torch.acos(torch.clamp(cosz_grid, -1.0, 1.0)))
        valid_theta = (theta_grid >= 0.0) & (theta_grid < SURFACE_THETA_MAX_DEG)
        theta_grid = theta_grid[valid_theta]
        theta_grid = theta_grid[torch.argsort(theta_grid)]
        return None, theta_grid, "honda-cosz theta"

    if USE_DETECTOR_ALPHA_GRID:
        alpha_grid = _as_tensor(ALPHA_DETECTOR_GRID_DEG, device="cpu", dtype=torch.float64).reshape(-1)
        theta_grid = detector_alpha_to_surface_theta(
            alpha_grid,
            detector_depth_m=DETECTOR_DEPTH_M,
            device="cpu",
            dtype=torch.float64,
        ).reshape(-1)

        valid = (theta_grid >= 0.0) & (theta_grid < SURFACE_THETA_MAX_DEG)
        if not torch.all(valid):
            dropped = int((~valid).sum().item())
            print(
                f"Dropping {dropped} detector-alpha values whose surface "
                f"theta is outside [0, {SURFACE_THETA_MAX_DEG}) deg."
            )
            alpha_grid = alpha_grid[valid]

        alpha_grid = alpha_grid[torch.argsort(alpha_grid)]
        return alpha_grid, None, "alpha"

    theta_grid = _as_tensor(THETA_SURFACE_GRID_DEG, device="cpu", dtype=torch.float64).reshape(-1)
    valid = (theta_grid >= 0.0) & (theta_grid < SURFACE_THETA_MAX_DEG)
    theta_grid = theta_grid[valid]
    theta_grid = theta_grid[torch.argsort(theta_grid)]
    return None, theta_grid, "theta"


def build_configs():
    output_config = OutputConfig(
        output_dir=OUTPUT_DIR,
        filename=OUTPUT_FILENAME,
        dtype=SAVE_DTYPE,
        compressed=True,
        overwrite=OVERWRITE,
        save_intermediate=False,
    )
    parallel_config = ParallelConfig(
        parallel=PARALLEL,
        n_jobs=N_JOBS,
        backend=PARALLEL_BACKEND,
    )

    output_config.validate()
    parallel_config.validate()
    return output_config, parallel_config


def main():
    output_config, parallel_config = build_configs()
    alpha_grid, theta_grid, angle_mode = prepared_angle_grids()

    generation_device = resolve_device(GENERATION_DEVICE)
    stack_device = resolve_device(STACK_DEVICE)

    print("\nHonda flux generation")
    print(f"Honda data dir   : {HONDA_DATA_DIR}")
    print(f"Output directory : {OUTPUT_DIR}")
    print(f"Selection        : {HONDA_SELECTION}")
    print(f"Particles        : {PARTICLES}")
    print(f"Detector depth m : {DETECTOR_DEPTH_M}")
    print(f"Angle mode       : {angle_mode}")
    print(f"Generation device: {generation_device}")
    print(f"Compute dtype    : {COMPUTE_DTYPE}")
    print(f"Save dtype       : {SAVE_DTYPE}")

    results = generate_flux_for_particles_angle_grid(
        particles=PARTICLES,
        alpha_grid_deg=alpha_grid,
        theta_grid_deg=theta_grid,
        detector_depth_m=DETECTOR_DEPTH_M,
        honda_data_dir=HONDA_DATA_DIR,
        selection=HONDA_SELECTION,
        energy_grid_GeV=ENERGY_GRID_GEV,
        h_grid_km=H_GRID_KM,
        output_config=output_config,
        parallel_config=parallel_config,
        save=SAVE,
        skip_existing=SKIP_EXISTING,
        device=generation_device,
        dtype=COMPUTE_DTYPE,
        debug=DEBUG,
    )

    n_files = sum(len(item["results"]) for item in results.values())
    print(f"\nFinished. Generated/visited {n_files} particle-angle results.")

    stacked = None
    if STACK_AFTER_GENERATION:
        print(
            "\nLoading generated files as batched tensors "
            f"(device={stack_device}, dtype={STACK_DTYPE}, group_by={STACK_GROUP_BY})"
        )

        stacked = load_directory(
            OUTPUT_DIR,
            map_location="cpu",
            dtype=STACK_DTYPE,
            device=stack_device,
            group_by=STACK_GROUP_BY,
            verbose=DEBUG,
        )

        for particle, data in stacked.items():
            print(
                f"Stacked {particle:12s}: "
                f"phi_E_theta_h={tuple(data['phi_E_theta_h'].shape)}, "
                f"device={data['phi_E_theta_h'].device}"
            )

    return {
        "generation_results": results,
        "stacked": stacked,
    }


if __name__ == "__main__":
    main()

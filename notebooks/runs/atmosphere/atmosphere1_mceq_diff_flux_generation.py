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
Run 1: generate mceq height-differential atmospheric flux files.

This script is intentionally parameter-heavy: edit the block below and run the
file to generate one torch file per particle and angle.

mceq itself is an external CPU/Python solver. The useful batching knobs here
are therefore:

    - CPU job batching: how many independent particle-angle mceq jobs are
      dispatched together and how many workers execute them.
    - Torch post-processing device: where smoothing/interpolation/profile
      reconstruction runs after each mceq solve.
    - Stacked dataset loading: whether generated files are loaded as batched
      tensors, ready for the flux_propagation pipeline.
"""



from __future__ import annotations

import os
from pathlib import Path

import torch


THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parents[3]


from tpeanuts.external.mceq.config import (  # noqa: E402
    GridConfig,
    MCEqModelConfig,
    SmoothingConfig,
)
from tpeanuts.io.io_atmosphere import load_directory, OutputConfig  # noqa: E402
from tpeanuts.util.parallel import ParallelConfig
from tpeanuts.atmosphere.geometry import detector_alpha_to_surface_theta  # noqa: E402
from tpeanuts.external.mceq.generator import generate_flux_for_particles_angle_grid  # noqa: E402

from tpeanuts.util.torch_util import _default_device, resolve_device
from tpeanuts.util.type import _as_tensor

# ============================================================
# Output
# ============================================================
DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_DATA_ROOT = Path(OUTPUT_ROOT / "data")
OUTPUT_ATMOSPHERE_ROOT = Path(OUTPUT_DATA_ROOT / "atmosphere")

OUTPUT_MCEQ_ROOT = Path(OUTPUT_ATMOSPHERE_ROOT / "mceq")
OUTPUT_DIR = str(OUTPUT_MCEQ_ROOT / "mceq_flux_diff_002")

OUTPUT_FILENAME = "diff_flux.pt"
SAVE_DTYPE = torch.float32
OVERWRITE = False
SAVE_INTERMEDIATE = True


# ============================================================
# Particles
# ============================================================

# Keys are user-facing names. Values are mceq solution names.
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

# Use one of the three grids:
#   - HONDA_COSZ_CENTERS: Honda/HKKM cos(theta_z) bin centres, converted
#     directly to the mceq surface theta grid.
#   - ALPHA_DETECTOR_GRID_DEG: angle at detector. It is converted to theta.
#   - THETA_SURFACE_GRID_DEG: angle at earth-surface intersection used by mceq.
#
# Honda's 20-bin zenith grid has centres from -0.95 to +0.95 in steps of 0.10.
# MCEq receives the surface zenith angle in the range [0, 90) deg, so this
# source-generation script uses the positive/down-going Honda centres by
# default. The negative/up-going Honda branch is a detector-propagation geometry
# question and is not a separate MCEq production angle here.
USE_HONDA_COSZ_GRID = True
USE_DETECTOR_ALPHA_GRID = False

HONDA_COSZ_CENTERS = torch.linspace(0.95, 0.05, 10)

ALPHA_MIN = 0
ALPHA_MAX = 180
ALPHA_N = 11

THETA_MIN = 0
THETA_MAX = 89.5
THETA_N = 37

ALPHA_DETECTOR_GRID_DEG = torch.linspace(ALPHA_MIN, ALPHA_MAX, ALPHA_N)
THETA_SURFACE_GRID_DEG = torch.linspace(THETA_MIN, THETA_MAX, THETA_N)

DETECTOR_DEPTH_M = 1000
SURFACE_THETA_MAX_DEG = 89.999


# ============================================================
# mceq physical models
# ============================================================

INTERACTION_MODEL = "QGSJETII04"
PRIMARY_MODEL = "HillasGaisser H3a"
density_MODEL = "ICECUBE"
MCEQ_INFO = False
PRIMARY_MODEL_FALLBACK_TO_DEFAULT = True


# ============================================================
# Numerical grids
# ============================================================

X_OBS_GCM2 = 1030.0
X_GRID_MIN = 1.0
X_GRID_MAX = X_OBS_GCM2
X_GRID_N   = 101
H_GRID_MIN = 0.0
H_GRID_MAX = 120.0
H_GRID_N   = 501

X_GRID_GCM2 = torch.linspace(X_GRID_MIN, X_GRID_MAX, X_GRID_N)
H_GRID_KM = torch.linspace(H_GRID_MIN, H_GRID_MAX, H_GRID_N)


# ============================================================
# Smoothing and derivative extraction
# ============================================================

SMOOTHING_METHOD = "spline"  # None, "none", "gaussian", "spline"
SMOOTHING = 1.0e-4
GAUSSIAN_SIGMA = 2.0
POSITIVE_ONLY = True


# ============================================================
# Runtime
# ============================================================

# Generation is CPU-bound because mceq is CPU-bound. When PARALLEL=True, using
# "cpu" here avoids multiple worker processes competing for the same CUDA
# context. Set to _default_device or "cuda" only when you intentionally want
# Torch post-processing inside each job to run on GPU.
GENERATION_DEVICE = "cpu"
COMPUTE_DTYPE = torch.float64

PARALLEL = True
N_JOBS = 8
PARALLEL_BACKEND = "loky"

# Chunking controls memory and scheduling granularity.
# None means dispatch the full grid in one call.
ANGLE_CHUNK_SIZE = 6
PARTICLE_CHUNK_SIZE = None

# After generation, optionally load all .pt files as stacked tensors.
# This is the part that benefits most directly from the batched Torch/CUDA
# adaptations in mceq/flux_propagation.
STACK_AFTER_GENERATION = True
STACK_DEVICE = _default_device
STACK_DTYPE = torch.float64
STACK_GROUP_BY = "particle"

SAVE = True
SKIP_EXISTING = True
DEBUG = True


def chunk_sequence(values, chunk_size):
    values = list(values)

    if chunk_size is None or chunk_size <= 0:
        yield values
        return

    for start in range(0, len(values), int(chunk_size)):
        yield values[start:start + int(chunk_size)]


def chunk_tensor(values, chunk_size):
    values_t = _as_tensor(values, device="cpu", dtype=torch.float64).reshape(-1)

    if chunk_size is None or chunk_size <= 0:
        yield values_t
        return

    for start in range(0, values_t.numel(), int(chunk_size)):
        yield values_t[start:start + int(chunk_size)]


def prepared_angle_grids():
    if USE_HONDA_COSZ_GRID:
        cosz_grid = _as_tensor(
            HONDA_COSZ_CENTERS,
            device="cpu",
            dtype=torch.float64,
        ).reshape(-1)

        valid_cosz = (cosz_grid > 0.0) & (cosz_grid <= 1.0)
        if not torch.all(valid_cosz):
            dropped = int((~valid_cosz).sum().item())
            print(
                f"Dropping {dropped} Honda cosZ centres outside "
                "(0, 1], because mceq source generation expects "
                "surface theta in [0, 90) deg."
            )
            cosz_grid = cosz_grid[valid_cosz]

        theta_grid = torch.rad2deg(torch.acos(torch.clamp(cosz_grid, -1.0, 1.0)))

        valid_theta = (theta_grid >= 0.0) & (theta_grid < SURFACE_THETA_MAX_DEG)
        if not torch.all(valid_theta):
            dropped = int((~valid_theta).sum().item())
            print(
                f"Dropping {dropped} Honda-derived theta values outside "
                f"[0, {SURFACE_THETA_MAX_DEG}) deg."
            )
            theta_grid = theta_grid[valid_theta]

        theta_grid = theta_grid[torch.argsort(theta_grid)]

        return None, theta_grid, theta_grid

    if USE_DETECTOR_ALPHA_GRID:
        alpha_grid = _as_tensor(
            ALPHA_DETECTOR_GRID_DEG,
            device="cpu",
            dtype=torch.float64,
        ).reshape(-1)

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
            theta_grid = theta_grid[valid]

        order = torch.argsort(alpha_grid)
        alpha_grid = alpha_grid[order]
        theta_grid = theta_grid[order]

        return alpha_grid, None, theta_grid

    theta_grid = _as_tensor(
        THETA_SURFACE_GRID_DEG,
        device="cpu",
        dtype=torch.float64,
    ).reshape(-1)

    valid = (theta_grid >= 0.0) & (theta_grid < SURFACE_THETA_MAX_DEG)
    if not torch.all(valid):
        dropped = int((~valid).sum().item())
        print(
            f"Dropping {dropped} theta values outside "
            f"[0, {SURFACE_THETA_MAX_DEG}) deg."
        )
        theta_grid = theta_grid[valid]

    theta_grid = theta_grid[torch.argsort(theta_grid)]

    return None, theta_grid, theta_grid


def merge_generation_results(accumulator, partial):
    for flavour_name, flavour_result in partial.items():
        if flavour_name not in accumulator:
            accumulator[flavour_name] = dict(flavour_result)
            accumulator[flavour_name]["results"] = dict(flavour_result["results"])
            continue

        accumulator[flavour_name]["results"].update(flavour_result["results"])


def build_configs():
    alpha_grid, theta_surface_grid, theta_config_grid = prepared_angle_grids()

    model_config = MCEqModelConfig(
        interaction_model=INTERACTION_MODEL,
        primary_model=PRIMARY_MODEL,
        density_model=density_MODEL,
        info=MCEQ_INFO,
    )

    try:
        model_config.validate()
    except ValueError as exc:
        if (
            PRIMARY_MODEL_FALLBACK_TO_DEFAULT
            and PRIMARY_MODEL is not None
            and "primary_model" in str(exc)
        ):
            print(
                f"Primary model '{PRIMARY_MODEL}' is not available in this "
                "environment. Falling back to mceq default primary_model=None."
            )
            model_config = MCEqModelConfig(
                interaction_model=INTERACTION_MODEL,
                primary_model=None,
                density_model=density_MODEL,
                info=MCEQ_INFO,
            )
            model_config.validate()
        else:
            raise

    grid_config = GridConfig(
        theta_grid_deg=theta_config_grid,
        X_grid_gcm2=X_GRID_GCM2,
        h_grid_km=H_GRID_KM,
        X_obs_gcm2=X_OBS_GCM2,
    )

    smoothing_config = SmoothingConfig(
        method=SMOOTHING_METHOD,
        smoothing=SMOOTHING,
        gaussian_sigma=GAUSSIAN_SIGMA,
        positive_only=POSITIVE_ONLY,
    )

    output_config = OutputConfig(
        output_dir=OUTPUT_DIR,
        filename=OUTPUT_FILENAME,
        dtype=SAVE_DTYPE,
        compressed=True,
        overwrite=OVERWRITE,
        save_intermediate=SAVE_INTERMEDIATE,
    )

    parallel_config = ParallelConfig(
        parallel=PARALLEL,
        n_jobs=N_JOBS,
        backend=PARALLEL_BACKEND,
    )

    grid_config.validate()
    smoothing_config.validate()
    output_config.validate()
    parallel_config.validate()

    return (
        model_config,
        grid_config,
        smoothing_config,
        output_config,
        parallel_config,
        alpha_grid,
        theta_surface_grid,
    )


def main():
    (
        model_config,
        grid_config,
        smoothing_config,
        output_config,
        parallel_config,
        alpha_grid,
        theta_surface_grid,
    ) = build_configs()

    generation_device = resolve_device(GENERATION_DEVICE)
    stack_device = resolve_device(STACK_DEVICE)

    print("\nmceq flux generation")
    print(f"Output directory : {OUTPUT_DIR}")
    print(f"Particles        : {PARTICLES}")
    print(f"Detector depth m : {DETECTOR_DEPTH_M}")
    print(f"Parallel         : {PARALLEL} (n_jobs={N_JOBS}, backend={PARALLEL_BACKEND})")
    print(f"Generation device: {generation_device}")
    print(f"Compute dtype    : {COMPUTE_DTYPE}")
    print(f"Save dtype       : {SAVE_DTYPE}")
    print(f"Angle chunk size : {ANGLE_CHUNK_SIZE}")
    print(f"Particle chunks  : {PARTICLE_CHUNK_SIZE}")

    angle_grid = alpha_grid if USE_DETECTOR_ALPHA_GRID else theta_surface_grid
    angle_mode = (
        "alpha"
        if USE_DETECTOR_ALPHA_GRID
        else "honda-cosz theta"
        if USE_HONDA_COSZ_GRID
        else "theta"
    )
    particle_items = list(PARTICLES.items())
    results = {}

    angle_chunks = list(chunk_tensor(angle_grid, ANGLE_CHUNK_SIZE))
    particle_chunks = list(chunk_sequence(particle_items, PARTICLE_CHUNK_SIZE))
    n_dispatches = len(angle_chunks) * len(particle_chunks)
    dispatch_idx = 0

    for particle_chunk in particle_chunks:
        particle_dict = dict(particle_chunk)

        for angle_chunk in angle_chunks:
            dispatch_idx += 1

            if DEBUG:
                print(
                    f"\nDispatch {dispatch_idx}/{n_dispatches}: "
                    f"{len(particle_dict)} particles x {angle_chunk.numel()} "
                    f"{angle_mode} values"
                )

            partial = generate_flux_for_particles_angle_grid(
                particles=particle_dict,
                alpha_grid_deg=angle_chunk if USE_DETECTOR_ALPHA_GRID else None,
                theta_grid_deg=None if USE_DETECTOR_ALPHA_GRID else angle_chunk,
                detector_depth_m=DETECTOR_DEPTH_M,
                model_config=model_config,
                grid_config=grid_config,
                smoothing_config=smoothing_config,
                output_config=output_config,
                parallel_config=parallel_config,
                save=SAVE,
                skip_existing=SKIP_EXISTING,
                device=generation_device,
                dtype=COMPUTE_DTYPE,
                debug=DEBUG,
            )

            merge_generation_results(results, partial)

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
            phi_shape = tuple(data["phi_E_theta_h"].shape)
            print(
                f"Stacked {particle:12s}: phi_E_theta_h={phi_shape}, "
                f"device={data['phi_E_theta_h'].device}"
            )

    return {
        "generation_results": results,
        "stacked": stacked,
    }


if __name__ == "__main__":
    main()

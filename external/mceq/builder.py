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
High-level builders for atmospheric height-flux datasets.

This module orchestrates the full pipeline:

    theta
        -> profile reconstruction
        -> Phi(E,h)
        -> flavour aggregation
        -> optional parallel execution

This module should not contain low-level physics operations.
"""



from __future__ import annotations

import time
from typing import Dict, Optional, Union

import torch
from tqdm import tqdm

from tpeanuts.util.type import _as_tensor
from tpeanuts.util.torch_util import _default_device

from tpeanuts.external.mceq.config import (
    RunConfig,
    MCEqModelConfig,
    GridConfig,
    SmoothingConfig,
)
from tpeanuts.io.io_atmosphere import (
    OutputConfig,
    save_phi_Eh_theta_result,
)
from tpeanuts.util.parallel import ParallelConfig

from tpeanuts.external.mceq.profiles import (
    production_profiles_all_energies_from_flux_gradient,
)

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None
    delayed = None


TensorLike = Union[float, int, torch.Tensor]


# ============================================================
# Internal utilities
# ============================================================

def split_run_config(config: RunConfig):
    return (
        config.model,
        config.grid,
        config.smoothing,
        config.output,
        config.parallel,
    )


def validate_particle_name(particle: str) -> None:
    if not isinstance(particle, str):
        raise TypeError("particle must be a string.")

    if particle.strip() == "":
        raise ValueError("particle cannot be empty.")


# ============================================================
# Single theta builder
# ============================================================

@torch.no_grad()
def build_theta_result(
    theta_deg: TensorLike,
    particle: str,
    model_config: Optional[MCEqModelConfig] = None,
    grid_config: Optional[GridConfig] = None,
    smoothing_config: Optional[SmoothingConfig] = None,
    *,
    save: bool = False,
    flavour_name: Optional[str] = None,
    output_config: Optional[OutputConfig] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
):
    dev = _default_device(device)

    validate_particle_name(particle)

    t0 = time.perf_counter()

    result = production_profiles_all_energies_from_flux_gradient(
        theta_deg=theta_deg,
        particle=particle,
        model_config=model_config,
        grid_config=grid_config,
        smoothing_config=smoothing_config,
        device=dev,
        dtype=dtype,
    )

    elapsed = time.perf_counter() - t0

    result["particle"] = particle
    result["flavour_name"] = flavour_name
    result["build_time_sec"] = elapsed

    if save:

        if output_config is None:
            raise ValueError(
                "output_config must be provided when save=True."
            )

        output_path = save_phi_Eh_theta_result(
            result=result,
            output_config=output_config,
            flavour_name=flavour_name,
        )

        result["output_path"] = output_path

    return result


# ============================================================
# One flavour / all theta
# ============================================================

@torch.no_grad()
def build_phi_E_theta_h_for_particle(
    theta_grid_deg: TensorLike,
    particle: str,
    model_config: Optional[MCEqModelConfig] = None,
    grid_config: Optional[GridConfig] = None,
    smoothing_config: Optional[SmoothingConfig] = None,
    output_config: Optional[OutputConfig] = None,
    *,
    flavour_name: Optional[str] = None,
    save: bool = True,
    show_progress: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
):
    dev = _default_device(device)

    validate_particle_name(particle)

    theta_grid = _as_tensor(
        theta_grid_deg,
        device="cpu",
        dtype=torch.float64,
    ).reshape(-1)

    iterator = theta_grid

    if show_progress:
        iterator = tqdm(
            theta_grid,
            desc=f"Building {particle}",
        )

    theta_results = {}

    for theta in iterator:

        theta_float = float(theta.item())

        result = build_theta_result(
            theta_deg=theta_float,
            particle=particle,
            model_config=model_config,
            grid_config=grid_config,
            smoothing_config=smoothing_config,
            save=save,
            flavour_name=flavour_name,
            output_config=output_config,
            device=dev,
            dtype=dtype,
        )

        theta_results[theta_float] = result

    return {
        "particle": particle,
        "flavour_name": flavour_name,
        "theta_grid_deg": theta_grid,
        "results": theta_results,
    }


# ============================================================
# Parallel theta execution
# ============================================================

def _parallel_theta_worker(
    theta_deg,
    particle,
    model_config,
    grid_config,
    smoothing_config,
    output_config,
    flavour_name,
    save,
    dtype,
):
    return build_theta_result(
        theta_deg=theta_deg,
        particle=particle,
        model_config=model_config,
        grid_config=grid_config,
        smoothing_config=smoothing_config,
        save=save,
        flavour_name=flavour_name,
        output_config=output_config,
        dtype=dtype,
    )


@torch.no_grad()
def build_phi_E_theta_h_for_particle_parallel(
    theta_grid_deg: TensorLike,
    particle: str,
    model_config: Optional[MCEqModelConfig] = None,
    grid_config: Optional[GridConfig] = None,
    smoothing_config: Optional[SmoothingConfig] = None,
    output_config: Optional[OutputConfig] = None,
    parallel_config: Optional[ParallelConfig] = None,
    *,
    flavour_name: Optional[str] = None,
    save: bool = True,
    dtype: torch.dtype = torch.float64,
):
    if Parallel is None:
        raise ImportError(
            "joblib is required for parallel execution."
        )

    theta_grid = _as_tensor(
        theta_grid_deg,
        device="cpu",
        dtype=torch.float64,
    ).reshape(-1)

    if parallel_config is None:
        parallel_config = ParallelConfig()

    parallel_config.validate()

    results_list = Parallel(
        n_jobs=parallel_config.n_jobs,
        backend=parallel_config.backend,
    )(
        delayed(_parallel_theta_worker)(
            theta_deg=float(theta.item()),
            particle=particle,
            model_config=model_config,
            grid_config=grid_config,
            smoothing_config=smoothing_config,
            output_config=output_config,
            flavour_name=flavour_name,
            save=save,
            dtype=dtype,
        )
        for theta in theta_grid
    )

    theta_results = {}

    for result in results_list:

        theta = float(result["theta_deg"].item())

        theta_results[theta] = result

    return {
        "particle": particle,
        "flavour_name": flavour_name,
        "theta_grid_deg": theta_grid,
        "results": theta_results,
    }


# ============================================================
# All flavours
# ============================================================

@torch.no_grad()
def build_all_flavours(
    config: RunConfig,
    *,
    save: bool = True,
    show_progress: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
):
    dev = _default_device(device)

    config.validate()

    (
        model_config,
        grid_config,
        smoothing_config,
        output_config,
        parallel_config,
    ) = split_run_config(config)

    all_results = {}

    flavour_iterator = config.flavours.items()

    if show_progress:
        flavour_iterator = tqdm(
            flavour_iterator,
            desc="Flavours",
        )

    for flavour_name, particle in flavour_iterator:

        t0 = time.perf_counter()

        if parallel_config.parallel:

            flavour_result = build_phi_E_theta_h_for_particle_parallel(
                theta_grid_deg=grid_config.theta_grid_deg,
                particle=particle,
                model_config=model_config,
                grid_config=grid_config,
                smoothing_config=smoothing_config,
                output_config=output_config,
                parallel_config=parallel_config,
                flavour_name=flavour_name,
                save=save,
                dtype=dtype,
            )

        else:

            flavour_result = build_phi_E_theta_h_for_particle(
                theta_grid_deg=grid_config.theta_grid_deg,
                particle=particle,
                model_config=model_config,
                grid_config=grid_config,
                smoothing_config=smoothing_config,
                output_config=output_config,
                flavour_name=flavour_name,
                save=save,
                show_progress=show_progress,
                device=dev,
                dtype=dtype,
            )

        elapsed = time.perf_counter() - t0

        flavour_result["build_time_sec"] = elapsed

        all_results[flavour_name] = flavour_result

    return {
        "config": config,
        "results": all_results,
    }

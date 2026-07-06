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
High-level builders for Atmosphere height-flux datasets.

This module orchestrates the full pipeline:

    theta
        -> profile reconstruction
        -> Phi(E,h)
        -> flavour aggregation
        -> optional parallel execution

This module should not contain low-level physics operations.

It is tpeanuts-native orchestration code: it never calls MCEq
directly. Each physics step (MCEq instantiation, cascade-equation
solve, depth/altitude conversion, smoothing, profile normalization) is
delegated to production_profiles_all_energies_from_flux_gradient (see
tpeanuts.external.mceq.profiles), which in turn drives MCEq through
tpeanuts.external.mceq.core/solver/depth. This module's own job is to
repeat that single-(theta, particle) computation across a grid of
zenith angles and across flavour/particle selections, optionally in
parallel via joblib, and to handle result bookkeeping (timing,
saving to disk via save_phi_Eh_theta_result).

Module functions:
    split_run_config:
        Unpack a RunConfig into its constituent sub-configs.
    validate_particle_name:
        Validate that a particle identifier is a non-empty string.
    build_theta_result:
        Build (and optionally save) the production-height profile and
        height-differential flux for a single (theta, particle) pair.
    build_phi_E_theta_h_for_particle:
        Serially loop build_theta_result over a zenith-angle grid for
        one particle/flavour.
    build_phi_E_theta_h_for_particle_parallel:
        Parallel (joblib) version of
        build_phi_E_theta_h_for_particle.
    build_all_flavours:
        Top-level entry point: loop over every flavour/particle in a
        RunConfig, dispatching to the serial or parallel per-particle
        builder according to the run's ParallelConfig.
"""



from __future__ import annotations

import time
from typing import Dict, Optional, Union

import torch
from tqdm import tqdm

from tpeanuts.util.type import as_tensor
from tpeanuts.util.torch_util import default_device

from tpeanuts.external.mceq.config import (
    RunConfig,
    MCEqModelConfig,
    GridConfig,
    SmoothingConfig,
)
from tpeanuts.medium.atmosphere.io import (
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
    """
    Unpack a RunConfig into its constituent sub-configs.

    Args:
        config: A RunConfig bundling the model, grid, smoothing, output
            and parallel-execution settings for a build run.

    Returns:
        Tuple (model_config, grid_config, smoothing_config,
        output_config, parallel_config) extracted from config.
    """
    return (
        config.model,
        config.grid,
        config.smoothing,
        config.output,
        config.parallel,
    )


def validate_particle_name(particle: str) -> None:
    """
    Validate that a particle identifier is a non-empty string.

    Args:
        particle: MCEq particle name (e.g. "numu", "mu+") to validate.

    Raises:
        TypeError: If particle is not a string.
        ValueError: If particle is an empty or whitespace-only string.
    """
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
    """
    Build (and optionally save) the production-height profile and
    height-differential flux for a single zenith angle and particle.

    Thin wrapper around
    production_profiles_all_energies_from_flux_gradient (which is what
    actually instantiates and solves MCEq) that adds wall-clock timing
    and optional persistence of the result dict to disk via
    save_phi_Eh_theta_result.

    Args:
        theta_deg: Zenith angle in degrees (0 <= theta_deg < 90) of the
            shower/neutrino trajectory; theta_deg=0 is vertical.
        particle: MCEq particle name (e.g. "numu", "mu+") for which to
            solve and reconstruct the production profile.
        model_config: Optional MCEqModelConfig selecting the
            interaction/primary/density models; see
            production_profiles_all_energies_from_flux_gradient.
        grid_config: Optional GridConfig providing the Atmosphere depth
            grid, altitude grid and observation depth.
        smoothing_config: Optional SmoothingConfig controlling
            depth-axis smoothing before differentiation.
        save: If True, persist the result to disk using
            output_config and flavour_name.
        flavour_name: Optional neutrino-flavour label (e.g. "nu_mu")
            stored in the result dict and used to name the saved file.
        output_config: OutputConfig describing where/how to save the
            result; required when save=True.
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used throughout the computation.

    Returns:
        The result dict produced by
        production_profiles_all_energies_from_flux_gradient, augmented
        with "particle", "flavour_name", "build_time_sec" (wall-clock
        seconds spent in the profile reconstruction), and, if save is
        True, "output_path" (the path the result was written to).

    Raises:
        ValueError: If save=True but output_config is None, or if
            particle is invalid.
    """
    dev = default_device(device)

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
    """
    Serially build production-height profiles for one particle across a
    grid of zenith angles.

    Loops build_theta_result over every value in theta_grid_deg,
    re-solving MCEq's cascade equation independently for each zenith
    angle (each angle changes the slant-depth geometry and hence the
    cascade development), and collects the per-angle results keyed by
    the (float) zenith angle in degrees.

    Args:
        theta_grid_deg: 1-D tensor-like grid of zenith angles in
            degrees, each satisfying 0 <= theta_deg < 90.
        particle: MCEq particle name (e.g. "numu", "mu+") for which to
            build profiles at every angle.
        model_config: Optional MCEqModelConfig selecting the
            interaction/primary/density models.
        grid_config: Optional GridConfig providing the Atmosphere depth
            grid, altitude grid and observation depth.
        smoothing_config: Optional SmoothingConfig controlling
            depth-axis smoothing before differentiation.
        output_config: OutputConfig describing where/how to save each
            per-angle result; required when save=True.
        flavour_name: Optional neutrino-flavour label stored in each
            result and used to name saved files.
        save: If True, persist each per-angle result to disk.
        show_progress: If True, display a tqdm progress bar over the
            zenith-angle grid.
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used throughout the computation.

    Returns:
        Dict with keys "particle", "flavour_name", "theta_grid_deg"
        (the input grid as a tensor) and "results" (a dict mapping each
        zenith angle in degrees to its build_theta_result output dict).
    """
    dev = default_device(device)

    validate_particle_name(particle)

    theta_grid = as_tensor(
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
    """
    Worker function executed in a joblib subprocess/thread to build the
    result for a single zenith angle.

    Thin pass-through to build_theta_result, factored out as a
    module-level function (rather than a closure) so it can be pickled
    and dispatched by joblib.delayed across worker processes. Always
    runs on the CPU (no explicit device override), since CUDA contexts
    are typically not shared across joblib worker processes.

    Args:
        theta_deg: Zenith angle in degrees for this worker's task.
        particle: MCEq particle name to solve for.
        model_config: Optional MCEqModelConfig for this task.
        grid_config: Optional GridConfig for this task.
        smoothing_config: Optional SmoothingConfig for this task.
        output_config: OutputConfig used if save=True.
        flavour_name: Optional flavour label stored in the result.
        save: If True, persist the result to disk.
        dtype: Floating dtype used for the computation.

    Returns:
        The result dict produced by build_theta_result for this
        zenith angle.
    """
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
    """
    Parallel (joblib-based) version of
    build_phi_E_theta_h_for_particle.

    Dispatches one independent MCEq solve per zenith angle in
    theta_grid_deg to a joblib worker pool (process- or thread-based,
    per parallel_config.backend), since each angle's cascade-equation
    solve is an independent computation. Each worker calls
    _parallel_theta_worker, which internally invokes
    build_theta_result.

    Args:
        theta_grid_deg: 1-D tensor-like grid of zenith angles in
            degrees, each satisfying 0 <= theta_deg < 90.
        particle: MCEq particle name (e.g. "numu", "mu+") for which to
            build profiles at every angle.
        model_config: Optional MCEqModelConfig selecting the
            interaction/primary/density models.
        grid_config: Optional GridConfig providing the Atmosphere depth
            grid, altitude grid and observation depth.
        smoothing_config: Optional SmoothingConfig controlling
            depth-axis smoothing before differentiation.
        output_config: OutputConfig describing where/how to save each
            per-angle result; required when save=True.
        parallel_config: Optional ParallelConfig controlling the
            joblib worker count (n_jobs) and backend; defaults to
            ParallelConfig() if omitted.
        flavour_name: Optional neutrino-flavour label stored in each
            result and used to name saved files.
        save: If True, persist each per-angle result to disk.
        dtype: Floating dtype used throughout the computation.

    Returns:
        Dict with keys "particle", "flavour_name", "theta_grid_deg"
        (the input grid as a tensor) and "results" (a dict mapping each
        zenith angle in degrees to its build_theta_result output dict).

    Raises:
        ImportError: If joblib is not installed.
        ValueError: If parallel_config fails validation.
    """
    if Parallel is None:
        raise ImportError(
            "joblib is required for parallel execution."
        )

    theta_grid = as_tensor(
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
    """
    Build production-height profiles and height-differential fluxes for
    every flavour/particle defined in a RunConfig.

    Top-level entry point of the MCEq pipeline: iterates over
    config.flavours (a mapping of flavour name -> MCEq particle name),
    and for each one builds the full zenith-angle grid of results by
    dispatching to either build_phi_E_theta_h_for_particle_parallel or
    build_phi_E_theta_h_for_particle, depending on
    config.parallel.parallel. Each flavour's build is itself a
    collection of independent MCEq cascade-equation solves, one per
    zenith angle in config.grid.theta_grid_deg.

    Args:
        config: RunConfig bundling the model, grid, smoothing, output,
            parallel-execution and flavour-name settings for the run.
        save: If True, persist each per-angle result to disk via the
            configured OutputConfig.
        show_progress: If True, display tqdm progress bars over
            flavours (and, in the serial path, over zenith angles).
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used throughout the computation.

    Returns:
        Dict with keys "config" (the input RunConfig) and "results" (a
        dict mapping each flavour name to its
        build_phi_E_theta_h_for_particle[_parallel] output dict,
        augmented with a "build_time_sec" wall-clock timing entry).

    Raises:
        ValueError: If config fails validation.
        ImportError: If parallel execution is requested but joblib is
            not installed.
    """
    dev = default_device(device)

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

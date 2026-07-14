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
High-level generation routines for mceq height-differential flux files.

This module orchestrates existing mceq blocks:

    - angle conversion between detector theta and surface/MCEq alpha
    - production-profile reconstruction from dPhi/dX
    - torch-file saving with metadata
    - serial/parallel loops over particles and angles

This is a tpeanuts-native orchestration layer: it never calls MCEq
directly. The only physics-bearing external calls happen one level
down, in production_profiles_all_energies_from_flux_gradient (which
drives the MCEq cascade-equation solve; see
tpeanuts.external.mceq.profiles) and in
theta_detector_to_alpha_surface plus underground_path_length (which convert
a detector-frame zenith angle theta into the surface/MCEq zenith angle
alpha and the straight-line distance travelled inside the Earth down to
the detector; see tpeanuts.medium.atmosphere.geometry). Everything in
this module is tpeanuts-native plumbing around those two calls: angle
bookkeeping, per-job output-path construction, metadata assembly, and
serial/parallel iteration over (particle, angle) job grids.

Module functions:
    generate_flux_for_particle_angle:
        Generate (and optionally save) the height-differential flux for
        one particle at one detector/surface angle.
    generate_flux_for_particles_angle_grid:
        Build a (particle x angle) job grid and execute it serially or
        in parallel (via run_task_dicts), grouping results by flavour.
"""



from __future__ import annotations

import time
from typing import Any, Dict, Optional, Union

import torch

from tpeanuts.medium.atmosphere.geometry import (
    alpha_surface_to_theta_detector,
    theta_detector_to_alpha_surface,
    underground_path_length,
)
from tpeanuts.external.mceq.config import (
    GridConfig,
    MCEqModelConfig,
    SmoothingConfig,
)
from tpeanuts.util.parallel import ParallelConfig
from tpeanuts.external.mceq.core import ensure_mceq_available
from tpeanuts.medium.atmosphere.io import (
    build_angle_output_path,
    OutputConfig,
    save_phi_Eh_theta_result,
)
from tpeanuts.util.parallel import run_task_dicts
from tpeanuts.external.mceq.profiles import (
    production_profiles_all_energies_from_flux_gradient,
)
from tpeanuts.util.torch_util import resolve_device, scalar_float
from tpeanuts.util.type import as_tensor


TensorLike = Union[float, int, torch.Tensor]


def _prepare_grid_config_for_alpha(
    grid_config: Optional[GridConfig],
    alpha_deg: float,
) -> GridConfig:
    """
    Build a GridConfig restricted to a single zenith angle, preserving
    the depth/altitude/observation-depth settings of a template config.

    Each (particle, angle) generation job in this module solves MCEq at
    one specific surface zenith angle. MCEq names this coordinate theta,
    while TPeanuts names the surface angle alpha, so the external
    GridConfig.theta_grid_deg field is set to [alpha_deg].
    while the Atmosphere depth grid (X_grid_gcm2), altitude grid
    (h_grid_km) and observation depth (X_obs_gcm2) are carried over
    unchanged from grid_config (or left at GridConfig defaults if
    grid_config is None).

    Args:
        grid_config: Optional template GridConfig to copy
            X_grid_gcm2/h_grid_km/X_obs_gcm2 from; if None, GridConfig
            defaults are used for those fields.
        alpha_deg: Surface zenith angle in degrees to solve at.

    Returns:
        A new GridConfig with theta_grid_deg=[alpha_deg] and the other
        grid fields copied from grid_config (or defaulted).
    """
    if grid_config is None:
        return GridConfig(
            theta_grid_deg=torch.tensor([alpha_deg], dtype=torch.float64),
        )

    return GridConfig(
        theta_grid_deg=torch.tensor([alpha_deg], dtype=torch.float64),
        X_grid_gcm2=grid_config.X_grid_gcm2,
        h_grid_km=grid_config.h_grid_km,
        X_obs_gcm2=grid_config.X_obs_gcm2,
    )


def _metadata_extra(
    *,
    alpha_deg: Optional[float],
    theta_deg: float,
    detector_depth_m: float,
    surface_distance_km: Optional[float],
    model_config: MCEqModelConfig,
    grid_config: GridConfig,
    smoothing_config: SmoothingConfig,
    build_time_sec: float,
) -> Dict[str, Any]:
    """
    Assemble a flat metadata dict describing one generation job, for
    storage alongside the saved flux result.

    Captures the detector/surface angle relationship, the resolved
    model selection, the grid sizes, the smoothing settings, and the
    wall-clock build time, so that a saved result file is
    self-describing without needing to re-derive these values from the
    original run configuration.

    Args:
        alpha_deg: Surface/MCEq zenith angle in degrees.
        theta_deg: Detector zenith angle in degrees.
        detector_depth_m: Depth of the detector below the surface, in
            metres, used in the detector/surface angle conversion.
        surface_distance_km: Straight-line distance in km from the
            surface entry point to the detector along the trajectory,
            or None if not computed.
        model_config: The MCEqModelConfig used for this job
            (interaction_model, primary_model, density_model).
        grid_config: The (single-angle) GridConfig used for this job.
        smoothing_config: The SmoothingConfig used for this job.
        build_time_sec: Wall-clock seconds spent solving/reconstructing
            the profile.

    Returns:
        Dict of plain Python scalars/strings suitable for serialization
        alongside the result tensors.
    """
    return {
        "theta_detector_deg": theta_deg,
        "alpha_surface_deg": alpha_deg,
        "alpha_mceq_deg": alpha_deg,
        "detector_depth_m": float(detector_depth_m),
        "surface_distance_km": surface_distance_km,
        "interaction_model": model_config.interaction_model,
        "primary_model": str(model_config.primary_model),
        "density_model": model_config.density_model,
        "X_obs_gcm2": float(grid_config.X_obs_gcm2),
        "X_grid_n": int(grid_config.X_grid_gcm2.numel()),
        "h_grid_n": int(grid_config.h_grid_km.numel()),
        "smoothing_method": smoothing_config.method,
        "smoothing": float(smoothing_config.smoothing),
        "gaussian_sigma": float(smoothing_config.gaussian_sigma),
        "positive_only": bool(smoothing_config.positive_only),
        "build_time_sec": float(build_time_sec),
    }


@torch.no_grad()
def generate_flux_for_particle_angle(
    particle: str,
    *,
    alpha_deg: Optional[TensorLike] = None,
    theta_deg: Optional[TensorLike] = None,
    detector_depth_m: float = 0.0,
    flavour_name: Optional[str] = None,
    model_config: Optional[MCEqModelConfig] = None,
    grid_config: Optional[GridConfig] = None,
    smoothing_config: Optional[SmoothingConfig] = None,
    output_config: Optional[OutputConfig] = None,
    save: bool = True,
    skip_existing: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Generate (and optionally save) the height-differential flux for one
    particle at one angle. TPeanuts convention is theta_deg for the
    detector angle and alpha_deg for the surface angle.

    If alpha_deg is not given, theta_deg is converted to the surface
    angle alpha via theta_detector_to_alpha_surface, while
    underground_path_length yields the straight-line distance travelled
    from the surface to the detector. The resulting alpha is then used
    to instantiate and solve MCEq's
    cascade equation via
    production_profiles_all_energies_from_flux_gradient, producing the
    production-height profile and height-differential flux phi_Eh for
    that single angle. If save=True, the result (together with
    descriptive metadata from _metadata_extra) is written to disk via
    save_phi_Eh_theta_result.

    Args:
        particle: MCEq particle name (e.g. "numu", "mu+") to solve for.
        alpha_deg: Surface/MCEq zenith angle in degrees (0 <= alpha_deg
            < 90); if omitted, it is computed from theta_deg.
        theta_deg: Detector-frame zenith angle in degrees; if omitted,
            it is computed from alpha_deg for bookkeeping and later
            atmosphere propagation.
        detector_depth_m: Depth of the detector below the surface, in
            metres, used in the theta <-> alpha conversion.
        flavour_name: Optional neutrino-flavour label stored in the
            result and used to name the saved file.
        model_config: Optional MCEqModelConfig selecting the
            interaction/primary/density models; defaults to
            MCEqModelConfig() if omitted.
        grid_config: Optional template GridConfig providing the
            Atmosphere depth grid, altitude grid and observation depth
            (MCEq's theta_grid_deg is overridden internally to the
            resolved surface alpha); defaults to GridConfig() fields if
            omitted.
        smoothing_config: Optional SmoothingConfig controlling
            depth-axis smoothing before differentiation; defaults to
            SmoothingConfig() if omitted.
        output_config: Optional OutputConfig describing where/how to
            save the result; defaults to OutputConfig() if omitted.
        save: If True, persist the result to disk.
        skip_existing: If True and an output file already exists at the
            expected path (and output_config.overwrite is False), skip
            the (expensive) MCEq solve and return early with
            "skipped": True.
        device: Output torch device, a device-name string, or a
            zero-argument callable returning one. None selects CUDA
            when available, else CPU.
        dtype: Floating dtype used throughout the computation.
        debug: If True, print progress/diagnostic messages.

    Returns:
        Result dict from production_profiles_all_energies_from_flux_gradient,
        augmented with "particle", "flavour_name", "theta_deg",
        optionally "alpha_deg", "metadata_extra", and (if save=True)
        "output_path". If skipped via skip_existing, a smaller dict
        with "particle", "flavour_name", "alpha_deg", "theta_deg",
        "output_path" and "skipped": True is returned instead.

    Raises:
        ValueError: If neither alpha_deg nor theta_deg is provided, or
            if model_config/smoothing_config/output_config/
            current_grid_config fail validation.
        ImportError: If the optional MCEq package is not installed.
    """
    if theta_deg is None and alpha_deg is None:
        raise ValueError("Provide either alpha_deg or theta_deg.")

    dev = resolve_device(device)

    if model_config is None:
        model_config = MCEqModelConfig()

    if smoothing_config is None:
        smoothing_config = SmoothingConfig()

    if output_config is None:
        output_config = OutputConfig()

    model_config.validate()
    smoothing_config.validate()
    output_config.validate()

    depth_km = detector_depth_m / 1.0e3
    alpha_value = None
    surface_distance_value = None

    if alpha_deg is None:
        theta_value = scalar_float(theta_deg)
        alpha_t = theta_detector_to_alpha_surface(
            theta_deg,
            depth_km,
            device=dev,
            dtype=dtype,
        )
        surface_distance_t = underground_path_length(
            theta_deg,
            depth_km=depth_km,
            device=dev,
            dtype=dtype,
        )
        alpha_value = scalar_float(alpha_t)
        surface_distance_value = scalar_float(surface_distance_t)
    else:
        alpha_value = scalar_float(alpha_deg)
        if theta_deg is None:
            theta_t = alpha_surface_to_theta_detector(
                alpha_deg,
                depth_km,
                device=dev,
                dtype=dtype,
            )
            theta_value = scalar_float(theta_t)
        else:
            theta_value = scalar_float(theta_deg)
        surface_distance_t = underground_path_length(
            theta_value,
            depth_km=depth_km,
            device=dev,
            dtype=dtype,
        )
        surface_distance_value = scalar_float(surface_distance_t)

    if not (0.0 <= alpha_value < 90.0):
        raise ValueError("MCEq source generation expects 0 <= alpha_deg < 90.")

    current_grid_config = _prepare_grid_config_for_alpha(
        grid_config,
        alpha_value,
    )
    current_grid_config.validate()

    output_path = None

    if save:
        output_path = build_angle_output_path(
            output_config=output_config,
            particle=particle,
            alpha_deg=alpha_value,
            flavour_name=flavour_name,
        )

        if skip_existing and not output_config.overwrite:
            import os

            if os.path.exists(output_path):
                if debug:
                    print(
                        f"Skipping existing {particle} theta_detector={theta_value:.3f} "
                        f"alpha_surface={alpha_value:.3f}: {output_path}"
                    )
                return {
                    "particle": particle,
                    "flavour_name": flavour_name,
                    "alpha_deg": alpha_value,
                    "theta_deg": theta_value,
                    "output_path": output_path,
                    "skipped": True,
                }

    if debug:
        print(
            f"Generating {particle} | theta_detector={theta_value:.3f} deg | "
            f"alpha_surface={alpha_value:.3f} deg"
        )

    t0 = time.perf_counter()

    result = production_profiles_all_energies_from_flux_gradient(
        alpha_deg=alpha_value,
        particle=particle,
        model_config=model_config,
        grid_config=current_grid_config,
        smoothing_config=smoothing_config,
        device=dev,
        dtype=dtype,
    )

    build_time_sec = time.perf_counter() - t0

    result["particle"] = particle
    result["flavour_name"] = flavour_name
    result["theta_deg"] = torch.as_tensor(theta_value, device=dev, dtype=dtype)
    result["alpha_deg"] = torch.as_tensor(alpha_value, device=dev, dtype=dtype)
    result["alpha_mceq_deg"] = torch.as_tensor(alpha_value, device=dev, dtype=dtype)

    result["metadata_extra"] = _metadata_extra(
        alpha_deg=alpha_value,
        theta_deg=theta_value,
        detector_depth_m=detector_depth_m,
        surface_distance_km=surface_distance_value,
        model_config=model_config,
        grid_config=current_grid_config,
        smoothing_config=smoothing_config,
        build_time_sec=build_time_sec,
    )

    if save:
        output_path = save_phi_Eh_theta_result(
            result=result,
            output_config=output_config,
            particle=particle,
            alpha_deg=alpha_value,
            theta_deg=theta_value,
            flavour_name=flavour_name,
        )
        result["output_path"] = output_path
        if debug:
            print(f"Saved {particle}: {output_path}. Time Execution: {build_time_sec} s.")

    return result


@torch.no_grad()
def generate_flux_for_particles_angle_grid(
    particles: Union[Dict[str, str], list[str], tuple[str, ...]],
    *,
    alpha_grid_deg: Optional[TensorLike] = None,
    theta_grid_deg: Optional[TensorLike] = None,
    detector_depth_m: float = 0.0,
    model_config: Optional[MCEqModelConfig] = None,
    grid_config: Optional[GridConfig] = None,
    smoothing_config: Optional[SmoothingConfig] = None,
    output_config: Optional[OutputConfig] = None,
    parallel_config: Optional[ParallelConfig] = None,
    save: bool = True,
    skip_existing: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Generate the height-differential flux for multiple particles across
    a shared grid of detector or surface angles, serially or in
    parallel.

    Builds the full (particle x angle) Cartesian product of generation
    jobs as a flat list of task dicts, then executes them via
    run_task_dicts, and finally regroups the flat results by flavour
    name into a nested dictionary keyed by flavour and detector angle.

    Args:
        particles: Either a dict mapping flavour name -> MCEq particle
            name, or a list/tuple of particle names (in which case each
            particle name is also used as its own flavour name).
        alpha_grid_deg: 1-D tensor-like grid of surface/MCEq zenith
            angles in degrees; mutually exclusive with theta_grid_deg.
        theta_grid_deg: 1-D tensor-like grid of detector-frame zenith
            angles in degrees; mutually exclusive with alpha_grid_deg.
        detector_depth_m: Depth of the detector below the surface, in
            metres, used in the theta <-> alpha conversion.
        model_config: Optional MCEqModelConfig selecting the
            interaction/primary/density models, shared by all jobs.
        grid_config: Optional template GridConfig providing the
            Atmosphere depth grid, altitude grid and observation depth,
            shared by all jobs.
        smoothing_config: Optional SmoothingConfig controlling
            depth-axis smoothing, shared by all jobs.
        output_config: Optional OutputConfig describing where/how to
            save each result.
        parallel_config: Optional ParallelConfig controlling whether
            and how jobs are executed in parallel; defaults to
            ParallelConfig(parallel=False) (serial execution) if
            omitted.
        save: If True, persist each per-job result to disk.
        skip_existing: If True, skip jobs whose output file already
            exists.
        device: Output torch device, a device-name string, or a
            zero-argument callable returning one.
        dtype: Floating dtype used throughout the computation.
        debug: If True, print progress/diagnostic messages and show a
            progress bar over tasks.

    Returns:
        Dict mapping each flavour name to a dict with keys "particle",
        "flavour_name", "angle_mode", "input_angle_grid_deg" and
        "results" (a dict mapping each resolved detector theta angle in
        degrees to its generate_flux_for_particle_angle output dict),
        matching the grouped shape used by the Honda generator.

    Raises:
        ValueError: If neither or both of alpha_grid_deg/theta_grid_deg
            are provided.
        ImportError: If the optional MCEq package is not installed, or
            if parallel execution is requested but joblib is missing.
    """
    if alpha_grid_deg is None and theta_grid_deg is None:
        raise ValueError("Provide either alpha_grid_deg or theta_grid_deg.")

    if alpha_grid_deg is not None and theta_grid_deg is not None:
        raise ValueError("Use either alpha_grid_deg or theta_grid_deg, not both.")

    if isinstance(particles, dict):
        particle_items = list(particles.items())
    else:
        particle_items = [(str(particle), str(particle)) for particle in particles]

    ensure_mceq_available()

    angle_mode = "alpha" if alpha_grid_deg is not None else "theta"
    angle_grid = alpha_grid_deg if alpha_grid_deg is not None else theta_grid_deg
    angle_grid = as_tensor(angle_grid, device="cpu", dtype=torch.float64).reshape(-1)

    if parallel_config is None:
        parallel_config = ParallelConfig(parallel=False)

    tasks = []

    for flavour_name, particle in particle_items:
        for angle in angle_grid:
            tasks.append(
                {
                    "particle": particle,
                    "alpha_deg": float(angle.item()) if angle_mode == "alpha" else None,
                    "theta_deg": float(angle.item()) if angle_mode == "theta" else None,
                    "detector_depth_m": detector_depth_m,
                    "flavour_name": flavour_name,
                    "model_config": model_config,
                    "grid_config": grid_config,
                    "smoothing_config": smoothing_config,
                    "output_config": output_config,
                    "save": save,
                    "skip_existing": skip_existing,
                    "device": device,
                    "dtype": dtype,
                    "debug": debug,
                }
            )

    if debug:
        print(
            f"Generating {len(tasks)} mceq flux jobs "
            f"({len(particle_items)} particles x {len(angle_grid)} angles). "
            f"parallel={parallel_config.parallel}"
        )

    results_flat = run_task_dicts(
        func=generate_flux_for_particle_angle,
        tasks=tasks,
        config=parallel_config,
        show_progress=debug,
        desc="mceq flux generation",
    )

    grouped: Dict[str, Dict[str, Any]] = {}

    for task, result in zip(tasks, results_flat):
        flavour_name = task["flavour_name"]
        particle = task["particle"]

        if flavour_name not in grouped:
            grouped[flavour_name] = {
                "particle": particle,
                "flavour_name": flavour_name,
                "angle_mode": angle_mode,
                "input_angle_grid_deg": angle_grid,
                "results": {},
            }

        grouped[flavour_name]["results"][scalar_float(result["theta_deg"])] = result

    return grouped

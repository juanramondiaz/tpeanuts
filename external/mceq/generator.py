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

    - angle conversion from detector to surface/mceq angle
    - production-profile reconstruction from dPhi/dX
    - torch-file saving with metadata
    - serial/parallel loops over particles and angles

This is a tpeanuts-native orchestration layer: it never calls MCEq
directly. The only physics-bearing external calls happen one level
down, in production_profiles_all_energies_from_flux_gradient (which
drives the MCEq cascade-equation solve; see
tpeanuts.external.mceq.profiles) and in
detector_alpha_to_surface_theta (which converts a detector-frame
incidence angle alpha into the surface/MCEq zenith angle theta and the
straight-line distance travelled inside the Atmosphere/Earth down to
the detector; see tpeanuts.medium.atmosphere.geometry). Everything in
this module is tpeanuts-native plumbing around those two calls: angle
bookkeeping, per-job output-path construction, metadata assembly, and
serial/parallel iteration over (particle, angle) job grids.

Module functions:
    generate_flux_for_particle_angle:
        Generate (and optionally save) the height-differential flux for
        one particle at one detector/surface angle.
    generate_flux_for_particle_angle_grid:
        Serially loop generate_flux_for_particle_angle over a grid of
        angles for one particle.
    generate_flux_for_particles_angle_grid:
        Build a (particle x angle) job grid and execute it serially or
        in parallel (via run_task_dicts), grouping results by flavour.
"""



from __future__ import annotations

import time
from typing import Any, Dict, Optional, Union

import torch

from tpeanuts.medium.atmosphere.geometry import detector_alpha_to_surface_theta
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
from tpeanuts.util.torch_util import default_device
from tpeanuts.util.type import as_tensor


TensorLike = Union[float, int, torch.Tensor]


def _debug_print(debug: bool, message: str) -> None:
    """
    Print a diagnostic message only when debugging is enabled.

    Args:
        debug: If True, print message; if False, do nothing.
        message: Text to print.
    """
    if debug:
        print(message)


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    """
    Resolve a device argument that may be a literal device, a device
    name, a zero-argument callable returning a device, or None.

    Args:
        device: A torch.device, a device-name string, a callable
            returning either of those, or None to auto-select (CUDA if
            available, else CPU).

    Returns:
        The resolved torch.device.
    """
    if callable(device):
        device = device()

    return default_device(device)


def _scalar_float(value: TensorLike) -> float:
    """
    Convert a tensor-like scalar to a plain Python float.

    Args:
        value: Scalar or tensor-like value; only the first element is
            used if more than one is given.

    Returns:
        The value as a Python float.
    """
    value_t = as_tensor(value, device="cpu", dtype=torch.float64)
    return float(value_t.detach().cpu().reshape(-1)[0].item())


def _prepare_grid_config_for_theta(
    grid_config: Optional[GridConfig],
    theta_deg: float,
) -> GridConfig:
    """
    Build a GridConfig restricted to a single zenith angle, preserving
    the depth/altitude/observation-depth settings of a template config.

    Each (particle, angle) generation job in this module solves MCEq at
    one specific zenith angle, so the GridConfig.theta_grid_deg used
    internally is overridden to a single-element tensor [theta_deg]
    while the Atmosphere depth grid (X_grid_gcm2), altitude grid
    (h_grid_km) and observation depth (X_obs_gcm2) are carried over
    unchanged from grid_config (or left at GridConfig defaults if
    grid_config is None).

    Args:
        grid_config: Optional template GridConfig to copy
            X_grid_gcm2/h_grid_km/X_obs_gcm2 from; if None, GridConfig
            defaults are used for those fields.
        theta_deg: Zenith angle in degrees to solve at.

    Returns:
        A new GridConfig with theta_grid_deg=[theta_deg] and the other
        grid fields copied from grid_config (or defaulted).
    """
    if grid_config is None:
        return GridConfig(
            theta_grid_deg=torch.tensor([theta_deg], dtype=torch.float64),
        )

    return GridConfig(
        theta_grid_deg=torch.tensor([theta_deg], dtype=torch.float64),
        X_grid_gcm2=grid_config.X_grid_gcm2,
        h_grid_km=grid_config.h_grid_km,
        X_obs_gcm2=grid_config.X_obs_gcm2,
    )


def _expected_output_path(
    *,
    output_config: OutputConfig,
    particle: str,
    theta_deg: float,
    alpha_deg: Optional[float],
    flavour_name: Optional[str],
) -> str:
    """
    Compute the output file path a generation job would write to,
    without actually running or saving anything.

    Used to implement skip_existing: the path is computed up front so
    an already-existing output file can short-circuit a (potentially
    expensive) MCEq solve.

    Args:
        output_config: OutputConfig describing the output directory/
            naming convention.
        particle: MCEq particle name.
        theta_deg: Surface/MCEq zenith angle in degrees.
        alpha_deg: Optional detector-frame incidence angle in degrees,
            included in the file name when available.
        flavour_name: Optional neutrino-flavour label included in the
            file name.

    Returns:
        The path (as a string) the corresponding result would be saved
        to.
    """
    return build_angle_output_path(
        output_config=output_config,
        theta_deg=theta_deg,
        particle=particle,
        flavour_name=flavour_name,
        alpha_deg=alpha_deg,
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
        alpha_deg: Detector-frame incidence angle in degrees, or None
            if the job was specified directly in terms of theta_deg.
        theta_deg: Surface/MCEq zenith angle in degrees (0 <= theta_deg
            < 90).
        detector_depth_m: Depth of the detector below the surface, in
            metres, used in the alpha -> theta conversion.
        surface_distance_km: Straight-line distance in km from the
            surface entry point to the detector along the trajectory,
            or None if not computed (i.e. theta_deg was given
            directly).
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
        "alpha_detector_deg": alpha_deg,
        "theta_surface_deg": theta_deg,
        "theta_mceq_deg": theta_deg,
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
    particle at one angle, specified either as a detector-frame
    incidence angle or directly as a surface/MCEq zenith angle.

    If theta_deg is not given, alpha_deg (the angle of incidence in the
    detector's own frame, e.g. relative to the local vertical at depth
    detector_depth_m) is first converted to the surface zenith angle
    theta via detector_alpha_to_surface_theta, which also yields the
    straight-line distance travelled from the surface to the detector.
    The resulting theta is then used to instantiate and solve MCEq's
    cascade equation via
    production_profiles_all_energies_from_flux_gradient, producing the
    production-height profile and height-differential flux phi_Eh for
    that single angle. If save=True, the result (together with
    descriptive metadata from _metadata_extra) is written to disk via
    save_phi_Eh_theta_result.

    Args:
        particle: MCEq particle name (e.g. "numu", "mu+") to solve for.
        alpha_deg: Detector-frame incidence angle in degrees; required
            if theta_deg is not given.
        theta_deg: Surface/MCEq zenith angle in degrees (0 <= theta_deg
            < 90); if given, takes precedence and alpha_deg is used
            only for bookkeeping/metadata (if also supplied).
        detector_depth_m: Depth of the detector below the surface, in
            metres, used in the alpha -> theta conversion.
        flavour_name: Optional neutrino-flavour label stored in the
            result and used to name the saved file.
        model_config: Optional MCEqModelConfig selecting the
            interaction/primary/density models; defaults to
            MCEqModelConfig() if omitted.
        grid_config: Optional template GridConfig providing the
            Atmosphere depth grid, altitude grid and observation depth
            (theta_grid_deg is overridden internally to the single
            resolved angle); defaults to GridConfig() fields if
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

    dev = _resolve_device(device)

    if model_config is None:
        model_config = MCEqModelConfig()

    if smoothing_config is None:
        smoothing_config = SmoothingConfig()

    if output_config is None:
        output_config = OutputConfig()

    model_config.validate()
    smoothing_config.validate()
    output_config.validate()

    alpha_value = None
    surface_distance_value = None

    if theta_deg is None:
        theta_t, surface_distance_t = detector_alpha_to_surface_theta(
            alpha_deg,
            detector_depth_m=detector_depth_m,
            device=dev,
            dtype=dtype,
            return_distance=True,
        )
        alpha_value = _scalar_float(alpha_deg)
        theta_value = _scalar_float(theta_t)
        surface_distance_value = _scalar_float(surface_distance_t)
    else:
        theta_value = _scalar_float(theta_deg)
        if alpha_deg is not None:
            alpha_value = _scalar_float(alpha_deg)

    current_grid_config = _prepare_grid_config_for_theta(
        grid_config,
        theta_value,
    )
    current_grid_config.validate()

    output_path = None

    if save:
        output_path = _expected_output_path(
            output_config=output_config,
            particle=particle,
            theta_deg=theta_value,
            alpha_deg=alpha_value,
            flavour_name=flavour_name,
        )

        if skip_existing and not output_config.overwrite:
            import os

            if os.path.exists(output_path):
                _debug_print(
                    debug,
                    f"Skipping existing {particle} alpha={alpha_value} "
                    f"theta={theta_value:.3f}: {output_path}",
                )
                return {
                    "particle": particle,
                    "flavour_name": flavour_name,
                    "alpha_deg": alpha_value,
                    "theta_deg": theta_value,
                    "output_path": output_path,
                    "skipped": True,
                }

    _debug_print(
        debug,
        f"Generating {particle} | alpha={alpha_value} deg | "
        f"theta={theta_value:.3f} deg",
    )

    t0 = time.perf_counter()

    result = production_profiles_all_energies_from_flux_gradient(
        theta_deg=theta_value,
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

    if alpha_value is not None:
        result["alpha_deg"] = torch.as_tensor(alpha_value, device=dev, dtype=dtype)

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
        _debug_print(debug, f"Saved {particle}: {output_path}. Time Execution: {build_time_sec} s.")

    return result


@torch.no_grad()
def generate_flux_for_particle_angle_grid(
    particle: str,
    *,
    alpha_grid_deg: Optional[TensorLike] = None,
    theta_grid_deg: Optional[TensorLike] = None,
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
    Serially generate the height-differential flux for one particle
    across a grid of detector or surface angles.

    Loops generate_flux_for_particle_angle over every value in
    alpha_grid_deg (detector-frame incidence angles) or
    theta_grid_deg (surface/MCEq zenith angles), exactly one of which
    must be given, and collects the per-angle results keyed by the
    resolved surface zenith angle in degrees.

    Args:
        particle: MCEq particle name (e.g. "numu", "mu+") to solve for.
        alpha_grid_deg: 1-D tensor-like grid of detector-frame
            incidence angles in degrees; mutually exclusive with
            theta_grid_deg.
        theta_grid_deg: 1-D tensor-like grid of surface/MCEq zenith
            angles in degrees; mutually exclusive with alpha_grid_deg.
        detector_depth_m: Depth of the detector below the surface, in
            metres, used in the alpha -> theta conversion.
        flavour_name: Optional neutrino-flavour label stored in each
            result and used to name saved files.
        model_config: Optional MCEqModelConfig selecting the
            interaction/primary/density models.
        grid_config: Optional template GridConfig providing the
            Atmosphere depth grid, altitude grid and observation depth.
        smoothing_config: Optional SmoothingConfig controlling
            depth-axis smoothing before differentiation.
        output_config: Optional OutputConfig describing where/how to
            save each result.
        save: If True, persist each per-angle result to disk.
        skip_existing: If True, skip angles whose output file already
            exists (see generate_flux_for_particle_angle).
        device: Output torch device, a device-name string, or a
            zero-argument callable returning one.
        dtype: Floating dtype used throughout the computation.
        debug: If True, print progress/diagnostic messages.

    Returns:
        Dict with keys "particle", "flavour_name", "angle_mode" ("alpha"
        or "theta", indicating which grid was provided),
        "input_angle_grid_deg" (the input grid as a tensor) and
        "results" (a dict mapping each resolved surface zenith angle in
        degrees to its generate_flux_for_particle_angle output dict).

    Raises:
        ValueError: If neither alpha_grid_deg nor theta_grid_deg is
            provided.
    """
    if alpha_grid_deg is None and theta_grid_deg is None:
        raise ValueError("Provide either alpha_grid_deg or theta_grid_deg.")

    angle_grid = alpha_grid_deg if alpha_grid_deg is not None else theta_grid_deg
    angle_grid = as_tensor(angle_grid, device="cpu", dtype=torch.float64).reshape(-1)

    results = {}

    for idx, angle in enumerate(angle_grid):
        angle_value = float(angle.item())
        _debug_print(
            debug,
            f"\n[{particle}] angle {idx + 1}/{len(angle_grid)} = {angle_value:.3f} deg",
        )

        result = generate_flux_for_particle_angle(
            particle=particle,
            alpha_deg=angle_value if alpha_grid_deg is not None else None,
            theta_deg=angle_value if theta_grid_deg is not None else None,
            detector_depth_m=detector_depth_m,
            flavour_name=flavour_name,
            model_config=model_config,
            grid_config=grid_config,
            smoothing_config=smoothing_config,
            output_config=output_config,
            save=save,
            skip_existing=skip_existing,
            device=device,
            dtype=dtype,
            debug=debug,
        )

        results[_scalar_float(result["theta_deg"])] = result

    return {
        "particle": particle,
        "flavour_name": flavour_name,
        "angle_mode": "alpha" if alpha_grid_deg is not None else "theta",
        "input_angle_grid_deg": angle_grid,
        "results": results,
    }


def _generate_particle_angle_task(
    *,
    particle: str,
    angle_deg: float,
    angle_mode: str,
    detector_depth_m: float,
    flavour_name: Optional[str],
    model_config: Optional[MCEqModelConfig],
    grid_config: Optional[GridConfig],
    smoothing_config: Optional[SmoothingConfig],
    output_config: Optional[OutputConfig],
    save: bool,
    skip_existing: bool,
    device: Optional[Union[str, torch.device]],
    dtype: torch.dtype,
    debug: bool,
):
    """
    Task wrapper executed (serially or via a parallel backend, see
    run_task_dicts) for a single (particle, angle) generation job.

    Thin pass-through to generate_flux_for_particle_angle, factored out
    as a module-level function with keyword-only, picklable arguments
    so it can be dispatched as a task by
    tpeanuts.util.parallel.run_task_dicts.

    Args:
        particle: MCEq particle name to solve for.
        angle_deg: The angle value for this task, interpreted according
            to angle_mode.
        angle_mode: Either "alpha" (angle_deg is a detector-frame
            incidence angle) or "theta" (angle_deg is a surface/MCEq
            zenith angle).
        detector_depth_m: Depth of the detector below the surface, in
            metres.
        flavour_name: Optional neutrino-flavour label.
        model_config: Optional MCEqModelConfig for this task.
        grid_config: Optional template GridConfig for this task.
        smoothing_config: Optional SmoothingConfig for this task.
        output_config: Optional OutputConfig for this task.
        save: If True, persist the result to disk.
        skip_existing: If True, skip if the output file already exists.
        device: Output torch device, device-name string, or callable.
        dtype: Floating dtype used for the computation.
        debug: If True, print progress/diagnostic messages.

    Returns:
        The result dict produced by generate_flux_for_particle_angle
        for this task.
    """
    return generate_flux_for_particle_angle(
        particle=particle,
        alpha_deg=angle_deg if angle_mode == "alpha" else None,
        theta_deg=angle_deg if angle_mode == "theta" else None,
        detector_depth_m=detector_depth_m,
        flavour_name=flavour_name,
        model_config=model_config,
        grid_config=grid_config,
        smoothing_config=smoothing_config,
        output_config=output_config,
        save=save,
        skip_existing=skip_existing,
        device=device,
        dtype=dtype,
        debug=debug,
    )


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
    run_task_dicts (which dispatches to _generate_particle_angle_task
    either serially or across a joblib worker pool, depending on
    parallel_config), and finally regroups the flat results by flavour
    name into the same nested shape produced by
    generate_flux_for_particle_angle_grid.

    Args:
        particles: Either a dict mapping flavour name -> MCEq particle
            name, or a list/tuple of particle names (in which case each
            particle name is also used as its own flavour name).
        alpha_grid_deg: 1-D tensor-like grid of detector-frame
            incidence angles in degrees; mutually exclusive with
            theta_grid_deg.
        theta_grid_deg: 1-D tensor-like grid of surface/MCEq zenith
            angles in degrees; mutually exclusive with alpha_grid_deg.
        detector_depth_m: Depth of the detector below the surface, in
            metres, used in the alpha -> theta conversion.
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
        "results" (a dict mapping each resolved surface zenith angle in
        degrees to its generate_flux_for_particle_angle output dict),
        matching the shape produced by
        generate_flux_for_particle_angle_grid.

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
                    "angle_deg": float(angle.item()),
                    "angle_mode": angle_mode,
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

    _debug_print(
        debug,
        f"Generating {len(tasks)} mceq flux jobs "
        f"({len(particle_items)} particles x {len(angle_grid)} angles). "
        f"parallel={parallel_config.parallel}",
    )

    results_flat = run_task_dicts(
        func=_generate_particle_angle_task,
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

        grouped[flavour_name]["results"][_scalar_float(result["theta_deg"])] = result

    return grouped

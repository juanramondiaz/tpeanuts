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
"""



from __future__ import annotations

import time
from typing import Any, Dict, Optional, Union

import torch

from tpeanuts.atmosphere.geometry import detector_alpha_to_surface_theta
from tpeanuts.external.mceq.config import (
    GridConfig,
    MCEqModelConfig,
    SmoothingConfig,
)
from tpeanuts.util.parallel import ParallelConfig
from tpeanuts.external.mceq.core import ensure_mceq_available
from tpeanuts.io.io_atmosphere import (
    build_angle_output_path,
    OutputConfig,
    save_phi_Eh_theta_result,
)
from tpeanuts.util.parallel import run_task_dicts
from tpeanuts.external.mceq.profiles import (
    production_profiles_all_energies_from_flux_gradient,
)
from tpeanuts.util.torch_util import _default_device
from tpeanuts.util.type import _as_tensor


TensorLike = Union[float, int, torch.Tensor]


def _debug_print(debug: bool, message: str) -> None:
    if debug:
        print(message)


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if callable(device):
        device = device()

    return _default_device(device)


def _scalar_float(value: TensorLike) -> float:
    value_t = _as_tensor(value, device="cpu", dtype=torch.float64)
    return float(value_t.detach().cpu().reshape(-1)[0].item())


def _prepare_grid_config_for_theta(
    grid_config: Optional[GridConfig],
    theta_deg: float,
) -> GridConfig:
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
    if alpha_grid_deg is None and theta_grid_deg is None:
        raise ValueError("Provide either alpha_grid_deg or theta_grid_deg.")

    angle_grid = alpha_grid_deg if alpha_grid_deg is not None else theta_grid_deg
    angle_grid = _as_tensor(angle_grid, device="cpu", dtype=torch.float64).reshape(-1)

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
    angle_grid = _as_tensor(angle_grid, device="cpu", dtype=torch.float64).reshape(-1)

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

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
Generate TPeanuts height-differential source flux files from Honda tables.

This module is tpeanuts-native logic built on top of the Honda/HKKM table
reader in ``external.honda.tables``: it does not call any external Python
package, only the local parser for the Honda ``.d.gz`` text files. Its job is
to turn the height-integrated Honda flux Phi(E, cosZ) and the Honda
production-height quantiles into the height-differential source flux

    Phi(E,h) = Phi(E; X_obs) * f(h|E,alpha)

used as the upper boundary condition for atmosphere-neutrino propagation
through the atmosphere and Earth. Phi(E; X_obs) is interpolated (log-log in
energy, linear in cos(zenith)) from the Honda flux table onto a requested
energy grid and surface/detector angle pair; f(h|E,alpha) is reconstructed as the
derivative of the production-height cumulative distribution built from the
Honda quantiles (or, for particles without a Honda height table such as tau
neutrinos, approximated as a flat density over the requested height grid,
or set to zero flux for particles absent from the Honda flavour set).

Angle convention: theta is the detector zenith angle in degrees, while
alpha is the surface zenith angle in degrees. Honda calls its binning
coordinate theta, but within TPeanuts this surface/table angle is alpha,
and cosZ = cos(alpha). If only detector theta is provided, it is converted
to surface alpha via ``medium.atmosphere.geometry.theta_detector_to_alpha_surface``
before table lookup, to account for detector depth.

Module functions:
    generate_flux_for_particle_angle(...)
        Build and optionally save one height-differential flux file
        Phi(E,h) for a single particle and zenith/detector angle.
    generate_flux_for_particles_angle_grid(...)
        Run generate_flux_for_particle_angle over a grid of particles and
        angles, optionally in parallel, and group results by flavour.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Optional, Union

import numpy as np
import torch

from tpeanuts.external.honda.tables import (
    TPEANUTS_TO_HONDA,
    HondaTableSelection,
    _select_height_source,
    load_honda_tables,
)
from tpeanuts.external.honda.math import (
    M2_TO_CM2_FLUX,
    density_from_quantiles,
    flat_height_density,
    interpolate_flux_cm2,
    interpolate_quantiles,
)
from tpeanuts.util.parallel import ParallelConfig
from tpeanuts.medium.atmosphere.geometry import (
    alpha_surface_to_theta_detector,
    theta_detector_to_alpha_surface,
)
from tpeanuts.medium.atmosphere.io import (
    build_angle_output_path,
    OutputConfig,
    save_phi_Eh_theta_result,
)
from tpeanuts.util.parallel import run_task_dicts
from tpeanuts.util.torch_util import resolve_device, scalar_float
from tpeanuts.util.type import as_tensor


TensorLike = Union[float, int, torch.Tensor]


def _metadata_extra(
    *,
    alpha_deg: float,
    theta_deg: float,
    cosz: float,
    selection: HondaTableSelection,
    honda_data_dir: str,
    flux_table: dict[str, Any],
    height_table: Optional[dict[str, Any]],
    source_honda_flavour: Optional[str],
    synthetic_zero_flux: bool,
    build_time_sec: float,
) -> dict[str, Any]:
    """
    Assemble provenance/units metadata for one generated Honda flux file.

    Args:
        alpha_deg: Surface/Honda zenith angle in degrees.
        theta_deg: Detector zenith angle in degrees.
        cosz: cos(alpha_deg) used for table interpolation.
        selection: Honda table variant that was loaded.
        honda_data_dir: Resolved Honda data directory.
        flux_table: Parsed flux table dict (for its "path").
        height_table: Parsed height table dict, or None.
        source_honda_flavour: Honda flux column used, or None if the
            particle has no Honda flavour (synthetic zero flux).
        synthetic_zero_flux: Whether the flux was set to all zeros because
            no Honda flavour matched the particle.
        build_time_sec: Wall-clock seconds spent building this result.

    Returns:
        JSON-serializable metadata dictionary describing units, file
        provenance, and the angle/height reconstruction conventions used.
    """
    return {
        "description": "Atmosphere height-differential flux generated from Honda/HKKM tables.",
        "source": "Honda/HKKM",
        "honda_data_dir": honda_data_dir,
        "honda_flux_file": flux_table["path"],
        "honda_height_file": None if height_table is None else height_table["path"],
        "honda_flavour": source_honda_flavour,
        "honda_units_original": "(m^2 s sr GeV)^-1",
        "tpeanuts_flux_units": "(cm^2 s sr GeV)^-1",
        "honda_flux_m2_to_tpeanuts_cm2_factor": M2_TO_CM2_FLUX,
        "cosz_center": float(cosz),
        "theta_detector_deg": float(theta_deg),
        "alpha_surface_deg": float(alpha_deg),
        "alpha_honda_deg": float(alpha_deg),
        "angle_relation": "cosZ = cos(alpha_surface); theta_deg is the detector angle used by TPeanuts atmosphere geometry, and alpha_deg is the surface/Honda table angle.",
        "site_code": selection.site_code,
        "season_code": selection.season_code,
        "solar": selection.solar,
        "mountain": selection.mountain,
        "angular_mode": selection.angular_mode,
        "height_reconstruction": "f_Eh is reconstructed as dCDF/dh from Honda production-height quantiles and normalized on h_grid_km.",
        "synthetic_zero_flux": bool(synthetic_zero_flux),
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
    honda_data_dir: str | Path | None = None,
    selection: HondaTableSelection = HondaTableSelection(),
    tables: Optional[dict[str, Any]] = None,
    energy_grid_GeV: Optional[TensorLike] = None,
    h_grid_km: Optional[TensorLike] = None,
    output_config: Optional[OutputConfig] = None,
    save: bool = True,
    skip_existing: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
    debug: bool = False,
) -> dict[str, Any]:
    """
    Build one Honda-derived height-differential flux file Phi(E,h).

    This is the main entry point of the Honda generator. For a single
    particle and surface/detector angle pair, it interpolates the Honda
    height-integrated flux Phi(E; X_obs) and production-height density
    f(h|E,alpha) (see module docstring) from the loaded Honda tables, forms
    Phi(E,h) = Phi(E; X_obs) * f(h|E,alpha), and optionally saves the
    result as a tpeanuts torch flux file using
    ``medium.atmosphere.io.save_phi_Eh_theta_result``.

    Args:
        particle: tpeanuts particle/flavour name (e.g. "numu", "antinue").
            Particles without a matching Honda flux flavour get a
            synthetic all-zero flux; particles without a height table fall
            back to a flat height density (see _select_height_source).
        alpha_deg: Surface/Honda zenith angle in degrees. Mutually
            exclusive with theta_deg when specifying an angle grid.
        theta_deg: Detector zenith angle in degrees. If alpha_deg is not
            supplied, theta_deg is converted internally to surface alpha via
            theta_detector_to_alpha_surface using detector_depth_m.
        detector_depth_m: Detector depth below the Earth surface in metres,
            used in the theta <-> alpha conversion.
        flavour_name: Optional grouping label stored in the output
            filename/metadata; defaults to particle when omitted downstream.
        honda_data_dir: Optional directory hint forwarded to
            find_honda_data_dir, used only when tables is None.
        selection: Honda table variant to load when tables is None.
        tables: Optional pre-loaded tables dict from load_honda_tables, to
            avoid re-reading the Honda files for every angle in a grid.
        energy_grid_GeV: Optional neutrino energy grid in GeV. Defaults to
            the Honda table's native energy grid.
        h_grid_km: Optional altitude grid in km for the height axis.
            Defaults to 501 points linearly spaced from 0 to 120 km.
        output_config: Output directory/filename/dtype settings used when
            save=True.
        save: If True, write the result to disk via
            save_phi_Eh_theta_result.
        skip_existing: If True and the expected output file already exists
            (and output_config.overwrite is False), skip recomputation and
            return early with "skipped": True.
        device: Optional torch device for the returned tensors.
        dtype: Real torch dtype for the returned tensors.
        debug: If True, print progress/skip/save messages.

    Returns:
        Dictionary with the computed tensors and metadata. Always includes
        "particle", "flavour_name", "theta_deg" (detector angle) and
        "alpha_deg" (surface/Honda angle). On a full (non-skipped)
        computation, also includes
        "E_grid_GeV"/"E_grid" (energy grid in GeV), "h_grid_km" (height grid
        in km), "phi_E_obs"/"phi_E" (Phi(E; X_obs) in (cm^2 s sr GeV)^-1),
        "f_Eh"/"f_E_h" (normalized height density in 1/km),
        "phi_Eh"/"phi_E_h" (the product, the height-differential flux),
        "cosz" (cos(alpha_deg)), "metadata_extra" (provenance/units
        metadata), and "output_path" when save=True. When skip_existing
        short-circuits, the dictionary instead contains
        "output_path"/"skipped": True without the tensor fields.

    Raises:
        ValueError: If neither alpha_deg nor theta_deg is given, or if the
            resolved alpha_deg falls outside 0 <= alpha_deg < 90.
    """
    if theta_deg is None and alpha_deg is None:
        raise ValueError("Provide either alpha_deg or theta_deg.")

    dev = resolve_device(device)

    if output_config is None:
        output_config = OutputConfig(output_dir="honda_height_flux_outputs", filename="diff_flux.pt")
    output_config.validate()

    depth_km = detector_depth_m / 1.0e3
    alpha_value = None
    if alpha_deg is None:
        theta_value = scalar_float(theta_deg)
        alpha_t = theta_detector_to_alpha_surface(
            theta_deg,
            depth_km,
            device=dev,
            dtype=dtype,
        )
        alpha_value = scalar_float(alpha_t)
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

    if not (0.0 <= alpha_value < 90.0):
        raise ValueError("Honda source generation expects 0 <= alpha_deg < 90.")

    output_path = None
    if save:
        output_path = build_angle_output_path(
            output_config=output_config,
            alpha_deg=alpha_value,
            particle=particle,
            flavour_name=flavour_name,
        )
        if skip_existing and Path(output_path).exists() and not output_config.overwrite:
            if debug:
                print(
                    f"Skipping existing {particle} "
                    f"theta_detector={theta_value:.3f} "
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

    if tables is None:
        tables = load_honda_tables(
            honda_data_dir=honda_data_dir,
            selection=selection,
            particles=[particle],
        )

    flux_table = tables["flux_table"]
    height_tables = tables["height_tables"]
    data_dir = tables["data_dir"]

    if energy_grid_GeV is None:
        energy_np = np.asarray(flux_table["energy_GeV"], dtype=float)
    else:
        energy_np = as_tensor(energy_grid_GeV, device="cpu", dtype=torch.float64).numpy()

    if h_grid_km is None:
        h_np = np.linspace(0.0, 120.0, 501, dtype=float)
    else:
        h_np = as_tensor(h_grid_km, device="cpu", dtype=torch.float64).numpy()

    cosz = float(np.cos(np.deg2rad(alpha_value)))
    honda_flavour = TPEANUTS_TO_HONDA.get(str(particle).lower())
    synthetic_zero_flux = honda_flavour is None

    t0 = time.perf_counter()

    if honda_flavour is None:
        phi_E_obs_np = np.zeros(energy_np.size, dtype=float)
    else:
        phi_E_obs_np = interpolate_flux_cm2(
            flux_table,
            honda_flavour=honda_flavour,
            cosz=cosz,
            energy_grid=energy_np,
        )

    height_table = _select_height_source(height_tables, particle)
    if height_table is None:
        f_Eh_np = flat_height_density(energy_np, h_np)
    else:
        probabilities, quantiles = interpolate_quantiles(
            height_table,
            cosz=cosz,
            energy_grid=energy_np,
        )
        f_Eh_np = density_from_quantiles(probabilities, quantiles, h_np)

    phi_Eh_np = phi_E_obs_np[:, None] * f_Eh_np
    build_time_sec = time.perf_counter() - t0

    E = torch.as_tensor(energy_np, device=dev, dtype=dtype)
    h = torch.as_tensor(h_np, device=dev, dtype=dtype)
    phi_E_obs = torch.as_tensor(phi_E_obs_np, device=dev, dtype=dtype)
    f_Eh = torch.as_tensor(f_Eh_np, device=dev, dtype=dtype)
    phi_Eh = torch.as_tensor(phi_Eh_np, device=dev, dtype=dtype)

    result = {
        "particle": particle,
        "flavour_name": flavour_name,
        "theta_deg": torch.as_tensor(theta_value, device=dev, dtype=dtype),
        "alpha_deg": torch.as_tensor(alpha_value, device=dev, dtype=dtype),
        "alpha_honda_deg": torch.as_tensor(alpha_value, device=dev, dtype=dtype),
        "E_grid_GeV": E,
        "E_grid": E,
        "h_grid_km": h,
        "phi_Eh": phi_Eh,
        "phi_E_h": phi_Eh,
        "phi_E_obs": phi_E_obs,
        "phi_E": phi_E_obs,
        "f_Eh": f_Eh,
        "f_E_h": f_Eh,
        "cosz": torch.as_tensor(cosz, device=dev, dtype=dtype),
        "metadata_extra": _metadata_extra(
            alpha_deg=alpha_value,
            theta_deg=theta_value,
            cosz=cosz,
            selection=selection,
            honda_data_dir=data_dir,
            flux_table=flux_table,
            height_table=height_table,
            source_honda_flavour=honda_flavour,
            synthetic_zero_flux=synthetic_zero_flux,
            build_time_sec=build_time_sec,
        ),
    }

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
            print(f"Saved Honda {particle}: {output_path}. Time: {build_time_sec:.3f} s.")

    return result


@torch.no_grad()
def generate_flux_for_particles_angle_grid(
    particles: dict[str, str] | list[str] | tuple[str, ...],
    *,
    alpha_grid_deg: Optional[TensorLike] = None,
    theta_grid_deg: Optional[TensorLike] = None,
    detector_depth_m: float = 0.0,
    honda_data_dir: str | Path | None = None,
    selection: HondaTableSelection = HondaTableSelection(),
    energy_grid_GeV: Optional[TensorLike] = None,
    h_grid_km: Optional[TensorLike] = None,
    output_config: Optional[OutputConfig] = None,
    parallel_config: Optional[ParallelConfig] = None,
    save: bool = True,
    skip_existing: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
    debug: bool = False,
) -> dict[str, Any]:
    """
    Build Honda-derived height-differential flux files over a particle/angle grid.

    Runs generate_flux_for_particle_angle for every combination of the
    requested particles and angle grid, sharing one set of loaded Honda
    tables across all of them, optionally in parallel
    (see ``util.parallel.run_task_dicts``), and groups the per-angle results
    by flavour.

    Args:
        particles: Either a dict mapping flavour_name -> particle name, or
            a list/tuple of particle names (in which case flavour_name
            equals particle for each).
        alpha_grid_deg: 1D grid of surface/Honda zenith angles in degrees.
            Mutually exclusive with theta_grid_deg.
        theta_grid_deg: 1D grid of detector zenith angles in degrees.
            Mutually exclusive with alpha_grid_deg.
        detector_depth_m: Detector depth below the Earth surface in metres,
            used in the theta <-> alpha conversion.
        honda_data_dir: Optional directory hint forwarded to
            find_honda_data_dir.
        selection: Honda table variant to load.
        energy_grid_GeV: Optional neutrino energy grid in GeV, forwarded to
            every per-angle call.
        h_grid_km: Optional altitude grid in km, forwarded to every
            per-angle call.
        output_config: Output directory/filename/dtype settings.
        parallel_config: Parallel-execution settings for the per-task grid;
            defaults to sequential execution.
        save: If True, write each result to disk.
        skip_existing: If True, skip recomputation for files that already
            exist.
        device: Optional torch device for the returned tensors.
        dtype: Real torch dtype for the returned tensors.
        debug: If True, print progress and per-job summary messages.

    Returns:
        Dictionary keyed by flavour_name. Each value is a dictionary with
        "particle", "flavour_name", "angle_mode" ("alpha" or "theta"),
        "input_angle_grid_deg" (the requested angle tensor), and "results"
        (a dict mapping the resolved detector theta_deg float to the corresponding
        generate_flux_for_particle_angle result dictionary).

    Raises:
        ValueError: If neither or both of alpha_grid_deg/theta_grid_deg are
            given.
    """
    if alpha_grid_deg is None and theta_grid_deg is None:
        raise ValueError("Provide either alpha_grid_deg or theta_grid_deg.")
    if alpha_grid_deg is not None and theta_grid_deg is not None:
        raise ValueError("Use either alpha_grid_deg or theta_grid_deg, not both.")

    if isinstance(particles, dict):
        particle_items = list(particles.items())
    else:
        particle_items = [(str(particle), str(particle)) for particle in particles]

    angle_mode = "alpha" if alpha_grid_deg is not None else "theta"
    angle_grid = alpha_grid_deg if alpha_grid_deg is not None else theta_grid_deg
    angle_grid = as_tensor(angle_grid, device="cpu", dtype=torch.float64).reshape(-1)

    output_config = output_config or OutputConfig(
        output_dir="honda_height_flux_outputs",
        filename="diff_flux.pt",
    )
    output_config.validate()

    if parallel_config is None:
        parallel_config = ParallelConfig(parallel=False)
    parallel_config.validate()

    particles_for_tables = [particle for _flavour_name, particle in particle_items]
    tables = load_honda_tables(
        honda_data_dir=honda_data_dir,
        selection=selection,
        particles=particles_for_tables,
    )

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
                    "honda_data_dir": honda_data_dir,
                    "selection": selection,
                    "tables": tables,
                    "energy_grid_GeV": energy_grid_GeV,
                    "h_grid_km": h_grid_km,
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
            f"Generating {len(tasks)} Honda flux jobs "
            f"({len(particle_items)} particles x {angle_grid.numel()} angles). "
            f"parallel={parallel_config.parallel}"
        )

    results_flat = run_task_dicts(
        func=generate_flux_for_particle_angle,
        tasks=tasks,
        config=parallel_config,
        show_progress=debug,
        desc="Honda flux generation",
    )

    grouped: dict[str, dict[str, Any]] = {}
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

        theta = result.get("theta_deg", task.get("theta_deg"))
        grouped[flavour_name]["results"][scalar_float(theta)] = result

    return grouped

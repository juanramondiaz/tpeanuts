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

    Phi(E,h) = Phi(E; X_obs) * f(h|E,theta)

used as the upper boundary condition for atmosphere-neutrino propagation
through the atmosphere and Earth. Phi(E; X_obs) is interpolated (log-log in
energy, linear in cos(zenith)) from the Honda flux table onto a requested
energy grid and zenith/detector angle; f(h|E,theta) is reconstructed as the
derivative of the production-height cumulative distribution built from the
Honda quantiles (or, for particles without a Honda height table such as tau
neutrinos, approximated as a flat density over the requested height grid,
or set to zero flux for particles absent from the Honda flavour set).

Angle convention: theta is the MCEq/TPeanuts atmosphere zenith angle in
degrees (0 = vertically downward, 90 = horizontal), and cosZ = cos(theta)
is the Honda binning coordinate. Detector angle alpha (also in degrees) is
optionally converted to a surface theta via
``medium.atmosphere.geometry.detector_alpha_to_surface_theta`` before table
lookup, to account for detector depth.

Module functions:
    load_honda_tables(...)
        Locate and parse the Honda flux table and per-particle
        production-height tables needed for a set of particles.
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
    HONDA_TO_TPEANUTS,
    TPEANUTS_TO_HONDA,
    HondaTableSelection,
    choose_flux_file,
    choose_height_file,
    find_honda_data_dir,
    read_flux_table,
    read_height_table,
)
from tpeanuts.util.parallel import ParallelConfig
from tpeanuts.medium.atmosphere.geometry import detector_alpha_to_surface_theta
from tpeanuts.medium.atmosphere.io import (
    build_angle_output_path,
    OutputConfig,
    save_phi_Eh_theta_result,
)
from tpeanuts.util.parallel import run_task_dicts
from tpeanuts.util.torch_util import default_device
from tpeanuts.util.type import as_tensor


TensorLike = Union[float, int, torch.Tensor]
M2_TO_CM2_FLUX = 1.0e-4


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if callable(device):
        device = device()
    return default_device(device)


def _scalar_float(value: TensorLike) -> float:
    value_t = as_tensor(value, device="cpu", dtype=torch.float64)
    return float(value_t.detach().cpu().reshape(-1)[0].item())


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
        alpha_deg=alpha_deg,
        particle=particle,
        flavour_name=flavour_name,
    )


def _interp_logx(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Linearly interpolate fp(xp) in log10(x), matching Honda's log-spaced energy grid."""
    return np.interp(np.log10(x), np.log10(xp), fp)


def _interpolate_flux_cm2(
    flux_table: dict[str, Any],
    *,
    honda_flavour: str,
    cosz: float,
    energy_grid: np.ndarray,
) -> np.ndarray:
    """
    Interpolate a Honda flux table onto a requested energy grid and cos(zenith).

    Interpolation is log-linear in energy (matching the Honda grid spacing)
    and linear in cos(zenith) between the 20 Honda zenith bin centers.

    Args:
        flux_table: Parsed Honda flux table, as returned by
            external.honda.tables.read_flux_table.
        honda_flavour: Honda flux column name (e.g. "NuMu", "NuEbar").
        cosz: cos(zenith) at the detector; cosz=1 is vertically downward.
        energy_grid: 1D array of neutrino energies in GeV at which to
            evaluate the flux.

    Returns:
        1D array of differential flux in (cm^2 s sr GeV)^-1 (converted from
        the Honda (m^2 s sr GeV)^-1 units via M2_TO_CM2_FLUX), matching
        energy_grid.
    """
    source_energy = flux_table["energy_GeV"]
    source_cosz = flux_table["cosz_center"]
    values = flux_table["flux_m2"][honda_flavour]

    by_cosz = np.empty((source_cosz.size, energy_grid.size), dtype=float)
    for iz in range(source_cosz.size):
        by_cosz[iz] = _interp_logx(energy_grid, source_energy, values[iz])

    flux_m2 = np.empty(energy_grid.size, dtype=float)
    for ie in range(energy_grid.size):
        flux_m2[ie] = np.interp(cosz, source_cosz, by_cosz[:, ie])

    return flux_m2 * M2_TO_CM2_FLUX


def _select_height_source(
    height_tables: dict[str, Optional[dict[str, Any]]],
    particle: str,
) -> Optional[dict[str, Any]]:
    """
    Pick a production-height table for a particle, with a tau-neutrino fallback.

    Honda does not publish production-height tables for tau neutrinos
    (they are not produced directly in cosmic-ray air showers). As an
    approximation, nutau/antinutau fall back to the numu/antinumu height
    table, since both come from the same hadronic shower region.

    Args:
        height_tables: Mapping from particle name to parsed Honda
            production-height table (or None if unavailable).
        particle: tpeanuts particle/flavour name.

    Returns:
        The matching production-height table dictionary, or None if no
        table (including no fallback) is available for this particle.
    """
    if height_tables.get(particle) is not None:
        return height_tables[particle]

    key = str(particle).lower()
    if "tau" in key:
        return height_tables.get("numu") if "anti" not in key else height_tables.get("antinumu")

    return None


def _interpolate_quantiles(
    height_table: dict[str, Any],
    *,
    cosz: float,
    energy_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate Honda production-height quantiles onto an energy grid and cos(zenith).

    Each quantile (e.g. the 10%, 20%, ... height of the production-height
    cumulative distribution) is interpolated independently: log-linear in
    energy, then linear in cos(zenith) between Honda zenith bin centers.

    Args:
        height_table: Parsed Honda production-height table, as returned by
            external.honda.tables.read_height_table.
        cosz: cos(zenith) at the detector.
        energy_grid: 1D array of neutrino energies in GeV.

    Returns:
        Pair (probabilities, q) where probabilities is the 1D array of
        quantile probability levels in [0, 1] and q is a 2D array of shape
        (n_energy, n_probabilities) with the corresponding production
        heights in km.
    """
    source_energy = height_table["energy_GeV"]
    source_cosz = height_table["cosz_center"]
    probabilities = height_table["probabilities"]
    quantiles = height_table["height_quantiles_km"]

    q_energy = np.empty((source_cosz.size, energy_grid.size, probabilities.size), dtype=float)
    for iz in range(source_cosz.size):
        for ip in range(probabilities.size):
            q_energy[iz, :, ip] = _interp_logx(
                energy_grid,
                source_energy,
                quantiles[iz, :, ip],
            )

    # Guarantee ascending xp for np.interp (Honda height tables are keyed by
    # zenith_bin, which yields *decreasing* cosz if not re-sorted at read time).
    sort_idx = np.argsort(source_cosz)
    source_cosz_asc = source_cosz[sort_idx]
    q_energy_asc = q_energy[sort_idx]

    q = np.empty((energy_grid.size, probabilities.size), dtype=float)
    for ie in range(energy_grid.size):
        for ip in range(probabilities.size):
            q[ie, ip] = np.interp(cosz, source_cosz_asc, q_energy_asc[:, ie, ip])

    return probabilities, q


def _density_from_quantiles(
    probabilities: np.ndarray,
    quantile_heights_km: np.ndarray,
    h_grid_km: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct a normalized production-height density f(h|E) from quantiles.

    For each energy row, the (probability, height) quantile pairs define a
    piecewise-linear cumulative distribution function (CDF) of production
    height, anchored at h=0 (CDF=0) and at the top of the grid (CDF=1). The
    density is obtained as the numerical derivative dCDF/dh on h_grid_km,
    clipped to be non-negative and renormalized so that its integral over
    h_grid_km equals 1. If the quantile data is degenerate (zero or
    non-finite normalization), a narrow delta-like density at h=0 is used
    instead.

    Args:
        probabilities: 1D array of quantile probability levels in [0, 1].
        quantile_heights_km: 2D array of shape (n_energy, n_probabilities)
            with production-height quantiles in km, one row per energy.
        h_grid_km: 1D altitude grid in km on which to evaluate the density.

    Returns:
        2D array of shape (n_energy, h_grid_km.size) with the
        height-density f(h|E) in 1/km, normalized so that
        integral f(h|E) dh = 1 for each energy row.
    """
    f = np.zeros((quantile_heights_km.shape[0], h_grid_km.size), dtype=float)

    for ie, heights in enumerate(quantile_heights_km):
        order = np.argsort(heights)
        h_sorted = heights[order]
        p_sorted = probabilities[order]

        h_nodes = np.concatenate(([0.0], h_sorted, [max(float(h_grid_km[-1]), float(h_sorted[-1]))]))
        p_nodes = np.concatenate(([0.0], p_sorted, [1.0]))

        unique_h, unique_indices = np.unique(h_nodes, return_index=True)
        unique_p = p_nodes[unique_indices]
        cdf = np.interp(h_grid_km, unique_h, unique_p, left=0.0, right=1.0)
        density = np.gradient(cdf, h_grid_km, edge_order=1)
        density = np.clip(density, 0.0, None)
        norm = np.trapz(density, x=h_grid_km)

        if norm > 0.0 and np.isfinite(norm):
            density = density / norm
        else:
            density = np.zeros_like(h_grid_km)
            density[0] = 1.0 / max(float(h_grid_km[1] - h_grid_km[0]), 1.0)
            density = density / np.trapz(density, x=h_grid_km)

        f[ie] = density

    return f


def _flat_height_density(energy_grid: np.ndarray, h_grid_km: np.ndarray) -> np.ndarray:
    """
    Build a uniform (energy-independent) production-height density.

    Used as a fallback when no Honda production-height table is available
    for a particle (e.g. no flavour-specific table and no tau-neutrino
    fallback). Approximates ignorance of the true height distribution by
    spreading the flux uniformly over the requested height grid.

    Args:
        energy_grid: 1D array of neutrino energies in GeV (only its size is
            used).
        h_grid_km: 1D altitude grid in km.

    Returns:
        2D array of shape (energy_grid.size, h_grid_km.size), uniform along
        the height axis and normalized so that integral f(h) dh = 1 for
        each energy row.
    """
    density = np.ones((energy_grid.size, h_grid_km.size), dtype=float)
    norm = np.trapz(density, x=h_grid_km, axis=1)
    return density / norm[:, None]


def load_honda_tables(
    *,
    honda_data_dir: str | Path | None,
    selection: HondaTableSelection,
    particles: list[str],
) -> dict[str, Any]:
    """
    Locate and parse the Honda flux table and per-particle height tables.

    This loads the data once so that it can be reused across many
    particle/angle combinations (see generate_flux_for_particles_angle_grid),
    instead of re-reading the gzip files for every call.

    Args:
        honda_data_dir: Optional directory hint forwarded to
            find_honda_data_dir.
        selection: Honda table variant (site, season, solar, mountain,
            angular mode) to load.
        particles: List of tpeanuts particle/flavour names for which a
            production-height table should be loaded (entries without a
            matching Honda height table are stored as None).

    Returns:
        Dictionary with keys "data_dir" (resolved Honda data directory as
        str), "flux_table" (parsed flux table dict, see
        external.honda.tables.read_flux_table), and "height_tables" (dict
        mapping each requested particle to its parsed production-height
        table, or None).
    """
    data_dir = find_honda_data_dir(honda_data_dir)
    flux_path = choose_flux_file(data_dir, selection)
    flux_table = read_flux_table(flux_path)

    height_tables: dict[str, Optional[dict[str, Any]]] = {}
    for particle in particles:
        path = choose_height_file(data_dir, selection, particle)
        height_tables[particle] = read_height_table(path) if path is not None else None

    return {
        "data_dir": str(data_dir),
        "flux_table": flux_table,
        "height_tables": height_tables,
    }


def _metadata_extra(
    *,
    alpha_deg: Optional[float],
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
        alpha_deg: Detector angle in degrees, if used, else None.
        theta_deg: Resolved atmosphere surface zenith angle in degrees.
        cosz: cos(theta_deg) used for table interpolation.
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
        "theta_surface_deg": float(theta_deg),
        "theta_mceq_deg": float(theta_deg),
        "alpha_detector_deg": alpha_deg,
        "angle_relation": "cosZ = cos(theta_surface); theta_surface is the MCEq/TPeanuts Atmosphere zenith angle. If alpha is provided, theta_surface is computed from detector alpha and detector depth.",
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
    particle and zenith/detector angle, it interpolates the Honda
    height-integrated flux Phi(E; X_obs) and production-height density
    f(h|E,theta) (see module docstring) from the loaded Honda tables, forms
    Phi(E,h) = Phi(E; X_obs) * f(h|E,theta), and optionally saves the
    result as a tpeanuts torch flux file using
    ``medium.atmosphere.io.save_phi_Eh_theta_result``.

    Args:
        particle: tpeanuts particle/flavour name (e.g. "numu", "antinue").
            Particles without a matching Honda flux flavour get a
            synthetic all-zero flux; particles without a height table fall
            back to a flat height density (see _select_height_source).
        alpha_deg: Detector zenith angle in degrees. Mutually exclusive
            with theta_deg; converted internally to a surface theta via
            detector_alpha_to_surface_theta using detector_depth_m.
        theta_deg: Atmosphere surface zenith angle in degrees (MCEq
            convention: 0 = vertically downward, 90 = horizontal). Mutually
            exclusive with alpha_deg. Must satisfy 0 <= theta_deg < 90.
        detector_depth_m: Detector depth below the Earth surface in metres,
            used only when alpha_deg is given.
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
        "particle", "flavour_name", "theta_deg" (and "alpha_deg" when
        applicable). On a full (non-skipped) computation, also includes
        "E_grid_GeV"/"E_grid" (energy grid in GeV), "h_grid_km" (height grid
        in km), "phi_E_obs"/"phi_E" (Phi(E; X_obs) in (cm^2 s sr GeV)^-1),
        "f_Eh"/"f_E_h" (normalized height density in 1/km),
        "phi_Eh"/"phi_E_h" (the product, the height-differential flux),
        "cosz" (cos(theta_deg)), "metadata_extra" (provenance/units
        metadata), and "output_path" when save=True. When skip_existing
        short-circuits, the dictionary instead contains
        "output_path"/"skipped": True without the tensor fields.

    Raises:
        ValueError: If neither alpha_deg nor theta_deg is given, or if the
            resolved theta_deg falls outside 0 <= theta_deg < 90.
    """
    if theta_deg is None and alpha_deg is None:
        raise ValueError("Provide either alpha_deg or theta_deg.")

    dev = _resolve_device(device)

    if output_config is None:
        output_config = OutputConfig(output_dir="honda_height_flux_outputs", filename="diff_flux.pt")
    output_config.validate()

    alpha_value = None
    if theta_deg is None:
        theta_t = detector_alpha_to_surface_theta(
            alpha_deg,
            detector_depth_m=detector_depth_m,
            device=dev,
            dtype=dtype,
        )
        alpha_value = _scalar_float(alpha_deg)
        theta_value = _scalar_float(theta_t)
    else:
        theta_value = _scalar_float(theta_deg)
        if alpha_deg is not None:
            alpha_value = _scalar_float(alpha_deg)

    if not (0.0 <= theta_value < 90.0):
        raise ValueError("Honda source generation expects 0 <= theta_deg < 90.")

    output_path = None
    if save:
        output_path = _expected_output_path(
            output_config=output_config,
            particle=particle,
            theta_deg=theta_value,
            alpha_deg=alpha_value,
            flavour_name=flavour_name,
        )
        if skip_existing and Path(output_path).exists() and not output_config.overwrite:
            if debug:
                print(f"Skipping existing {particle} theta={theta_value:.3f}: {output_path}")
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

    cosz = float(np.cos(np.deg2rad(theta_value)))
    honda_flavour = TPEANUTS_TO_HONDA.get(str(particle).lower())
    synthetic_zero_flux = honda_flavour is None

    t0 = time.perf_counter()

    if honda_flavour is None:
        phi_E_obs_np = np.zeros(energy_np.size, dtype=float)
    else:
        phi_E_obs_np = _interpolate_flux_cm2(
            flux_table,
            honda_flavour=honda_flavour,
            cosz=cosz,
            energy_grid=energy_np,
        )

    height_table = _select_height_source(height_tables, particle)
    if height_table is None:
        f_Eh_np = _flat_height_density(energy_np, h_np)
    else:
        probabilities, quantiles = _interpolate_quantiles(
            height_table,
            cosz=cosz,
            energy_grid=energy_np,
        )
        f_Eh_np = _density_from_quantiles(probabilities, quantiles, h_np)

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

    if alpha_value is not None:
        result["alpha_deg"] = torch.as_tensor(alpha_value, device=dev, dtype=dtype)

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


def _generate_particle_angle_task(**kwargs):
    """Keyword-only adapter so generate_flux_for_particle_angle can run as a parallel task."""
    return generate_flux_for_particle_angle(**kwargs)


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
        alpha_grid_deg: 1D grid of detector zenith angles in degrees.
            Mutually exclusive with theta_grid_deg.
        theta_grid_deg: 1D grid of atmosphere surface zenith angles in
            degrees. Mutually exclusive with alpha_grid_deg.
        detector_depth_m: Detector depth below the Earth surface in metres,
            used only with alpha_grid_deg.
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
        (a dict mapping the resolved theta_deg float to the corresponding
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
        func=_generate_particle_angle_task,
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
        grouped[flavour_name]["results"][_scalar_float(theta)] = result

    return grouped

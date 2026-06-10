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
Run 2: propagate mceq_run1 production fluxes to detector-level fluxes.

Input:
    tpeanuts/data/flux/mceq_run1

Output:
    tpeanuts/data/flux/detector1

One torch file is written for every produced particle and every detector alpha
or theta value. Each output stores flux(E, final_flavour) after:

    mceq production profile -> atmosphere propagation -> earth propagation
    -> integration over production height h.
"""



from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch


THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parents[3]


from tpeanuts.core.pmns import PMNS  # noqa: E402
from tpeanuts.io.io_earth import load_earth_density_from_csv  # noqa: E402
from tpeanuts.flux_propagation.pipeline_atmosphere import (  # noqa: E402
    integrate_height_and_sum_flavours,
    integrate_initial_and_surface_fluxes,
    propagate_atmosphere_coherent,
    propagate_earth_coherent,
    select_particle_angle_flux,
)
from tpeanuts.io.io_flux_propagation import (  # noqa: E402
    build_detector_flux_path,
    save_detector_flux_result,
)
from tpeanuts.io.io_atmosphere import load_directory  # noqa: E402
from tpeanuts.util.torch_util import _default_device, resolve_device  # noqa: E402
from tpeanuts.util.type import _as_tensor  # noqa: E402


# ============================================================
# Input / output
# ============================================================
DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_DATA_ROOT = Path(OUTPUT_ROOT / "data")
OUTPUT_ATMOSPHERE_ROOT = Path(OUTPUT_DATA_ROOT / "atmosphere")

OUTPUT_MCEQ_ROOT = Path(OUTPUT_ATMOSPHERE_ROOT / "mceq")

INPUT_DIR = str(OUTPUT_MCEQ_ROOT / "mceq_flux_diff_002")

DETECTOR_ROOT = Path(OUTPUT_ATMOSPHERE_ROOT / "detector")
OUTPUT_DIR = str(DETECTOR_ROOT / "detector_flux_002")

OUTPUT_FILENAME = "detector_flux.pt"
EARTH_DENSITY_FILE = str(PACKAGE_DIR / "data" / "density" / "earth_density.csv")

OVERWRITE = False
SAVE_DTYPE = torch.float32


# ============================================================
# Physics
# ============================================================

DM21_EV2 = 7.42e-5
DM3L_EV2 = 2.517e-3

THETA12 = 0.59
THETA13 = 0.15
THETA23 = 0.78
DELTA_CP = 1.20

DETECTOR_DEPTH_M = 1000.0
MATTER_IN_atmosphere = True
REUNITARIZE_earth = False


# ============================================================
# Runtime / batching controls
# ============================================================

LOAD_DEVICE = "cpu"
PROPAGATION_DEVICE = _default_device
COMPUTE_DTYPE = torch.float64
LOAD_DTYPE = torch.float64

# Energy chunking is the main memory knob. Propagation is batched over
# (energy_chunk, all_heights), so memory grows roughly with:
#     ENERGY_CHUNK_SIZE * n_h * atmosphere_n_steps
# Use None to process all energies in one batch.
ENERGY_CHUNK_SIZE: Optional[int] = 8

# Height chunking is intentionally not used by default because integration over
# h is more accurate and simpler when every production height is available in
# the same trapezoid call.
HEIGHT_MAX_COUNT: Optional[int] = None
ENERGY_MAX_COUNT: Optional[int] = None
ANGLE_MAX_COUNT: Optional[int] = None

atmosphere_STEPS = 120
TRAJECTORY_STEPS = 120

PARTICLES: Optional[tuple[str, ...]] = None
SKIP_EXISTING = True
DEBUG = True


def evenly_spaced_indices(n_items: int, max_count: Optional[int] = None) -> torch.Tensor:
    if max_count is None or max_count >= n_items:
        return torch.arange(n_items, dtype=torch.long)

    return torch.linspace(
        0,
        n_items - 1,
        int(max_count),
        dtype=torch.float64,
    ).round().to(torch.long).unique()


def chunk_indices(indices: torch.Tensor, chunk_size: Optional[int]):
    indices = indices.reshape(-1).to(dtype=torch.long)

    if chunk_size is None or chunk_size <= 0:
        yield indices
        return

    for start in range(0, indices.numel(), int(chunk_size)):
        yield indices[start:start + int(chunk_size)]


def infer_antinu(particle: str) -> bool:
    key = str(particle).lower()
    return "anti" in key or "bar" in key


def angle_value(selected: Dict[str, object]) -> tuple[Optional[float], float]:
    alpha = selected.get("alpha_deg", None)
    theta = selected.get("theta_deg", None)

    alpha_value = None if alpha is None else float(alpha)
    theta_value = float(theta)

    return alpha_value, theta_value


def thin_selected(
    selected: Dict[str, object],
    *,
    energy_indices: torch.Tensor,
    height_indices: torch.Tensor,
) -> Dict[str, object]:
    thinned = dict(selected)

    E = selected["E_grid_GeV"][energy_indices]
    h = selected["h_grid_km"][height_indices]
    phi_Eh = selected["phi_Eh"][energy_indices][:, height_indices]

    thinned["E_grid_GeV"] = E
    thinned["h_grid_km"] = h
    thinned["phi_Eh"] = phi_Eh
    thinned["phi_E_theta"] = torch.trapezoid(phi_Eh, x=h, dim=-1)

    if selected.get("f_Eh", None) is not None:
        thinned["f_Eh"] = selected["f_Eh"][energy_indices][:, height_indices]

    return thinned


def propagate_particle_angle(
    *,
    selected: Dict[str, object],
    pmns: PMNS,
    density,
    dm21: torch.Tensor,
    dm3l: torch.Tensor,
    antinu: bool,
    energy_indices: torch.Tensor,
    height_indices: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    detector_flux_chunks = []
    surface_flux_chunks = []
    initial_flux_chunks = []
    detector_probability_chunks = []
    surface_probability_chunks = []
    source_flux_chunks = []
    E_chunks = []

    for chunk in chunk_indices(energy_indices, ENERGY_CHUNK_SIZE):
        selected_chunk = thin_selected(
            selected,
            energy_indices=chunk,
            height_indices=height_indices,
        )

        atmosphere = propagate_atmosphere_coherent(
            selected_chunk,
            pmns,
            dm21,
            dm3l,
            detector_depth_m=DETECTOR_DEPTH_M,
            antinu=antinu,
            matter=MATTER_IN_atmosphere,
            atmosphere_n_steps=atmosphere_STEPS,
            trajectory_steps=TRAJECTORY_STEPS,
            device=device,
            dtype=dtype,
            debug=False,
        )

        earth = propagate_earth_coherent(
            atmosphere,
            pmns,
            dm21,
            dm3l,
            detector_depth_m=DETECTOR_DEPTH_M,
            density=density,
            antinu=antinu,
            reunitarize_earth=REUNITARIZE_earth,
            device=device,
            dtype=dtype,
            debug=False,
        )

        source_surface = integrate_initial_and_surface_fluxes(
            {selected_chunk["particle"]: selected_chunk},
            {selected_chunk["particle"]: atmosphere},
            device=device,
            dtype=dtype,
        )

        integrated = integrate_height_and_sum_flavours(
            {selected_chunk["particle"]: earth},
            device=device,
            dtype=dtype,
        )

        particle = selected_chunk["particle"]
        detector_flux = integrated["detector_flux_total_Ei"]
        initial_flux = source_surface["initial_flux_by_beta"][particle]
        surface_flux = source_surface["surface_flux_by_beta"][particle]
        surface_probability = source_surface["surface_probability_by_beta"][particle]
        source_flux = torch.sum(initial_flux, dim=-1)
        detector_probability = detector_flux / source_flux[..., None].clamp_min(
            torch.finfo(dtype).tiny
        )

        E_chunks.append(selected_chunk["E_grid_GeV"])
        detector_flux_chunks.append(detector_flux)
        surface_flux_chunks.append(surface_flux)
        initial_flux_chunks.append(initial_flux)
        detector_probability_chunks.append(detector_probability)
        surface_probability_chunks.append(surface_probability)
        source_flux_chunks.append(source_flux)

    return {
        "E_grid_GeV": torch.cat(E_chunks, dim=0),
        "detector_flux_Ei": torch.cat(detector_flux_chunks, dim=0),
        "surface_flux_Ei": torch.cat(surface_flux_chunks, dim=0),
        "initial_flux_Ei": torch.cat(initial_flux_chunks, dim=0),
        "detector_probability_Ei": torch.cat(detector_probability_chunks, dim=0),
        "surface_probability_Ei": torch.cat(surface_probability_chunks, dim=0),
        "source_flux_E": torch.cat(source_flux_chunks, dim=0),
    }


def main():
    load_device = resolve_device(LOAD_DEVICE)
    propagation_device = resolve_device(PROPAGATION_DEVICE)

    print("\nDetector flux propagation")
    print(f"Input directory      : {INPUT_DIR}")
    print(f"Output directory     : {OUTPUT_DIR}")
    print(f"earth density file   : {EARTH_DENSITY_FILE}")
    print(f"Load device          : {load_device}")
    print(f"Propagation device   : {propagation_device}")
    print(f"Energy chunk size    : {ENERGY_CHUNK_SIZE}")
    print(f"atmosphere steps     : {atmosphere_STEPS}")
    print(f"Trajectory steps     : {TRAJECTORY_STEPS}")

    flux_data = load_directory(
        INPUT_DIR,
        map_location="cpu",
        dtype=LOAD_DTYPE,
        device=load_device,
        group_by="particle",
        verbose=DEBUG,
    )

    particles = list(PARTICLES) if PARTICLES is not None else sorted(flux_data.keys())

    pmns = PMNS(
        THETA12,
        THETA13,
        THETA23,
        DELTA_CP,
        device=propagation_device,
        real_dtype=COMPUTE_DTYPE,
    )

    density = load_earth_density_from_csv(
        EARTH_DENSITY_FILE,
        device=propagation_device,
        dtype=COMPUTE_DTYPE,
    )

    dm21 = torch.as_tensor(DM21_EV2, device=propagation_device, dtype=COMPUTE_DTYPE)
    dm3l = torch.as_tensor(DM3L_EV2, device=propagation_device, dtype=COMPUTE_DTYPE)

    saved_paths = []
    t0 = time.perf_counter()

    for particle in particles:
        if particle not in flux_data:
            print(f"Skipping missing particle: {particle}")
            continue

        group = flux_data[particle]
        n_angles = int(group["theta_grid_deg"].numel())
        angle_indices = evenly_spaced_indices(n_angles, ANGLE_MAX_COUNT)
        energy_indices_full = evenly_spaced_indices(
            int(group["E_grid_GeV"].numel()),
            ENERGY_MAX_COUNT,
        )
        height_indices = evenly_spaced_indices(
            int(group["h_grid_km"].numel()),
            HEIGHT_MAX_COUNT,
        )

        for i_angle, angle_index in enumerate(angle_indices.tolist()):
            selected = select_particle_angle_flux(
                flux_data,
                particle,
                angle_index=angle_index,
                device=propagation_device,
                dtype=COMPUTE_DTYPE,
            )

            alpha_deg, theta_deg = angle_value(selected)
            output_path = build_detector_flux_path(
                OUTPUT_DIR,
                particle,
                alpha_deg=alpha_deg,
                theta_deg=theta_deg,
                base_filename=OUTPUT_FILENAME,
            )

            if SKIP_EXISTING and output_path and Path(output_path).exists() and not OVERWRITE:
                if DEBUG:
                    print(f"Skipping existing: {output_path}")
                saved_paths.append(output_path)
                continue

            if DEBUG:
                alpha_text = "None" if alpha_deg is None else f"{alpha_deg:.3f}"
                print(
                    f"\n{particle} | angle {i_angle + 1}/{angle_indices.numel()} "
                    f"| alpha={alpha_text} deg | theta={theta_deg:.3f} deg "
                    f"| antinu={infer_antinu(particle)}"
                )

            propagated = propagate_particle_angle(
                selected=selected,
                pmns=pmns,
                density=density,
                dm21=dm21,
                dm3l=dm3l,
                antinu=infer_antinu(particle),
                energy_indices=energy_indices_full,
                height_indices=height_indices,
                device=propagation_device,
                dtype=COMPUTE_DTYPE,
            )

            h_grid = _as_tensor(
                selected["h_grid_km"][height_indices],
                device=propagation_device,
                dtype=COMPUTE_DTYPE,
            )

            result = {
                "particle": particle,
                "antinu": torch.as_tensor(infer_antinu(particle)),
                "alpha_deg": None if alpha_deg is None else torch.as_tensor(alpha_deg),
                "theta_deg": torch.as_tensor(theta_deg),
                "E_grid_GeV": propagated["E_grid_GeV"],
                "h_grid_km": h_grid,
                "detector_flux_Ei": propagated["detector_flux_Ei"],
                "surface_flux_Ei": propagated["surface_flux_Ei"],
                "initial_flux_Ei": propagated["initial_flux_Ei"],
                "detector_probability_Ei": propagated["detector_probability_Ei"],
                "surface_probability_Ei": propagated["surface_probability_Ei"],
                "source_flux_E": propagated["source_flux_E"],
                "metadata_extra": {
                    "input_dir": INPUT_DIR,
                    "earth_density_file": EARTH_DENSITY_FILE,
                    "detector_depth_m": float(DETECTOR_DEPTH_M),
                    "matter_in_atmosphere": bool(MATTER_IN_atmosphere),
                    "reunitarize_earth": bool(REUNITARIZE_earth),
                    "atmosphere_steps": int(atmosphere_STEPS),
                    "trajectory_steps": int(TRAJECTORY_STEPS),
                    "energy_chunk_size": ENERGY_CHUNK_SIZE,
                    "energy_max_count": ENERGY_MAX_COUNT,
                    "height_max_count": HEIGHT_MAX_COUNT,
                    "angle_index": int(angle_index),
                    "pmns": {
                        "theta12": float(THETA12),
                        "theta13": float(THETA13),
                        "theta23": float(THETA23),
                        "delta": float(DELTA_CP),
                    },
                    "mass_splittings_ev2": {
                        "DeltamSq21": float(DM21_EV2),
                        "DeltamSq3l": float(DM3L_EV2),
                    },
                },
            }

            saved_path = save_detector_flux_result(
                result,
                OUTPUT_DIR,
                particle=particle,
                alpha_deg=alpha_deg,
                theta_deg=theta_deg,
                base_filename=OUTPUT_FILENAME,
                dtype=SAVE_DTYPE,
                overwrite=OVERWRITE,
            )
            saved_paths.append(saved_path)
            print(f"Saved: {saved_path}")

    elapsed = time.perf_counter() - t0
    print(f"\nFinished detector propagation. Files visited/saved: {len(saved_paths)}")
    print(f"Elapsed: {elapsed:.3f} s")

    return saved_paths


if __name__ == "__main__":
    main()

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
Fully coherent solar-neutrino propagation pipeline.

This module implements the high-level coherent solar-to-detector workflow:
a neutrino is produced at some radius rho inside the Sun in a coherent
superposition of mass eigenstates, that superposition is evolved as a
flavour-amplitude state vector (not just probabilities) from the production
point, through the solar interior (matter-affected mixing), across vacuum to
the Earth, and optionally through Earth matter to an underground detector,
where it is finally converted to flavour-transition probabilities. It keeps
amplitudes through the solar, vacuum, and Earth stages, and projects to
flavour probabilities only where diagnostics or final observables require
it — this is what distinguishes it from ``pipeline_incoherent`` (which
collapses to incoherent mass-eigenstate *weights* right after solar
production, discarding inter-state phase information) and from
``pipeline_legacypeanuts`` (the original incoherent NumPy/Numba reference
implementation, kept for validation). Shared setup tasks such as profile
preparation and nadir exposure handling live in ``pipeline.pipeline_common``.
Every entry point takes a single ``PropagationConfig`` instead of separate
oscillation/exposure/earth/solar keyword arguments.

1. solar coherent propagation from production to solar surface.
2. vacuum coherent propagation from solar surface to earth.
3. earth coherent propagation from earth surface to detector.
4. Final conversion from coherent amplitudes to flavour probabilities.
5. Optional one-year nadir-exposure integration.

Flavour convention:

    [nue, numu, nutau] -> [0, 1, 2]

Module functions:
    propagate_solar_to_detector_coherent(...)
        Run the full coherent solar-to-detector propagation for an energy
        grid and return states, probabilities, and metadata.
    build_coherent_pipeline_metadata(...)
        Build the JSON-serializable metadata block stored with a saved
        coherent result.
    save_coherent_solar_detector_result(...)
        Save a coherent propagation result to a torch file.
    load_coherent_solar_detector_result(...)
        Load a saved coherent propagation result from a torch file.
    detector_conversion_spectra_from_coherent_result(...)
        Recompute per-initial-flavour detector conversion-probability spectra
        from a previously run coherent result.
    run_and_save_solar_to_detector_coherent(...)
        Run the coherent pipeline and immediately save its result.
"""



from __future__ import annotations

import json
import os
from typing import Any, Optional

import torch

from tpeanuts.coherent.evolution import solar_surface_state
from tpeanuts.core.common.probability import probability_coherent_state
from tpeanuts.medium.earth.evolutor import earth_evolutor
from tpeanuts.medium.earth.exposure_table import integrate_exposure, prepare_nadir_exposure
from tpeanuts.util.io import safe_filename_name
from tpeanuts.medium.solar.profile import SolarProfile, build_solar_profile
from tpeanuts.pipeline.config import ProductionMode, PropagationConfig
from tpeanuts.pipeline.pipeline_common import (
    FLAVOUR_ORDER,
    StateLike,
    prepare_earth_distance,
    prepare_earth_profile,
    prepare_initial_state,
)
from tpeanuts.util.constant import R_SUN_KM
from tpeanuts.util.math import trapz_weights
from tpeanuts.util.torch_util import as_1d_tensor, cast_tensor_tree
from tpeanuts.util.type import TensorLike, as_tensor, cdtype_from_real
from tpeanuts.medium.vacuum.evolutor import vacuum_evolutor


def _prepare_production_grid(
    production_mode: ProductionMode,
    rho0: Optional[TensorLike],
    source: Optional[str],
    solar_profile: SolarProfile,
    *,
    context,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Prepare the solar production radius grid and integration weights.

    Args:
        production_mode: Point, coherent-profile, or incoherent-profile mode.
        rho0: Production radius for point mode.
        source: Solar source key for profile modes.
        solar_profile: Prepared SolarProfile.
        context: Runtime device/dtype.

    Returns:
        Tuple ``(rho, weights, metadata)``.
    """
    if production_mode == "point":
        if rho0 is None:
            rho0 = 0.08
        rho = as_1d_tensor(rho0, name="rho0", device=context.device, dtype=context.dtype)
        if rho.numel() != 1:
            raise ValueError("production_mode='point' requires a scalar rho0.")
        weights = torch.ones_like(rho)
        return rho, weights, {"mode": "point", "source": None}

    if source is None:
        raise ValueError(
            "source must be provided when production_mode is 'coherent' or 'incoherent'."
        )

    rho = solar_profile.radius
    weights = solar_profile.normalized_fraction(source)
    norm = torch.trapezoid(weights, x=rho)
    if not torch.isfinite(norm).item() or norm <= 0.0:
        raise ValueError(f"solar source '{source}' has zero or invalid production normalization.")

    return rho, weights, {
        "mode": production_mode,
        "source": source,
        "normalization": float(norm.detach().cpu().item()),
    }


@torch.no_grad()
def propagate_solar_to_detector_coherent(
    *,
    E_MeV: TensorLike,
    config: PropagationConfig,
    initial_state: StateLike = "nue",
    solar_profile: Optional[SolarProfile] = None,
    earth_density: Optional[object] = None,
    eta: Optional[TensorLike] = None,
    debug: bool = False,
) -> dict[str, Any]:
    """Propagate solar neutrino amplitudes coherently to the detector.

    Physical scenario: a neutrino is created in flavour state
    ``initial_state`` at one or more production radii rho (in units of the
    solar radius, set by ``config.production_mode``/``config.rho0``/
    ``config.source``) inside the Sun. Its flavour-amplitude state vector is
    evolved coherently through the solar matter density profile to the solar
    surface, then through vacuum across the Sun-Earth distance, and finally
    (per nadir angle eta) through the Earth's matter density to the detector
    at depth ``config.detector_depth_m``. Probabilities are obtained from the
    amplitudes only at the points where they are needed (surface, Earth
    arrival, detector); the coherent phase relationship between mass
    eigenstates is preserved end-to-end except when
    ``config.production_mode="incoherent"`` is requested as a comparison
    mode, in which case probabilities (not amplitudes) are averaged over the
    production-radius distribution.

    Args:
        E_MeV: Scalar or one-dimensional neutrino energy grid in MeV.
        config: Runtime, oscillation, exposure, Earth, and solar settings
            shared by every tpeanuts pipeline. ``config.production_mode``
            selects between a single production radius ("point",
            ``config.rho0``), a coherent superposition over the solar
            production-radius distribution ("coherent"), or an incoherent
            probability average over that distribution ("incoherent",
            comparison-only) for ``config.source`` (e.g. "8B", "pp").
        initial_state: Initial flavour label (e.g. "nue") or explicit
            coherent flavour amplitudes (complex, unit-normalised, ordered
            [nue, numu, nutau]).
        solar_profile: Optional already-built SolarProfile object. When
            omitted, one is built from ``config.solar``.
        earth_density: Optional already-built EarthProfile-compatible
            object. When omitted, one is built from ``config.earth``.
        eta: Optional explicit nadir-angle grid in radians; when omitted, the
            grid is built from ``config.exposure``.
        debug: Print pipeline sizes when True.

    Returns:
        Dictionary with coherent states and flavour probabilities at the
        solar surface and Earth arrival (both per-rho and production-mode-
        reduced), the Earth evolution operators, detector-level coherent
        states/probabilities per nadir angle (and per rho when
        ``production_mode="incoherent"``), the optional exposure-integrated
        detector probabilities, the energy/rho/eta grids, the Sun-Earth
        distance in km, and a ``metadata_extra`` block describing the run
        (production mode, density/source files, exposure, detector depth,
        antinu flag, reunitarization).
    """
    context = config.runtime
    cdtype = cdtype_from_real(context.dtype)

    E = as_1d_tensor(E_MeV, name="E_MeV", device=context.device, dtype=context.dtype)
    solar_profile_obj = build_solar_profile(
        solar_profile,
        params=config.solar,
        context=context,
    )

    earth_density, density_path = prepare_earth_profile(
        earth_density,
        earth=config.earth,
        context=context,
    )

    earth_distance, distance_metadata = prepare_earth_distance(
        config.earth_distance_km,
        sun_earth_distance_path=config.sun_earth_distance_path,
        use_sun_earth_distance_table=config.use_sun_earth_distance_table,
        context=context,
    )

    rho_grid, rho_weights, production_metadata = _prepare_production_grid(
        config.production_mode,
        config.rho0,
        config.source,
        solar_profile_obj,
        context=context,
    )
    rho_quadrature = trapz_weights(rho_grid) * rho_weights

    eta_grid, exposure, exposure_metadata = prepare_nadir_exposure(
        eta,
        exposure=config.exposure,
        context=context,
    )

    state0 = prepare_initial_state(initial_state, context=context)

    n_E = E.numel()
    n_rho = rho_grid.numel()
    n_eta = eta_grid.numel()

    if debug:
        print(
            "coherent solar detector pipeline: "
            f"n_E={n_E}, n_rho={n_rho}, n_eta={n_eta}, "
            f"production_mode={config.production_mode}"
        )

    surface_states = solar_surface_state(
        state0,
        config.oscillation,
        E,
        rho_grid,
        profile=solar_profile_obj,
        context=context,
    )

    if surface_states.shape != (n_E, n_rho, 3):
        surface_states = surface_states.reshape(n_E, n_rho, 3)

    L_vac_km = earth_distance - torch.as_tensor(R_SUN_KM, device=context.device, dtype=context.dtype)
    if torch.any(L_vac_km < 0.0).item():
        raise ValueError("earth_distance_km must be at least one solar radius.")

    U_vac = vacuum_evolutor(
        config.oscillation,
        E,
        L_vac_km,
        context=context,
    )
    earth_arrival_states = torch.einsum(
        "eij,erj->eri",
        U_vac,
        surface_states,
    )

    surface_probabilities_rho = probability_coherent_state(
        surface_states,
        real_dtype=context.dtype,
    )
    earth_probabilities_rho = probability_coherent_state(
        earth_arrival_states,
        real_dtype=context.dtype,
    )

    if config.production_mode == "coherent":
        surface_states_stage = torch.sum(
            surface_states * rho_quadrature.reshape(1, n_rho, 1).to(cdtype),
            dim=1,
        )
        earth_states_stage = torch.sum(
            earth_arrival_states * rho_quadrature.reshape(1, n_rho, 1).to(cdtype),
            dim=1,
        )
        surface_probabilities = probability_coherent_state(
            surface_states_stage,
            real_dtype=context.dtype,
        )
        earth_probabilities = probability_coherent_state(
            earth_states_stage,
            real_dtype=context.dtype,
        )
    elif config.production_mode == "incoherent":
        surface_states_stage = None
        earth_states_stage = None
        surface_probabilities = torch.sum(
            surface_probabilities_rho * rho_quadrature.reshape(1, n_rho, 1),
            dim=1,
        )
        earth_probabilities = torch.sum(
            earth_probabilities_rho * rho_quadrature.reshape(1, n_rho, 1),
            dim=1,
        )
    else:
        surface_states_stage = surface_states[:, 0, :]
        earth_states_stage = earth_arrival_states[:, 0, :]
        surface_probabilities = surface_probabilities_rho[:, 0, :]
        earth_probabilities = earth_probabilities_rho[:, 0, :]

    detector_states_eta_rho = None
    detector_probabilities_eta_rho = None

    earth_operators = earth_evolutor(
        profile_earth=earth_density,
        oscillation=config.oscillation,
        E=E.reshape(n_E, 1),
        eta=eta_grid.reshape(1, n_eta),
        depth_m=config.detector_depth_m,
        reunitarize=config.reunitarize_earth,
    )

    if config.production_mode == "incoherent":
        detector_states_eta = None
        detector_probabilities_eta_rho = probability_coherent_state(
            torch.einsum(
                "etij,erj->etri",
                earth_operators,
                earth_arrival_states,
            ),
            real_dtype=context.dtype,
        )
    else:
        detector_states_eta = torch.einsum(
            "etij,ej->eti",
            earth_operators,
            earth_states_stage,
        )
        detector_probabilities_eta = probability_coherent_state(
            detector_states_eta,
            real_dtype=context.dtype,
        )

    if config.production_mode == "incoherent":
        detector_probabilities_eta = torch.sum(
            detector_probabilities_eta_rho
            * rho_quadrature.reshape(1, 1, n_rho, 1),
            dim=2,
        )
        detector_states_eta = None

    if config.exposure.integrate_exposure:
        detector_probabilities_integrated = integrate_exposure(
            detector_probabilities_eta,
            eta_grid,
            exposure,
        )
    else:
        detector_probabilities_integrated = None

    result = {
        "mode": "fully_coherent_solar_to_detector",
        "production_mode": config.production_mode,
        "flavour_order": FLAVOUR_ORDER,
        "E_MeV": E,
        "rho_grid": rho_grid,
        "rho_weights": rho_weights,
        "eta": eta_grid,
        "nadir_exposure": exposure,
        "earth_distance_km": earth_distance,
        "initial_state": state0 if torch.is_tensor(state0) else str(state0),
        "surface_states_rho": surface_states,
        "earth_arrival_states_rho": earth_arrival_states,
        "surface_probabilities_rho": surface_probabilities_rho,
        "earth_probabilities_rho": earth_probabilities_rho,
        "surface_states": surface_states_stage,
        "earth_arrival_states": earth_states_stage,
        "surface_probabilities": surface_probabilities,
        "earth_probabilities": earth_probabilities,
        "earth_operators": earth_operators,
        "detector_states_eta": detector_states_eta,
        "detector_probabilities_eta": detector_probabilities_eta,
        "detector_probabilities_eta_rho": detector_probabilities_eta_rho,
        "detector_probabilities_integrated": detector_probabilities_integrated,
        "metadata_extra": {
            "description": (
                "solar neutrino propagation from production to detector with "
                "coherent amplitudes until the final probability projection. "
                "production_mode='incoherent' is provided as a comparison mode."
            ),
            "density_file": density_path,
            "production": production_metadata,
            "sun_earth_distance": distance_metadata,
            "exposure": exposure_metadata,
            "detector_depth_m": float(config.detector_depth_m),
            "antinu": bool(config.oscillation.antinu) if isinstance(config.oscillation.antinu, bool) else "tensor",
            "reunitarize_earth": bool(config.reunitarize_earth),
        },
    }

    return result


def build_coherent_pipeline_metadata(result: dict[str, Any]) -> dict[str, Any]:
    """Build the JSON-serializable metadata block for a coherent result.

    Args:
        result: Result dictionary returned by
            ``propagate_solar_to_detector_coherent``; its tensor entries are
            summarised by shape, and any ``result["metadata_extra"]`` dict
            is merged in (overriding the defaults below).

    Returns:
        Dictionary with a human-readable description, storage format/
        extension, flavour order, production mode, and per-tensor shapes.
    """
    tensor_shapes = {
        key: tuple(value.shape)
        for key, value in result.items()
        if isinstance(value, torch.Tensor)
    }

    metadata = {
        "description": "Fully coherent solar-to-detector propagation result.",
        "format": "torch",
        "extension": ".pt",
        "flavour_order": result.get("flavour_order", FLAVOUR_ORDER),
        "production_mode": result.get("production_mode"),
        "tensor_shapes": tensor_shapes,
    }

    extra = result.get("metadata_extra")
    if isinstance(extra, dict):
        metadata.update(extra)

    return metadata


def save_coherent_solar_detector_result(
    result: dict[str, Any],
    output_dir: str,
    *,
    filename: str = "coherentsolardetector.pt",
    dtype: Optional[torch.dtype] = torch.float32,
    overwrite: bool = False,
) -> str:
    """Save a coherent solar-to-detector propagation result to a torch file.

    Args:
        result: Result dictionary returned by
            ``propagate_solar_to_detector_coherent``.
        output_dir: Directory the file is written into (created if needed).
        filename: Output filename; extension is normalised to ``.pt`` unless
            already ``.pt``/``.pth``.
        dtype: Floating-point dtype tensors are cast to before saving. None
            keeps the original dtype.
        overwrite: If False and the target file already exists, the existing
            path is returned without rewriting it.

    Returns:
        Path of the saved (or pre-existing) torch file.
    """
    os.makedirs(output_dir, exist_ok=True)

    base, ext = os.path.splitext(filename)
    if ext.lower() not in {".pt", ".pth"}:
        ext = ".pt"
    output_path = os.path.join(output_dir, safe_filename_name(base) + ext)

    if os.path.exists(output_path) and not overwrite:
        return output_path

    metadata = build_coherent_pipeline_metadata(result)

    payload = {
        "metadata": metadata,
        "metadata_json": json.dumps(metadata, indent=2),
        "data": cast_tensor_tree(
            result,
            dtype=dtype,
            device="cpu",
        ),
    }

    torch.save(payload, output_path)

    return output_path


def load_coherent_solar_detector_result(
    input_path: str,
    *,
    map_location: str | torch.device = "cpu",
    dtype: Optional[torch.dtype] = torch.float64,
    device: Optional[str | torch.device] = None,
) -> dict[str, Any]:
    """Load a coherent solar-to-detector result saved by ``save_coherent_solar_detector_result``.

    Args:
        input_path: Path to the saved torch file.
        map_location: torch.load map_location used while reading the file.
        dtype: Floating-point dtype tensors are cast to after loading. None
            keeps the dtype stored on disk.
        device: Device tensors are moved to. Defaults to ``map_location``.

    Returns:
        Dictionary with the saved result fields (states, probabilities,
        grids, ...) plus ``"metadata"``/``"metadata_json"`` when present in
        the file.
    """
    try:
        loaded = torch.load(
            input_path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        loaded = torch.load(
            input_path,
            map_location=map_location,
        )

    data = loaded.get("data", loaded)
    if device is None:
        device = map_location

    data = cast_tensor_tree(
        data,
        dtype=dtype,
        device=device,
    )

    if "metadata" in loaded:
        data["metadata"] = loaded["metadata"]
    if "metadata_json" in loaded:
        data["metadata_json"] = loaded["metadata_json"]

    return data


@torch.no_grad()
def detector_conversion_spectra_from_coherent_result(
    result: dict[str, Any],
    *,
    config: PropagationConfig,
    initial_flavours: tuple[str, ...] = ("nue", "numu", "nutau"),
    production_mode: Optional[ProductionMode] = None,
    solar_profile: Optional[SolarProfile] = None,
    rho_chunk_size: int = 32,
) -> dict[str, Any]:
    """Recompute per-initial-flavour detector conversion spectra from a coherent result.

    Reuses the energy/rho/eta grids, Sun-Earth distance, and (already
    computed) Earth evolution operators stored in a previous
    ``propagate_solar_to_detector_coherent`` result to cheaply recompute
    P(initial flavour beta -> final flavour i) detector-conversion spectra
    for one or more *different* initial flavours, without rebuilding the
    Earth evolutor. This is useful for building a full 3x3 conversion-
    probability matrix after running the expensive pipeline once. The solar
    surface state is recomputed in chunks of ``rho_chunk_size`` production
    radii to bound memory use.

    Args:
        result: Result dictionary returned by
            ``propagate_solar_to_detector_coherent`` (used for its grids,
            Sun-Earth distance, and Earth evolution operators).
        config: Runtime and oscillation settings shared by every tpeanuts
            pipeline (must match the oscillation parameters used to produce
            ``result``).
        initial_flavours: Initial flavours to evaluate conversion spectra
            for.
        production_mode: Production-radius treatment ("point", "coherent",
            "incoherent"); defaults to ``result["production_mode"]`` (or
            "incoherent" if absent).
        solar_profile: Optional already-built SolarProfile object. When
            omitted, one is built from ``config.solar``.
        rho_chunk_size: Number of production-radius points processed per
            batch when recomputing solar surface states.

    Returns:
        Dictionary with the initial and final flavour orders, the energy/eta
        grids, the eta-resolved conversion probabilities
        (``conversion_probabilities_eta``), and the exposure-weighted,
        eta-integrated conversion probabilities (``conversion_probabilities``),
        both shaped ``(n_initial_flavours, n_E, ...)``.

    Raises:
        ValueError: If ``rho_chunk_size`` is not positive, or
            ``result["earth_operators"]`` does not have the expected shape.
    """
    context = config.runtime
    cdtype = cdtype_from_real(context.dtype)

    E = as_1d_tensor(result["E_MeV"], name="E_MeV", device=context.device, dtype=context.dtype)
    rho_grid = as_1d_tensor(result["rho_grid"], name="rho_grid", device=context.device, dtype=context.dtype)
    rho_weights = as_1d_tensor(result["rho_weights"], name="rho_weights", device=context.device, dtype=context.dtype)
    eta_grid = as_1d_tensor(result["eta"], name="eta", device=context.device, dtype=context.dtype)
    exposure = as_1d_tensor(result["nadir_exposure"], name="nadir_exposure", device=context.device, dtype=context.dtype)
    earth_distance = as_tensor(result["earth_distance_km"], device=context.device, dtype=context.dtype)
    earth_operators = as_tensor(result["earth_operators"], device=context.device, dtype=cdtype)

    if production_mode is None:
        production_mode = result.get("production_mode", "incoherent")

    if rho_chunk_size <= 0:
        raise ValueError("rho_chunk_size must be positive.")

    solar_profile_obj = build_solar_profile(
        solar_profile,
        params=config.solar,
        context=context,
    )

    n_E = E.numel()
    n_rho = rho_grid.numel()
    n_eta = eta_grid.numel()

    if earth_operators.shape != (n_E, n_eta, 3, 3):
        raise ValueError(
            "earth_operators must have shape (n_E, n_eta, 3, 3). "
            f"Got {tuple(earth_operators.shape)}."
        )

    L_vac_km = earth_distance - torch.as_tensor(R_SUN_KM, device=context.device, dtype=context.dtype)
    U_vac = vacuum_evolutor(
        config.oscillation,
        E,
        L_vac_km,
        context=context,
    )

    eta_quadrature = trapz_weights(eta_grid) * exposure
    eta_quadrature = eta_quadrature / torch.sum(eta_quadrature).clamp_min(torch.finfo(context.dtype).tiny)
    rho_quadrature = trapz_weights(rho_grid) * rho_weights

    spectra = torch.empty((len(initial_flavours), n_E, 3), device=context.device, dtype=context.dtype)
    eta_spectra = torch.empty((len(initial_flavours), n_E, n_eta, 3), device=context.device, dtype=context.dtype)

    for i_flavour, initial_flavour in enumerate(initial_flavours):
        if production_mode == "coherent":
            earth_state_integral = torch.zeros((n_E, 3), device=context.device, dtype=cdtype)
        else:
            detector_eta_accum = torch.zeros((n_E, n_eta, 3), device=context.device, dtype=context.dtype)

        for start in range(0, n_rho, int(rho_chunk_size)):
            stop = min(start + int(rho_chunk_size), n_rho)
            rho_chunk = rho_grid[start:stop]

            surface_states = solar_surface_state(
                initial_flavour,
                config.oscillation,
                E,
                rho_chunk,
                profile=solar_profile_obj,
                context=context,
            )
            surface_states = surface_states.reshape(n_E, rho_chunk.numel(), 3)
            earth_arrival_states = torch.einsum("eij,erj->eri", U_vac, surface_states)

            if production_mode == "coherent":
                weights_chunk = rho_quadrature[start:stop].reshape(1, -1, 1)
                earth_state_integral = earth_state_integral + torch.sum(
                    earth_arrival_states * weights_chunk.to(cdtype),
                    dim=1,
                )
            elif production_mode == "point":
                detector_states = torch.einsum(
                    "etij,ej->eti",
                    earth_operators,
                    earth_arrival_states[:, 0, :],
                )
                detector_eta_accum = probability_coherent_state(
                    detector_states,
                    real_dtype=context.dtype,
                )
                break
            else:
                detector_probabilities = probability_coherent_state(
                    torch.einsum(
                        "etij,erj->etri",
                        earth_operators,
                        earth_arrival_states,
                    ),
                    real_dtype=context.dtype,
                )
                weights_chunk = rho_quadrature[start:stop].reshape(1, 1, -1, 1)
                detector_eta_accum = detector_eta_accum + torch.sum(
                    detector_probabilities * weights_chunk,
                    dim=2,
                )

        if production_mode == "coherent":
            detector_states = torch.einsum("etij,ej->eti", earth_operators, earth_state_integral)
            detector_eta = probability_coherent_state(
                detector_states,
                real_dtype=context.dtype,
            )
        else:
            detector_eta = detector_eta_accum

        eta_spectra[i_flavour] = detector_eta
        spectra[i_flavour] = torch.sum(
            detector_eta * eta_quadrature.reshape(1, n_eta, 1),
            dim=1,
        )

    return {
        "initial_flavours": list(initial_flavours),
        "final_flavours": result.get("flavour_order", FLAVOUR_ORDER),
        "E_MeV": E,
        "eta": eta_grid,
        "conversion_probabilities": spectra,
        "conversion_probabilities_eta": eta_spectra,
        "production_mode": production_mode,
    }


@torch.no_grad()
def run_and_save_solar_to_detector_coherent(
    output_dir: str,
    *,
    filename: str = "coherentsolardetector.pt",
    overwrite: bool = False,
    save_dtype: Optional[torch.dtype] = torch.float32,
    **pipeline_kwargs,
) -> dict[str, Any]:
    """Run the coherent solar-to-detector pipeline and immediately save its result.

    Args:
        output_dir: Directory the result file is written into.
        filename: Output filename passed to
            ``save_coherent_solar_detector_result``.
        overwrite: If False and the target file already exists, the existing
            path is kept (file not rewritten).
        save_dtype: Floating-point dtype tensors are cast to before saving.
        **pipeline_kwargs: Forwarded to
            ``propagate_solar_to_detector_coherent`` (e.g. ``E_MeV``,
            ``config``, ``initial_state``).

    Returns:
        The result dictionary from ``propagate_solar_to_detector_coherent``
        with an added ``"output_path"`` entry pointing at the saved file.
    """
    result = propagate_solar_to_detector_coherent(**pipeline_kwargs)
    output_path = save_coherent_solar_detector_result(
        result,
        output_dir,
        filename=filename,
        dtype=save_dtype,
        overwrite=overwrite,
    )
    result["output_path"] = output_path

    return result

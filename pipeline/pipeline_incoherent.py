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
Incoherent solar-neutrino propagation pipeline.

This module implements the torch-native incoherent solar-to-detector
workflow: instead of carrying a coherent flavour-amplitude state vector, it
works directly with the *incoherent* mixture of mass-eigenstate weights
produced by adiabatic conversion inside the Sun — i.e. it tracks how much of
the flux is nu1, nu2, nu3 (no relative phase information), because solar
densities and baselines decohere the mass eigenstates well before they reach
Earth. It follows the legacy peanuts
physical structure (see ``pipeline_legacypeanuts``, the original NumPy/Numba
reference implementation that this module reproduces in torch) while
delegating shared setup tasks to ``pipeline.pipeline_common`` and Earth
exposure integration to ``medium.earth.exposure_table``.

1. The solar block produces incoherent mass-eigenstate weights using the
   adiabatic approximation.
2. vacuum propagation to earth does not preserve or use phases between mass
   eigenstates, so the mass weights are unchanged.
3. earth propagation is applied with ``massbasis=True`` as an incoherent
   mixture of incident mass eigenstates.
4. Final detector probabilities are integrated over the nadir exposure.

Flavour convention:

    [nue, numu, nutau] -> [0, 1, 2]

Module functions:
    propagate_solar_to_detector_incoherent(...)
        Run the full incoherent solar-to-detector propagation for an energy
        grid and one solar source, returning mass weights, probabilities,
        grids, and metadata.
    build_incoherent_pipeline_metadata(...)
        Build the JSON-serializable metadata block stored with a saved
        incoherent result.
    save_incoherent_solar_detector_result(...)
        Save an incoherent propagation result to a torch file.
    load_incoherent_solar_detector_result(...)
        Load a saved incoherent propagation result from a torch file.
    run_and_save_solar_to_detector_incoherent(...)
        Run the incoherent pipeline and immediately save its result.
"""



from __future__ import annotations

import json
import os
from typing import Any, Optional

import torch

from tpeanuts.medium.earth.exposure_table import integrate_exposure, prepare_nadir_exposure
from tpeanuts.medium.earth.probability import pearth
from tpeanuts.util.io import safe_filename_name
from tpeanuts.medium.solar.probability import (
    psolar,
    solar_probability_mass,
)
from tpeanuts.medium.solar.profile import build_solar_profile
from tpeanuts.util.type import TensorLike, as_tensor

from tpeanuts.pipeline.config import PropagationConfig
from tpeanuts.pipeline.pipeline_common import (
    FLAVOUR_ORDER,
    prepare_earth_distance,
    prepare_earth_profile,
)
from tpeanuts.util.torch_util import as_1d_tensor, cast_tensor_tree


def _normalised_source_fraction(
    solar_profile,
    source: str,
) -> torch.Tensor:
    """Return a normalized solar production fraction for one source.

    Args:
        solar_profile: SolarProfile-like object exposing normalized_fraction().
        source: Solar source key.

    Returns:
        Normalized source-production weights on the solar radius grid.

    Raises:
        ValueError: If the source normalization is invalid.
    """
    fraction = solar_profile.normalized_fraction(source)
    norm = torch.trapezoid(fraction, x=solar_profile.radius)
    if not torch.isfinite(norm).item() or norm <= 0.0:
        raise ValueError(f"solar source '{source}' has zero or invalid production normalization.")
    return fraction


@torch.no_grad()
def propagate_solar_to_detector_incoherent(
    *,
    E_MeV: TensorLike,
    config: PropagationConfig,
    source: str,
    solar_profile: Optional[object] = None,
    earth_density: Optional[object] = None,
    eta: Optional[TensorLike] = None,
    legacy_precision: bool = False,
    debug: bool = False,
) -> dict[str, Any]:
    """Propagate solar neutrinos to the detector as incoherent mass weights.

    Physical scenario: neutrinos are produced as electron neutrinos across
    the solar production-radius distribution of ``source`` (e.g. "8B", "pp"),
    convert adiabatically into a flavour-incoherent mixture of mass
    eigenstate weights inside the Sun (``solar_probability_mass``), and exit
    the Sun carrying only those weights — relative phases between mass
    eigenstates are not tracked because they decohere over the long vacuum
    baseline to Earth, so vacuum propagation leaves the mass weights
    unchanged. At the detector, the incoherent mass-eigenstate mixture is
    propagated through Earth matter per nadir angle eta
    (``pearth(..., massbasis=True)``) and the resulting flavour
    probabilities are optionally integrated over a one-year nadir exposure.
    This is the torch-native counterpart of the original NumPy/Numba
    ``pipeline_legacypeanuts`` workflow: it never carries flavour
    amplitudes/phases after solar production, only mass-eigenstate weights.

    Args:
        E_MeV: Scalar or one-dimensional neutrino energy grid in MeV.
        config: Runtime, oscillation, exposure, Earth, and solar settings
            shared by every tpeanuts pipeline.
        source: Solar neutrino source key (e.g. "8B", "pp") selecting the
            production-radius distribution.
        solar_profile: Optional already-built SolarProfile object. When
            omitted, one is built from ``config.solar``.
        earth_density: Optional already-built EarthProfile-compatible
            object. When omitted, one is built from ``config.earth``.
        eta: Optional explicit nadir-angle grid in radians; when omitted, the
            grid is built from ``config.exposure``.
        legacy_precision: If True, evaluate the solar matter-mixing angles
            with the legacy peanuts ``Vk`` prefactor for bit-comparable
            validation (see ``medium.solar.matter_mixing``).
        debug: Print pipeline sizes when True.

    Returns:
        Dictionary with the solar and Earth-arrival mass-eigenstate weights
        (``solar_mass_weights``, identical to ``earth_arrival_mass_weights``
        since vacuum propagation does not alter them), the corresponding
        flavour probabilities, the detector flavour probabilities per nadir
        angle (``detector_probabilities_eta``) and optionally exposure-
        integrated (``detector_probabilities_integrated``), the energy/rho/eta
        grids, the Sun-Earth distance in km, and a ``metadata_extra`` block
        describing the run (density/source files, exposure, detector depth,
        antinu flag, reunitarization, legacy_precision flag).
    """
    context = config.runtime
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

    eta_grid, exposure, exposure_metadata = prepare_nadir_exposure(
        eta,
        exposure=config.exposure,
        context=context,
    )

    fraction = _normalised_source_fraction(solar_profile_obj, source)

    if debug:
        print(
            "Incoherent solar detector pipeline: "
            f"n_E={E.numel()}, n_eta={eta_grid.numel()}, source={source}"
        )

    mass_weights = solar_probability_mass(
        config.oscillation,
        E,
        solar_profile_obj,
        source,
        legacy_precision=legacy_precision,
    )

    solar_probabilities = psolar(
        config.oscillation,
        E,
        solar_profile_obj,
        source,
        legacy_precision=legacy_precision,
    )

    earth_arrival_mass_weights = mass_weights.clone()
    earth_arrival_probabilities = solar_probabilities.clone()

    detector_probabilities_eta = pearth(
        nustate=earth_arrival_mass_weights,
        profile_earth=earth_density,
        oscillation=config.oscillation,
        E_MeV=E,
        eta=eta_grid,
        depth_m=config.detector_depth_m,
        method="analytical",
        massbasis=True,
        reunitarize=config.reunitarize_earth,
    )

    if config.exposure.integrate_exposure:
        detector_probabilities_integrated = integrate_exposure(
            detector_probabilities_eta,
            eta_grid,
            exposure,
        )
    else:
        detector_probabilities_integrated = None

    return {
        "mode": "incoherent_solar_to_detector",
        "flavour_order": FLAVOUR_ORDER,
        "mass_order": ["nu1", "nu2", "nu3"],
        "source": source,
        "E_MeV": E,
        "rho_grid": solar_profile_obj.radius,
        "rho_weights": fraction,
        "eta": eta_grid,
        "nadir_exposure": exposure,
        "earth_distance_km": as_tensor(earth_distance, device=context.device, dtype=context.dtype),
        "solar_mass_weights": mass_weights,
        "earth_arrival_mass_weights": earth_arrival_mass_weights,
        "solar_probabilities": solar_probabilities,
        "earth_arrival_probabilities": earth_arrival_probabilities,
        "detector_probabilities_eta": detector_probabilities_eta,
        "detector_probabilities_integrated": detector_probabilities_integrated,
        "metadata_extra": {
            "description": (
                "solar neutrino propagation following the legacy peanuts "
                "incoherent workflow: solar_probability_mass -> Pearth(massbasis=True)."
            ),
            "density_file": density_path,
            "source": source,
            "sun_earth_distance": distance_metadata,
            "exposure": exposure_metadata,
            "detector_depth_m": float(config.detector_depth_m),
            "antinu": bool(config.oscillation.antinu) if isinstance(config.oscillation.antinu, bool) else "tensor",
            "reunitarize_earth": bool(config.reunitarize_earth),
            "legacy_precision": bool(legacy_precision),
        },
    }


def build_incoherent_pipeline_metadata(result: dict[str, Any]) -> dict[str, Any]:
    """Build the JSON-serializable metadata block for an incoherent result.

    Args:
        result: Result dictionary returned by
            ``propagate_solar_to_detector_incoherent``; its tensor entries
            are summarised by shape, and any ``result["metadata_extra"]``
            dict is merged in (overriding the defaults below).

    Returns:
        Dictionary with a human-readable description, storage format/
        extension, flavour/mass order, solar source, and per-tensor shapes.
    """
    tensor_shapes = {
        key: tuple(value.shape)
        for key, value in result.items()
        if isinstance(value, torch.Tensor)
    }

    metadata = {
        "description": "Incoherent legacy-style solar-to-detector propagation result.",
        "format": "torch",
        "extension": ".pt",
        "flavour_order": result.get("flavour_order", FLAVOUR_ORDER),
        "mass_order": result.get("mass_order", ["nu1", "nu2", "nu3"]),
        "source": result.get("source"),
        "tensor_shapes": tensor_shapes,
    }

    extra = result.get("metadata_extra")
    if isinstance(extra, dict):
        metadata.update(extra)

    return metadata


def save_incoherent_solar_detector_result(
    result: dict[str, Any],
    output_dir: str,
    *,
    filename: str = "Incoherentsolardetector.pt",
    dtype: Optional[torch.dtype] = torch.float32,
    overwrite: bool = False,
) -> str:
    """Save an incoherent solar-to-detector propagation result to a torch file.

    Args:
        result: Result dictionary returned by
            ``propagate_solar_to_detector_incoherent``.
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

    metadata = build_incoherent_pipeline_metadata(result)

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


def load_incoherent_solar_detector_result(
    input_path: str,
    *,
    map_location: str | torch.device = "cpu",
    dtype: Optional[torch.dtype] = torch.float64,
    device: Optional[str | torch.device] = None,
) -> dict[str, Any]:
    """Load an incoherent solar-to-detector result saved by ``save_incoherent_solar_detector_result``.

    Args:
        input_path: Path to the saved torch file.
        map_location: torch.load map_location used while reading the file.
        dtype: Floating-point dtype tensors are cast to after loading. None
            keeps the dtype stored on disk.
        device: Device tensors are moved to. Defaults to ``map_location``.

    Returns:
        Dictionary with the saved result fields (mass weights, probabilities,
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
def run_and_save_solar_to_detector_incoherent(
    output_dir: str,
    *,
    filename: str = "Incoherentsolardetector.pt",
    overwrite: bool = False,
    save_dtype: Optional[torch.dtype] = torch.float32,
    **pipeline_kwargs,
) -> dict[str, Any]:
    """Run the incoherent solar-to-detector pipeline and immediately save its result.

    Args:
        output_dir: Directory the result file is written into.
        filename: Output filename passed to
            ``save_incoherent_solar_detector_result``.
        overwrite: If False and the target file already exists, the existing
            path is kept (file not rewritten).
        save_dtype: Floating-point dtype tensors are cast to before saving.
        **pipeline_kwargs: Forwarded to
            ``propagate_solar_to_detector_incoherent`` (e.g. ``E_MeV``,
            ``config``, ``source``).

    Returns:
        The result dictionary from ``propagate_solar_to_detector_incoherent``
        with an added ``"output_path"`` entry pointing at the saved file.
    """
    result = propagate_solar_to_detector_incoherent(**pipeline_kwargs)
    output_path = save_incoherent_solar_detector_result(
        result,
        output_dir,
        filename=filename,
        dtype=save_dtype,
        overwrite=overwrite,
    )
    result["output_path"] = output_path

    return result

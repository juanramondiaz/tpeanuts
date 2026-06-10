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

This pipeline follows the legacy peanuts solar workflow:

1. The solar block produces incoherent mass-eigenstate weights using the
   adiabatic approximation.
2. vacuum propagation to earth does not preserve or use phases between mass
   eigenstates, so the mass weights are unchanged.
3. earth propagation is applied with ``massbasis=True`` as an incoherent
   mixture of incident mass eigenstates.
4. Final detector probabilities are integrated over the nadir exposure.

Flavour convention:

    [nue, numu, nutau] -> [0, 1, 2]
"""



from __future__ import annotations

import json
import os
from typing import Any, Literal, Optional, Union

import torch

from tpeanuts.earth.exposure import build_nadir_exposure
from tpeanuts.io.io_earth import load_earth_density_from_csv
from tpeanuts.earth.probabilities import pearth
from tpeanuts.io.io_common import cast_tensor_tree, safe_filename_name
from tpeanuts.solar.probabilities import psolar, solar_flux_mass
from tpeanuts.util.type import _as_tensor

from tpeanuts.flux_propagation.pipeline_coherent import (
    FLAVOUR_ORDER,
    _as_1d_tensor,
    _build_pmns,
    _default_earth_density_path,
    _exposure_integral,
    _prepare_earth_distance,
    _prepare_solar_profile,
    _resolve_device,
)

TensorLike = Union[float, int, torch.Tensor]
ExposureSource = Literal["math", "cache", "csv", "legacy"]


def _normalised_source_fraction(
    solar_profile,
    source: str,
) -> torch.Tensor:
    fraction = solar_profile.normalized_fraction(source)
    norm = torch.trapezoid(fraction, x=solar_profile.radius)
    if not torch.isfinite(norm).item() or norm <= 0.0:
        raise ValueError(f"solar source '{source}' has zero or invalid production normalization.")
    return fraction


def _prepare_exposure(
    eta: Optional[TensorLike],
    *,
    detector_latitude_rad: Optional[float],
    exposure_source: ExposureSource,
    exposure_csv_path: Optional[str],
    exposure_angle: str,
    exposure_daynight: Optional[Literal["day", "night"]],
    exposure_d1: float,
    exposure_d2: float,
    exposure_ns: int,
    exposure_cache_dir: str,
    exposure_use_cache: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if eta is not None:
        eta_grid = _as_1d_tensor(eta, name="eta", device=device, dtype=dtype)
        exposure = torch.ones_like(eta_grid)
        exposure = exposure / torch.trapezoid(exposure, x=eta_grid).clamp_min(torch.finfo(dtype).tiny)
        return eta_grid, exposure, {
            "source": "user_eta_uniform",
            "normalized": True,
        }

    if detector_latitude_rad is None and exposure_source in ("math", "cache", "legacy"):
        raise ValueError(
            "detector_latitude_rad is required when eta is not provided "
            "and exposure_source is math/cache/legacy."
        )

    exposure_table = build_nadir_exposure(
        source=exposure_source,
        lam_rad=detector_latitude_rad,
        d1=exposure_d1,
        d2=exposure_d2,
        ns=exposure_ns,
        daynight=exposure_daynight,
        normalized=True,
        csv_path=exposure_csv_path,
        angle=exposure_angle,
        cache_dir=exposure_cache_dir,
        use_cache=exposure_use_cache,
        device=device,
        dtype=dtype,
    )

    return exposure_table.eta, exposure_table.exposure, {
        "source": exposure_source,
        "d1": exposure_d1,
        "d2": exposure_d2,
        "ns": exposure_ns,
        "daynight": exposure_daynight,
        "normalized": True,
        "detector_latitude_rad": detector_latitude_rad,
    }


@torch.no_grad()
def propagate_solar_to_detector_incoherent(
    *,
    E_MeV: TensorLike,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    source: str,
    pmns: Optional[object] = None,
    theta12: Optional[TensorLike] = None,
    theta13: Optional[TensorLike] = None,
    theta23: Optional[TensorLike] = None,
    delta: Optional[TensorLike] = None,
    solar_profile: Optional[object] = None,
    earth_density: Optional[object] = None,
    earth_density_file: Optional[str] = None,
    tabulated_earth_density: bool = False,
    earth_distance_km: Optional[TensorLike] = None,
    sun_earth_distance_path: Optional[str] = None,
    use_sun_earth_distance_table: bool = True,
    eta: Optional[TensorLike] = None,
    detector_depth_m: float = 0.0,
    detector_latitude_rad: Optional[float] = None,
    exposure_source: ExposureSource = "math",
    exposure_csv_path: Optional[str] = None,
    exposure_angle: str = "Nadir",
    exposure_daynight: Optional[Literal["day", "night"]] = None,
    exposure_d1: float = 0.0,
    exposure_d2: float = 365.0,
    exposure_ns: int = 1000,
    exposure_cache_dir: str = "cache_exposure",
    exposure_use_cache: bool = False,
    integrate_exposure: bool = True,
    antinu: Union[bool, torch.Tensor] = False,
    reunitarize_earth: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
    debug: bool = False,
) -> dict[str, Any]:
    dev = _resolve_device(device)
    E = _as_1d_tensor(E_MeV, name="E_MeV", device=dev, dtype=dtype)

    pmns_obj = _build_pmns(
        pmns,
        theta12=theta12,
        theta13=theta13,
        theta23=theta23,
        delta=delta,
        device=dev,
        dtype=dtype,
    )
    solar_profile_obj = _prepare_solar_profile(
        solar_profile,
        device=dev,
        dtype=dtype,
    )

    if earth_density is None:
        density_path = earth_density_file or _default_earth_density_path()
        earth_density = load_earth_density_from_csv(
            density_path,
            tabulated_density=tabulated_earth_density,
            device=dev,
            dtype=dtype,
        )
    else:
        density_path = earth_density_file

    earth_distance, distance_metadata = _prepare_earth_distance(
        earth_distance_km,
        sun_earth_distance_path=sun_earth_distance_path,
        use_sun_earth_distance_table=use_sun_earth_distance_table,
        device=dev,
        dtype=dtype,
    )

    eta_grid, exposure, exposure_metadata = _prepare_exposure(
        eta,
        detector_latitude_rad=detector_latitude_rad,
        exposure_source=exposure_source,
        exposure_csv_path=exposure_csv_path,
        exposure_angle=exposure_angle,
        exposure_daynight=exposure_daynight,
        exposure_d1=exposure_d1,
        exposure_d2=exposure_d2,
        exposure_ns=exposure_ns,
        exposure_cache_dir=exposure_cache_dir,
        exposure_use_cache=exposure_use_cache,
        device=dev,
        dtype=dtype,
    )

    fraction = _normalised_source_fraction(solar_profile_obj, source)

    if debug:
        print(
            "Incoherent solar detector pipeline: "
            f"n_E={E.numel()}, n_eta={eta_grid.numel()}, source={source}"
        )

    mass_weights = solar_flux_mass(
        pmns_obj.theta12,
        pmns_obj.theta13,
        DeltamSq21,
        DeltamSq3l,
        E,
        solar_profile_obj.radius,
        solar_profile_obj.density,
        fraction,
    )

    solar_probabilities = psolar(
        pmns_obj,
        DeltamSq21,
        DeltamSq3l,
        E,
        solar_profile_obj.radius,
        solar_profile_obj.density,
        fraction,
        antinu=antinu,
    )

    earth_arrival_mass_weights = mass_weights.clone()
    earth_arrival_probabilities = solar_probabilities.clone()

    detector_probabilities_eta = pearth(
        nustate=earth_arrival_mass_weights,
        density=earth_density,
        pmns=pmns_obj,
        dm21_eV2=DeltamSq21,
        dm3l_eV2=DeltamSq3l,
        E_MeV=E,
        eta=eta_grid,
        depth_m=detector_depth_m,
        method="analytical",
        antinu=antinu,
        massbasis=True,
        reunitarize=reunitarize_earth,
    )

    if integrate_exposure:
        detector_probabilities_integrated = _exposure_integral(
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
        "earth_distance_km": _as_tensor(earth_distance, device=dev, dtype=dtype),
        "solar_mass_weights": mass_weights,
        "earth_arrival_mass_weights": earth_arrival_mass_weights,
        "solar_probabilities": solar_probabilities,
        "earth_arrival_probabilities": earth_arrival_probabilities,
        "detector_probabilities_eta": detector_probabilities_eta,
        "detector_probabilities_integrated": detector_probabilities_integrated,
        "metadata_extra": {
            "description": (
                "solar neutrino propagation following the legacy peanuts "
                "incoherent workflow: solar_flux_mass -> Pearth(massbasis=True)."
            ),
            "density_file": density_path,
            "source": source,
            "sun_earth_distance": distance_metadata,
            "exposure": exposure_metadata,
            "detector_depth_m": float(detector_depth_m),
            "antinu": bool(antinu) if isinstance(antinu, bool) else "tensor",
            "reunitarize_earth": bool(reunitarize_earth),
        },
    }


def build_incoherent_pipeline_metadata(result: dict[str, Any]) -> dict[str, Any]:
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

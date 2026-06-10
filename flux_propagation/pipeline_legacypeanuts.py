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
Legacy peanuts solar-neutrino propagation pipeline.

This module intentionally calls the original NumPy/Numba implementation in the
``peanuts`` package and saves the results with the same torch-file convention
used by the new propagation pipelines.

The legacy solar workflow is incoherent:

1. ``peanuts.solar.solar_flux_mass`` builds mass-eigenstate weights.
2. ``peanuts.solar.Psolar`` builds solar flavour probabilities.
3. ``peanuts.earth.Pearth(..., massbasis=True)`` propagates the incoherent
   mass mixture through earth.
4. The detector probabilities are integrated over the legacy nadir exposure.
"""



from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import torch

from tpeanuts.io.io_common import cast_tensor_tree, safe_filename_name
from tpeanuts.io.io_solar import load_sun_earth_distance
from tpeanuts.util.constant import SUN_EARTH_DISTANCE_KM
from tpeanuts.util.type import _as_tensor

from tpeanuts.flux_propagation.pipeline_coherent import (
    FLAVOUR_ORDER,
    _as_1d_tensor,
    _resolve_device,
)

TensorLike = Union[float, int, torch.Tensor]


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _legacy_data_dir() -> Path:
    return _package_root() / "data" / "peanuts"



def _legacy_expected_data_dir() -> Path:
    return _package_root() / "data"


def _default_legacy_earth_density_path() -> str:
    return str(_legacy_data_dir() / "earth_density.csv")


def _ensure_legacy_default_data_aliases() -> None:
    src_dir = _legacy_data_dir()
    dst_dir = _legacy_expected_data_dir()

    if not src_dir.is_dir():
        raise FileNotFoundError(f"Legacy peanuts data directory not found: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    for src in src_dir.iterdir():
        if not src.is_file():
            continue

        dst = dst_dir / src.name
        if dst.exists():
            continue

        shutil.copyfile(src, dst)


def _torch_to_numpy_1d(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().reshape(-1)


def _prepare_legacy_pmns(
    pmns: Optional[object],
    *,
    theta12: Optional[TensorLike],
    theta13: Optional[TensorLike],
    theta23: Optional[TensorLike],
    delta: Optional[TensorLike],
) -> object:
    if pmns is not None:
        return pmns

    missing = [
        name
        for name, value in {
            "theta12": theta12,
            "theta13": theta13,
            "theta23": theta23,
            "delta": delta,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(
            "Provide pmns or all PMNS angles: " + ", ".join(missing)
        )

    from peanuts.pmns import PMNS as LegacyPMNS

    return LegacyPMNS(
        float(theta12),
        float(theta13),
        float(theta23),
        float(delta),
    )


def _prepare_legacy_solar_model(
    solar_model: Optional[object],
    *,
    solar_model_file: Optional[str],
    solar_flux_file: Optional[str],
) -> object:
    if solar_model is not None:
        return solar_model
    _ensure_legacy_default_data_aliases()

    from peanuts.solar import SolarModel

    data_dir = _legacy_data_dir()
    return SolarModel(
        solar_model_file=solar_model_file or str(data_dir / "nudistr_b16_agss09.dat"),
        flux_file=solar_flux_file or str(data_dir / "fluxes_b16.dat"),
    )


def _prepare_legacy_earth_density(
    earth_density: Optional[object],
    *,
    earth_density_file: Optional[str],
    tabulated_earth_density: bool,
) -> object:
    if earth_density is not None:
        return earth_density

    from peanuts.earth import earthdensity

    return earthdensity(
        density_file=earth_density_file or _default_legacy_earth_density_path(),
        tabulated_density=tabulated_earth_density,
    )


def _prepare_legacy_exposure(
    eta: Optional[TensorLike],
    *,
    detector_latitude_rad: Optional[float],
    exposure_d1: float,
    exposure_d2: float,
    exposure_ns: int,
    exposure_normalized: bool,
    exposure_from_file: Optional[str],
    exposure_angle: str,
    exposure_daynight: Optional[Literal["day", "night"]],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, dict[str, Any], float]:
    if eta is not None:
        eta_t = _as_1d_tensor(eta, name="eta", device=device, dtype=dtype)
        exposure_t = torch.ones_like(eta_t)
        exposure_t = exposure_t / torch.trapezoid(exposure_t, x=eta_t).clamp_min(torch.finfo(dtype).tiny)
        eta_np = _torch_to_numpy_1d(eta_t)
        exposure_np = _torch_to_numpy_1d(exposure_t)
        deta = float(np.pi / max(eta_np.size, 1))
        return eta_np, exposure_np, eta_t, exposure_t, {
            "source": "user_eta_uniform",
            "normalized": True,
        }, deta

    if detector_latitude_rad is None and exposure_from_file is None:
        raise ValueError("detector_latitude_rad is required when eta and exposure_from_file are not provided.")

    from peanuts.time_average import NadirExposure

    table = NadirExposure(
        lam=-1 if detector_latitude_rad is None else float(detector_latitude_rad),
        d1=exposure_d1,
        d2=exposure_d2,
        ns=exposure_ns,
        normalized=exposure_normalized,
        from_file=exposure_from_file,
        angle=exposure_angle,
        daynight=exposure_daynight,
    )

    eta_np = np.asarray(table[:, 0], dtype=np.float64)
    exposure_np = np.asarray(table[:, 1], dtype=np.float64)
    eta_t = torch.as_tensor(eta_np, device=device, dtype=dtype)
    exposure_t = torch.as_tensor(exposure_np, device=device, dtype=dtype)
    deta = float(np.pi / exposure_ns)

    return eta_np, exposure_np, eta_t, exposure_t, {
        "source": "legacy_NadirExposure",
        "d1": exposure_d1,
        "d2": exposure_d2,
        "ns": exposure_ns,
        "daynight": exposure_daynight,
        "normalized": bool(exposure_normalized),
        "detector_latitude_rad": detector_latitude_rad,
        "from_file": exposure_from_file,
        "angle": exposure_angle,
    }, deta


def _prepare_earth_distance_metadata(
    earth_distance_km: Optional[TensorLike],
    *,
    sun_earth_distance_path: Optional[str],
    use_sun_earth_distance_table: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if earth_distance_km is not None:
        distance = _as_tensor(earth_distance_km, device=device, dtype=dtype)
        return distance, {
            "source": "user",
            "distance_km": float(distance.reshape(-1)[0].detach().cpu().item()),
            "note": "Not used by the incoherent legacy solar workflow.",
        }

    if use_sun_earth_distance_table:
        table = load_sun_earth_distance(
            sun_earth_distance_path,
            device=device,
            dtype=dtype,
        )
        distance = torch.mean(table["distance_km"])
        return distance, {
            "source": "sun_earth_distance_table_mean",
            "path": sun_earth_distance_path,
            "n_dates": len(table["date"]),
            "distance_km": float(distance.detach().cpu().item()),
            "note": "Not used by the incoherent legacy solar workflow.",
        }

    distance = torch.tensor(SUN_EARTH_DISTANCE_KM, device=device, dtype=dtype)
    return distance, {
        "source": "SUN_EARTH_DISTANCE_KM",
        "distance_km": float(distance.detach().cpu().item()),
        "note": "Not used by the incoherent legacy solar workflow.",
    }


@torch.no_grad()
def propagate_solar_to_detector_legacypeanuts(
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
    solar_model: Optional[object] = None,
    solar_model_file: Optional[str] = None,
    solar_flux_file: Optional[str] = None,
    earth_density: Optional[object] = None,
    earth_density_file: Optional[str] = None,
    tabulated_earth_density: bool = False,
    earth_distance_km: Optional[TensorLike] = None,
    sun_earth_distance_path: Optional[str] = None,
    use_sun_earth_distance_table: bool = True,
    eta: Optional[TensorLike] = None,
    detector_depth_m: float = 0.0,
    detector_latitude_rad: Optional[float] = None,
    exposure_d1: float = 0.0,
    exposure_d2: float = 365.0,
    exposure_ns: int = 1000,
    exposure_normalized: bool = True,
    exposure_from_file: Optional[str] = None,
    exposure_angle: str = "Nadir",
    exposure_daynight: Optional[Literal["day", "night"]] = None,
    integrate_exposure: bool = True,
    antinu: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
    debug: bool = False,
) -> dict[str, Any]:
    dev = _resolve_device(device)
    E = _as_1d_tensor(E_MeV, name="E_MeV", device=dev, dtype=dtype)
    E_np = _torch_to_numpy_1d(E)

    dm21 = float(_as_tensor(DeltamSq21, device="cpu", dtype=torch.float64).reshape(-1)[0].item())
    dm3l = float(_as_tensor(DeltamSq3l, device="cpu", dtype=torch.float64).reshape(-1)[0].item())

    legacy_pmns = _prepare_legacy_pmns(
        pmns,
        theta12=theta12,
        theta13=theta13,
        theta23=theta23,
        delta=delta,
    )
    legacy_solar = _prepare_legacy_solar_model(
        solar_model,
        solar_model_file=solar_model_file,
        solar_flux_file=solar_flux_file,
    )
    legacy_earth = _prepare_legacy_earth_density(
        earth_density,
        earth_density_file=earth_density_file,
        tabulated_earth_density=tabulated_earth_density,
    )

    eta_np, exposure_np, eta_t, exposure_t, exposure_metadata, deta = _prepare_legacy_exposure(
        eta,
        detector_latitude_rad=detector_latitude_rad,
        exposure_d1=exposure_d1,
        exposure_d2=exposure_d2,
        exposure_ns=exposure_ns,
        exposure_normalized=exposure_normalized,
        exposure_from_file=exposure_from_file,
        exposure_angle=exposure_angle,
        exposure_daynight=exposure_daynight,
        device=dev,
        dtype=dtype,
    )

    earth_distance, distance_metadata = _prepare_earth_distance_metadata(
        earth_distance_km,
        sun_earth_distance_path=sun_earth_distance_path,
        use_sun_earth_distance_table=use_sun_earth_distance_table,
        device=dev,
        dtype=dtype,
    )
    _ensure_legacy_default_data_aliases()
    from peanuts.solar import Psolar, solar_flux_mass
    from peanuts.earth import Pearth

    radius = np.asarray(legacy_solar.radius(), dtype=np.float64)
    density = np.asarray(legacy_solar.density(), dtype=np.float64)
    fraction = np.asarray(legacy_solar.fraction(source), dtype=np.float64)

    n_E = E_np.size
    n_eta = eta_np.size

    if debug:
        print(
            "Legacy peanuts solar detector pipeline: "
            f"n_E={n_E}, n_eta={n_eta}, source={source}"
        )

    mass_weights_np = np.empty((n_E, 3), dtype=np.float64)
    solar_probabilities_np = np.empty((n_E, 3), dtype=np.float64)
    detector_probabilities_eta_np = np.empty((n_E, n_eta, 3), dtype=np.float64)
    detector_probabilities_integrated_np = np.zeros((n_E, 3), dtype=np.float64)

    for i_E, energy in enumerate(E_np):
        mass_weights = solar_flux_mass(
            legacy_pmns.theta12,
            legacy_pmns.theta13,
            dm21,
            dm3l,
            float(energy),
            radius,
            density,
            fraction,
        )
        solar_probabilities = Psolar(
            legacy_pmns,
            dm21,
            dm3l,
            float(energy),
            radius,
            density,
            fraction,
        )

        mass_weights_np[i_E] = np.asarray(mass_weights, dtype=np.float64)
        solar_probabilities_np[i_E] = np.asarray(solar_probabilities, dtype=np.float64)

        for i_eta, eta_value in enumerate(eta_np):
            detector_prob = Pearth(
                mass_weights_np[i_E],
                legacy_earth,
                legacy_pmns,
                dm21,
                dm3l,
                float(energy),
                float(eta_value),
                float(detector_depth_m),
                mode="analytical",
                massbasis=True,
                antinu=antinu,
            )
            detector_probabilities_eta_np[i_E, i_eta] = np.asarray(detector_prob, dtype=np.float64)

        if integrate_exposure:
            detector_probabilities_integrated_np[i_E] = np.sum(
                detector_probabilities_eta_np[i_E] * exposure_np[:, None],
                axis=0,
            ) * deta

    mass_weights_t = torch.as_tensor(mass_weights_np, device=dev, dtype=dtype)
    solar_probabilities_t = torch.as_tensor(solar_probabilities_np, device=dev, dtype=dtype)
    detector_probabilities_eta_t = torch.as_tensor(detector_probabilities_eta_np, device=dev, dtype=dtype)
    detector_probabilities_integrated_t = (
        torch.as_tensor(detector_probabilities_integrated_np, device=dev, dtype=dtype)
        if integrate_exposure
        else None
    )

    return {
        "mode": "legacy_peanuts_solar_to_detector",
        "flavour_order": FLAVOUR_ORDER,
        "mass_order": ["nu1", "nu2", "nu3"],
        "source": source,
        "E_MeV": E,
        "rho_grid": torch.as_tensor(radius, device=dev, dtype=dtype),
        "rho_weights": torch.as_tensor(fraction, device=dev, dtype=dtype),
        "eta": eta_t,
        "nadir_exposure": exposure_t,
        "earth_distance_km": earth_distance,
        "solar_mass_weights": mass_weights_t,
        "earth_arrival_mass_weights": mass_weights_t.clone(),
        "solar_probabilities": solar_probabilities_t,
        "earth_arrival_probabilities": solar_probabilities_t.clone(),
        "detector_probabilities_eta": detector_probabilities_eta_t,
        "detector_probabilities_integrated": detector_probabilities_integrated_t,
        "metadata_extra": {
            "description": (
                "solar-to-detector propagation computed with the original "
                "legacy peanuts NumPy/Numba implementation."
            ),
            "legacy_package": "peanuts",
            "solar_model_file": solar_model_file or str(_legacy_data_dir() / "nudistr_b16_agss09.dat"),
            "solar_flux_file": solar_flux_file or str(_legacy_data_dir() / "fluxes_b16.dat"),
            "density_file": earth_density_file or _default_legacy_earth_density_path(),
            "source": source,
            "sun_earth_distance": distance_metadata,
            "exposure": exposure_metadata,
            "detector_depth_m": float(detector_depth_m),
            "antinu": bool(antinu),
        },
    }


def build_legacypeanuts_pipeline_metadata(result: dict[str, Any]) -> dict[str, Any]:
    tensor_shapes = {
        key: tuple(value.shape)
        for key, value in result.items()
        if isinstance(value, torch.Tensor)
    }

    metadata = {
        "description": "Legacy peanuts solar-to-detector propagation result.",
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


def save_legacypeanuts_solar_detector_result(
    result: dict[str, Any],
    output_dir: str,
    *,
    filename: str = "Legacypeanutssolardetector.pt",
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

    metadata = build_legacypeanuts_pipeline_metadata(result)

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


def load_legacypeanuts_solar_detector_result(
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
def run_and_save_solar_to_detector_legacypeanuts(
    output_dir: str,
    *,
    filename: str = "Legacypeanutssolardetector.pt",
    overwrite: bool = False,
    save_dtype: Optional[torch.dtype] = torch.float32,
    **pipeline_kwargs,
) -> dict[str, Any]:
    result = propagate_solar_to_detector_legacypeanuts(**pipeline_kwargs)
    output_path = save_legacypeanuts_solar_detector_result(
        result,
        output_dir,
        filename=filename,
        dtype=save_dtype,
        overwrite=overwrite,
    )
    result["output_path"] = output_path

    return result

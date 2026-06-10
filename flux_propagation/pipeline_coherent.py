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

The pipeline connects the existing low-level blocks without duplicating their
physics:

1. solar coherent propagation from production to solar surface.
2. vacuum coherent propagation from solar surface to earth.
3. earth coherent propagation from earth surface to detector.
4. Final conversion from coherent amplitudes to flavour probabilities.
5. Optional one-year nadir-exposure integration.

Flavour convention:

    [nue, numu, nutau] -> [0, 1, 2]
"""



from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, Optional, Union

import torch

from tpeanuts.coherent.evolution import solar_surface_state
from tpeanuts.core.pmns import PMNS
from tpeanuts.earth.evolutor import earth_evolutor
from tpeanuts.earth.exposure import build_nadir_exposure
from tpeanuts.io.io_earth import load_earth_density_from_csv
from tpeanuts.io.io_common import cast_tensor_tree, safe_filename_name
from tpeanuts.io.io_solar import load_sun_earth_distance
from tpeanuts.solar.profiles import SolarProfile, load_default_solar_profile
from tpeanuts.util.constant import R_SUN_KM, SUN_EARTH_DISTANCE_KM
from tpeanuts.util.torch_util import _default_device
from tpeanuts.util.type import _as_tensor, _cdtype_from_real, _state_tensor
from tpeanuts.vacuum.probabilities import vacuum_evolutor

TensorLike = Union[float, int, torch.Tensor]
StateLike = Union[str, list[complex], tuple[complex, ...], torch.Tensor]
ProductionMode = Literal["point", "coherent", "incoherent"]
ExposureSource = Literal["math", "cache", "csv", "legacy"]

FLAVOUR_ORDER = ["nue", "numu", "nutau"]


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_earth_density_path() -> str:
    return str(_package_root() / "data" / "density" / "earth_density.csv")


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if callable(device):
        device = device()
    return _default_device(device)


def _as_1d_tensor(
    value: TensorLike,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    out = _as_tensor(value, device=device, dtype=dtype)
    if out.ndim == 0:
        out = out[None]
    if out.ndim != 1:
        raise ValueError(f"{name} must be scalar or one-dimensional.")
    return out


def _build_pmns(
    pmns: Optional[object],
    *,
    theta12: Optional[TensorLike],
    theta13: Optional[TensorLike],
    theta23: Optional[TensorLike],
    delta: Optional[TensorLike],
    device: torch.device,
    dtype: torch.dtype,
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

    return PMNS(
        theta12=theta12,
        theta13=theta13,
        theta23=theta23,
        delta=delta,
        device=device,
        real_dtype=dtype,
    )


def _prepare_initial_state(
    initial_state: StateLike,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> StateLike:
    if isinstance(initial_state, str):
        return initial_state
    return _state_tensor(
        initial_state,
        device=device,
        dtype=_cdtype_from_real(dtype),
    )


def _prepare_solar_profile(
    solar_profile: Optional[SolarProfile],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> SolarProfile:
    if solar_profile is None:
        return load_default_solar_profile(device=device, dtype=dtype)

    if solar_profile.radius.device == device and solar_profile.radius.dtype == dtype:
        return solar_profile

    return SolarProfile(
        radius=solar_profile.radius.to(device=device, dtype=dtype),
        density=solar_profile.density.to(device=device, dtype=dtype),
        fractions={
            key: value.to(device=device, dtype=dtype)
            for key, value in solar_profile.fractions.items()
        },
        fluxes={
            key: value.to(device=device, dtype=dtype)
            for key, value in solar_profile.fluxes.items()
        },
    )


def _prepare_earth_distance(
    earth_distance_km: Optional[TensorLike],
    *,
    sun_earth_distance_path: Optional[str],
    use_sun_earth_distance_table: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if earth_distance_km is not None:
        distance = _as_tensor(earth_distance_km, device=device, dtype=dtype)
        return distance, {"source": "user", "distance_km": float(distance.reshape(-1)[0].detach().cpu().item())}

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
        }

    distance = torch.tensor(SUN_EARTH_DISTANCE_KM, device=device, dtype=dtype)
    return distance, {
        "source": "SUN_EARTH_DISTANCE_KM",
        "distance_km": float(distance.detach().cpu().item()),
    }


def _prepare_production_grid(
    production_mode: ProductionMode,
    rho0: Optional[TensorLike],
    source: Optional[str],
    solar_profile: SolarProfile,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if production_mode == "point":
        if rho0 is None:
            rho0 = 0.08
        rho = _as_1d_tensor(rho0, name="rho0", device=device, dtype=dtype)
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


def _integrate_over_rho(
    values: torch.Tensor,
    weights: torch.Tensor,
    rho: torch.Tensor,
) -> torch.Tensor:
    weighted = values * weights.reshape(*((1,) * (values.ndim - 2)), -1, 1)
    return torch.trapezoid(weighted, x=rho, dim=-2)


def _integrate_probabilities_over_rho(
    probabilities: torch.Tensor,
    weights: torch.Tensor,
    rho: torch.Tensor,
) -> torch.Tensor:
    weighted = probabilities * weights.reshape(*((1,) * (probabilities.ndim - 2)), -1, 1)
    return torch.trapezoid(weighted, x=rho, dim=-2)


def _apply_earth_to_state(
    U_earth: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    if U_earth.ndim == 2:
        return torch.einsum("ij,j->i", U_earth, state)

    if state.ndim == 1:
        return torch.einsum("...ij,j->...i", U_earth, state)

    return torch.einsum("...ij,...j->...i", U_earth, state)


def _exposure_integral(
    probabilities_eta: torch.Tensor,
    eta: torch.Tensor,
    exposure: torch.Tensor,
) -> torch.Tensor:
    weighted = probabilities_eta * exposure.reshape(*((1,) * (probabilities_eta.ndim - 2)), -1, 1)
    return torch.trapezoid(weighted, x=eta, dim=-2)


def _trapz_weights(grid: torch.Tensor) -> torch.Tensor:
    if grid.ndim != 1:
        raise ValueError("Trapz weights require a one-dimensional grid.")
    if grid.numel() == 1:
        return torch.ones_like(grid)

    weights = torch.empty_like(grid)
    weights[0] = 0.5 * (grid[1] - grid[0])
    weights[-1] = 0.5 * (grid[-1] - grid[-2])
    if grid.numel() > 2:
        weights[1:-1] = 0.5 * (grid[2:] - grid[:-2])
    return weights


@torch.no_grad()
def propagate_solar_to_detector_coherent(
    *,
    E_MeV: TensorLike,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    initial_state: StateLike = "nue",
    pmns: Optional[object] = None,
    theta12: Optional[TensorLike] = None,
    theta13: Optional[TensorLike] = None,
    theta23: Optional[TensorLike] = None,
    delta: Optional[TensorLike] = None,
    production_mode: ProductionMode = "point",
    rho0: Optional[TensorLike] = None,
    source: Optional[str] = None,
    solar_profile: Optional[SolarProfile] = None,
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
    cdtype = _cdtype_from_real(dtype)

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

    rho_grid, rho_weights, production_metadata = _prepare_production_grid(
        production_mode,
        rho0,
        source,
        solar_profile_obj,
        device=dev,
        dtype=dtype,
    )

    if eta is not None:
        eta_grid = _as_1d_tensor(eta, name="eta", device=dev, dtype=dtype)
        exposure = torch.ones_like(eta_grid)
        exposure = exposure / torch.trapezoid(exposure, x=eta_grid).clamp_min(torch.finfo(dtype).tiny)
        exposure_metadata = {
            "source": "user_eta_uniform",
            "normalized": True,
        }
    else:
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
            device=dev,
            dtype=dtype,
        )
        eta_grid = exposure_table.eta
        exposure = exposure_table.exposure
        exposure_metadata = {
            "source": exposure_source,
            "d1": exposure_d1,
            "d2": exposure_d2,
            "ns": exposure_ns,
            "daynight": exposure_daynight,
            "normalized": True,
            "detector_latitude_rad": detector_latitude_rad,
        }

    state0 = _prepare_initial_state(initial_state, device=dev, dtype=dtype)

    n_E = E.numel()
    n_rho = rho_grid.numel()
    n_eta = eta_grid.numel()

    if debug:
        print(
            "coherent solar detector pipeline: "
            f"n_E={n_E}, n_rho={n_rho}, n_eta={n_eta}, "
            f"production_mode={production_mode}"
        )

    surface_states = solar_surface_state(
        state0,
        pmns_obj,
        DeltamSq21,
        DeltamSq3l,
        E,
        rho_grid,
        profile=solar_profile_obj,
        antinu=antinu,
        device=dev,
        dtype=dtype,
    )

    if surface_states.shape != (n_E, n_rho, 3):
        surface_states = surface_states.reshape(n_E, n_rho, 3)

    L_vac_km = earth_distance - torch.as_tensor(R_SUN_KM, device=dev, dtype=dtype)
    if torch.any(L_vac_km < 0.0).item():
        raise ValueError("earth_distance_km must be at least one solar radius.")

    U_vac = vacuum_evolutor(
        pmns_obj,
        DeltamSq21,
        DeltamSq3l,
        E,
        L_vac_km,
        antinu=antinu,
        device=dev,
        dtype=dtype,
    )
    earth_arrival_states = torch.einsum(
        "eij,erj->eri",
        U_vac,
        surface_states,
    )

    surface_probabilities_rho = torch.abs(surface_states) ** 2
    earth_probabilities_rho = torch.abs(earth_arrival_states) ** 2

    if production_mode == "coherent":
        surface_states_stage = _integrate_over_rho(
            surface_states,
            rho_weights,
            rho_grid,
        )
        earth_states_stage = _integrate_over_rho(
            earth_arrival_states,
            rho_weights,
            rho_grid,
        )
        surface_probabilities = torch.abs(surface_states_stage) ** 2
        earth_probabilities = torch.abs(earth_states_stage) ** 2
    elif production_mode == "incoherent":
        surface_states_stage = None
        earth_states_stage = None
        surface_probabilities = _integrate_probabilities_over_rho(
            surface_probabilities_rho,
            rho_weights,
            rho_grid,
        )
        earth_probabilities = _integrate_probabilities_over_rho(
            earth_probabilities_rho,
            rho_weights,
            rho_grid,
        )
    else:
        surface_states_stage = surface_states[:, 0, :]
        earth_states_stage = earth_arrival_states[:, 0, :]
        surface_probabilities = surface_probabilities_rho[:, 0, :]
        earth_probabilities = earth_probabilities_rho[:, 0, :]

    detector_states_eta_rho = None
    detector_probabilities_eta_rho = None

    earth_operators = earth_evolutor(
        density=earth_density,
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        pmns=pmns_obj,
        E=E.reshape(n_E, 1),
        eta=eta_grid.reshape(1, n_eta),
        depth_m=detector_depth_m,
        antinu=antinu,
        reunitarize=reunitarize_earth,
    )

    if production_mode == "incoherent":
        detector_states_eta = None
        detector_probabilities_eta_rho = torch.abs(
            torch.einsum(
                "etij,erj->etri",
                earth_operators,
                earth_arrival_states,
            )
        ) ** 2
    else:
        detector_states_eta = torch.einsum(
            "etij,ej->eti",
            earth_operators,
            earth_states_stage,
        )
        detector_probabilities_eta = torch.abs(detector_states_eta) ** 2

    if production_mode == "incoherent":
        detector_probabilities_eta = _integrate_probabilities_over_rho(
            detector_probabilities_eta_rho,
            rho_weights,
            rho_grid,
        )
        detector_states_eta = None

    if integrate_exposure:
        detector_probabilities_integrated = _exposure_integral(
            detector_probabilities_eta,
            eta_grid,
            exposure,
        )
    else:
        detector_probabilities_integrated = None

    result = {
        "mode": "fully_coherent_solar_to_detector",
        "production_mode": production_mode,
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
            "detector_depth_m": float(detector_depth_m),
            "antinu": bool(antinu) if isinstance(antinu, bool) else "tensor",
            "reunitarize_earth": bool(reunitarize_earth),
        },
    }

    return result


def build_coherent_pipeline_metadata(result: dict[str, Any]) -> dict[str, Any]:
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
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    pmns: Optional[object] = None,
    theta12: Optional[TensorLike] = None,
    theta13: Optional[TensorLike] = None,
    theta23: Optional[TensorLike] = None,
    delta: Optional[TensorLike] = None,
    initial_flavours: tuple[str, ...] = ("nue", "numu", "nutau"),
    production_mode: Optional[ProductionMode] = None,
    solar_profile: Optional[SolarProfile] = None,
    antinu: Union[bool, torch.Tensor, None] = None,
    rho_chunk_size: int = 32,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    dev = _resolve_device(device)
    cdtype = _cdtype_from_real(dtype)

    E = _as_1d_tensor(result["E_MeV"], name="E_MeV", device=dev, dtype=dtype)
    rho_grid = _as_1d_tensor(result["rho_grid"], name="rho_grid", device=dev, dtype=dtype)
    rho_weights = _as_1d_tensor(result["rho_weights"], name="rho_weights", device=dev, dtype=dtype)
    eta_grid = _as_1d_tensor(result["eta"], name="eta", device=dev, dtype=dtype)
    exposure = _as_1d_tensor(result["nadir_exposure"], name="nadir_exposure", device=dev, dtype=dtype)
    earth_distance = _as_tensor(result["earth_distance_km"], device=dev, dtype=dtype)
    earth_operators = _as_tensor(result["earth_operators"], device=dev, dtype=cdtype)

    if production_mode is None:
        production_mode = result.get("production_mode", "incoherent")

    if antinu is None:
        metadata = result.get("metadata", {})
        antinu = bool(metadata.get("antinu", False)) if isinstance(metadata, dict) else False

    if rho_chunk_size <= 0:
        raise ValueError("rho_chunk_size must be positive.")

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

    n_E = E.numel()
    n_rho = rho_grid.numel()
    n_eta = eta_grid.numel()

    if earth_operators.shape != (n_E, n_eta, 3, 3):
        raise ValueError(
            "earth_operators must have shape (n_E, n_eta, 3, 3). "
            f"Got {tuple(earth_operators.shape)}."
        )

    L_vac_km = earth_distance - torch.as_tensor(R_SUN_KM, device=dev, dtype=dtype)
    U_vac = vacuum_evolutor(
        pmns_obj,
        DeltamSq21,
        DeltamSq3l,
        E,
        L_vac_km,
        antinu=antinu,
        device=dev,
        dtype=dtype,
    )

    eta_quadrature = _trapz_weights(eta_grid) * exposure
    eta_quadrature = eta_quadrature / torch.sum(eta_quadrature).clamp_min(torch.finfo(dtype).tiny)
    rho_quadrature = _trapz_weights(rho_grid) * rho_weights

    spectra = torch.empty((len(initial_flavours), n_E, 3), device=dev, dtype=dtype)
    eta_spectra = torch.empty((len(initial_flavours), n_E, n_eta, 3), device=dev, dtype=dtype)

    for i_flavour, initial_flavour in enumerate(initial_flavours):
        if production_mode == "coherent":
            earth_state_integral = torch.zeros((n_E, 3), device=dev, dtype=cdtype)
        else:
            detector_eta_accum = torch.zeros((n_E, n_eta, 3), device=dev, dtype=dtype)

        for start in range(0, n_rho, int(rho_chunk_size)):
            stop = min(start + int(rho_chunk_size), n_rho)
            rho_chunk = rho_grid[start:stop]

            surface_states = solar_surface_state(
                initial_flavour,
                pmns_obj,
                DeltamSq21,
                DeltamSq3l,
                E,
                rho_chunk,
                profile=solar_profile_obj,
                antinu=antinu,
                device=dev,
                dtype=dtype,
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
                detector_eta_accum = torch.abs(detector_states) ** 2
                break
            else:
                detector_probabilities = torch.abs(
                    torch.einsum(
                        "etij,erj->etri",
                        earth_operators,
                        earth_arrival_states,
                    )
                ) ** 2
                weights_chunk = rho_quadrature[start:stop].reshape(1, 1, -1, 1)
                detector_eta_accum = detector_eta_accum + torch.sum(
                    detector_probabilities * weights_chunk,
                    dim=2,
                )

        if production_mode == "coherent":
            detector_states = torch.einsum("etij,ej->eti", earth_operators, earth_state_integral)
            detector_eta = torch.abs(detector_states) ** 2
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

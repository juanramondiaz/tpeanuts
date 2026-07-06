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
used by the new propagation pipelines. It exists purely as a ground-truth
reference: ``pipeline_incoherent`` reimplements the same physical workflow in
torch, and this module lets that reimplementation be validated against the
original code path (same physical scenario, different numerical backend —
per-energy, per-eta Python loops over the legacy NumPy/Numba routines instead
of batched torch tensor operations).

The legacy solar workflow is incoherent:

1. ``peanuts.solar.solar_flux_mass`` builds mass-eigenstate weights.
2. ``peanuts.solar.Psolar`` builds solar flavour probabilities.
3. ``peanuts.earth.Pearth(..., massbasis=True)`` propagates the incoherent
   mass mixture through earth.
4. The detector probabilities are integrated over the legacy nadir exposure.

Module functions:
    propagate_solar_to_detector_legacypeanuts(...)
        Run the full legacy-peanuts solar-to-detector propagation for an
        energy grid and one solar source, returning mass weights,
        probabilities, grids, and metadata.
    build_legacypeanuts_pipeline_metadata(...)
        Build the JSON-serializable metadata block stored with a saved
        legacy-peanuts result.
    save_legacypeanuts_solar_detector_result(...)
        Save a legacy-peanuts propagation result to a torch file.
    load_legacypeanuts_solar_detector_result(...)
        Load a saved legacy-peanuts propagation result from a torch file.
    run_and_save_solar_to_detector_legacypeanuts(...)
        Run the legacy-peanuts pipeline and immediately save its result.
"""



from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.medium.earth.exposure_table import ExposureParameters
from tpeanuts.medium.earth.profile import EarthParameters
from tpeanuts.util.io import safe_filename_name
from tpeanuts.medium.solar.io import load_sun_earth_distance
from tpeanuts.util.constant import SUN_EARTH_DISTANCE_KM
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import TensorLike, as_tensor

from tpeanuts.pipeline.config import PropagationConfig
from tpeanuts.pipeline.pipeline_common import (
    FLAVOUR_ORDER,
)
from tpeanuts.util.torch_util import as_1d_tensor, cast_tensor_tree



def _package_root() -> Path:
    """Return the tpeanuts package root directory."""
    return Path(__file__).resolve().parents[1]


def _legacy_data_dir() -> Path:
    """Return the directory bundling the legacy peanuts reference data files."""
    return _package_root() / "data" / "peanuts"



def _legacy_expected_data_dir() -> Path:
    """Return the data directory the legacy ``peanuts`` package expects to find its files in."""
    return _package_root() / "data"


def _default_legacy_earth_density_path() -> str:
    """Return the bundled legacy Earth-density CSV path."""
    return str(_legacy_data_dir() / "earth_density.csv")


def _ensure_legacy_default_data_aliases() -> None:
    """Copy bundled legacy data files into the path layout the ``peanuts`` package expects.

    The legacy ``peanuts`` package reads its solar/Earth data files from a
    fixed relative location; this copies (without overwriting) the data
    bundled at ``_legacy_data_dir()`` into ``_legacy_expected_data_dir()`` so
    that legacy calls succeed without requiring the caller to pass explicit
    file paths.

    Raises:
        FileNotFoundError: If the bundled legacy data directory is missing.
    """
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
    """Detach, move to CPU, and flatten a tensor into a 1D NumPy array for legacy calls."""
    return x.detach().cpu().numpy().reshape(-1)


def _prepare_legacy_pmns(oscillation: OscillationParameters) -> object:
    """Build a legacy ``peanuts.pmns.PMNS`` object from tpeanuts oscillation parameters.

    Args:
        oscillation: Built pmns object plus mass splittings and antinu
            selection; only the three mixing angles and the CP phase
            (radians) are used here.

    Returns:
        A legacy ``peanuts.pmns.PMNS`` instance constructed from the same
        theta12, theta13, theta23, delta values (as plain Python floats).
    """
    from peanuts.pmns import PMNS as LegacyPMNS

    pmns = oscillation.pmns
    return LegacyPMNS(
        float(pmns.params.theta12.detach().cpu()),
        float(pmns.params.theta13.detach().cpu()),
        float(pmns.params.theta23.detach().cpu()),
        float(pmns.params.delta.detach().cpu()),
    )


def _prepare_legacy_solar_model(
    solar_model: Optional[object],
    *,
    solar_model_file: Optional[str],
    solar_flux_file: Optional[str],
) -> object:
    """Return an existing legacy ``SolarModel`` or build the default one.

    Args:
        solar_model: Existing legacy ``peanuts.solar.SolarModel`` instance.
            When provided, it is returned unchanged.
        solar_model_file: Path to the solar density/composition data file
            (legacy ``nudistr_*`` format) used when building a new model.
            Defaults to the bundled ``nudistr_b16_agss09.dat``.
        solar_flux_file: Path to the solar neutrino flux-normalisation file
            used when building a new model. Defaults to the bundled
            ``fluxes_b16.dat``.

    Returns:
        The given ``solar_model``, or a newly constructed legacy
        ``peanuts.solar.SolarModel``.
    """
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
    earth: EarthParameters,
) -> object:
    """Return an existing legacy Earth-density object or build the default one.

    Args:
        earth_density: Existing legacy ``peanuts.earth.earthdensity``
            instance. When provided, it is returned unchanged.
        earth: Earth electron-density profile construction settings; only
            ``profile_perturbative_kwargs["density_file"]`` and
            ``["tabulated_density"]`` are used when building a new object.

    Returns:
        The given ``earth_density``, or a newly constructed legacy
        ``peanuts.earth.earthdensity`` built from the bundled or
        user-specified density CSV.
    """
    if earth_density is not None:
        return earth_density

    from peanuts.earth import earthdensity

    kwargs = earth.profile_perturbative_kwargs or {}
    return earthdensity(
        density_file=kwargs.get("density_file") or _default_legacy_earth_density_path(),
        tabulated_density=kwargs.get("tabulated_density", False),
    )


def _prepare_legacy_exposure(
    eta: Optional[TensorLike],
    *,
    exposure: ExposureParameters,
    exposure_normalized: bool,
    context: RuntimeContext,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, dict[str, Any], float]:
    """Prepare legacy peanuts nadir exposure in NumPy and torch formats.

    Args:
        eta: Optional explicit eta grid. When provided, a uniform normalized
            exposure over that grid is used.
        exposure: Exposure-table construction settings (latitude, day-of-year
            window, CSV path, angle convention).
        exposure_normalized: Request normalized legacy exposure.
        context: Runtime device/dtype.

    Returns:
        Tuple with NumPy eta, NumPy exposure, torch eta, torch exposure,
        metadata, and legacy rectangle-rule spacing.
    """
    device, dtype = context.device, context.dtype

    if eta is not None:
        eta_t = as_1d_tensor(eta, name="eta", device=device, dtype=dtype)
        exposure_t = torch.ones_like(eta_t)
        exposure_t = exposure_t / torch.trapezoid(exposure_t, x=eta_t).clamp_min(torch.finfo(dtype).tiny)
        eta_np = _torch_to_numpy_1d(eta_t)
        exposure_np = _torch_to_numpy_1d(exposure_t)
        deta = float(np.pi / max(eta_np.size, 1))
        return eta_np, exposure_np, eta_t, exposure_t, {
            "source": "user_eta_uniform",
            "normalized": True,
        }, deta

    if exposure.detector_latitude_rad is None and exposure.exposure_csv_path is None:
        raise ValueError("detector_latitude_rad is required when eta and exposure_csv_path are not provided.")

    from peanuts.time_average import NadirExposure

    table = NadirExposure(
        lam=-1 if exposure.detector_latitude_rad is None else float(exposure.detector_latitude_rad),
        d1=exposure.exposure_d1,
        d2=exposure.exposure_d2,
        ns=exposure.exposure_ns,
        normalized=exposure_normalized,
        from_file=exposure.exposure_csv_path,
        angle=exposure.exposure_angle,
        daynight=exposure.exposure_daynight,
    )

    eta_np = np.asarray(table[:, 0], dtype=np.float64)
    exposure_np = np.asarray(table[:, 1], dtype=np.float64)
    eta_t = torch.as_tensor(eta_np, device=device, dtype=dtype)
    exposure_t = torch.as_tensor(exposure_np, device=device, dtype=dtype)
    deta = float(np.pi / exposure.exposure_ns)

    return eta_np, exposure_np, eta_t, exposure_t, {
        "source": "legacy_NadirExposure",
        "d1": exposure.exposure_d1,
        "d2": exposure.exposure_d2,
        "ns": exposure.exposure_ns,
        "daynight": exposure.exposure_daynight,
        "normalized": bool(exposure_normalized),
        "detector_latitude_rad": exposure.detector_latitude_rad,
        "from_file": exposure.exposure_csv_path,
        "angle": exposure.exposure_angle,
    }, deta


def _prepare_earth_distance_metadata(
    earth_distance_km: Optional[TensorLike],
    *,
    sun_earth_distance_path: Optional[str],
    use_sun_earth_distance_table: bool,
    context: RuntimeContext,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Select a Sun-Earth distance and build descriptive metadata for it.

    The legacy incoherent solar workflow does not actually use the
    Sun-Earth distance in its physics (vacuum propagation is not applied to
    mass-eigenstate weights), so this only resolves a value for
    record-keeping/comparison with the coherent pipelines; every returned
    metadata branch carries an explicit note to that effect.

    Args:
        earth_distance_km: Explicit user-provided distance in km.
        sun_earth_distance_path: Optional path to the tabulated Sun-Earth
            distance file.
        use_sun_earth_distance_table: If True and ``earth_distance_km`` is
            None, use the tabulated distance mean instead of the constant
            fallback.
        context: Runtime device/dtype.

    Returns:
        Tuple ``(distance, metadata)`` where ``distance`` is a scalar tensor
        in km and ``metadata`` describes its source.
    """
    device, dtype = context.device, context.dtype

    if earth_distance_km is not None:
        distance = as_tensor(earth_distance_km, device=device, dtype=dtype)
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
    config: PropagationConfig,
    source: str,
    solar_model: Optional[object] = None,
    solar_model_file: Optional[str] = None,
    solar_flux_file: Optional[str] = None,
    earth_density: Optional[object] = None,
    eta: Optional[TensorLike] = None,
    exposure_normalized: bool = True,
    debug: bool = False,
) -> dict[str, Any]:
    """Propagate solar neutrinos to the detector using the legacy peanuts implementation.

    Physical scenario: identical to ``pipeline_incoherent.
    propagate_solar_to_detector_incoherent`` (solar production of incoherent
    mass-eigenstate weights for source ``source``, unchanged across vacuum,
    propagated through Earth matter per nadir angle, optionally
    exposure-integrated) but every physics call is delegated to the original
    NumPy/Numba ``peanuts`` package via explicit per-energy, per-eta Python
    loops, rather than batched torch operations. This pipeline exists to
    produce reference numbers that the torch pipelines can be validated
    against.

    Args:
        E_MeV: Scalar or one-dimensional neutrino energy grid in MeV.
        config: Runtime, oscillation, exposure, Earth, and solar settings
            shared by every tpeanuts pipeline (only ``config.oscillation``,
            ``config.earth``, ``config.exposure``, ``config.detector_depth_m``
            are used; solar-profile construction settings are not since the
            legacy ``SolarModel`` is built separately).
        source: Solar neutrino source key (e.g. "8B", "pp") selecting the
            production-radius distribution.
        solar_model: Optional already-built legacy ``peanuts.solar.
            SolarModel``. When omitted, one is built from
            ``solar_model_file``/``solar_flux_file``.
        solar_model_file: Path to the legacy solar density/composition data
            file used when building a new solar model.
        solar_flux_file: Path to the legacy solar flux-normalisation file
            used when building a new solar model.
        earth_density: Optional already-built legacy
            ``peanuts.earth.earthdensity`` object. When omitted, one is
            built from ``config.earth``.
        eta: Optional explicit nadir-angle grid in radians.
        exposure_normalized: If True, request a normalized legacy nadir
            exposure table.
        debug: Print pipeline sizes when True.

    Returns:
        Dictionary with the solar and Earth-arrival mass-eigenstate weights,
        the corresponding flavour probabilities, the detector flavour
        probabilities per nadir angle (``detector_probabilities_eta``) and
        optionally exposure-integrated
        (``detector_probabilities_integrated``), the energy/rho/eta grids,
        the (unused-but-recorded) Sun-Earth distance in km, and a
        ``metadata_extra`` block describing the run (legacy data file paths,
        exposure, detector depth, antinu flag).
    """
    context = config.runtime
    E = as_1d_tensor(E_MeV, name="E_MeV", device=context.device, dtype=context.dtype)
    E_np = _torch_to_numpy_1d(E)

    dm21 = float(config.oscillation.DeltamSq21.reshape(-1)[0].item())
    dm3l = float(config.oscillation.DeltamSq3l.reshape(-1)[0].item())
    antinu = bool(config.oscillation.antinu)

    legacy_pmns = _prepare_legacy_pmns(config.oscillation)
    legacy_solar = _prepare_legacy_solar_model(
        solar_model,
        solar_model_file=solar_model_file,
        solar_flux_file=solar_flux_file,
    )
    legacy_earth = _prepare_legacy_earth_density(
        earth_density,
        earth=config.earth,
    )

    eta_np, exposure_np, eta_t, exposure_t, exposure_metadata, deta = _prepare_legacy_exposure(
        eta,
        exposure=config.exposure,
        exposure_normalized=exposure_normalized,
        context=context,
    )

    earth_distance, distance_metadata = _prepare_earth_distance_metadata(
        config.earth_distance_km,
        sun_earth_distance_path=config.sun_earth_distance_path,
        use_sun_earth_distance_table=config.use_sun_earth_distance_table,
        context=context,
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
                float(config.detector_depth_m),
                mode="analytical",
                massbasis=True,
                antinu=antinu,
            )
            detector_probabilities_eta_np[i_E, i_eta] = np.asarray(detector_prob, dtype=np.float64)

        if config.exposure.integrate_exposure:
            detector_probabilities_integrated_np[i_E] = np.sum(
                detector_probabilities_eta_np[i_E] * exposure_np[:, None],
                axis=0,
            ) * deta

    mass_weights_t = torch.as_tensor(mass_weights_np, device=context.device, dtype=context.dtype)
    solar_probabilities_t = torch.as_tensor(solar_probabilities_np, device=context.device, dtype=context.dtype)
    detector_probabilities_eta_t = torch.as_tensor(detector_probabilities_eta_np, device=context.device, dtype=context.dtype)
    detector_probabilities_integrated_t = (
        torch.as_tensor(detector_probabilities_integrated_np, device=context.device, dtype=context.dtype)
        if config.exposure.integrate_exposure
        else None
    )

    earth_kwargs = config.earth.profile_perturbative_kwargs or {}

    return {
        "mode": "legacy_peanuts_solar_to_detector",
        "flavour_order": FLAVOUR_ORDER,
        "mass_order": ["nu1", "nu2", "nu3"],
        "source": source,
        "E_MeV": E,
        "rho_grid": torch.as_tensor(radius, device=context.device, dtype=context.dtype),
        "rho_weights": torch.as_tensor(fraction, device=context.device, dtype=context.dtype),
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
            "density_file": earth_kwargs.get("density_file") or _default_legacy_earth_density_path(),
            "source": source,
            "sun_earth_distance": distance_metadata,
            "exposure": exposure_metadata,
            "detector_depth_m": float(config.detector_depth_m),
            "antinu": bool(antinu),
        },
    }


def build_legacypeanuts_pipeline_metadata(result: dict[str, Any]) -> dict[str, Any]:
    """Build the JSON-serializable metadata block for a legacy-peanuts result.

    Args:
        result: Result dictionary returned by
            ``propagate_solar_to_detector_legacypeanuts``; its tensor
            entries are summarised by shape, and any
            ``result["metadata_extra"]`` dict is merged in (overriding the
            defaults below).

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
    """Save a legacy-peanuts solar-to-detector propagation result to a torch file.

    Args:
        result: Result dictionary returned by
            ``propagate_solar_to_detector_legacypeanuts``.
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
    """Load a legacy-peanuts result saved by ``save_legacypeanuts_solar_detector_result``.

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
def run_and_save_solar_to_detector_legacypeanuts(
    output_dir: str,
    *,
    filename: str = "Legacypeanutssolardetector.pt",
    overwrite: bool = False,
    save_dtype: Optional[torch.dtype] = torch.float32,
    **pipeline_kwargs,
) -> dict[str, Any]:
    """Run the legacy-peanuts solar-to-detector pipeline and immediately save its result.

    Args:
        output_dir: Directory the result file is written into.
        filename: Output filename passed to
            ``save_legacypeanuts_solar_detector_result``.
        overwrite: If False and the target file already exists, the existing
            path is kept (file not rewritten).
        save_dtype: Floating-point dtype tensors are cast to before saving.
        **pipeline_kwargs: Forwarded to
            ``propagate_solar_to_detector_legacypeanuts`` (e.g. ``E_MeV``,
            ``config``, ``source``).

    Returns:
        The result dictionary from
        ``propagate_solar_to_detector_legacypeanuts`` with an added
        ``"output_path"`` entry pointing at the saved file.
    """
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

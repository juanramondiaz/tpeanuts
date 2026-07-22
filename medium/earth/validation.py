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
Validation helpers comparing tpeanuts.medium.earth against legacy peanuts.

The legacy Earth implementation is scalar and NumPy/Numba-based. The helpers
in this module therefore compare one trajectory at a time for pointwise Earth
probabilities, and one scalar energy at a time for exposure-integrated
probabilities.

Module functions:
    ensure_legacy_importable(...)
        Return the bundled legacy peanuts package path.
    legacy_modules(...)
        Import the legacy PMNS and Earth modules.
    legacy_pmns_from_torch(...)
        Build a legacy peanuts.pmns.PMNS object from a torch PMNS object.
    default_new_earth_density_path(...)
        Return the clean Earth-density CSV path used by the new profile.
    default_legacy_earth_density_path(...)
        Return the bundled legacy Earth-density CSV path.
    legacy_earth_density(...)
        Build a legacy peanuts.earth.earthdensity object.
    build_validation_profiles(...)
        Build matching new and legacy Earth profile objects.
    compare_earth_probability_state_with_legacy(...)
        Compare pointwise Earth probabilities against peanuts.earth.Pearth.
    compare_earth_probability_exposure_with_legacy(...)
        Compare exposure-integrated Earth probabilities against
        peanuts.earth.Pearth_integrated.
"""



from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

import tpeanuts.config.default as default
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.numerical.geometry import OdeMethod
from tpeanuts.medium.earth.exposure_integration import earth_probability_exposure
from tpeanuts.medium.earth.exposure_table import ExposureParameters
from tpeanuts.medium.earth.probability import PearthMethod, earth_probability_state
from tpeanuts.medium.earth.profile import EarthParameters, EarthProfile
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import TensorLike


def ensure_legacy_importable() -> Path:
    """Return the bundled legacy peanuts package path.

    Returns:
        Path pointing to the local ``peanuts`` package used for legacy
        validation.
    """
    package_dir = Path(__file__).resolve().parents[2]
    return package_dir / "peanuts"


@lru_cache(maxsize=1)
def legacy_modules():
    """Import the legacy PMNS and Earth modules.

    Returns:
        Pair ``(legacy_pmns, legacy_earth)`` corresponding to
        ``peanuts.pmns`` and ``peanuts.earth``.
    """
    ensure_legacy_importable()
    legacy_pmns = importlib.import_module("peanuts.pmns")
    legacy_earth = importlib.import_module("peanuts.earth")
    return legacy_pmns, legacy_earth


def legacy_pmns_from_torch(pmns_torch: object):
    """Build a legacy PMNS object from a torch-native PMNS object.

    Args:
        pmns_torch: PMNS-like object exposing a ``params`` attribute with
            tensor fields ``theta12``, ``theta13``, ``theta23``, and
            ``delta``.

    Returns:
        Instance of ``peanuts.pmns.PMNS`` with scalar angles.
    """
    legacy_pmns_module, _ = legacy_modules()

    return legacy_pmns_module.PMNS(
        float(pmns_torch.params.theta12.detach().cpu()),
        float(pmns_torch.params.theta13.detach().cpu()),
        float(pmns_torch.params.theta23.detach().cpu()),
        float(pmns_torch.params.delta.detach().cpu()),
    )


def default_legacy_earth_density_path() -> str:
    """Return the bundled legacy Earth-density CSV path.

    Returns:
        String path to ``data/peanuts/earth_density.csv``. The validation
        helpers use this file by default so the new and legacy implementations
        are compared with identical density input.
    """
    package_dir = Path(__file__).resolve().parents[2]
    return str(package_dir / "data" / "peanuts" / "earth_density.csv")


def default_new_earth_density_path() -> str:
    """Return the clean Earth-density CSV path for the new implementation.

    Returns:
        String path to the canonical PREM-derived even-power fit.
    """
    package_dir = Path(__file__).resolve().parents[2]
    return str(package_dir / default.earth_density_dir / default.earth_density_filename)


def legacy_earth_density(
    density_file: Optional[str] = None,
    *,
    tabulated_density: bool = False,
    custom_density: bool = False,
):
    """Build a legacy ``peanuts.earth.earthdensity`` object.

    Args:
        density_file: Optional Earth-density CSV path. When omitted, legacy
            peanuts uses its bundled default file.
        tabulated_density: Forwarded to the legacy constructor.
        custom_density: Forwarded to the legacy constructor.

    Returns:
        Legacy Earth density object.
    """
    _, legacy_earth = legacy_modules()

    return legacy_earth.earthdensity(
        density_file=density_file,
        tabulated_density=tabulated_density,
        custom_density=custom_density,
    )


def legacy_nadir_exposure(
    eta: Optional[TensorLike],
    *,
    exposure: ExposureParameters,
    normalized: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], float]:
    """Build the nadir grid and weights used by the legacy backend."""
    if eta is not None:
        eta_array = np.asarray(torch.as_tensor(eta).detach().cpu(), dtype=np.float64).reshape(-1)
        weights = np.ones_like(eta_array)
        norm = np.trapz(weights, x=eta_array)
        weights = weights / max(float(norm), np.finfo(np.float64).tiny)
        return eta_array, weights, {"source": "user_eta_uniform", "normalized": True}, float(np.pi / max(eta_array.size, 1))

    if exposure.detector_latitude_rad is None and exposure.exposure_csv_path is None:
        raise ValueError(
            "detector_latitude_rad is required when eta and exposure_csv_path are omitted."
        )
    from peanuts.time_average import NadirExposure

    table = NadirExposure(
        lam=-1 if exposure.detector_latitude_rad is None else float(exposure.detector_latitude_rad),
        d1=exposure.exposure_d1,
        d2=exposure.exposure_d2,
        ns=exposure.exposure_ns,
        normalized=normalized,
        from_file=exposure.exposure_csv_path,
        angle=exposure.exposure_angle,
        daynight=exposure.exposure_daynight,
    )
    eta_array = np.asarray(table[:, 0], dtype=np.float64)
    weights = np.asarray(table[:, 1], dtype=np.float64)
    return eta_array, weights, {
        "source": "legacy_NadirExposure",
        "normalized": bool(normalized),
        "d1": exposure.exposure_d1,
        "d2": exposure.exposure_d2,
        "ns": exposure.exposure_ns,
        "daynight": exposure.exposure_daynight,
        "detector_latitude_rad": exposure.detector_latitude_rad,
        "from_file": exposure.exposure_csv_path,
        "angle": exposure.exposure_angle,
    }, float(np.pi / exposure.exposure_ns)


def legacy_earth_probabilities(
    incident_state: TensorLike,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    eta: TensorLike,
    depth_m: float,
    *,
    density: object,
    massbasis: bool,
    method: PearthMethod = "analytical",
) -> np.ndarray:
    """Evaluate legacy Pearth over energy and nadir grids."""
    _, legacy_earth = legacy_modules()
    pmns = legacy_pmns_from_torch(oscillation.pmns)
    energy = np.asarray(torch.as_tensor(E_MeV).detach().cpu(), dtype=np.float64).reshape(-1)
    eta_array = np.asarray(torch.as_tensor(eta).detach().cpu(), dtype=np.float64).reshape(-1)
    states = np.asarray(torch.as_tensor(incident_state).detach().cpu())
    if states.ndim == 1:
        states = np.broadcast_to(states, (energy.size, states.shape[0]))
    if states.shape != (energy.size, 3):
        raise ValueError("incident_state must have shape (3,) or (n_energy, 3).")
    out = np.empty((energy.size, eta_array.size, 3), dtype=np.float64)
    for i_energy, energy_value in enumerate(energy):
        for i_eta, eta_value in enumerate(eta_array):
            out[i_energy, i_eta] = np.asarray(
                legacy_earth.Pearth(
                    states[i_energy],
                    density,
                    pmns,
                    float(oscillation.mass_spectrum.DeltamSq21),
                    float(oscillation.mass_spectrum.DeltamSq3l),
                    float(energy_value),
                    float(eta_value),
                    float(depth_m),
                    mode=str(method),
                    massbasis=bool(massbasis),
                    full_oscillation=False,
                    antinu=bool(oscillation.antinu),
                ),
                dtype=np.float64,
            )
    return out


def legacy_integrate_probabilities(
    probabilities_eta: np.ndarray,
    eta: np.ndarray,
    exposure: np.ndarray,
    *,
    method: str = "legacy_rectangle",
    deta: Optional[float] = None,
) -> np.ndarray:
    """Integrate legacy angular probabilities with an explicit historical rule."""
    if method == "legacy_rectangle":
        if deta is None:
            raise ValueError("deta is required for legacy_rectangle integration.")
        return np.sum(probabilities_eta * exposure[None, :, None], axis=-2) * float(deta)
    if method == "trapezoid":
        return np.trapz(
            probabilities_eta * exposure[None, :, None],
            x=eta,
            axis=-2,
        )
    raise ValueError("method must be 'legacy_rectangle' or 'trapezoid'.")


def legacy_earth_probabilities_integrated(
    incident_state: TensorLike,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    depth_m: float,
    *,
    density: object,
    exposure: ExposureParameters,
    normalized: bool,
    method: PearthMethod = "analytical",
) -> np.ndarray:
    """Evaluate the original Peanuts exposure-integrated Earth routine."""
    if exposure.detector_latitude_rad is None and exposure.exposure_csv_path is None:
        raise ValueError(
            "detector_latitude_rad is required when exposure_csv_path is omitted."
        )
    _, legacy_earth = legacy_modules()
    pmns = legacy_pmns_from_torch(oscillation.pmns)
    energy = np.asarray(torch.as_tensor(E_MeV).detach().cpu(), dtype=np.float64).reshape(-1)
    states = np.asarray(torch.as_tensor(incident_state).detach().cpu())
    if states.ndim == 1:
        states = np.broadcast_to(states, (energy.size, states.shape[0]))
    if states.shape != (energy.size, 3):
        raise ValueError("incident_state must have shape (3,) or (n_energy, 3).")
    return np.stack(
        [
            np.asarray(
                legacy_earth.Pearth_integrated(
                    states[index],
                    density,
                    pmns,
                    float(oscillation.mass_spectrum.DeltamSq21),
                    float(oscillation.mass_spectrum.DeltamSq3l),
                    float(value),
                    float(depth_m),
                    mode=str(method),
                    full_oscillation=False,
                    antinu=bool(oscillation.antinu),
                    lam=(
                        -1.0
                        if exposure.detector_latitude_rad is None
                        else float(exposure.detector_latitude_rad)
                    ),
                    d1=float(exposure.exposure_d1),
                    d2=float(exposure.exposure_d2),
                    ns=int(exposure.exposure_ns),
                    normalized=bool(normalized),
                    from_file=exposure.exposure_csv_path,
                    angle=exposure.exposure_angle,
                    daynight=exposure.exposure_daynight,
                ),
                dtype=np.float64,
            )
            for index, value in enumerate(energy)
        ],
        axis=0,
    )


def build_validation_profiles(
    *,
    profile_earth: Optional[EarthProfile] = None,
    density_file: Optional[str] = None,
    profile_perturbative_name: str = default.earth_profile_perturbative_name,
    profile_perturbative_kwargs: Optional[dict[str, Any]] = None,
    tabulated_density: bool = False,
    custom_density: bool = False,
    context: RuntimeContext = RuntimeContext.resolve(default.earth_device, default.dtype),
) -> tuple[EarthProfile, object]:
    """Build matching new and legacy Earth profiles.

    Args:
        profile_earth: Optional pre-built torch-native EarthProfile. If
            omitted, one is constructed from the remaining arguments.
        density_file: Optional density file passed to both implementations.
        profile_perturbative_name: New perturbative profile selector.
        profile_perturbative_kwargs: Optional kwargs for the selected new
            perturbative profile. ``density_file`` is inserted when provided.
        tabulated_density: Forwarded to the legacy density constructor.
        custom_density: Forwarded to the legacy density constructor.
        context: Runtime device/dtype for a newly created EarthProfile.

    Returns:
        Pair ``(profile_earth, legacy_density)``.
    """
    if profile_earth is None:
        kwargs = dict(profile_perturbative_kwargs or {})
        kwargs.setdefault(
            "density_file",
            density_file if density_file is not None else default_new_earth_density_path(),
        )

        profile_earth = EarthProfile(
            params=EarthParameters(
                profile_perturbative_name=profile_perturbative_name,
                profile_perturbative_kwargs=kwargs,
            ),
            context=context,
        )

    legacy_density = legacy_earth_density(
        density_file=density_file,
        tabulated_density=tabulated_density,
        custom_density=custom_density,
    )

    return profile_earth, legacy_density


def _numpy_state(nustate: TensorLike, *, complex_state: bool) -> np.ndarray:
    dtype = np.complex128 if complex_state else float
    if torch.is_tensor(nustate):
        return np.asarray(nustate.detach().cpu().numpy(), dtype=dtype)
    return np.asarray(nustate, dtype=dtype)


def _diff_summary(torch_value: np.ndarray, legacy_value: np.ndarray) -> dict[str, Any]:
    abs_diff = np.abs(torch_value - legacy_value)
    return {
        "torch": torch_value,
        "legacy": legacy_value,
        "abs_diff": abs_diff,
        "max_abs": float(np.max(abs_diff)),
    }


def compare_earth_probability_state_with_legacy(
    nustate: TensorLike,
    oscillation: OscillationParameters,
    E_MeV: float,
    eta: float,
    depth_m: float,
    *,
    profile_earth: Optional[EarthProfile] = None,
    density_file: Optional[str] = None,
    method: PearthMethod = "analytical",
    massbasis: bool = True,
    reunitarize: bool = default.earth_reunitarize,
    nsteps: int = default.earth_probability_nsteps,
    ode_method: OdeMethod | None = default.earth_numerical_method,
    context: RuntimeContext = RuntimeContext.resolve(default.earth_device, default.dtype),
) -> dict[str, Any]:
    """Compare ``earth_probability_state`` with legacy ``peanuts.earth.Pearth``.

    Args:
        nustate: Initial state. Interpreted as mass weights when
            ``massbasis=True`` and coherent flavour amplitudes otherwise.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Scalar neutrino energy in MeV.
        eta: Scalar detector nadir angle in radians.
        depth_m: Detector depth in metres.
        profile_earth: Optional torch-native EarthProfile.
        density_file: Optional density CSV used to construct both profiles.
        method: ``"analytical"`` or ``"numerical"``.
        massbasis: Basis convention passed to both implementations.
        reunitarize: Whether to reunitarize the analytical torch evolutor.
        nsteps: Number of numerical steps for torch numerical mode.
        ode_method: Numerical sampling rule for torch numerical mode.
        context: Runtime device/dtype for the new implementation.

    Returns:
        Dictionary with ``torch``, ``legacy``, ``abs_diff`` and ``max_abs``.
    """
    method = str(method).lower().strip()
    device, dtype = context.device, context.dtype
    profile_earth, legacy_density = build_validation_profiles(
        profile_earth=profile_earth,
        density_file=density_file,
        context=context,
    )
    _, legacy_earth = legacy_modules()
    legacy_pmns = legacy_pmns_from_torch(oscillation.pmns)

    torch_p = earth_probability_state(
        _numpy_state(nustate, complex_state=not massbasis),
        profile_earth,
        oscillation,
        torch.tensor(E_MeV, device=device, dtype=dtype),
        torch.tensor(eta, device=device, dtype=dtype),
        float(depth_m),
        method=method,
        massbasis=bool(massbasis),
        full_oscillation=False,
        nsteps=nsteps,
        ode_method=ode_method,
        context=context,
        reunitarize=reunitarize,
    )

    legacy_p = legacy_earth.Pearth(
        _numpy_state(nustate, complex_state=not massbasis),
        legacy_density,
        legacy_pmns,
        float(oscillation.mass_spectrum.DeltamSq21),
        float(oscillation.mass_spectrum.DeltamSq3l),
        float(E_MeV),
        float(eta),
        float(depth_m),
        mode=method,
        massbasis=bool(massbasis),
        full_oscillation=False,
        antinu=bool(oscillation.antinu),
    )

    return _diff_summary(
        torch.as_tensor(torch_p).detach().cpu().numpy(),
        np.asarray(legacy_p, dtype=float),
    )


def compare_earth_probability_exposure_with_legacy(
    nustate: TensorLike,
    oscillation: OscillationParameters,
    E_MeV: float,
    depth_m: float,
    *,
    profile_earth: Optional[EarthProfile] = None,
    density_file: Optional[str] = None,
    method: PearthMethod = "analytical",
    exposure: ExposureParameters = ExposureParameters(),
    normalized_exposure: bool = default.earth_normalized_exposure,
    chunk_eta: Optional[int] = default.earth_chunk_eta,
    reunitarize: bool = default.earth_reunitarize,
    nsteps: int = default.earth_probability_nsteps,
    ode_method: OdeMethod | None = default.earth_numerical_method,
    context: RuntimeContext = RuntimeContext.resolve(default.earth_device, default.dtype),
) -> dict[str, Any]:
    """Compare exposure-integrated ``earth_probability_exposure`` with legacy peanuts.

    Args:
        nustate: Initial mass-basis weights. The legacy integrated function
            always calls ``Pearth(..., massbasis=True)``.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Scalar neutrino energy in MeV.
        depth_m: Detector depth in metres.
        profile_earth: Optional torch-native EarthProfile.
        density_file: Optional density CSV used to construct both profiles.
        method: ``"analytical"`` or ``"numerical"``.
        exposure: Exposure-table construction settings (latitude, day-of-year
            window, CSV path, angle convention).
        normalized_exposure: Whether to normalize the exposure weights.
        cache_dir: Cache directory used by the new exposure builder.
        chunk_eta: Number of eta samples per torch batch.
        reunitarize: Whether to reunitarize the analytical torch evolutor.
        nsteps: Number of numerical steps for torch numerical mode.
        ode_method: Numerical sampling rule for torch numerical mode.
        context: Runtime device/dtype for the new implementation.

    Returns:
        Dictionary with ``torch``, ``legacy``, ``abs_diff`` and ``max_abs``.
    """
    method = str(method).lower().strip()
    device, dtype = context.device, context.dtype
    profile_earth, legacy_density = build_validation_profiles(
        profile_earth=profile_earth,
        density_file=density_file,
        context=context,
    )
    _, legacy_earth = legacy_modules()
    legacy_pmns = legacy_pmns_from_torch(oscillation.pmns)

    torch_p = earth_probability_exposure(
        _numpy_state(nustate, complex_state=False),
        profile_earth,
        oscillation,
        torch.tensor(E_MeV, device=device, dtype=dtype),
        float(depth_m),
        method=method,
        massbasis=True,
        exposure=exposure,
        normalized_exposure=normalized_exposure,
        context=context,
        chunk_eta=chunk_eta,
        reunitarize=reunitarize,
        nsteps=nsteps,
        ode_method=ode_method,
    )

    legacy_p = legacy_earth.Pearth_integrated(
        _numpy_state(nustate, complex_state=False),
        legacy_density,
        legacy_pmns,
        float(oscillation.mass_spectrum.DeltamSq21),
        float(oscillation.mass_spectrum.DeltamSq3l),
        float(E_MeV),
        float(depth_m),
        mode=method,
        full_oscillation=False,
        antinu=bool(oscillation.antinu),
        lam=float(exposure.detector_latitude_rad),
        d1=float(exposure.exposure_d1),
        d2=float(exposure.exposure_d2),
        ns=int(exposure.exposure_ns),
        normalized=bool(normalized_exposure),
        from_file=exposure.exposure_csv_path,
        angle=exposure.exposure_angle,
        daynight=exposure.exposure_daynight,
    )

    return _diff_summary(
        torch.as_tensor(torch_p).detach().cpu().numpy(),
        np.asarray(legacy_p, dtype=float),
    )

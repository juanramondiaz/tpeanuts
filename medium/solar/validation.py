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
Validation helpers comparing tpeanuts.medium.solar against legacy peanuts.

Module functions:
    ensure_legacy_importable(...)
        Return the bundled legacy peanuts package path.
    legacy_modules(...)
        Import the legacy PMNS module and a global-model-free copy of the
        legacy solar module.
    compare_solar_probability_state_with_legacy(...)
        Compare torch-native ``solar_probability_state`` flavour probabilities
        against legacy ``peanuts.solar.Psolar`` for a single source and
        energy.
"""



from __future__ import annotations

import importlib
import types
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.medium.solar.profile import SolarParameters, SolarProfile
from tpeanuts.medium.solar.probability import solar_probability_state
from tpeanuts.medium.solar.io import default_legacy_data_dir
from tpeanuts.util.context import RuntimeContext


def ensure_legacy_importable() -> Path:
    """Return the bundled legacy peanuts package path.

    Returns:
        Path pointing to the local ``peanuts`` package used for legacy
        validation. Importing is delegated to Python's normal import
        machinery; this helper only centralizes the expected location.
    """
    package_dir = Path(__file__).resolve().parents[2]

    return package_dir / "peanuts"


@lru_cache(maxsize=1)
def legacy_modules():
    """Import the legacy PMNS module and a global-model-free legacy solar module.

    The original ``peanuts/solar.py`` instantiates a module-level global
    ``SolarModel()`` at import time, which eagerly reads the legacy data
    files from disk. To avoid that side effect (and to allow repeated,
    independent comparisons), this helper loads the module source, strips
    the ``model = SolarModel()`` global-instantiation line, and executes the
    patched source into a fresh module object under a different name
    ("peanuts.solar_no_global_model").

    Returns:
        Pair ``(legacy_pmns, legacy_solar)`` corresponding to the imported
        ``peanuts.pmns`` module and the patched ``peanuts.solar`` module
        (without its global ``SolarModel`` instance).
    """
    ensure_legacy_importable()
    legacy_pmns = importlib.import_module("peanuts.pmns")
    importlib.import_module("peanuts.matter_mixing")
    importlib.import_module("peanuts.files")

    package_dir = Path(__file__).resolve().parents[2]
    solar_path = package_dir / "peanuts" / "solar.py"

    source = solar_path.read_text(encoding="utf-8")
    source = source.replace("\nmodel = SolarModel()\n", "\n")

    legacy_solar = types.ModuleType("peanuts.solar_no_global_model")
    legacy_solar.__file__ = str(solar_path)
    legacy_solar.__package__ = "peanuts"
    exec(compile(source, str(solar_path), "exec"), legacy_solar.__dict__)

    return legacy_pmns, legacy_solar


def legacy_solar_model(
    solar_model: Optional[object] = None,
    *,
    solar_model_file: Optional[str] = None,
    solar_flux_file: Optional[str] = None,
) -> object:
    """Return an explicit legacy SolarModel without modifying package data."""
    if solar_model is not None:
        return solar_model
    _, legacy_solar = legacy_modules()
    data_dir = default_legacy_data_dir()
    return legacy_solar.SolarModel(
        solar_model_file=solar_model_file or str(data_dir / "nudistr_b16_agss09.dat"),
        flux_file=solar_flux_file or str(data_dir / "fluxes_b16.dat"),
    )


def legacy_solar_mass_weights(
    oscillation: OscillationParameters,
    E_MeV: np.ndarray,
    source: str,
    *,
    solar_model: object,
) -> np.ndarray:
    """Evaluate legacy incoherent mass weights over a one-dimensional energy grid."""
    legacy_pmns_module, legacy_solar = legacy_modules()
    pmns = legacy_pmns_module.PMNS(
        float(oscillation.pmns.params.theta12.detach().cpu()),
        float(oscillation.pmns.params.theta13.detach().cpu()),
        float(oscillation.pmns.params.theta23.detach().cpu()),
        float(oscillation.pmns.params.delta.detach().cpu()),
    )
    energy = np.asarray(E_MeV, dtype=np.float64).reshape(-1)
    return np.stack(
        [
            np.asarray(
                legacy_solar.solar_flux_mass(
                    pmns.theta12,
                    pmns.theta13,
                    float(oscillation.mass_spectrum.DeltamSq21),
                    float(oscillation.mass_spectrum.DeltamSq3l),
                    float(value),
                    solar_model.radius(),
                    solar_model.density(),
                    solar_model.fraction(source),
                ),
                dtype=np.float64,
            )
            for value in energy
        ],
        axis=0,
    )


def legacy_solar_flavour_probabilities(
    oscillation: OscillationParameters,
    E_MeV: np.ndarray,
    source: str,
    *,
    solar_model: object,
) -> np.ndarray:
    """Evaluate legacy Psolar independently for validation diagnostics."""
    legacy_pmns_module, legacy_solar = legacy_modules()
    pmns = legacy_pmns_module.PMNS(
        float(oscillation.pmns.params.theta12.detach().cpu()),
        float(oscillation.pmns.params.theta13.detach().cpu()),
        float(oscillation.pmns.params.theta23.detach().cpu()),
        float(oscillation.pmns.params.delta.detach().cpu()),
    )
    energy = np.asarray(E_MeV, dtype=np.float64).reshape(-1)
    return np.stack(
        [
            np.asarray(
                legacy_solar.Psolar(
                    pmns,
                    float(oscillation.mass_spectrum.DeltamSq21),
                    float(oscillation.mass_spectrum.DeltamSq3l),
                    float(value),
                    solar_model.radius(),
                    solar_model.density(),
                    solar_model.fraction(source),
                ),
                dtype=np.float64,
            )
            for value in energy
        ],
        axis=0,
    )


def compare_solar_probability_state_with_legacy(
    source: str,
    oscillation: OscillationParameters,
    E_MeV: float,
    *,
    context: RuntimeContext = RuntimeContext.resolve("cpu", torch.float64),
    legacy_precision: bool = False,
) -> dict[str, np.ndarray | float]:
    """Compare ``solar_probability_state`` with legacy ``peanuts.solar.Psolar`` for one source.

    Builds the torch-native default B16 solar profile, evaluates
    :func:`tpeanuts.medium.solar.probability.solar_probability_state` for the
    requested source and energy, then independently constructs the legacy
    ``SolarModel``/``PMNS`` objects from the original (non-torch) data files
    and evaluates ``peanuts.solar.Psolar`` with the same physical inputs, so
    the two flavour-probability vectors can be compared directly.

    Args:
        source: Solar source key available in both the torch profile
            (``SolarProfile.default``) and the legacy ``SolarModel``
            (e.g. "8B", "pp", "hep").
        oscillation: Built pmns object plus mass splittings (DeltamSq21,
            DeltamSq3l in eV^2) and antinu selection.
        E_MeV: Scalar neutrino energy in MeV.
        context: Runtime device/dtype for the torch-native calculation.
        legacy_precision: If True, evaluate the underlying matter-mixing
            angles with the legacy peanuts ``Vk`` prefactor for
            bit-comparable validation (see ``medium.solar.matter_mixing``).

    Returns:
        Dictionary with:
            "torch": Torch-native flavour-probability vector as a NumPy
                array, shape (3,).
            "legacy": Legacy flavour-probability vector as a NumPy array,
                shape (3,).
            "abs_diff": Elementwise absolute difference between the two,
                shape (3,).
            "max_abs": Maximum absolute difference across flavours.
    """
    legacy_pmns_module, legacy_solar = legacy_modules()

    # Always use the B16 AGSS09 profile here: the legacy peanuts reference
    # was generated with B16, so the comparison must use the same model
    # regardless of what the package-wide default profile is.
    _solar_dir = Path(__file__).resolve().parents[2] / "data" / "solar"
    _b16_params = SolarParameters(
        model_path=str(_solar_dir / "nudistr_b16_agss09.csv"),
        fluxes_path=str(_solar_dir / "fluxes_b16.csv"),
    )
    profile = SolarProfile.default(params=_b16_params, context=context)
    torch_p = solar_probability_state(
        oscillation,
        torch.tensor(E_MeV, device=profile.device, dtype=profile.dtype),
        profile,
        source,
        legacy_precision=legacy_precision,
    )

    package_dir = Path(__file__).resolve().parents[2]
    legacy_data = package_dir / "data" / "peanuts"

    legacy_model = legacy_solar.SolarModel(
        solar_model_file=str(legacy_data / "nudistr_b16_agss09.dat"),
        flux_file=str(legacy_data / "fluxes_b16.dat"),
        spectrum_files={
            "8B": str(legacy_data / "8B_shape_Ortiz_et_al.csv"),
            "hep": str(legacy_data / "hep_shape.csv"),
            "pp": str(legacy_data / "pp_shape.csv"),
            "17F": str(legacy_data / "f17_shape.csv"),
            "7Beground": str(legacy_data / "be7ground_shape.csv"),
            "7Beexcited": str(legacy_data / "be7excited_shape.csv"),
            "13N": str(legacy_data / "n13_shape.csv"),
            "15O": str(legacy_data / "o15_shape.csv"),
        },
    )
    pmns_torch = oscillation.pmns
    legacy_pmns = legacy_pmns_module.PMNS(
        float(pmns_torch.params.theta12.detach().cpu()),
        float(pmns_torch.params.theta13.detach().cpu()),
        float(pmns_torch.params.theta23.detach().cpu()),
        float(pmns_torch.params.delta.detach().cpu()),
    )
    legacy_p = legacy_solar.Psolar(
        legacy_pmns,
        float(oscillation.mass_spectrum.DeltamSq21),
        float(oscillation.mass_spectrum.DeltamSq3l),
        E_MeV,
        legacy_model.radius(),
        legacy_model.density(),
        legacy_model.fraction(source),
    )

    torch_np = torch_p.detach().cpu().numpy()
    legacy_np = np.asarray(legacy_p, dtype=float)
    abs_diff = np.abs(torch_np - legacy_np)

    return {
        "torch": torch_np,
        "legacy": legacy_np,
        "abs_diff": abs_diff,
        "max_abs": float(np.max(abs_diff)),
    }

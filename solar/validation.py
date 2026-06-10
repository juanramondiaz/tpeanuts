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
Validation helpers comparing tpeanuts.solar against legacy peanuts.
"""



from __future__ import annotations

import importlib
import types
from pathlib import Path

import numpy as np
import torch

from tpeanuts.solar.profiles import load_default_solar_profile
from tpeanuts.solar.probabilities import psolar


def ensure_legacy_importable() -> Path:
    package_dir = Path(__file__).resolve().parents[1]

    return package_dir / "peanuts"


def legacy_modules():
    ensure_legacy_importable()
    legacy_pmns = importlib.import_module("peanuts.pmns")
    importlib.import_module("peanuts.matter_mixing")
    importlib.import_module("peanuts.files")

    package_dir = Path(__file__).resolve().parents[1]
    solar_path = package_dir / "peanuts" / "solar.py"

    source = solar_path.read_text(encoding="utf-8")
    source = source.replace("\nmodel = SolarModel()\n", "\n")

    legacy_solar = types.ModuleType("peanuts.solar_no_global_model")
    legacy_solar.__file__ = str(solar_path)
    legacy_solar.__package__ = "peanuts"
    exec(compile(source, str(solar_path), "exec"), legacy_solar.__dict__)

    return legacy_pmns, legacy_solar


def compare_psolar_with_legacy(
    source: str,
    pmns_torch,
    dm21_eV2: float,
    dm3l_eV2: float,
    E_MeV: float,
    *,
    device="cpu",
    dtype=torch.float64,
) -> dict[str, np.ndarray | float]:
    legacy_pmns_module, legacy_solar = legacy_modules()

    profile = load_default_solar_profile(device=device, dtype=dtype)
    torch_p = psolar(
        pmns_torch,
        dm21_eV2,
        dm3l_eV2,
        torch.tensor(E_MeV, device=profile.device, dtype=profile.dtype),
        profile.radius,
        profile.density,
        profile.production_fraction(source),
    )

    package_dir = Path(__file__).resolve().parents[1]
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
    legacy_pmns = legacy_pmns_module.PMNS(
        float(pmns_torch.theta12.detach().cpu()),
        float(pmns_torch.theta13.detach().cpu()),
        float(pmns_torch.theta23.detach().cpu()),
        float(pmns_torch.delta.detach().cpu()),
    )
    legacy_p = legacy_solar.Psolar(
        legacy_pmns,
        dm21_eV2,
        dm3l_eV2,
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

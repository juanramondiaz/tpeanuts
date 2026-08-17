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

"""Pytest-compatible tests for tpeanuts.medium.solar.io (density/composition).

Production/flux/spectrum loaders now live in ``source.solar.io`` -- see
``source/solar/test/test1_io.py``.
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from tpeanuts.medium.solar.io import (
    load_solar_composition,
    load_solar_density,
)
from tpeanuts.medium.vacuum.io import load_sun_earth_distance
import tpeanuts.config.default as default
from tpeanuts.util.io import package_dir
from tpeanuts.util.test_utils import assert_close


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def assert_same_device(actual: torch.device, expected: torch.device) -> None:
    assert actual.type == torch.device(expected).type


def test_package_and_configured_data_directories_exist():
    root = package_dir()
    solar_dir = root / default.solar_data_dir
    legacy_dir = root / default.legacy_data_dir

    assert root.exists()
    assert solar_dir.exists()
    assert legacy_dir.exists()
    assert solar_dir.is_dir()
    assert legacy_dir.is_dir()


def test_load_solar_density_from_canonical_csv(tmp_path):
    density_path = tmp_path / "density.csv"
    pd.DataFrame(
        {
            "radius": [0.0, 0.5, 1.0],
            "electron_density_mol_cm3": [100.0, 10.0, 1.0],
            "neutron_density_mol_cm3": [80.0, 8.0, 0.8],
        }
    ).to_csv(density_path, index=False)

    density = load_solar_density(density_path, device=DEVICE, dtype=DTYPE)

    assert_close(density["electron_density"], torch.tensor([100.0, 10.0, 1.0], device=DEVICE, dtype=DTYPE), name="electron density")
    assert_close(density["neutron_density"], torch.tensor([80.0, 8.0, 0.8], device=DEVICE, dtype=DTYPE), name="neutron density")


@pytest.mark.parametrize(
    ("radius", "density", "message"),
    [
        ([0.0, 0.5, 0.5], [10.0, 5.0, 1.0], "strictly increasing"),
        ([0.0, 0.5, 1.01], [10.0, 5.0, 1.0], "0 <= r/R_sun <= 1"),
        ([0.0, 0.5, 1.0], [10.0, float("nan"), 1.0], "non-finite"),
        ([0.0, 0.5, 1.0], [10.0, -1.0, 1.0], "non-negative"),
    ],
)
def test_load_solar_density_rejects_invalid_grid_or_values(
    tmp_path, radius, density, message,
):
    path = tmp_path / "density.csv"
    pd.DataFrame(
        {"radius": radius, "electron_density_mol_cm3": density}
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        load_solar_density(path, device=DEVICE, dtype=DTYPE)


def test_load_sun_earth_distance_from_synthetic_csv(tmp_path):
    path = tmp_path / "distance.csv"
    pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-07-01"],
            "distance_km": [147.1e6, 152.1e6],
            "distance_AU": [0.983, 1.017],
        }
    ).to_csv(path, index=False)

    distance = load_sun_earth_distance(path, device=DEVICE, dtype=DTYPE)

    assert distance["date"] == ["2026-01-01", "2026-07-01"]
    assert_close(distance["distance_km"], torch.tensor([147.1e6, 152.1e6], device=DEVICE, dtype=DTYPE), name="distance km")
    assert_close(distance["distance_AU"], torch.tensor([0.983, 1.017], device=DEVICE, dtype=DTYPE), name="distance AU")


def test_load_sun_earth_distance_rejects_missing_required_columns(tmp_path):
    path = tmp_path / "bad_distance.csv"
    pd.DataFrame({"date": ["2026-01-01"], "distance_km": [147.1e6]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_sun_earth_distance(path, device=DEVICE, dtype=DTYPE)


def _struct_nu_columns() -> list[str]:
    leading = [
        "R_sun", "mass_sun", "L_sun", "logR", "logT", "logP", "logRho",
        "Csound", "dm", "nu_pp", "nu_pep", "nu_hep", "nu_7Be", "nu_8B",
        "nu_13N", "nu_15O", "nu_17F", "log_ne",
    ]
    isotopes = [
        "H1", "He4", "He3", "C12", "C13", "N14", "N15", "O16", "O17", "O18",
        "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc",
        "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
    ]
    return leading + isotopes


def test_load_solar_composition_pure_hydrogen_and_helium_give_exact_ratio(tmp_path):
    # Pure H1 (bare proton: 1 electron, 0 neutrons) -> n_n/n_e = 0 exactly.
    # Pure He4 (bare alpha: Z=N=2) -> n_n/n_e = 1 exactly. Both are simple,
    # hand-verifiable closed-form checks of the isotope (A, Z) table and the
    # fully-ionized-plasma ratio formula, independent of any real solar
    # model's absolute density normalization (which this function never
    # reads -- see load_solar_composition's docstring).
    columns = _struct_nu_columns()
    row_pure_h1 = {c: 0.0 for c in columns}
    row_pure_h1["R_sun"] = 0.0
    row_pure_h1["H1"] = 1.0

    row_pure_he4 = {c: 0.0 for c in columns}
    row_pure_he4["R_sun"] = 1.0
    row_pure_he4["He4"] = 1.0

    path = tmp_path / "struct_nu_synthetic.dat"
    with open(path, "w") as f:
        f.write(" ".join(columns) + "\n")
        f.write(" ".join(str(row_pure_h1[c]) for c in columns) + "\n")
        f.write(" ".join(str(row_pure_he4[c]) for c in columns) + "\n")

    composition = load_solar_composition(path, device=DEVICE, dtype=DTYPE)

    assert set(composition) == {"radius", "neutron_to_electron_ratio"}
    assert_close(composition["radius"], torch.tensor([0.0, 1.0], device=DEVICE, dtype=DTYPE), name="composition radius")
    assert_close(
        composition["neutron_to_electron_ratio"],
        torch.tensor([0.0, 1.0], device=DEVICE, dtype=DTYPE),
        name="pure H1 / pure He4 neutron-to-electron ratio",
    )


def test_default_bundled_solar_composition_is_finite_and_decreases_outward():
    composition = load_solar_composition(device=DEVICE, dtype=DTYPE)

    assert composition["radius"].ndim == 1
    assert composition["neutron_to_electron_ratio"].shape == composition["radius"].shape
    assert torch.isfinite(composition["neutron_to_electron_ratio"]).all()
    assert bool(torch.all(composition["neutron_to_electron_ratio"] >= 0.0))
    # The solar core is helium-enriched by hydrogen burning (more neutrons
    # per free electron than the near-primordial envelope), so the ratio
    # should be higher at the core than at the surface.
    assert float(composition["neutron_to_electron_ratio"][0]) > float(
        composition["neutron_to_electron_ratio"][-1]
    )


def test_legacy_provider_loads_density():
    density = load_solar_density(provider="legacy", device=DEVICE, dtype=DTYPE)

    assert density["radius"].shape == density["electron_density"].shape

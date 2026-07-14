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

"""Pytest-compatible tests for tpeanuts.medium.solar.io."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from tpeanuts.medium.solar.io import (
    as_tensor,
    default_legacy_data_dir,
    default_solar_data_dir,
    load_b16_fluxes,
    load_b16_solar_model,
    load_spectrum_csv,
    load_sun_earth_distance,
    package_dir,
)
from tpeanuts.util.test_utils import assert_close


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def assert_same_device(actual: torch.device, expected: torch.device) -> None:
    assert actual.type == torch.device(expected).type


def test_package_and_default_data_directories_exist():
    root = package_dir()
    solar_dir = default_solar_data_dir()
    legacy_dir = default_legacy_data_dir()

    assert root.exists()
    assert solar_dir.exists()
    assert legacy_dir.exists()
    assert solar_dir.is_dir()
    assert legacy_dir.is_dir()


def test_as_tensor_copies_readonly_numpy_arrays_and_sets_device_dtype():
    array = np.asarray([1.0, 2.0, 3.0])
    array.setflags(write=False)

    value = as_tensor(array, device=DEVICE, dtype=DTYPE)

    assert_same_device(value.device, DEVICE)
    assert value.dtype == DTYPE
    assert_close(value, torch.tensor([1.0, 2.0, 3.0], device=DEVICE, dtype=DTYPE), name="readonly numpy conversion")


def test_load_b16_solar_model_from_synthetic_csv(tmp_path):
    path = tmp_path / "model.csv"
    pd.DataFrame(
        {
            "radius": [0.0, 0.5, 1.0],
            "density_log_10": [2.0, 1.0, 0.0],
            "pp fraction": [0.0, 1.0, 0.0],
            "8B fraction": [1.0, 0.0, 0.0],
            "not_a_fraction": [9.0, 9.0, 9.0],
        }
    ).to_csv(path, index=False)

    model = load_b16_solar_model(path, device=DEVICE, dtype=DTYPE)

    assert set(model) == {"radius", "density", "fractions"}
    assert_same_device(model["radius"].device, DEVICE)
    assert model["radius"].dtype == DTYPE
    assert_close(model["radius"], torch.tensor([0.0, 0.5, 1.0], device=DEVICE, dtype=DTYPE), name="radius")
    assert_close(model["density"], torch.tensor([100.0, 10.0, 1.0], device=DEVICE, dtype=DTYPE), name="density linear scale")
    assert sorted(model["fractions"]) == ["8B", "pp"]
    assert_close(model["fractions"]["pp"], torch.tensor([0.0, 1.0, 0.0], device=DEVICE, dtype=DTYPE), name="pp fraction")


def test_load_b16_fluxes_from_synthetic_csv(tmp_path):
    path = tmp_path / "fluxes.csv"
    pd.DataFrame(
        {
            "fraction": ["pp", "8B", "hep"],
            "flux": [6.0e10, 4.5e6, 8.0e3],
        }
    ).to_csv(path, index=False)

    fluxes = load_b16_fluxes(path, device=DEVICE, dtype=DTYPE)

    assert sorted(fluxes) == ["8B", "hep", "pp"]
    for value in fluxes.values():
        assert value.shape == ()
        assert_same_device(value.device, DEVICE)
        assert value.dtype == DTYPE
    assert_close(fluxes["8B"], torch.tensor(4.5e6, device=DEVICE, dtype=DTYPE), name="8B flux")


def test_load_spectrum_csv_uses_default_first_two_columns(tmp_path):
    path = tmp_path / "spectrum.csv"
    pd.DataFrame(
        {
            "Energy": [0.0, 1.0, 2.0],
            "Spectrum": [0.0, 0.5, 0.0],
            "ignored": [9.0, 9.0, 9.0],
        }
    ).to_csv(path, index=False)

    spectrum = load_spectrum_csv(path, device=DEVICE, dtype=DTYPE)

    assert_close(spectrum["energy"], torch.tensor([0.0, 1.0, 2.0], device=DEVICE, dtype=DTYPE), name="default energy column")
    assert_close(spectrum["spectrum"], torch.tensor([0.0, 0.5, 0.0], device=DEVICE, dtype=DTYPE), name="default spectrum column")


def test_load_spectrum_csv_accepts_explicit_column_names(tmp_path):
    path = tmp_path / "spectrum_named.csv"
    pd.DataFrame(
        {
            "E_MeV": [0.1, 0.2],
            "weight": [1.0, 3.0],
        }
    ).to_csv(path, index=False)

    spectrum = load_spectrum_csv(path, energy_column="E_MeV", spectrum_column="weight", device=DEVICE, dtype=DTYPE)

    assert_close(spectrum["energy"], torch.tensor([0.1, 0.2], device=DEVICE, dtype=DTYPE), name="explicit energy")
    assert_close(spectrum["spectrum"], torch.tensor([1.0, 3.0], device=DEVICE, dtype=DTYPE), name="explicit spectrum")


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


def test_default_bundled_solar_model_has_expected_sources_and_finite_values():
    model = load_b16_solar_model(device=DEVICE, dtype=DTYPE)

    assert model["radius"].ndim == 1
    assert model["density"].shape == model["radius"].shape
    assert torch.isfinite(model["radius"]).all()
    assert torch.isfinite(model["density"]).all()
    assert bool(torch.all(torch.diff(model["radius"]) > 0.0))
    assert bool(torch.all(model["radius"] >= 0.0))
    assert bool(torch.all(model["radius"] <= 1.0))
    assert bool(torch.all(model["density"] > 0.0))
    assert {"pp", "8B", "7Be", "hep"}.issubset(model["fractions"])
    for fraction in model["fractions"].values():
        assert fraction.shape == model["radius"].shape
        assert torch.isfinite(fraction).all()


def test_default_bundled_fluxes_are_positive_and_include_main_sources():
    fluxes = load_b16_fluxes(device=DEVICE, dtype=DTYPE)

    assert {"pp", "8B", "7Be", "hep"}.issubset(fluxes)
    for value in fluxes.values():
        assert value.shape == ()
        assert torch.isfinite(value)
        assert bool(value > 0.0)

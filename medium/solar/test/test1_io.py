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

import pandas as pd
import pytest
import torch

from tpeanuts.medium.solar.io import (
    available_solar_spectrum_sources,
    load_solar_composition,
    load_solar_fluxes,
    load_solar_density,
    load_solar_production,
    load_spectrum_csv,
    load_solar_spectrum,
    load_sun_earth_distance,
)
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


def test_load_solar_density_and_production_from_canonical_csvs(tmp_path):
    density_path = tmp_path / "density.csv"
    production_path = tmp_path / "production.csv"
    pd.DataFrame(
        {
            "radius": [0.0, 0.5, 1.0],
            "electron_density_mol_cm3": [100.0, 10.0, 1.0],
            "neutron_density_mol_cm3": [80.0, 8.0, 0.8],
        }
    ).to_csv(density_path, index=False)
    pd.DataFrame(
        {
            "radius": [0.0, 0.5, 1.0],
            "pp fraction": [0.0, 1.0, 0.0],
            "8B fraction": [1.0, 0.0, 0.0],
        }
    ).to_csv(production_path, index=False)

    density = load_solar_density(density_path, device=DEVICE, dtype=DTYPE)
    production = load_solar_production(production_path, device=DEVICE, dtype=DTYPE)

    assert_close(density["electron_density"], torch.tensor([100.0, 10.0, 1.0], device=DEVICE, dtype=DTYPE), name="electron density")
    assert_close(density["neutron_density"], torch.tensor([80.0, 8.0, 0.8], device=DEVICE, dtype=DTYPE), name="neutron density")
    assert sorted(production["fractions"]) == ["8B", "pp"]


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


def test_load_solar_production_rejects_nonfinite_and_nonpositive_distributions(tmp_path):
    nonfinite = tmp_path / "nonfinite.csv"
    zero = tmp_path / "zero.csv"
    pd.DataFrame(
        {"radius": [0.0, 0.5, 1.0], "pp fraction": [0.0, float("nan"), 1.0]}
    ).to_csv(nonfinite, index=False)
    pd.DataFrame(
        {"radius": [0.0, 0.5, 1.0], "pp fraction": [0.0, 0.0, 0.0]}
    ).to_csv(zero, index=False)
    with pytest.raises(ValueError, match="non-finite"):
        load_solar_production(nonfinite, device=DEVICE, dtype=DTYPE)
    with pytest.raises(ValueError, match="non-positive normalization"):
        load_solar_production(zero, device=DEVICE, dtype=DTYPE)


def test_load_solar_fluxes_from_synthetic_csv(tmp_path):
    path = tmp_path / "fluxes.csv"
    pd.DataFrame(
        {
            "fraction": ["pp", "8B", "hep"],
            "flux": [6.0e10, 4.5e6, 8.0e3],
        }
    ).to_csv(path, index=False)

    fluxes = load_solar_fluxes(path, device=DEVICE, dtype=DTYPE)

    assert sorted(fluxes) == ["8B", "hep", "pp"]
    for value in fluxes.values():
        assert value.shape == ()
        assert_same_device(value.device, DEVICE)
        assert value.dtype == DTYPE
    assert_close(fluxes["8B"], torch.tensor(4.5e6, device=DEVICE, dtype=DTYPE), name="8B flux")


@pytest.mark.parametrize(
    ("table", "message"),
    [
        ({"source": ["pp"], "flux": [1.0]}, "missing required columns"),
        ({"fraction": ["pp", "pp"], "flux": [1.0, 2.0]}, "duplicate sources"),
        ({"fraction": ["pp"], "flux": [-1.0]}, "non-negative"),
        ({"fraction": ["pp"], "flux": [float("inf")]}, "non-finite"),
    ],
)
def test_load_solar_fluxes_rejects_invalid_tables(tmp_path, table, message):
    path = tmp_path / "flux.csv"
    pd.DataFrame(table).to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        load_solar_fluxes(path, device=DEVICE, dtype=DTYPE)


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


def test_load_legacy_solar_spectrum_by_source_and_variant():
    spectrum = load_solar_spectrum(
        "8B", provider="legacy", variant="ortiz", device=DEVICE, dtype=DTYPE
    )
    assert spectrum["energy"].ndim == 1
    assert spectrum["spectrum"].shape == spectrum["energy"].shape
    assert bool(torch.all(spectrum["spectrum"] >= 0))


def test_load_bahcall_solar_spectrum_for_every_registered_source():
    # Regression test: the bundled bahcall/spectrum/*.csv headers must match
    # load_solar_spectrum's expected "energy_MeV"/"spectrum" columns (the
    # bahcall provider previously shipped "Energy"/"Spectrum" headers, which
    # raised KeyError for every source under provider="bahcall").
    for source in available_solar_spectrum_sources("bahcall"):
        spectrum = load_solar_spectrum(
            source, provider="bahcall", device=DEVICE, dtype=DTYPE
        )
        assert spectrum["energy"].ndim == 1
        assert spectrum["spectrum"].shape == spectrum["energy"].shape
        assert bool(torch.all(spectrum["spectrum"] >= 0))


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


def test_default_bundled_fluxes_are_positive_and_include_main_sources():
    fluxes = load_solar_fluxes(device=DEVICE, dtype=DTYPE)

    assert {"pp", "8B", "7Be", "hep"}.issubset(fluxes)
    for value in fluxes.values():
        assert value.shape == ()
        assert torch.isfinite(value)
        assert bool(value > 0.0)


def test_legacy_provider_loads_all_runtime_products():
    density = load_solar_density(provider="legacy", device=DEVICE, dtype=DTYPE)
    production = load_solar_production(provider="legacy", device=DEVICE, dtype=DTYPE)
    fluxes = load_solar_fluxes(provider="legacy", device=DEVICE, dtype=DTYPE)

    assert density["radius"].shape == density["electron_density"].shape
    assert {"pp", "7Be", "8B", "hep"}.issubset(production["fractions"])
    assert {"pp", "7Be", "8B", "hep"}.issubset(fluxes)


def test_default_production_measure_matches_configured_default_provider():
    # No path/provider override: resolves tpeanuts.config.default.solar_provider
    # for the measure lookup exactly like passing that provider explicitly
    # -- kept provider-agnostic so this test does not go stale if the
    # configured default provider changes.
    default_result = load_solar_production(device=DEVICE, dtype=DTYPE)
    explicit_result = load_solar_production(
        provider=default.solar_provider, device=DEVICE, dtype=DTYPE,
    )
    assert default_result["production_measure"] == explicit_result["production_measure"]


def test_zenodo_and_legacy_providers_resolve_to_radial_pdf():
    zenodo = load_solar_production(provider="zenodo", device=DEVICE, dtype=DTYPE)
    legacy = load_solar_production(provider="legacy", device=DEVICE, dtype=DTYPE)

    assert zenodo["production_measure"] == "radial_pdf"
    assert legacy["production_measure"] == "radial_pdf"


def test_bahcall_production_fractions_sum_to_one_per_shell():
    # Direct evidence for the "shell_fraction" classification: Bahcall's
    # bp2004_production.csv tabulates, for each (non-uniform) radial shell,
    # the share of that source's total neutrino production occurring there
    # -- a plain sum over shells recovers ~1, while trapezoidal integration
    # over the non-uniform radius grid does not (it would instead pick up an
    # arbitrary factor tied to the local grid spacing).
    production = load_solar_production(provider="bahcall", device=DEVICE, dtype=DTYPE)
    for source, fraction in production["fractions"].items():
        total = float(fraction.sum())
        assert total == pytest.approx(1.0, abs=1e-3), (
            f"{source} shell fractions should sum to ~1, got {total}"
        )
        trapz_total = float(torch.trapz(fraction, x=production["radius"]))
        assert trapz_total == pytest.approx(0.0, abs=1e-2), (
            f"{source} trapz-integrated over its own non-uniform shell grid "
            f"should be far from 1 (confirming it is not a radial_pdf), got "
            f"{trapz_total}"
        )


def test_explicit_production_measure_overrides_provider_default():
    production = load_solar_production(
        provider="bahcall", production_measure="radial_pdf", device=DEVICE, dtype=DTYPE,
    )
    assert production["production_measure"] == "radial_pdf"


def test_production_measure_rejects_unknown_value():
    with pytest.raises(ValueError, match="production_measure"):
        load_solar_production(
            provider="bahcall", production_measure="bogus", device=DEVICE, dtype=DTYPE,
        )


def test_explicit_path_without_provider_defaults_to_radial_pdf(tmp_path):
    # An arbitrary custom production table with no provider/measure hint
    # keeps the historical trapz-integration behaviour rather than silently
    # guessing "shell_fraction".
    production_path = tmp_path / "production.csv"
    pd.DataFrame(
        {
            "radius": [0.0, 0.5, 1.0],
            "pp fraction": [0.0, 1.0, 0.0],
        }
    ).to_csv(production_path, index=False)

    production = load_solar_production(production_path, device=DEVICE, dtype=DTYPE)
    assert production["production_measure"] == "radial_pdf"

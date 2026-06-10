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
Spyder-friendly tests and visual diagnostics for tpeanuts.io.io_earth.

This script checks:

    1. density CSV parsing and delta-column detection.
    2. Loading the real earth density CSV file.
    3. Tabulated-density loading mode.
    4. Attaching the CSV loader to earthdensity.
    5. Nadir-exposure CSV loading and angle-mode conversion.
    6. Nadir-exposure cache save/load roundtrip.
    7. Expected IO validation errors.
    8. Visual diagnostics for loaded density and exposure data.

Run directly in Spyder or from a terminal:

    python tpeanuts/tests/earth/test5_io.py
"""



from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch


# ============================================================
# Import bootstrap
# ============================================================

THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parents[3]



from tpeanuts.earth.density import EarthDensity  # noqa: E402
from tpeanuts.io.io_earth import (  # noqa: E402
    _cache_filename,
    _convert_csv_angle_mode,
    _read_csv_exposure_column,
    attach_csv_loader_to_density_class,
    extract_delta_columns,
    load_earth_density_from_csv,
    nadir_exposure_from_cache,
    nadir_exposure_from_csv,
    parse_density_table,
    save_nadir_exposure_to_cache,
)
from tpeanuts.earth.exposure_math import make_eta_grid  # noqa: E402
from tpeanuts.util.test_utils import (  # noqa: E402
    assert_raises,
    assert_true,
    run_test_suite,
)


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device("cpu")
DTYPE = torch.float64

REAL_DENSITY_FILE = PACKAGE_DIR / "data" / "density" / "earth_density.csv"
TESTS_DIR = THIS_FILE.parents[1]

DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_TEST_ROOT = Path(OUTPUT_ROOT / "test")
OUTPUT_DIR = Path(OUTPUT_TEST_ROOT / "earth" / Path(__file__).stem)
TEMP_DIR = OUTPUT_DIR
CACHE_DIR = OUTPUT_DIR / "cache"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

torch.set_printoptions(
    precision=10,
    sci_mode=True,
    linewidth=160,
)


# ============================================================
# Local helpers
# ============================================================

def build_density_dataframe():
    return pd.DataFrame(
        {
            "rj": [0.25, 0.50, 0.75, 1.00],
            "alpha": [10.0, 8.0, 5.0, 2.0],
            "beta": [1.0, 0.5, 0.2, 0.1],
            "gamma": [0.10, 0.05, 0.02, 0.01],
            "delta1": [0.010, 0.020, 0.030, 0.040],
            "delta2": [0.001, 0.002, 0.003, 0.004],
        }
    )


def write_density_csv(path: Path):
    table = build_density_dataframe()
    table.to_csv(path, index=False)
    print(f"Created temporary density CSV file: {path}")
    return table


def write_exposure_csv(path: Path, values):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Exposure"])
        for value in values:
            writer.writerow([float(value)])

    print(f"Created temporary exposure CSV file: {path}")


def assert_tensor_close(actual, expected, message, atol=1.0e-12, rtol=1.0e-10):
    actual_t = torch.as_tensor(actual)
    expected_t = torch.as_tensor(expected, dtype=actual_t.dtype, device=actual_t.device)

    max_diff = torch.max(torch.abs(actual_t - expected_t)).item()

    print(f"Checking: {message}")
    print("  actual shape  :", tuple(actual_t.shape))
    print("  expected shape:", tuple(expected_t.shape))
    print("  max abs diff  :", f"{max_diff:.6e}")

    assert_true(
        torch.allclose(actual_t, expected_t, atol=atol, rtol=rtol),
        message,
    )


# ============================================================
# tests
# ============================================================

def test_extract_delta_columns():
    table = build_density_dataframe()
    delta_names = extract_delta_columns(table)

    print("\nDelta-column extraction:")
    print("columns      :", list(table.columns))
    print("delta columns:", delta_names)

    assert_true(delta_names == ["delta1", "delta2"], "Delta columns must be detected in table order")


def test_parse_density_table_polynomial_mode():
    table = build_density_dataframe()

    density = parse_density_table(
        table,
        tabulated_density=False,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("\nParsed polynomial density table:")
    print("type         :", type(density))
    print("rj           :", density.rj)
    print("alpha        :", density.alpha)
    print("beta         :", density.beta)
    print("gamma        :", density.gamma)
    print("deltas shape :", tuple(density.deltas.shape))
    print("tabulated    :", density.tabulated)

    assert_true(isinstance(density, EarthDensity), "parse_density_table must return earthdensity")
    assert_true(density.rj.shape == (4,), "rj shape is correct")
    assert_true(density.deltas.shape == (2, 4), "Two delta columns must produce shape (2, Ns)")
    assert_true(density.device == DEVICE, "density tensors must be on the requested device")
    assert_true(density.dtype == DTYPE, "density tensors must use the requested dtype")
    assert_true(density.tabulated is False, "Polynomial mode must set tabulated=False")
    assert_tensor_close(density.deltas[0], torch.tensor(table["delta1"].to_numpy(), dtype=DTYPE), "delta1 values are preserved")


def test_parse_density_table_tabulated_mode_zeroes_higher_terms():
    table = build_density_dataframe()

    density = parse_density_table(
        table,
        tabulated_density=True,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("\nParsed tabulated density table:")
    print("beta        :", density.beta)
    print("gamma       :", density.gamma)
    print("deltas shape:", tuple(density.deltas.shape))
    print("tabulated   :", density.tabulated)

    assert_true(density.tabulated is True, "Tabulated mode must set tabulated=True")
    assert_tensor_close(density.beta, torch.zeros_like(density.beta), "Tabulated beta values are zero")
    assert_tensor_close(density.gamma, torch.zeros_like(density.gamma), "Tabulated gamma values are zero")
    assert_true(density.deltas.shape == (0, density.rj.numel()), "Tabulated deltas must be empty")


def test_load_earth_density_from_csv_synthetic_and_real_files():
    synthetic_path = TEMP_DIR / "synthetic_earth_density.csv"
    table = write_density_csv(synthetic_path)

    density_synthetic = load_earth_density_from_csv(
        str(synthetic_path),
        tabulated_density=False,
        device=DEVICE,
        dtype=DTYPE,
    )

    density_real = load_earth_density_from_csv(
        str(REAL_DENSITY_FILE),
        tabulated_density=False,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("\nLoaded density CSV files:")
    print("synthetic rj shape:", tuple(density_synthetic.rj.shape))
    print("real rj shape     :", tuple(density_real.rj.shape))
    print("real deltas shape :", tuple(density_real.deltas.shape))
    print("real rj min/max   :", density_real.rj.min().item(), density_real.rj.max().item())

    assert_tensor_close(density_synthetic.rj, torch.tensor(table["rj"].to_numpy(), dtype=DTYPE), "Synthetic rj values are loaded")
    assert_true(density_real.rj.ndim == 1, "Real density rj must be one-dimensional")
    assert_true(density_real.rj.numel() > 0, "Real density file must contain shells")
    assert_true(torch.isfinite(density_real.alpha).all().item(), "Real alpha values must be finite")
    assert_true(torch.all(density_real.rj[1:] >= density_real.rj[:-1]).item(), "Real shell radii must be sorted increasingly")


def test_attach_csv_loader_to_density_class():
    synthetic_path = TEMP_DIR / "synthetic_earth_density_loader.csv"
    write_density_csv(synthetic_path)

    attach_csv_loader_to_density_class()
    density = EarthDensity.from_csv(
        str(synthetic_path),
        tabulated_density=False,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("\nAttached CSV loader:")
    print("EarthDensity.from_csv:", EarthDensity.from_csv)
    print("loaded rj shape      :", tuple(density.rj.shape))

    assert_true(hasattr(EarthDensity, "from_csv"), "EarthDensity must receive from_csv static method")
    assert_true(isinstance(density, EarthDensity), "EarthDensity.from_csv must return earthdensity")


def test_read_and_convert_exposure_csv_modes():
    csv_path = TEMP_DIR / "synthetic_exposure_modes.csv"
    raw_values = torch.linspace(1.0, 2.0, 9, dtype=DTYPE)
    write_exposure_csv(csv_path, raw_values)

    raw = _read_csv_exposure_column(str(csv_path))
    eta = make_eta_grid(raw.numel(), device=DEVICE, dtype=DTYPE)

    nadir = _convert_csv_angle_mode(raw, eta, angle="Nadir", dtype=DTYPE)
    zenith = _convert_csv_angle_mode(raw, eta, angle="Zenith", dtype=DTYPE)
    coszenith = _convert_csv_angle_mode(raw, eta, angle="CosZenith", dtype=DTYPE)

    print("\nExposure CSV angle conversion:")
    print("raw      :", raw)
    print("nadir    :", nadir)
    print("zenith   :", zenith)
    print("coszenith:", coszenith)

    assert_tensor_close(raw, raw_values, "Raw exposure column is read correctly")
    assert_tensor_close(nadir, raw_values, "Nadir mode preserves raw exposure")
    assert_tensor_close(zenith, torch.flip(raw_values, dims=(0,)), "Zenith mode reverses raw exposure")
    assert_true(coszenith.shape == raw_values.shape, "CosZenith conversion preserves shape")
    assert_true(torch.isfinite(coszenith).all().item(), "CosZenith conversion must be finite")
    assert_true((coszenith >= 0.0).all().item(), "CosZenith conversion must be non-negative after clamping")


def test_nadir_exposure_from_csv_daynight_slices():
    csv_path = TEMP_DIR / "synthetic_exposure_daynight.csv"
    raw_values = torch.arange(1, 10, dtype=DTYPE)
    write_exposure_csv(csv_path, raw_values)

    eta_full, exposure_full = nadir_exposure_from_csv(
        str(csv_path),
        angle="Nadir",
        daynight=None,
        device=DEVICE,
        dtype=DTYPE,
    )

    eta_night, exposure_night = nadir_exposure_from_csv(
        str(csv_path),
        angle="Nadir",
        daynight="night",
        device=DEVICE,
        dtype=DTYPE,
    )

    eta_day, exposure_day = nadir_exposure_from_csv(
        str(csv_path),
        angle="Nadir",
        daynight="day",
        device=DEVICE,
        dtype=DTYPE,
    )

    print("\nNadir exposure CSV day/night slices:")
    print("full eta/exposure shapes :", tuple(eta_full.shape), tuple(exposure_full.shape))
    print("night eta/exposure shapes:", tuple(eta_night.shape), tuple(exposure_night.shape))
    print("day eta/exposure shapes  :", tuple(eta_day.shape), tuple(exposure_day.shape))
    print("night exposure:", exposure_night)
    print("day exposure  :", exposure_day)

    assert_true(eta_full.shape == (9,), "Full CSV exposure must keep all points")
    assert_true(eta_night.shape == (4,), "Night CSV slice must keep lower half for ns=9")
    assert_true(eta_day.shape == (4,), "Day CSV slice must keep upper half for ns=9")
    assert_tensor_close(exposure_night, raw_values[:4], "Night exposure slice is the lower half")
    assert_tensor_close(exposure_day, raw_values[5:], "Day exposure slice is the upper half")


def test_nadir_exposure_cache_roundtrip():
    eta = torch.linspace(0.0, torch.pi, 11, device=DEVICE, dtype=DTYPE)
    exposure = 0.5 + torch.sin(eta) ** 2

    path = save_nadir_exposure_to_cache(
        eta,
        exposure,
        lam_rad=0.72,
        d1=0.0,
        d2=365.0,
        ns=11,
        daynight=None,
        cache_dir=str(CACHE_DIR),
    )

    expected_path = _cache_filename(
        str(CACHE_DIR),
        0.72,
        0.0,
        365.0,
        11,
        None,
    )

    eta_loaded, exposure_loaded = nadir_exposure_from_cache(
        lam_rad=0.72,
        d1=0.0,
        d2=365.0,
        ns=11,
        daynight=None,
        cache_dir=str(CACHE_DIR),
        device=DEVICE,
        dtype=DTYPE,
    )

    print("\nNadir exposure cache roundtrip:")
    print("saved path   :", path)
    print("expected path:", expected_path)
    print("eta loaded   :", eta_loaded)
    print("W loaded     :", exposure_loaded)

    assert_true(path == expected_path, "Cache save path must match deterministic cache filename")
    assert_true(os.path.isfile(path), "Cache file must exist after saving")
    assert_tensor_close(eta_loaded, eta, "Cached eta values roundtrip")
    assert_tensor_close(exposure_loaded, exposure, "Cached exposure values roundtrip")


def test_invalid_io_inputs_raise_errors():
    missing_path = TEMP_DIR / "does_not_exist.csv"
    bad_exposure_path = TEMP_DIR / "bad_exposure.csv"

    pd.DataFrame({"NotExposure": [1.0, 2.0]}).to_csv(bad_exposure_path, index=False)

    print("\nInvalid IO input checks:")
    print("Missing density file should raise FileNotFoundError")
    assert_raises(
        FileNotFoundError,
        load_earth_density_from_csv,
        str(missing_path),
        device=DEVICE,
        dtype=DTYPE,
    )

    print("Exposure CSV without 'Exposure' column should raise ValueError")
    assert_raises(
        ValueError,
        _read_csv_exposure_column,
        str(bad_exposure_path),
    )

    print("Unknown CSV angle mode should raise ValueError")
    assert_raises(
        ValueError,
        _convert_csv_angle_mode,
        torch.ones(5, dtype=DTYPE),
        torch.linspace(0.0, torch.pi, 5, dtype=DTYPE),
        angle="BadMode",
        dtype=DTYPE,
    )

    print("Missing cache file should raise FileNotFoundError")
    assert_raises(
        FileNotFoundError,
        nadir_exposure_from_cache,
        lam_rad=9.99,
        d1=0.0,
        d2=1.0,
        ns=3,
        cache_dir=str(CACHE_DIR),
        device=DEVICE,
        dtype=DTYPE,
    )


# ============================================================
# Visualization
# ============================================================

def plot_loaded_density_coefficients(savefig=False):
    density = load_earth_density_from_csv(
        str(REAL_DENSITY_FILE),
        tabulated_density=False,
        device=DEVICE,
        dtype=DTYPE,
    )

    r = density.rj.detach().cpu()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(r, density.alpha.detach().cpu(), marker="o", label="alpha")
    ax.plot(r, density.beta.detach().cpu(), marker="s", label="beta")
    ax.plot(r, density.gamma.detach().cpu(), marker="^", label="gamma")
    ax.set_xlabel(r"Shell radius $r_j/R_E$")
    ax.set_ylabel("density polynomial coefficient")
    ax.set_title("earth density coefficients loaded from CSV")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = OUTPUT_DIR / "earth_io_density_coefficients.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"\nSaved plot: {path}")
    plt.show()


def plot_exposure_csv_modes(savefig=False):
    csv_path = TEMP_DIR / "synthetic_exposure_plot_modes.csv"
    raw_values = torch.linspace(0.5, 1.5, 31, dtype=DTYPE) ** 2
    write_exposure_csv(csv_path, raw_values)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for mode in ["Nadir", "Zenith", "CosZenith"]:
        eta, exposure = nadir_exposure_from_csv(
            str(csv_path),
            angle=mode,
            daynight=None,
            device=DEVICE,
            dtype=DTYPE,
        )
        ax.plot(eta.detach().cpu(), exposure.detach().cpu(), label=mode)

    ax.set_xlabel(r"Nadir angle $\eta$ [rad]")
    ax.set_ylabel(r"Exposure weight $W(\eta)$")
    ax.set_title("Exposure CSV angle-mode conversion")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = OUTPUT_DIR / "earth_io_exposure_csv_modes.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"Saved plot: {path}")
    plt.show()


def plot_cache_roundtrip_exposure(savefig=False):
    eta = torch.linspace(0.0, torch.pi, 41, device=DEVICE, dtype=DTYPE)
    exposure = 0.5 + torch.sin(eta) ** 2

    save_nadir_exposure_to_cache(
        eta,
        exposure,
        lam_rad=0.33,
        d1=10.0,
        d2=50.0,
        ns=41,
        cache_dir=str(CACHE_DIR),
    )

    eta_loaded, exposure_loaded = nadir_exposure_from_cache(
        lam_rad=0.33,
        d1=10.0,
        d2=50.0,
        ns=41,
        cache_dir=str(CACHE_DIR),
        device=DEVICE,
        dtype=DTYPE,
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(eta.detach().cpu(), exposure.detach().cpu(), lw=2.0, label="Original")
    ax.scatter(eta_loaded.detach().cpu(), exposure_loaded.detach().cpu(), s=18, label="Loaded from cache")
    ax.set_xlabel(r"Nadir angle $\eta$ [rad]")
    ax.set_ylabel(r"Exposure weight $W(\eta)$")
    ax.set_title("Nadir-exposure cache roundtrip")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = OUTPUT_DIR / "earth_io_cache_roundtrip.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"Saved plot: {path}")
    plt.show()


def test_visualization_outputs(savefig=False):
    plot_loaded_density_coefficients(savefig=savefig)
    plot_exposure_csv_modes(savefig=savefig)
    plot_cache_roundtrip_exposure(savefig=savefig)

    expected_files = [
        OUTPUT_DIR / "earth_io_density_coefficients.png",
        OUTPUT_DIR / "earth_io_exposure_csv_modes.png",
        OUTPUT_DIR / "earth_io_cache_roundtrip.png",
    ]

    for path in expected_files:
        print(f"Checking plot file: {path}")
        if savefig:
            assert_true(path.is_file(), f"Plot was not created: {path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    tests = [
        test_extract_delta_columns,
        test_parse_density_table_polynomial_mode,
        test_parse_density_table_tabulated_mode_zeroes_higher_terms,
        test_load_earth_density_from_csv_synthetic_and_real_files,
        test_attach_csv_loader_to_density_class,
        test_read_and_convert_exposure_csv_modes,
        test_nadir_exposure_from_csv_daynight_slices,
        test_nadir_exposure_cache_roundtrip,
        test_invalid_io_inputs_raise_errors,
        test_visualization_outputs,
    ]

    run_test_suite(
        tests,
        suite_name="earth IO tests",
        verbose_traceback=True,
    )

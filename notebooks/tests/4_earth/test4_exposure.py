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
Spyder-friendly tests and visual diagnostics for tpeanuts.earth.exposure.

This script checks:

    1. Nadir-angle grid construction and day/night slicing.
    2. Low-level analytical exposure integrals.
    3. NadirExposureTable normalization and interpolation.
    4. build_nadir_exposure(source="math") without cache.
    5. build_nadir_exposure(source="csv") for simple input files.
    6. Input validation errors.
    7. Visual diagnostics for exposure profiles and interpolation.

Run directly in Spyder or from a terminal:

    python tpeanuts/tests/earth/test4_exposure.py
"""



from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch


# ============================================================
# Import bootstrap
# ============================================================

THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parents[3]



from tpeanuts.earth.exposure import (  # noqa: E402
    NadirExposureTable,
    build_nadir_exposure,
    nadir_exposure_from_math,
)
from tpeanuts.earth.exposure_math import (  # noqa: E402
    IntegralAngle,
    IntegralDay,
    IndefiniteIntegralDay,
    make_eta_grid,
)
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

TESTS_DIR = THIS_FILE.parents[1]

DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_TEST_ROOT = Path(OUTPUT_ROOT / "test")
OUTPUT_DIR = Path(OUTPUT_TEST_ROOT / "earth" / Path(__file__).stem)
os.makedirs(OUTPUT_DIR, exist_ok=True)

LATITUDE_RAD = 0.72

torch.set_printoptions(
    precision=10,
    sci_mode=True,
    linewidth=160,
)


# ============================================================
# Local helpers
# ============================================================

def assert_tensor_close(actual, expected, message, atol=1.0e-10, rtol=1.0e-8):
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


def trapz_norm(y, x):
    return torch.trapz(y, x=x).item()


def write_exposure_csv(path: Path, values):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Exposure"])
        for value in values:
            writer.writerow([float(value)])

    print(f"Created temporary CSV file: {path}")


def build_synthetic_table():
    eta = torch.linspace(0.0, torch.pi, 21, device=DEVICE, dtype=DTYPE)
    exposure = 0.25 + torch.sin(eta) ** 2
    return NadirExposureTable(eta=eta, exposure=exposure)


# ============================================================
# tests
# ============================================================

def test_make_eta_grid_and_daynight_slices():
    eta_full = make_eta_grid(9, daynight=None, device=DEVICE, dtype=DTYPE)
    eta_day = make_eta_grid(9, daynight="day", device=DEVICE, dtype=DTYPE)
    eta_night = make_eta_grid(9, daynight="night", device=DEVICE, dtype=DTYPE)

    print("\nEta grids:")
    print("full :", eta_full)
    print("day  :", eta_day)
    print("night:", eta_night)

    assert_true(eta_full.shape == (9,), "Full eta grid must contain ns points")
    assert_true(eta_day.shape == (4,), "Day slice for ns=9 must contain ceil/floor upper half")
    assert_true(eta_night.shape == (4,), "Night slice for ns=9 must contain lower half")
    assert_tensor_close(eta_full[0], torch.tensor(0.0, dtype=DTYPE), "Eta grid starts at zero")
    assert_tensor_close(eta_full[-1], torch.tensor(torch.pi, dtype=DTYPE), "Eta grid ends at pi")
    assert_true(torch.all(eta_full[1:] >= eta_full[:-1]).item(), "Eta grid must be sorted increasingly")


def test_indefinite_integral_day_is_finite():
    T = torch.tensor(0.15, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(1.00, device=DEVICE, dtype=DTYPE)
    lam = torch.tensor(LATITUDE_RAD, device=DEVICE, dtype=DTYPE)

    value = IndefiniteIntegralDay(
        T,
        eta,
        lam,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("\nIndefiniteIntegralDay:")
    print("T     :", T.item())
    print("eta   :", eta.item())
    print("lambda:", lam.item())
    print("value :", value)

    assert_true(torch.is_complex(value), "IndefiniteIntegralDay must return a complex tensor")
    assert_true(torch.isfinite(value.real).all().item(), "Primitive real part must be finite")
    assert_true(torch.isfinite(value.imag).all().item(), "Primitive imaginary part must be finite")


def test_integral_angle_and_day_are_finite():
    eta = torch.tensor(1.10, device=DEVICE, dtype=DTYPE)
    lam = torch.tensor(LATITUDE_RAD, device=DEVICE, dtype=DTYPE)

    angle_weight = IntegralAngle(
        eta,
        lam,
        a1=0.20,
        a2=2.80,
        device=DEVICE,
        dtype=DTYPE,
    )

    day_weight = IntegralDay(
        eta,
        lam,
        d1=20.0,
        d2=260.0,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("\nAnalytical exposure weights:")
    print("IntegralAngle:", angle_weight)
    print("IntegralDay  :", day_weight)

    assert_true(angle_weight.shape == (), "IntegralAngle must return a scalar for scalar eta")
    assert_true(day_weight.shape == (), "IntegralDay must return a scalar for scalar eta")
    assert_true(torch.isfinite(angle_weight).all().item(), "IntegralAngle output must be finite")
    assert_true(torch.isfinite(day_weight).all().item(), "IntegralDay output must be finite")


def test_nadir_exposure_table_normalization_and_interp():
    table = build_synthetic_table()
    raw_norm = trapz_norm(table.exposure, table.eta)

    table.normalize_()
    normalized_norm = trapz_norm(table.exposure, table.eta)

    query = torch.tensor([0.0, 0.50, 1.00, 2.00, torch.pi], device=DEVICE, dtype=DTYPE)
    interpolated = table.interp(query)

    print("\nNadirExposureTable normalization and interpolation:")
    print("raw integral       :", f"{raw_norm:.10f}")
    print("normalized integral:", f"{normalized_norm:.10f}")
    print("query eta          :", query)
    print("interpolated W     :", interpolated)

    assert_true(table.device == DEVICE, "Table device property is correct")
    assert_true(table.dtype == DTYPE, "Table dtype property is correct")
    assert_true(abs(normalized_norm - 1.0) < 1.0e-10, "Normalized exposure must integrate to one")
    assert_true(interpolated.shape == query.shape, "Interpolated exposure must match query shape")
    assert_true(torch.isfinite(interpolated).all().item(), "Interpolated exposure must be finite")
    assert_true((interpolated >= 0.0).all().item(), "Synthetic normalized exposure must be non-negative")


def test_nadir_exposure_from_math_shapes_and_finiteness():
    eta, exposure = nadir_exposure_from_math(
        LATITUDE_RAD,
        ns=7,
        d1=0.0,
        d2=365.0,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("\nnadir_exposure_from_math:")
    print("eta shape     :", tuple(eta.shape))
    print("exposure shape:", tuple(exposure.shape))
    print("eta           :", eta)
    print("exposure      :", exposure)

    assert_true(eta.shape == (7,), "Math eta table must have ns points")
    assert_true(exposure.shape == eta.shape, "Math exposure shape must match eta shape")
    assert_true(torch.isfinite(exposure).all().item(), "Math exposure must be finite")


def test_build_nadir_exposure_math_without_cache():
    table = build_nadir_exposure(
        source="math",
        lam_rad=LATITUDE_RAD,
        ns=7,
        d1=0.0,
        d2=365.0,
        normalized=False,
        use_cache=False,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("\nbuild_nadir_exposure(source='math', use_cache=False):")
    print("table type:", type(table))
    print("eta shape :", tuple(table.eta.shape))
    print("W shape   :", tuple(table.exposure.shape))
    print("W min/max :", table.exposure.min().item(), table.exposure.max().item())

    assert_true(isinstance(table, NadirExposureTable), "Math builder must return NadirExposureTable")
    assert_true(table.eta.shape == (7,), "Math builder eta shape is correct")
    assert_true(table.exposure.shape == table.eta.shape, "Math builder exposure shape is correct")
    assert_true(torch.isfinite(table.exposure).all().item(), "Math builder exposure must be finite")


def test_build_nadir_exposure_csv_modes_and_daynight():
    csv_path = OUTPUT_DIR / "synthetic_exposure.csv"
    values = torch.linspace(1.0, 2.0, 9, dtype=DTYPE)
    write_exposure_csv(csv_path, values)

    table_nadir = build_nadir_exposure(
        source="csv",
        csv_path=str(csv_path),
        angle="Nadir",
        daynight=None,
        normalized=True,
        use_cache=False,
        device=DEVICE,
        dtype=DTYPE,
    )

    table_zenith = build_nadir_exposure(
        source="csv",
        csv_path=str(csv_path),
        angle="Zenith",
        daynight=None,
        normalized=False,
        use_cache=False,
        device=DEVICE,
        dtype=DTYPE,
    )

    table_night = build_nadir_exposure(
        source="csv",
        csv_path=str(csv_path),
        angle="Nadir",
        daynight="night",
        normalized=False,
        use_cache=False,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("\nCSV exposure tables:")
    print("nadir eta shape :", tuple(table_nadir.eta.shape))
    print("zenith exposure :", table_zenith.exposure)
    print("night eta       :", table_night.eta)
    print("normalized area :", trapz_norm(table_nadir.exposure, table_nadir.eta))

    assert_true(table_nadir.eta.shape == (9,), "CSV Nadir table must keep all points")
    assert_true(table_zenith.eta.shape == (9,), "CSV Zenith table must keep all points")
    assert_true(table_night.eta.shape == (4,), "CSV night table must keep the lower half")
    assert_tensor_close(table_zenith.exposure, torch.flip(values, dims=(0,)), "Zenith mode reverses exposure values")
    assert_true(abs(trapz_norm(table_nadir.exposure, table_nadir.eta) - 1.0) < 1.0e-10, "CSV normalized exposure integrates to one")


def test_invalid_inputs_raise_errors():
    print("\nInvalid input checks:")
    print("IntegralDay with d1 > d2 should raise ValueError")
    assert_raises(
        ValueError,
        IntegralDay,
        torch.tensor(1.0, device=DEVICE, dtype=DTYPE),
        torch.tensor(LATITUDE_RAD, device=DEVICE, dtype=DTYPE),
        300.0,
        20.0,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("build_nadir_exposure(source='math') without lam_rad should raise ValueError")
    assert_raises(
        ValueError,
        build_nadir_exposure,
        source="math",
        use_cache=False,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("build_nadir_exposure(source='csv') without csv_path should raise ValueError")
    assert_raises(
        ValueError,
        build_nadir_exposure,
        source="csv",
        use_cache=False,
        device=DEVICE,
        dtype=DTYPE,
    )


# ============================================================
# Visualization
# ============================================================

def plot_math_exposure_profiles(savefig=False):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    eta, exposure = nadir_exposure_from_math(
        LATITUDE_RAD,
        ns=7,
        d1=0.0,
        d2=365.0,
        device=DEVICE,
        dtype=DTYPE,
    )

    ax.plot(
        eta.detach().cpu(),
        exposure.detach().cpu(),
        marker="o",
        label=f"Latitude {LATITUDE_RAD:.2f} rad",
    )

    ax.axvline(float(torch.pi / 2.0), color="black", ls="--", lw=1.0, label=r"$\pi/2$")
    ax.set_xlabel(r"Nadir angle $\eta$ [rad]")
    ax.set_ylabel(r"Raw exposure weight $W(\eta)$")
    ax.set_title("Analytical nadir exposure profiles")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = OUTPUT_DIR / "earth_exposure_math_profiles.png"
    if savefig:
        fig.savefig(path, dpi=200)
    plt.show()

    if savefig:
        print(f"\nSaved plot: {path}")


def plot_table_interpolation(savefig=False):
    table = build_synthetic_table()
    table.normalize_()

    eta_query = torch.linspace(0.0, torch.pi, 200, device=DEVICE, dtype=DTYPE)
    exposure_query = table.interp(eta_query)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(eta_query.detach().cpu(), exposure_query.detach().cpu(), label="Linear interpolation")
    ax.scatter(table.eta.detach().cpu(), table.exposure.detach().cpu(), s=20, color="tab:red", label="Table nodes")
    ax.set_xlabel(r"Nadir angle $\eta$ [rad]")
    ax.set_ylabel(r"Normalized exposure $W(\eta)$")
    ax.set_title("NadirExposureTable interpolation diagnostic")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = OUTPUT_DIR / "earth_exposure_table_interpolation.png"
    if savefig:
        fig.savefig(path, dpi=200)
    plt.show()

    if savefig:
        print(f"Saved plot: {path}")


def plot_csv_angle_modes(savefig=False):
    csv_path = OUTPUT_DIR / "synthetic_exposure_plot.csv"
    values = torch.linspace(0.5, 1.5, 31, dtype=DTYPE) ** 2
    write_exposure_csv(csv_path, values)

    modes = ["Nadir", "Zenith", "CosZenith"]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for mode in modes:
        table = build_nadir_exposure(
            source="csv",
            csv_path=str(csv_path),
            angle=mode,
            normalized=True,
            use_cache=False,
            device=DEVICE,
            dtype=DTYPE,
        )

        ax.plot(
            table.eta.detach().cpu(),
            table.exposure.detach().cpu(),
            label=mode,
        )

    ax.set_xlabel(r"Nadir angle $\eta$ [rad]")
    ax.set_ylabel(r"Normalized exposure $W(\eta)$")
    ax.set_title("CSV exposure angle-mode conversion")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = OUTPUT_DIR / "earth_exposure_csv_angle_modes.png"
    if savefig:
        fig.savefig(path, dpi=200)
    plt.show()

    if savefig:
        print(f"Saved plot: {path}")


def test_visualization_outputs(savefig=False):
    plot_math_exposure_profiles(savefig=savefig)
    plot_table_interpolation(savefig=savefig)
    plot_csv_angle_modes(savefig=savefig)

    expected_files = [
        OUTPUT_DIR / "earth_exposure_math_profiles.png",
        OUTPUT_DIR / "earth_exposure_table_interpolation.png",
        OUTPUT_DIR / "earth_exposure_csv_angle_modes.png",
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
        test_make_eta_grid_and_daynight_slices,
        test_indefinite_integral_day_is_finite,
        test_integral_angle_and_day_are_finite,
        test_nadir_exposure_table_normalization_and_interp,
        test_nadir_exposure_from_math_shapes_and_finiteness,
        test_build_nadir_exposure_math_without_cache,
        test_build_nadir_exposure_csv_modes_and_daynight,
        test_invalid_inputs_raise_errors,
        test_visualization_outputs,
    ]

    run_test_suite(
        tests,
        suite_name="earth EXPOSURE tests",
        verbose_traceback=True,
    )

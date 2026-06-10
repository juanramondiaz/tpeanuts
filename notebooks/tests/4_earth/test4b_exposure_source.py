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
Spyder-friendly comparison of tpeanuts.earth.exposure sources.

This script compares the exposure sources supported by build_nadir_exposure:

    1. source="math"
    2. source="cache"
    3. source="csv"
    4. source="legacy" using tpeanuts/peanuts.

The comparison is numerical and graphical. The cache and CSV sources are seeded
from the math table so that IO roundtrips can be checked exactly. The legacy
source is loaded from the original NumPy peanuts package in tpeanuts/peanuts.

Run directly in Spyder or from a terminal:

    python tpeanuts/tests/earth/test4b_exposure_source.py
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
)
from tpeanuts.io.io_earth import save_nadir_exposure_to_cache  # noqa: E402
from tpeanuts.util.test_utils import (  # noqa: E402
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
CACHE_DIR = OUTPUT_DIR / "exposure_source_cache"
LEGACY_DIR = PACKAGE_DIR / "peanuts"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

LATITUDE_RAD = 0.72
D1 = 0.0
D2 = 365.0
NS = 9

SOURCE_ORDER = ["math", "cache", "csv", "legacy"]

torch.set_printoptions(
    precision=10,
    sci_mode=True,
    linewidth=160,
)


# ============================================================
# Shared source builder
# ============================================================

_SOURCE_TABLES = None
_SOURCE_ERRORS = None


def write_exposure_csv(path: Path, eta: torch.Tensor, exposure: torch.Tensor):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Exposure"])
        for value in exposure.detach().cpu():
            writer.writerow([float(value)])

    print(f"Created CSV source from math table: {path}")
    print("CSV eta grid is implicit and reconstructed by nadir_exposure_from_csv.")


def get_source_tables():
    global _SOURCE_TABLES, _SOURCE_ERRORS

    if _SOURCE_TABLES is not None:
        return _SOURCE_TABLES, _SOURCE_ERRORS

    tables = {}
    errors = {}

    print("\nBuilding reference source='math' table...")
    math_table = build_nadir_exposure(
        source="math",
        lam_rad=LATITUDE_RAD,
        d1=D1,
        d2=D2,
        ns=NS,
        normalized=False,
        use_cache=False,
        device=DEVICE,
        dtype=DTYPE,
    )
    tables["math"] = math_table

    print("\nSeeding source='cache' from the math table...")
    save_nadir_exposure_to_cache(
        math_table.eta,
        math_table.exposure,
        lam_rad=LATITUDE_RAD,
        d1=D1,
        d2=D2,
        ns=NS,
        daynight=None,
        cache_dir=str(CACHE_DIR),
    )

    tables["cache"] = build_nadir_exposure(
        source="cache",
        lam_rad=LATITUDE_RAD,
        d1=D1,
        d2=D2,
        ns=NS,
        normalized=False,
        use_cache=False,
        cache_dir=str(CACHE_DIR),
        device=DEVICE,
        dtype=DTYPE,
    )

    print("\nSeeding source='csv' from the math table...")
    csv_path = OUTPUT_DIR / "exposure_source_math_seed.csv"
    write_exposure_csv(csv_path, math_table.eta, math_table.exposure)

    tables["csv"] = build_nadir_exposure(
        source="csv",
        csv_path=str(csv_path),
        angle="Nadir",
        daynight=None,
        normalized=False,
        use_cache=False,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("\nBuilding source='legacy' table from tpeanuts/peanuts...")
    assert_true(LEGACY_DIR.is_dir(), f"Legacy peanuts directory not found: {LEGACY_DIR}")
    tables["legacy"] = build_nadir_exposure(
        source="legacy",
        lam_rad=LATITUDE_RAD,
        d1=D1,
        d2=D2,
        ns=NS,
        daynight=None,
        normalized=False,
        use_cache=False,
        device=DEVICE,
        dtype=DTYPE,
    )

    _SOURCE_TABLES = tables
    _SOURCE_ERRORS = errors

    return tables, errors


def trapz_area(table: NadirExposureTable) -> float:
    return torch.trapz(table.exposure, x=table.eta).item()


def max_abs_difference(table_a: NadirExposureTable, table_b: NadirExposureTable) -> float:
    if table_a.eta.shape == table_b.eta.shape and torch.allclose(table_a.eta, table_b.eta):
        exposure_b = table_b.exposure
    else:
        exposure_b = table_b.interp(table_a.eta)

    return torch.max(torch.abs(table_a.exposure - exposure_b)).item()


def normalized_clone(table: NadirExposureTable) -> NadirExposureTable:
    out = NadirExposureTable(
        eta=table.eta.clone(),
        exposure=table.exposure.clone(),
    )
    out.normalize_()
    return out


# ============================================================
# tests
# ============================================================

def test_sources_load_with_consistent_shapes():
    tables, errors = get_source_tables()

    print("\nLoaded exposure sources:")
    for source in SOURCE_ORDER:
        if source in tables:
            table = tables[source]
            print(f"{source:7s}: eta shape={tuple(table.eta.shape)}, exposure shape={tuple(table.exposure.shape)}, area={trapz_area(table): .6e}")
        else:
            exc = errors.get(source)
            print(f"{source:7s}: unavailable ({type(exc).__name__}: {exc})")

    for required in SOURCE_ORDER:
        assert_true(required in tables, f"source='{required}' must be available")
        assert_true(isinstance(tables[required], NadirExposureTable), f"source='{required}' must return NadirExposureTable")
        assert_true(tables[required].eta.shape == (NS,), f"source='{required}' eta shape must be (NS,)")
        assert_true(tables[required].exposure.shape == (NS,), f"source='{required}' exposure shape must be (NS,)")
        assert_true(torch.isfinite(tables[required].exposure).all().item(), f"source='{required}' exposure must be finite")


def test_math_cache_csv_are_exact_roundtrips():
    tables, _ = get_source_tables()
    reference = tables["math"]

    diff_cache = max_abs_difference(reference, tables["cache"])
    diff_csv = max_abs_difference(reference, tables["csv"])

    print("\nExact roundtrip checks:")
    print("max |math - cache|:", f"{diff_cache:.6e}")
    print("max |math - csv|  :", f"{diff_csv:.6e}")

    assert_true(diff_cache < 1.0e-12, "Cache source must reproduce the seeded math table")
    assert_true(diff_csv < 1.0e-12, "CSV source must reproduce the seeded math table in Nadir mode")


def test_source_integrals_are_finite():
    tables, _ = get_source_tables()

    print("\nSource trapezoidal integrals:")
    for source, table in tables.items():
        area = trapz_area(table)
        print(f"{source:7s}: area = {area: .10e}")
        assert_true(torch.isfinite(torch.tensor(area)).item(), f"source='{source}' integral must be finite")


def test_legacy_source_comparison():
    tables, _ = get_source_tables()

    diff = max_abs_difference(tables["math"], tables["legacy"])

    print("\nLegacy source comparison:")
    print("max |math - legacy|:", f"{diff:.6e}")
    print("math area          :", f"{trapz_area(tables['math']): .10e}")
    print("legacy area        :", f"{trapz_area(tables['legacy']): .10e}")

    assert_true(isinstance(tables["legacy"], NadirExposureTable), "source='legacy' must return NadirExposureTable")
    assert_true(torch.isfinite(tables["legacy"].exposure).all().item(), "Legacy exposure must be finite")
    assert_true(torch.isfinite(torch.tensor(diff)).item(), "Legacy difference must be finite")


def test_normalized_sources_integrate_to_one():
    tables, _ = get_source_tables()

    print("\nNormalized source integrals:")
    for source, table in tables.items():
        normalized = normalized_clone(table)
        area = trapz_area(normalized)
        print(f"{source:7s}: normalized area = {area: .10e}")
        assert_true(abs(area - 1.0) < 1.0e-10, f"source='{source}' normalized exposure must integrate to one")


# ============================================================
# Visualization
# ============================================================

def plot_raw_source_overlay(savefig=False):
    tables, errors = get_source_tables()

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    for source in SOURCE_ORDER:
        if source not in tables:
            continue

        table = tables[source]
        ax.plot(
            table.eta.detach().cpu(),
            table.exposure.detach().cpu(),
            marker="o",
            label=source,
        )

    ax.axvline(float(torch.pi / 2.0), color="black", ls="--", lw=1.0)
    ax.set_xlabel(r"Nadir angle $\eta$ [rad]")
    ax.set_ylabel(r"Raw exposure $W(\eta)$")
    ax.set_title("Exposure source comparison: raw tables")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = OUTPUT_DIR / "earth_exposure_source_raw_overlay.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"\nSaved plot: {path}")
    plt.show()


def plot_difference_from_math(savefig=False):
    tables, _ = get_source_tables()
    reference = tables["math"]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    for source in SOURCE_ORDER:
        if source == "math" or source not in tables:
            continue

        table = tables[source]

        if table.eta.shape == reference.eta.shape and torch.allclose(table.eta, reference.eta):
            exposure = table.exposure
        else:
            exposure = table.interp(reference.eta)

        diff = torch.abs(exposure - reference.exposure)

        ax.semilogy(
            reference.eta.detach().cpu(),
            torch.clamp(diff.detach().cpu(), min=1.0e-30),
            marker="o",
            label=f"{source} - math",
        )

    ax.set_xlabel(r"Nadir angle $\eta$ [rad]")
    ax.set_ylabel(r"$|W_\mathrm{source} - W_\mathrm{math}|$")
    ax.set_title("Exposure source absolute differences")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = OUTPUT_DIR / "earth_exposure_source_difference_from_math.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"Saved plot: {path}")
    plt.show()


def plot_normalized_source_overlay(savefig=False):
    tables, _ = get_source_tables()

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    for source in SOURCE_ORDER:
        if source not in tables:
            continue

        table = normalized_clone(tables[source])
        ax.plot(
            table.eta.detach().cpu(),
            table.exposure.detach().cpu(),
            marker="o",
            label=source,
        )

    ax.axvline(float(torch.pi / 2.0), color="black", ls="--", lw=1.0)
    ax.set_xlabel(r"Nadir angle $\eta$ [rad]")
    ax.set_ylabel(r"Normalized exposure $W(\eta)$")
    ax.set_title("Exposure source comparison: normalized tables")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = OUTPUT_DIR / "earth_exposure_source_normalized_overlay.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"Saved plot: {path}")
    plt.show()


def test_visualization_outputs(savefig=False):
    plot_raw_source_overlay(savefig=savefig)
    plot_difference_from_math(savefig=savefig)
    plot_normalized_source_overlay(savefig=savefig)

    expected_files = [
        OUTPUT_DIR / "earth_exposure_source_raw_overlay.png",
        OUTPUT_DIR / "earth_exposure_source_difference_from_math.png",
        OUTPUT_DIR / "earth_exposure_source_normalized_overlay.png",
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
        test_sources_load_with_consistent_shapes,
        test_math_cache_csv_are_exact_roundtrips,
        test_source_integrals_are_finite,
        test_legacy_source_comparison,
        test_normalized_sources_integrate_to_one,
        test_visualization_outputs,
    ]

    run_test_suite(
        tests,
        suite_name="earth EXPOSURE SOURCE COMPARISON tests",
        verbose_traceback=True,
    )

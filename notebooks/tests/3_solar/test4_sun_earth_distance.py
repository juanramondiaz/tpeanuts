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
Spyder-friendly checks for Sun-earth distance constants and input data.
"""



from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch

THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parents[3]


from tpeanuts.io.io_solar import load_sun_earth_distance
from tpeanuts.util.test_utils import assert_true, run_test_suite
from tpeanuts.util.constant import AU_KM, AU_M, R_SUN, R_SUN_KM, SUN_EARTH_DISTANCE_KM

DEVICE = torch.device("cpu")
DTYPE = torch.float64
DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_TEST_ROOT = Path(OUTPUT_ROOT / "test")
OUTPUT_DIR = Path(OUTPUT_TEST_ROOT / "solar" / Path(__file__).stem)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_solar_and_orbital_constants():
    print("\nsolar and orbital constants:")
    print(f"R_SUN       = {R_SUN:.6e} m")
    print(f"R_SUN_KM    = {R_SUN_KM:.6e} km")
    print(f"AU_KM       = {AU_KM:.6e} km")
    print(f"AU_M        = {AU_M:.6e} m")
    print(f"default LSE = {SUN_EARTH_DISTANCE_KM:.6e} km")

    assert_true(abs(R_SUN / 1.0e3 - R_SUN_KM) < 1.0e-9, "solar radius km conversion must be consistent")
    assert_true(abs(AU_M / 1.0e3 - AU_KM) < 1.0e-6, "AU m/km conversion must be consistent")
    assert_true(abs(SUN_EARTH_DISTANCE_KM - AU_KM) < 1.0e-12, "Default Sun-earth distance must be 1 AU")


def test_load_sun_earth_distance_table():
    table = load_sun_earth_distance(device=DEVICE, dtype=DTYPE)

    dates = table["date"]
    distance_km = table["distance_km"]
    distance_au = table["distance_AU"]
    reconstructed_au = distance_km / AU_KM
    max_au_error = torch.max(torch.abs(reconstructed_au - distance_au)).item()

    print("\nSun-earth distance table:")
    print(f"number of rows: {len(dates)}")
    print(f"first date    : {dates[0]}")
    print(f"last date     : {dates[-1]}")
    print(f"min distance  : {torch.min(distance_km).item():.6f} km")
    print(f"max distance  : {torch.max(distance_km).item():.6f} km")
    print(f"mean distance : {torch.mean(distance_km).item():.6f} km")
    print(f"max AU error  : {max_au_error:.6e}")

    assert_true(len(dates) == distance_km.numel(), "Date count must match distance grid")
    assert_true(distance_km.numel() > 300, "Distance table should contain a yearly grid")
    assert_true(torch.all(distance_km > 0.0).item(), "All Sun-earth distances must be positive")
    assert_true(max_au_error < 1.0e-12, "distance_AU must match distance_km / AU_KM")


def plot_sun_earth_distance(savefig=False):
    table = load_sun_earth_distance(device=DEVICE, dtype=DTYPE)
    day = torch.arange(table["distance_AU"].numel(), dtype=DTYPE) + 1.0

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(day.cpu(), table["distance_AU"].cpu(), color="tab:blue")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="1 AU")
    ax.set_xlabel("Day index")
    ax.set_ylabel("Sun-earth distance [AU]")
    ax.set_title("Sun-earth distance table")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = OUTPUT_DIR / "solar_sun_earth_distance.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"\nSaved plot: {path}")
    plt.show()


def test_visualization_outputs(savefig=False):
    plot_sun_earth_distance(savefig=savefig)
    path = OUTPUT_DIR / "solar_sun_earth_distance.png"
    if savefig:
        assert_true(path.is_file(), f"Plot was not created: {path}")


if __name__ == "__main__":
    tests = [
        test_solar_and_orbital_constants,
        test_load_sun_earth_distance_table,
        test_visualization_outputs,
    ]

    run_test_suite(tests, suite_name="SUN-earth DISTANCE CONSTANTS AND data tests", verbose_traceback=True)

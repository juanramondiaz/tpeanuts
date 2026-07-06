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
Spyder-friendly tests for coherent solar coordinate conversions.
"""



from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch

THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parents[3]


from tpeanuts.coherent.coordinates import (
    distance_to_solar_radius_fraction,
    production_to_surface_path_length,
    solar_path_grid,
    solar_radius_fraction_to_distance,
    solar_shell_widths,
)
from tpeanuts.solar.profiles import load_default_solar_profile
from tpeanuts.util.test_utils import assert_true, run_test_suite
from tpeanuts.util.constant import R_SUN, R_SUN_KM

DEVICE = torch.device("cpu")
DTYPE = torch.float64
DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_TEST_ROOT = Path(OUTPUT_ROOT / "test")
OUTPUT_DIR = Path(OUTPUT_TEST_ROOT / "solar" / Path(__file__).stem)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_radius_fraction_distance_roundtrip():
    rho = torch.tensor([0.0, 0.05, 0.25, 0.70, 1.0], device=DEVICE, dtype=DTYPE)
    radius_m = solar_radius_fraction_to_distance(rho, unit="m", device=DEVICE, dtype=DTYPE)
    radius_km = solar_radius_fraction_to_distance(rho, unit="km", device=DEVICE, dtype=DTYPE)
    rho_from_m = distance_to_solar_radius_fraction(radius_m, unit="m", device=DEVICE, dtype=DTYPE)
    rho_from_km = distance_to_solar_radius_fraction(radius_km, unit="km", device=DEVICE, dtype=DTYPE)

    print("\nsolar radius conversion roundtrip:")
    print("rho       :", rho)
    print("radius [m]:", radius_m)
    print("radius [km]:", radius_km)
    print("rho from m :", rho_from_m)
    print("rho from km:", rho_from_km)

    assert_true(torch.allclose(radius_m / 1.0e3, radius_km, atol=1.0e-9), "m/km solar radius conversion must be consistent")
    assert_true(torch.allclose(rho_from_m, rho, atol=1.0e-12), "Roundtrip through meters must recover rho")
    assert_true(torch.allclose(rho_from_km, rho, atol=1.0e-12), "Roundtrip through kilometers must recover rho")


def test_production_to_surface_path_length():
    rho0 = torch.tensor([0.0, 0.10, 0.50, 0.95, 1.0], device=DEVICE, dtype=DTYPE)
    length_m = production_to_surface_path_length(rho0, unit="m", device=DEVICE, dtype=DTYPE)
    length_km = production_to_surface_path_length(rho0, unit="km", device=DEVICE, dtype=DTYPE)
    expected_m = (1.0 - rho0) * R_SUN

    print("\nRadial path length from production point to solar surface:")
    print("rho0      :", rho0)
    print("length [m]:", length_m)
    print("length [km]:", length_km)

    assert_true(torch.allclose(length_m, expected_m, atol=1.0e-6), "Path length in meters must match (1-rho0) R_SUN")
    assert_true(torch.allclose(length_m / 1.0e3, length_km, atol=1.0e-9), "Path length m/km conversion must be consistent")
    assert_true(abs(length_km[0].item() - R_SUN_KM) < 1.0e-9, "Central production path must be one solar radius")
    assert_true(abs(length_km[-1].item()) < 1.0e-12, "Surface production path must be zero")


def test_path_grid_from_profile_radius():
    profile = load_default_solar_profile(device=DEVICE, dtype=DTYPE)
    rho0 = torch.tensor(0.08, device=DEVICE, dtype=DTYPE)
    grid = solar_path_grid(rho0, profile_radius=profile.radius, device=DEVICE, dtype=DTYPE)
    widths_km = solar_shell_widths(grid, unit="km", device=DEVICE, dtype=DTYPE)
    total_km = torch.sum(widths_km)
    expected_km = production_to_surface_path_length(rho0, unit="km", device=DEVICE, dtype=DTYPE)

    print("\nsolar path grid built from the solar model radius samples:")
    print(f"rho0              : {rho0.item():.6f}")
    print(f"number of points  : {grid.numel()}")
    print(f"first grid point  : {grid[0].item():.6f}")
    print(f"last grid point   : {grid[-1].item():.6f}")
    print(f"min shell width km: {torch.min(widths_km).item():.6e}")
    print(f"max shell width km: {torch.max(widths_km).item():.6e}")
    print(f"total path km     : {total_km.item():.6f}")
    print(f"expected path km  : {expected_km.item():.6f}")

    assert_true(torch.isclose(grid[0], rho0, atol=1.0e-12).item(), "Path grid must start at rho0")
    assert_true(torch.isclose(grid[-1], torch.tensor(1.0, dtype=DTYPE), atol=1.0e-12).item(), "Path grid must end at the solar surface")
    assert_true(torch.all(widths_km >= 0.0).item(), "Shell widths must be non-negative")
    assert_true(torch.allclose(total_km, expected_km, atol=1.0e-8), "Shell widths must sum to the production-to-surface path")


def plot_coordinate_conversions(savefig=False):
    profile = load_default_solar_profile(device=DEVICE, dtype=DTYPE)
    rho0 = torch.tensor(0.08, device=DEVICE, dtype=DTYPE)
    grid = solar_path_grid(rho0, profile_radius=profile.radius, device=DEVICE, dtype=DTYPE)
    radius_km = solar_radius_fraction_to_distance(grid, unit="km", device=DEVICE, dtype=DTYPE)
    widths_km = solar_shell_widths(grid, unit="km", device=DEVICE, dtype=DTYPE)
    midpoint = 0.5 * (grid[:-1] + grid[1:])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(grid.cpu(), radius_km.cpu(), marker=".", linewidth=1.0)
    axes[0].set_xlabel("solar radius fraction r / R_sun")
    axes[0].set_ylabel("Physical radius [km]")
    axes[0].set_title("solar coordinate conversion")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(midpoint.cpu(), widths_km.cpu(), marker=".", linewidth=1.0)
    axes[1].set_xlabel("Shell midpoint r / R_sun")
    axes[1].set_ylabel("Shell width [km]")
    axes[1].set_title("solar propagation shell widths")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = OUTPUT_DIR / "coherent_solar_coordinate_conversion.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"\nSaved plot: {path}")
    plt.show()


def test_visualization_outputs(savefig=False):
    plot_coordinate_conversions(savefig=savefig)
    path = OUTPUT_DIR / "coherent_solar_coordinate_conversion.png"
    if savefig:
        assert_true(path.is_file(), f"Plot was not created: {path}")


if __name__ == "__main__":
    tests = [
        test_radius_fraction_distance_roundtrip,
        test_production_to_surface_path_length,
        test_path_grid_from_profile_radius,
        test_visualization_outputs,
    ]

    run_test_suite(tests, suite_name="coherent solar COORDINATE tests", verbose_traceback=True)

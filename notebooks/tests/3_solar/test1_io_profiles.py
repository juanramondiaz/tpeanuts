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
Spyder-friendly tests for tpeanuts.solar IO and profiles.
"""



from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch

THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parents[3]


from tpeanuts.io.io_solar import load_b16_fluxes, load_b16_solar_model
from tpeanuts.solar.profiles import load_default_solar_profile
from tpeanuts.util.test_utils import assert_true, run_test_suite

DEVICE = torch.device("cpu")
DTYPE = torch.float64
DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_TEST_ROOT = Path(OUTPUT_ROOT / "test")
OUTPUT_DIR = Path(OUTPUT_TEST_ROOT / "solar" / Path(__file__).stem)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_load_b16_solar_model():
    data = load_b16_solar_model(device=DEVICE, dtype=DTYPE)

    print("\nLoaded B16 solar model:")
    print("radius shape :", tuple(data["radius"].shape))
    print("density shape:", tuple(data["density"].shape))
    print("sources      :", sorted(data["fractions"].keys()))

    assert_true(data["radius"].ndim == 1, "Radius must be 1D")
    assert_true(data["density"].shape == data["radius"].shape, "density shape must match radius")
    assert_true("8B" in data["fractions"], "8B source must be available")
    assert_true(torch.isfinite(data["density"]).all().item(), "density must be finite")


def test_load_b16_fluxes():
    fluxes = load_b16_fluxes(device=DEVICE, dtype=DTYPE)

    print("\nLoaded B16 fluxes:")
    for key in sorted(fluxes):
        print(f"{key:4s}: {fluxes[key].item():.6e}")

    assert_true("pp" in fluxes, "pp flux must be available")
    assert_true("8B" in fluxes, "8B flux must be available")
    assert_true(all(value.item() > 0.0 for value in fluxes.values()), "fluxes must be positive")


def test_profile_interpolation_and_normalization():
    profile = load_default_solar_profile(device=DEVICE, dtype=DTYPE)
    r_query = torch.linspace(0.0, 1.0, 50, device=DEVICE, dtype=DTYPE)

    density_q = profile.electron_density(r_query)
    frac = profile.normalized_fraction("8B")
    area = torch.trapz(frac, x=profile.radius).item()

    print("\nProfile interpolation:")
    print("density query shape:", tuple(density_q.shape))
    print("8B normalized area :", f"{area:.10f}")

    assert_true(density_q.shape == r_query.shape, "Interpolated density shape must match query")
    assert_true(torch.isfinite(density_q).all().item(), "Interpolated density must be finite")
    assert_true(abs(area - 1.0) < 1.0e-10, "Normalized source fraction must integrate to one")


def plot_solar_profile(savefig=False):
    profile = load_default_solar_profile(device=DEVICE, dtype=DTYPE)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].semilogy(profile.radius.cpu(), profile.density.cpu())
    axes[0].set_xlabel(r"solar radius $r/R_\odot$")
    axes[0].set_ylabel(r"Electron density $n_e$ [mol cm$^{-3}$]")
    axes[0].set_title("solar electron density")
    axes[0].grid(True, alpha=0.3)

    for source in ["pp", "7Be", "8B", "13N", "15O", "17F", 'hep', "pep"]:
        axes[1].plot(profile.radius.cpu(), profile.normalized_fraction(source).cpu(), label=source)

    axes[1].set_xlabel(r"solar radius $r/R_\odot$")
    axes[1].set_ylabel("Normalized production fraction")
    axes[1].set_title("solar production profiles")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()

    path = OUTPUT_DIR / "solar_profile_density_and_sources.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"\nSaved plot: {path}")
    plt.show()


def test_visualization_outputs(savefig=False):
    plot_solar_profile(savefig=savefig)
    path = OUTPUT_DIR / "solar_profile_density_and_sources.png"
    if savefig:
        assert_true(path.is_file(), f"Plot was not created: {path}")


if __name__ == "__main__":
    tests = [
        test_load_b16_solar_model,
        test_load_b16_fluxes,
        test_profile_interpolation_and_normalization,
        test_visualization_outputs,
    ]

    run_test_suite(tests, suite_name="solar IO AND PROFILE tests", verbose_traceback=True)

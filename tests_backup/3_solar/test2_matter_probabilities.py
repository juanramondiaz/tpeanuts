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
Spyder-friendly tests for solar matter mixing and probabilities.
"""



from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch

THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parents[3]


from tpeanuts.core.pmns import PMNS
from tpeanuts.solar.matter_mixing import DeltamSqee, Vk, th12_M, th13_M
from tpeanuts.solar.profiles import load_default_solar_profile
from tpeanuts.solar.probabilities import Tei, psolar, solar_flux_mass
from tpeanuts.solar.validation import compare_psolar_with_legacy
from tpeanuts.util.test_utils import assert_true, run_test_suite

DEVICE = torch.device("cpu")
DTYPE = torch.float64
DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_TEST_ROOT = Path(OUTPUT_ROOT / "test")
OUTPUT_DIR = Path(OUTPUT_TEST_ROOT / "solar" / Path(__file__).stem)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DM21 = 7.42e-5
DM3L = 2.517e-3


def build_pmns():
    return PMNS(0.59, 0.15, 0.78, 1.20, device=DEVICE, real_dtype=DTYPE)


def test_matter_mixing_shapes():
    E = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)
    ne = torch.tensor([100.0, 50.0, 10.0], device=DEVICE, dtype=DTYPE)

    th13m = th13_M(0.59, 0.15, DM21, DM3L, E, ne)
    th12m = th12_M(0.59, 0.15, DM21, DM3L, E, ne)
    vk = Vk(DM21, E, ne)
    dmee = DeltamSqee(0.59, DM21, DM3L)

    print("\nMatter mixing:")
    print("Vk      :", vk)
    print("Dmee    :", dmee)
    print("theta13M:", th13m)
    print("theta12M:", th12m)

    assert_true(th13m.shape == E.shape, "theta13M shape must match energy")
    assert_true(th12m.shape == E.shape, "theta12M shape must match energy")
    assert_true(torch.isfinite(th13m).all().item(), "theta13M must be finite")
    assert_true(torch.isfinite(th12m).all().item(), "theta12M must be finite")


def test_solar_probabilities_are_normalized():
    profile = load_default_solar_profile(device=DEVICE, dtype=DTYPE)
    pmns = build_pmns()
    E = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    weights = solar_flux_mass(pmns.theta12, pmns.theta13, DM21, DM3L, E, profile.radius, profile.density, profile.production_fraction("8B"))
    P = psolar(pmns, DM21, DM3L, E, profile.radius, profile.density, profile.production_fraction("8B"))

    print("\nsolar 8B probabilities:")
    print("mass weights:", weights)
    print("P           :", P)
    print("P sums      :", P.sum(-1))

    assert_true(weights.shape == (3, 3), "Mass weights must have shape (NE, 3)")
    assert_true(P.shape == (3, 3), "Probabilities must have shape (NE, 3)")
    assert_true(torch.allclose(P.sum(-1), torch.ones(3, dtype=DTYPE), atol=1.0e-10), "Probabilities must sum to one")


def test_compare_psolar_with_legacy():
    pmns = build_pmns()
    result = compare_psolar_with_legacy("8B", pmns, DM21, DM3L, 5.0, device=DEVICE, dtype=DTYPE)

    print("\nTorch vs legacy Psolar:")
    print("torch   :", result["torch"])
    print("legacy  :", result["legacy"])
    print("abs diff:", result["abs_diff"])
    print("max abs :", f"{result['max_abs']:.6e}")

    assert_true(result["max_abs"] < 1.0e-10, "Torch Psolar must match legacy for 8B at 5 MeV")


def plot_psolar_vs_energy(savefig=False):
    profile = load_default_solar_profile(device=DEVICE, dtype=DTYPE)
    pmns = build_pmns()
    E = torch.linspace(0.5, 15.0, 80, device=DEVICE, dtype=DTYPE)

    fig, ax = plt.subplots(figsize=(8, 4.8))

    for source in ["pp", "7Be", "8B", "13N", "15O"]:
        P = psolar(pmns, DM21, DM3L, E, profile.radius, profile.density, profile.production_fraction(source))
        ax.plot(E.cpu(), P[:, 0].cpu(), label=source)

    ax.set_xlabel("Energy [MeV]")
    ax.set_ylabel(r"$P_{e}$")
    ax.set_title("solar electron-flavour probability by source")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = OUTPUT_DIR / "solar_psolar_pe_vs_energy.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"\nSaved plot: {path}")
    plt.show()


def test_visualization_outputs(savefig=False):
    plot_psolar_vs_energy(savefig=savefig)
    path = OUTPUT_DIR / "solar_psolar_pe_vs_energy.png"
    if savefig:
        assert_true(path.is_file(), f"Plot was not created: {path}")


if __name__ == "__main__":
    tests = [
        test_matter_mixing_shapes,
        test_solar_probabilities_are_normalized,
        test_compare_psolar_with_legacy,
        test_visualization_outputs,
    ]

    run_test_suite(tests, suite_name="solar MATTER AND PROBABILITY tests", verbose_traceback=True)

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
Spyder-friendly analytical-vs-numerical probability comparison.

This script compares:

    1. pearth(..., method="analytical") against pearth(..., method="numerical").
    2. Case-A through-earth trajectories.
    3. Case-B underground shallow trajectories.
    4. Numerical full-path evolution against the numerical final state.
    5. Mass-basis diagnostic differences.
    6. Visual diagnostics of analytical/numerical agreement.

The tight agreement checks are made in flavour basis, where both methods act
directly on a coherent flavour state.

Run directly in Spyder or from a terminal:

    python tpeanuts/tests/earth/test6b_prob_compare.py
"""



from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch


# ============================================================
# Import bootstrap
# ============================================================

THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parents[3]



from tpeanuts.core.pmns import PMNS  # noqa: E402
from tpeanuts.io.io_earth import load_earth_density_from_csv  # noqa: E402
from tpeanuts.earth.probabilities import pearth  # noqa: E402
from tpeanuts.util.test_utils import (  # noqa: E402
    assert_true,
    run_test_suite,
)


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device("cpu")
DTYPE = torch.float64
CDTYPE = torch.complex128

REAL_DENSITY_FILE = PACKAGE_DIR / "data" / "density" / "earth_density.csv"
TESTS_DIR = THIS_FILE.parents[1]

DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_TEST_ROOT = Path(OUTPUT_ROOT / "test")
OUTPUT_DIR = Path(OUTPUT_TEST_ROOT / "earth" / Path(__file__).stem)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DM21_EV2 = 7.42e-5
DM3L_EV2 = 2.517e-3

DEPTH_SURFACE_M = 0.0
DEPTH_UNDERGROUND_M = 1000.0

NSTEPS_COMPARE = 80

FLAVOUR_LABELS = [r"$\nu_e$", r"$\nu_\mu$", r"$\nu_\tau$"]

torch.set_printoptions(
    precision=10,
    sci_mode=True,
    linewidth=160,
)


# ============================================================
# Shared fixtures and helpers
# ============================================================

def build_pmns():
    return PMNS(
        theta12=0.59,
        theta13=0.15,
        theta23=0.78,
        delta=1.20,
        device=DEVICE,
        real_dtype=DTYPE,
    )


def load_density():
    return load_earth_density_from_csv(
        str(REAL_DENSITY_FILE),
        tabulated_density=False,
        device=DEVICE,
        dtype=DTYPE,
    )


def electron_flavour_state():
    return torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=CDTYPE)


def first_mass_state():
    return torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)


def analytical_probability(state, E, eta, depth_m, *, massbasis):
    return pearth(
        nustate=state,
        density=load_density(),
        pmns=build_pmns(),
        dm21_eV2=DM21_EV2,
        dm3l_eV2=DM3L_EV2,
        E_MeV=E,
        eta=eta,
        depth_m=depth_m,
        method="analytical",
        massbasis=massbasis,
        reunitarize=True,
    )


def numerical_probability(state, E, eta, depth_m, *, massbasis, nsteps=NSTEPS_COMPARE, full_oscillation=False):
    return pearth(
        nustate=state,
        density=load_density(),
        pmns=build_pmns(),
        dm21_eV2=DM21_EV2,
        dm3l_eV2=DM3L_EV2,
        E_MeV=E,
        eta=eta,
        depth_m=depth_m,
        method="numerical",
        massbasis=massbasis,
        full_oscillation=full_oscillation,
        nsteps=nsteps,
        ode_method="midpoint",
        device=DEVICE,
        dtype=DTYPE,
    )


def max_abs_difference(a, b):
    return torch.max(torch.abs(a - b)).item()


def assert_probability_vector(P, message, atol=1.0e-8):
    row_sum = torch.sum(P, dim=-1)

    print(f"Checking: {message}")
    print("  P shape  :", tuple(P.shape))
    print("  min/max P:", P.min().item(), P.max().item())
    print("  sum(P)   :", row_sum)

    assert_true(torch.isfinite(P).all().item(), f"{message}: probabilities must be finite")
    assert_true((P >= -atol).all().item(), f"{message}: probabilities must be non-negative within tolerance")
    assert_true(torch.allclose(row_sum, torch.ones_like(row_sum), atol=atol, rtol=atol), f"{message}: probabilities must sum to one")


def compare_flavour_case(name, E_value, eta_value, depth_m, tolerance):
    state = electron_flavour_state()
    E = torch.tensor(E_value, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(eta_value, device=DEVICE, dtype=DTYPE)

    P_analytical = analytical_probability(
        state,
        E,
        eta,
        depth_m,
        massbasis=False,
    )

    P_numerical = numerical_probability(
        state,
        E,
        eta,
        depth_m,
        massbasis=False,
    )

    diff = max_abs_difference(P_analytical, P_numerical)

    print(f"\n{name}:")
    print("E [MeV]       :", E_value)
    print("eta [rad]     :", eta_value)
    print("depth [m]     :", depth_m)
    print("P analytical  :", P_analytical)
    print("P numerical   :", P_numerical)
    print("max |diff|    :", f"{diff:.6e}")
    print("tolerance     :", f"{tolerance:.6e}")

    assert_probability_vector(P_analytical, f"{name} analytical probabilities")
    assert_probability_vector(P_numerical, f"{name} numerical probabilities")
    assert_true(diff < tolerance, f"{name}: analytical and numerical probabilities must agree within tolerance")

    return P_analytical, P_numerical, diff


# ============================================================
# tests
# ============================================================

def test_case_a_flavourbasis_analytical_vs_numerical():
    compare_flavour_case(
        name="Case A through-earth flavour-basis comparison",
        E_value=1000.0,
        eta_value=0.60,
        depth_m=DEPTH_SURFACE_M,
        tolerance=5.0e-3,
    )


def test_case_b_flavourbasis_analytical_vs_numerical():
    compare_flavour_case(
        name="Case B underground flavour-basis comparison",
        E_value=1000.0,
        eta_value=2.40,
        depth_m=DEPTH_UNDERGROUND_M,
        tolerance=5.0e-6,
    )


def test_numerical_full_oscillation_final_matches_final_only():
    state = electron_flavour_state()
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.60, device=DEVICE, dtype=DTYPE)

    evolution, x = numerical_probability(
        state,
        E,
        eta,
        DEPTH_SURFACE_M,
        massbasis=False,
        full_oscillation=True,
    )

    final_only = numerical_probability(
        state,
        E,
        eta,
        DEPTH_SURFACE_M,
        massbasis=False,
        full_oscillation=False,
    )

    diff = max_abs_difference(evolution[-1], final_only)

    print("\nNumerical full-path consistency:")
    print("evolution shape:", tuple(evolution.shape))
    print("x shape        :", tuple(x.shape))
    print("final path     :", evolution[-1])
    print("final only     :", final_only)
    print("max |diff|     :", f"{diff:.6e}")

    assert_true(evolution.shape == (NSTEPS_COMPARE + 1, 3), "Full numerical evolution must have shape (nsteps + 1, 3)")
    assert_true(x.shape == (NSTEPS_COMPARE + 1,), "Numerical x grid must have nsteps + 1 points")
    assert_probability_vector(evolution[-1], "Final full-path numerical probabilities")
    assert_true(diff < 1.0e-12, "Final full-path state must match final-only numerical result")


def test_massbasis_diagnostic_difference_is_finite():
    state = first_mass_state()
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.60, device=DEVICE, dtype=DTYPE)

    P_analytical = analytical_probability(
        state,
        E,
        eta,
        DEPTH_SURFACE_M,
        massbasis=True,
    )

    P_numerical = numerical_probability(
        state,
        E,
        eta,
        DEPTH_SURFACE_M,
        massbasis=True,
    )

    diff = max_abs_difference(P_analytical, P_numerical)

    print("\nMass-basis diagnostic comparison:")
    print("P analytical:", P_analytical)
    print("P numerical :", P_numerical)
    print("max |diff|  :", f"{diff:.6e}")
    print("Note        : mass-basis comparison is diagnostic; flavour-basis is the tight consistency check.")

    assert_probability_vector(P_analytical, "Mass-basis analytical probabilities")
    assert_probability_vector(P_numerical, "Mass-basis numerical probabilities")
    assert_true(torch.isfinite(torch.tensor(diff)).item(), "Mass-basis analytical/numerical difference must be finite")


# ============================================================
# Visualization
# ============================================================

def plot_case_a_probability_bars(savefig=False):
    P_analytical, P_numerical, _ = compare_flavour_case(
        name="Case A plot data",
        E_value=1000.0,
        eta_value=0.60,
        depth_m=DEPTH_SURFACE_M,
        tolerance=5.0e-3,
    )

    x = torch.arange(3).detach().cpu().numpy()
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - width / 2.0, P_analytical.detach().cpu().numpy(), width=width, label="Analytical")
    ax.bar(x + width / 2.0, P_numerical.detach().cpu().numpy(), width=width, label="Numerical")
    ax.set_xticks(x)
    ax.set_xticklabels(FLAVOUR_LABELS)
    ax.set_ylabel("Final flavour probability")
    ax.set_title("Case-A analytical vs numerical probabilities")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = OUTPUT_DIR / "earth_prob_compare_case_a_bars.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"\nSaved plot: {path}")
    plt.show()


def plot_error_vs_eta(savefig=False):
    state = electron_flavour_state()
    eta_values = torch.tensor([0.25, 0.45, 0.65, 0.85, 1.05], device=DEVICE, dtype=DTYPE)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)

    errors = []

    for eta in eta_values:
        P_analytical = analytical_probability(
            state,
            E,
            eta,
            DEPTH_SURFACE_M,
            massbasis=False,
        )

        P_numerical = numerical_probability(
            state,
            E,
            eta,
            DEPTH_SURFACE_M,
            massbasis=False,
            nsteps=60,
        )

        errors.append(max_abs_difference(P_analytical, P_numerical))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.semilogy(eta_values.detach().cpu(), errors, marker="o")
    ax.set_xlabel(r"Nadir angle $\eta$ [rad]")
    ax.set_ylabel(r"max $|P_\mathrm{analytical} - P_\mathrm{numerical}|$")
    ax.set_title("Analytical-vs-numerical probability error")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = OUTPUT_DIR / "earth_prob_compare_error_vs_eta.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"Saved plot: {path}")
    plt.show()


def plot_numerical_path_evolution(savefig=False):
    state = electron_flavour_state()
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.60, device=DEVICE, dtype=DTYPE)

    evolution, x = numerical_probability(
        state,
        E,
        eta,
        DEPTH_SURFACE_M,
        massbasis=False,
        full_oscillation=True,
    )

    P_analytical = analytical_probability(
        state,
        E,
        eta,
        DEPTH_SURFACE_M,
        massbasis=False,
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    for idx, label in enumerate(FLAVOUR_LABELS):
        ax.plot(x.detach().cpu(), evolution[:, idx].detach().cpu(), label=f"Numerical {label}")
        ax.axhline(P_analytical[idx].item(), ls="--", lw=1.0, alpha=0.75, label=f"Analytical final {label}")

    ax.set_xlabel("Trajectory coordinate x")
    ax.set_ylabel("Flavour probability")
    ax.set_title("Numerical path evolution with analytical final reference")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()

    path = OUTPUT_DIR / "earth_prob_compare_numerical_path.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"Saved plot: {path}")
    plt.show()


def test_visualization_outputs(savefig=False):
    plot_case_a_probability_bars(savefig=savefig)
    plot_error_vs_eta(savefig=savefig)
    plot_numerical_path_evolution(savefig=savefig)

    expected_files = [
        OUTPUT_DIR / "earth_prob_compare_case_a_bars.png",
        OUTPUT_DIR / "earth_prob_compare_error_vs_eta.png",
        OUTPUT_DIR / "earth_prob_compare_numerical_path.png",
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
        test_case_a_flavourbasis_analytical_vs_numerical,
        test_case_b_flavourbasis_analytical_vs_numerical,
        test_numerical_full_oscillation_final_matches_final_only,
        test_massbasis_diagnostic_difference_is_finite,
        test_visualization_outputs,
    ]

    run_test_suite(
        tests,
        suite_name="earth PROBABILITY ANALYTICAL VS NUMERICAL tests",
        verbose_traceback=True,
    )

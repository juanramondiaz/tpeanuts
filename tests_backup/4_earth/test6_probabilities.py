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
Spyder-friendly tests and visual diagnostics for tpeanuts.earth.probabilities.

This script checks:

    1. Probability-module imports and helper broadcasting.
    2. Mass-basis analytical probabilities.
    3. Flavour-basis coherent probabilities.
    4. Above-horizon identity behaviour at zero detector depth.
    5. Energy/eta broadcasting.
    6. Antineutrino analytical path.
    7. Visual diagnostics for earth probabilities.

Run directly in Spyder or from a terminal:

    python tpeanuts/tests/earth/test6_probabilities.py
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
from tpeanuts.earth.probabilities import (  # noqa: E402
    _broadcast_mass_weights,
    pearth,
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


def assert_probability_vector(P, message, atol=1.0e-8):
    row_sum = torch.sum(P, dim=-1)

    print(f"Checking: {message}")
    print("  P shape       :", tuple(P.shape))
    print("  min/max P     :", P.min().item(), P.max().item())
    print("  row sums      :", row_sum)

    assert_true(torch.isfinite(P).all().item(), f"{message}: probabilities must be finite")
    assert_true((P >= -atol).all().item(), f"{message}: probabilities must be non-negative within tolerance")
    assert_tensor_close(row_sum, torch.ones_like(row_sum), f"{message}: probabilities sum to one", atol=atol, rtol=atol)


def call_pearth(
    nustate,
    E,
    eta,
    *,
    massbasis=True,
    antinu=False,
    depth_m=DEPTH_SURFACE_M,
):
    return pearth(
        nustate=nustate,
        density=load_density(),
        pmns=build_pmns(),
        dm21_eV2=DM21_EV2,
        dm3l_eV2=DM3L_EV2,
        E_MeV=E,
        eta=eta,
        depth_m=depth_m,
        method="analytical",
        antinu=antinu,
        massbasis=massbasis,
        reunitarize=True,
    )


# ============================================================
# tests
# ============================================================

def test_broadcast_mass_weights_helper():
    weights = torch.tensor([0.70, 0.20, 0.10], device=DEVICE, dtype=DTYPE)
    probs_i_to_alpha = torch.zeros((4, 5, 3, 3), device=DEVICE, dtype=DTYPE)

    weights_b = _broadcast_mass_weights(weights, probs_i_to_alpha)

    print("\nBroadcast mass weights:")
    print("weights shape          :", tuple(weights.shape))
    print("probs_i_to_alpha shape :", tuple(probs_i_to_alpha.shape))
    print("broadcast weights shape:", tuple(weights_b.shape))

    assert_true(weights_b.shape == (1, 1, 3), "Mass weights must broadcast over leading probability dimensions")


def test_massbasis_scalar_probabilities_are_normalized():
    nustate = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)

    P = call_pearth(
        nustate,
        E=torch.tensor(1000.0, device=DEVICE, dtype=DTYPE),
        eta=torch.tensor(0.40, device=DEVICE, dtype=DTYPE),
        massbasis=True,
    )

    print("\nMass-basis scalar probabilities:")
    print("P:", P)

    assert_true(P.shape == (3,), "Scalar mass-basis output must have shape (3,)")
    assert_probability_vector(P, "Scalar mass-basis probabilities")


def test_flavourbasis_scalar_probabilities_are_normalized():
    psi_e = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)

    P = call_pearth(
        psi_e,
        E=torch.tensor(1200.0, device=DEVICE, dtype=DTYPE),
        eta=torch.tensor(0.60, device=DEVICE, dtype=DTYPE),
        massbasis=False,
    )

    print("\nFlavour-basis scalar probabilities:")
    print("P:", P)

    assert_true(P.shape == (3,), "Scalar flavour-basis output must have shape (3,)")
    assert_probability_vector(P, "Scalar flavour-basis probabilities")


def test_above_horizon_identity_limits():
    density = load_density()
    pmns = build_pmns()

    mass_state_1 = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)
    flavour_e = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)

    eta_above = torch.tensor(2.20, device=DEVICE, dtype=DTYPE)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)

    P_mass = pearth(
        mass_state_1,
        density,
        pmns,
        DM21_EV2,
        DM3L_EV2,
        E,
        eta_above,
        DEPTH_SURFACE_M,
        method="analytical",
        massbasis=True,
        reunitarize=True,
    )

    P_flavour = pearth(
        flavour_e,
        density,
        pmns,
        DM21_EV2,
        DM3L_EV2,
        E,
        eta_above,
        DEPTH_SURFACE_M,
        method="analytical",
        massbasis=False,
        reunitarize=True,
    )

    expected_mass = torch.abs(pmns.pmns_matrix()[:, 0]) ** 2
    expected_flavour = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)

    print("\nAbove-horizon identity limits:")
    print("P_mass          :", P_mass)
    print("expected mass   :", expected_mass)
    print("P_flavour       :", P_flavour)
    print("expected flavour:", expected_flavour)

    assert_tensor_close(P_mass, expected_mass.real, "Above-horizon mass-basis probabilities reduce to PMNS column")
    assert_tensor_close(P_flavour, expected_flavour, "Above-horizon flavour-basis probabilities keep the input flavour")


def test_energy_eta_grid_probabilities():
    weights = torch.tensor([0.55, 0.30, 0.15], device=DEVICE, dtype=DTYPE)
    E = torch.tensor([800.0, 2000.0, 6000.0], device=DEVICE, dtype=DTYPE)
    eta = torch.tensor([0.30, 1.10], device=DEVICE, dtype=DTYPE)

    P = call_pearth(
        weights,
        E=E,
        eta=eta,
        massbasis=True,
    )

    print("\nEnergy-eta grid probabilities:")
    print("E shape  :", tuple(E.shape))
    print("eta shape:", tuple(eta.shape))
    print("P shape  :", tuple(P.shape))
    print("P:")
    print(P)

    assert_true(P.shape == (3, 2, 3), "Energy vector and eta vector must produce shape (NE, Neta, 3)")
    assert_probability_vector(P, "Energy-eta grid probabilities")


def test_antineutrino_probabilities_are_valid():
    weights = torch.tensor([0.20, 0.50, 0.30], device=DEVICE, dtype=DTYPE)
    E = torch.tensor([1000.0, 5000.0], device=DEVICE, dtype=DTYPE)
    eta = torch.tensor([0.35, 1.00], device=DEVICE, dtype=DTYPE)

    P_nu = call_pearth(
        weights,
        E=E,
        eta=eta,
        massbasis=True,
        antinu=False,
    )

    P_antinu = call_pearth(
        weights,
        E=E,
        eta=eta,
        massbasis=True,
        antinu=True,
    )

    diff = torch.max(torch.abs(P_nu - P_antinu)).item()

    print("\nNeutrino vs antineutrino probabilities:")
    print("P_nu    :", P_nu)
    print("P_antinu:", P_antinu)
    print("max |difference|:", f"{diff:.6e}")

    assert_true(P_antinu.shape == (2, 3), "Antineutrino paired batch output must have shape (2, 3)")
    assert_probability_vector(P_antinu, "Antineutrino probabilities")
    assert_true(diff > 0.0, "Neutrino and antineutrino probabilities should not be exactly identical for these inputs")


# ============================================================
# Visualization
# ============================================================

def plot_probabilities_vs_eta_for_mass_states(savefig=False):
    density = load_density()
    pmns = build_pmns()

    eta = torch.linspace(0.0, torch.pi, 17, device=DEVICE, dtype=DTYPE)
    E = torch.full_like(eta, 2000.0)

    mass_states = [
        ("Initial mass state 1", torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)),
        ("Initial mass state 2", torch.tensor([0.0, 1.0, 0.0], device=DEVICE, dtype=DTYPE)),
        ("Initial mass state 3", torch.tensor([0.0, 0.0, 1.0], device=DEVICE, dtype=DTYPE)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), sharey=True)

    for ax, (title, weights) in zip(axes, mass_states):
        P = pearth(
            weights,
            density,
            pmns,
            DM21_EV2,
            DM3L_EV2,
            E,
            eta,
            DEPTH_SURFACE_M,
            method="analytical",
            massbasis=True,
            reunitarize=True,
        )

        for idx, label in enumerate(FLAVOUR_LABELS):
            ax.plot(eta.detach().cpu(), P[:, idx].detach().cpu(), marker="o", label=label)

        ax.axvline(float(torch.pi / 2.0), color="black", ls="--", lw=1.0)
        ax.set_title(title)
        ax.set_xlabel(r"Nadir angle $\eta$ [rad]")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Final flavour probability")
    axes[-1].legend(loc="best")
    fig.suptitle("earth probabilities vs nadir angle")
    fig.tight_layout()

    path = OUTPUT_DIR / "earth_probabilities_vs_eta_mass_states.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"\nSaved plot: {path}")
    plt.show()


def plot_energy_eta_probability_map(savefig=False):
    density = load_density()
    pmns = build_pmns()

    weights = torch.tensor([0.55, 0.30, 0.15], device=DEVICE, dtype=DTYPE)
    E = torch.tensor([500.0, 1000.0, 3000.0, 10000.0], device=DEVICE, dtype=DTYPE)
    eta = torch.linspace(0.0, torch.pi, 13, device=DEVICE, dtype=DTYPE)

    P = pearth(
        weights,
        density,
        pmns,
        DM21_EV2,
        DM3L_EV2,
        E,
        eta,
        DEPTH_SURFACE_M,
        method="analytical",
        massbasis=True,
        reunitarize=True,
    )

    P_e = P[..., 0].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 4.8))
    image = ax.imshow(
        P_e,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[
            float(eta[0].item()),
            float(eta[-1].item()),
            float(E[0].item()),
            float(E[-1].item()),
        ],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    ax.axvline(float(torch.pi / 2.0), color="white", ls="--", lw=1.0)
    ax.set_xlabel(r"Nadir angle $\eta$ [rad]")
    ax.set_ylabel("Energy [MeV]")
    ax.set_title(r"Electron-flavour probability $P_e(E,\eta)$")
    fig.colorbar(image, ax=ax, label=r"$P_e$")
    fig.tight_layout()

    path = OUTPUT_DIR / "earth_probability_energy_eta_map.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"Saved plot: {path}")
    plt.show()


def plot_massbasis_vs_flavourbasis_example(savefig=False):
    density = load_density()
    pmns = build_pmns()

    eta = torch.linspace(0.0, torch.pi, 17, device=DEVICE, dtype=DTYPE)
    E = torch.full_like(eta, 1500.0)

    mass_weights = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)
    flavour_e = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)

    P_mass = pearth(
        mass_weights,
        density,
        pmns,
        DM21_EV2,
        DM3L_EV2,
        E,
        eta,
        DEPTH_SURFACE_M,
        method="analytical",
        massbasis=True,
        reunitarize=True,
    )

    P_flavour = pearth(
        flavour_e,
        density,
        pmns,
        DM21_EV2,
        DM3L_EV2,
        E,
        eta,
        DEPTH_SURFACE_M,
        method="analytical",
        massbasis=False,
        reunitarize=True,
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(eta.detach().cpu(), P_mass[:, 0].detach().cpu(), marker="o", label=r"Mass basis: $P_e$")
    ax.plot(eta.detach().cpu(), P_flavour[:, 0].detach().cpu(), marker="s", label=r"Flavour basis: $P_e$")
    ax.axvline(float(torch.pi / 2.0), color="black", ls="--", lw=1.0)
    ax.set_xlabel(r"Nadir angle $\eta$ [rad]")
    ax.set_ylabel(r"Electron-flavour probability $P_e$")
    ax.set_title("Mass-basis and flavour-basis input comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = OUTPUT_DIR / "earth_probability_mass_vs_flavour_basis.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"Saved plot: {path}")
    plt.show()


def test_visualization_outputs(savefig=False):
    plot_probabilities_vs_eta_for_mass_states(savefig=savefig)
    plot_energy_eta_probability_map(savefig=savefig)
    plot_massbasis_vs_flavourbasis_example(savefig=savefig)

    expected_files = [
        OUTPUT_DIR / "earth_probabilities_vs_eta_mass_states.png",
        OUTPUT_DIR / "earth_probability_energy_eta_map.png",
        OUTPUT_DIR / "earth_probability_mass_vs_flavour_basis.png",
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
        test_broadcast_mass_weights_helper,
        test_massbasis_scalar_probabilities_are_normalized,
        test_flavourbasis_scalar_probabilities_are_normalized,
        test_above_horizon_identity_limits,
        test_energy_eta_grid_probabilities,
        test_antineutrino_probabilities_are_valid,
        test_visualization_outputs,
    ]

    run_test_suite(
        tests,
        suite_name="earth PROBABILITIES tests",
        verbose_traceback=True,
    )

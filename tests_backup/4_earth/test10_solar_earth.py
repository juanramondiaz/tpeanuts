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
Spyder-friendly solar + earth propagation test.

This test starts from a solar production source, computes the mass-eigenstate
weights after adiabatic solar propagation, and then propagates those weights
through the earth for several nadir angles.

It compares neutrino and antineutrino earth propagation numerically and
graphically. The antineutrino branch uses the same solar production weights as
a diagnostic input, so differences shown here isolate the earth matter-sign
effect.
"""



from __future__ import annotations

import os
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# ============================================================
# Import bootstrap
# ============================================================

THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parents[3]



from tpeanuts.core.pmns import PMNS  # noqa: E402
from tpeanuts.io.io_earth import load_earth_density_from_csv  # noqa: E402
from tpeanuts.earth.probabilities import pearth  # noqa: E402
from tpeanuts.solar.probabilities import solar_flux_mass  # noqa: E402
from tpeanuts.solar.profiles import load_default_solar_profile  # noqa: E402
from tpeanuts.util.test_utils import assert_true, run_test_suite  # noqa: E402


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device("cpu")
DTYPE = torch.float64

DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_TEST_ROOT = Path(OUTPUT_ROOT / "test")
OUTPUT_DIR = Path(OUTPUT_TEST_ROOT / "earth" / Path(__file__).stem)
os.makedirs(OUTPUT_DIR, exist_ok=True)

EARTH_DENSITY_FILE = PACKAGE_DIR / "data" / "density" / "earth_density.csv"
LEGACY_data_DIR = PACKAGE_DIR / "data" / "peanuts"

SOURCE = "8B"
SPECTRUM_FILE = LEGACY_data_DIR / "8B_shape_Ortiz_et_al.csv"

TH12 = np.arctan(sqrt(0.469))
TH13 = np.arcsin(sqrt(0.01))
TH23 = 0.85521
DELTA = 3.4034
DM21 = 7.9e-5
DM3L = 2.46e-3
DEPTH_M = 0.0

ENERGIES_MEV = torch.tensor([1.0, 2.0, 5.0, 10.0, 15.0], device=DEVICE, dtype=DTYPE)
ETA_VALUES = torch.tensor([0.0, np.pi / 6.0, np.pi / 3.0, np.pi / 2.0], device=DEVICE, dtype=DTYPE)

FLAVOUR_LABELS = [r"$\nu_e$", r"$\nu_\mu$", r"$\nu_\tau$"]
FLAVOUR_NAMES = ["nu_e", "nu_mu", "nu_tau"]

torch.set_printoptions(precision=10, sci_mode=True, linewidth=160)
np.set_printoptions(precision=10, suppress=False, linewidth=160)


# ============================================================
# Fixtures
# ============================================================

def build_pmns():
    return PMNS(TH12, TH13, TH23, DELTA, device=DEVICE, real_dtype=DTYPE)


def load_spectrum_on_grid(energies: torch.Tensor) -> torch.Tensor:
    table = pd.read_csv(
        SPECTRUM_FILE,
        comment="#",
        names=["Energy", "Spectrum"],
    ).dropna()

    energy_np = table["Energy"].astype(float).to_numpy()
    spectrum_np = table["Spectrum"].astype(float).to_numpy()

    values = np.interp(
        energies.detach().cpu().numpy(),
        energy_np,
        spectrum_np,
        left=0.0,
        right=0.0,
    )

    return torch.as_tensor(values, device=energies.device, dtype=energies.dtype)


def solar_mass_weights(energies: torch.Tensor) -> torch.Tensor:
    profile = load_default_solar_profile(device=DEVICE, dtype=DTYPE)
    pmns = build_pmns()

    return solar_flux_mass(
        pmns.theta12,
        pmns.theta13,
        DM21,
        DM3L,
        energies,
        profile.radius,
        profile.density,
        profile.production_fraction(SOURCE),
    )


def flavour_probabilities_from_mass_weights(weights: torch.Tensor, *, antinu: bool) -> torch.Tensor:
    pmns = build_pmns()
    U = pmns.pmns_matrix().to(device=weights.device)

    if antinu:
        U = torch.conj(U)

    probs_i_to_alpha = torch.abs(U) ** 2

    return torch.einsum("ai,Ei->Ea", probs_i_to_alpha, weights)


def earth_final_probabilities(weights: torch.Tensor, *, antinu: bool) -> torch.Tensor:
    density = load_earth_density_from_csv(
        str(EARTH_DENSITY_FILE),
        tabulated_density=False,
        device=DEVICE,
        dtype=DTYPE,
    )
    pmns = build_pmns()

    E_grid = ENERGIES_MEV[:, None].expand(ENERGIES_MEV.numel(), ETA_VALUES.numel())
    eta_grid = ETA_VALUES[None, :].expand(ENERGIES_MEV.numel(), ETA_VALUES.numel())

    return pearth(
        nustate=weights,
        density=density,
        pmns=pmns,
        dm21_eV2=DM21,
        dm3l_eV2=DM3L,
        E_MeV=E_grid,
        eta=eta_grid,
        depth_m=DEPTH_M,
        method="analytical",
        antinu=antinu,
        massbasis=True,
        reunitarize=True,
    )


_CACHE = None


def compute_solar_earth_case():
    global _CACHE

    if _CACHE is not None:
        return _CACHE

    spectrum = load_spectrum_on_grid(ENERGIES_MEV)
    weights = solar_mass_weights(ENERGIES_MEV)

    initial_prob_nu = flavour_probabilities_from_mass_weights(weights, antinu=False)
    initial_prob_antinu = flavour_probabilities_from_mass_weights(weights, antinu=True)

    initial_flux_nu = spectrum[:, None] * initial_prob_nu
    initial_flux_antinu = spectrum[:, None] * initial_prob_antinu

    final_prob_nu = earth_final_probabilities(weights, antinu=False)
    final_prob_antinu = earth_final_probabilities(weights, antinu=True)

    final_flux_nu = spectrum[:, None, None] * final_prob_nu
    final_flux_antinu = spectrum[:, None, None] * final_prob_antinu

    _CACHE = {
        "spectrum": spectrum,
        "weights": weights,
        "initial_prob_nu": initial_prob_nu,
        "initial_prob_antinu": initial_prob_antinu,
        "initial_flux_nu": initial_flux_nu,
        "initial_flux_antinu": initial_flux_antinu,
        "final_prob_nu": final_prob_nu,
        "final_prob_antinu": final_prob_antinu,
        "final_flux_nu": final_flux_nu,
        "final_flux_antinu": final_flux_antinu,
    }

    return _CACHE


# ============================================================
# tests
# ============================================================

def test_solar_mass_weights_are_normalized():
    data = compute_solar_earth_case()
    weights = data["weights"]

    print("\nsolar mass weights from source production:")
    print(weights)
    print("weight sums:", weights.sum(dim=-1))

    assert_true(weights.shape == (ENERGIES_MEV.numel(), 3), "solar mass weights must have shape (NE, 3)")
    assert_true(torch.allclose(weights.sum(dim=-1), torch.ones(ENERGIES_MEV.numel(), dtype=DTYPE), atol=1.0e-10), "solar mass weights must sum to one")
    assert_true(torch.isfinite(weights).all().item(), "solar mass weights must be finite")


def test_initial_neutrino_antineutrino_match_for_same_mass_weights():
    data = compute_solar_earth_case()
    diff = torch.max(torch.abs(data["initial_prob_nu"] - data["initial_prob_antinu"])).item()

    print("\nInitial flavour probabilities before earth:")
    print("neutrino    :", data["initial_prob_nu"])
    print("antineutrino:", data["initial_prob_antinu"])
    print(f"max |nu - antinu| before earth = {diff:.6e}")

    assert_true(diff < 1.0e-14, "Initial nu/antinu flavour probabilities match for identical mass weights")


def test_final_probabilities_are_normalized_for_each_eta():
    data = compute_solar_earth_case()

    for label, probs in [("neutrino", data["final_prob_nu"]), ("antineutrino", data["final_prob_antinu"])]:
        sums = probs.sum(dim=-1)
        max_error = torch.max(torch.abs(sums - 1.0)).item()

        print(f"\nFinal probability sums for {label}:")
        print(sums)
        print(f"max normalization error = {max_error:.6e}")

        assert_true(probs.shape == (ENERGIES_MEV.numel(), ETA_VALUES.numel(), 3), f"{label} final probabilities must have shape (NE, Neta, 3)")
        assert_true(max_error < 1.0e-8, f"{label} final probabilities must sum to one")
        assert_true(torch.isfinite(probs).all().item(), f"{label} final probabilities must be finite")


def test_neutrino_antineutrino_earth_difference_is_finite():
    data = compute_solar_earth_case()
    diff = torch.abs(data["final_prob_nu"] - data["final_prob_antinu"])

    print("\nNeutrino vs antineutrino after earth:")
    print("max absolute probability difference:", f"{torch.max(diff).item():.6e}")
    print("mean absolute probability difference:", f"{torch.mean(diff).item():.6e}")

    assert_true(torch.isfinite(diff).all().item(), "Neutrino-antineutrino earth differences must be finite")


# ============================================================
# Visualization
# ============================================================

def plot_initial_flux_comparison(savefig=False):
    data = compute_solar_earth_case()
    E = ENERGIES_MEV.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    for idx, label in enumerate(FLAVOUR_LABELS):
        ax.loglog(E, data["initial_flux_nu"][:, idx].detach().cpu().numpy(), marker="o", label=f"nu {label}")
        ax.loglog(E, data["initial_flux_antinu"][:, idx].detach().cpu().numpy(), marker="x", linestyle="--", label=f"antinu {label}")

    ax.set_xlabel("Energy [MeV]")
    ax.set_ylabel("Arbitrary flux units")
    ax.set_title(f"Initial flavour spectra from solar {SOURCE} production")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()

    path = OUTPUT_DIR / "earth_solar_initial_nu_antinu_spectra.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"\nSaved plot: {path}")
    plt.show()


def plot_final_flux_by_eta(savefig=False):
    data = compute_solar_earth_case()
    E = ENERGIES_MEV.detach().cpu().numpy()

    fig, axes = plt.subplots(2, ETA_VALUES.numel(), figsize=(14.0, 7.0), sharex=True, sharey=True)

    for eta_index, eta_value in enumerate(ETA_VALUES.detach().cpu().numpy()):
        for idx, label in enumerate(FLAVOUR_LABELS):
            axes[0, eta_index].loglog(E, data["final_flux_nu"][:, eta_index, idx].detach().cpu().numpy(), marker="o", label=label)
            axes[1, eta_index].loglog(E, data["final_flux_antinu"][:, eta_index, idx].detach().cpu().numpy(), marker="x", linestyle="--", label=label)

        axes[0, eta_index].set_title(rf"nu, $\eta={eta_value / np.pi:.2f}\pi$")
        axes[1, eta_index].set_title(rf"antinu, $\eta={eta_value / np.pi:.2f}\pi$")

        for row in range(2):
            axes[row, eta_index].grid(True, which="both", alpha=0.3)
            axes[row, eta_index].set_xlabel("Energy [MeV]")

    axes[0, 0].set_ylabel("Final flux")
    axes[1, 0].set_ylabel("Final flux")
    axes[0, -1].legend(fontsize=8)
    fig.suptitle(f"Final solar+earth flavour spectra from {SOURCE} source")
    fig.tight_layout()

    path = OUTPUT_DIR / "earth_solar_final_spectra_by_eta.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"Saved plot: {path}")
    plt.show()


def plot_neutrino_antineutrino_difference(savefig=False):
    data = compute_solar_earth_case()
    E = ENERGIES_MEV.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharey=True)

    for flavour_index, ax in enumerate(axes):
        for eta_index, eta_value in enumerate(ETA_VALUES.detach().cpu().numpy()):
            diff = (
                data["final_prob_nu"][:, eta_index, flavour_index]
                - data["final_prob_antinu"][:, eta_index, flavour_index]
            )
            ax.semilogx(E, diff.detach().cpu().numpy(), marker="o", label=rf"$\eta={eta_value / np.pi:.2f}\pi$")

        ax.axhline(0.0, color="black", lw=1.0)
        ax.set_xlabel("Energy [MeV]")
        ax.set_title(FLAVOUR_LABELS[flavour_index])
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(r"$P_\nu - P_{\bar\nu}$ after earth")
    axes[-1].legend(fontsize=8)
    fig.suptitle("earth matter-sign effect: neutrinos vs antineutrinos")
    fig.tight_layout()

    path = OUTPUT_DIR / "earth_solar_nu_antinu_probability_difference.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"Saved plot: {path}")
    plt.show()


def test_visualization_outputs(savefig=False):
    plot_initial_flux_comparison(savefig=savefig)
    plot_final_flux_by_eta(savefig=savefig)
    plot_neutrino_antineutrino_difference(savefig=savefig)

    expected = [
        OUTPUT_DIR / "earth_solar_initial_nu_antinu_spectra.png",
        OUTPUT_DIR / "earth_solar_final_spectra_by_eta.png",
        OUTPUT_DIR / "earth_solar_nu_antinu_probability_difference.png",
    ]

    for path in expected:
        if savefig:
            assert_true(path.is_file(), f"Plot was not created: {path}")


if __name__ == "__main__":
    tests = [
        test_solar_mass_weights_are_normalized,
        test_initial_neutrino_antineutrino_match_for_same_mass_weights,
        test_final_probabilities_are_normalized_for_each_eta,
        test_neutrino_antineutrino_earth_difference_is_finite,
        test_visualization_outputs,
    ]

    run_test_suite(tests, suite_name="solar + earth PROPAGATION tests", verbose_traceback=True)

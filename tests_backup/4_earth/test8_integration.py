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
Spyder-friendly tests for tpeanuts.earth.integration.

This script validates exposure-integrated earth probabilities and compares the
current torch implementation with the original NumPy peanuts implementation in
tpeanuts/peanuts.

The test seeds a temporary exposure cache with the legacy NadirExposure table.
Then pearth_integrated reads that same cache, so the comparison isolates
the probability integration itself.

It also builds simple initial flavour energy spectra and shows the final
flavour spectra after earth propagation.

Run directly in Spyder or from a terminal:

    python tpeanuts/tests/earth/test8_integration.py
"""



from __future__ import annotations

import importlib
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# ============================================================
# Import bootstrap
# ============================================================

THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parents[3]




from tpeanuts.core.pmns import PMNS  # noqa: E402
from tpeanuts.earth.integration import (  # noqa: E402
    _load_exposure_table,
    _prepare_energy_grid,
    pearth_integrated,
)
from tpeanuts.io.io_earth import (  # noqa: E402
    load_earth_density_from_csv,
    save_nadir_exposure_to_cache,
)
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

TORCH_DENSITY_FILE = PACKAGE_DIR / "data" / "density" / "earth_density.csv"
LEGACY_DENSITY_FILE = PACKAGE_DIR / "data" / "peanuts" / "earth_density.csv"
LEGACY_DIR = PACKAGE_DIR / "peanuts"

TESTS_DIR = THIS_FILE.parents[1]
DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_TEST_ROOT = Path(OUTPUT_ROOT / "test")
OUTPUT_DIR = Path(OUTPUT_TEST_ROOT / "earth" / Path(__file__).stem)
CACHE_DIR = OUTPUT_DIR / "cache"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

DM21_EV2 = 7.42e-5
DM3L_EV2 = 2.517e-3
DEPTH_M = 0.0

THETA12 = 0.59
THETA13 = 0.15
THETA23 = 0.78
DELTA_CP = 1.20

LATITUDE_RAD = 0.72
D1 = 0.0
D2 = 365.0
NS = 11

ENERGIES_MEV = torch.tensor([500.0, 1000.0, 3000.0, 10000.0], device=DEVICE, dtype=DTYPE)

FLAVOUR_NAMES = ["nu_e", "nu_mu", "nu_tau"]
FLAVOUR_LABELS = [r"$\nu_e$", r"$\nu_\mu$", r"$\nu_\tau$"]

ABS_TOL = 5.0e-10
REL_TOL = 5.0e-9
SUM_TOL_RAW = 2.0e-3

torch.set_printoptions(precision=10, sci_mode=True, linewidth=160)
np.set_printoptions(precision=10, suppress=False, linewidth=160)


# ============================================================
# Fixtures
# ============================================================

def load_legacy_modules():
    assert_true(LEGACY_DIR.is_dir(), f"Legacy peanuts directory not found: {LEGACY_DIR}")
    legacy_pmns = importlib.import_module("peanuts.pmns")
    legacy_earth = importlib.import_module("peanuts.earth")
    legacy_time_average = importlib.import_module("peanuts.time_average")
    return legacy_pmns, legacy_earth, legacy_time_average


def build_torch_objects():
    density = load_earth_density_from_csv(
        str(TORCH_DENSITY_FILE),
        tabulated_density=False,
        device=DEVICE,
        dtype=DTYPE,
    )

    pmns = PMNS(
        THETA12,
        THETA13,
        THETA23,
        DELTA_CP,
        device=DEVICE,
        real_dtype=DTYPE,
    )

    return density, pmns


def build_legacy_objects():
    legacy_pmns, legacy_earth, legacy_time_average = load_legacy_modules()

    density = legacy_earth.earthdensity(str(LEGACY_DENSITY_FILE))
    pmns = legacy_pmns.PMNS(THETA12, THETA13, THETA23, DELTA_CP)

    return density, pmns, legacy_earth, legacy_time_average


def flavour_state_torch(index: int) -> torch.Tensor:
    state = torch.zeros(3, device=DEVICE, dtype=CDTYPE)
    state[index] = 1.0 + 0.0j
    return state


def flavour_state_numpy(index: int) -> np.ndarray:
    state = np.zeros(3, dtype=np.complex128)
    state[index] = 1.0 + 0.0j
    return state


_EXPOSURE_CACHE = None


def seed_legacy_exposure_cache():
    global _EXPOSURE_CACHE

    if _EXPOSURE_CACHE is not None:
        return _EXPOSURE_CACHE

    _, _, legacy_time_average = load_legacy_modules()

    exposure_np = legacy_time_average.NadirExposure(
        lam=LATITUDE_RAD,
        d1=D1,
        d2=D2,
        ns=NS,
        normalized=False,
        from_file=None,
        angle="Nadir",
        daynight=None,
    )

    eta = torch.tensor(exposure_np[:, 0], device=DEVICE, dtype=DTYPE)
    exposure = torch.tensor(exposure_np[:, 1], device=DEVICE, dtype=DTYPE)

    path = save_nadir_exposure_to_cache(
        eta,
        exposure,
        lam_rad=LATITUDE_RAD,
        d1=D1,
        d2=D2,
        ns=NS,
        daynight=None,
        cache_dir=str(CACHE_DIR),
    )

    print("\nSeeded legacy exposure cache:")
    print("path       :", path)
    print("eta        :", eta)
    print("exposure   :", exposure)
    print("trapz area :", torch.trapz(exposure, x=eta).item())

    _EXPOSURE_CACHE = {
        "eta": eta,
        "exposure": exposure,
        "path": path,
    }

    return _EXPOSURE_CACHE


def legacy_integrated_probability(flavour_index: int, energy_mev: float) -> np.ndarray:
    legacy_density, legacy_pmns, legacy_earth, _ = build_legacy_objects()
    exposure = seed_legacy_exposure_cache()

    state = flavour_state_numpy(flavour_index)
    eta_np = exposure["eta"].detach().cpu().numpy()
    w_np = exposure["exposure"].detach().cpu().numpy()

    out = np.zeros(3, dtype=float)
    deta = np.pi / NS

    for eta_value, weight in zip(eta_np, w_np):
        out += (
            legacy_earth.Pearth(
                state,
                legacy_density,
                legacy_pmns,
                DM21_EV2,
                DM3L_EV2,
                float(energy_mev),
                float(eta_value),
                DEPTH_M,
                mode="analytical",
                massbasis=False,
                antinu=False,
            )
            * float(weight)
            * deta
        )

    return out


def torch_integrated_probability(flavour_index: int, energies: torch.Tensor) -> torch.Tensor:
    torch_density, torch_pmns = build_torch_objects()
    seed_legacy_exposure_cache()

    return pearth_integrated(
        nustate=flavour_state_torch(flavour_index),
        density=torch_density,
        pmns=torch_pmns,
        dm21_eV2=DM21_EV2,
        dm3l_eV2=DM3L_EV2,
        E_MeV=energies,
        depth_m=DEPTH_M,
        method="analytical",
        antinu=False,
        massbasis=False,
        lam_rad=LATITUDE_RAD,
        d1=D1,
        d2=D2,
        ns=NS,
        cache_dir=str(CACHE_DIR),
        normalized_exposure=False,
        from_file=None,
        device=DEVICE,
        dtype=DTYPE,
        chunk_eta=4,
        reunitarize=False,
    )


_COMPARISON_CACHE = None


def compute_comparison():
    global _COMPARISON_CACHE

    if _COMPARISON_CACHE is not None:
        return _COMPARISON_CACHE

    torch_probs = np.zeros((3, ENERGIES_MEV.numel(), 3), dtype=float)
    legacy_probs = np.zeros_like(torch_probs)

    print("\nComputing integrated probabilities:")
    print("Torch density file  :", TORCH_DENSITY_FILE)
    print("Legacy density file :", LEGACY_DENSITY_FILE)
    print("Exposure cache dir  :", CACHE_DIR)
    print("Energies [MeV]      :", ENERGIES_MEV)

    for initial_index, initial_name in enumerate(FLAVOUR_NAMES):
        P_torch = torch_integrated_probability(initial_index, ENERGIES_MEV)
        torch_probs[initial_index] = P_torch.detach().cpu().numpy()

        print(f"\nInitial flavour: {initial_name}")

        for energy_index, energy in enumerate(ENERGIES_MEV.detach().cpu().numpy()):
            P_legacy = legacy_integrated_probability(initial_index, float(energy))
            legacy_probs[initial_index, energy_index] = P_legacy

            diff = np.max(np.abs(torch_probs[initial_index, energy_index] - P_legacy))
            print(
                f"  E={energy:8.1f} MeV | "
                f"legacy={P_legacy} | "
                f"torch={torch_probs[initial_index, energy_index]} | "
                f"max_abs_diff={diff:.3e}"
            )

    abs_diff = np.abs(torch_probs - legacy_probs)
    rel_diff = abs_diff / np.maximum(np.abs(legacy_probs), 1.0e-15)

    metrics = {
        "max_abs": float(np.max(abs_diff)),
        "mean_abs": float(np.mean(abs_diff)),
        "rms_abs": float(np.sqrt(np.mean(abs_diff**2))),
        "max_rel": float(np.max(rel_diff)),
        "mean_rel": float(np.mean(rel_diff)),
        "rms_rel": float(np.sqrt(np.mean(rel_diff**2))),
        "torch_sum_error": float(np.max(np.abs(np.sum(torch_probs, axis=-1) - np.sum(legacy_probs, axis=-1)))),
    }

    _COMPARISON_CACHE = {
        "torch": torch_probs,
        "legacy": legacy_probs,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "metrics": metrics,
    }

    return _COMPARISON_CACHE


def initial_spectra(energies: torch.Tensor) -> torch.Tensor:
    x = energies / 1000.0
    phi_e = 1.00 * x ** -2.20 * torch.exp(-energies / 13000.0)
    phi_mu = 0.55 * x ** -2.05 * torch.exp(-energies / 16000.0)
    phi_tau = 0.12 * x ** -1.80 * torch.exp(-energies / 20000.0)
    return torch.stack([phi_e, phi_mu, phi_tau], dim=-1)


def final_spectra_from_probabilities(probabilities: np.ndarray, spectra_initial: np.ndarray) -> np.ndarray:
    row_sums = np.sum(probabilities, axis=-1, keepdims=True)
    probabilities = probabilities / np.maximum(row_sums, 1.0e-30)

    final = np.zeros_like(spectra_initial)

    for initial_index in range(3):
        for energy_index in range(spectra_initial.shape[0]):
            final[energy_index] += spectra_initial[energy_index, initial_index] * probabilities[initial_index, energy_index]

    return final


# ============================================================
# tests
# ============================================================

def test_prepare_energy_grid():
    scalar, squeeze_scalar = _prepare_energy_grid(
        1000.0,
        device=DEVICE,
        dtype=DTYPE,
    )

    vector, squeeze_vector = _prepare_energy_grid(
        ENERGIES_MEV,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("\n_prepare_energy_grid:")
    print("scalar:", scalar, "squeeze:", squeeze_scalar)
    print("vector:", vector, "squeeze:", squeeze_vector)

    assert_true(scalar.shape == (1,), "Scalar energy must become one-element grid")
    assert_true(squeeze_scalar is True, "Scalar energy must request squeeze")
    assert_true(vector.shape == ENERGIES_MEV.shape, "Vector energy shape must be preserved")
    assert_true(squeeze_vector is False, "Vector energy must not request squeeze")


def test_load_seeded_exposure_table():
    seed_legacy_exposure_cache()

    table = _load_exposure_table(
        from_file=None,
        lam_rad=LATITUDE_RAD,
        d1=D1,
        d2=D2,
        ns=NS,
        cache_dir=str(CACHE_DIR),
        normalized_exposure=False,
        angle="Nadir",
        daynight=None,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("\nLoaded seeded exposure table:")
    print("eta shape     :", tuple(table.eta.shape))
    print("exposure shape:", tuple(table.exposure.shape))
    print("trapz area    :", torch.trapz(table.exposure, x=table.eta).item())

    assert_true(table.eta.shape == (NS,), "Loaded eta table must have NS samples")
    assert_true(table.exposure.shape == (NS,), "Loaded exposure table must have NS samples")
    assert_true(torch.isfinite(table.exposure).all().item(), "Loaded exposure must be finite")


def test_integrated_probabilities_match_legacy():
    comparison = compute_comparison()
    metrics = comparison["metrics"]

    print("\nPrecision report:")
    print(f"  max abs diff  : {metrics['max_abs']:.6e}")
    print(f"  mean abs diff : {metrics['mean_abs']:.6e}")
    print(f"  RMS abs diff  : {metrics['rms_abs']:.6e}")
    print(f"  max rel diff  : {metrics['max_rel']:.6e}")
    print(f"  mean rel diff : {metrics['mean_rel']:.6e}")
    print(f"  RMS rel diff  : {metrics['rms_rel']:.6e}")
    print(f"  sum diff      : {metrics['torch_sum_error']:.6e}")
    print(f"  abs tolerance : {ABS_TOL:.6e}")
    print(f"  rel tolerance : {REL_TOL:.6e}")

    assert_true(metrics["max_abs"] < ABS_TOL, "Integrated torch probabilities must match legacy absolute precision")
    assert_true(metrics["max_rel"] < REL_TOL, "Integrated torch probabilities must match legacy relative precision")
    assert_true(metrics["torch_sum_error"] < SUM_TOL_RAW, "Integrated raw normalization drift must match legacy")


def test_scalar_energy_output_shape():
    P = torch_integrated_probability(
        0,
        torch.tensor(1000.0, device=DEVICE, dtype=DTYPE),
    )

    print("\nScalar integrated probability:")
    print(P)

    assert_true(P.shape == (3,), "Scalar integrated output must have shape (3,)")
    assert_true(torch.isfinite(P).all().item(), "Scalar integrated output must be finite")


def test_numerical_method_runs():
    torch_density, torch_pmns = build_torch_objects()
    seed_legacy_exposure_cache()

    P = pearth_integrated(
        nustate=flavour_state_torch(0),
        density=torch_density,
        pmns=torch_pmns,
        dm21_eV2=DM21_EV2,
        dm3l_eV2=DM3L_EV2,
        E_MeV=torch.tensor(1000.0, device=DEVICE, dtype=DTYPE),
        depth_m=DEPTH_M,
        method="numerical",
        antinu=False,
        massbasis=False,
        lam_rad=LATITUDE_RAD,
        d1=D1,
        d2=D2,
        ns=NS,
        cache_dir=str(CACHE_DIR),
        normalized_exposure=False,
        device=DEVICE,
        dtype=DTYPE,
        chunk_eta=4,
        nsteps=20,
        ode_method="midpoint",
    )

    print("\nNumerical-method integrated probability:")
    print(P)

    assert_true(P.shape == (3,), "Numerical integrated output must have shape (3,)")
    assert_true(torch.isfinite(P).all().item(), "Numerical integrated output must be finite")


# ============================================================
# Visualization
# ============================================================

def plot_integrated_probability_spectra(savefig=False):
    comparison = compute_comparison()
    torch_probs = comparison["torch"]
    legacy_probs = comparison["legacy"]
    energies = ENERGIES_MEV.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharey=True)

    for initial_index, ax in enumerate(axes):
        for final_index, label in enumerate(FLAVOUR_LABELS):
            ax.plot(energies, legacy_probs[initial_index, :, final_index], marker="o", label=f"legacy {label}")
            ax.plot(energies, torch_probs[initial_index, :, final_index], marker="x", linestyle="--", label=f"torch {label}")

        ax.set_xscale("log")
        ax.set_title(f"Initial {FLAVOUR_LABELS[initial_index]}")
        ax.set_xlabel("Energy [MeV]")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Integrated final flavour probability")
    axes[-1].legend(fontsize=7, ncol=2)
    fig.suptitle("Integrated earth probabilities: torch vs legacy")
    fig.tight_layout()

    path = OUTPUT_DIR / "earth_integration_probability_spectra.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"\nSaved plot: {path}")
    plt.show()


def plot_initial_and_final_flux_spectra(savefig=False):
    comparison = compute_comparison()
    energies_t = ENERGIES_MEV
    energies = energies_t.detach().cpu().numpy()

    phi_initial_t = initial_spectra(energies_t)
    phi_initial = phi_initial_t.detach().cpu().numpy()

    phi_final_torch = final_spectra_from_probabilities(comparison["torch"], phi_initial)
    phi_final_legacy = final_spectra_from_probabilities(comparison["legacy"], phi_initial)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)

    for idx, label in enumerate(FLAVOUR_LABELS):
        axes[0].loglog(energies, phi_initial[:, idx], marker="o", label=label)
        axes[1].loglog(energies, phi_final_legacy[:, idx], marker="o", label=f"legacy {label}")
        axes[1].loglog(energies, phi_final_torch[:, idx], marker="x", linestyle="--", label=f"torch {label}")

    axes[0].set_title("Initial flavour energy spectra")
    axes[1].set_title("Final flavour energy spectra after earth")

    for ax in axes:
        ax.set_xlabel("Energy [MeV]")
        ax.set_ylabel("Arbitrary flux units")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()

    path = OUTPUT_DIR / "earth_integration_initial_final_spectra.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"Saved plot: {path}")
    plt.show()


def plot_integration_error_heatmap(savefig=False):
    comparison = compute_comparison()
    max_error = np.max(comparison["abs_diff"], axis=-1)
    energies = ENERGIES_MEV.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    image = ax.imshow(
        max_error,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[float(energies[0]), float(energies[-1]), -0.5, 2.5],
        cmap="magma",
    )
    ax.set_xscale("log")
    ax.set_yticks(range(3))
    ax.set_yticklabels(FLAVOUR_LABELS)
    ax.set_xlabel("Energy [MeV]")
    ax.set_ylabel("Initial flavour")
    ax.set_title("Max final-flavour absolute integration error")
    fig.colorbar(image, ax=ax, label="max absolute difference")
    fig.tight_layout()

    path = OUTPUT_DIR / "earth_integration_error_heatmap.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"Saved plot: {path}")
    plt.show()


def test_visualization_outputs(savefig=False):
    plot_integrated_probability_spectra(savefig=savefig)
    plot_initial_and_final_flux_spectra(savefig=savefig)
    plot_integration_error_heatmap(savefig=savefig)

    expected_files = [
        OUTPUT_DIR / "earth_integration_probability_spectra.png",
        OUTPUT_DIR / "earth_integration_initial_final_spectra.png",
        OUTPUT_DIR / "earth_integration_error_heatmap.png",
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
        test_prepare_energy_grid,
        test_load_seeded_exposure_table,
        test_integrated_probabilities_match_legacy,
        test_scalar_energy_output_shape,
        test_numerical_method_runs,
        test_visualization_outputs,
    ]

    run_test_suite(
        tests,
        suite_name="earth INTEGRATION tests",
        verbose_traceback=True,
    )

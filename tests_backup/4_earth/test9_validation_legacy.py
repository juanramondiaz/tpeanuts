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
Spyder-friendly validation against the original NumPy peanuts implementation.

This script compares the current torch implementation with the legacy NumPy
implementation stored in:

    tpeanuts/peanuts

The legacy folder is treated as read-only.

The validation checks flavour-basis probabilities for initial:

    nu_e, nu_mu, nu_tau

at several nadir angles that cross the earth. It reports absolute and relative
precision metrics and creates visual comparison plots.

Run directly in Spyder or from a terminal:

    python tpeanuts/tests/earth/test10_validation_legacy.py
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

TORCH_DENSITY_FILE = PACKAGE_DIR / "data" / "density" / "earth_density.csv"
LEGACY_DENSITY_FILE = PACKAGE_DIR / "data" / "peanuts" / "earth_density.csv"
LEGACY_DIR = PACKAGE_DIR / "peanuts"
TESTS_DIR = THIS_FILE.parents[1]
DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_TEST_ROOT = Path(OUTPUT_ROOT / "test")
OUTPUT_DIR = Path(OUTPUT_TEST_ROOT / "earth" / Path(__file__).stem)

os.makedirs(OUTPUT_DIR, exist_ok=True)

DM21_EV2 = 7.42e-5
DM3L_EV2 = 2.517e-3
DEPTH_M = 0.0

THETA12 = 0.59
THETA13 = 0.15
THETA23 = 0.78
DELTA_CP = 1.20

ENERGY_MEV = 1000.0
ETA_VALUES = np.array([0.10, 0.35, 0.60, 0.85, 1.10, 1.35], dtype=float)

FLAVOUR_NAMES = ["nu_e", "nu_mu", "nu_tau"]
FLAVOUR_LABELS = [r"$\nu_e$", r"$\nu_\mu$", r"$\nu_\tau$"]

ABS_TOL = 2.0e-6
REL_TOL = 2.0e-5
SUM_TOL_RAW = 1.0e-3

torch.set_printoptions(
    precision=10,
    sci_mode=True,
    linewidth=160,
)
np.set_printoptions(
    precision=10,
    suppress=False,
    linewidth=160,
)


# ============================================================
# Legacy import and fixtures
# ============================================================

def install_legacy_peanuts_alias():
    assert_true(LEGACY_DIR.is_dir(), f"Legacy directory not found: {LEGACY_DIR}")


def load_legacy_modules():
    install_legacy_peanuts_alias()

    legacy_pmns = importlib.import_module("peanuts.pmns")
    legacy_earth = importlib.import_module("peanuts.earth")

    return legacy_pmns, legacy_earth


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
    legacy_pmns, legacy_earth = load_legacy_modules()

    density = legacy_earth.earthdensity(str(LEGACY_DENSITY_FILE))
    pmns = legacy_pmns.PMNS(
        THETA12,
        THETA13,
        THETA23,
        DELTA_CP,
    )

    return density, pmns, legacy_earth


def flavour_state_numpy(index: int) -> np.ndarray:
    state = np.zeros(3, dtype=np.complex128)
    state[index] = 1.0 + 0.0j
    return state


def flavour_state_torch(index: int) -> torch.Tensor:
    state = torch.zeros(3, device=DEVICE, dtype=CDTYPE)
    state[index] = 1.0 + 0.0j
    return state


# ============================================================
# Comparison core
# ============================================================

_COMPARISON_CACHE = None


def compute_comparison_table():
    global _COMPARISON_CACHE

    if _COMPARISON_CACHE is not None:
        return _COMPARISON_CACHE

    torch_density, torch_pmns = build_torch_objects()
    legacy_density, legacy_pmns, legacy_earth = build_legacy_objects()

    torch_probs = np.zeros((len(FLAVOUR_NAMES), len(ETA_VALUES), 3), dtype=float)
    legacy_probs = np.zeros_like(torch_probs)

    print("\nComputing legacy-vs-torch earth probabilities:")
    print(f"Energy [MeV]  : {ENERGY_MEV}")
    print(f"Depth [m]     : {DEPTH_M}")
    print(f"Torch density file  : {TORCH_DENSITY_FILE}")
    print(f"Legacy density file : {LEGACY_DENSITY_FILE}")
    print(f"Legacy folder : {LEGACY_DIR}")

    for flavour_index, flavour_name in enumerate(FLAVOUR_NAMES):
        state_np = flavour_state_numpy(flavour_index)
        state_t = flavour_state_torch(flavour_index)

        print(f"\nInitial flavour: {flavour_name}")

        for eta_index, eta_value in enumerate(ETA_VALUES):
            legacy_p = legacy_earth.Pearth(
                state_np,
                legacy_density,
                legacy_pmns,
                DM21_EV2,
                DM3L_EV2,
                ENERGY_MEV,
                float(eta_value),
                DEPTH_M,
                mode="analytical",
                massbasis=False,
                antinu=False,
            )

            torch_p = pearth(
                state_t,
                torch_density,
                torch_pmns,
                DM21_EV2,
                DM3L_EV2,
                torch.tensor(ENERGY_MEV, device=DEVICE, dtype=DTYPE),
                torch.tensor(float(eta_value), device=DEVICE, dtype=DTYPE),
                DEPTH_M,
                method="analytical",
                massbasis=False,
                antinu=False,
                reunitarize=False,
            )

            legacy_probs[flavour_index, eta_index, :] = np.asarray(legacy_p, dtype=float)
            torch_probs[flavour_index, eta_index, :] = torch_p.detach().cpu().numpy()

            diff = np.max(np.abs(torch_probs[flavour_index, eta_index] - legacy_probs[flavour_index, eta_index]))

            print(
                f"  eta={eta_value:.3f} rad | "
                f"legacy={legacy_probs[flavour_index, eta_index]} | "
                f"torch={torch_probs[flavour_index, eta_index]} | "
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
        "torch_sum_error": float(np.max(np.abs(np.sum(torch_probs, axis=-1) - 1.0))),
        "legacy_sum_error": float(np.max(np.abs(np.sum(legacy_probs, axis=-1) - 1.0))),
    }

    _COMPARISON_CACHE = {
        "torch": torch_probs,
        "legacy": legacy_probs,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "metrics": metrics,
    }

    return _COMPARISON_CACHE


def print_precision_report(comparison):
    metrics = comparison["metrics"]

    print("\nPrecision report:")
    print(f"  max abs diff       : {metrics['max_abs']:.6e}")
    print(f"  mean abs diff      : {metrics['mean_abs']:.6e}")
    print(f"  RMS abs diff       : {metrics['rms_abs']:.6e}")
    print(f"  max rel diff       : {metrics['max_rel']:.6e}")
    print(f"  mean rel diff      : {metrics['mean_rel']:.6e}")
    print(f"  RMS rel diff       : {metrics['rms_rel']:.6e}")
    print(f"  max torch sum err  : {metrics['torch_sum_error']:.6e}")
    print(f"  max legacy sum err : {metrics['legacy_sum_error']:.6e}")
    print(f"  abs tolerance      : {ABS_TOL:.6e}")
    print(f"  rel tolerance      : {REL_TOL:.6e}")
    print(f"  raw sum tolerance  : {SUM_TOL_RAW:.6e}")
    print("  note               : comparison uses raw peanuts evolution without reunitarization")


# ============================================================
# tests
# ============================================================

def test_legacy_import_and_objects():
    legacy_pmns, legacy_earth = load_legacy_modules()
    legacy_density, legacy_pmns_object, _ = build_legacy_objects()

    print("\nLegacy import check:")
    print("legacy pmns module :", legacy_pmns)
    print("legacy earth module:", legacy_earth)
    print("legacy density rj shape:", legacy_density.rj.shape)
    print("legacy PMNS matrix shape:", legacy_pmns_object.pmns.shape)

    assert_true(legacy_density.rj.size > 0, "Legacy density must contain earth shells")
    assert_true(legacy_pmns_object.pmns.shape == (3, 3), "Legacy PMNS matrix must have shape (3, 3)")


def test_legacy_torch_probability_precision():
    comparison = compute_comparison_table()
    print_precision_report(comparison)

    metrics = comparison["metrics"]

    assert_true(metrics["max_abs"] < ABS_TOL, "Torch and legacy probabilities must agree in absolute precision")
    assert_true(metrics["max_rel"] < REL_TOL, "Torch and legacy probabilities must agree in relative precision")
    assert_true(metrics["torch_sum_error"] < SUM_TOL_RAW, "Raw torch probabilities must remain close to normalized")
    assert_true(metrics["legacy_sum_error"] < SUM_TOL_RAW, "Raw legacy probabilities must remain close to normalized")
    assert_true(
        abs(metrics["torch_sum_error"] - metrics["legacy_sum_error"]) < 1.0e-12,
        "Torch and legacy raw normalization drift must match",
    )


def test_each_flavour_angle_probability_is_finite():
    comparison = compute_comparison_table()

    torch_probs = comparison["torch"]
    legacy_probs = comparison["legacy"]

    print("\nFinite and normalized probability checks:")

    for flavour_index, flavour_name in enumerate(FLAVOUR_NAMES):
        for eta_index, eta_value in enumerate(ETA_VALUES):
            torch_p = torch_probs[flavour_index, eta_index]
            legacy_p = legacy_probs[flavour_index, eta_index]

            print(
                f"{flavour_name:6s} eta={eta_value:.3f} | "
                f"sum_torch={np.sum(torch_p):.12f} | "
                f"sum_legacy={np.sum(legacy_p):.12f}"
            )

            assert_true(np.all(np.isfinite(torch_p)), "Torch probabilities must be finite")
            assert_true(np.all(np.isfinite(legacy_p)), "Legacy probabilities must be finite")
            assert_true(np.all(torch_p >= -1.0e-12), "Torch probabilities must be non-negative within tolerance")
            assert_true(np.all(legacy_p >= -1.0e-12), "Legacy probabilities must be non-negative within tolerance")


# ============================================================
# Visualization
# ============================================================

def plot_probability_curves_by_initial_flavour(savefig=False):
    comparison = compute_comparison_table()
    torch_probs = comparison["torch"]
    legacy_probs = comparison["legacy"]

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharey=True)

    for flavour_index, ax in enumerate(axes):
        for final_index, final_label in enumerate(FLAVOUR_LABELS):
            ax.plot(
                ETA_VALUES,
                legacy_probs[flavour_index, :, final_index],
                marker="o",
                linestyle="-",
                label=f"legacy {final_label}",
            )
            ax.plot(
                ETA_VALUES,
                torch_probs[flavour_index, :, final_index],
                marker="x",
                linestyle="--",
                label=f"torch {final_label}",
            )

        ax.set_title(f"Initial {FLAVOUR_LABELS[flavour_index]}")
        ax.set_xlabel(r"Nadir angle $\eta$ [rad]")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Final flavour probability")
    axes[-1].legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle("Legacy NumPy vs current Torch earth probabilities")
    fig.tight_layout()

    path = OUTPUT_DIR / "earth_legacy_validation_probability_curves.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"\nSaved plot: {path}")
    plt.show()


def plot_absolute_error_by_initial_flavour(savefig=False):
    comparison = compute_comparison_table()
    abs_diff = comparison["abs_diff"]

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharey=True)

    for flavour_index, ax in enumerate(axes):
        for final_index, final_label in enumerate(FLAVOUR_LABELS):
            ax.semilogy(
                ETA_VALUES,
                np.clip(abs_diff[flavour_index, :, final_index], 1.0e-30, None),
                marker="o",
                label=final_label,
            )

        ax.axhline(ABS_TOL, color="black", linestyle="--", linewidth=1.0, label="tolerance" if flavour_index == 0 else None)
        ax.set_title(f"Initial {FLAVOUR_LABELS[flavour_index]}")
        ax.set_xlabel(r"Nadir angle $\eta$ [rad]")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Absolute probability difference")
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle("Absolute error: Torch vs legacy")
    fig.tight_layout()

    path = OUTPUT_DIR / "earth_legacy_validation_absolute_error.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"Saved plot: {path}")
    plt.show()


def plot_max_error_heatmap(savefig=False):
    comparison = compute_comparison_table()
    max_error = np.max(comparison["abs_diff"], axis=-1)

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    image = ax.imshow(
        max_error,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[
            float(ETA_VALUES[0]),
            float(ETA_VALUES[-1]),
            -0.5,
            len(FLAVOUR_NAMES) - 0.5,
        ],
        cmap="magma",
    )

    ax.set_yticks(range(len(FLAVOUR_NAMES)))
    ax.set_yticklabels(FLAVOUR_LABELS)
    ax.set_xlabel(r"Nadir angle $\eta$ [rad]")
    ax.set_ylabel("Initial flavour")
    ax.set_title("Maximum final-flavour absolute difference")
    fig.colorbar(image, ax=ax, label="max absolute difference")
    fig.tight_layout()

    path = OUTPUT_DIR / "earth_legacy_validation_max_error_heatmap.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"Saved plot: {path}")
    plt.show()


def test_visualization_outputs(savefig=False):
    plot_probability_curves_by_initial_flavour(savefig=savefig)
    plot_absolute_error_by_initial_flavour(savefig=savefig)
    plot_max_error_heatmap(savefig=savefig)

    expected_files = [
        OUTPUT_DIR / "earth_legacy_validation_probability_curves.png",
        OUTPUT_DIR / "earth_legacy_validation_absolute_error.png",
        OUTPUT_DIR / "earth_legacy_validation_max_error_heatmap.png",
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
        test_legacy_import_and_objects,
        test_legacy_torch_probability_precision,
        test_each_flavour_angle_probability_is_finite,
        test_visualization_outputs,
    ]

    run_test_suite(
        tests,
        suite_name="earth LEGACY VALIDATION tests",
        verbose_traceback=True,
    )

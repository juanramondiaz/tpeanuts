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
Spyder-friendly SNO validation test for the torch-native solar block.

This script follows the solar part of run_SNO_test.py:

* solar electron density and 8B/hep production fractions,
* SNO electron-survival curves,
* distorted 8B and hep spectra,
* direct numerical comparison against the legacy peanuts implementation.
"""



from __future__ import annotations

import os
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parents[3]


from tpeanuts.core.pmns import PMNS
from tpeanuts.solar.probabilities import psolar
from tpeanuts.solar.profiles import load_default_solar_profile
from tpeanuts.solar.validation import legacy_modules
from tpeanuts.util.test_utils import assert_true, run_test_suite

DEVICE = torch.device("cpu")
DTYPE = torch.float64

DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_TEST_ROOT = Path(OUTPUT_ROOT / "test")
OUTPUT_DIR = Path(OUTPUT_TEST_ROOT / "solar" / Path(__file__).stem)
os.makedirs(OUTPUT_DIR, exist_ok=True)

solar_data_DIR = PACKAGE_DIR / "data" / "solar"
LEGACY_data_DIR = PACKAGE_DIR / "data" / "peanuts"

TH12 = np.arctan(sqrt(0.469))
TH13 = np.arcsin(sqrt(0.01))
TH23 = 0.85521
DELTA = 3.4034
DM21 = 7.9e-5
DM3L = 2.46e-3

SOURCES = ("8B", "hep")
FLAVOUR_LABELS = [r"$\nu_e$", r"$\nu_\mu$", r"$\nu_\tau$"]


def build_torch_pmns():
    return PMNS(TH12, TH13, TH23, DELTA, device=DEVICE, real_dtype=DTYPE)


def build_legacy_model_and_pmns():
    legacy_pmns_module, legacy_solar = legacy_modules()
    legacy_model = legacy_solar.SolarModel(
        solar_model_file=str(LEGACY_data_DIR / "nudistr_b16_agss09.dat"),
        flux_file=str(LEGACY_data_DIR / "fluxes_b16.dat"),
        spectrum_files={
            "8B": str(LEGACY_data_DIR / "8B_shape_Ortiz_et_al.csv"),
            "hep": str(LEGACY_data_DIR / "hep_shape.csv"),
            "pp": str(LEGACY_data_DIR / "pp_shape.csv"),
            "17F": str(LEGACY_data_DIR / "f17_shape.csv"),
            "7Beground": str(LEGACY_data_DIR / "be7ground_shape.csv"),
            "7Beexcited": str(LEGACY_data_DIR / "be7excited_shape.csv"),
            "13N": str(LEGACY_data_DIR / "n13_shape.csv"),
            "15O": str(LEGACY_data_DIR / "o15_shape.csv"),
        },
    )
    legacy_pmns = legacy_pmns_module.PMNS(TH12, TH13, TH23, DELTA)

    return legacy_solar, legacy_model, legacy_pmns


def torch_psolar(source, energies):
    profile = load_default_solar_profile(device=DEVICE, dtype=DTYPE)
    energy_t = torch.as_tensor(energies, device=DEVICE, dtype=DTYPE)
    probs = psolar(
        build_torch_pmns(),
        DM21,
        DM3L,
        energy_t,
        profile.radius,
        profile.density,
        profile.production_fraction(source),
    )

    return probs.detach().cpu().numpy()


def legacy_psolar(source, energies):
    legacy_solar, legacy_model, legacy_pmns = build_legacy_model_and_pmns()
    return np.asarray(
        [
            legacy_solar.Psolar(
                legacy_pmns,
                DM21,
                DM3L,
                float(energy),
                legacy_model.radius(),
                legacy_model.density(),
                legacy_model.fraction(source),
            )
            for energy in np.asarray(energies, dtype=float)
        ],
        dtype=float,
    )


def read_sno_file(source, data_dir):
    path = data_dir / f"SNO_{source}.csv"

    if data_dir == solar_data_DIR:
        table = pd.read_csv(path)
    else:
        table = pd.read_csv(path, comment="#", names=["energy", "Pnuenue"])

    table = table.dropna()
    table["energy"] = table["energy"].astype(float)
    table["Pnuenue"] = table["Pnuenue"].astype(float)

    return table


def read_legacy_spectrum(source):
    filename = "8B_shape_Ortiz_et_al.csv" if source == "8B" else "hep_shape.csv"
    table = pd.read_csv(
        LEGACY_data_DIR / filename,
        comment="#",
        names=["Energy", "Spectrum"],
    )
    table = table.dropna()
    table["Energy"] = table["Energy"].astype(float)
    table["Spectrum"] = table["Spectrum"].astype(float)

    return table


def sno_survival_on_grid(source, energies):
    table = read_sno_file(source, solar_data_DIR)
    return np.interp(energies, table.energy.to_numpy(), table.Pnuenue.to_numpy())


def test_sno_input_files_are_consistent():
    print("\nSNO input file consistency:")

    for source in SOURCES:
        torch_sno = read_sno_file(source, solar_data_DIR)
        legacy_sno = read_sno_file(source, LEGACY_data_DIR)
        legacy_on_torch_grid = np.interp(
            torch_sno.energy.to_numpy(),
            legacy_sno.energy.to_numpy(),
            legacy_sno.Pnuenue.to_numpy(),
        )
        max_abs = float(np.max(np.abs(torch_sno.Pnuenue.to_numpy() - legacy_on_torch_grid)))

        print(f"{source}: torch rows={len(torch_sno)}, legacy rows={len(legacy_sno)}, max |delta SNO|={max_abs:.3e}")

        assert_true(len(torch_sno) > 10, f"{source} torch SNO file must contain enough points")
        assert_true(len(legacy_sno) > 10, f"{source} legacy SNO file must contain enough points")
        assert_true(max_abs < 1.0e-12, f"{source} SNO files must agree between data/solar and data/peanuts")


def test_torch_legacy_psolar_precision():
    energies = np.linspace(1.0, 20.0, 39)

    print("\nTorch vs legacy solar probability precision:")
    for source in SOURCES:
        torch_p = torch_psolar(source, energies)
        legacy_p = legacy_psolar(source, energies)
        abs_diff = np.abs(torch_p - legacy_p)
        max_abs = float(np.max(abs_diff))
        rms_abs = float(np.sqrt(np.mean(abs_diff**2)))
        max_norm = float(np.max(np.abs(torch_p.sum(axis=1) - 1.0)))

        print(f"{source}: max |P_torch - P_legacy|={max_abs:.3e}, RMS={rms_abs:.3e}, max normalization error={max_norm:.3e}")

        assert_true(max_abs < 1.0e-10, f"{source} torch probabilities must match legacy")
        assert_true(max_norm < 1.0e-12, f"{source} torch probabilities must sum to one")


def test_sno_reference_curve_agreement():
    energies = np.linspace(1.1, 18.0, 80)

    print("\nTorch electron survival probability compared with SNO reference curves:")
    for source in SOURCES:
        torch_p = torch_psolar(source, energies)
        sno_pee = sno_survival_on_grid(source, energies)
        diff = torch_p[:, 0] - sno_pee
        max_abs = float(np.max(np.abs(diff)))
        rms_abs = float(np.sqrt(np.mean(diff**2)))

        print(f"{source}: max |Pee_torch - Pee_SNO|={max_abs:.3e}, RMS={rms_abs:.3e}")

        assert_true(np.isfinite(max_abs), f"{source} SNO comparison must be finite")
        assert_true(max_abs < 0.12, f"{source} torch Pee should remain close to the SNO reference curve")


def test_distorted_spectrum_precision():
    print("\nDistorted spectrum precision:")

    for source in SOURCES:
        spectrum = read_legacy_spectrum(source)
        energies = spectrum.Energy.to_numpy()
        shape = spectrum.Spectrum.to_numpy()[:, None]
        valid = (energies >= 1.0) & (energies <= 20.0)

        torch_distorted = shape * torch_psolar(source, energies)
        legacy_distorted = shape * legacy_psolar(source, energies)
        diff = np.abs(torch_distorted[valid] - legacy_distorted[valid])
        scale = max(float(np.max(np.abs(legacy_distorted[valid]))), 1.0e-30)
        max_abs = float(np.max(diff))
        max_rel = max_abs / scale

        print(f"{source}: max |distorted_torch - distorted_legacy|={max_abs:.3e}, max relative={max_rel:.3e}")

        assert_true(max_abs < 1.0e-10, f"{source} distorted spectrum must match legacy")


def plot_density_and_fractions(savefig=False):
    profile = load_default_solar_profile(device=DEVICE, dtype=DTYPE)
    _, legacy_model, _ = build_legacy_model_and_pmns()

    radius = profile.radius.detach().cpu().numpy()
    density = profile.density.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(radius, density, label="Torch B16")
    axes[0].plot(legacy_model.radius(), legacy_model.density(), "--", label="Legacy B16")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("solar radius fraction")
    axes[0].set_ylabel(r"$n_e(r)$ [mol/cm$^3$]")
    axes[0].set_title("solar electron density")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for source in SOURCES:
        axes[1].plot(radius, profile.production_fraction(source).detach().cpu().numpy(), label=f"Torch {source}")
        axes[1].plot(legacy_model.radius(), legacy_model.fraction(source), "--", label=f"Legacy {source}")

    axes[1].set_xlabel("solar radius fraction")
    axes[1].set_ylabel("Production fraction")
    axes[1].set_title("solar production regions")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    path = OUTPUT_DIR / "solar_sno_density_and_fractions.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"Saved plot: {path}")
    plt.show()


def plot_probabilities(source, savefig=False):
    energies = np.arange(1.0, 20.0, 0.1)
    torch_p = torch_psolar(source, energies)
    legacy_p = legacy_psolar(source, energies)
    sno = read_sno_file(source, solar_data_DIR)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    for idx, label in enumerate(FLAVOUR_LABELS):
        ax.plot(energies, torch_p[:, idx], label=f"Torch {label}")
        ax.plot(energies, legacy_p[:, idx], "--", linewidth=1.0, label=f"Legacy {label}")

    ax.plot(sno.energy, sno.Pnuenue, "k:", linewidth=2.0, label=r"SNO $\nu_e \rightarrow \nu_e$")
    ax.set_xlabel("Energy [MeV]")
    ax.set_ylabel(r"$P(\nu_e \rightarrow \nu_\alpha)$")
    ax.set_title(f"{source} solar neutrino probabilities")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)

    fig.tight_layout()
    path = OUTPUT_DIR / f"solar_sno_probabilities_{source}.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"Saved plot: {path}")
    plt.show()


def plot_distorted_spectrum(source, savefig=False):
    spectrum = read_legacy_spectrum(source)
    energies = spectrum.Energy.to_numpy()
    shape = spectrum.Spectrum.to_numpy()

    torch_p = torch_psolar(source, energies)
    legacy_p = legacy_psolar(source, energies)
    sno_pee = sno_survival_on_grid(source, energies)

    torch_distorted = shape[:, None] * torch_p
    legacy_distorted = shape[:, None] * legacy_p
    sno_distorted = shape * sno_pee

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    for idx, label in enumerate(FLAVOUR_LABELS):
        ax.plot(energies, torch_distorted[:, idx], label=f"Torch {label}")
        ax.plot(energies, legacy_distorted[:, idx], "--", linewidth=1.0, label=f"Legacy {label}")

    ax.plot(energies, sno_distorted, "k:", linewidth=2.0, label=r"SNO $\nu_e$")
    ax.plot(energies, shape, color="0.35", linestyle="-.", label=r"Undistorted $\nu_e$")
    ax.set_xlabel("Energy [MeV]")
    ax.set_ylabel(f"{source} spectral shape")
    ax.set_title(f"{source} distorted solar spectrum")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)

    fig.tight_layout()
    path = OUTPUT_DIR / f"solar_sno_distorted_spectrum_{source}.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"Saved plot: {path}")
    plt.show()


def test_visualization_outputs(savefig=False):
    print("\nGenerating SNO validation plots:")
    plot_density_and_fractions(savefig=savefig)

    expected_paths = [OUTPUT_DIR / "solar_sno_density_and_fractions.png"]
    for source in SOURCES:
        plot_probabilities(source, savefig=savefig)
        plot_distorted_spectrum(source, savefig=savefig)
        expected_paths.extend(
            [
                OUTPUT_DIR / f"solar_sno_probabilities_{source}.png",
                OUTPUT_DIR / f"solar_sno_distorted_spectrum_{source}.png",
            ]
        )

    for path in expected_paths:
        if savefig:
            assert_true(path.is_file(), f"Plot was not created: {path}")


if __name__ == "__main__":
    tests = [
        test_sno_input_files_are_consistent,
        test_torch_legacy_psolar_precision,
        test_sno_reference_curve_agreement,
        test_distorted_spectrum_precision,
        test_visualization_outputs,
    ]

    run_test_suite(tests, suite_name="solar SNO TORCH VS LEGACY tests", verbose_traceback=True)

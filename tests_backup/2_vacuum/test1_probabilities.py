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
Spyder-friendly tests for tpeanuts.vacuum.probabilities.

The tests print intermediate numerical diagnostics and generate visual checks
for vacuum oscillations. Plots are saved under OUTPUT_ROOT/test/vacuum and every
plot ends with plt.show() so it can be inspected interactively from Spyder.
"""



from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch

THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parents[3]


from tpeanuts.core.pmns import PMNS
from tpeanuts.util.test_utils import assert_true, run_test_suite
from tpeanuts.vacuum.probabilities import (
    pvacuum,
    vacuum_evolved_state,
    vacuum_evolutor,
    vacuum_probability_matrix,
)

DEVICE = torch.device("cpu")
DTYPE = torch.float64
DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_TEST_ROOT = Path(OUTPUT_ROOT / "test")
OUTPUT_DIR = Path(OUTPUT_TEST_ROOT / "vacuum" / Path(__file__).stem)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DM21 = 7.42e-5
DM3L = 2.517e-3
FLAVOURS = [r"$\nu_e$", r"$\nu_\mu$", r"$\nu_\tau$"]


def build_pmns(delta=1.20):
    return PMNS(0.59, 0.15, 0.78, delta, device=DEVICE, real_dtype=DTYPE)


def _try_legacy():
    try:
        from peanuts.pmns import PMNS as LegacyPMNS
        from peanuts.vacuum import Pvacuum as legacy_pvacuum

        return LegacyPMNS, legacy_pvacuum
    except Exception as exc:
        print("\nLegacy peanuts import was not available.")
        print(f"Reason: {type(exc).__name__}: {exc}")
        return None, None


def test_zero_baseline_identity():
    pmns = build_pmns()
    S = vacuum_evolutor(pmns, DM21, DM3L, E_MeV=1000.0, L_km=0.0, device=DEVICE, dtype=DTYPE)
    P = pvacuum([1.0, 0.0, 0.0], pmns, DM21, DM3L, 1000.0, 0.0, massbasis=False, device=DEVICE, dtype=DTYPE)

    print("\nZero-baseline evolutor:")
    print(S)
    print("Final probabilities for initial nu_e:", P)

    identity = torch.eye(3, dtype=torch.complex128, device=DEVICE)
    assert_true(torch.allclose(S, identity, atol=1.0e-12), "vacuum evolutor must be identity at L=0")
    assert_true(torch.allclose(P, torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE), atol=1.0e-12), "nu_e must remain nu_e at L=0")


def test_probability_matrix_conservation():
    pmns = build_pmns()
    energies = torch.tensor([250.0, 1000.0, 3000.0, 10000.0], dtype=DTYPE, device=DEVICE)
    baseline = 735.0
    P = vacuum_probability_matrix(pmns, DM21, DM3L, energies, baseline, device=DEVICE, dtype=DTYPE)

    col_sums = P.sum(dim=-2)
    row_sums = P.sum(dim=-1)

    print("\nProbability matrices P[final flavour, initial flavour]:")
    print(P)
    print("Column sums, fixed initial flavour:", col_sums)
    print("Row sums, fixed final flavour:", row_sums)

    assert_true(P.shape == (4, 3, 3), "Probability matrix must have shape (NE, 3, 3)")
    assert_true(torch.isfinite(P).all().item(), "Probabilities must be finite")
    assert_true(torch.allclose(col_sums, torch.ones_like(col_sums), atol=1.0e-10), "Each initial flavour column must sum to one")
    assert_true(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1.0e-10), "Each final flavour row must sum to one")


def test_coherent_flavour_probabilities_sum_to_one():
    pmns = build_pmns()
    energies = torch.linspace(500.0, 5000.0, 7, dtype=DTYPE, device=DEVICE)
    initial_mu = torch.tensor([0.0, 1.0, 0.0], dtype=torch.complex128, device=DEVICE)
    P = pvacuum(initial_mu, pmns, DM21, DM3L, energies, 1300.0, massbasis=False, device=DEVICE, dtype=DTYPE)

    print("\ncoherent initial nu_mu probabilities:")
    print(P)
    print("Probability sums:", P.sum(dim=-1))

    assert_true(P.shape == (7, 3), "Batched coherent probabilities must have shape (NE, 3)")
    assert_true(torch.allclose(P.sum(dim=-1), torch.ones(7, dtype=DTYPE), atol=1.0e-10), "coherent final probabilities must sum to one")


def test_mass_basis_weights_are_normalized():
    pmns = build_pmns()
    weights = torch.tensor([0.20, 0.30, 0.50], dtype=DTYPE, device=DEVICE)
    baselines = torch.tensor([0.0, 295.0, 1300.0, 12000.0], dtype=DTYPE, device=DEVICE)
    P = pvacuum(weights, pmns, DM21, DM3L, 2000.0, baselines, massbasis=True, device=DEVICE, dtype=DTYPE)

    print("\nFinal flavour probabilities from incoherent mass weights:")
    print(P)
    print("Probability sums:", P.sum(dim=-1))

    assert_true(P.shape == (4, 3), "Mass-basis result must have shape (NL, 3)")
    assert_true(torch.all(P >= -1.0e-12).item(), "Mass-basis probabilities must be non-negative")
    assert_true(torch.allclose(P.sum(dim=-1), torch.ones(4, dtype=DTYPE), atol=1.0e-10), "Mass-basis probabilities must sum to one")


def test_antineutrino_equals_neutrino_when_delta_zero():
    pmns = build_pmns(delta=0.0)
    initial_e = torch.tensor([1.0, 0.0, 0.0], dtype=torch.complex128, device=DEVICE)
    energies = torch.tensor([300.0, 800.0, 2500.0, 8000.0], dtype=DTYPE, device=DEVICE)

    P_nu = pvacuum(initial_e, pmns, DM21, DM3L, energies, 1300.0, antinu=False, massbasis=False, device=DEVICE, dtype=DTYPE)
    P_anu = pvacuum(initial_e, pmns, DM21, DM3L, energies, 1300.0, antinu=True, massbasis=False, device=DEVICE, dtype=DTYPE)

    print("\nNeutrino probabilities at delta=0:")
    print(P_nu)
    print("Antineutrino probabilities at delta=0:")
    print(P_anu)
    print("Max absolute difference:", torch.max(torch.abs(P_nu - P_anu)).item())

    assert_true(torch.allclose(P_nu, P_anu, atol=1.0e-12), "nu and antinu vacuum probabilities must match when delta=0")


def test_compare_with_legacy_pvacuum():
    LegacyPMNS, legacy_pvacuum = _try_legacy()
    if LegacyPMNS is None:
        print("Skipping legacy comparison because legacy peanuts could not be imported.")
        return

    pmns = build_pmns()
    legacy_pmns = LegacyPMNS(0.59, 0.15, 0.78, 1.20)

    cases = [
        ("flavour nu_e", [1.0, 0.0, 0.0], False, 900.0, 295.0),
        ("flavour nu_mu", [0.0, 1.0, 0.0], False, 2500.0, 1300.0),
        ("mass weights", [0.20, 0.30, 0.50], True, 5000.0, 12000.0),
    ]

    max_errors = []
    print("\nTorch vacuum probabilities compared with legacy peanuts:")

    for label, state, massbasis, energy, baseline in cases:
        torch_value = pvacuum(state, pmns, DM21, DM3L, energy, baseline, massbasis=massbasis, antinu=False, device=DEVICE, dtype=DTYPE)
        legacy_value = torch.as_tensor(
            legacy_pvacuum(state, legacy_pmns, DM21, DM3L, energy, baseline, antinu=False, massbasis=massbasis),
            dtype=DTYPE,
            device=DEVICE,
        )
        diff = torch.abs(torch_value - legacy_value)
        max_errors.append(torch.max(diff).item())

        print(f"\nCase: {label}")
        print("torch   :", torch_value)
        print("legacy  :", legacy_value)
        print("abs diff:", diff)
        print("max abs :", f"{torch.max(diff).item():.6e}")

    max_error = max(max_errors)
    print(f"\nGlobal maximum legacy comparison error: {max_error:.6e}")
    assert_true(max_error < 1.0e-10, "Torch vacuum probabilities must match legacy peanuts for neutrinos")


def plot_vacuum_probabilities_vs_baseline(savefig=False):
    pmns = build_pmns()
    baselines = torch.linspace(0.0, 13000.0, 220, dtype=DTYPE, device=DEVICE)
    initial_mu = torch.tensor([0.0, 1.0, 0.0], dtype=torch.complex128, device=DEVICE)
    P = pvacuum(initial_mu, pmns, DM21, DM3L, 1000.0, baselines, massbasis=False, device=DEVICE, dtype=DTYPE)

    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, label in enumerate(FLAVOURS):
        ax.plot(baselines.cpu(), P[:, idx].cpu(), label=label)

    ax.set_xlabel("Baseline L [km]")
    ax.set_ylabel("Final flavour probability")
    ax.set_title("vacuum oscillations from an initial muon neutrino at 1 GeV")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = OUTPUT_DIR / "vacuum_probabilities_vs_baseline.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"\nSaved plot: {path}")
    plt.show()


def plot_neutrino_antineutrino_comparison(savefig=False):
    pmns = build_pmns(delta=1.20)
    energies = torch.logspace(torch.log10(torch.tensor(100.0)), torch.log10(torch.tensor(10000.0)), 160, dtype=DTYPE, device=DEVICE)
    initial_mu = torch.tensor([0.0, 1.0, 0.0], dtype=torch.complex128, device=DEVICE)

    P_nu = pvacuum(initial_mu, pmns, DM21, DM3L, energies, 1300.0, antinu=False, massbasis=False, device=DEVICE, dtype=DTYPE)
    P_anu = pvacuum(initial_mu, pmns, DM21, DM3L, energies, 1300.0, antinu=True, massbasis=False, device=DEVICE, dtype=DTYPE)
    diff = P_nu - P_anu

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

    for idx, label in enumerate(FLAVOURS):
        axes[0].plot(energies.cpu(), P_nu[:, idx].cpu(), label=f"neutrino {label}")
        axes[0].plot(energies.cpu(), P_anu[:, idx].cpu(), "--", label=f"antineutrino {label}")
        axes[1].plot(energies.cpu(), diff[:, idx].cpu(), label=label)

    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_xlabel("Energy [MeV]")
    axes[1].set_xlabel("Energy [MeV]")
    axes[0].set_ylabel("Final flavour probability")
    axes[1].set_ylabel("P(neutrino) - P(antineutrino)")
    axes[0].set_title("vacuum probabilities at L = 1300 km")
    axes[1].set_title("CP-phase vacuum difference")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()

    path = OUTPUT_DIR / "vacuum_neutrino_antineutrino_comparison.png"
    if savefig:
        fig.savefig(path, dpi=200)
    if savefig:
        print(f"\nSaved plot: {path}")
    plt.show()


def test_visualization_outputs(savefig=False):
    plot_vacuum_probabilities_vs_baseline(savefig=savefig)
    plot_neutrino_antineutrino_comparison(savefig=savefig)

    expected = [
        OUTPUT_DIR / "vacuum_probabilities_vs_baseline.png",
        OUTPUT_DIR / "vacuum_neutrino_antineutrino_comparison.png",
    ]
    for path in expected:
        if savefig:
            assert_true(path.is_file(), f"Plot was not created: {path}")


if __name__ == "__main__":
    tests = [
        test_zero_baseline_identity,
        test_probability_matrix_conservation,
        test_coherent_flavour_probabilities_sum_to_one,
        test_mass_basis_weights_are_normalized,
        test_antineutrino_equals_neutrino_when_delta_zero,
        test_compare_with_legacy_pvacuum,
        test_visualization_outputs,
    ]

    run_test_suite(tests, suite_name="vacuum PROBABILITY tests", verbose_traceback=True)

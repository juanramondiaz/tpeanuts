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
Verbose tests and diagnostic plots for peanuts_torch.potentials.

Run with:

    python tests/core/test_potentials.py

or with pytest:

    pytest tests/core/test_potentials.py -v -s

This script tests:

    1. matter_potential(n, antinu)
    2. kinetic_potential(mSq, E)
    3. broadcasting behaviour
    4. neutrino / antineutrino sign flip
    5. energy dependence k ~ 1/E
    6. density dependence V ~ n

It also generates plots under OUTPUT_ROOT/test/core/test0_potential.
"""



from __future__ import annotations

import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt

import tpeanuts.util.constant as constant

from tpeanuts.core.potential import (
    matter_potential,
    kinetic_potential,
)


# ============================================================
# Global configuration
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_TEST_ROOT = Path(OUTPUT_ROOT / "test")
OUTPUT_DIR = Path(OUTPUT_TEST_ROOT / "core" / Path(__file__).stem)
os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.set_printoptions(
    precision=12,
    sci_mode=True,
    linewidth=160,
)

from tpeanuts.util.test_utils import (
    printoptions,
    banner, section, print_ok, print_fail, 
    assert_close, assert_true, run_test_suite
    )
printoptions()

# ============================================================
# tests
# ============================================================

def test_matter_potential_formula():

    section("TEST: matter_potential formula")

    n = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=DTYPE, device=DEVICE)

    V = matter_potential(n, antinu=False)

    V_expected = constant.R_E * 3.868e-7 * n

    print("n [mol/cm^3]:")
    print(n)

    print("V neutrino:")
    print(V)

    print("V expected:")
    print(V_expected)

    assert_close(
        V,
        V_expected,
        name="V = R_E * 3.868e-7 * n",
    )


def test_matter_potential_antinu_sign():

    section("TEST: matter_potential antineutrino sign")

    n = torch.linspace(
        0.0,
        10.0,
        11,
        dtype=DTYPE,
        device=DEVICE,
    )

    V_nu = matter_potential(n, antinu=False)
    V_anti = matter_potential(n, antinu=True)

    print("V neutrino:")
    print(V_nu)

    print("V antineutrino:")
    print(V_anti)

    assert_close(
        V_anti,
        -V_nu,
        name="Antineutrino matter potential is -V_neutrino",
    )


def test_matter_potential_shape_dtype_device():

    section("TEST: matter_potential shape, dtype and device")

    n = torch.ones((4, 5), dtype=DTYPE, device=DEVICE)

    V = matter_potential(n, antinu=False)

    print("n shape:", n.shape)
    print("V shape:", V.shape)
    print("n dtype:", n.dtype)
    print("V dtype:", V.dtype)
    print("device:", V.device)

    assert_true(V.shape == n.shape, "matter_potential preserves shape")
    assert_true(V.dtype == n.dtype, "matter_potential preserves dtype")
    assert_true(V.device == n.device, "matter_potential preserves device")


def test_kinetic_potential_scalar_energy():

    section("TEST: kinetic_potential with scalar energy")

    mSq = torch.tensor(
        [0.0, 7.42e-5, 2.517e-3],
        dtype=DTYPE,
        device=DEVICE,
    )

    E = torch.tensor(1000.0, dtype=DTYPE, device=DEVICE)

    k = kinetic_potential(mSq, E)

    k_expected = (
        constant.R_E
        * 0.5
        * 1.0e-12
        * mSq
        / E
        / constant.HBARC_MeV_m
    )

    print("mSq [eV^2]:")
    print(mSq)

    print("E [MeV]:")
    print(E)

    print("k:")
    print(k)

    print("k expected:")
    print(k_expected)

    assert_true(k.shape == (3,), "k has shape (3,) for scalar E and mSq=(3,)")
    assert_close(
        k,
        k_expected,
        name="kinetic_potential scalar energy formula",
    )


def test_kinetic_potential_energy_grid():

    section("TEST: kinetic_potential with energy grid")

    mSq = torch.tensor(
        [0.0, 7.42e-5, 2.517e-3],
        dtype=DTYPE,
        device=DEVICE,
    )

    E = torch.logspace(
        2.0,
        5.0,
        8,
        dtype=DTYPE,
        device=DEVICE,
    )

    k = kinetic_potential(mSq, E)

    k_expected = (
        constant.R_E
        * 0.5
        * 1.0e-12
        * mSq[None, :]
        / E[:, None]
        / constant.HBARC_MeV_m
    )

    print("E grid [MeV]:")
    print(E)

    print("k shape:", k.shape)
    print("k:")
    print(k)

    assert_true(k.shape == (E.numel(), 3), "k has shape (Ne, 3)")
    assert_close(
        k,
        k_expected,
        name="kinetic_potential energy-grid broadcasting",
    )


def test_kinetic_potential_batched_mass_and_energy():

    section("TEST: kinetic_potential with batched mSq and E")

    mSq = torch.tensor(
        [
            [0.0, 7.42e-5, 2.517e-3],
            [0.0, 7.42e-5, 2.517e-3],
            [0.0, 7.42e-5, 2.517e-3],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )

    E = torch.tensor(
        [500.0, 1000.0, 5000.0],
        dtype=DTYPE,
        device=DEVICE,
    )

    k = kinetic_potential(mSq, E)

    k_expected = (
        constant.R_E
        * 0.5
        * 1.0e-12
        * mSq
        / E[:, None]
        / constant.HBARC_MeV_m
    )

    print("mSq shape:", mSq.shape)
    print("E shape:", E.shape)
    print("k shape:", k.shape)
    print("k:")
    print(k)

    assert_true(k.shape == (3, 3), "k has shape (batch, 3)")
    assert_close(
        k,
        k_expected,
        name="kinetic_potential batched formula",
    )


def test_kinetic_potential_inverse_energy_scaling():

    section("TEST: kinetic_potential inverse-energy scaling")

    mSq = torch.tensor(
        [0.0, 7.42e-5, 2.517e-3],
        dtype=DTYPE,
        device=DEVICE,
    )

    E1 = torch.tensor(1000.0, dtype=DTYPE, device=DEVICE)
    E2 = torch.tensor(2000.0, dtype=DTYPE, device=DEVICE)

    k1 = kinetic_potential(mSq, E1)
    k2 = kinetic_potential(mSq, E2)

    ratio = k1[1:] / k2[1:]

    print("k(E=1000 MeV):")
    print(k1)

    print("k(E=2000 MeV):")
    print(k2)

    print("ratio k(E1)/k(E2):")
    print(ratio)

    assert_close(
        ratio,
        torch.full_like(ratio, 2.0),
        name="k scales as 1/E",
    )


def test_kinetic_potential_requires_tensors():

    section("TEST: kinetic_potential requires torch tensors")

    mSq = [0.0, 7.42e-5, 2.517e-3]
    E = torch.tensor(1000.0, dtype=DTYPE, device=DEVICE)

    try:
        kinetic_potential(mSq, E)
    except TypeError:
        print_ok("TypeError raised when mSq_eV2 is not a torch.Tensor")
    else:
        print_fail("TypeError was not raised for non-tensor mSq_eV2")
        raise AssertionError("kinetic_potential should require tensor mSq_eV2")

    mSq_t = torch.tensor(mSq, dtype=DTYPE, device=DEVICE)
    E_bad = 1000.0

    try:
        kinetic_potential(mSq_t, E_bad)
    except TypeError:
        print_ok("TypeError raised when E_MeV is not a torch.Tensor")
    else:
        print_fail("TypeError was not raised for non-tensor E_MeV")
        raise AssertionError("kinetic_potential should require tensor E_MeV")


# ============================================================
# Plotting diagnostics
# ============================================================

def plot_matter_potential_vs_density(savefig=False):

    section("PLOT: matter potential vs density")

    n = torch.linspace(
        0.0,
        15.0,
        300,
        dtype=DTYPE,
        device=DEVICE,
    )

    V_nu = matter_potential(n, antinu=False)
    V_anti = matter_potential(n, antinu=True)

    n_cpu = n.detach().cpu().numpy()
    V_nu_cpu = V_nu.detach().cpu().numpy()
    V_anti_cpu = V_anti.detach().cpu().numpy()

    plt.figure(figsize=(8, 5))
    plt.plot(n_cpu, V_nu_cpu, label=r"neutrino")
    plt.plot(n_cpu, V_anti_cpu, label=r"antineutrino")
    plt.xlabel(r"Electron density $n_e$ [mol/cm$^3$]")
    plt.ylabel(r"Dimensionless matter potential $V$")
    plt.title("Matter potential versus electron density")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "matter_potential_vs_density.png")
    if savefig:
        plt.savefig(path, dpi=200)
    #plt.show()
    if savefig:
        print(f"Saved: {path}")


def plot_kinetic_potential_vs_energy(savefig=False):

    section("PLOT: kinetic potential vs energy")

    E = torch.logspace(
        1.0,
        6.0,
        500,
        dtype=DTYPE,
        device=DEVICE,
    )

    mSq = torch.tensor(
        [0.0, 7.42e-5, 2.517e-3],
        dtype=DTYPE,
        device=DEVICE,
    )

    k = kinetic_potential(mSq, E)

    E_cpu = E.detach().cpu().numpy()
    k_cpu = k.detach().cpu().numpy()

    plt.figure(figsize=(8, 5))
    plt.loglog(E_cpu, k_cpu[:, 1], label=r"$\Delta m^2_{21}$")
    plt.loglog(E_cpu, k_cpu[:, 2], label=r"$\Delta m^2_{31}$")
    plt.xlabel(r"Energy $E$ [MeV]")
    plt.ylabel(r"Dimensionless kinetic potential $k$")
    plt.title("Kinetic potential versus energy")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "kinetic_potential_vs_energy.png")
    if savefig:
        plt.savefig(path, dpi=200)
    #plt.show()

    if savefig:
        print(f"Saved: {path}")


def plot_ratio_matter_to_kinetic(savefig=False):

    section("PLOT: matter-to-kinetic scale comparison")

    E = torch.logspace(
        1.0,
        6.0,
        500,
        dtype=DTYPE,
        device=DEVICE,
    )

    n = torch.tensor(5.0, dtype=DTYPE, device=DEVICE)

    mSq31 = torch.tensor(
        [2.517e-3],
        dtype=DTYPE,
        device=DEVICE,
    )

    V = matter_potential(n, antinu=False)
    k31 = kinetic_potential(mSq31, E).squeeze(-1)

    ratio = torch.abs(V / k31)

    E_cpu = E.detach().cpu().numpy()
    ratio_cpu = ratio.detach().cpu().numpy()

    plt.figure(figsize=(8, 5))
    plt.loglog(E_cpu, ratio_cpu)
    plt.xlabel(r"Energy $E$ [MeV]")
    plt.ylabel(r"$|V/k_{31}|$")
    plt.title(r"Matter scale compared with kinetic scale, $n_e=5$ mol/cm$^3$")
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "matter_to_kinetic_ratio.png")
    if savefig:
        plt.savefig(path, dpi=200)
    #plt.show()

    if savefig:
        print(f"Saved: {path}")


# ============================================================
# Main
# ============================================================

# ============================================================
# Runner
# ============================================================

def run_test0_potential_tests(verbose_traceback=False):
    tests = [
        test_matter_potential_formula,
        test_matter_potential_antinu_sign,
        test_matter_potential_shape_dtype_device,
        test_kinetic_potential_scalar_energy,
        test_kinetic_potential_energy_grid,
        test_kinetic_potential_batched_mass_and_energy,
        test_kinetic_potential_inverse_energy_scaling,
        test_kinetic_potential_requires_tensors,
        plot_matter_potential_vs_density,
        plot_kinetic_potential_vs_energy,
        plot_ratio_matter_to_kinetic,
    ]
    return run_test_suite(
        tests,
        suite_name="core TEST0 POTENTIAL tests",
        verbose_traceback=verbose_traceback,
    )


if __name__ == "__main__":
    run_test0_potential_tests(verbose_traceback=True)

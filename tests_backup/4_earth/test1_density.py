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
Spyder-friendly tests and visual diagnostics for EarthDensity.

This script tests:

    1. Loading the earth density CSV file.
    2. Tensor shapes and dtype/device consistency.
    3. Shell-crossing coordinates x_j(eta).
    4. Trajectory-dependent polynomial coefficients a,b,c.
    5. Direct density evaluation n_e(x, eta).
    6. Radial earth density profile visualization.
    7. density along trajectory visualization for several eta values.

Input file:

    data/density/earth_density.csv

Run directly in Spyder or from terminal:

    python tpeanuts/tests/earth/test_earth_density.py
"""



from __future__ import annotations

import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt

from tpeanuts.util.test_utils import (
    run_test_suite,
    assert_true,
    assert_close,
)
from tpeanuts.util.torch_util import _default_device
from tpeanuts.io.io_earth import load_earth_density_from_csv


# ============================================================
# Configuration
# ============================================================

DEVICE = _default_device()
DTYPE = torch.float64

THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parents[3]
TESTS_DIR = THIS_FILE.parents[1]

density_FILE = PACKAGE_DIR / "data" / "density" / "earth_density.csv"

DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_TEST_ROOT = Path(OUTPUT_ROOT / "test")
OUTPUT_DIR = Path(OUTPUT_TEST_ROOT / "earth" / Path(__file__).stem)
os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.set_printoptions(
    precision=10,
    sci_mode=True,
    linewidth=160,
)


# ============================================================
# Shared loader
# ============================================================

def load_density():
    density = load_earth_density_from_csv(
        density_FILE,
        tabulated_density=False,
        device=DEVICE,
        dtype=DTYPE,
    )
    
    return density


# ============================================================
# tests
# ============================================================

def test_density_file_exists():
    print(f"Checking density file: {density_FILE}")
    assert_true(
        os.path.isfile(density_FILE),
        f"density file not found: {density_FILE}",
    )


def test_density_loads_correctly():
    density = load_density()

    print("\nLoaded earth density object:")
    print("rj shape     :", density.rj.shape)
    print("alpha shape  :", density.alpha.shape)
    print("beta shape   :", density.beta.shape)
    print("gamma shape  :", density.gamma.shape)
    print("deltas shape :", density.deltas.shape)
    print("Test Device  :", DEVICE)
    print("device       :", density.device)
    print("dtype        :", density.dtype)

    assert_true(density.rj.ndim == 1, "rj must be 1D")
    assert_true(density.alpha.shape == density.rj.shape, "alpha shape mismatch")
    assert_true(density.beta.shape == density.rj.shape, "beta shape mismatch")
    assert_true(density.gamma.shape == density.rj.shape, "gamma shape mismatch")
    assert_true(density.deltas.ndim == 2, "deltas must have shape (Nd, Ns)")
    assert_true(density.deltas.shape[1] == density.rj.numel(), "deltas second dimension must match number of shells")
    assert_true(density.device == DEVICE, "density tensors are on wrong device")
    assert_true(density.dtype == DTYPE, "density tensors have wrong dtype")


def test_shell_radii_are_valid():
    density = load_density()

    rj = density.rj.detach().cpu()

    print("\nShell radii rj:")
    print(rj)

    assert_true(torch.isfinite(density.rj).all().item(), "rj contains NaN or Inf")
    assert_true((density.rj >= 0.0).all().item(), "rj must be >= 0")
    assert_true((density.rj <= 1.0 + 1.0e-12).all().item(), "rj must be <= 1")
    assert_true(torch.all(density.rj[1:] >= density.rj[:-1]).item(), "rj must be sorted increasingly")


def test_shell_crossings_scalar_eta():
    density = load_density()

    eta = torch.tensor(0.40, device=DEVICE, dtype=DTYPE)

    xj, crossed, idx0 = density.shells_x(eta)

    print("\nScalar eta shell crossings:")
    print("eta     :", eta.item())
    print("xj shape:", xj.shape)
    print("crossed :", crossed)
    print("idx0    :", idx0)
    print("xj      :", xj)

    assert_true(xj.shape == density.rj.shape, "xj shape mismatch")
    assert_true(crossed.shape == density.rj.shape, "crossed shape mismatch")
    assert_true(idx0.ndim == 0, "idx0 should be scalar")
    assert_true(torch.isfinite(xj).all().item(), "xj contains NaN or Inf")


def test_shell_crossings_batched_eta():
    density = load_density()

    eta = torch.linspace(
        0.0,
        torch.pi / 2.0 - 1.0e-4,
        8,
        device=DEVICE,
        dtype=DTYPE,
    )

    xj, crossed, idx0 = density.shells_x(eta)

    print("\nBatched eta shell crossings:")
    print("eta shape    :", eta.shape)
    print("xj shape     :", xj.shape)
    print("crossed shape:", crossed.shape)
    print("idx0 shape   :", idx0.shape)

    assert_true(xj.shape == (eta.numel(), density.rj.numel()), "batched xj shape mismatch")
    assert_true(crossed.shape == xj.shape, "batched crossed shape mismatch")
    assert_true(idx0.shape == eta.shape, "batched idx0 shape mismatch")
    assert_true(torch.isfinite(xj).all().item(), "batched xj contains NaN or Inf")


def test_parameters_abc_shapes():
    density = load_density()

    eta = torch.tensor(
        [0.0, 0.3, 0.7, 1.0],
        device=DEVICE,
        dtype=DTYPE,
    )

    coeffs_abc, xj, crossed = density.parameters_abc(eta)

    print("\nparameters_abc:")
    print("eta shape        :", eta.shape)
    print("coeffs_abc shape :", coeffs_abc.shape)
    print("xj shape         :", xj.shape)
    print("crossed shape    :", crossed.shape)

    assert_true(coeffs_abc.shape == (eta.numel(), density.rj.numel(), 3), "coeffs_abc shape mismatch")
    assert_true(xj.shape == (eta.numel(), density.rj.numel()), "xj shape mismatch")
    assert_true(crossed.shape == xj.shape, "crossed shape mismatch")
    assert_true(torch.isfinite(coeffs_abc).all().item(), "coeffs_abc contains NaN or Inf")


def test_full_parameters_shapes():
    density = load_density()

    eta = torch.tensor(
        [0.0, 0.5, 1.0],
        device=DEVICE,
        dtype=DTYPE,
    )

    coeffs, xj, crossed = density.parameters(eta)

    expected_ncoeff = 3 + density.deltas.shape[0]

    print("\nparameters:")
    print("coeffs shape :", coeffs.shape)
    print("expected     :", (eta.numel(), density.rj.numel(), expected_ncoeff))

    assert_true(coeffs.shape == (eta.numel(), density.rj.numel(), expected_ncoeff), "full parameters shape mismatch")
    assert_true(torch.isfinite(coeffs).all().item(), "full parameters contains NaN or Inf")


def test_density_x_eta_values_are_finite():
    density = load_density()

    x = torch.linspace(
        0.0,
        1.0,
        200,
        device=DEVICE,
        dtype=DTYPE,
    )

    eta = torch.tensor(0.4, device=DEVICE, dtype=DTYPE)

    ne = density.density_x_eta(x, eta)

    print("\ndensity_x_eta:")
    print("x shape  :", x.shape)
    print("ne shape :", ne.shape)
    print("min ne   :", ne.min().item())
    print("max ne   :", ne.max().item())

    assert_true(ne.shape == x.shape, "density_x_eta shape mismatch")
    assert_true(torch.isfinite(ne).all().item(), "density_x_eta contains NaN or Inf")
    assert_true((ne >= -1.0e-12).all().item(), "density should be non-negative up to numerical tolerance")


def test_density_call_wrapper():
    density = load_density()

    r = torch.tensor(0.5, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.3, device=DEVICE, dtype=DTYPE)

    ne1 = density.density_x_eta(r, eta)
    ne2 = density.call(r, eta)
    ne3 = density(r, eta)

    print("\ndensity call wrappers:")
    print("density_x_eta:", ne1.item())
    print("call         :", ne2.item())
    print("__call__     :", ne3.item())

    assert_close(
        ne2.item(),
        ne1.item(),
        atol=1.0e-12,
        rtol=1.0e-12,
        message="density.call does not match density_x_eta",
    )

    assert_close(
        ne3.item(),
        ne1.item(),
        atol=1.0e-12,
        rtol=1.0e-12,
        message="density.__call__ does not match density_x_eta",
    )


# ============================================================
# Visualization
# ============================================================

def plot_radial_density_profile(savefig=False):
    density = load_density()

    r = torch.linspace(
        0.0,
        1.0,
        1000,
        device=DEVICE,
        dtype=DTYPE,
    )

    eta = torch.zeros_like(r)

    ne = density.density_x_eta(r, eta)

    r_cpu = r.detach().cpu().numpy()
    ne_cpu = ne.detach().cpu().numpy()

    plt.figure(figsize=(8, 5))
    plt.plot(r_cpu, ne_cpu)
    plt.xlabel(r"Dimensionless earth radius $r/R_E$")
    plt.ylabel(r"Electron density $n_e$ [mol cm$^{-3}$]")
    plt.title("earth electron density profile")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "earth_density_profile_r.png")
    if savefig:
        plt.savefig(path, dpi=200)
    plt.show()

    if savefig:
        print(f"\nSaved plot: {path}")


def plot_density_along_trajectories(savefig=False):
    density = load_density()

    eta_values = torch.tensor(
        [0.0, 0.3, 0.6, 1.0],
        device=DEVICE,
        dtype=DTYPE,
    )

    x = torch.linspace(
        0.0,
        1.0,
        1000,
        device=DEVICE,
        dtype=DTYPE,
    )

    x_cpu = x.detach().cpu().numpy()

    plt.figure(figsize=(8, 5))

    for eta in eta_values:
        ne = density.density_x_eta(x, eta)

        plt.plot(
            x_cpu,
            ne.detach().cpu().numpy(),
            label=rf"$\eta={eta.item():.2f}$ rad",
        )

    plt.xlabel(r"Trajectory coordinate $x$")
    plt.ylabel(r"Electron density $n_e(x,\eta)$ [mol cm$^{-3}$]")
    plt.title("earth density along different nadir trajectories")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "earth_density_x_eta.png")
    if savefig:
        plt.savefig(path, dpi=200)
    plt.show()

    if savefig:
        print(f"Saved plot: {path}")


def plot_shell_crossings(savefig=False):
    density = load_density()

    eta = torch.linspace(
        0.0,
        torch.pi / 2.0 - 1.0e-5,
        300,
        device=DEVICE,
        dtype=DTYPE,
    )

    xj, crossed, _ = density.shells_x(eta)

    eta_cpu = eta.detach().cpu().numpy()
    xj_cpu = xj.detach().cpu().numpy()
    crossed_cpu = crossed.detach().cpu().numpy()

    plt.figure(figsize=(8, 5))

    for j in range(density.rj.numel()):
        y = xj_cpu[:, j].copy()
        y[~crossed_cpu[:, j]] = float("nan")

        plt.plot(
            eta_cpu,
            y,
            lw=1.0,
            alpha=0.8,
        )

    plt.xlabel(r"Nadir angle $\eta$ [rad]")
    plt.ylabel(r"Shell crossing coordinate $x_j(\eta)$")
    plt.title("earth shell-crossing coordinates")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "earth_shell_crossings.png")
    if savefig:
        plt.savefig(path, dpi=200)
    plt.show()

    if savefig:
        print(f"Saved plot: {path}")


def test_visualization_outputs(savefig=False):
    plot_radial_density_profile(savefig=savefig)
    plot_density_along_trajectories(savefig=savefig)
    plot_shell_crossings(savefig=savefig)

    expected_files = [
        "earth_density_profile_r.png",
        "earth_density_x_eta.png",
        "earth_shell_crossings.png",
    ]

    for filename in expected_files:
        path = os.path.join(OUTPUT_DIR, filename)
        if savefig:
            assert_true(os.path.isfile(path), f"Plot was not created: {path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    tests = [
        test_density_file_exists,
        test_density_loads_correctly,
        test_shell_radii_are_valid,
        test_shell_crossings_scalar_eta,
        test_shell_crossings_batched_eta,
        test_parameters_abc_shapes,
        test_full_parameters_shapes,
        test_density_x_eta_values_are_finite,
        test_density_call_wrapper,
        test_visualization_outputs,
    ]

    run_test_suite(
        tests,
        suite_name="earth density tests",
        verbose_traceback=True,
    )

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
Visual diagnostics for depth utilities.

This script generates plots for:

    - compute_vertical_depth_from_mceq
    - compute_slant_depth_from_vertical_depth
    - compute_slant_depth_from_mceq
    - compute_dXdh

Run directly in Spyder.
"""



import torch
import matplotlib.pyplot as plt

from tpeanuts.util.test_utils import run_test_suite

from tpeanuts.external.mceq.config import (
    MCEqModelConfig,
)

from tpeanuts.external.mceq.core import (
    init_mceq,
)

from tpeanuts.external.mceq.density import (
    atmospheric_mass_density_profile_from_mceq,
)

from tpeanuts.external.mceq.depth import (
    compute_vertical_depth_from_density,
    compute_vertical_depth_from_mceq,
    compute_slant_depth_from_vertical_depth,
    compute_slant_depth_from_mceq,
    compute_dXdh,
)


# ============================================================
# Configuration
# ============================================================

DEVICE = "cpu"
DTYPE = torch.float64

THETA_LIST = [0.0, 30.0, 60.0, 75.0]

N_H = 1000
H_MIN_KM = 0.0
H_MAX_KM = 100.0

MODEL_CONFIG = MCEqModelConfig(
    interaction_model="SIBYLL23D",
    density_model="CORSIKA",
)


# ============================================================
# Height grid
# ============================================================

h_grid = torch.linspace(
    H_MIN_KM,
    H_MAX_KM,
    N_H,
    device=DEVICE,
    dtype=DTYPE,
)


# ============================================================
# Plot 1
# Vertical depth from mceq
# ============================================================

def plot_vertical_depth_from_mceq():

    print("\nComputing vertical depth from mceq...")

    mceq = init_mceq(
        theta_deg=0.0,
        config=MODEL_CONFIG,
    )

    rho = atmospheric_mass_density_profile_from_mceq(
        h_km=h_grid,
        mceq=mceq,
        device=DEVICE,
        dtype=DTYPE,
    )

    X_vertical = compute_vertical_depth_from_mceq(
        h_km=h_grid,
        mceq=mceq,
        config=MODEL_CONFIG,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("rho shape       :", rho.shape)
    print("X_vertical shape:", X_vertical.shape)

    print("X(0 km)   =", float(X_vertical[0].item()), "g/cm^2")
    print("X(100 km) =", float(X_vertical[-1].item()), "g/cm^2")

    fig = plt.figure(figsize=(8, 6))

    plt.plot(
        X_vertical.cpu().numpy(),
        h_grid.cpu().numpy(),
        lw=2,
        label=r"$X_{\rm vertical}(h)$",
    )

    plt.xlabel(r"Vertical depth $X(h)$ [g/cm$^2$]")
    plt.ylabel(r"Altitude $h$ [km]")

    plt.title("Vertical Atmospheric Depth from mceq")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# Plot 2
# Slant-depth comparison
# ============================================================

def plot_slant_depth_comparison():

    print("\nComparing slant-depth approximations...")

    fig = plt.figure(figsize=(9, 7))

    for theta_deg in THETA_LIST:

        print(f"\nTheta = {theta_deg:.1f} deg")

        mceq = init_mceq(
            theta_deg=theta_deg,
            config=MODEL_CONFIG,
        )

        X_vertical = compute_vertical_depth_from_mceq(
            h_km=h_grid,
            mceq=mceq,
            config=MODEL_CONFIG,
            device=DEVICE,
            dtype=DTYPE,
        )

        X_planar = compute_slant_depth_from_vertical_depth(
            X_vertical_gcm2=X_vertical,
            theta_deg=theta_deg,
            device=DEVICE,
            dtype=DTYPE,
        )

        X_mceq = compute_slant_depth_from_mceq(
            h_km=h_grid,
            theta_deg=theta_deg,
            mceq=mceq,
            config=MODEL_CONFIG,
            device=DEVICE,
            dtype=DTYPE,
        )

        rel_diff = torch.abs(X_mceq - X_planar) / torch.clamp(
            torch.abs(X_mceq),
            min=1.0e-30,
        )

        print(
            "max relative difference =",
            float(torch.max(rel_diff).item())
        )

        plt.plot(
            X_planar.cpu().numpy(),
            h_grid.cpu().numpy(),
            lw=2,
            ls="--",
            label=f"Planar θ={theta_deg:.0f}°",
        )

        plt.plot(
            X_mceq.cpu().numpy(),
            h_grid.cpu().numpy(),
            lw=2,
            label=f"mceq θ={theta_deg:.0f}°",
        )

    plt.xlabel(r"Slant depth $X(h,\theta)$ [g/cm$^2$]")
    plt.ylabel(r"Altitude $h$ [km]")

    plt.title("Planar vs mceq Slant Depth")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# Plot 3
# dX/dh
# ============================================================

def plot_dXdh():

    print("\nComputing dX/dh...")

    mceq = init_mceq(
        theta_deg=0.0,
        config=MODEL_CONFIG,
    )

    rho = atmospheric_mass_density_profile_from_mceq(
        h_km=h_grid,
        mceq=mceq,
        device=DEVICE,
        dtype=DTYPE,
    )

    X_vertical = compute_vertical_depth_from_mceq(
        h_km=h_grid,
        mceq=mceq,
        config=MODEL_CONFIG,
        device=DEVICE,
        dtype=DTYPE,
    )

    dXdh = compute_dXdh(
        X_gcm2=X_vertical,
        h_km=h_grid,
        device=DEVICE,
        dtype=DTYPE,
    )

    expected = -1.0e5 * rho

    rel_err = torch.abs(dXdh - expected) / torch.clamp(
        torch.abs(expected),
        min=1.0e-30,
    )

    print("dXdh min =", float(dXdh.min().item()))
    print("dXdh max =", float(dXdh.max().item()))

    print(
        "max relative error =",
        float(torch.max(rel_err[5:-5]).item())
    )

    fig = plt.figure(figsize=(8, 6))

    plt.plot(
        dXdh.cpu().numpy(),
        h_grid.cpu().numpy(),
        lw=2,
        label=r"Numerical $dX/dh$",
    )

    plt.plot(
        expected.cpu().numpy(),
        h_grid.cpu().numpy(),
        lw=2,
        ls="--",
        label=r"$-10^5 \rho(h)$",
    )

    plt.xlabel(r"$dX/dh$ [g/cm$^2$/km]")
    plt.ylabel(r"Altitude $h$ [km]")

    plt.title(r"Consistency Check: $dX/dh = -10^5 \rho(h)$")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# Plot 4
# Relative difference
# ============================================================

def plot_relative_difference_planar_vs_mceq():

    print("\nPlotting relative differences...")

    fig = plt.figure(figsize=(8, 6))

    for theta_deg in THETA_LIST:

        mceq = init_mceq(
            theta_deg=theta_deg,
            config=MODEL_CONFIG,
        )

        X_vertical = compute_vertical_depth_from_mceq(
            h_km=h_grid,
            mceq=mceq,
            config=MODEL_CONFIG,
            device=DEVICE,
            dtype=DTYPE,
        )

        X_planar = compute_slant_depth_from_vertical_depth(
            X_vertical_gcm2=X_vertical,
            theta_deg=theta_deg,
            device=DEVICE,
            dtype=DTYPE,
        )

        X_mceq = compute_slant_depth_from_mceq(
            h_km=h_grid,
            theta_deg=theta_deg,
            mceq=mceq,
            config=MODEL_CONFIG,
            device=DEVICE,
            dtype=DTYPE,
        )

        rel_diff = (
            torch.abs(X_mceq - X_planar)
            / torch.clamp(torch.abs(X_mceq), min=1.0e-30)
        )

        plt.plot(
            rel_diff.cpu().numpy(),
            h_grid.cpu().numpy(),
            lw=2,
            label=f"{theta_deg:.0f}°",
        )

    plt.xlabel("Relative difference")
    plt.ylabel(r"Altitude $h$ [km]")

    plt.title("Relative Difference: Planar vs mceq Slant Depth")

    plt.grid(True)
    plt.legend(title=r"$\theta$")

    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

def run_depth_visual_tests():

    print("\n" + "=" * 80)
    print("DEPTH VISUAL tests")
    print("=" * 80)

    plot_vertical_depth_from_mceq()

    plot_slant_depth_comparison()

    plot_dXdh()

    plot_relative_difference_planar_vs_mceq()

    print("\nFinished visual diagnostics.")


def test_depth_visual_diagnostics():
    run_depth_visual_tests()


def run_mceq_depth_visual_tests(verbose_traceback=False):
    return run_test_suite(
        [test_depth_visual_diagnostics],
        suite_name="mceq DEPTH VISUAL tests",
        verbose_traceback=verbose_traceback,
    )


if __name__ == "__main__":
    run_mceq_depth_visual_tests(verbose_traceback=True)

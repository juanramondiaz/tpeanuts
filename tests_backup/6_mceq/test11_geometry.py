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
Visual test for spherical earth angle conversion.

Relation:

    R_E sin(alpha) = (R_E - h_d) sin(theta)

where:

    alpha : zenith angle at earth's surface, used by mceq
    theta : zenith angle at detector
    h_d   : detector depth below surface
"""



import torch
import matplotlib.pyplot as plt

from tpeanuts.util.test_utils import run_test_suite

from tpeanuts.atmosphere.geometry import (
    alpha_surface_to_theta_detector,
    theta_detector_to_alpha_surface,
    alpha_max_for_detector_depth
    )
# ============================================================
# Constants
# ============================================================

R_E_KM = 6371.0

DEVICE = "cpu"
DTYPE = torch.float64

DEPTHS_KM = [
   1.0,
    10.0,
    20.0,
    50.0,
    100.0,
    200.0,
]


# ============================================================
# Plots
# ============================================================

def plot_alpha_to_theta():

    plt.figure(figsize=(9, 6))

    for h_d in DEPTHS_KM:

        alpha_max = float(alpha_max_for_detector_depth(h_d).item())

        alpha_grid = torch.linspace(
            0.0,
            180,
            1000,
            device=DEVICE,
            dtype=DTYPE,
        )

        theta_grid = alpha_surface_to_theta_detector(
            alpha_grid,
            h_d_km=h_d,
        )

        plt.plot(
            alpha_grid.cpu().numpy(),
            theta_grid.cpu().numpy(),
            lw=2,
            label=rf"$h_d={h_d:g}$ km",
        )
        """
        plt.plot(
            alpha_grid.cpu().numpy(),
            alpha_grid.cpu().numpy() - theta_grid.cpu().numpy(),
            lw=2,
            label= "a-theta",
            )
        """
        

    plt.xlabel(r"Surface zenith angle $\alpha$ [deg]")
    plt.ylabel(r"Detector zenith angle $\theta$ [deg]")

    plt.title(
        r"Angle conversion: "
        r"$\theta=\arcsin\left(\frac{R_E-h_d}{R_E}\sin\alpha\right)$"
        
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_theta_to_alpha():

    plt.figure(figsize=(9, 6))

    theta_grid = torch.linspace(
        0.0,
        180,
        1000,
        device=DEVICE,
        dtype=DTYPE,
    )

    for h_d in DEPTHS_KM:

        alpha_grid = theta_detector_to_alpha_surface(
            theta_grid,
            h_d_km=h_d,
        )

        plt.plot(
            theta_grid.cpu().numpy(),
            alpha_grid.cpu().numpy(),
            lw=2,
            label=rf"$h_d={h_d:g}$ km",
        )

    plt.xlabel(r"Detector zenith angle $\theta$ [deg]")
    plt.ylabel(r"Surface zenith angle $\alpha$ [deg]")

    plt.title(
        r"Inverse conversion: "
        r"$\alpha=\arcsin\left(\frac{R_E}{R_E-h_d}\sin\theta\right)$"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_difference_theta_minus_alpha():

    plt.figure(figsize=(9, 6))

    alpha_grid = torch.linspace(
        0.0,
        180,
        500,
        device=DEVICE,
        dtype=DTYPE,
    )

    for h_d in DEPTHS_KM:

        theta_grid = alpha_surface_to_theta_detector(
            alpha_grid,
            h_d_km=h_d,
        )

        diff = alpha_grid-theta_grid 

        plt.plot(
            alpha_grid.cpu().numpy(),
            diff.cpu().numpy(),
            lw=2,
            label=rf"$h_d={h_d:g}$ km",
        )

    plt.xlabel(r"Detector zenith angle $\theta$ [deg]")
    plt.ylabel(r"$\alpha-\theta$ [deg]")

    plt.title("Angular correction due to detector depth")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_inverse_consistency():

    plt.figure(figsize=(9, 6))

    theta_grid = torch.linspace(
        0.0,
        90.0,
        500,
        device=DEVICE,
        dtype=DTYPE,
    )

    for h_d in DEPTHS_KM:

        alpha_grid = theta_detector_to_alpha_surface(
            theta_grid,
            h_d_km=h_d,
        )

        theta_rec = alpha_surface_to_theta_detector(
            alpha_grid,
            h_d_km=h_d,
        )

        error = torch.abs(theta_rec - theta_grid)

        plt.semilogy(
            theta_grid.cpu().numpy(),
            torch.clamp(error, min=1.0e-16).cpu().numpy(),
            lw=2,
            label=rf"$h_d={h_d:g}$ km",
        )

        print(
            f"h_d={h_d:g} km | max inverse error = "
            f"{float(error.max().item()):.3e} deg"
        )

    plt.xlabel(r"Detector zenith angle $\theta$ [deg]")
    plt.ylabel(r"Inverse error [deg]")

    plt.title("Inverse consistency check")

    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Runner
# ============================================================

def run_angle_geometry_visual_test():

    print("\n" + "=" * 80)
    print("ANGLE GEOMETRY VISUAL TEST")
    print("=" * 80)

    for h_d in DEPTHS_KM:
        alpha_max = float(alpha_max_for_detector_depth(h_d).item())
        print(
            f"h_d = {h_d:g} km | "
            f"alpha_max = {alpha_max:.8f} deg"
        )

    plot_alpha_to_theta()
    plot_theta_to_alpha()
    plot_difference_theta_minus_alpha()
    plot_inverse_consistency()

    print("\nFinished angle geometry visual test.")


def test_angle_geometry_visual_diagnostics():
    run_angle_geometry_visual_test()


def run_mceq_geometry_visual_tests(verbose_traceback=False):
    return run_test_suite(
        [test_angle_geometry_visual_diagnostics],
        suite_name="mceq GEOMETRY VISUAL tests",
        verbose_traceback=verbose_traceback,
    )


if __name__ == "__main__":
    run_mceq_geometry_visual_tests(verbose_traceback=True)

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
Spyder-friendly tests and visual diagnostics for earth geometry utilities.

This script tests:

    1. Detector radius fraction.
    2. eta -> eta_prime transformation.
    3. Detector x-coordinate.
    4. Case-B chord length.
    5. eta-region classification.
    6. eta-range validation.
    7. Geometry plots.

Run directly in Spyder:

    python tpeanuts/tests/earth/test_earth_geometry.py
"""



from __future__ import annotations

import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt

import tpeanuts.util.constant as constant

from tpeanuts.util.test_utils import (
    run_test_suite,
    assert_true,
    assert_close,
    assert_raises,
)

from tpeanuts.earth.geometry import (
    detector_radius_fraction,
    eta_prime_from_eta,
    detector_x_coordinate,
    chord_length_case_b,
    classify_eta_regions,
    validate_eta_range,
)


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

THIS_FILE = Path(__file__).resolve()
TESTS_DIR = THIS_FILE.parents[1]

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
# tests
# ============================================================

def test_detector_radius_fraction_surface():
    depth_m = 0.0

    r_d = detector_radius_fraction(
        depth_m,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("\nDetector radius fraction at surface:")
    print("depth_m =", depth_m)
    print("r_d     =", r_d.item())

    assert_close(
        r_d.item(),
        1.0,
        atol=1.0e-14,
        rtol=1.0e-14,
        message="r_d should be 1 at the surface.",
    )


def test_detector_radius_fraction_depth():
    depth_m = 1000.0

    r_d = detector_radius_fraction(
        depth_m,
        device=DEVICE,
        dtype=DTYPE,
    )

    expected = 1.0 - depth_m / float(constant.R_E)

    print("\nDetector radius fraction at depth:")
    print("depth_m  =", depth_m)
    print("R_E      =", constant.R_E)
    print("r_d      =", r_d.item())
    print("expected =", expected)

    assert_close(
        r_d.item(),
        expected,
        atol=1.0e-14,
        rtol=1.0e-14,
        message="r_d formula mismatch.",
    )


def test_eta_prime_surface_identity():
    eta = torch.linspace(
        0.0,
        torch.pi / 2.0,
        20,
        device=DEVICE,
        dtype=DTYPE,
    )

    r_d = torch.tensor(1.0, device=DEVICE, dtype=DTYPE)

    eta_p = eta_prime_from_eta(
        eta,
        r_d,
    )

    print("\neta_prime at surface:")
    print("max|eta_prime - eta| =", torch.max(torch.abs(eta_p - eta)).item())

    assert_true(
        torch.allclose(eta_p, eta, atol=1.0e-12, rtol=1.0e-12),
        "For r_d=1 and eta in [0, pi/2], eta_prime should equal eta.",
    )


def test_eta_prime_depth_is_smaller():
    eta = torch.linspace(
        0.0,
        torch.pi / 2.0 - 1.0e-6,
        100,
        device=DEVICE,
        dtype=DTYPE,
    )

    r_d = detector_radius_fraction(
        1000.0,
        device=DEVICE,
        dtype=DTYPE,
    )

    eta_p = eta_prime_from_eta(
        eta,
        r_d,
    )

    print("\neta_prime at depth:")
    print("min eta_prime - eta =", torch.min(eta_p - eta).item())
    print("max eta_prime - eta =", torch.max(eta_p - eta).item())

    assert_true(
        (eta_p <= eta + 1.0e-12).all().item(),
        "For r_d<1, eta_prime should be <= eta in [0, pi/2].",
    )


def test_detector_x_coordinate():
    eta = torch.tensor(
        [0.0, torch.pi / 4.0, torch.pi / 2.0],
        device=DEVICE,
        dtype=DTYPE,
    )

    r_d = torch.tensor(1.0, device=DEVICE, dtype=DTYPE)

    x_d = detector_x_coordinate(
        eta,
        r_d,
    )

    expected = torch.cos(eta)

    print("\nDetector x-coordinate:")
    print("eta      =", eta)
    print("x_d      =", x_d)
    print("expected =", expected)

    assert_true(
        torch.allclose(x_d, expected, atol=1.0e-12, rtol=1.0e-12),
        "x_d should equal r_d cos(eta).",
    )


def test_chord_length_case_b_surface():
    eta = torch.linspace(
        torch.pi / 2.0,
        torch.pi,
        100,
        device=DEVICE,
        dtype=DTYPE,
    )

    r_d = torch.tensor(1.0, device=DEVICE, dtype=DTYPE)

    dx = chord_length_case_b(
        eta,
        r_d,
    )

    print("\nCase-B chord length at surface:")
    print("min dx =", torch.min(dx).item())
    print("max dx =", torch.max(dx).item())

    assert_true(
        torch.isfinite(dx).all().item(),
        "Chord length contains NaN or Inf.",
    )

    assert_true(
        (dx >= -1.0e-12).all().item(),
        "Chord length should be non-negative up to numerical tolerance.",
    )

    assert_close(
        dx[0].item(),
        0.0,
        atol=1.0e-10,
        rtol=1.0e-10,
        message="At eta=pi/2 and surface, chord length should be zero.",
    )


def test_chord_length_case_b_depth():
    eta = torch.linspace(
        torch.pi / 2.0,
        torch.pi,
        100,
        device=DEVICE,
        dtype=DTYPE,
    )

    r_d = detector_radius_fraction(
        1000.0,
        device=DEVICE,
        dtype=DTYPE,
    )

    dx = chord_length_case_b(
        eta,
        r_d,
    )

    print("\nCase-B chord length at detector depth:")
    print("min dx =", torch.min(dx).item())
    print("max dx =", torch.max(dx).item())

    assert_true(
        torch.isfinite(dx).all().item(),
        "Chord length contains NaN or Inf.",
    )

    assert_true(
        (dx >= -1.0e-12).all().item(),
        "Chord length should be non-negative up to numerical tolerance.",
    )


def test_classify_eta_regions_surface():
    eta = torch.tensor(
        [
            0.0,
            0.4,
            torch.pi / 2.0,
            2.0,
            torch.pi,
        ],
        device=DEVICE,
        dtype=DTYPE,
    )

    above, mask_a, mask_b = classify_eta_regions(
        eta,
        depth_m=0.0,
    )

    print("\nRegion classification at surface:")
    print("eta    =", eta)
    print("above  =", above)
    print("mask A =", mask_a)
    print("mask B =", mask_b)

    assert_true(above[2:].all().item(), "At depth=0, eta>=pi/2 should be above horizon.")
    assert_true(mask_a[:2].all().item(), "eta<pi/2 should be Case A.")
    assert_true((~mask_b).all().item(), "At depth=0, above-horizon values should not be Case B.")


def test_classify_eta_regions_depth():
    eta = torch.tensor(
        [
            0.0,
            0.4,
            torch.pi / 2.0,
            2.0,
            torch.pi,
        ],
        device=DEVICE,
        dtype=DTYPE,
    )

    above, mask_a, mask_b = classify_eta_regions(
        eta,
        depth_m=1000.0,
    )

    print("\nRegion classification at depth:")
    print("eta    =", eta)
    print("above  =", above)
    print("mask A =", mask_a)
    print("mask B =", mask_b)

    assert_true((~above).all().item(), "At depth>0 there should be no above-horizon identity mask.")
    assert_true(mask_a[:2].all().item(), "eta<pi/2 should be Case A.")
    assert_true(mask_b[2:].all().item(), "eta>=pi/2 should be Case B.")


def test_validate_eta_range_valid():
    eta = torch.linspace(
        0.0,
        torch.pi,
        100,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("\nValidating eta range [0, pi]")
    validate_eta_range(eta)

    assert_true(True, "Valid eta range accepted.")


def test_validate_eta_range_invalid_low():
    eta = torch.tensor(
        [-0.1, 0.2],
        device=DEVICE,
        dtype=DTYPE,
    )

    print("\nValidating invalid eta range, low value:")
    print(eta)

    assert_raises(
        ValueError,
        validate_eta_range,
        eta,
    )


def test_validate_eta_range_invalid_high():
    eta = torch.tensor(
        [0.2, torch.pi + 0.1],
        device=DEVICE,
        dtype=DTYPE,
    )

    print("\nValidating invalid eta range, high value:")
    print(eta)

    assert_raises(
        ValueError,
        validate_eta_range,
        eta,
    )


# ============================================================
# Visualization
# ============================================================

def plot_eta_prime_transformation(savefig=False):
    eta = torch.linspace(
        0.0,
        torch.pi / 2.0,
        500,
        device=DEVICE,
        dtype=DTYPE,
    )

    depths = [0.0, 1000.0, 3000.0, 10000.0]

    plt.figure(figsize=(8, 5))

    for depth_m in depths:
        r_d = detector_radius_fraction(
            depth_m,
            device=DEVICE,
            dtype=DTYPE,
        )

        eta_p = eta_prime_from_eta(
            eta,
            r_d,
        )

        plt.plot(
            eta.detach().cpu().numpy(),
            eta_p.detach().cpu().numpy(),
            label=rf"$d={depth_m:.0f}$ m",
        )

    plt.xlabel(r"Nadir angle $\eta$ [rad]")
    plt.ylabel(r"Transformed angle $\eta'$ [rad]")
    plt.title(r"Geometry transformation $\eta'=\arcsin(r_d\sin\eta)$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "earth_geometry_eta_prime.png")
    if savefig:
        plt.savefig(path, dpi=200)
    plt.show()

    if savefig:
        print(f"\nSaved plot: {path}")


def plot_detector_x_coordinate(savefig=False):
    eta = torch.linspace(
        0.0,
        torch.pi,
        700,
        device=DEVICE,
        dtype=DTYPE,
    )

    depths = [0.0, 1000.0, 3000.0, 10000.0]

    plt.figure(figsize=(8, 5))

    for depth_m in depths:
        r_d = detector_radius_fraction(
            depth_m,
            device=DEVICE,
            dtype=DTYPE,
        )

        x_d = detector_x_coordinate(
            eta,
            r_d,
        )

        plt.plot(
            eta.detach().cpu().numpy(),
            x_d.detach().cpu().numpy(),
            label=rf"$d={depth_m:.0f}$ m",
        )

    plt.xlabel(r"Nadir angle $\eta$ [rad]")
    plt.ylabel(r"Detector coordinate $x_d=r_d\cos\eta$")
    plt.title("Detector trajectory coordinate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "earth_geometry_detector_x.png")
    if savefig:
        plt.savefig(path, dpi=200)
    plt.show()

    if savefig:
        print(f"Saved plot: {path}")


def plot_case_b_chord_length(savefig=False):
    eta = torch.linspace(
        torch.pi / 2.0,
        torch.pi,
        500,
        device=DEVICE,
        dtype=DTYPE,
    )

    depths = [0.0, 1000.0, 3000.0, 10000.0]

    plt.figure(figsize=(8, 5))

    for depth_m in depths:
        r_d = detector_radius_fraction(
            depth_m,
            device=DEVICE,
            dtype=DTYPE,
        )

        dx = chord_length_case_b(
            eta,
            r_d,
        )

        plt.plot(
            eta.detach().cpu().numpy(),
            dx.detach().cpu().numpy(),
            label=rf"$d={depth_m:.0f}$ m",
        )

    plt.xlabel(r"Nadir angle $\eta$ [rad]")
    plt.ylabel(r"Case-B path length $\Delta x$")
    plt.title("Case-B earth chord length")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "earth_geometry_case_b_chord.png")
    if savefig:
        plt.savefig(path, dpi=200)
    plt.show()

    if savefig:
        print(f"Saved plot: {path}")


def test_visualization_outputs(savefig=False):
    plot_eta_prime_transformation(savefig=savefig)
    plot_detector_x_coordinate(savefig=savefig)
    plot_case_b_chord_length(savefig=savefig)

    expected_files = [
        "earth_geometry_eta_prime.png",
        "earth_geometry_detector_x.png",
        "earth_geometry_case_b_chord.png",
    ]

    for filename in expected_files:
        path = os.path.join(OUTPUT_DIR, filename)

        if savefig:
            assert_true(
                os.path.isfile(path),
                f"Plot was not created: {path}",
            )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    tests = [
        test_detector_radius_fraction_surface,
        test_detector_radius_fraction_depth,
        test_eta_prime_surface_identity,
        test_eta_prime_depth_is_smaller,
        test_detector_x_coordinate,
        test_chord_length_case_b_surface,
        test_chord_length_case_b_depth,
        test_classify_eta_regions_surface,
        test_classify_eta_regions_depth,
        test_validate_eta_range_valid,
        test_validate_eta_range_invalid_low,
        test_validate_eta_range_invalid_high,
        test_visualization_outputs,
    ]

    run_test_suite(
        tests,
        suite_name="earth GEOMETRY tests",
        verbose_traceback=True,
    )

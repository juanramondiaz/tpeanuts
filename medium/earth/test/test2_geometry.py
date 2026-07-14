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
Pytest-compatible tests for tpeanuts.medium.earth.geometry.

The diagnostic plots from the historical backup tests live in notebooks; this
file keeps only fast numerical sanity checks that can run automatically.
"""

from __future__ import annotations

import math

import pytest
import torch

import tpeanuts.util.constant as constant
from tpeanuts.medium.earth.geometry import (
    build_earth_trajectory,
    chord_length_case_b,
    classify_eta_regions,
    detector_radius_fraction,
    detector_x_coordinate,
    eta_prime_from_eta,
    validate_eta_range,
)
from tpeanuts.medium.earth.profile import EarthParameters, EarthProfile
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.test_utils import assert_close


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _two_shell_profile() -> EarthProfile:
    """Synthetic two-shell profile: constant density 2.0 for r<0.5, 1.0 for 0.5<r<=1.0."""
    rj = torch.tensor([0.5, 1.0], device=DEVICE, dtype=DTYPE)
    coefficients = torch.tensor(
        [[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        device=DEVICE,
        dtype=DTYPE,
    )
    params = EarthParameters(
        profile_perturbative_name="even_power",
        profile_perturbative_kwargs={"rj": rj, "coefficients": coefficients},
    )
    return EarthProfile(params=params, context=RuntimeContext.resolve(DEVICE, DTYPE))


def test_detector_radius_fraction_surface():
    r_d = detector_radius_fraction(0.0, device=DEVICE, dtype=DTYPE)

    assert_close(r_d, torch.tensor(1.0, dtype=DTYPE), atol=1.0e-14, rtol=1.0e-14, name="r_d at surface")


def test_detector_radius_fraction_depth_formula():
    depth_m = 1000.0

    r_d = detector_radius_fraction(depth_m, device=DEVICE, dtype=DTYPE)
    expected = 1.0 - depth_m / float(constant.R_E)

    assert_close(r_d, torch.tensor(expected, dtype=DTYPE), atol=1.0e-14, rtol=1.0e-14, name="r_d depth formula")


def test_eta_prime_surface_identity():
    eta = torch.linspace(0.0, math.pi / 2.0, 20, device=DEVICE, dtype=DTYPE)
    r_d = torch.tensor(1.0, device=DEVICE, dtype=DTYPE)

    eta_p = eta_prime_from_eta(eta, r_d)

    assert_close(eta_p, eta, atol=1.0e-12, rtol=1.0e-12, name="eta_prime equals eta at surface")


def test_eta_prime_exact_formula():
    eta = torch.tensor([0.1, 0.5, 1.0, 1.4], device=DEVICE, dtype=DTYPE)
    r_d = torch.tensor(0.85, device=DEVICE, dtype=DTYPE)

    eta_p = eta_prime_from_eta(eta, r_d)
    expected = torch.asin(r_d * torch.sin(eta))

    assert_close(eta_p, expected, name="eta_prime closed-form formula")


def test_eta_prime_depth_is_smaller_or_equal():
    eta = torch.linspace(0.0, math.pi / 2.0 - 1.0e-6, 100, device=DEVICE, dtype=DTYPE)
    r_d = detector_radius_fraction(1000.0, device=DEVICE, dtype=DTYPE)

    eta_p = eta_prime_from_eta(eta, r_d)

    assert torch.all(eta_p <= eta + 1.0e-12)


def test_detector_x_coordinate_formula():
    eta = torch.tensor([0.0, math.pi / 4.0, math.pi / 2.0, math.pi], device=DEVICE, dtype=DTYPE)
    r_d = torch.tensor(0.9, device=DEVICE, dtype=DTYPE)

    x_d = detector_x_coordinate(eta, r_d)
    expected = r_d * torch.cos(eta)

    assert_close(x_d, expected, name="x_d = r_d cos(eta)")
    assert float(x_d[-1]) < 0.0, "x_d should be negative for eta beyond pi/2."


def test_chord_length_case_b_surface_boundary():
    eta = torch.linspace(math.pi / 2.0, math.pi, 100, device=DEVICE, dtype=DTYPE)
    r_d = torch.tensor(1.0, device=DEVICE, dtype=DTYPE)

    dx = chord_length_case_b(eta, r_d)

    assert torch.all(torch.isfinite(dx))
    assert torch.all(dx >= -1.0e-12)
    assert_close(dx[0], torch.tensor(0.0, dtype=DTYPE), atol=1.0e-10, rtol=1.0e-10, name="dx=0 at eta=pi/2, r_d=1")


def test_chord_length_case_b_exact_formula():
    eta = torch.tensor([1.6, 2.1, 2.8, math.pi], device=DEVICE, dtype=DTYPE)
    r_d = torch.tensor(0.95, device=DEVICE, dtype=DTYPE)

    dx = chord_length_case_b(eta, r_d)
    expected = r_d * torch.cos(eta) + torch.sqrt(1.0 - r_d**2 * torch.sin(eta) ** 2)

    assert_close(dx, expected, name="chord_length_case_b closed-form formula")


def test_chord_length_case_b_depth_is_finite_and_nonnegative():
    eta = torch.linspace(math.pi / 2.0, math.pi, 100, device=DEVICE, dtype=DTYPE)
    r_d = detector_radius_fraction(1000.0, device=DEVICE, dtype=DTYPE)

    dx = chord_length_case_b(eta, r_d)

    assert torch.all(torch.isfinite(dx))
    assert torch.all(dx >= -1.0e-12)


def test_classify_eta_regions_surface():
    eta = torch.tensor([0.0, 0.4, math.pi / 2.0, 2.0, math.pi], device=DEVICE, dtype=DTYPE)

    above, mask_a, mask_b = classify_eta_regions(eta, depth_m=0.0)

    assert torch.all(above[2:])
    assert torch.all(mask_a[:2])
    assert torch.all(~mask_b)


def test_classify_eta_regions_depth():
    eta = torch.tensor([0.0, 0.4, math.pi / 2.0, 2.0, math.pi], device=DEVICE, dtype=DTYPE)

    above, mask_a, mask_b = classify_eta_regions(eta, depth_m=1000.0)

    assert torch.all(~above)
    assert torch.all(mask_a[:2])
    assert torch.all(mask_b[2:])


def test_classify_eta_regions_masks_partition_domain():
    eta = torch.linspace(0.0, math.pi, 200, device=DEVICE, dtype=DTYPE)

    for depth_m in (0.0, 500.0):
        above, mask_a, mask_b = classify_eta_regions(eta, depth_m=depth_m)
        total = above.to(torch.int64) + mask_a.to(torch.int64) + mask_b.to(torch.int64)
        assert torch.all(total == 1), f"masks do not partition eta domain at depth_m={depth_m}"


def test_validate_eta_range_valid_and_boundary():
    eta = torch.linspace(0.0, math.pi, 100, device=DEVICE, dtype=DTYPE)
    validate_eta_range(eta)

    boundary = torch.tensor([0.0, math.pi], device=DEVICE, dtype=DTYPE)
    validate_eta_range(boundary)


def test_validate_eta_range_invalid_low():
    eta = torch.tensor([-0.1, 0.2], device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="eta must be between 0 and pi"):
        validate_eta_range(eta)


def test_validate_eta_range_invalid_high():
    eta = torch.tensor([0.2, math.pi + 0.1], device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="eta must be between 0 and pi"):
        validate_eta_range(eta)


def test_build_earth_trajectory_earth_crossing_mode():
    profile = _two_shell_profile()
    eta = torch.tensor(0.3, device=DEVICE, dtype=DTYPE)

    trajectory = build_earth_trajectory(
        profile_earth=profile,
        eta=eta,
        depth_m=0.0,
        nsteps=50,
        method="midpoint",
        device=DEVICE,
        dtype=DTYPE,
        evolution_scale_m=constant.R_E,
    )

    assert trajectory.meta["mode"] == "earth_crossing"
    assert trajectory.x.shape == (51,)
    assert trajectory.dx_evolution.shape == (50,)
    assert trajectory.sample_x.shape == (50,)
    assert torch.all(torch.isfinite(trajectory.x))
    assert torch.all(torch.isfinite(trajectory.dx_evolution))

    x_d = detector_x_coordinate(eta, torch.tensor(1.0, device=DEVICE, dtype=DTYPE))
    assert_close(trajectory.x[-1], x_d, name="earth_crossing trajectory ends at detector coordinate")


def test_build_earth_trajectory_local_constant_mode():
    profile = _two_shell_profile()
    eta = torch.tensor(2.0, device=DEVICE, dtype=DTYPE)

    trajectory = build_earth_trajectory(
        profile_earth=profile,
        eta=eta,
        depth_m=0.0,
        nsteps=30,
        method="midpoint",
        device=DEVICE,
        dtype=DTYPE,
        evolution_scale_m=constant.R_E,
    )

    r_d = torch.tensor(1.0, device=DEVICE, dtype=DTYPE)
    delta_x = chord_length_case_b(eta, r_d)

    assert trajectory.meta["mode"] == "local_constant"
    assert_close(trajectory.x[0], torch.tensor(0.0, dtype=DTYPE), atol=1.0e-14, rtol=1.0e-14, name="local_constant starts at 0")
    assert_close(trajectory.x[-1], delta_x, name="local_constant trajectory ends at case-B chord length")


def test_build_earth_trajectory_rejects_non_scalar_eta():
    profile = _two_shell_profile()
    eta = torch.tensor([0.1, 0.2], device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="only supports scalar eta"):
        build_earth_trajectory(
            profile_earth=profile,
            eta=eta,
            depth_m=0.0,
            nsteps=10,
            method="midpoint",
            device=DEVICE,
            dtype=DTYPE,
            evolution_scale_m=constant.R_E,
        )


def test_build_earth_trajectory_rejects_non_positive_evolution_scale():
    profile = _two_shell_profile()

    with pytest.raises(ValueError, match="evolution_scale_m must be positive"):
        build_earth_trajectory(
            profile_earth=profile,
            eta=torch.tensor(0.3, device=DEVICE, dtype=DTYPE),
            depth_m=0.0,
            nsteps=10,
            method="midpoint",
            device=DEVICE,
            dtype=DTYPE,
            evolution_scale_m=0.0,
        )


def test_build_earth_trajectory_rejects_invalid_nsteps():
    profile = _two_shell_profile()

    with pytest.raises(ValueError, match="nsteps must be at least 1"):
        build_earth_trajectory(
            profile_earth=profile,
            eta=torch.tensor(0.3, device=DEVICE, dtype=DTYPE),
            depth_m=0.0,
            nsteps=0,
            method="midpoint",
            device=DEVICE,
            dtype=DTYPE,
            evolution_scale_m=constant.R_E,
        )

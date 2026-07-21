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

"""Pytest-compatible checks for core flux utilities."""

from __future__ import annotations

import pytest
import torch

from tpeanuts.core.common.flux import (
    flux_integrated,
    flux_integrated_angular,
    flux_integrated_coordinate,
    flux_state,
    flux_transition,
)
from tpeanuts.util.test_utils import assert_close


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_flux_transition_applies_beta_alpha_matrix_to_initial_flux():
    probability = torch.eye(3, device=DEVICE, dtype=DTYPE).expand(2, 3, 3)
    initial = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=DEVICE, dtype=DTYPE
    )
    assert_close(flux_transition(probability, initial), initial, name="identity flux transition")


def test_flux_integrated_coordinate_reduces_requested_non_flavour_axis():
    coordinate = torch.tensor([0.0, 1.0, 2.0], device=DEVICE, dtype=DTYPE)
    flux = torch.ones((2, 3, 3), device=DEVICE, dtype=DTYPE)
    result = flux_integrated_coordinate(flux, coordinate, dim=1)
    assert_close(result, 2.0 * torch.ones((2, 3), device=DEVICE, dtype=DTYPE), name="coordinate integration")


def test_flux_from_probability_scalar_flux_scales_all_flavours():
    probability = torch.tensor([0.2, 0.3, 0.5], device=DEVICE, dtype=DTYPE)

    flux = flux_state(probability, 10.0)

    assert_close(flux, torch.tensor([2.0, 3.0, 5.0], device=DEVICE, dtype=DTYPE), name="scalar flux scaling")


def test_flux_from_probability_with_spectrum_scales_probability_grid():
    probability = torch.tensor(
        [
            [0.2, 0.3, 0.5],
            [0.1, 0.4, 0.5],
        ],
        device=DEVICE,
        dtype=DTYPE,
    )
    flux = torch.tensor([10.0, 20.0], device=DEVICE, dtype=DTYPE)
    spectrum = torch.tensor([0.5, 2.0], device=DEVICE, dtype=DTYPE)

    out = flux_state(probability, flux, spectrum)
    expected = probability * (flux * spectrum)[:, None]

    assert out.shape == probability.shape
    assert_close(out, expected, name="flux and spectrum scaling")


def test_flux_from_probability_preserves_total_flux_for_normalized_probabilities():
    probability = torch.tensor(
        [
            [0.2, 0.3, 0.5],
            [0.1, 0.4, 0.5],
            [0.7, 0.2, 0.1],
        ],
        device=DEVICE,
        dtype=DTYPE,
    )
    flux = torch.tensor([10.0, 20.0, 5.0], device=DEVICE, dtype=DTYPE)

    out = flux_state(probability, flux)

    assert_close(out.sum(dim=-1), flux, name="total flux conservation for normalized probabilities")


def test_flux_from_probability_broadcasts_source_and_energy_axes():
    probability = torch.tensor(
        [
            [
                [0.2, 0.3, 0.5],
                [0.1, 0.4, 0.5],
                [0.3, 0.3, 0.4],
                [0.5, 0.2, 0.3],
            ],
            [
                [0.6, 0.3, 0.1],
                [0.2, 0.2, 0.6],
                [0.1, 0.7, 0.2],
                [0.4, 0.4, 0.2],
            ],
        ],
        device=DEVICE,
        dtype=DTYPE,
    )
    flux = torch.tensor([10.0, 20.0], device=DEVICE, dtype=DTYPE)
    spectrum = torch.tensor(
        [
            [1.0, 0.5, 0.25, 0.125],
            [2.0, 1.0, 0.5, 0.25],
        ],
        device=DEVICE,
        dtype=DTYPE,
    )

    out = flux_state(probability, flux, spectrum)
    expected = probability * flux[:, None, None] * spectrum[:, :, None]

    assert out.shape == (2, 4, 3)
    assert_close(out, expected, name="source-energy-flavour flux grid")


def test_flux_from_probability_accepts_full_grid_flux():
    probability = torch.full((2, 3, 3), 1.0 / 3.0, device=DEVICE, dtype=DTYPE)
    flux = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        device=DEVICE,
        dtype=DTYPE,
    )

    out = flux_state(probability, flux)
    expected = probability * flux[..., None]

    assert_close(out, expected, name="full leading-grid flux")


def test_flux_from_probability_rejects_probability_without_three_flavours():
    with pytest.raises(ValueError, match="final flavour dimension 3"):
        flux_state(torch.ones((2, 2), device=DEVICE, dtype=DTYPE), 1.0)


def test_flux_from_probability_rejects_too_many_flux_dimensions():
    probability = torch.ones((2, 3), device=DEVICE, dtype=DTYPE) / 3.0
    flux = torch.ones((2, 3, 1), device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="more dimensions than probability"):
        flux_state(probability, flux)


def test_flux_from_probability_rejects_too_many_spectrum_dimensions():
    probability = torch.ones((2, 3), device=DEVICE, dtype=DTYPE) / 3.0
    spectrum = torch.ones((2, 3, 1), device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="more dimensions than probability"):
        flux_state(probability, 1.0, spectrum)


def test_flux_integrated_matches_manual_trapezoidal_rule():
    E = torch.tensor([100.0, 200.0, 500.0, 1000.0], device=DEVICE, dtype=DTYPE)
    flux = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [0.5, 1.0, 1.5],
            [0.1, 0.2, 0.3],
        ],
        device=DEVICE,
        dtype=DTYPE,
    )

    out = flux_integrated(flux, E)
    expected = torch.trapezoid(flux, x=E, dim=-2)

    assert out.shape == (3,)
    assert_close(out, expected, name="flux_integrated matches torch.trapezoid")


def test_flux_integrated_constant_flux_over_uniform_grid():
    E = torch.linspace(100.0, 1100.0, 11, device=DEVICE, dtype=DTYPE)
    flux = torch.ones((11, 3), device=DEVICE, dtype=DTYPE) * 2.0

    out = flux_integrated(flux, E)

    assert_close(
        out,
        torch.full((3,), 2000.0, device=DEVICE, dtype=DTYPE),
        name="constant flux integrates to flux * energy range",
    )


def test_flux_integrated_preserves_angle_axis_with_explicit_energy_dim():
    E = torch.tensor([100.0, 500.0, 1000.0], device=DEVICE, dtype=DTYPE)
    flux = torch.rand((3, 4, 3), device=DEVICE, dtype=DTYPE)  # (E, angle, flavour)

    out = flux_integrated(flux, E, energy_dim=-3)
    expected = torch.trapezoid(flux, x=E, dim=-3)

    assert out.shape == (4, 3)
    assert_close(out, expected, name="flux_integrated preserves angle axis")


def test_flux_integrated_rejects_flux_without_three_flavours():
    E = torch.tensor([100.0, 200.0], device=DEVICE, dtype=DTYPE)
    flux = torch.ones((2, 2), device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="final flavour dimension 3"):
        flux_integrated(flux, E)


def test_flux_integrated_rejects_energy_dim_on_flavour_axis():
    E = torch.tensor([100.0, 200.0, 300.0], device=DEVICE, dtype=DTYPE)
    flux = torch.ones((4, 3), device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="must not select the flavour axis"):
        flux_integrated(flux, E, energy_dim=-1)


def test_flux_integrated_rejects_mismatched_energy_grid():
    E = torch.tensor([100.0, 200.0], device=DEVICE, dtype=DTYPE)
    flux = torch.ones((4, 3), device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="must match E_grid_MeV"):
        flux_integrated(flux, E)


# ---------------------------------------------------------------------------
# N=4 (3+1 sterile) checks -- flux_state, flux_integrated
# ---------------------------------------------------------------------------


def test_flux_state_accepts_four_flavours():
    probability = torch.tensor([0.4, 0.3, 0.2, 0.1], device=DEVICE, dtype=DTYPE)

    flux = flux_state(probability, 10.0)

    assert flux.shape == (4,)
    assert_close(flux, torch.tensor([4.0, 3.0, 2.0, 1.0], device=DEVICE, dtype=DTYPE), name="four-flavour scalar flux scaling")


def test_flux_integrated_accepts_four_flavours():
    E = torch.tensor([100.0, 200.0, 500.0, 1000.0], device=DEVICE, dtype=DTYPE)
    flux = torch.rand((4, 4), device=DEVICE, dtype=DTYPE)

    out = flux_integrated(flux, E)
    expected = torch.trapezoid(flux, x=E, dim=-2)

    assert out.shape == (4,)
    assert_close(out, expected, name="four-flavour flux_integrated")


# ---------------------------------------------------------------------------
# flux_integrated_angular
# ---------------------------------------------------------------------------


def test_flux_integrated_angular_matches_manual_trapezoidal_rule():
    theta = torch.tensor([0.0, 45.0, 90.0, 135.0, 180.0], device=DEVICE, dtype=DTYPE)
    flux = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0],
        ],
        device=DEVICE,
        dtype=DTYPE,
    )

    out = flux_integrated_angular(flux, theta)

    theta_rad = torch.deg2rad(theta)
    sin_theta = torch.sin(theta_rad)
    expected = 2.0 * torch.pi * torch.trapezoid(flux * sin_theta[:, None], x=theta_rad, dim=-2)

    assert out.shape == (3,)
    assert_close(out, expected, name="flux_integrated_angular matches manual trapezoidal rule")


def test_flux_integrated_angular_isotropic_flux_matches_closed_form():
    # A constant (isotropic) flux integrated over the full sky, assuming
    # azimuthal symmetry, equals flux * 4*pi (the full solid angle).
    theta = torch.linspace(0.0, 180.0, 361, device=DEVICE, dtype=DTYPE)
    flux = torch.ones((361, 3), device=DEVICE, dtype=DTYPE) * 2.0

    out = flux_integrated_angular(flux, theta)

    assert_close(
        out,
        torch.full((3,), 2.0 * 4.0 * torch.pi, device=DEVICE, dtype=DTYPE),
        atol=1.0e-3,
        rtol=1.0e-6,
        name="isotropic flux integrates to flux * 4*pi",
    )


def test_flux_integrated_angular_preserves_energy_axis_with_explicit_angular_dim():
    theta = torch.tensor([0.0, 90.0, 180.0], device=DEVICE, dtype=DTYPE)
    flux = torch.rand((4, 3, 3), device=DEVICE, dtype=DTYPE)  # (E, angle, flavour)

    out = flux_integrated_angular(flux, theta, angular_dim=-2)
    theta_rad = torch.deg2rad(theta)
    sin_theta = torch.sin(theta_rad)
    expected = 2.0 * torch.pi * torch.trapezoid(flux * sin_theta[:, None], x=theta_rad, dim=-2)

    assert out.shape == (4, 3)
    assert_close(out, expected, name="flux_integrated_angular preserves energy axis")


def test_flux_integrated_angular_accepts_four_flavours():
    theta = torch.tensor([10.0, 90.0, 170.0], device=DEVICE, dtype=DTYPE)
    flux = torch.rand((3, 4), device=DEVICE, dtype=DTYPE)

    out = flux_integrated_angular(flux, theta)

    theta_rad = torch.deg2rad(theta)
    sin_theta = torch.sin(theta_rad)
    expected = 2.0 * torch.pi * torch.trapezoid(flux * sin_theta[:, None], x=theta_rad, dim=-2)

    assert out.shape == (4,)
    assert_close(out, expected, name="four-flavour flux_integrated_angular")


def test_flux_integrated_angular_rejects_flux_without_three_or_four_flavours():
    theta = torch.tensor([0.0, 90.0], device=DEVICE, dtype=DTYPE)
    flux = torch.ones((2, 2), device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="3 or 4"):
        flux_integrated_angular(flux, theta)


def test_flux_integrated_angular_rejects_angular_dim_on_flavour_axis():
    theta = torch.tensor([0.0, 90.0, 180.0], device=DEVICE, dtype=DTYPE)
    flux = torch.ones((4, 3), device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="must not select the flavour axis"):
        flux_integrated_angular(flux, theta, angular_dim=-1)


def test_flux_integrated_angular_rejects_mismatched_theta_grid():
    theta = torch.tensor([0.0, 90.0], device=DEVICE, dtype=DTYPE)
    flux = torch.ones((4, 3), device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="must match theta_deg"):
        flux_integrated_angular(flux, theta)

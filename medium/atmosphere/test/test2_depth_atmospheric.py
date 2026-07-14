#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pytest-compatible tests for tpeanuts.medium.atmosphere.depth."""

from __future__ import annotations

import math

import pytest
import torch

from tpeanuts.medium.atmosphere.depth import (
    alpha_deg_to_cos,
    atmosphere_slant_depth,
    atmosphere_vertical_depth,
    compute_dXdh,
    interpolate_flux_at_Xobs,
)


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_alpha_deg_to_cos_scalar_and_tensor_values():
    assert math.isclose(alpha_deg_to_cos(60.0), 0.5, rel_tol=1.0e-14, abs_tol=1.0e-14)

    alpha = torch.tensor([0.0, 30.0, 60.0], dtype=DTYPE)
    cos_alpha = alpha_deg_to_cos(alpha)
    expected = torch.cos(torch.deg2rad(alpha))

    torch.testing.assert_close(cos_alpha, expected, rtol=1.0e-14, atol=1.0e-14)


def test_alpha_deg_to_cos_rejects_horizontal_and_upward_angles():
    for alpha in (90.0, 120.0):
        with pytest.raises(ValueError):
            alpha_deg_to_cos(alpha)


def test_vertical_depth_constant_density_matches_closed_form():
    h = torch.tensor([0.0, 1.0, 2.0, 5.0], device=DEVICE, dtype=DTYPE)
    rho = torch.full_like(h, 1.2e-3)

    X = atmosphere_vertical_depth(h, rho, device=DEVICE, dtype=DTYPE)
    expected = rho[0] * (h[-1] - h) * 1.0e5

    torch.testing.assert_close(X, expected, rtol=1.0e-14, atol=1.0e-12)
    assert torch.all(torch.diff(X) <= 0.0)
    torch.testing.assert_close(X[-1], torch.tensor(0.0, device=DEVICE, dtype=DTYPE), rtol=0.0, atol=0.0)


def test_vertical_depth_matches_manual_trapezoid_for_nonuniform_grid():
    h = torch.tensor([0.0, 0.5, 2.0, 5.0], device=DEVICE, dtype=DTYPE)
    rho = torch.tensor([1.0e-3, 0.8e-3, 0.3e-3, 0.1e-3], device=DEVICE, dtype=DTYPE)

    X = atmosphere_vertical_depth(h, rho, device=DEVICE, dtype=DTYPE)
    segment = 0.5 * (rho[:-1] + rho[1:]) * torch.diff(h * 1.0e5)
    expected = torch.tensor(
        [segment.sum(), segment[1:].sum(), segment[2:].sum(), 0.0],
        device=DEVICE,
        dtype=DTYPE,
    )

    torch.testing.assert_close(X, expected, rtol=1.0e-14, atol=1.0e-12)


def test_vertical_depth_rejects_bad_shapes_and_nonmonotonic_altitude():
    h = torch.tensor([0.0, 1.0, 0.5], device=DEVICE, dtype=DTYPE)
    rho = torch.ones_like(h)

    with pytest.raises(ValueError):
        atmosphere_vertical_depth(h, rho, device=DEVICE, dtype=DTYPE)
    with pytest.raises(ValueError):
        atmosphere_vertical_depth(h.reshape(1, 3), rho, device=DEVICE, dtype=DTYPE)
    with pytest.raises(ValueError):
        atmosphere_vertical_depth(torch.tensor([0.0, 1.0], device=DEVICE, dtype=DTYPE), rho, device=DEVICE, dtype=DTYPE)


def test_slant_depth_projects_by_secant_and_broadcasts():
    X = torch.tensor([1000.0, 500.0, 0.0], device=DEVICE, dtype=DTYPE)
    alpha = torch.tensor([0.0, 60.0], device=DEVICE, dtype=DTYPE)

    X_slant = atmosphere_slant_depth(X, alpha, device=DEVICE, dtype=DTYPE)
    expected = X[None, :] / torch.cos(torch.deg2rad(alpha))[:, None]

    assert X_slant.shape == (2, 3)
    torch.testing.assert_close(X_slant, expected, rtol=1.0e-14, atol=1.0e-12)


def test_compute_dXdh_constant_density_depth_derivative():
    h = torch.linspace(0.0, 5.0, 6, device=DEVICE, dtype=DTYPE)
    rho = torch.full_like(h, 1.2e-3)
    X = atmosphere_vertical_depth(h, rho, device=DEVICE, dtype=DTYPE)

    dXdh = compute_dXdh(X, h, device=DEVICE, dtype=DTYPE)

    torch.testing.assert_close(dXdh, torch.full_like(h, -120.0), rtol=1.0e-14, atol=1.0e-12)


def test_interpolate_flux_at_Xobs_linear_and_log_modes():
    X = torch.tensor([0.0, 10.0, 20.0], device=DEVICE, dtype=DTYPE)
    flux = torch.tensor(
        [
            [1.0, 10.0],
            [3.0, 30.0],
            [5.0, 50.0],
        ],
        device=DEVICE,
        dtype=DTYPE,
    )

    linear = interpolate_flux_at_Xobs(X, flux, 5.0, log_interp=False, device=DEVICE, dtype=DTYPE)
    log_interp = interpolate_flux_at_Xobs(X, flux, 5.0, log_interp=True, device=DEVICE, dtype=DTYPE)

    torch.testing.assert_close(linear, torch.tensor([2.0, 20.0], device=DEVICE, dtype=DTYPE), rtol=1.0e-14, atol=1.0e-12)
    torch.testing.assert_close(log_interp, torch.sqrt(flux[0] * flux[1]), rtol=1.0e-14, atol=1.0e-12)


def test_interpolate_flux_at_Xobs_rejects_invalid_inputs():
    X = torch.tensor([0.0, 10.0, 5.0], device=DEVICE, dtype=DTYPE)
    flux = torch.ones(3, 2, device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError):
        interpolate_flux_at_Xobs(X, flux, 3.0, device=DEVICE, dtype=DTYPE)
    with pytest.raises(ValueError):
        interpolate_flux_at_Xobs(torch.tensor([0.0, 10.0], device=DEVICE, dtype=DTYPE), flux, 3.0, device=DEVICE, dtype=DTYPE)
    with pytest.raises(ValueError):
        interpolate_flux_at_Xobs(torch.tensor([0.0, 10.0, 20.0], device=DEVICE, dtype=DTYPE), flux, 25.0, device=DEVICE, dtype=DTYPE)

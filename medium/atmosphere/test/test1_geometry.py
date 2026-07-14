#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pytest-compatible tests for tpeanuts.medium.atmosphere.geometry."""

from __future__ import annotations

import math

import pytest
import torch

from tpeanuts.medium.atmosphere.geometry import (
    alpha_max_for_detector_depth,
    alpha_surface_to_theta_detector,
    altitude_along_detector_path,
    atmosphere_path_grid,
    atmosphere_path_length,
    eta_to_theta,
    theta_detector_to_alpha_surface,
    theta_to_eta,
    total_path_length,
    underground_path_length,
)
from tpeanuts.util.constant import R_E_KM


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_theta_eta_roundtrip_and_known_limits():
    theta = torch.tensor([0.0, 90.0, 180.0], device=DEVICE, dtype=DTYPE)

    eta = theta_to_eta(theta, device=DEVICE, dtype=DTYPE)
    theta_back = eta_to_theta(eta, device=DEVICE, dtype=DTYPE)

    expected_eta = torch.tensor([math.pi, math.pi / 2.0, 0.0], device=DEVICE, dtype=DTYPE)
    torch.testing.assert_close(eta, expected_eta, rtol=1.0e-14, atol=1.0e-14)
    torch.testing.assert_close(theta_back, theta, rtol=1.0e-14, atol=1.0e-12)


def test_surface_detector_angle_roundtrip_for_physical_downward_branch():
    depth = torch.tensor(2.0, device=DEVICE, dtype=DTYPE)
    alpha = torch.linspace(0.0, 70.0, 21, device=DEVICE, dtype=DTYPE)

    theta = alpha_surface_to_theta_detector(alpha, depth, device=DEVICE, dtype=DTYPE)
    alpha_back = theta_detector_to_alpha_surface(theta, depth, device=DEVICE, dtype=DTYPE)

    torch.testing.assert_close(alpha_back, alpha, rtol=1.0e-12, atol=1.0e-10)


def test_alpha_max_matches_detector_radius_over_earth_radius():
    depth = torch.tensor([0.0, 2.0, 10.0], device=DEVICE, dtype=DTYPE)

    alpha_max = alpha_max_for_detector_depth(depth, device=DEVICE, dtype=DTYPE)
    expected = torch.rad2deg(torch.asin((R_E_KM - depth) / R_E_KM))

    torch.testing.assert_close(alpha_max, expected, rtol=1.0e-10, atol=1.0e-6)
    assert torch.all(alpha_max <= 90.0)


def test_underground_path_surface_detector_and_vertical_depth_limit():
    theta = torch.tensor([0.0, 45.0, 90.0], device=DEVICE, dtype=DTYPE)

    at_surface = underground_path_length(theta, 0.0, device=DEVICE, dtype=DTYPE)
    vertical_depth = underground_path_length(torch.tensor(0.0, device=DEVICE, dtype=DTYPE), 2.0, device=DEVICE, dtype=DTYPE)

    torch.testing.assert_close(at_surface, torch.zeros_like(theta), rtol=0.0, atol=1.0e-12)
    torch.testing.assert_close(vertical_depth, torch.tensor(2.0, device=DEVICE, dtype=DTYPE), rtol=1.0e-13, atol=1.0e-13)


def test_total_path_length_decomposes_into_underground_plus_atmosphere():
    h = torch.tensor([1.0, 10.0, 30.0], device=DEVICE, dtype=DTYPE)[:, None]
    theta = torch.tensor([0.0, 45.0, 85.0], device=DEVICE, dtype=DTYPE)[None, :]
    depth = torch.tensor(2.0, device=DEVICE, dtype=DTYPE)

    total = total_path_length(h, theta, depth, device=DEVICE, dtype=DTYPE)
    underground = underground_path_length(theta, depth, device=DEVICE, dtype=DTYPE)
    atmosphere = atmosphere_path_length(h, theta, depth, device=DEVICE, dtype=DTYPE)

    assert total.shape == (3, 3)
    torch.testing.assert_close(total, underground + atmosphere, rtol=1.0e-13, atol=1.0e-10)


def test_vertical_total_and_atmosphere_lengths_match_height_and_depth():
    h = torch.tensor([0.0, 10.0, 50.0], device=DEVICE, dtype=DTYPE)
    depth = torch.tensor(2.0, device=DEVICE, dtype=DTYPE)

    total = total_path_length(h, 0.0, depth, device=DEVICE, dtype=DTYPE)
    atmosphere = atmosphere_path_length(h, 0.0, depth, device=DEVICE, dtype=DTYPE)

    torch.testing.assert_close(total, h + depth, rtol=1.0e-13, atol=1.0e-11)
    torch.testing.assert_close(atmosphere, h, rtol=1.0e-13, atol=1.0e-11)


def test_altitude_along_path_has_detector_surface_and_production_limits():
    h = torch.tensor(20.0, device=DEVICE, dtype=DTYPE)
    theta = torch.tensor(35.0, device=DEVICE, dtype=DTYPE)
    depth = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)
    s_surface = underground_path_length(theta, depth, device=DEVICE, dtype=DTYPE)
    s_total = total_path_length(h, theta, depth, device=DEVICE, dtype=DTYPE)
    s = torch.stack([torch.zeros_like(s_surface), s_surface, s_total])

    altitude = altitude_along_detector_path(s, theta, depth, device=DEVICE, dtype=DTYPE)

    torch.testing.assert_close(altitude[0], -depth, rtol=1.0e-13, atol=1.0e-10)
    torch.testing.assert_close(altitude[1], torch.tensor(0.0, device=DEVICE, dtype=DTYPE), rtol=0.0, atol=1.0e-10)
    torch.testing.assert_close(altitude[2], h, rtol=1.0e-13, atol=1.0e-10)


def test_atmosphere_path_grid_endpoints_and_monotonic_altitude():
    h = torch.tensor([10.0, 25.0], device=DEVICE, dtype=DTYPE)
    theta = torch.tensor([0.0, 60.0], device=DEVICE, dtype=DTYPE)
    depth = torch.tensor(2.0, device=DEVICE, dtype=DTYPE)

    s_grid, h_grid = atmosphere_path_grid(h, theta, depth, n_steps=25, device=DEVICE, dtype=DTYPE)
    L_atm = atmosphere_path_length(h, theta, depth, device=DEVICE, dtype=DTYPE)

    assert s_grid.shape == (2, 25)
    assert h_grid.shape == (2, 25)
    torch.testing.assert_close(s_grid[:, 0], torch.zeros(2, device=DEVICE, dtype=DTYPE), rtol=0.0, atol=0.0)
    torch.testing.assert_close(s_grid[:, -1], L_atm, rtol=1.0e-13, atol=1.0e-10)
    torch.testing.assert_close(h_grid[:, 0], torch.zeros(2, device=DEVICE, dtype=DTYPE), rtol=0.0, atol=1.0e-10)
    torch.testing.assert_close(h_grid[:, -1], h, rtol=1.0e-13, atol=1.0e-10)
    assert torch.all(torch.diff(h_grid, dim=-1) >= -1.0e-10)


def test_invalid_detector_radius_raises():
    with pytest.raises(ValueError):
        underground_path_length(0.0, R_E_KM, device=DEVICE, dtype=DTYPE)

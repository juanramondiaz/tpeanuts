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

from tpeanuts.core.common.flux import flux_from_probability
from tpeanuts.util.test_utils import assert_close


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_flux_from_probability_scalar_flux_scales_all_flavours():
    probability = torch.tensor([0.2, 0.3, 0.5], device=DEVICE, dtype=DTYPE)

    flux = flux_from_probability(probability, 10.0)

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

    out = flux_from_probability(probability, flux, spectrum)
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

    out = flux_from_probability(probability, flux)

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

    out = flux_from_probability(probability, flux, spectrum)
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

    out = flux_from_probability(probability, flux)
    expected = probability * flux[..., None]

    assert_close(out, expected, name="full leading-grid flux")


def test_flux_from_probability_rejects_probability_without_three_flavours():
    with pytest.raises(ValueError, match="final flavour dimension 3"):
        flux_from_probability(torch.ones((2, 2), device=DEVICE, dtype=DTYPE), 1.0)


def test_flux_from_probability_rejects_too_many_flux_dimensions():
    probability = torch.ones((2, 3), device=DEVICE, dtype=DTYPE) / 3.0
    flux = torch.ones((2, 3, 1), device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="more dimensions than probability"):
        flux_from_probability(probability, flux)


def test_flux_from_probability_rejects_too_many_spectrum_dimensions():
    probability = torch.ones((2, 3), device=DEVICE, dtype=DTYPE) / 3.0
    spectrum = torch.ones((2, 3, 1), device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="more dimensions than probability"):
        flux_from_probability(probability, 1.0, spectrum)

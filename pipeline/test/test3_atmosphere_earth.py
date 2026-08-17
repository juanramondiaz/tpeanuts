#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pytest-compatible tests for tpeanuts.pipeline.atmosphere_earth detector-flux helpers."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.pipeline.atmosphere_earth import (
    detector_flux_from_production,
    integrate_detector_flux_over_height,
    propagate_atmosphere_grid_to_detector,
    sum_detected_flavours,
)
from tpeanuts.util.context import RuntimeContext


DTYPE = torch.float64


def _config() -> PropagationConfig:
    context = RuntimeContext.resolve("cpu", DTYPE)
    oscillation = PropagationConfig.oscillation_parameters_from_preset(
        "_SM_NUFIT52_NO", context=context
    )
    return PropagationConfig(runtime=context, oscillation=oscillation)


def _production(flavour: str, theta_deg: float) -> dict[str, object]:
    return {
        "particle": flavour,
        "E_grid_GeV": torch.tensor([1.0, 2.0], dtype=DTYPE),
        "h_grid_km": torch.tensor([10.0, 20.0], dtype=DTYPE),
        "theta_deg": theta_deg,
        "phi_Eh": torch.ones((2, 2), dtype=DTYPE),
    }


def test_detector_flux_helpers_weight_integrate_and_sum_sources():
    h = torch.tensor([0.0, 1.0, 2.0], dtype=DTYPE)
    production = torch.tensor([[2.0, 2.0, 2.0]], dtype=DTYPE)
    probabilities = torch.tensor(
        [[[[0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.5, 0.3, 0.2]]]],
        dtype=DTYPE,
    ).reshape(1, 3, 3)

    detector = detector_flux_from_production(production, probabilities)
    integrated = integrate_detector_flux_over_height(h, detector)
    total = sum_detected_flavours({"nue": integrated, "numu": integrated})

    torch.testing.assert_close(integrated, torch.tensor([[2.0, 1.2, 0.8]], dtype=DTYPE))
    torch.testing.assert_close(total, 2.0 * integrated)


def test_integrate_detector_flux_over_height_rejects_mismatched_grid():
    h = torch.tensor([0.0, 1.0], dtype=DTYPE)
    detector_flux = torch.zeros((1, 3, 3), dtype=DTYPE)

    with pytest.raises(ValueError, match="h_grid_km"):
        integrate_detector_flux_over_height(h, detector_flux)


def test_sum_detected_flavours_requires_at_least_one_entry():
    with pytest.raises(ValueError, match="At least one"):
        sum_detected_flavours({})


def test_sum_detected_flavours_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="must share a shape"):
        sum_detected_flavours({
            "nue": torch.zeros(3, dtype=DTYPE),
            "numu": torch.zeros(2, dtype=DTYPE),
        })


def test_atmosphere_grid_pipeline_returns_full_transition_and_integrated_fluxes():
    productions = {
        flavour: [_production(flavour, 120.0), _production(flavour, 140.0)]
        for flavour in ("nue", "numu", "nutau")
    }
    result = propagate_atmosphere_grid_to_detector(
        productions,
        _config(),
        integrate_angular=True,
        integrate_energy=True,
    )
    assert result.transition_probability_theta_Eh_beta_alpha.shape == (2, 2, 2, 3, 3)
    assert result.probability_height_theta_E_beta_alpha.shape == (2, 2, 3, 3)
    assert result.detector_flux_theta_Eh_beta.shape == (2, 2, 2, 3)
    assert result.detector_flux_height_theta_E_beta.shape == (2, 2, 3)
    assert result.detector_flux_angular_E_beta.shape == (2, 3)
    assert result.detector_rate.shape == (3,)


def test_atmosphere_earth_pipeline_returns_earth_perturbative_diagnostics():
    productions = {
        flavour: [_production(flavour, 140.0)]
        for flavour in ("nue", "numu", "nutau")
    }
    result = propagate_atmosphere_grid_to_detector(
        productions,
        _config(),
        return_diagnostics=True,
    )

    diagnostics = result.detector_results[0].perturbative_diagnostics
    assert diagnostics is not None
    assert diagnostics.max_first_order_norm.shape == torch.Size([2])
    assert diagnostics.validity_code.dtype == torch.int8


def test_atmosphere_grid_pipeline_propagates_analytic_eigenvalues_and_reunitarize_atmosphere():
    # config.analytic_eigenvalues flows into the Earth leg (pipeline/atmosphere_earth.py's
    # earth_evolutor_from_zenith call, always the perturbative/analytical path)
    # and config.reunitarize_atmosphere flows into the atmosphere leg
    # (pipeline/atmosphere.py's direct atmosphere_evolutor call); neither
    # should change the pipeline's shapes or blow up finiteness.
    productions = {
        flavour: [_production(flavour, 120.0), _production(flavour, 140.0)]
        for flavour in ("nue", "numu", "nutau")
    }
    default_config = _config()
    flagged_config = dataclasses.replace(
        default_config,
        analytic_eigenvalues=True,
        reunitarize_atmosphere=True,
        reunitarize_earth=True,
    )

    default_result = propagate_atmosphere_grid_to_detector(
        productions, default_config, integrate_angular=True, integrate_energy=True,
    )
    flagged_result = propagate_atmosphere_grid_to_detector(
        productions, flagged_config, integrate_angular=True, integrate_energy=True,
    )

    assert flagged_result.detector_rate.shape == default_result.detector_rate.shape
    assert torch.isfinite(flagged_result.detector_rate).all()
    # A loose smoke-test tolerance, not a precision check: this toy production
    # grid uses very low (1-2 GeV) energies and a coarse (2x2) grid where
    # reunitarize's SVD projection can shift the result by more than
    # floating-point noise -- the point here is that the flags propagate and
    # stay in the right physical ballpark, not bit-for-bit agreement.
    torch.testing.assert_close(
        flagged_result.detector_rate, default_result.detector_rate, atol=1.0e-3, rtol=1.0e-2,
    )

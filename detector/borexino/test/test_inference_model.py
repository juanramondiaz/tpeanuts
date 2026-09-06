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

"""Pytest-compatible tests for BorexinoEventRateModel (continuous and line sources)."""

from __future__ import annotations

import torch

from tpeanuts.detector.borexino.event_rate import line_event_rate
from tpeanuts.detector.borexino.inference_model import BorexinoEventRateModel
from tpeanuts.inference.model_solar import SolarSMOscillationModel
from tpeanuts.medium.solar.probability import solar_probability_state
from tpeanuts.medium.solar.profile import build_solar_medium
from tpeanuts.source.solar import build_solar_source
from tpeanuts.util.context import RuntimeContext


DTYPE = torch.float64
DEVICE = torch.device("cpu")


def _model_for(source_name: str) -> BorexinoEventRateModel:
    context = RuntimeContext.resolve(DEVICE, DTYPE)
    oscillation_model, _ = SolarSMOscillationModel.from_preset(
        "_SM_NUFIT52_NO", free=("theta12",), context=context,
    )
    medium = build_solar_medium(None, context=context)
    source = build_solar_source(None, context=context)
    bin_edges = torch.linspace(0.2, 1.2, 5, device=DEVICE, dtype=DTYPE)
    return BorexinoEventRateModel(
        oscillation_model=oscillation_model,
        medium=medium,
        source=source,
        source_names=(source_name,),
        bin_edges_MeV=bin_edges,
        exposure_days=1000.0,
    )


def test_predict_continuous_source_returns_finite_nonnegative_counts():
    model = _model_for("8B")
    theta = torch.tensor([0.59], dtype=DTYPE)  # theta12, the model's default free parameter

    counts = model.predict(theta)

    assert counts.shape == (4,)
    assert torch.isfinite(counts).all()
    assert torch.all(counts >= 0.0)


def test_predict_line_source_matches_direct_line_event_rate_computation():
    # End-to-end check for the SolarLineSpectrum branch of predict(): not
    # just "does it run", but that it reproduces line_event_rate called
    # directly with the same physical inputs.
    model = _model_for("7Be")
    theta = torch.tensor([0.59], dtype=DTYPE)

    counts = model.predict(theta)

    oscillation = model.oscillation_model.oscillation(theta)
    spectrum = model.source.spectrum_table("7Be")
    probabilities = solar_probability_state(
        oscillation, spectrum.energy_MeV, model.medium, model.source, "7Be",
    )
    expected = line_event_rate(
        probabilities, model.source.total_flux("7Be"), spectrum.weights, spectrum.energy_MeV,
        model.bin_edges_MeV, exposure_days=model.exposure_days,
    )

    assert counts.shape == (4,)
    assert torch.isfinite(counts).all()
    assert torch.all(counts >= 0.0)
    torch.testing.assert_close(counts, expected)


def test_predict_line_and_continuous_sources_are_both_differentiable():
    for source_name in ("8B", "7Be"):
        model = _model_for(source_name)
        theta = torch.tensor([0.59], dtype=DTYPE, requires_grad=True)

        counts = model.predict(theta)
        counts.sum().backward()

        assert theta.grad is not None
        assert torch.isfinite(theta.grad).all()

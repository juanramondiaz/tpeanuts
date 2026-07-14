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

"""Pytest-compatible checks for perturbative density-profile models."""

from __future__ import annotations

import pytest
import torch

from tpeanuts.core.perturbative.models.even_power.profile_layered import (
    EvenPowerProfileLayered,
)
from tpeanuts.core.perturbative.models.even_power.profile_segment import (
    EvenPowerProfileSegment,
)
from tpeanuts.core.perturbative.models.interface import (
    PerturbativeOuterSegment,
    PerturbativeSegmentBatch,
)
from tpeanuts.core.perturbative.models.model_selection import (
    perturbative_profile_selection,
)
from tpeanuts.core.perturbative.models.prem.profile_layered import (
    PremTabulatedProfile,
)
from tpeanuts.core.perturbative.models.prem.profile_segment import PremProfileSegment
from tpeanuts.util.test_utils import assert_close


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CDTYPE = torch.complex128


def tensor(value, *, dtype=DTYPE):
    return torch.as_tensor(value, device=DEVICE, dtype=dtype)


def test_perturbative_interface_dataclasses_preserve_payloads():
    x1 = tensor([0.0, 0.2])
    x2 = tensor([0.2, 0.5])
    crossed = torch.tensor([True, False], device=DEVICE)
    payload = {"coefficients": tensor([[1.0, 0.0, 0.0], [2.0, 0.1, 0.0]])}

    segments = PerturbativeSegmentBatch(x1=x1, x2=x2, crossed=crossed, model_data=payload)
    outer = PerturbativeOuterSegment(
        x_start=x1,
        model_data=payload,
        has_any=crossed,
        has_two=torch.tensor([False, True], device=DEVICE),
    )

    assert segments.model_data is payload
    assert outer.model_data is payload
    assert_close(segments.x2 - segments.x1, tensor([0.2, 0.3]), name="segment lengths")
    assert outer.has_two.shape == crossed.shape


def test_perturbative_profile_selection_accepts_supported_aliases():
    even_coefficients = tensor(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.1, 0.0],
        ]
    )
    prem_coefficients = tensor(
        [
            [1.0, 0.0],
            [2.0, 0.1],
        ]
    )

    even = perturbative_profile_selection(
        "even-power",
        {"coefficients": even_coefficients, "device": DEVICE, "dtype": DTYPE},
    )
    prem = perturbative_profile_selection(
        "prem500",
        {"coefficients": prem_coefficients, "rj": tensor([0.5, 1.0]), "device": DEVICE, "dtype": DTYPE},
    )

    assert isinstance(even, EvenPowerProfileLayered)
    assert isinstance(prem, PremTabulatedProfile)


def test_perturbative_profile_selection_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown perturbative profile model"):
        perturbative_profile_selection("not_a_model", {})


def test_even_power_constant_segment_has_no_perturbation_and_zero_residual():
    segment = EvenPowerProfileSegment.constant(
        x1=tensor(0.1),
        x2=tensor(0.6),
        density=tensor(1.7),
        device=DEVICE,
        dtype=DTYPE,
    )
    la = tensor([0.2, 0.5, 0.9], dtype=CDTYPE)[:, None]
    lb = tensor([0.2, 0.5, 0.9], dtype=CDTYPE)[None, :]

    residual = segment.residual_integral(la, lb)

    assert_close(segment.length, tensor(0.5), name="constant even-power length")
    assert_close(segment.average, tensor(1.7), name="constant even-power average")
    assert not bool(segment.zero_mask)
    assert not bool(segment.has_perturbation())
    assert_close(residual, torch.zeros((3, 3), device=DEVICE, dtype=CDTYPE), name="constant residual")


def test_even_power_segment_average_matches_polynomial_integral():
    x1 = tensor(0.0)
    x2 = tensor(2.0)
    a = tensor(1.0)
    b = tensor(0.3)
    c = tensor(0.05)
    segment = EvenPowerProfileSegment(x1=x1, x2=x2, a=a, b=b, c=c, device=DEVICE, dtype=DTYPE)

    expected_average = a + b * (x2**3 - x1**3) / (3.0 * (x2 - x1)) + c * (x2**5 - x1**5) / (5.0 * (x2 - x1))

    assert_close(segment.average, expected_average, name="even-power polynomial average")
    assert bool(segment.has_perturbation())
    assert segment.coefficients.shape == (3,)


def test_even_power_residual_integral_is_finite_and_diagonal_zero():
    segment = EvenPowerProfileSegment(
        x1=tensor(0.0),
        x2=tensor(0.7),
        a=tensor(1.0),
        b=tensor(0.2),
        c=tensor(0.05),
        device=DEVICE,
        dtype=DTYPE,
    )
    eigenvalues = tensor([0.2, 0.5, 0.9], dtype=CDTYPE)
    residual = segment.residual_integral(eigenvalues[:, None], eigenvalues[None, :])

    assert residual.shape == (3, 3)
    assert torch.isfinite(residual.real).all()
    assert torch.isfinite(residual.imag).all()
    assert_close(torch.diagonal(residual), torch.zeros(3, device=DEVICE, dtype=CDTYPE), name="even residual diagonal")


def test_even_power_layered_evaluate_shift_gather_and_segments():
    coefficients = tensor(
        [
            [1.0, 0.1, 0.01],
            [2.0, 0.2, 0.02],
            [3.0, 0.3, 0.03],
        ]
    )
    profile = EvenPowerProfileLayered(coefficients=coefficients.unsqueeze(0), device=DEVICE, dtype=DTYPE)
    x = tensor(2.0)
    layer_index = torch.tensor([1], device=DEVICE)

    value = profile.evaluate(x, layer_index=layer_index)
    shifted = profile.shifted(tensor(0.5))
    gathered = profile.gather_layers(torch.tensor([99], device=DEVICE))
    xj_all = tensor([[0.2, 0.5, 0.9]])
    crossed = torch.tensor([[True, False, True]], device=DEVICE)
    segments = profile.ordered_segments(xj_all, crossed)
    outer = profile.outermost_segment(xj_all, crossed)

    expected_value = coefficients[1, 0] + coefficients[1, 1] * x**2 + coefficients[1, 2] * x**4
    expected_shifted_a = coefficients[:, 0] + coefficients[:, 1] * 0.5 + coefficients[:, 2] * 0.5**2

    assert_close(value, expected_value.unsqueeze(0), name="even layered evaluate")
    assert_close(shifted.coefficients[..., 0], expected_shifted_a.unsqueeze(0), name="even shifted constant term")
    assert_close(gathered, coefficients[-1].unsqueeze(0), name="even gather clamps layer index")
    assert_close(segments.x1, tensor([[0.5, 0.2, 0.0]]), name="even ordered x1")
    assert_close(segments.x2, tensor([[0.9, 0.5, 0.2]]), name="even ordered x2")
    assert torch.equal(segments.crossed, torch.tensor([[True, False, True]], device=DEVICE))
    assert_close(outer.x_start, tensor([0.2]), name="even outer start")
    assert bool(outer.has_any[0])
    assert bool(outer.has_two[0])


def test_even_power_layered_segment_model_builds_segment_batch():
    coefficients = tensor(
        [
            [1.0, 0.1, 0.01],
            [2.0, 0.2, 0.02],
        ]
    )
    profile = EvenPowerProfileLayered(coefficients=coefficients, device=DEVICE, dtype=DTYPE)
    segments = PerturbativeSegmentBatch(
        x1=tensor([0.0, 0.5]),
        x2=tensor([0.5, 1.0]),
        crossed=torch.tensor([True, True], device=DEVICE),
        model_data=coefficients,
    )

    segment_model = profile.segment_model(segments, device=DEVICE, dtype=DTYPE)
    constant_model = profile.constant_segment_model(x1=tensor(0.0), x2=tensor(1.0), density=tensor(1.2), device=DEVICE, dtype=DTYPE)

    assert segment_model.average.shape == (2,)
    assert segment_model.has_perturbation().shape == (2,)
    assert_close(constant_model.average, tensor(1.2), name="even constant segment model average")
    assert not bool(constant_model.has_perturbation())


def test_prem_segment_average_and_constant_residual():
    segment = PremProfileSegment(
        x1=tensor(0.0),
        x2=tensor(2.0),
        a=tensor(1.0),
        b=tensor(0.3),
        device=DEVICE,
        dtype=DTYPE,
    )
    constant = PremProfileSegment(
        x1=tensor(0.2),
        x2=tensor(0.8),
        a=tensor(1.7),
        b=tensor(0.0),
        device=DEVICE,
        dtype=DTYPE,
    )
    la = tensor([0.2, 0.5, 0.9], dtype=CDTYPE)[:, None]
    lb = tensor([0.2, 0.5, 0.9], dtype=CDTYPE)[None, :]

    expected_average = tensor(1.0) + tensor(0.3) * (segment.x1**2 + segment.x1 * segment.x2 + segment.x2**2) / 3.0
    residual = constant.residual_integral(la, lb)

    assert_close(segment.average, expected_average, name="prem segment average")
    assert bool(segment.has_perturbation())
    assert not bool(constant.has_perturbation())
    assert_close(residual, torch.zeros((3, 3), device=DEVICE, dtype=CDTYPE), name="prem constant residual")


def test_prem_residual_integral_is_finite_and_diagonal_zero():
    segment = PremProfileSegment(
        x1=tensor(0.0),
        x2=tensor(0.7),
        a=tensor(1.0),
        b=tensor(0.2),
        device=DEVICE,
        dtype=DTYPE,
    )
    eigenvalues = tensor([0.2, 0.5, 0.9], dtype=CDTYPE)
    residual = segment.residual_integral(eigenvalues[:, None], eigenvalues[None, :])

    assert residual.shape == (3, 3)
    assert torch.isfinite(residual.real).all()
    assert torch.isfinite(residual.imag).all()
    assert_close(torch.diagonal(residual), torch.zeros(3, device=DEVICE, dtype=CDTYPE), name="prem residual diagonal")


def test_prem_tabulated_profile_shift_evaluate_segments_and_models():
    coefficients = tensor(
        [
            [1.0, 0.1],
            [2.0, 0.2],
            [3.0, 0.3],
        ]
    )
    profile = PremTabulatedProfile(
        rj=tensor([0.2, 0.5, 0.9]),
        coefficients=coefficients,
        device=DEVICE,
        dtype=DTYPE,
    )
    shifted = profile.shifted(tensor(0.5))
    x = tensor(2.0)
    layer_index = torch.tensor(1, device=DEVICE)
    xj_all = tensor([[0.2, 0.5, 0.9]])
    crossed = torch.tensor([[True, False, True]], device=DEVICE)
    segments = shifted.ordered_segments(xj_all, crossed)
    segment_model = shifted.segment_model(segments, device=DEVICE, dtype=DTYPE)
    constant_model = shifted.constant_segment_model(x1=tensor(0.0), x2=tensor(1.0), density=tensor(1.2), device=DEVICE, dtype=DTYPE)
    outer = shifted.outermost_segment(xj_all, crossed)

    expected_shifted_a = coefficients[:, 0] + coefficients[:, 1] * 0.5
    expected_value = shifted.coefficients[1, 0] + shifted.coefficients[1, 1] * x**2

    assert_close(shifted.coefficients[..., 0], expected_shifted_a, name="prem shifted constant term")
    assert_close(shifted.evaluate(x, layer_index=layer_index), expected_value, name="prem evaluate")
    assert_close(segments.x1, tensor([[0.5, 0.2, 0.0]]), name="prem ordered x1")
    assert segment_model.average.shape == (1, 3)
    assert_close(constant_model.average, tensor(1.2), name="prem constant segment model average")
    assert not bool(constant_model.has_perturbation())
    assert_close(outer.x_start, tensor([0.2]), name="prem outer start")
    assert bool(outer.has_any[0])
    assert bool(outer.has_two[0])

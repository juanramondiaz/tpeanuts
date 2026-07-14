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

"""Pytest-compatible checks for generic core evolutor utilities."""

from __future__ import annotations

import pytest
import torch

from tpeanuts.core.common.evolutor import (
    apply_evolutor_to_state,
    compose_segment_evolutors,
)
from tpeanuts.util.test_utils import assert_close


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CDTYPE = torch.complex128


def rotation(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta)
    s = torch.sin(theta)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)
    rows = [
        torch.stack([c, s, zeros], dim=-1),
        torch.stack([-s, c, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1),
    ]
    return torch.stack(rows, dim=-2).to(dtype=CDTYPE)


def identity3(batch_shape=()) -> torch.Tensor:
    return torch.eye(3, device=DEVICE, dtype=CDTYPE).expand(*batch_shape, 3, 3)


def test_apply_evolutor_to_state_identity_returns_state():
    state = torch.tensor([1.0 + 0.0j, 0.2 - 0.3j, -0.4 + 0.1j], device=DEVICE, dtype=CDTYPE)
    evolved = apply_evolutor_to_state(identity3(), state)

    assert evolved.shape == (3,)
    assert_close(evolved, state, name="identity evolutor leaves state unchanged")


def test_apply_evolutor_to_state_matches_matrix_vector_product():
    theta = torch.tensor(0.37, device=DEVICE, dtype=DTYPE)
    U = rotation(theta)
    state = torch.tensor([1.0 + 0.0j, 0.0 + 0.0j, 0.5j], device=DEVICE, dtype=CDTYPE)

    evolved = apply_evolutor_to_state(U, state)
    expected = U @ state

    assert_close(evolved, expected, name="evolutor @ state")


def test_apply_evolutor_to_state_broadcasts_state_over_batch():
    theta = torch.tensor([0.1, 0.2, 0.3], device=DEVICE, dtype=DTYPE)
    U = rotation(theta)
    state = torch.tensor([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], device=DEVICE, dtype=CDTYPE)

    evolved = apply_evolutor_to_state(U, state)
    expected = torch.matmul(U, state[..., None]).squeeze(-1)

    assert evolved.shape == (3, 3)
    assert_close(evolved, expected, name="batched evolutor broadcast state")


def test_apply_evolutor_to_state_batched_state_matches_manual_product():
    theta = torch.tensor([0.1, 0.2, 0.3], device=DEVICE, dtype=DTYPE)
    U = rotation(theta)
    state = torch.tensor(
        [
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
        ],
        device=DEVICE,
        dtype=CDTYPE,
    )

    evolved = apply_evolutor_to_state(U, state)
    expected = torch.matmul(U, state[..., None]).squeeze(-1)

    assert evolved.shape == (3, 3)
    assert_close(evolved, expected, name="batched state evolution")


def test_apply_evolutor_to_state_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="evolutor must have final dimensions"):
        apply_evolutor_to_state(torch.ones((3, 2), device=DEVICE, dtype=CDTYPE), torch.ones(3, device=DEVICE, dtype=CDTYPE))

    with pytest.raises(ValueError, match="state must have last dimension"):
        apply_evolutor_to_state(identity3(), torch.ones(2, device=DEVICE, dtype=CDTYPE))


def test_compose_segment_evolutors_left_matches_historical_reversed_product():
    U0 = rotation(torch.tensor(0.1, device=DEVICE, dtype=DTYPE))
    U1 = rotation(torch.tensor(0.2, device=DEVICE, dtype=DTYPE))
    U2 = rotation(torch.tensor(0.3, device=DEVICE, dtype=DTYPE))
    segments = torch.stack([U0, U1, U2], dim=0)

    composed = compose_segment_evolutors(segments, multiply="left")
    expected = U2 @ U1 @ U0

    assert_close(composed, expected, name="left composition reversed product")


def test_compose_segment_evolutors_right_matches_left_to_right_product():
    U0 = rotation(torch.tensor(0.1, device=DEVICE, dtype=DTYPE))
    U1 = rotation(torch.tensor(0.2, device=DEVICE, dtype=DTYPE))
    U2 = rotation(torch.tensor(0.3, device=DEVICE, dtype=DTYPE))
    segments = torch.stack([U0, U1, U2], dim=0)

    composed = compose_segment_evolutors(segments, multiply="right")
    expected = U0 @ U1 @ U2

    assert_close(composed, expected, name="right composition left-to-right product")


def test_compose_segment_evolutors_handles_batched_segments_and_custom_dim():
    theta = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        device=DEVICE,
        dtype=DTYPE,
    )
    segments = rotation(theta)
    segments_custom_dim = torch.movedim(segments, 1, 0)

    composed = compose_segment_evolutors(segments_custom_dim, segment_dim=0, multiply="right")
    expected = torch.stack(
        [
            segments[0, 0] @ segments[0, 1] @ segments[0, 2],
            segments[1, 0] @ segments[1, 1] @ segments[1, 2],
        ],
        dim=0,
    )

    assert composed.shape == (2, 3, 3)
    assert_close(composed, expected, name="custom segment_dim batched composition")


def test_compose_segment_evolutors_single_segment_returns_that_segment():
    segment = rotation(torch.tensor([0.25, 0.50], device=DEVICE, dtype=DTYPE)).unsqueeze(-3)

    composed = compose_segment_evolutors(segment)

    assert_close(composed, segment[..., 0, :, :], name="single segment composition")


def test_compose_segment_evolutors_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="final shape"):
        compose_segment_evolutors(torch.ones((2, 3, 2), device=DEVICE, dtype=CDTYPE))

    with pytest.raises(ValueError, match="multiply"):
        compose_segment_evolutors(identity3((2,)), multiply="middle")

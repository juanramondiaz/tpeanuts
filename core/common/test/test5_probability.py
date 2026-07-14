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

"""Pytest-compatible checks for core probability utilities."""

from __future__ import annotations

import pytest
import torch

from tpeanuts.core.common.neutrino import flavour_index
from tpeanuts.core.common.probability import (
    check_probability_conservation,
    check_probability_matrix,
    normalize_probability_columns,
    probability_coherent,
    probability_coherent_state,
    probability_from_evolutor,
    probability_incoherent,
    probability_transition,
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


def test_flavour_index_accepts_aliases_and_rejects_unknown_labels():
    cases = {
        "e": 0,
        "electron": 0,
        "nue": 0,
        "nu_e": 0,
        "mu": 1,
        "muon": 1,
        "numu": 1,
        "nu_mu": 1,
        "tau": 2,
        "nutau": 2,
        "nu_tau": 2,
        0: 0,
        1: 1,
        2: 2,
    }

    for label, expected in cases.items():
        assert flavour_index(label) == expected

    with pytest.raises(ValueError, match="Unknown flavour label"):
        flavour_index("sterile")

    with pytest.raises(ValueError, match="Flavour index"):
        flavour_index(3)


def test_probability_transition_identity_evolutor_is_identity_matrix():
    S = torch.eye(3, device=DEVICE, dtype=CDTYPE)

    P = probability_transition(S)

    assert P.shape == (3, 3)
    assert P.dtype == DTYPE
    assert_close(P, torch.eye(3, device=DEVICE, dtype=DTYPE), name="identity probability matrix")


def test_probability_transition_known_rotation_channels():
    theta = torch.tensor(0.3, device=DEVICE, dtype=DTYPE)
    S = rotation(theta)
    P = probability_transition(S)
    expected = torch.tensor(
        [
            [torch.cos(theta) ** 2, torch.sin(theta) ** 2, 0.0],
            [torch.sin(theta) ** 2, torch.cos(theta) ** 2, 0.0],
            [0.0, 0.0, 1.0],
        ],
        device=DEVICE,
        dtype=DTYPE,
    )

    assert_close(P, expected, name="known rotation probability matrix")
    assert_close(probability_transition(S, alpha="e", beta="mu"), expected[1, 0], name="P(e->mu)")
    assert_close(probability_transition(P, alpha="e", input_is_probability=True), expected[0, 0], name="P(e->e)")


def test_probability_transition_beta_without_alpha_raises():
    with pytest.raises(ValueError, match="beta cannot be provided without alpha"):
        probability_transition(torch.eye(3, device=DEVICE, dtype=CDTYPE), beta="mu")


def test_probability_coherent_state_returns_component_moduli():
    state = torch.tensor([1.0 + 0.0j, 0.0 + 2.0j, 3.0 - 4.0j], device=DEVICE, dtype=CDTYPE)

    P = probability_coherent_state(state)

    assert_close(P, torch.tensor([1.0, 4.0, 25.0], device=DEVICE, dtype=DTYPE), name="coherent state moduli")


def test_probability_coherent_applies_evolutor_before_projection():
    theta = torch.tensor(0.3, device=DEVICE, dtype=DTYPE)
    S = rotation(theta)
    state = torch.tensor([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], device=DEVICE, dtype=CDTYPE)

    P = probability_coherent(S, state)
    expected = probability_coherent_state(S @ state)

    assert_close(P, expected, name="coherent probability from evolved state")


def test_probability_incoherent_probability_matrix_matches_einsum():
    theta = torch.tensor(0.3, device=DEVICE, dtype=DTYPE)
    P = probability_transition(rotation(theta))
    weights = torch.tensor([10.0, 5.0, 1.0], device=DEVICE, dtype=DTYPE)

    out = probability_incoherent(P, weights)
    expected = torch.einsum("ba,a->b", P, weights)

    assert_close(out, expected, name="incoherent P @ weights")


def test_probability_incoherent_with_pmns_matrix_projects_mass_weights():
    S = torch.eye(3, device=DEVICE, dtype=CDTYPE)
    theta = torch.tensor(0.2, device=DEVICE, dtype=DTYPE)
    U = rotation(theta)
    weights = torch.tensor([0.7, 0.2, 0.1], device=DEVICE, dtype=DTYPE)

    out = probability_incoherent(S, weights, pmns=U)
    expected = torch.einsum("ai,i->a", torch.abs(U) ** 2, weights)

    assert_close(out, expected.to(dtype=DTYPE), name="mass weights through PMNS matrix")


def test_probability_from_evolutor_dispatches_coherent_and_massbasis_paths():
    S = rotation(torch.tensor(0.2, device=DEVICE, dtype=DTYPE))
    coherent_state = torch.tensor([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], device=DEVICE, dtype=CDTYPE)
    mass_weights = torch.tensor([0.7, 0.2, 0.1], device=DEVICE, dtype=DTYPE)
    U = torch.eye(3, device=DEVICE, dtype=CDTYPE)

    coherent = probability_from_evolutor(S, coherent_state)
    incoherent = probability_from_evolutor(S, mass_weights, massbasis=True, pmns=U)

    assert_close(coherent, probability_coherent(S, coherent_state), name="coherent dispatch")
    assert_close(incoherent, probability_incoherent(S, mass_weights, pmns=U), name="massbasis dispatch")


def test_probability_from_evolutor_massbasis_requires_pmns():
    with pytest.raises(ValueError, match="pmns is required"):
        probability_from_evolutor(
            torch.eye(3, device=DEVICE, dtype=CDTYPE),
            torch.ones(3, device=DEVICE, dtype=DTYPE),
            massbasis=True,
        )


def test_normalize_probability_columns_makes_columns_sum_to_one():
    P = torch.tensor(
        [
            [0.6, 0.2, 0.3],
            [0.3, 0.5, 0.4],
            [0.2, 0.1, 0.5],
        ],
        device=DEVICE,
        dtype=DTYPE,
    )

    P_norm = normalize_probability_columns(P)

    assert_close(P_norm.sum(dim=-2), torch.ones(3, device=DEVICE, dtype=DTYPE), name="normalized column sums")


def test_check_probability_conservation_and_matrix_accept_valid_unitary_matrix():
    P = probability_transition(rotation(torch.tensor(0.3, device=DEVICE, dtype=DTYPE)))

    assert check_probability_conservation(P)
    assert check_probability_matrix(P)


def test_check_probability_conservation_and_matrix_reject_invalid_matrices():
    P = probability_transition(rotation(torch.tensor(0.3, device=DEVICE, dtype=DTYPE)))
    P_bad_norm = P.clone()
    P_bad_norm[0, 0] += 0.1
    P_bad_negative = P.clone()
    P_bad_negative[0, 0] = -0.1
    P_bad_nan = P.clone()
    P_bad_nan[0, 0] = float("nan")

    assert not check_probability_conservation(P_bad_norm)
    assert not check_probability_matrix(P_bad_norm)
    assert not check_probability_matrix(P_bad_negative)
    assert not check_probability_matrix(P_bad_nan)

    with pytest.raises(ValueError, match="Probability conservation failed"):
        check_probability_conservation(P_bad_norm, raise_error=True)

    with pytest.raises(ValueError, match="negative values"):
        check_probability_matrix(P_bad_negative, raise_error=True)

    with pytest.raises(ValueError, match="NaN or Inf"):
        check_probability_matrix(P_bad_nan, raise_error=True)


def test_batched_probability_matrix_and_incoherent_weights():
    theta = torch.tensor([0.1, 0.2, 0.3], device=DEVICE, dtype=DTYPE)
    P = probability_transition(rotation(theta))
    weights = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ],
        device=DEVICE,
        dtype=DTYPE,
    )

    out = probability_incoherent(P, weights)
    expected = torch.einsum("...ba,...a->...b", P, weights)

    assert P.shape == (3, 3, 3)
    assert out.shape == (3, 3)
    assert_close(P.sum(dim=-2), torch.ones((3, 3), device=DEVICE, dtype=DTYPE), name="batched conservation")
    assert_close(out, expected, name="batched incoherent probabilities")

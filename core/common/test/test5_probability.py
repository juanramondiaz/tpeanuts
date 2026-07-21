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

from tpeanuts.core.common.neutrino import flavour_index, flavour_state
from tpeanuts.core.common.probability import (
    check_probability_conservation,
    check_probability_matrix,
    normalize_probability_columns,
    probability_coherent,
    probability_coherent_state,
    probability_integrated,
    probability_integrated_angular,
    probability_state,
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
        "s": 3,
        "sterile": 3,
        "nus": 3,
        "nu_s": 3,
        0: 0,
        1: 1,
        2: 2,
        3: 3,
    }

    for label, expected in cases.items():
        assert flavour_index(label) == expected

    with pytest.raises(ValueError, match="Unknown flavour label"):
        flavour_index("not_a_flavour")

    with pytest.raises(ValueError, match="Flavour index"):
        flavour_index(4)


def test_flavour_state_builds_three_flavour_state_by_default():
    state = flavour_state("mu", device=DEVICE, dtype=DTYPE)

    assert state.shape == (3,)
    assert state.dtype == CDTYPE
    assert_close(state, torch.tensor([0.0, 1.0, 0.0], device=DEVICE, dtype=CDTYPE), name="default 3-flavour state")


def test_flavour_state_builds_four_flavour_sterile_state():
    state = flavour_state("sterile", device=DEVICE, dtype=DTYPE, n_flavours=4)

    assert state.shape == (4,)
    assert_close(
        state,
        torch.tensor([0.0, 0.0, 0.0, 1.0], device=DEVICE, dtype=CDTYPE),
        name="four-flavour sterile state",
    )


def test_flavour_state_rejects_sterile_index_without_four_flavours():
    with pytest.raises(ValueError, match="out of range"):
        flavour_state("sterile", device=DEVICE, dtype=DTYPE)


def test_flavour_state_rejects_invalid_n_flavours():
    with pytest.raises(ValueError, match="n_flavours must be 3 or 4"):
        flavour_state("e", device=DEVICE, dtype=DTYPE, n_flavours=5)


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


# Fase 6: probability.py accepts N=4 (3+1 sterile) alongside the N=3 Standard Model case.


def test_probability_coherent_state_accepts_four_flavours():
    state = torch.tensor(
        [1.0 + 0.0j, 0.0 + 2.0j, 3.0 - 4.0j, 1.0 + 1.0j], device=DEVICE, dtype=CDTYPE
    )

    P = probability_coherent_state(state)

    assert P.shape == (4,)
    assert_close(
        P,
        torch.tensor([1.0, 4.0, 25.0, 2.0], device=DEVICE, dtype=DTYPE),
        name="four-flavour coherent state moduli",
    )


def test_probability_coherent_state_rejects_invalid_last_dimension():
    state = torch.ones(5, device=DEVICE, dtype=CDTYPE)

    with pytest.raises(ValueError, match="3 or 4"):
        probability_coherent_state(state)


def test_probability_coherent_applies_evolutor_before_projection():
    theta = torch.tensor(0.3, device=DEVICE, dtype=DTYPE)
    S = rotation(theta)
    state = torch.tensor([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], device=DEVICE, dtype=CDTYPE)

    P = probability_coherent(S, state)
    expected = probability_coherent_state(S @ state)

    assert_close(P, expected, name="coherent probability from evolved state")


def test_probability_incoherent_accepts_four_flavour_weights():
    P = probability_transition(torch.eye(4, device=DEVICE, dtype=CDTYPE))
    weights = torch.tensor([10.0, 5.0, 1.0, 2.0], device=DEVICE, dtype=DTYPE)

    out = probability_incoherent(P, weights)

    assert out.shape == (4,)
    assert_close(out, weights, name="four-flavour incoherent identity pass-through")


def test_probability_incoherent_rejects_invalid_weights_last_dimension():
    P = probability_transition(torch.eye(3, device=DEVICE, dtype=CDTYPE))
    weights = torch.ones(5, device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="3 or 4"):
        probability_incoherent(P, weights)


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

    coherent = probability_state(S, coherent_state)
    incoherent = probability_state(S, mass_weights, massbasis=True, pmns=U)

    assert_close(coherent, probability_coherent(S, coherent_state), name="coherent dispatch")
    assert_close(incoherent, probability_incoherent(S, mass_weights, pmns=U), name="massbasis dispatch")


def test_probability_from_evolutor_massbasis_requires_pmns():
    with pytest.raises(ValueError, match="pmns is required"):
        probability_state(
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


# ---------------------------------------------------------------------------
# probability_transition / probability_state -- explicit N=4 (3+1 sterile) checks
# ---------------------------------------------------------------------------


def test_probability_transition_four_flavour_identity_evolutor_is_identity_matrix():
    S = torch.eye(4, device=DEVICE, dtype=CDTYPE)

    P = probability_transition(S)

    assert P.shape == (4, 4)
    assert_close(P, torch.eye(4, device=DEVICE, dtype=DTYPE), name="four-flavour identity probability matrix")


def test_probability_transition_four_flavour_selects_sterile_channel_by_alias():
    S = torch.eye(4, device=DEVICE, dtype=CDTYPE)
    S[3, 0] = 1.0
    S[0, 0] = 0.0

    P_es = probability_transition(S, alpha="e", beta="sterile")

    assert_close(P_es, torch.tensor(1.0, device=DEVICE, dtype=DTYPE), name="P(e->sterile) via alias")


def test_probability_state_four_flavour_coherent_and_massbasis_dispatch():
    S = torch.eye(4, device=DEVICE, dtype=CDTYPE)
    coherent_state = torch.tensor([0.0, 1.0, 0.0, 0.0], device=DEVICE, dtype=CDTYPE)
    mass_weights = torch.tensor([0.4, 0.3, 0.2, 0.1], device=DEVICE, dtype=DTYPE)
    U = torch.eye(4, device=DEVICE, dtype=CDTYPE)

    coherent = probability_state(S, coherent_state)
    incoherent = probability_state(S, mass_weights, massbasis=True, pmns=U)

    assert coherent.shape == (4,)
    assert incoherent.shape == (4,)
    assert_close(coherent, probability_coherent(S, coherent_state), name="four-flavour coherent dispatch")
    assert_close(incoherent, probability_incoherent(S, mass_weights, pmns=U), name="four-flavour massbasis dispatch")


# ---------------------------------------------------------------------------
# probability_integrated
# ---------------------------------------------------------------------------


def test_probability_integrated_matches_manual_weighted_average():
    E = torch.tensor([100.0, 200.0, 500.0, 1000.0], device=DEVICE, dtype=DTYPE)
    P = torch.tensor(
        [
            [0.9, 0.05, 0.05],
            [0.7, 0.2, 0.1],
            [0.5, 0.3, 0.2],
            [0.3, 0.4, 0.3],
        ],
        device=DEVICE,
        dtype=DTYPE,
    )
    spectrum = torch.tensor([1.0, 2.0, 1.0, 0.5], device=DEVICE, dtype=DTYPE)

    out = probability_integrated(P, E, spectrum)
    expected = torch.trapezoid(P * spectrum[:, None], x=E, dim=-2) / torch.trapezoid(spectrum, x=E)

    assert out.shape == (3,)
    assert_close(out, expected, name="probability_integrated matches manual weighted average")


def test_probability_integrated_preserves_angle_axis_with_explicit_energy_dim():
    E = torch.tensor([100.0, 500.0, 1000.0], device=DEVICE, dtype=DTYPE)
    P = torch.rand((3, 4, 3), device=DEVICE, dtype=DTYPE)
    P = P / P.sum(dim=-1, keepdim=True)
    spectrum = torch.tensor([1.0, 1.0, 1.0], device=DEVICE, dtype=DTYPE)

    out = probability_integrated(P, E, spectrum, energy_dim=-3)
    expected = torch.trapezoid(P * spectrum[:, None, None], x=E, dim=-3) / torch.trapezoid(spectrum, x=E)

    assert out.shape == (4, 3)
    assert_close(out, expected, name="probability_integrated preserves angle axis")


def test_probability_integrated_stays_in_unit_interval_for_normalized_probabilities():
    E = torch.linspace(100.0, 1000.0, 6, device=DEVICE, dtype=DTYPE)
    P = torch.rand((6, 3), device=DEVICE, dtype=DTYPE)
    P = P / P.sum(dim=-1, keepdim=True)
    spectrum = torch.rand(6, device=DEVICE, dtype=DTYPE) + 0.1

    out = probability_integrated(P, E, spectrum)

    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)
    assert_close(out.sum(), torch.tensor(1.0, device=DEVICE, dtype=DTYPE), atol=1.0e-12, rtol=1.0e-12, name="weighted-average probabilities still sum to one")


def test_probability_integrated_accepts_four_flavours():
    E = torch.tensor([500.0, 1000.0, 3000.0], device=DEVICE, dtype=DTYPE)
    P = torch.rand((3, 4), device=DEVICE, dtype=DTYPE)
    P = P / P.sum(dim=-1, keepdim=True)
    spectrum = torch.tensor([1.0, 2.0, 1.0], device=DEVICE, dtype=DTYPE)

    out = probability_integrated(P, E, spectrum)
    expected = torch.trapezoid(P * spectrum[:, None], x=E, dim=-2) / torch.trapezoid(spectrum, x=E)

    assert out.shape == (4,)
    assert_close(out, expected, name="four-flavour probability_integrated")


def test_probability_integrated_rejects_probability_without_three_or_four_flavours():
    E = torch.tensor([100.0, 200.0], device=DEVICE, dtype=DTYPE)
    P = torch.ones((2, 2), device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="3 or 4"):
        probability_integrated(P, E, torch.ones(2, device=DEVICE, dtype=DTYPE))


def test_probability_integrated_rejects_energy_dim_on_flavour_axis():
    E = torch.tensor([100.0, 200.0, 300.0], device=DEVICE, dtype=DTYPE)
    P = torch.ones((4, 3), device=DEVICE, dtype=DTYPE) / 3.0

    with pytest.raises(ValueError, match="must not select the flavour axis"):
        probability_integrated(P, E, torch.ones(3, device=DEVICE, dtype=DTYPE), energy_dim=-1)


def test_probability_integrated_rejects_mismatched_energy_grid():
    E = torch.tensor([100.0, 200.0], device=DEVICE, dtype=DTYPE)
    P = torch.ones((4, 3), device=DEVICE, dtype=DTYPE) / 3.0

    with pytest.raises(ValueError, match="must match E_grid_MeV"):
        probability_integrated(P, E, torch.ones(4, device=DEVICE, dtype=DTYPE))


# ---------------------------------------------------------------------------
# probability_integrated_angular
# ---------------------------------------------------------------------------


def test_probability_integrated_angular_matches_manual_weighted_average():
    theta = torch.tensor([0.0, 45.0, 90.0, 135.0, 180.0], device=DEVICE, dtype=DTYPE)
    P = torch.rand((5, 3), device=DEVICE, dtype=DTYPE)
    P = P / P.sum(dim=-1, keepdim=True)

    out = probability_integrated_angular(P, theta)

    theta_rad = torch.deg2rad(theta)
    sin_theta = torch.sin(theta_rad)
    expected = torch.trapezoid(P * sin_theta[:, None], x=theta_rad, dim=-2) / torch.trapezoid(sin_theta, x=theta_rad)

    assert out.shape == (3,)
    assert_close(out, expected, name="probability_integrated_angular matches manual weighted average")


def test_probability_integrated_angular_stays_in_unit_interval_and_conserves_sum():
    theta = torch.linspace(0.0, 180.0, 7, device=DEVICE, dtype=DTYPE)
    P = torch.rand((7, 3), device=DEVICE, dtype=DTYPE)
    P = P / P.sum(dim=-1, keepdim=True)

    out = probability_integrated_angular(P, theta)

    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)
    assert_close(out.sum(), torch.tensor(1.0, device=DEVICE, dtype=DTYPE), atol=1.0e-12, rtol=1.0e-12, name="angular-averaged probabilities still sum to one")


def test_probability_integrated_angular_accepts_four_flavours():
    theta = torch.tensor([10.0, 90.0, 170.0], device=DEVICE, dtype=DTYPE)
    P = torch.rand((3, 4), device=DEVICE, dtype=DTYPE)
    P = P / P.sum(dim=-1, keepdim=True)

    out = probability_integrated_angular(P, theta)

    theta_rad = torch.deg2rad(theta)
    sin_theta = torch.sin(theta_rad)
    expected = torch.trapezoid(P * sin_theta[:, None], x=theta_rad, dim=-2) / torch.trapezoid(sin_theta, x=theta_rad)

    assert out.shape == (4,)
    assert_close(out, expected, name="four-flavour probability_integrated_angular")


def test_probability_integrated_angular_rejects_probability_without_three_or_four_flavours():
    theta = torch.tensor([0.0, 90.0], device=DEVICE, dtype=DTYPE)
    P = torch.ones((2, 2), device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="3 or 4"):
        probability_integrated_angular(P, theta)


def test_probability_integrated_angular_rejects_angular_dim_on_flavour_axis():
    theta = torch.tensor([0.0, 90.0, 180.0], device=DEVICE, dtype=DTYPE)
    P = torch.ones((4, 3), device=DEVICE, dtype=DTYPE) / 3.0

    with pytest.raises(ValueError, match="must not select the flavour axis"):
        probability_integrated_angular(P, theta, angular_dim=-1)


def test_probability_integrated_angular_rejects_mismatched_theta_grid():
    theta = torch.tensor([0.0, 90.0], device=DEVICE, dtype=DTYPE)
    P = torch.ones((4, 3), device=DEVICE, dtype=DTYPE) / 3.0

    with pytest.raises(ValueError, match="must match theta_deg"):
        probability_integrated_angular(P, theta)


def test_probability_weighted_average_reduces_arbitrary_coordinate():
    from tpeanuts.core.common.probability import probability_weighted_average

    coordinate = torch.tensor([0.0, 1.0, 2.0], device=DEVICE, dtype=DTYPE)
    probability = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 1.0, 0.0]]],
        device=DEVICE,
        dtype=DTYPE,
    )
    weight = torch.ones((1, 3), device=DEVICE, dtype=DTYPE)
    result = probability_weighted_average(probability, coordinate, weight, dim=1)
    expected = torch.tensor([[0.5, 0.5, 0.0]], device=DEVICE, dtype=DTYPE)
    assert_close(result, expected, name="generic probability weighted average")

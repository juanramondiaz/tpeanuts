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

"""
Verbose tests for tpeanuts.core.probabilities.

Run with:

    pytest tests/core/test_probabilities.py -v -s

or directly:

    python tests/core/test_probabilities.py
"""



from __future__ import annotations

import torch

from tpeanuts.core.segment_evolution import perturbative_segment_evolutor
from tpeanuts.core.probabilities import (
    flavour_index,
    probability_matrix_from_evolutor,
    transition_probability,
    survival_probability,
    apply_probability_matrix_to_flux,
    probability_columns_sum,
    check_probability_conservation,
    check_probability_matrix,
    normalize_probability_columns,
)

from tpeanuts.util.test_utils import (
    ATOL, RTOL, printoptions,
    banner, section, print_ok, print_fail,
    max_abs_error, assert_close, assert_true,
    default_inputs, build_pmns, run_test_suite
    )
printoptions()


# ============================================================
# Fixtures
# ============================================================


def build_segment_evolutor():
    pmns = build_pmns()

    S = perturbative_segment_evolutor(
        DeltamSq21=torch.tensor(7.42e-5, dtype=torch.float64),
        DeltamSq3l=torch.tensor(2.517e-3, dtype=torch.float64),
        pmns=pmns,
        E_MeV=torch.tensor(1000.0, dtype=torch.float64),
        x1=torch.tensor(0.20, dtype=torch.float64),
        x2=torch.tensor(0.70, dtype=torch.float64),
        a=torch.tensor(1.10, dtype=torch.float64),
        b=torch.tensor(0.20, dtype=torch.float64),
        c=torch.tensor(0.05, dtype=torch.float64),
        antinu=False,
    )

    return S


def build_unitary_test_operator():
    theta = torch.tensor(0.3, dtype=torch.float64)
    c = torch.cos(theta)
    s = torch.sin(theta)

    S = torch.tensor(
        [
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.complex128,
    )

    return S


# ============================================================
# tests
# ============================================================

def test_flavour_index():

    section("TEST: flavour_index")

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
        idx = flavour_index(label)
        print(f"{label!r} -> {idx}")
        assert_true(idx == expected, f"flavour_index({label!r}) = {expected}")

    try:
        flavour_index("invalid")
    except ValueError:
        print_ok("Invalid flavour label raises ValueError")
    else:
        print_fail("Invalid flavour label did not raise ValueError")
        raise AssertionError("Invalid flavour label did not raise ValueError")


def test_probability_matrix_from_evolutor_identity():

    section("TEST: probability_matrix_from_evolutor identity")

    S = torch.eye(3, dtype=torch.complex128)

    P = probability_matrix_from_evolutor(S)

    P_expected = torch.eye(3, dtype=torch.float64)

    print("P:")
    print(P)

    assert_true(P.shape == (3, 3), "P has shape (3, 3)")
    assert_true(P.dtype == torch.float64, "P dtype is float64")
    assert_close(P, P_expected, name="Identity evolutor gives identity probability")


def test_probability_matrix_from_known_rotation():

    section("TEST: probability_matrix_from_evolutor known rotation")

    S = build_unitary_test_operator()
    P = probability_matrix_from_evolutor(S)

    theta = torch.tensor(0.3, dtype=torch.float64)
    c = torch.cos(theta)
    s = torch.sin(theta)

    P_expected = torch.tensor(
        [
            [c**2, s**2, 0.0],
            [s**2, c**2, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    print("S:")
    print(S)
    print("P:")
    print(P)

    assert_close(
        P,
        P_expected,
        name="Known rotation probability matrix",
        atol=1.0e-12,
        rtol=1.0e-10,
    )


def test_transition_probability():

    section("TEST: transition_probability")

    S = build_unitary_test_operator()
    P = probability_matrix_from_evolutor(S)

    P_e_to_mu_from_S = transition_probability(
        S,
        alpha="e",
        beta="mu",
        input_is_probability=False,
    )

    P_e_to_mu_from_P = transition_probability(
        P,
        alpha="e",
        beta="mu",
        input_is_probability=True,
    )

    expected = P[1, 0]

    print(f"P(e -> mu) from S = {P_e_to_mu_from_S.item():.12e}")
    print(f"P(e -> mu) from P = {P_e_to_mu_from_P.item():.12e}")
    print(f"Expected           = {expected.item():.12e}")

    assert_close(P_e_to_mu_from_S, expected, name="Transition probability from S")
    assert_close(P_e_to_mu_from_P, expected, name="Transition probability from P")


def test_survival_probability():

    section("TEST: survival_probability")

    S = build_unitary_test_operator()
    P = probability_matrix_from_evolutor(S)

    P_e_surv = survival_probability(
        P,
        alpha="e",
        input_is_probability=True,
    )

    expected = P[0, 0]

    print(f"P(e -> e) = {P_e_surv.item():.12e}")

    assert_close(P_e_surv, expected, name="Survival probability P(e -> e)")


def test_probability_columns_sum():

    section("TEST: probability_columns_sum")

    S = build_unitary_test_operator()
    P = probability_matrix_from_evolutor(S)

    colsum = probability_columns_sum(P)
    expected = torch.ones(3, dtype=P.dtype)

    print("P:")
    print(P)
    print("Column sums:")
    print(colsum)

    assert_close(
        colsum,
        expected,
        name="Column sums equal one",
        atol=1.0e-12,
        rtol=1.0e-10,
    )


def test_check_probability_conservation():

    section("TEST: check_probability_conservation")

    S = build_unitary_test_operator()
    P = probability_matrix_from_evolutor(S)

    ok = check_probability_conservation(P)

    print("P:")
    print(P)
    print("Conservation check:", ok)

    assert_true(ok, "Probability conservation passes for unitary operator")

    P_bad = P.clone()
    P_bad[0, 0] += 0.1

    ok_bad = check_probability_conservation(P_bad)

    print("P_bad:")
    print(P_bad)
    print("Conservation check for P_bad:", ok_bad)

    assert_true(not ok_bad, "Probability conservation fails for modified matrix")


def test_check_probability_matrix():

    section("TEST: check_probability_matrix")

    S = build_unitary_test_operator()
    P = probability_matrix_from_evolutor(S)

    ok = check_probability_matrix(P)

    print("P:")
    print(P)
    print("Probability matrix check:", ok)

    assert_true(ok, "Probability matrix passes all checks")

    P_bad = P.clone()
    P_bad[0, 0] = -0.1

    ok_bad = check_probability_matrix(P_bad)

    print("P_bad:")
    print(P_bad)
    print("Probability matrix check for P_bad:", ok_bad)

    assert_true(not ok_bad, "Probability matrix check fails for negative value")


def test_normalize_probability_columns():

    section("TEST: normalize_probability_columns")

    P = torch.tensor(
        [
            [0.6, 0.2, 0.3],
            [0.3, 0.5, 0.4],
            [0.2, 0.1, 0.5],
        ],
        dtype=torch.float64,
    )

    P_norm = normalize_probability_columns(P)
    colsum = probability_columns_sum(P_norm)

    print("P:")
    print(P)
    print("P_norm:")
    print(P_norm)
    print("Column sums:")
    print(colsum)

    assert_close(
        colsum,
        torch.ones(3, dtype=torch.float64),
        name="Normalized columns sum to one",
        atol=1.0e-12,
        rtol=1.0e-10,
    )


def test_apply_probability_matrix_to_flux_identity():

    section("TEST: apply_probability_matrix_to_flux identity")

    P = torch.eye(3, dtype=torch.float64)

    flux_initial = torch.tensor(
        [1.0, 2.0, 3.0],
        dtype=torch.float64,
    )

    flux_final = apply_probability_matrix_to_flux(
        P,
        flux_initial,
    )

    print("flux_initial:", flux_initial)
    print("flux_final  :", flux_final)

    assert_close(
        flux_final,
        flux_initial,
        name="Identity probability leaves flux unchanged",
        atol=1.0e-12,
        rtol=1.0e-10,
    )


def test_apply_probability_matrix_to_flux_known_matrix():

    section("TEST: apply_probability_matrix_to_flux known matrix")

    S = build_unitary_test_operator()
    P = probability_matrix_from_evolutor(S)

    flux_initial = torch.tensor(
        [10.0, 5.0, 1.0],
        dtype=torch.float64,
    )

    flux_final = apply_probability_matrix_to_flux(
        P,
        flux_initial,
    )

    flux_expected = P @ flux_initial

    print("P:")
    print(P)
    print("flux_initial:", flux_initial)
    print("flux_final  :", flux_final)

    assert_close(
        flux_final,
        flux_expected,
        name="flux transformation flux_final = P @ flux_initial",
        atol=1.0e-12,
        rtol=1.0e-10,
    )


def test_probability_matrix_from_segment_evolutor():

    section("TEST: probability_matrix_from real segment evolutor")

    S = build_segment_evolutor()
    P = probability_matrix_from_evolutor(S)

    colsum = probability_columns_sum(P)

    print("S:")
    print(S)
    print("P:")
    print(P)
    print("Column sums:")
    print(colsum)

    assert_true(P.shape == (3, 3), "Segment probability matrix has shape (3, 3)")
    assert_true(torch.isfinite(P).all().item(), "Segment probability matrix contains finite values")
    assert_true((P >= -1.0e-8).all().item(), "Segment probabilities are non-negative within tolerance")

    ok = check_probability_conservation(
        P,
        atol=1.0e-3,
        rtol=1.0e-3,
    )

    assert_true(ok, "Segment probability approximately conserves probability")


def test_batched_probability_matrix_and_flux():

    section("TEST: batched probability matrix and flux")

    theta = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    c = torch.cos(theta)
    s = torch.sin(theta)

    S = torch.zeros((3, 3, 3), dtype=torch.complex128)

    S[:, 0, 0] = c
    S[:, 0, 1] = s
    S[:, 1, 0] = -s
    S[:, 1, 1] = c
    S[:, 2, 2] = 1.0

    P = probability_matrix_from_evolutor(S)

    flux_initial = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ],
        dtype=torch.float64,
    )

    flux_final = apply_probability_matrix_to_flux(P, flux_initial)
    flux_expected = torch.einsum("...ba,...a->...b", P, flux_initial)

    print("P shape:", P.shape)
    print("flux_initial shape:", flux_initial.shape)
    print("flux_final shape:", flux_final.shape)
    print("P:")
    print(P)
    print("flux_final:")
    print(flux_final)

    assert_true(P.shape == (3, 3, 3), "Batched P has shape (batch, 3, 3)")
    assert_true(flux_final.shape == (3, 3), "Batched flux_final has shape (batch, 3)")
    assert_close(
        flux_final,
        flux_expected,
        name="Batched flux transformation",
        atol=1.0e-12,
        rtol=1.0e-10,
    )

    colsum = probability_columns_sum(P)

    assert_close(
        colsum,
        torch.ones_like(colsum),
        name="Batched probability column conservation",
        atol=1.0e-12,
        rtol=1.0e-10,
    )


# ============================================================
# Main
# ============================================================

# ============================================================
# Runner
# ============================================================

def run_test8_probabilities_tests(verbose_traceback=False):
    tests = [
        test_flavour_index,
        test_probability_matrix_from_evolutor_identity,
        test_probability_matrix_from_known_rotation,
        test_transition_probability,
        test_survival_probability,
        test_probability_columns_sum,
        test_check_probability_conservation,
        test_check_probability_matrix,
        test_normalize_probability_columns,
        test_apply_probability_matrix_to_flux_identity,
        test_apply_probability_matrix_to_flux_known_matrix,
        test_probability_matrix_from_segment_evolutor,
        test_batched_probability_matrix_and_flux,
    ]
    return run_test_suite(
        tests,
        suite_name="core TEST8 PROBABILITIES tests",
        verbose_traceback=verbose_traceback,
    )


if __name__ == "__main__":
    run_test8_probabilities_tests(verbose_traceback=True)

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
Verbose tests for peanuts_torch.core.pmns.

Run with:

    pytest tests/core/test_pmns.py -v -s

or directly:

    python tests/core/test_pmns.py

This version prints detailed diagnostics, matrix errors,
unitarity precision, shapes, dtypes, and batch behaviour.
"""



from __future__ import annotations

import math
import torch

from tpeanuts.core.pmns import PMNS

from tpeanuts.util.test_utils import (
    ATOL, RTOL, printoptions,
    banner, section, print_ok, print_fail,
    max_abs_error, assert_close, assert_true,
    default_inputs, build_pmns, run_test_suite
    )
printoptions()

# ============================================================
# tests
# ============================================================

def test_pmns_shapes_scalar():

    section("TEST: Scalar shapes")

    pmns = build_pmns()

    print("R12 shape :", pmns.R12().shape)
    print("R13 shape :", pmns.R13().shape)
    print("R23 shape :", pmns.R23().shape)
    print("Delta shape :", pmns.Delta().shape)
    print("PMNS shape :", pmns.pmns_matrix().shape)
    print("Reduced shape :", pmns.reduced().shape)

    assert pmns.R12().shape == (3, 3)
    assert pmns.R13().shape == (3, 3)
    assert pmns.R23().shape == (3, 3)
    assert pmns.Delta().shape == (3, 3)
    assert pmns.pmns_matrix().shape == (3, 3)
    assert pmns.reduced().shape == (3, 3)

    print_ok("All scalar shapes are correct")


def test_pmns_dtype():

    section("TEST: Dtypes")

    pmns = build_pmns()

    print("theta12 dtype :", pmns.theta12.dtype)
    print("PMNS dtype    :", pmns.pmns_matrix().dtype)
    print("Reduced dtype :", pmns.reduced().dtype)

    assert pmns.theta12.dtype == torch.float64
    assert pmns.pmns_matrix().dtype == torch.complex128
    assert pmns.reduced().dtype == torch.complex128

    print_ok("All dtypes are correct")


def test_reduced_definition():

    section("TEST: Reduced matrix definition")

    pmns = build_pmns()

    Ured_expected = pmns.R13() @ pmns.R12()
    Ured = pmns.reduced()

    print("Ured:")
    print(Ured)

    assert_close(
        Ured,
        Ured_expected,
        name="Ured = R13 @ R12",
    )


def test_full_pmns_definition():

    section("TEST: Full PMNS definition")

    pmns = build_pmns()

    U_expected = (
        pmns.R23()
        @ pmns.Delta()
        @ pmns.R13()
        @ pmns.Delta().conj()
        @ pmns.R12()
    )

    U = pmns.pmns_matrix()

    print("PMNS matrix:")
    print(U)

    assert_close(
        U,
        U_expected,
        name="U_PMNS construction",
    )


def test_cached_matrices():

    section("TEST: Cached matrices")

    pmns = build_pmns()

    assert_close(
        pmns.U,
        pmns.reduced(),
        name="Cached reduced matrix",
    )

    assert_close(
        pmns.pmns,
        pmns.pmns_matrix(),
        name="Cached PMNS matrix",
    )


def test_pmns_unitarity():

    section("TEST: Full PMNS unitarity")

    pmns = build_pmns()

    U = pmns.pmns_matrix()

    I = torch.eye(
        3,
        dtype=U.dtype,
        device=U.device,
    )

    left = U.conj().transpose(-2, -1) @ U
    right = U @ U.conj().transpose(-2, -1)

    print("U†U:")
    print(left)

    print("UU†:")
    print(right)

    assert_close(
        left,
        I,
        name="U†U = I",
    )

    assert_close(
        right,
        I,
        name="UU† = I",
    )


def test_reduced_unitarity():

    section("TEST: Reduced matrix unitarity")

    pmns = build_pmns()

    Ured = pmns.reduced()

    I = torch.eye(
        3,
        dtype=Ured.dtype,
        device=Ured.device,
    )

    left = Ured.conj().transpose(-2, -1) @ Ured
    right = Ured @ Ured.conj().transpose(-2, -1)

    print("Ured† Ured:")
    print(left)

    print("Ured Ured†:")
    print(right)

    assert_close(
        left,
        I,
        name="Ured†Ured = I",
    )

    assert_close(
        right,
        I,
        name="UredUred† = I",
    )


def test_dagger():

    section("TEST: Hermitian conjugate")

    pmns = build_pmns()

    U = pmns.pmns_matrix()
    Udagger_expected = U.conj().transpose(-2, -1)

    assert_close(
        pmns.dagger(),
        Udagger_expected,
        name="dagger() correctness",
    )


def test_batch_shapes():

    section("TEST: Batch shapes")

    theta12 = torch.tensor([0.58, 0.59, 0.60], dtype=torch.float64)
    theta13 = torch.tensor([0.14, 0.15, 0.16], dtype=torch.float64)
    theta23 = torch.tensor([0.77, 0.78, 0.79], dtype=torch.float64)
    delta = torch.tensor([0.0, 1.2, math.pi], dtype=torch.float64)

    pmns = PMNS(
        theta12=theta12,
        theta13=theta13,
        theta23=theta23,
        delta=delta,
        real_dtype=torch.float64,
    )

    print("Batch PMNS shape:", pmns.pmns_matrix().shape)
    print("Batch Reduced shape:", pmns.reduced().shape)

    assert pmns.pmns_matrix().shape == (3, 3, 3)
    assert pmns.reduced().shape == (3, 3, 3)

    print_ok("Batch shapes are correct")


def test_batch_unitarity():

    section("TEST: Batch unitarity")

    theta12 = torch.tensor([0.58, 0.59, 0.60], dtype=torch.float64)
    theta13 = torch.tensor([0.14, 0.15, 0.16], dtype=torch.float64)
    theta23 = torch.tensor([0.77, 0.78, 0.79], dtype=torch.float64)
    delta = torch.tensor([0.0, 1.2, math.pi], dtype=torch.float64)

    pmns = PMNS(
        theta12=theta12,
        theta13=theta13,
        theta23=theta23,
        delta=delta,
        real_dtype=torch.float64,
    )

    U = pmns.pmns_matrix()

    I = torch.eye(
        3,
        dtype=U.dtype,
        device=U.device,
    ).expand(3, 3, 3)

    left = U.conj().transpose(-2, -1) @ U

    print("Batch U†U:")
    print(left)

    assert_close(
        left,
        I,
        name="Batch unitarity",
    )


def test_zero_angles_identity():

    section("TEST: Zero-angle identity limit")

    pmns = PMNS(
        theta12=0.0,
        theta13=0.0,
        theta23=0.0,
        delta=0.0,
        real_dtype=torch.float64,
    )

    I = torch.eye(3, dtype=torch.complex128)

    assert_close(pmns.R12(), I, name="R12 identity")
    assert_close(pmns.R13(), I, name="R13 identity")
    assert_close(pmns.R23(), I, name="R23 identity")
    assert_close(pmns.Delta(), I, name="Delta identity")
    assert_close(pmns.reduced(), I, name="Reduced identity")
    assert_close(pmns.pmns_matrix(), I, name="PMNS identity")


# ============================================================
# Main
# ============================================================

# ============================================================
# Runner
# ============================================================

def run_test1_PMNS_tests(verbose_traceback=False):
    tests = [
        test_pmns_shapes_scalar,
        test_pmns_dtype,
        test_reduced_definition,
        test_full_pmns_definition,
        test_cached_matrices,
        test_pmns_unitarity,
        test_reduced_unitarity,
        test_dagger,
        test_batch_shapes,
        test_batch_unitarity,
        test_zero_angles_identity,
    ]
    return run_test_suite(
        tests,
        suite_name="core TEST1 PMNS tests",
        verbose_traceback=verbose_traceback,
    )


if __name__ == "__main__":
    run_test1_PMNS_tests(verbose_traceback=True)

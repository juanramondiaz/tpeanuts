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
Verbose tests for tpeanuts.core.evolution.

Run with:

    pytest tests/core/test_evolution.py -v -s

or directly:

    python tests/core/test_evolution.py
"""



from __future__ import annotations

import torch

from tpeanuts.core.hamiltonian import (
    reduced_hamiltonian_from_polynomial_density,
)

from tpeanuts.core.spectral import (
    hamiltonian_trace_from_ki_and_V,
    hamiltonian_spectral_data,
)

from tpeanuts.core.evolution import (
    identity_evolutor_like,
    enforce_identity_for_zero_length,
    constant_density_evolutor_from_spectral,
    constant_density_evolutor,
    matrix_exp_evolutor,
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


def build_segment():
    pmns = build_pmns()

    H, ki, V, L, zero_mask = reduced_hamiltonian_from_polynomial_density(
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

    trace_H = hamiltonian_trace_from_ki_and_V(ki, V)

    return H, ki, V, L, zero_mask, trace_H


def build_batch_segment():
    pmns = build_pmns()

    E = torch.tensor([500.0, 1000.0, 5000.0], dtype=torch.float64)
    x1 = torch.tensor([0.10, 0.20, 0.30], dtype=torch.float64)
    x2 = torch.tensor([0.40, 0.60, 0.90], dtype=torch.float64)

    a = torch.tensor([1.00, 1.10, 1.20], dtype=torch.float64)
    b = torch.tensor([0.10, 0.20, 0.30], dtype=torch.float64)
    c = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64)

    H, ki, V, L, zero_mask = reduced_hamiltonian_from_polynomial_density(
        DeltamSq21=7.42e-5,
        DeltamSq3l=2.517e-3,
        pmns=pmns,
        E_MeV=E,
        x1=x1,
        x2=x2,
        a=a,
        b=b,
        c=c,
        antinu=False,
    )

    trace_H = hamiltonian_trace_from_ki_and_V(ki, V)

    return H, ki, V, L, zero_mask, trace_H


# ============================================================
# tests
# ============================================================

def test_identity_evolutor_like_scalar():

    section("TEST: identity_evolutor_like scalar")

    L = torch.tensor(0.5, dtype=torch.float64)

    I = identity_evolutor_like(
        L,
        device=L.device,
        dtype=torch.complex128,
    )

    I_expected = torch.eye(3, dtype=torch.complex128)

    print("I:")
    print(I)

    assert_true(I.shape == (3, 3), "Scalar identity evolutor has shape (3, 3)")
    assert_close(I, I_expected, name="Scalar identity evolutor")


def test_identity_evolutor_like_batch():

    section("TEST: identity_evolutor_like batch")

    L = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)

    I = identity_evolutor_like(
        L,
        device=L.device,
        dtype=torch.complex128,
    )

    I_expected = torch.eye(3, dtype=torch.complex128).expand(3, 3, 3)

    print("I batch shape:", I.shape)
    print(I)

    assert_true(I.shape == (3, 3, 3), "Batch identity evolutor has shape (batch, 3, 3)")
    assert_close(I, I_expected, name="Batch identity evolutor")


def test_enforce_identity_for_zero_length_scalar():

    section("TEST: enforce_identity_for_zero_length scalar")

    U = torch.ones((3, 3), dtype=torch.complex128)
    L = torch.tensor(0.0, dtype=torch.float64)
    zero_mask = torch.tensor(True)

    U_corrected = enforce_identity_for_zero_length(
        U,
        L,
        zero_mask,
    )

    I = torch.eye(3, dtype=torch.complex128)

    print("U corrected:")
    print(U_corrected)

    assert_close(U_corrected, I, name="Scalar zero-length segment gives identity")


def test_enforce_identity_for_zero_length_batch():

    section("TEST: enforce_identity_for_zero_length batch")

    U = torch.ones((3, 3, 3), dtype=torch.complex128)

    L = torch.tensor([0.0, 0.5, 0.0], dtype=torch.float64)
    zero_mask = L == 0.0

    U_corrected = enforce_identity_for_zero_length(
        U,
        L,
        zero_mask,
    )

    I = torch.eye(3, dtype=torch.complex128)

    print("zero_mask:", zero_mask)
    print("U corrected:")
    print(U_corrected)

    assert_close(U_corrected[0], I, name="Batch element 0 replaced by identity")
    assert_close(U_corrected[2], I, name="Batch element 2 replaced by identity")
    assert_close(U_corrected[1], U[1], name="Batch element 1 remains unchanged")


def test_constant_density_evolutor_from_spectral_vs_matrix_exp():

    section("TEST: constant_density_evolutor_from_spectral vs matrix_exp")

    H, ki, V, L, zero_mask, trace_H = build_segment()

    spectral = hamiltonian_spectral_data(
        H,
        trace_H=trace_H,
    )

    U_spectral = constant_density_evolutor_from_spectral(
        lam=spectral["lam"],
        M=spectral["M"],
        trace_H=spectral["trace"],
        L=L,
    )

    U_exp = torch.matrix_exp(
        -1j * H * L.to(dtype=H.dtype)
    )

    print("U_spectral:")
    print(U_spectral)
    print("U_exp:")
    print(U_exp)

    assert_close(
        U_spectral,
        U_exp,
        name="Spectral evolutor equals torch.matrix_exp",
        atol=1.0e-7,
        rtol=1.0e-6,
    )


def test_constant_density_evolutor_high_level_vs_matrix_exp():

    section("TEST: constant_density_evolutor high-level vs matrix_exp_evolutor")

    H, ki, V, L, zero_mask, trace_H = build_segment()

    U0 = constant_density_evolutor(
        H=H,
        L=L,
        trace_H=trace_H,
        zero_mask=zero_mask,
    )

    U_exp = matrix_exp_evolutor(
        H=H,
        L=L,
        zero_mask=zero_mask,
    )

    print("U0:")
    print(U0)
    print("U_exp:")
    print(U_exp)

    assert_close(
        U0,
        U_exp,
        name="constant_density_evolutor equals matrix_exp_evolutor",
        atol=1.0e-7,
        rtol=1.0e-6,
    )


def test_matrix_exp_evolutor_direct():

    section("TEST: matrix_exp_evolutor direct formula")

    H, ki, V, L, zero_mask, trace_H = build_segment()

    U = matrix_exp_evolutor(
        H=H,
        L=L,
        zero_mask=zero_mask,
    )

    U_expected = torch.matrix_exp(
        -1j * H * L.to(dtype=H.dtype)
    )

    print("U:")
    print(U)

    assert_close(
        U,
        U_expected,
        name="matrix_exp_evolutor direct formula",
        atol=1.0e-12,
        rtol=1.0e-10,
    )


def test_unitarity_constant_density_evolutor():

    section("TEST: unitarity of constant_density_evolutor")

    H, ki, V, L, zero_mask, trace_H = build_segment()

    U0 = constant_density_evolutor(
        H=H,
        L=L,
        trace_H=trace_H,
        zero_mask=zero_mask,
    )

    I = torch.eye(3, dtype=U0.dtype, device=U0.device)

    left = U0.conj().transpose(-1, -2) @ U0
    right = U0 @ U0.conj().transpose(-1, -2)

    print("U0†U0:")
    print(left)

    print("U0U0†:")
    print(right)

    assert_close(
        left,
        I,
        name="U0†U0 = I",
        atol=1.0e-7,
        rtol=1.0e-6,
    )

    assert_close(
        right,
        I,
        name="U0U0† = I",
        atol=1.0e-7,
        rtol=1.0e-6,
    )


def test_zero_length_constant_density_evolutor():

    section("TEST: zero-length constant_density_evolutor")

    H, ki, V, L, zero_mask, trace_H = build_segment()

    L0 = torch.tensor(0.0, dtype=L.dtype)
    zero_mask0 = torch.tensor(True)

    U0 = constant_density_evolutor(
        H=H,
        L=L0,
        trace_H=trace_H,
        zero_mask=zero_mask0,
    )

    I = torch.eye(3, dtype=H.dtype, device=H.device)

    print("U0 zero length:")
    print(U0)

    assert_close(
        U0,
        I,
        name="Zero-length evolutor is identity",
        atol=1.0e-12,
        rtol=1.0e-10,
    )


def test_batch_constant_density_evolutor_vs_matrix_exp():

    section("TEST: batched constant_density_evolutor vs matrix_exp_evolutor")

    H, ki, V, L, zero_mask, trace_H = build_batch_segment()

    U0 = constant_density_evolutor(
        H=H,
        L=L,
        trace_H=trace_H,
        zero_mask=zero_mask,
    )

    U_exp = matrix_exp_evolutor(
        H=H,
        L=L,
        zero_mask=zero_mask,
    )

    print("H batch shape:", H.shape)
    print("L batch shape:", L.shape)
    print("U0 batch shape:", U0.shape)

    assert_true(U0.shape == (3, 3, 3), "Batched evolutor has shape (batch, 3, 3)")

    assert_close(
        U0,
        U_exp,
        name="Batched spectral evolutor equals batched matrix_exp",
        atol=1.0e-7,
        rtol=1.0e-6,
    )


def test_batch_unitarity():

    section("TEST: batched unitarity")

    H, ki, V, L, zero_mask, trace_H = build_batch_segment()

    U0 = constant_density_evolutor(
        H=H,
        L=L,
        trace_H=trace_H,
        zero_mask=zero_mask,
    )

    I = torch.eye(3, dtype=U0.dtype, device=U0.device).expand(3, 3, 3)

    left = U0.conj().transpose(-1, -2) @ U0
    right = U0 @ U0.conj().transpose(-1, -2)

    print("Batched U0†U0:")
    print(left)

    assert_close(
        left,
        I,
        name="Batched U0†U0 = I",
        atol=1.0e-7,
        rtol=1.0e-6,
    )

    assert_close(
        right,
        I,
        name="Batched U0U0† = I",
        atol=1.0e-7,
        rtol=1.0e-6,
    )


def test_batch_zero_length_enforcement():

    section("TEST: batched zero-length enforcement")

    pmns = build_pmns()

    E = torch.tensor([1000.0, 1000.0, 1000.0], dtype=torch.float64)
    x1 = torch.tensor([0.20, 0.20, 0.20], dtype=torch.float64)
    x2 = torch.tensor([0.20, 0.70, 0.20], dtype=torch.float64)

    a = torch.tensor([1.10, 1.10, 1.10], dtype=torch.float64)
    b = torch.tensor([0.20, 0.20, 0.20], dtype=torch.float64)
    c = torch.tensor([0.05, 0.05, 0.05], dtype=torch.float64)

    H, ki, V, L, zero_mask = reduced_hamiltonian_from_polynomial_density(
        DeltamSq21=7.42e-5,
        DeltamSq3l=2.517e-3,
        pmns=pmns,
        E_MeV=E,
        x1=x1,
        x2=x2,
        a=a,
        b=b,
        c=c,
        antinu=False,
    )

    trace_H = hamiltonian_trace_from_ki_and_V(ki, V)

    U0 = constant_density_evolutor(
        H=H,
        L=L,
        trace_H=trace_H,
        zero_mask=zero_mask,
    )

    I = torch.eye(3, dtype=U0.dtype, device=U0.device)

    print("L:", L)
    print("zero_mask:", zero_mask)
    print("U0:")
    print(U0)

    assert_close(
        U0[0],
        I,
        name="Batch zero-length element 0 is identity",
        atol=1.0e-12,
        rtol=1.0e-10,
    )

    assert_close(
        U0[2],
        I,
        name="Batch zero-length element 2 is identity",
        atol=1.0e-12,
        rtol=1.0e-10,
    )

    assert_true(
        torch.isfinite(U0[1]).all().item(),
        "Non-zero batch element is finite",
    )


# ============================================================
# Main
# ============================================================

# ============================================================
# Runner
# ============================================================

def run_test4_evolution_tests(verbose_traceback=False):
    tests = [
        test_identity_evolutor_like_scalar,
        test_identity_evolutor_like_batch,
        test_enforce_identity_for_zero_length_scalar,
        test_enforce_identity_for_zero_length_batch,
        test_constant_density_evolutor_from_spectral_vs_matrix_exp,
        test_constant_density_evolutor_high_level_vs_matrix_exp,
        test_matrix_exp_evolutor_direct,
        test_unitarity_constant_density_evolutor,
        test_zero_length_constant_density_evolutor,
        test_batch_constant_density_evolutor_vs_matrix_exp,
        test_batch_unitarity,
        test_batch_zero_length_enforcement,
    ]
    return run_test_suite(
        tests,
        suite_name="core TEST4 EVOLUTION tests",
        verbose_traceback=verbose_traceback,
    )


if __name__ == "__main__":
    run_test4_evolution_tests(verbose_traceback=True)

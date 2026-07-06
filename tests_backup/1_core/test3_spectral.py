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
Verbose tests for peanuts_torch.core.spectral.

Run with:

    pytest tests/core/test_spectral.py -v -s

or directly:

    python tests/core/test_spectral.py
"""



from __future__ import annotations

import torch

from tpeanuts.core.hamiltonian import (
    reduced_hamiltonian_from_polynomial_density,
)

from tpeanuts.core.spectral import (
    identity3_like,
    hamiltonian_trace_from_ki_and_V,
    traceless_hamiltonian,
    hermitize,
    traceless_invariant_c1,
    traceless_eigenvalues,
    spectral_projectors_traceless,
    hamiltonian_spectral_data,
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


def build_hamiltonian():
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

    return H, ki, V, L, zero_mask


# ============================================================
# tests
# ============================================================

def test_identity3_like():

    section("TEST: identity3_like")

    H, *_ = build_hamiltonian()

    I = identity3_like(H)
    I_expected = torch.eye(3, dtype=H.dtype, device=H.device)

    print("I:")
    print(I)

    assert_true(I.shape == (3, 3), "I has shape (3, 3)")
    assert_true(I.dtype == H.dtype, "I has same dtype as H")
    assert_true(I.device == H.device, "I has same device as H")

    assert_close(I, I_expected, name="identity3_like correctness")


def test_hamiltonian_trace_from_ki_and_V():

    section("TEST: hamiltonian_trace_from_ki_and_V")

    H, ki, V, *_ = build_hamiltonian()

    trace_expected = torch.diagonal(H, dim1=-2, dim2=-1).sum(dim=-1)
    trace_from_ki = hamiltonian_trace_from_ki_and_V(ki, V)

    print(f"trace from H      = {trace_expected}")
    print(f"trace from ki,V   = {trace_from_ki}")

    assert_close(
        trace_from_ki,
        trace_expected,
        name="Tr(H) = k1 + k2 + k3 + V",
    )


def test_traceless_hamiltonian():

    section("TEST: traceless_hamiltonian")

    H, *_ = build_hamiltonian()

    T, trace_H = traceless_hamiltonian(H)

    trace_T = torch.diagonal(T, dim1=-2, dim2=-1).sum(dim=-1)
    trace_H_expected = torch.diagonal(H, dim1=-2, dim2=-1).sum(dim=-1)

    print("H:")
    print(H)
    print("T:")
    print(T)
    print(f"Tr(H) = {trace_H}")
    print(f"Tr(T) = {trace_T}")

    assert_close(trace_H, trace_H_expected, name="Trace of H")
    assert_close(trace_T, torch.zeros_like(trace_T), name="Trace of T is zero", atol=1.0e-9, rtol=1.0e-9)

    H_reconstructed = T + trace_H * torch.eye(3, dtype=H.dtype, device=H.device) / 3.0

    assert_close(H_reconstructed, H, name="H = T + Tr(H)/3 I")


def test_hermitize():

    section("TEST: hermitize")

    H, *_ = build_hamiltonian()

    H_nonherm = H.clone()
    H_nonherm[0, 1] += 1.0e-6 + 2.0e-6j

    H_herm = hermitize(H_nonherm)
    herm_error = torch.max(torch.abs(H_herm - H_herm.conj().transpose(-1, -2)))

    print("Hermitized H:")
    print(H_herm)
    print(f"max|H-H†| = {herm_error.item():.6e}")

    assert_close(
        H_herm,
        H_herm.conj().transpose(-1, -2),
        name="Hermitized matrix is Hermitian",
        atol=1.0e-14,
        rtol=1.0e-12,
    )


def test_traceless_invariant_c1():

    section("TEST: traceless_invariant_c1")

    H, *_ = build_hamiltonian()

    T, _ = traceless_hamiltonian(H)
    T = hermitize(T)

    c1 = traceless_invariant_c1(T)

    T2 = T @ T
    c1_expected = -torch.diagonal(T2, dim1=-2, dim2=-1).sum(dim=-1) / 2.0

    print(f"c1          = {c1}")
    print(f"c1 expected = {c1_expected}")

    assert_close(c1, c1_expected, name="c1 = -Tr(T^2)/2")


def test_traceless_eigenvalues():

    section("TEST: traceless_eigenvalues")

    H, *_ = build_hamiltonian()

    T, _ = traceless_hamiltonian(H)
    T = hermitize(T)

    lam = traceless_eigenvalues(T)
    lam_expected = torch.linalg.eigvalsh(T).to(dtype=T.dtype)

    print("lambda:")
    print(lam)
    print(f"sum(lambda) = {lam.sum()}")

    assert_true(lam.shape == (3,), "lambda has shape (3,)")
    assert_true(torch.isfinite(lam).all().item(), "lambda contains only finite values")
    assert_close(lam, lam_expected, name="Eigenvalues match torch.linalg.eigvalsh")
    assert_close(lam.sum(), torch.zeros((), dtype=lam.dtype), name="Sum of traceless eigenvalues is zero", atol=1.0e-9, rtol=1.0e-9)


def test_spectral_projectors():

    section("TEST: spectral_projectors_traceless")

    H, *_ = build_hamiltonian()

    T, _ = traceless_hamiltonian(H)
    T = hermitize(T)

    M, lam, c1 = spectral_projectors_traceless(T)

    I = torch.eye(3, dtype=T.dtype, device=T.device)

    print("lambda:")
    print(lam)
    print("c1:")
    print(c1)
    print("Projectors M shape:", M.shape)

    assert_true(M.shape == (3, 3, 3), "M has shape (3, 3, 3)")
    assert_true(torch.isfinite(M).all().item(), "M contains only finite values")

    sum_M = M.sum(dim=-3)

    assert_close(sum_M, I, name="Sum_a M_a = I", atol=1.0e-8, rtol=1.0e-7)

    for a in range(3):
        Ma = M[a]

        assert_close(
            Ma,
            Ma.conj().transpose(-1, -2),
            name=f"M_{a} is Hermitian",
            atol=1.0e-8,
            rtol=1.0e-7,
        )

        assert_close(
            Ma @ Ma,
            Ma,
            name=f"M_{a}^2 = M_{a}",
            atol=1.0e-7,
            rtol=1.0e-6,
        )

    for a in range(3):
        for b in range(3):
            if a != b:
                assert_close(
                    M[a] @ M[b],
                    torch.zeros_like(I),
                    name=f"M_{a} M_{b} = 0",
                    atol=1.0e-7,
                    rtol=1.0e-6,
                )

    T_reconstructed = sum(lam[a] * M[a] for a in range(3))

    assert_close(
        T_reconstructed,
        T,
        name="T = Sum_a lambda_a M_a",
        atol=1.0e-7,
        rtol=1.0e-6,
    )


def test_hamiltonian_spectral_data():

    section("TEST: hamiltonian_spectral_data")

    H, ki, V, *_ = build_hamiltonian()

    trace_H = hamiltonian_trace_from_ki_and_V(ki, V)

    data = hamiltonian_spectral_data(H, trace_H=trace_H)

    T = data["T"]
    trace = data["trace"]
    lam = data["lam"]
    c1 = data["c1"]
    M = data["M"]

    print("Returned keys:", list(data.keys()))
    print("T shape:", T.shape)
    print("trace shape:", trace.shape)
    print("lambda shape:", lam.shape)
    print("c1 shape:", c1.shape)
    print("M shape:", M.shape)

    assert_true(set(data.keys()) == {"T", "trace", "lam", "c1", "M"}, "Returned keys are correct")
    assert_true(T.shape == (3, 3), "T shape is correct")
    assert_true(lam.shape == (3,), "lambda shape is correct")
    assert_true(M.shape == (3, 3, 3), "M shape is correct")

    I = torch.eye(3, dtype=H.dtype, device=H.device)

    H_reconstructed = sum(
        (lam[a] + trace / 3.0) * M[a]
        for a in range(3)
    )

    assert_close(
        H_reconstructed,
        H,
        name="H = Sum_a (lambda_a + Tr(H)/3) M_a",
        atol=1.0e-7,
        rtol=1.0e-6,
    )

    assert_close(
        M.sum(dim=-3),
        I,
        name="Spectral data projectors sum to identity",
        atol=1.0e-8,
        rtol=1.0e-7,
    )


def test_batch_spectral_data():

    section("TEST: batched spectral data")

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
    data = hamiltonian_spectral_data(H, trace_H=trace_H)

    T = data["T"]
    lam = data["lam"]
    M = data["M"]

    print("H batch shape:", H.shape)
    print("T batch shape:", T.shape)
    print("lambda batch shape:", lam.shape)
    print("M batch shape:", M.shape)

    assert_true(T.shape == (3, 3, 3), "Batched T has shape (batch, 3, 3)")
    assert_true(lam.shape == (3, 3), "Batched lambda has shape (batch, 3)")
    assert_true(M.shape == (3, 3, 3, 3), "Batched M has shape (batch, 3, 3, 3)")

    I = torch.eye(3, dtype=H.dtype, device=H.device).expand(3, 3, 3)

    assert_close(
        M.sum(dim=-3),
        I,
        name="Batched projectors sum to identity",
        atol=1.0e-8,
        rtol=1.0e-7,
    )

    H_reconstructed = (
        (lam[..., 0, None, None] + trace_H[..., None, None] / 3.0) * M[..., 0, :, :]
        + (lam[..., 1, None, None] + trace_H[..., None, None] / 3.0) * M[..., 1, :, :]
        + (lam[..., 2, None, None] + trace_H[..., None, None] / 3.0) * M[..., 2, :, :]
    )

    assert_close(
        H_reconstructed,
        H,
        name="Batched H spectral reconstruction",
        atol=1.0e-7,
        rtol=1.0e-6,
    )


# ============================================================
# Main
# ============================================================

# ============================================================
# Runner
# ============================================================

def run_test3_spectral_tests(verbose_traceback=False):
    tests = [
        test_identity3_like,
        test_hamiltonian_trace_from_ki_and_V,
        test_traceless_hamiltonian,
        test_hermitize,
        test_traceless_invariant_c1,
        test_traceless_eigenvalues,
        test_spectral_projectors,
        test_hamiltonian_spectral_data,
        test_batch_spectral_data,
    ]
    return run_test_suite(
        tests,
        suite_name="core TEST3 SPECTRAL tests",
        verbose_traceback=verbose_traceback,
    )


if __name__ == "__main__":
    run_test3_spectral_tests(verbose_traceback=True)

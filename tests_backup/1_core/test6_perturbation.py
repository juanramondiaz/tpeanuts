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
Verbose tests for tpeanuts.core.perturbation.

Run with:

    pytest tests/core/test_perturbation.py -v -s

or directly:

    python tests/core/test_perturbation.py
"""



from __future__ import annotations

import torch

from tpeanuts.core.hamiltonian import (
    average_polynomial_density,
    reduced_hamiltonian_from_polynomial_density,
)

from tpeanuts.core.spectral import (
    hamiltonian_trace_from_ki_and_V,
    hamiltonian_spectral_data,
)

from tpeanuts.core.evolution import (
    constant_density_evolutor,
)

from tpeanuts.core.perturbation import (
    has_density_perturbation,
    polynomial_density_delta_constant,
    first_order_integral_Iab,
    first_order_correction_from_projectors,
    first_order_density_correction,
    perturbative_segment_evolutor,
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

def build_segment(
    *,
    b_value: float | None = None,
    c_value: float | None = None,
    zero_length: bool = False,
):
    p = default_inputs()
    pmns = build_pmns()

    if b_value is not None:
        p["b"] = torch.tensor(b_value, dtype=torch.float64)

    if c_value is not None:
        p["c"] = torch.tensor(c_value, dtype=torch.float64)

    if zero_length:
        p["x2"] = p["x1"].clone()

    H, ki, V, L, zero_mask = reduced_hamiltonian_from_polynomial_density(
        DeltamSq21=p["DeltamSq21"],
        DeltamSq3l=p["DeltamSq3l_NO"],
        pmns=pmns,
        E_MeV=p["E_MeV"],
        x1=p["x1"],
        x2=p["x2"],
        a=p["a"],
        b=p["b"],
        c=p["c"],
        antinu=False,
    )

    naverage, L2, zero_mask2 = average_polynomial_density(
        x1=p["x1"],
        x2=p["x2"],
        a=p["a"],
        b=p["b"],
        c=p["c"],
    )

    trace_H = hamiltonian_trace_from_ki_and_V(
        ki,
        V,
    )

    spectral = hamiltonian_spectral_data(
        H,
        trace_H=trace_H,
    )

    return H, ki, V, L, zero_mask, naverage, trace_H, spectral, p


# ============================================================
# tests
# ============================================================

def test_has_density_perturbation():

    section("TEST: has_density_perturbation")

    b0 = torch.tensor(0.0, dtype=torch.float64)
    c0 = torch.tensor(0.0, dtype=torch.float64)

    b1 = torch.tensor(0.2, dtype=torch.float64)
    c1 = torch.tensor(0.0, dtype=torch.float64)

    b2 = torch.tensor(0.0, dtype=torch.float64)
    c2 = torch.tensor(0.05, dtype=torch.float64)

    b_batch = torch.tensor([0.0, 0.0, 0.2], dtype=torch.float64)
    c_batch = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)

    print("Case b=0, c=0:", has_density_perturbation(b0, c0))
    print("Case b!=0:", has_density_perturbation(b1, c1))
    print("Case c!=0:", has_density_perturbation(b2, c2))
    print("Batch case:", has_density_perturbation(b_batch, c_batch))

    assert_true(not has_density_perturbation(b0, c0), "No perturbation when b = c = 0")
    assert_true(has_density_perturbation(b1, c1), "Perturbation detected when b != 0")
    assert_true(has_density_perturbation(b2, c2), "Perturbation detected when c != 0")
    assert_true(has_density_perturbation(b_batch, c_batch), "Batch perturbation detected")


def test_polynomial_density_delta_constant():

    section("TEST: polynomial_density_delta_constant")

    H, ki, V, L, zero_mask, naverage, trace_H, spectral, p = build_segment()

    atilde = polynomial_density_delta_constant(
        p["a"],
        naverage,
    )

    expected = p["a"] - naverage

    print(f"a        = {p['a'].item():.12e}")
    print(f"naverage = {naverage.item():.12e}")
    print(f"atilde   = {atilde.item():.12e}")

    assert_close(
        atilde,
        expected,
        name="atilde = a - naverage",
        atol=1.0e-14,
        rtol=1.0e-12,
    )


def test_first_order_integral_Iab_shape_and_finiteness():

    section("TEST: first_order_integral_Iab shape and finiteness")

    H, ki, V, L, zero_mask, naverage, trace_H, spectral, p = build_segment()

    lam = spectral["lam"]
    eig_H = lam + trace_H[..., None] / 3.0

    la = eig_H[..., :, None]
    lb = eig_H[..., None, :]

    atilde = polynomial_density_delta_constant(
        p["a"],
        naverage,
    )

    Iab = first_order_integral_Iab(
        la=la,
        lb=lb,
        atilde=atilde,
        b=p["b"],
        c=p["c"],
        x2=p["x2"],
        x1=p["x1"],
    )

    print("la:")
    print(la)
    print("lb:")
    print(lb)
    print("Iab:")
    print(Iab)

    assert_true(Iab.shape == (3, 3), "Iab has shape (3, 3)")
    assert_true(torch.isfinite(Iab).all().item(), "Iab contains only finite values")


def test_first_order_integral_zero_diagonal():

    section("TEST: first_order_integral_Iab diagonal elements")

    H, ki, V, L, zero_mask, naverage, trace_H, spectral, p = build_segment()

    lam = spectral["lam"]
    eig_H = lam + trace_H[..., None] / 3.0

    la = eig_H[..., :, None]
    lb = eig_H[..., None, :]

    atilde = polynomial_density_delta_constant(
        p["a"],
        naverage,
    )

    Iab = first_order_integral_Iab(
        la=la,
        lb=lb,
        atilde=atilde,
        b=p["b"],
        c=p["c"],
        x2=p["x2"],
        x1=p["x1"],
    )

    diag = torch.diagonal(
        Iab,
        dim1=-2,
        dim2=-1,
    )

    print("diag(Iab):")
    print(diag)

    assert_close(
        diag,
        torch.zeros_like(diag),
        name="Iab diagonal is zero when la = lb",
        atol=1.0e-12,
        rtol=1.0e-10,
    )


def test_first_order_correction_from_projectors_zero_input():

    section("TEST: first_order_correction_from_projectors with zero Vcorr")

    H, ki, V, L, zero_mask, naverage, trace_H, spectral, p = build_segment()

    M = spectral["M"]
    Vcorr = torch.zeros((3, 3), dtype=M.dtype, device=M.device)

    u1 = first_order_correction_from_projectors(
        M=M,
        Vcorr=Vcorr,
    )

    print("u1:")
    print(u1)

    assert_close(
        u1,
        torch.zeros_like(u1),
        name="u1 is zero when Vcorr is zero",
        atol=1.0e-14,
        rtol=1.0e-12,
    )


def test_first_order_density_correction_shape_and_finiteness():

    section("TEST: first_order_density_correction shape and finiteness")

    H, ki, V, L, zero_mask, naverage, trace_H, spectral, p = build_segment()

    u1 = first_order_density_correction(
        M=spectral["M"],
        lam=spectral["lam"],
        trace_H=spectral["trace"],
        x1=p["x1"],
        x2=p["x2"],
        a=p["a"],
        b=p["b"],
        c=p["c"],
        naverage=naverage,
        antinu=False,
    )

    print("u1:")
    print(u1)
    print("max|u1| =", torch.max(torch.abs(u1)).item())

    assert_true(u1.shape == (3, 3), "u1 has shape (3, 3)")
    assert_true(torch.isfinite(u1).all().item(), "u1 contains only finite values")


def test_first_order_density_correction_zero_for_constant_density():

    section("TEST: first_order_density_correction is zero for constant density")

    H, ki, V, L, zero_mask, naverage, trace_H, spectral, p = build_segment(
        b_value=0.0,
        c_value=0.0,
    )

    u1 = first_order_density_correction(
        M=spectral["M"],
        lam=spectral["lam"],
        trace_H=spectral["trace"],
        x1=p["x1"],
        x2=p["x2"],
        a=p["a"],
        b=p["b"],
        c=p["c"],
        naverage=naverage,
        antinu=False,
    )

    print("u1 constant density:")
    print(u1)
    print("max|u1| =", torch.max(torch.abs(u1)).item())

    assert_close(
        u1,
        torch.zeros_like(u1),
        name="u1 = 0 for constant density",
        atol=1.0e-12,
        rtol=1.0e-10,
    )


def test_perturbative_segment_equals_u0_for_constant_density():

    section("TEST: perturbative_segment_evolutor equals u0 for constant density")

    H, ki, V, L, zero_mask, naverage, trace_H, spectral, p = build_segment(
        b_value=0.0,
        c_value=0.0,
    )

    U = perturbative_segment_evolutor(
        H=H,
        L=L,
        x1=p["x1"],
        x2=p["x2"],
        a=p["a"],
        b=p["b"],
        c=p["c"],
        naverage=naverage,
        trace_H=trace_H,
        zero_mask=zero_mask,
        antinu=False,
    )

    U0 = constant_density_evolutor(
        H=H,
        L=L,
        trace_H=trace_H,
        zero_mask=zero_mask,
    )

    print("U perturbative:")
    print(U)
    print("U0:")
    print(U0)

    assert_close(
        U,
        U0,
        name="U = U0 when b = c = 0",
        atol=1.0e-10,
        rtol=1.0e-8,
    )


def test_perturbative_segment_nonconstant_density_finite():

    section("TEST: perturbative_segment_evolutor non-constant density")

    H, ki, V, L, zero_mask, naverage, trace_H, spectral, p = build_segment()

    U = perturbative_segment_evolutor(
        H=H,
        L=L,
        x1=p["x1"],
        x2=p["x2"],
        a=p["a"],
        b=p["b"],
        c=p["c"],
        naverage=naverage,
        trace_H=trace_H,
        zero_mask=zero_mask,
        antinu=False,
    )

    U0 = constant_density_evolutor(
        H=H,
        L=L,
        trace_H=trace_H,
        zero_mask=zero_mask,
    )

    delta_U = U - U0

    print("U:")
    print(U)
    print("U0:")
    print(U0)
    print("U-U0:")
    print(delta_U)
    print("max|U-U0| =", torch.max(torch.abs(delta_U)).item())

    assert_true(U.shape == (3, 3), "U has shape (3, 3)")
    assert_true(torch.isfinite(U).all().item(), "U contains only finite values")
    assert_true(torch.isfinite(delta_U).all().item(), "U-U0 contains only finite values")


def test_zero_length_perturbative_segment_identity():

    section("TEST: zero-length perturbative segment gives identity")

    H, ki, V, L, zero_mask, naverage, trace_H, spectral, p = build_segment(
        zero_length=True,
    )

    U = perturbative_segment_evolutor(
        H=H,
        L=L,
        x1=p["x1"],
        x2=p["x2"],
        a=p["a"],
        b=p["b"],
        c=p["c"],
        naverage=naverage,
        trace_H=trace_H,
        zero_mask=zero_mask,
        antinu=False,
    )

    I = torch.eye(
        3,
        dtype=U.dtype,
        device=U.device,
    )

    print("L:", L)
    print("zero_mask:", zero_mask)
    print("U:")
    print(U)

    assert_close(
        U,
        I,
        name="Zero-length perturbative evolutor is identity",
        atol=1.0e-12,
        rtol=1.0e-10,
    )


def test_batched_perturbative_segment():

    section("TEST: batched perturbative_segment_evolutor")

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

    naverage, _, _ = average_polynomial_density(
        x1=x1,
        x2=x2,
        a=a,
        b=b,
        c=c,
    )

    trace_H = hamiltonian_trace_from_ki_and_V(
        ki,
        V,
    )

    U = perturbative_segment_evolutor(
        H=H,
        L=L,
        x1=x1,
        x2=x2,
        a=a,
        b=b,
        c=c,
        naverage=naverage,
        trace_H=trace_H,
        zero_mask=zero_mask,
        antinu=False,
    )

    print("H shape:", H.shape)
    print("L shape:", L.shape)
    print("U shape:", U.shape)
    print("U:")
    print(U)

    assert_true(U.shape == (3, 3, 3), "Batched U has shape (batch, 3, 3)")
    assert_true(torch.isfinite(U).all().item(), "Batched U contains only finite values")


def test_batched_zero_length_perturbative_segment():

    section("TEST: batched zero-length perturbative segment")

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

    naverage, _, _ = average_polynomial_density(
        x1=x1,
        x2=x2,
        a=a,
        b=b,
        c=c,
    )

    trace_H = hamiltonian_trace_from_ki_and_V(
        ki,
        V,
    )

    U = perturbative_segment_evolutor(
        H=H,
        L=L,
        x1=x1,
        x2=x2,
        a=a,
        b=b,
        c=c,
        naverage=naverage,
        trace_H=trace_H,
        zero_mask=zero_mask,
        antinu=False,
    )

    I = torch.eye(
        3,
        dtype=U.dtype,
        device=U.device,
    )

    print("L:", L)
    print("zero_mask:", zero_mask)
    print("U:")
    print(U)

    assert_close(
        U[0],
        I,
        name="Batch element 0 zero-length gives identity",
        atol=1.0e-12,
        rtol=1.0e-10,
    )

    assert_close(
        U[2],
        I,
        name="Batch element 2 zero-length gives identity",
        atol=1.0e-12,
        rtol=1.0e-10,
    )

    assert_true(
        torch.isfinite(U[1]).all().item(),
        "Batch element 1 is finite",
    )


# ============================================================
# Main
# ============================================================

# ============================================================
# Runner
# ============================================================

def run_test6_perturbation_tests(verbose_traceback=False):
    tests = [
        test_has_density_perturbation,
        test_polynomial_density_delta_constant,
        test_first_order_integral_Iab_shape_and_finiteness,
        test_first_order_integral_zero_diagonal,
        test_first_order_correction_from_projectors_zero_input,
        test_first_order_density_correction_shape_and_finiteness,
        test_first_order_density_correction_zero_for_constant_density,
        test_perturbative_segment_equals_u0_for_constant_density,
        test_perturbative_segment_nonconstant_density_finite,
        test_zero_length_perturbative_segment_identity,
        test_batched_perturbative_segment,
        test_batched_zero_length_perturbative_segment,
    ]
    return run_test_suite(
        tests,
        suite_name="core TEST6 PERTURBATION tests",
        verbose_traceback=verbose_traceback,
    )


if __name__ == "__main__":
    run_test6_perturbation_tests(verbose_traceback=True)

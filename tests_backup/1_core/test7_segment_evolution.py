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
Verbose tests for tpeanuts.core.segment_evolution.

Run with:

    pytest tests/core/test_segment_evolution.py -v -s

or directly:

    python tests/core/test_segment_evolution.py
"""



from __future__ import annotations

import torch


from tpeanuts.core.hamiltonian import (
    average_polynomial_density,
    reduced_hamiltonian_from_polynomial_density,
)

from tpeanuts.core.spectral import (
    hamiltonian_trace_from_ki_and_V,
)

from tpeanuts.core.evolution import (
    constant_density_evolutor,
)

from tpeanuts.core.perturbation import (
    perturbative_segment_evolutor as perturbative_segment_evolutor_core,
)

from tpeanuts.core.segment_evolution import (
    perturbative_segment_evolutor,
    constant_density_segment_evolutor,
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


def reference_segment_objects(
    *,
    b_value: float | None = None,
    c_value: float | None = None,
    zero_length: bool = False,
    antinu: bool = False,
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
        antinu=antinu,
    )

    naverage, _, _ = average_polynomial_density(
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

    return p, pmns, H, ki, V, L, zero_mask, naverage, trace_H


# ============================================================
# tests
# ============================================================

def test_segment_perturbative_matches_core():

    section("TEST: high-level perturbative_segment_evolutor matches core")

    p, pmns, H, ki, V, L, zero_mask, naverage, trace_H = reference_segment_objects()

    U_high = perturbative_segment_evolutor(
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
        debug=True,
    )

    U_ref = perturbative_segment_evolutor_core(
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

    print("U_high:")
    print(U_high)
    print("U_ref:")
    print(U_ref)

    assert_true(U_high.shape == (3, 3), "U_high has shape (3, 3)")
    assert_true(torch.isfinite(U_high).all().item(), "U_high contains only finite values")

    assert_close(
        U_high,
        U_ref,
        name="High-level segment evolutor matches perturbation core",
        atol=1.0e-10,
        rtol=1.0e-8,
    )


def test_segment_constant_density_matches_evolution_core():

    section("TEST: constant_density_segment_evolutor matches evolution core")

    p, pmns, H, ki, V, L, zero_mask, naverage, trace_H = reference_segment_objects()

    U_high = constant_density_segment_evolutor(
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
        debug=True,
    )

    U_ref = constant_density_evolutor(
        H=H,
        L=L,
        trace_H=trace_H,
        zero_mask=zero_mask,
    )

    print("U_high:")
    print(U_high)
    print("U_ref:")
    print(U_ref)

    assert_true(U_high.shape == (3, 3), "U_high has shape (3, 3)")
    assert_true(torch.isfinite(U_high).all().item(), "U_high contains only finite values")

    assert_close(
        U_high,
        U_ref,
        name="High-level constant-density segment evolutor matches core",
        atol=1.0e-10,
        rtol=1.0e-8,
    )


def test_segment_constant_density_case_perturbative_equals_constant():

    section("TEST: perturbative segment equals constant segment when b=c=0")

    p, pmns, *_ = reference_segment_objects(
        b_value=0.0,
        c_value=0.0,
    )

    U_pert = perturbative_segment_evolutor(
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

    U_const = constant_density_segment_evolutor(
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

    print("U_pert:")
    print(U_pert)
    print("U_const:")
    print(U_const)

    assert_close(
        U_pert,
        U_const,
        name="Perturbative high-level equals constant-density high-level for b=c=0",
        atol=1.0e-10,
        rtol=1.0e-8,
    )


def test_segment_zero_length_identity():

    section("TEST: zero-length high-level segment gives identity")

    p, pmns, *_ = reference_segment_objects(
        zero_length=True,
    )

    U_pert = perturbative_segment_evolutor(
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

    U_const = constant_density_segment_evolutor(
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

    I = torch.eye(
        3,
        dtype=U_pert.dtype,
        device=U_pert.device,
    )

    print("U_pert zero length:")
    print(U_pert)

    print("U_const zero length:")
    print(U_const)

    assert_close(
        U_pert,
        I,
        name="Perturbative high-level zero-length segment is identity",
        atol=1.0e-12,
        rtol=1.0e-10,
    )

    assert_close(
        U_const,
        I,
        name="Constant high-level zero-length segment is identity",
        atol=1.0e-12,
        rtol=1.0e-10,
    )


def test_segment_unitarity_constant_density():

    section("TEST: high-level constant-density segment unitarity")

    p, pmns, *_ = reference_segment_objects()

    U = constant_density_segment_evolutor(
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

    I = torch.eye(
        3,
        dtype=U.dtype,
        device=U.device,
    )

    left = U.conj().transpose(-1, -2) @ U
    right = U @ U.conj().transpose(-1, -2)

    print("U†U:")
    print(left)
    print("UU†:")
    print(right)

    assert_close(
        left,
        I,
        name="Constant-density segment U†U = I",
        atol=1.0e-7,
        rtol=1.0e-6,
    )

    assert_close(
        right,
        I,
        name="Constant-density segment UU† = I",
        atol=1.0e-7,
        rtol=1.0e-6,
    )


def test_segment_perturbative_near_unitarity():

    section("TEST: perturbative high-level segment near-unitarity")

    p, pmns, *_ = reference_segment_objects()

    U = perturbative_segment_evolutor(
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

    I = torch.eye(
        3,
        dtype=U.dtype,
        device=U.device,
    )

    left = U.conj().transpose(-1, -2) @ U
    err = torch.max(torch.abs(left - I)).item()

    print("U†U:")
    print(left)
    print(f"max|U†U-I| = {err:.6e}")

    assert_true(torch.isfinite(U).all().item(), "Perturbative segment U is finite")
    assert_true(err < 1.0e-3, "Perturbative segment is approximately unitary")


def test_segment_antineutrino_finite_and_different():

    section("TEST: antineutrino high-level segment")

    p, pmns, *_ = reference_segment_objects()

    U_nu = perturbative_segment_evolutor(
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

    U_anti = perturbative_segment_evolutor(
        DeltamSq21=p["DeltamSq21"],
        DeltamSq3l=p["DeltamSq3l_NO"],
        pmns=pmns,
        E_MeV=p["E_MeV"],
        x1=p["x1"],
        x2=p["x2"],
        a=p["a"],
        b=p["b"],
        c=p["c"],
        antinu=True,
    )

    diff = torch.max(torch.abs(U_nu - U_anti)).item()

    print("U_nu:")
    print(U_nu)
    print("U_anti:")
    print(U_anti)
    print(f"max|U_nu-U_anti| = {diff:.6e}")

    assert_true(torch.isfinite(U_nu).all().item(), "Neutrino segment U is finite")
    assert_true(torch.isfinite(U_anti).all().item(), "Antineutrino segment U is finite")
    assert_true(diff > 0.0, "Antineutrino segment differs from neutrino segment")


def test_segment_batch_shapes_and_finiteness():

    section("TEST: batched high-level segment shapes and finiteness")

    pmns = build_pmns()

    E = torch.tensor([500.0, 1000.0, 5000.0], dtype=torch.float64)
    x1 = torch.tensor([0.10, 0.20, 0.30], dtype=torch.float64)
    x2 = torch.tensor([0.40, 0.60, 0.90], dtype=torch.float64)

    a = torch.tensor([1.00, 1.10, 1.20], dtype=torch.float64)
    b = torch.tensor([0.10, 0.20, 0.30], dtype=torch.float64)
    c = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64)

    U_pert = perturbative_segment_evolutor(
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

    U_const = constant_density_segment_evolutor(
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

    print("U_pert shape:", U_pert.shape)
    print("U_const shape:", U_const.shape)
    print("U_pert:")
    print(U_pert)

    assert_true(U_pert.shape == (3, 3, 3), "Batched perturbative U has shape (batch, 3, 3)")
    assert_true(U_const.shape == (3, 3, 3), "Batched constant U has shape (batch, 3, 3)")
    assert_true(torch.isfinite(U_pert).all().item(), "Batched perturbative U contains only finite values")
    assert_true(torch.isfinite(U_const).all().item(), "Batched constant U contains only finite values")


def test_segment_batch_zero_length_identity():

    section("TEST: batched high-level segment zero-length identity")

    pmns = build_pmns()

    E = torch.tensor([1000.0, 1000.0, 1000.0], dtype=torch.float64)
    x1 = torch.tensor([0.20, 0.20, 0.20], dtype=torch.float64)
    x2 = torch.tensor([0.20, 0.70, 0.20], dtype=torch.float64)

    a = torch.tensor([1.10, 1.10, 1.10], dtype=torch.float64)
    b = torch.tensor([0.20, 0.20, 0.20], dtype=torch.float64)
    c = torch.tensor([0.05, 0.05, 0.05], dtype=torch.float64)

    U = perturbative_segment_evolutor(
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

    I = torch.eye(
        3,
        dtype=U.dtype,
        device=U.device,
    )

    print("U:")
    print(U)

    assert_close(
        U[0],
        I,
        name="Batch zero-length element 0 is identity",
        atol=1.0e-12,
        rtol=1.0e-10,
    )

    assert_close(
        U[2],
        I,
        name="Batch zero-length element 2 is identity",
        atol=1.0e-12,
        rtol=1.0e-10,
    )

    assert_true(
        torch.isfinite(U[1]).all().item(),
        "Batch non-zero element is finite",
    )


# ============================================================
# Main
# ============================================================

# ============================================================
# Runner
# ============================================================

def run_test7_segment_evolution_tests(verbose_traceback=False):
    tests = [
        test_segment_perturbative_matches_core,
        test_segment_constant_density_matches_evolution_core,
        test_segment_constant_density_case_perturbative_equals_constant,
        test_segment_zero_length_identity,
        test_segment_unitarity_constant_density,
        test_segment_perturbative_near_unitarity,
        test_segment_antineutrino_finite_and_different,
        test_segment_batch_shapes_and_finiteness,
        test_segment_batch_zero_length_identity,
    ]
    return run_test_suite(
        tests,
        suite_name="core TEST7 SEGMENT EVOLUTION tests",
        verbose_traceback=verbose_traceback,
    )


if __name__ == "__main__":
    run_test7_segment_evolution_tests(verbose_traceback=True)

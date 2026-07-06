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
Verbose tests for tpeanuts.core.integrals.

Run with:

    pytest tests/core/test_integrals.py -v -s

or directly:

    python tests/core/test_integrals.py
"""



from __future__ import annotations

import torch

from tpeanuts.core.hamiltonian import (
    reduced_hamiltonian_from_polynomial_density,
)
from tpeanuts.core.spectral import (
    traceless_hamiltonian,
    traceless_invariant_c1,
    traceless_eigenvalues,
)
from tpeanuts.core.integration import (
    c0,
    c1,
    lambdas_cardano,
    lambdas_eigvalsh,
    Iab,
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
    p = default_inputs()
    pmns = build_pmns()

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

    return H, ki, V, L, zero_mask, p, pmns


def numerical_Iab_reference(
    la: torch.Tensor,
    lb: torch.Tensor,
    atilde: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    *,
    n_steps: int = 20001,
) -> torch.Tensor:
    dtype = torch.float64
    cdtype = torch.complex128

    xs = torch.linspace(
        x1.item(),
        x2.item(),
        n_steps,
        dtype=dtype,
        device=la.device,
    ).to(dtype=cdtype)

    la = la.to(dtype=cdtype)
    lb = lb.to(dtype=cdtype)

    density_pert = (
        atilde.to(dtype=cdtype)
        + b.to(dtype=cdtype) * xs**2
        + c.to(dtype=cdtype) * xs**4
    )

    integrand = (
        torch.exp(-1j * la * (x2.to(dtype=cdtype) - xs))
        * density_pert
        * torch.exp(-1j * lb * (xs - x1.to(dtype=cdtype)))
    )

    return torch.trapz(integrand, xs)


# ============================================================
# tests
# ============================================================


def test_c1_matches_invariant():

    section("TEST: c1 vs invariant c1")

    H, ki, V, L, zero_mask, p, pmns = build_segment()

    T, _ = traceless_hamiltonian(H)
    T = 0.5 * (T + T.conj().transpose(-1, -2))

    c1_inv = traceless_invariant_c1(T).real

    naverage = (
        p["a"] * L
        + p["b"] * (p["x2"]**3 - p["x1"]**3) / 3.0
        + p["c"] * (p["x2"]**5 - p["x1"]**5) / 5.0
    ) / L

    c1_formula = c1(
        ki=ki,
        th12=pmns.theta12,
        th13=pmns.theta13,
        n_e=naverage,
        antinu=False,
    )

    print(f"c1 invariant = {c1_inv}")
    print(f"c1 formula   = {c1_formula}")

    assert_close(
        c1_formula,
        c1_inv,
        name="c1 formula matches -Tr(T^2)/2",
        atol=1.0e-8,
        rtol=1.0e-6,
    )


def test_c0_characteristic_polynomial():

    section("TEST: c0 in characteristic polynomial")

    H, ki, V, L, zero_mask, p, pmns = build_segment()

    T, _ = traceless_hamiltonian(H)
    T = 0.5 * (T + T.conj().transpose(-1, -2))

    lam = traceless_eigenvalues(T).real

    naverage = (
        p["a"] * L
        + p["b"] * (p["x2"]**3 - p["x1"]**3) / 3.0
        + p["c"] * (p["x2"]**5 - p["x1"]**5) / 5.0
    ) / L

    c0_value = c0(
        ki=ki,
        th12=pmns.theta12,
        th13=pmns.theta13,
        n_e=naverage,
        antinu=False,
    )

    c1_value = c1(
        ki=ki,
        th12=pmns.theta12,
        th13=pmns.theta13,
        n_e=naverage,
        antinu=False,
    )

    residual = lam**3 + c1_value * lam + c0_value

    print("lambda:")
    print(lam)
    print(f"c0 = {c0}")
    print(f"c1 = {c1}")
    print("Polynomial residual lambda^3 + c1 lambda + c0:")
    print(residual)

    assert_close(
        residual,
        torch.zeros_like(residual),
        name="Characteristic polynomial residual",
        atol=1.0e-7,
        rtol=1.0e-6,
    )


def test_lambdas_eigvalsh():

    section("TEST: lambdas_eigvalsh")

    H, *_ = build_segment()

    T, _ = traceless_hamiltonian(H)
    T = 0.5 * (T + T.conj().transpose(-1, -2))

    lam = lambdas_eigvalsh(T)
    lam_expected = torch.linalg.eigvalsh(T).real

    print("lambda eigvalsh:")
    print(lam)

    assert_close(
        lam,
        lam_expected,
        name="lambdas_eigvalsh equals torch.linalg.eigvalsh",
        atol=1.0e-12,
        rtol=1.0e-10,
    )


def test_lambdas_cardano_polynomial_residual():

    section("TEST: lambdas_cardano polynomial residual")

    H, ki, V, L, zero_mask, p, pmns = build_segment()

    naverage = (
        p["a"] * L
        + p["b"] * (p["x2"]**3 - p["x1"]**3) / 3.0
        + p["c"] * (p["x2"]**5 - p["x1"]**5) / 5.0
    ) / L

    c0_value = c0(
        ki=ki,
        th12=pmns.theta12,
        th13=pmns.theta13,
        n_e=naverage,
        antinu=False,
    )

    c1_value = c1(
        ki=ki,
        th12=pmns.theta12,
        th13=pmns.theta13,
        n_e=naverage,
        antinu=False,
    )

    lam_cardano = lambdas_cardano(c0_value, c1_value)

    residual = lam_cardano**3 + c1_value.to(dtype=lam_cardano.dtype) * lam_cardano + c0_value.to(dtype=lam_cardano.dtype)

    print("lambda Cardano:")
    print(lam_cardano)
    print("Polynomial residual:")
    print(residual)
    print("max residual:", torch.max(torch.abs(residual)).item())

    assert_true(torch.isfinite(lam_cardano).all().item(), "Cardano lambdas are finite")
    assert_close(
        residual,
        torch.zeros_like(residual),
        name="Cardano roots satisfy polynomial",
        atol=1.0e-6,
        rtol=1.0e-5,
    )


def test_Iab_zero_when_la_equals_lb():

    section("TEST: Iab returns zero for la == lb")

    la = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.complex128)
    lb = la.clone()

    atilde = torch.tensor(-0.1, dtype=torch.float64)
    b = torch.tensor(0.2, dtype=torch.float64)
    c = torch.tensor(0.05, dtype=torch.float64)
    x1 = torch.tensor(0.2, dtype=torch.float64)
    x2 = torch.tensor(0.7, dtype=torch.float64)

    Iab_value = Iab(
        la=la,
        lb=lb,
        atilde=atilde,
        b=b,
        c=c,
        x2=x2,
        x1=x1,
    )

    print("Iab:")
    print(Iab_value)

    assert_close(
        Iab_value,
        torch.zeros_like(Iab_value),
        name="Iab is zero when la == lb",
        atol=1.0e-14,
        rtol=1.0e-12,
    )


def test_Iab_polynomial_zero_average_condition():

    section("TEST: density perturbation integrates to zero")

    p = default_inputs()

    L = p["x2"] - p["x1"]
    naverage = (
        p["a"] * L
        + p["b"] * (p["x2"]**3 - p["x1"]**3) / 3.0
        + p["c"] * (p["x2"]**5 - p["x1"]**5) / 5.0
    ) / L

    atilde = p["a"] - naverage

    integral = (
        atilde * L
        + p["b"] * (p["x2"]**3 - p["x1"]**3) / 3.0
        + p["c"] * (p["x2"]**5 - p["x1"]**5) / 5.0
    )

    print(f"naverage = {naverage.item():.12e}")
    print(f"atilde   = {atilde.item():.12e}")
    print(f"Integral perturbation = {integral.item():.12e}")

    assert_close(
        integral,
        torch.tensor(0.0, dtype=torch.float64),
        name="Integral of perturbation density is zero",
        atol=1.0e-14,
        rtol=1.0e-12,
    )


def test_Iab_against_numerical_quadrature_full_branch():

    section("TEST: Iab against numerical quadrature, full branch")

    p = default_inputs()

    L = p["x2"] - p["x1"]
    naverage = (
        p["a"] * L
        + p["b"] * (p["x2"]**3 - p["x1"]**3) / 3.0
        + p["c"] * (p["x2"]**5 - p["x1"]**5) / 5.0
    ) / L

    atilde = p["a"] - naverage

    la = torch.tensor([[2.7]], dtype=torch.complex128)
    lb = torch.tensor([[0.4]], dtype=torch.complex128)

    I_analytic = Iab(
        la=la,
        lb=lb,
        atilde=atilde,
        b=p["b"],
        c=p["c"],
        x2=p["x2"],
        x1=p["x1"],
        small_ratio=1.0e-8,
    )

    I_numeric = numerical_Iab_reference(
        la=la.squeeze(),
        lb=lb.squeeze(),
        atilde=atilde,
        b=p["b"],
        c=p["c"],
        x1=p["x1"],
        x2=p["x2"],
        n_steps=30001,
    ).reshape(1, 1)

    print("I analytic:")
    print(I_analytic)
    print("I numeric:")
    print(I_numeric)

    assert_close(
        I_analytic,
        I_numeric,
        name="Iab full branch matches numerical quadrature",
        atol=5.0e-7,
        rtol=5.0e-6,
    )


def test_Iab_against_numerical_quadrature_taylor_branch():

    section("TEST: Iab against numerical quadrature, Taylor branch")

    p = default_inputs()

    L = p["x2"] - p["x1"]
    naverage = (
        p["a"] * L
        + p["b"] * (p["x2"]**3 - p["x1"]**3) / 3.0
        + p["c"] * (p["x2"]**5 - p["x1"]**5) / 5.0
    ) / L

    atilde = p["a"] - naverage

    la = torch.tensor([[1.0001]], dtype=torch.complex128)
    lb = torch.tensor([[1.0000]], dtype=torch.complex128)

    I_analytic = Iab(
        la=la,
        lb=lb,
        atilde=atilde,
        b=p["b"],
        c=p["c"],
        x2=p["x2"],
        x1=p["x1"],
        small_ratio=1.0e-2,
    )

    I_numeric = numerical_Iab_reference(
        la=la.squeeze(),
        lb=lb.squeeze(),
        atilde=atilde,
        b=p["b"],
        c=p["c"],
        x1=p["x1"],
        x2=p["x2"],
        n_steps=30001,
    ).reshape(1, 1)

    print("I analytic Taylor:")
    print(I_analytic)
    print("I numeric:")
    print(I_numeric)

    assert_close(
        I_analytic,
        I_numeric,
        name="Iab Taylor branch matches numerical quadrature",
        atol=5.0e-7,
        rtol=5.0e-6,
    )


def test_Iab_batched_shape_and_finiteness():

    section("TEST: Iab batched shape and finiteness")

    la = torch.tensor(
        [
            [[1.0, 1.2, 1.4], [1.6, 1.8, 2.0], [2.2, 2.4, 2.6]],
            [[0.5, 0.7, 0.9], [1.1, 1.3, 1.5], [1.7, 1.9, 2.1]],
        ],
        dtype=torch.complex128,
    )

    lb = torch.tensor(
        [
            [[0.3, 0.4, 0.5], [0.6, 0.7, 0.8], [0.9, 1.0, 1.1]],
            [[0.2, 0.35, 0.55], [0.75, 0.95, 1.15], [1.35, 1.55, 1.75]],
        ],
        dtype=torch.complex128,
    )

    atilde = torch.tensor([-0.10, -0.12], dtype=torch.float64)
    b = torch.tensor([0.20, 0.22], dtype=torch.float64)
    c = torch.tensor([0.05, 0.06], dtype=torch.float64)
    x1 = torch.tensor([0.20, 0.30], dtype=torch.float64)
    x2 = torch.tensor([0.70, 0.90], dtype=torch.float64)

    Iab_value = Iab(
        la=la,
        lb=lb,
        atilde=atilde,
        b=b,
        c=c,
        x2=x2,
        x1=x1,
    )

    print("Iab shape:", Iab_value.shape)
    print("Iab:")
    print(Iab_value)

    assert_true(Iab_value.shape == (2, 3, 3), "Iab batched shape is (batch, 3, 3)")
    assert_true(torch.isfinite(Iab_value).all().item(), "Iab batched values are finite")


# ============================================================
# Runner
# ============================================================

def run_test5_integration_tests(verbose_traceback=False):
    tests = [
        test_c1_matches_invariant,
        test_c0_characteristic_polynomial,
        test_lambdas_eigvalsh,
        test_lambdas_cardano_polynomial_residual,
        test_Iab_zero_when_la_equals_lb,
        test_Iab_polynomial_zero_average_condition,
        test_Iab_against_numerical_quadrature_full_branch,
        test_Iab_against_numerical_quadrature_taylor_branch,
        test_Iab_batched_shape_and_finiteness,
    ]
    return run_test_suite(
        tests,
        suite_name="core TEST5 INTEGRATION tests",
        verbose_traceback=verbose_traceback,
    )


if __name__ == "__main__":
    run_test5_integration_tests(verbose_traceback=True)

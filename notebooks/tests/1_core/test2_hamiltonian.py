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
Verbose tests for peanuts_torch.core.hamiltonian.

Run with:

    pytest tests/core/test_hamiltonian.py -v -s

or directly:

    python tests/core/test_hamiltonian.py
"""



from __future__ import annotations

import torch

from tpeanuts.core.hamiltonian import (
    kinetic_mass_vector,
    average_polynomial_density,
    matter_potential_from_polynomial_average,
    reduced_mixing_matrix,
    kinetic_hamiltonian_reduced,
    matter_hamiltonian_reduced,
    reduced_hamiltonian,
    reduced_hamiltonian_from_polynomial_density,
)

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

def test_kinetic_mass_vector_normal_ordering():

    section("TEST: kinetic_mass_vector normal ordering")

    p = default_inputs()

    ki = kinetic_mass_vector(
        DeltamSq21=p["DeltamSq21"],
        DeltamSq3l=p["DeltamSq3l_NO"],
        E_MeV=p["E_MeV"],
    )

    print("ki normal ordering:")
    print(ki)

    assert_true(ki.shape == (3,), "ki has shape (3,)")
    assert_true(torch.isfinite(ki).all().item(), "ki contains only finite values")

    assert_true(ki[0].abs().item() < 1.0e-14, "ki[0] is zero for normal ordering")
    assert_true(ki[1].item() > 0.0, "ki[1] is positive")
    assert_true(ki[2].item() > 0.0, "ki[2] is positive")


def test_kinetic_mass_vector_inverted_ordering():

    section("TEST: kinetic_mass_vector inverted ordering")

    p = default_inputs()

    ki = kinetic_mass_vector(
        DeltamSq21=p["DeltamSq21"],
        DeltamSq3l=p["DeltamSq3l_IO"],
        E_MeV=p["E_MeV"],
    )

    print("ki inverted ordering:")
    print(ki)

    assert_true(ki.shape == (3,), "ki has shape (3,)")
    assert_true(torch.isfinite(ki).all().item(), "ki contains only finite values")

    assert_true(ki[0].item() < 0.0, "ki[0] is negative for inverted ordering")
    assert_true(ki[1].abs().item() < 1.0e-14, "ki[1] is zero for inverted ordering")
    assert_true(ki[2].item() < 0.0, "ki[2] is negative for inverted ordering")


def test_average_polynomial_density():

    section("TEST: average_polynomial_density")

    p = default_inputs()

    naverage, L, zero_mask = average_polynomial_density(
        x1=p["x1"],
        x2=p["x2"],
        a=p["a"],
        b=p["b"],
        c=p["c"],
    )

    L_expected = p["x2"] - p["x1"]

    naverage_expected = (
        p["a"] * L_expected
        + p["b"] * (p["x2"]**3 - p["x1"]**3) / 3.0
        + p["c"] * (p["x2"]**5 - p["x1"]**5) / 5.0
    ) / L_expected

    print(f"L          = {L.item():.12e}")
    print(f"naverage   = {naverage.item():.12e}")
    print(f"zero_mask  = {zero_mask.item()}")

    assert_close(L, L_expected, name="Segment length L")
    assert_close(naverage, naverage_expected, name="Average polynomial density")
    assert_true(not bool(zero_mask.item()), "zero_mask is False for non-zero segment")


def test_average_polynomial_density_zero_length():

    section("TEST: average_polynomial_density zero-length segment")

    p = default_inputs()

    naverage, L, zero_mask = average_polynomial_density(
        x1=p["x1"],
        x2=p["x1"],
        a=p["a"],
        b=p["b"],
        c=p["c"],
    )

    print(f"L          = {L.item():.12e}")
    print(f"naverage   = {naverage.item():.12e}")
    print(f"zero_mask  = {zero_mask.item()}")

    assert_close(L, torch.tensor(0.0, dtype=torch.float64), name="Zero segment length")
    assert_close(naverage, torch.tensor(0.0, dtype=torch.float64), name="Average density is set to zero")
    assert_true(bool(zero_mask.item()), "zero_mask is True for zero-length segment")


def test_matter_potential_from_polynomial_average():

    section("TEST: matter_potential_from_polynomial_average")

    p = default_inputs()

    V, naverage, L, zero_mask = matter_potential_from_polynomial_average(
        x1=p["x1"],
        x2=p["x2"],
        a=p["a"],
        b=p["b"],
        c=p["c"],
        antinu=False,
    )

    V_anti, *_ = matter_potential_from_polynomial_average(
        x1=p["x1"],
        x2=p["x2"],
        a=p["a"],
        b=p["b"],
        c=p["c"],
        antinu=True,
    )

    print(f"naverage   = {naverage.item():.12e}")
    print(f"L          = {L.item():.12e}")
    print(f"V neutrino = {V.item():.12e}")
    print(f"V antinu   = {V_anti.item():.12e}")
    print(f"zero_mask  = {zero_mask.item()}")

    assert_true(torch.isfinite(V).all().item(), "Matter potential is finite")
    assert_close(V_anti, -V, name="Antineutrino potential sign flip", atol=1.0e-12, rtol=1.0e-10)


def test_reduced_mixing_matrix():

    section("TEST: reduced_mixing_matrix")

    pmns = build_pmns()

    Ured = reduced_mixing_matrix(pmns)
    Ured_expected = pmns.reduced()

    Ured_anti = reduced_mixing_matrix(pmns, antinu=True)
    Ured_anti_expected = pmns.reduced().conj()

    print("Ured:")
    print(Ured)

    assert_close(Ured, Ured_expected, name="Reduced mixing matrix extraction")
    assert_close(Ured_anti, Ured_anti_expected, name="Reduced mixing antineutrino conjugation")


def test_kinetic_hamiltonian_reduced():

    section("TEST: kinetic_hamiltonian_reduced")

    p = default_inputs()
    pmns = build_pmns()

    ki = kinetic_mass_vector(
        DeltamSq21=p["DeltamSq21"],
        DeltamSq3l=p["DeltamSq3l_NO"],
        E_MeV=p["E_MeV"],
    )

    Ured = reduced_mixing_matrix(pmns)
    Hkin = kinetic_hamiltonian_reduced(
        ki=ki,
        Ured=Ured,
    )
    ki_expected = ki.to(device=Ured.device, dtype=Ured.dtype)
    Hkin_expected = Ured @ torch.diag(ki_expected) @ Ured.transpose(-1, -2)

    print("Hkin:")
    print(Hkin)

    assert_true(Hkin.shape == (3, 3), "Hkin has shape (3, 3)")
    assert_true(torch.isfinite(Hkin).all().item(), "Hkin contains only finite values")
    assert_close(Hkin, Hkin_expected, name="Hkin = Ured diag(ki) Ured^T")


def test_matter_hamiltonian_reduced():

    section("TEST: matter_hamiltonian_reduced")

    V = torch.tensor(1.2345, dtype=torch.float64)

    Hmat = matter_hamiltonian_reduced(V)

    Hmat_expected = torch.tensor(
        [
            [1.2345, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.complex128,
    )

    print("Hmat:")
    print(Hmat)

    assert_true(Hmat.shape == (3, 3), "Hmat has shape (3, 3)")
    assert_close(Hmat, Hmat_expected, name="Hmat = diag(V, 0, 0)")


def test_reduced_hamiltonian_consistency():

    section("TEST: reduced_hamiltonian consistency")

    p = default_inputs()
    pmns = build_pmns()

    V = torch.tensor(0.25, dtype=torch.float64)

    H = reduced_hamiltonian(
        DeltamSq21=p["DeltamSq21"],
        DeltamSq3l=p["DeltamSq3l_NO"],
        pmns=pmns,
        E_MeV=p["E_MeV"],
        V=V,
        antinu=False,
    )

    ki = kinetic_mass_vector(
        DeltamSq21=p["DeltamSq21"],
        DeltamSq3l=p["DeltamSq3l_NO"],
        E_MeV=p["E_MeV"],
    )

    Ured = reduced_mixing_matrix(pmns)

    H_expected = (
        kinetic_hamiltonian_reduced(ki, Ured)
        + matter_hamiltonian_reduced(V)
    )

    print("H:")
    print(H)

    assert_true(H.shape == (3, 3), "H has shape (3, 3)")
    assert_true(torch.isfinite(H).all().item(), "H contains only finite values")
    assert_close(H, H_expected, name="Reduced Hamiltonian consistency")


def test_reduced_hamiltonian_from_polynomial_density_consistency():

    section("TEST: reduced_hamiltonian_from_polynomial_density consistency")

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

    V2, naverage, L2, zero_mask2 = matter_potential_from_polynomial_average(
        x1=p["x1"],
        x2=p["x2"],
        a=p["a"],
        b=p["b"],
        c=p["c"],
        antinu=False,
    )

    H_expected = reduced_hamiltonian(
        DeltamSq21=p["DeltamSq21"],
        DeltamSq3l=p["DeltamSq3l_NO"],
        pmns=pmns,
        E_MeV=p["E_MeV"],
        V=V2,
        antinu=False,
    )

    print(f"L          = {L.item():.12e}")
    print(f"V          = {V.item():.12e}")
    print(f"naverage   = {naverage.item():.12e}")
    print(f"zero_mask  = {zero_mask.item()}")
    print("H:")
    print(H)

    assert_close(V, V2, name="Polynomial-density V consistency")
    assert_close(L, L2, name="Polynomial-density L consistency")
    assert_true(bool(zero_mask.item()) == bool(zero_mask2.item()), "zero_mask consistency")
    assert_close(H, H_expected, name="Hamiltonian from polynomial density consistency")


def test_batch_reduced_hamiltonian():

    section("TEST: batched reduced_hamiltonian_from_polynomial_density")

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

    print("H batch shape:", H.shape)
    print("ki batch shape:", ki.shape)
    print("V batch shape:", V.shape)
    print("L batch shape:", L.shape)
    print("zero_mask:", zero_mask)

    assert_true(H.shape == (3, 3, 3), "Batched H has shape (batch, 3, 3)")
    assert_true(ki.shape == (3, 3), "Batched ki has shape (batch, 3)")
    assert_true(V.shape == (3,), "Batched V has shape (batch,)")
    assert_true(L.shape == (3,), "Batched L has shape (batch,)")
    assert_true(torch.isfinite(H).all().item(), "Batched H contains only finite values")


def test_zero_length_hamiltonian_from_polynomial_density():

    section("TEST: zero-length reduced_hamiltonian_from_polynomial_density")

    p = default_inputs()
    pmns = build_pmns()

    H, ki, V, L, zero_mask = reduced_hamiltonian_from_polynomial_density(
        DeltamSq21=p["DeltamSq21"],
        DeltamSq3l=p["DeltamSq3l_NO"],
        pmns=pmns,
        E_MeV=p["E_MeV"],
        x1=p["x1"],
        x2=p["x1"],
        a=p["a"],
        b=p["b"],
        c=p["c"],
        antinu=False,
    )

    print(f"L         = {L.item():.12e}")
    print(f"V         = {V.item():.12e}")
    print(f"zero_mask = {zero_mask.item()}")
    print("H:")
    print(H)

    assert_close(L, torch.tensor(0.0, dtype=torch.float64), name="Zero-length L")
    assert_close(V, torch.tensor(0.0, dtype=torch.float64), name="Zero-length V set to zero")
    assert_true(bool(zero_mask.item()), "zero_mask is True")

    assert_true(torch.isfinite(H).all().item(), "H remains finite for zero-length segment")


# ============================================================
# Main
# ============================================================

# ============================================================
# Runner
# ============================================================

def run_test2_hamiltonian_tests(verbose_traceback=False):
    tests = [
        test_kinetic_mass_vector_normal_ordering,
        test_kinetic_mass_vector_inverted_ordering,
        test_average_polynomial_density,
        test_average_polynomial_density_zero_length,
        test_matter_potential_from_polynomial_average,
        test_reduced_mixing_matrix,
        test_kinetic_hamiltonian_reduced,
        test_matter_hamiltonian_reduced,
        test_reduced_hamiltonian_consistency,
        test_reduced_hamiltonian_from_polynomial_density_consistency,
        test_batch_reduced_hamiltonian,
        test_zero_length_hamiltonian_from_polynomial_density,
    ]
    return run_test_suite(
        tests,
        suite_name="core TEST2 HAMILTONIAN tests",
        verbose_traceback=verbose_traceback,
    )


if __name__ == "__main__":
    run_test2_hamiltonian_tests(verbose_traceback=True)

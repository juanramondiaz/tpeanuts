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
Verbose end-to-end core pipeline test for peanuts-torch.

Run with:

    pytest tests/core/test_core_pipeline.py -v -s

or directly:

    python tests/core/test_core_pipeline.py

This test makes explicit the full flow through the core modules:

    1. pmns.py
    2. hamiltonian.py
    3. spectral.py
    4. evolution.py
    5. integrals.py
    6. perturbation.py
    7. segment_evolution.py
    8. probabilities.py

The goal is to verify that the complete core block works consistently from
mixing parameters to final oscillated flux.
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
    reduced_hamiltonian_from_polynomial_density,
)

from tpeanuts.core.spectral import (
    hamiltonian_trace_from_ki_and_V,
    traceless_hamiltonian,
    hamiltonian_spectral_data,
)

from tpeanuts.core.evolution import (
    constant_density_evolutor,
    matrix_exp_evolutor,
)

from tpeanuts.core.integration import (
    Iab,
)

from tpeanuts.core.perturbation import (
    polynomial_density_delta_constant,
    first_order_density_correction,
    perturbative_segment_evolutor as perturbative_segment_evolutor_core,
)

from tpeanuts.core.segment_evolution import (
    perturbative_segment_evolutor,
    constant_density_segment_evolutor,
)

from tpeanuts.core.probabilities import (
    probability_matrix_from_evolutor,
    probability_columns_sum,
    check_probability_matrix,
    apply_probability_matrix_to_flux,
    transition_probability,
    survival_probability,
)


from tpeanuts.util.test_utils import (
    ATOL, RTOL, printoptions,
    banner, section, print_ok, print_fail, step,
    max_abs_error, assert_close, assert_true,
    default_inputs, build_pmns, run_test_suite
    )
printoptions()

# ============================================================
# Full scalar pipeline
# ============================================================

def test_core_pipeline_scalar():

    banner("peanuts-TORCH core PIPELINE TEST: SCALAR SEGMENT")

    p = default_inputs()

    # --------------------------------------------------------
    # STEP 1: PMNS
    # --------------------------------------------------------

    step(1, "pmns.py: build full and reduced mixing matrices")

    pmns = build_pmns()

    U_full = pmns.pmns_matrix()
    U_red = pmns.reduced()

    print("Full PMNS matrix U_PMNS:")
    print(U_full)

    print("Reduced matrix U_red = R13 @ R12:")
    print(U_red)

    I = torch.eye(3, dtype=U_full.dtype, device=U_full.device)

    assert_close(
        U_full.conj().transpose(-1, -2) @ U_full,
        I,
        name="Full PMNS unitarity",
        atol=1.0e-12,
        rtol=1.0e-10,
    )

    assert_close(
        U_red.conj().transpose(-1, -2) @ U_red,
        I,
        name="Reduced PMNS unitarity",
        atol=1.0e-12,
        rtol=1.0e-10,
    )

    # --------------------------------------------------------
    # STEP 2: Hamiltonian
    # --------------------------------------------------------

    step(2, "hamiltonian.py: build kinetic terms, matter potential and Hamiltonian")

    ki = kinetic_mass_vector(
        DeltamSq21=p["DeltamSq21"],
        DeltamSq3l=p["DeltamSq3l_NO"],
        E_MeV=p["E_MeV"],
    )

    naverage, L, zero_mask = average_polynomial_density(
        x1=p["x1"],
        x2=p["x2"],
        a=p["a"],
        b=p["b"],
        c=p["c"],
    )

    V, naverage2, L2, zero_mask2 = matter_potential_from_polynomial_average(
        x1=p["x1"],
        x2=p["x2"],
        a=p["a"],
        b=p["b"],
        c=p["c"],
        antinu=False,
    )

    Ured_from_obj = reduced_mixing_matrix(pmns)

    Hkin = kinetic_hamiltonian_reduced(
        ki=ki,
        Ured=Ured_from_obj,
    )

    Hmat = matter_hamiltonian_reduced(V)

    H_manual = Hkin + Hmat

    H, ki2, V2, L3, zero_mask3 = reduced_hamiltonian_from_polynomial_density(
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

    print("ki:")
    print(ki)

    print(f"L          = {L.item():.12e}")
    print(f"naverage   = {naverage.item():.12e}")
    print(f"V          = {V.item():.12e}")
    print(f"zero_mask  = {zero_mask.item()}")

    print("Hkin:")
    print(Hkin)

    print("Hmat:")
    print(Hmat)

    print("H = Hkin + Hmat:")
    print(H)

    assert_close(naverage, naverage2, name="Average density consistency")
    assert_close(L, L2, name="Segment length consistency")
    assert_close(ki, ki2, name="Kinetic vector consistency")
    assert_close(V, V2, name="Matter potential consistency")
    assert_close(L, L3, name="Hamiltonian wrapper length consistency")
    assert_close(H_manual, H, name="Manual Hamiltonian equals wrapper Hamiltonian")
    assert_true(torch.isfinite(H).all().item(), "Hamiltonian contains only finite values")

    # --------------------------------------------------------
    # STEP 3: Spectral decomposition
    # --------------------------------------------------------

    step(3, "spectral.py: trace split, eigenvalues and projectors")

    trace_H = hamiltonian_trace_from_ki_and_V(
        ki,
        V,
    )

    T, trace_H2 = traceless_hamiltonian(
        H,
        trace_H=trace_H,
    )

    spectral = hamiltonian_spectral_data(
        H,
        trace_H=trace_H,
    )

    lam = spectral["lam"]
    M = spectral["M"]

    trace_T = torch.diagonal(T, dim1=-2, dim2=-1).sum(dim=-1)

    print(f"Tr(H) = {trace_H}")
    print(f"Tr(T) = {trace_T}")
    print("lambda:")
    print(lam)
    print("Projectors M:")
    print(M)

    assert_close(trace_H, trace_H2, name="Trace consistency")
    assert_close(trace_T, torch.zeros_like(trace_T), name="T is traceless", atol=1.0e-9, rtol=1.0e-9)
    assert_close(M.sum(dim=-3), I, name="Sum of projectors equals identity", atol=1.0e-8, rtol=1.0e-7)

    H_spectral = sum(
        (lam[a] + trace_H / 3.0) * M[a]
        for a in range(3)
    )

    assert_close(
        H_spectral,
        H,
        name="Spectral reconstruction of H",
        atol=1.0e-7,
        rtol=1.0e-6,
    )

    # --------------------------------------------------------
    # STEP 4: Constant-density evolution
    # --------------------------------------------------------

    step(4, "evolution.py: constant-density evolutor")

    U0 = constant_density_evolutor(
        H=H,
        L=L,
        trace_H=trace_H,
        zero_mask=zero_mask,
    )

    U0_exp = matrix_exp_evolutor(
        H=H,
        L=L,
        zero_mask=zero_mask,
    )

    print("U0 spectral:")
    print(U0)

    print("U0 matrix_exp:")
    print(U0_exp)

    assert_close(
        U0,
        U0_exp,
        name="Constant-density spectral evolutor equals matrix_exp",
        atol=1.0e-7,
        rtol=1.0e-6,
    )

    assert_close(
        U0.conj().transpose(-1, -2) @ U0,
        I,
        name="Constant-density U0 is unitary",
        atol=1.0e-7,
        rtol=1.0e-6,
    )

    # --------------------------------------------------------
    # STEP 5: Integrals
    # --------------------------------------------------------

    step(5, "integrals.py: compute analytical I_ab integrals")

    atilde = polynomial_density_delta_constant(
        p["a"],
        naverage,
    )

    eig_H = lam + trace_H[..., None] / 3.0

    la = eig_H[..., :, None]
    lb = eig_H[..., None, :]

    Iab_integral = Iab(
        la=la,
        lb=lb,
        atilde=atilde,
        b=p["b"],
        c=p["c"],
        x2=p["x2"],
        x1=p["x1"],
    )

    print(f"atilde = {atilde.item():.12e}")
    print("Iab:")
    print(Iab_integral)

    assert_true(Iab_integral.shape == (3, 3), "Iab has shape (3, 3)")
    assert_true(torch.isfinite(Iab_integral).all().item(), "Iab contains only finite values")

    assert_close(
        torch.diagonal(Iab_integral),
        torch.zeros(3, dtype=Iab_integral.dtype),
        name="Iab diagonal is zero",
        atol=1.0e-12,
        rtol=1.0e-10,
    )

    # --------------------------------------------------------
    # STEP 6: Perturbative correction
    # --------------------------------------------------------

    step(6, "perturbation.py: compute first-order correction and segment evolutor")

    u1 = first_order_density_correction(
        M=M,
        lam=lam,
        trace_H=trace_H,
        x1=p["x1"],
        x2=p["x2"],
        a=p["a"],
        b=p["b"],
        c=p["c"],
        naverage=naverage,
        antinu=False,
    )

    U_core = perturbative_segment_evolutor_core(
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

    print("u1:")
    print(u1)
    print("U_core = U0 + u1:")
    print(U_core)
    print(f"max|u1| = {torch.max(torch.abs(u1)).item():.6e}")

    assert_true(u1.shape == (3, 3), "u1 has shape (3, 3)")
    assert_true(torch.isfinite(u1).all().item(), "u1 contains only finite values")
    assert_close(U_core, U0 + u1, name="Perturbative core U = U0 + u1", atol=1.0e-10, rtol=1.0e-8)

    # --------------------------------------------------------
    # STEP 7: High-level segment evolution wrapper
    # --------------------------------------------------------

    step(7, "segment_evolution.py: high-level segment evolutor")

    U_segment = perturbative_segment_evolutor(
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

    U_segment_const = constant_density_segment_evolutor(
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

    print("U_segment high-level:")
    print(U_segment)

    print("U_segment_const high-level:")
    print(U_segment_const)

    assert_close(
        U_segment,
        U_core,
        name="High-level perturbative segment equals core perturbative segment",
        atol=1.0e-10,
        rtol=1.0e-8,
    )

    assert_close(
        U_segment_const,
        U0,
        name="High-level constant segment equals core constant evolution",
        atol=1.0e-10,
        rtol=1.0e-8,
    )

    # --------------------------------------------------------
    # STEP 8: Probabilities and flux
    # --------------------------------------------------------

    step(8, "probabilities.py: probability matrix and final flux")

    P = probability_matrix_from_evolutor(U_segment)

    colsum = probability_columns_sum(P)

    flux_initial = p["flux_initial"]
    flux_final = apply_probability_matrix_to_flux(P, flux_initial)

    Pee = survival_probability(
        P,
        alpha="e",
        input_is_probability=True,
    )

    Pem = transition_probability(
        P,
        alpha="e",
        beta="mu",
        input_is_probability=True,
    )

    Pet = transition_probability(
        P,
        alpha="e",
        beta="tau",
        input_is_probability=True,
    )

    print("Probability matrix P[beta, alpha] = |U[beta, alpha]|^2:")
    print(P)

    print("Column sums sum_beta P[beta, alpha]:")
    print(colsum)

    print("Initial flux [phi_e, phi_mu, phi_tau]:")
    print(flux_initial)

    print("Final flux:")
    print(flux_final)

    print(f"P(e -> e)   = {Pee.item():.12e}")
    print(f"P(e -> mu)  = {Pem.item():.12e}")
    print(f"P(e -> tau) = {Pet.item():.12e}")
    print(f"Sum e-column = {(Pee + Pem + Pet).item():.12e}")

    assert_true(P.shape == (3, 3), "Probability matrix has shape (3, 3)")
    assert_true(torch.isfinite(P).all().item(), "Probability matrix contains only finite values")
    assert_true((P >= -1.0e-8).all().item(), "Probabilities are non-negative within tolerance")

    ok_prob = check_probability_matrix(
        P,
        atol=1.0e-3,
        rtol=1.0e-3,
        raise_error=False,
    )

    assert_true(ok_prob, "Probability matrix passes approximate conservation check")
    assert_true(flux_final.shape == (3,), "Final flux has shape (3,)")
    assert_true(torch.isfinite(flux_final).all().item(), "Final flux contains only finite values")

    banner("SCALAR core PIPELINE TEST PASSED")


# ============================================================
# Batched pipeline
# ============================================================

def test_core_pipeline_batched():

    banner("peanuts-TORCH core PIPELINE TEST: BATCHED SEGMENTS")

    p = default_inputs()

    pmns = build_pmns()

    E = torch.tensor([500.0, 1000.0, 5000.0], dtype=torch.float64)
    x1 = torch.tensor([0.10, 0.20, 0.30], dtype=torch.float64)
    x2 = torch.tensor([0.40, 0.60, 0.90], dtype=torch.float64)

    a = torch.tensor([1.00, 1.10, 1.20], dtype=torch.float64)
    b = torch.tensor([0.10, 0.20, 0.30], dtype=torch.float64)
    c = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64)

    step(1, "segment_evolution.py: build batched perturbative segment evolutors")

    U_batch = perturbative_segment_evolutor(
        DeltamSq21=p["DeltamSq21"],
        DeltamSq3l=p["DeltamSq3l_NO"],
        pmns=pmns,
        E_MeV=E,
        x1=x1,
        x2=x2,
        a=a,
        b=b,
        c=c,
        antinu=False,
    )

    print("U_batch shape:", U_batch.shape)
    print("U_batch:")
    print(U_batch)

    assert_true(U_batch.shape == (3, 3, 3), "Batched segment evolutor has shape (batch, 3, 3)")
    assert_true(torch.isfinite(U_batch).all().item(), "Batched segment evolutor is finite")

    step(2, "probabilities.py: convert batched evolutors into probabilities")

    P_batch = probability_matrix_from_evolutor(U_batch)
    colsum = probability_columns_sum(P_batch)

    print("P_batch shape:", P_batch.shape)
    print("P_batch:")
    print(P_batch)
    print("Column sums:")
    print(colsum)

    assert_true(P_batch.shape == (3, 3, 3), "Batched probability matrix has shape (batch, 3, 3)")
    assert_true(torch.isfinite(P_batch).all().item(), "Batched probability matrix is finite")

    step(3, "probabilities.py: apply batched probabilities to batched fluxes")

    flux_initial = torch.tensor(
        [
            [1.0, 2.0, 0.1],
            [1.5, 1.0, 0.2],
            [0.5, 3.0, 0.3],
        ],
        dtype=torch.float64,
    )

    flux_final = apply_probability_matrix_to_flux(
        P_batch,
        flux_initial,
    )

    print("flux_initial:")
    print(flux_initial)
    print("flux_final:")
    print(flux_final)

    assert_true(flux_final.shape == (3, 3), "Batched final flux has shape (batch, 3)")
    assert_true(torch.isfinite(flux_final).all().item(), "Batched final flux is finite")

    banner("BATCHED core PIPELINE TEST PASSED")


# ============================================================
# Main
# ============================================================

# ============================================================
# Runner
# ============================================================

def run_test9_core_pipeline_tests(verbose_traceback=False):
    tests = [
        test_core_pipeline_scalar,
        test_core_pipeline_batched,
    ]
    return run_test_suite(
        tests,
        suite_name="core TEST9 CORE PIPELINE tests",
        verbose_traceback=verbose_traceback,
    )


if __name__ == "__main__":
    run_test9_core_pipeline_tests(verbose_traceback=True)

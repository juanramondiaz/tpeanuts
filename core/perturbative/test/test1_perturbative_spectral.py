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

"""Pytest-compatible checks for perturbative spectral decomposition."""

from __future__ import annotations

import pytest
import torch

from tpeanuts.core.perturbative.spectral import (
    hamiltonian_spectral_data,
    hamiltonian_spectral_projectors_traceless,
    hamiltonian_traceless,
    hamiltonian_traceless_c0,
    hamiltonian_traceless_c1,
    hamiltonian_traceless_eigenvalues,
)
from tpeanuts.util.test_utils import assert_close


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CDTYPE = torch.complex128


def make_hermitian() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0 + 0.0j, 0.2 + 0.1j, 0.05 - 0.03j],
            [0.2 - 0.1j, 2.0 + 0.0j, 0.3 + 0.2j],
            [0.05 + 0.03j, 0.3 - 0.2j, 3.0 + 0.0j],
        ],
        device=DEVICE,
        dtype=CDTYPE,
    )


def eye3(batch_shape=()) -> torch.Tensor:
    return torch.eye(3, device=DEVICE, dtype=CDTYPE).expand(*batch_shape, 3, 3)


def test_hamiltonian_traceless_reconstructs_original_hamiltonian():
    H = make_hermitian()

    T, trace_H = hamiltonian_traceless(H)
    reconstructed = T + trace_H * eye3() / 3.0
    trace_T = torch.diagonal(T, dim1=-2, dim2=-1).sum(dim=-1)

    assert T.shape == (3, 3)
    assert trace_H.shape == ()
    assert_close(trace_T, torch.zeros((), device=DEVICE, dtype=CDTYPE), name="trace(T)=0")
    assert_close(reconstructed, H, name="H = T + tr(H) I/3")


def test_hamiltonian_traceless_uses_supplied_trace():
    H = make_hermitian()
    trace_H = torch.diagonal(H, dim1=-2, dim2=-1).sum(dim=-1)

    T_auto, trace_auto = hamiltonian_traceless(H)
    T_supplied, trace_supplied = hamiltonian_traceless(H, trace_H=trace_H)

    assert_close(trace_supplied, trace_auto, name="supplied trace")
    assert_close(T_supplied, T_auto, name="supplied trace traceless H")


def test_traceless_invariants_match_trace_definitions():
    H = make_hermitian()
    T, _ = hamiltonian_traceless(H)
    T2 = T @ T
    T3 = T2 @ T

    c0 = hamiltonian_traceless_c0(T)
    c1 = hamiltonian_traceless_c1(T, T2=T2)
    expected_c0 = -torch.diagonal(T3, dim1=-2, dim2=-1).sum(dim=-1) / 3.0
    expected_c1 = -torch.diagonal(T2, dim1=-2, dim2=-1).sum(dim=-1) / 2.0

    assert_close(c0, expected_c0, name="c0=-tr(T^3)/3")
    assert_close(c1, expected_c1, name="c1=-tr(T^2)/2")


def test_traceless_eigenvalues_match_torch_eigvalsh_and_sum_zero():
    H = make_hermitian()
    T, _ = hamiltonian_traceless(H)

    lam = hamiltonian_traceless_eigenvalues(T)
    expected = torch.linalg.eigvalsh(T)

    assert lam.shape == (3,)
    assert torch.isfinite(lam.real).all()
    assert_close(lam, expected.to(dtype=CDTYPE), name="traceless eigenvalues")
    assert_close(lam.sum(), torch.zeros((), device=DEVICE, dtype=CDTYPE), name="sum eigenvalues")


def test_traceless_eigenvalues_reject_nan_or_inf():
    T = make_hermitian()
    T[0, 0] = torch.tensor(float("nan"), device=DEVICE, dtype=CDTYPE)

    with pytest.raises(FloatingPointError, match="NaN or Inf"):
        hamiltonian_traceless_eigenvalues(T)


def test_spectral_projectors_are_complete_orthogonal_and_reconstruct_T():
    H = make_hermitian()
    T, _ = hamiltonian_traceless(H)
    M, lam, c1 = hamiltonian_spectral_projectors_traceless(T)
    identity = eye3()

    assert M.shape == (3, 3, 3)
    assert lam.shape == (3,)
    assert c1.shape == ()
    assert torch.isfinite(M.real).all()
    assert torch.isfinite(M.imag).all()
    assert_close(M.sum(dim=0), identity, name="sum projectors")
    assert_close((lam[:, None, None] * M).sum(dim=0), T, name="spectral reconstruction of T")

    for a in range(3):
        assert_close(M[a].conj().transpose(-2, -1), M[a], name=f"M{a} Hermitian")
        assert_close(M[a] @ M[a], M[a], name=f"M{a} idempotent")
        for b in range(3):
            if a != b:
                assert_close(M[a] @ M[b], torch.zeros((3, 3), device=DEVICE, dtype=CDTYPE), name=f"M{a}M{b}=0")


def test_spectral_projectors_accept_precomputed_inputs():
    H = make_hermitian()
    T, _ = hamiltonian_traceless(H)
    T2 = T @ T
    lam = hamiltonian_traceless_eigenvalues(T)
    c1 = hamiltonian_traceless_c1(T, T2=T2)

    M_auto, lam_auto, c1_auto = hamiltonian_spectral_projectors_traceless(T)
    M_pre, lam_pre, c1_pre = hamiltonian_spectral_projectors_traceless(T, lam=lam, c1=c1, T2=T2)

    assert_close(M_pre, M_auto, name="projectors with precomputed inputs")
    assert_close(lam_pre, lam_auto, name="precomputed eigenvalues")
    assert_close(c1_pre, c1_auto, name="precomputed c1")


def test_hamiltonian_spectral_data_returns_consistent_keys_and_values():
    H = make_hermitian()
    data = hamiltonian_spectral_data(H)

    assert set(data) == {"T", "trace", "lam", "c1", "M"}
    assert data["T"].shape == (3, 3)
    assert data["trace"].shape == ()
    assert data["lam"].shape == (3,)
    assert data["M"].shape == (3, 3, 3)
    assert_close(data["M"].sum(dim=0), eye3(), name="spectral data projector completeness")
    assert_close(
        data["T"] + data["trace"] * eye3() / 3.0,
        H,
        name="spectral data reconstructs H",
    )


def test_batched_spectral_data_shapes_and_reconstruction():
    H0 = make_hermitian()
    H = torch.stack([H0, 1.7 * H0 + 0.2 * eye3()], dim=0)

    data = hamiltonian_spectral_data(H)
    reconstructed_T = (data["lam"][..., :, None, None] * data["M"]).sum(dim=-3)
    reconstructed_H = data["T"] + data["trace"][..., None, None] * eye3((2,)) / 3.0

    assert data["T"].shape == (2, 3, 3)
    assert data["trace"].shape == (2,)
    assert data["lam"].shape == (2, 3)
    assert data["M"].shape == (2, 3, 3, 3)
    assert_close(data["M"].sum(dim=-3), eye3((2,)), name="batched projector completeness")
    assert_close(reconstructed_T, data["T"], name="batched spectral T reconstruction")
    assert_close(reconstructed_H, H, name="batched spectral H reconstruction")


def test_nearly_degenerate_hamiltonian_projectors_remain_finite():
    H = torch.zeros((3, 3), device=DEVICE, dtype=CDTYPE)

    M, lam, c1 = hamiltonian_spectral_projectors_traceless(H)

    assert lam.shape == (3,)
    assert c1.shape == ()
    assert M.shape == (3, 3, 3)
    assert torch.isfinite(M.real).all()
    assert torch.isfinite(M.imag).all()

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
    hamiltonian_traceless_e3,
    hamiltonian_traceless_eigenvalues,
    spectral_projector_residuals,
    _spectral_degeneracy_mask,
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

    for a in range(3):
        assert_close(M[a].conj().transpose(-2, -1), M[a], name=f"M{a} Hermitian")

    residuals = spectral_projector_residuals(M, T, lam)
    tol = 1.0e-12
    for name, value in residuals.items():
        assert float(value) < tol, f"{name} residual too large: {float(value):.3e}"


def test_spectral_projector_residuals_detect_broken_decomposition():
    H = make_hermitian()
    T, _ = hamiltonian_traceless(H)
    M, lam, _ = hamiltonian_spectral_projectors_traceless(T)

    broken = M.clone()
    broken[0] = broken[0] + 0.1 * eye3()

    residuals = spectral_projector_residuals(broken, T, lam)
    assert float(residuals["completeness"]) > 1.0e-3
    assert float(residuals["reconstruction"]) > 1.0e-3


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


# ---------------------------------------------------------------------------
# N=4 (3+1 sterile) spectral projector formula
# ---------------------------------------------------------------------------

def make_hermitian4(seed: int = 0) -> torch.Tensor:
    """Random Hermitian 4x4 matrix (not yet traceless)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(4, 4, generator=generator, dtype=torch.float64).to(dtype=CDTYPE)
    A = A + 1j * torch.randn(4, 4, generator=generator, dtype=torch.float64).to(dtype=CDTYPE)
    A = A.to(device=DEVICE)
    return A + A.conj().transpose(-1, -2)


def eye4(batch_shape=()) -> torch.Tensor:
    return torch.eye(4, device=DEVICE, dtype=CDTYPE).expand(*batch_shape, 4, 4)


def test_traceless_e3_matches_negative_c0():
    H = make_hermitian4(seed=1)
    T, _ = hamiltonian_traceless(H)

    e3 = hamiltonian_traceless_e3(T)
    c0 = hamiltonian_traceless_c0(T)

    assert_close(e3, -c0, name="e3 == -c0")


@pytest.mark.parametrize("seed", range(10))
def test_n4_spectral_projectors_satisfy_all_invariants(seed):
    H = make_hermitian4(seed=seed)
    data = hamiltonian_spectral_data(H)

    assert data["T"].shape == (4, 4)
    assert data["lam"].shape == (4,)
    assert data["M"].shape == (4, 4, 4)
    assert torch.isfinite(data["M"].real).all()
    assert torch.isfinite(data["M"].imag).all()

    for a in range(4):
        assert_close(
            data["M"][a].conj().transpose(-2, -1), data["M"][a], name=f"M{a} Hermitian",
        )

    residuals = spectral_projector_residuals(data["M"], data["T"], data["lam"])
    tol = 1.0e-9
    for name, value in residuals.items():
        assert float(value) < tol, f"seed={seed} {name} residual too large: {float(value):.3e}"


@pytest.mark.parametrize("seed", range(10))
def test_n4_spectral_projectors_match_eigh_eigenvectors(seed):
    """Cross-check against an independent ground truth (not just self-consistency).

    A sign error in e2/e3 could still pass the completeness/idempotency/
    orthogonality checks in some cases; comparing directly against
    ``torch.linalg.eigh``'s eigenvector outer products v_a v_a^dagger is an
    independent verification that the closed-form formula is *correct*, not
    merely internally consistent.
    """
    H = make_hermitian4(seed=seed)
    data = hamiltonian_spectral_data(H)

    lam_eigh, V = torch.linalg.eigh(data["T"])
    M_eigh = torch.einsum("ia,ja->aij", V, V.conj()).to(dtype=CDTYPE)

    assert_close(data["lam"], lam_eigh.to(dtype=CDTYPE), atol=1.0e-9, rtol=1.0e-9, name="lam vs eigh")
    assert_close(data["M"], M_eigh, atol=1.0e-9, rtol=1.0e-9, name="M vs eigh eigenvector outer products")


def test_n4_batched_spectral_data_shapes_and_reconstruction():
    H0 = make_hermitian4(seed=2)
    H = torch.stack([H0, 1.7 * H0 + 0.2 * eye4()], dim=0)

    data = hamiltonian_spectral_data(H)
    reconstructed_T = (data["lam"][..., :, None, None] * data["M"]).sum(dim=-3)
    reconstructed_H = data["T"] + data["trace"][..., None, None] * eye4((2,)) / 4.0

    assert data["T"].shape == (2, 4, 4)
    assert data["trace"].shape == (2,)
    assert data["lam"].shape == (2, 4)
    assert data["M"].shape == (2, 4, 4, 4)
    assert_close(data["M"].sum(dim=-3), eye4((2,)), name="batched N=4 projector completeness")
    assert_close(reconstructed_T, data["T"], name="batched N=4 spectral T reconstruction")
    assert_close(reconstructed_H, H, name="batched N=4 spectral H reconstruction")


def test_n4_block_diagonal_sm_embedding_decouples_exactly():
    """A 4x4 H that is exactly block-diagonal (3x3 SM block + decoupled 4th
    eigenvalue) must reduce to the 3x3 SM eigen-decomposition exactly on the
    active block, with the 4th eigenvector purely sterile -- independent of
    the traceless-shift bookkeeping (compares the *full*, non-traceless H).
    """
    H3 = make_hermitian()  # existing 3x3 helper
    H4 = torch.zeros((4, 4), device=DEVICE, dtype=CDTYPE)
    H4[:3, :3] = H3
    H4[3, 3] = 5.0 + 0.0j

    lam3, V3 = torch.linalg.eigh(H3)
    lam4, V4 = torch.linalg.eigh(H4)

    sterile_idx = int((lam4 - 5.0).abs().argmin())
    active_idx = [i for i in range(4) if i != sterile_idx]

    assert_close(
        lam4[active_idx].sort().values, lam3.sort().values,
        name="active eigenvalues match the 3x3 SM spectrum exactly",
    )
    v_sterile = V4[:, sterile_idx].abs()
    assert_close(
        v_sterile, torch.tensor([0.0, 0.0, 0.0, 1.0], device=DEVICE, dtype=DTYPE),
        name="4th eigenvector is purely sterile",
    )


def test_n4_unsupported_flavour_count_raises():
    H5 = torch.eye(5, device=DEVICE, dtype=CDTYPE)
    T, _ = hamiltonian_traceless(H5)

    with pytest.raises(ValueError, match="supports N in"):
        hamiltonian_spectral_projectors_traceless(T)


# ---------------------------------------------------------------------------
# Degeneracy detection and eigh fallback
# ---------------------------------------------------------------------------

def random_unitary(n: int, seed: int, device, dtype) -> torch.Tensor:
    """Random n x n unitary matrix (via QR of a random complex Gaussian matrix)."""
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    generator = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(n, n, generator=generator, dtype=real_dtype).to(dtype=dtype)
    A = A + 1j * torch.randn(n, n, generator=generator, dtype=real_dtype).to(dtype=dtype)
    A = A.to(device=device)
    Q, _ = torch.linalg.qr(A)
    return Q


def traceless_from_eigenvalues(lam: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """Build a Hermitian traceless T = U diag(lam) U^H from prescribed eigenvalues.

    ``lam`` is assumed to already sum to zero along its last dimension.
    """
    D = torch.diag_embed(lam)
    return U @ D @ U.conj().transpose(-1, -2)


@pytest.mark.parametrize(
    "real_dtype,cplx_dtype,tol",
    [
        (torch.float64, torch.complex128, 1.0e-10),
        (torch.float32, torch.complex64, 1.0e-4),
    ],
)
@pytest.mark.parametrize("N", [3, 4])
def test_exact_degeneracy_projectors_reconstruct_precisely(real_dtype, cplx_dtype, tol, N):
    """A T with an exactly repeated eigenvalue must still decompose accurately.

    Before the eigh fallback, the closed-form formula's denominator vanishes
    exactly at a repeated eigenvalue and only the ``_DENOM_EPS`` safety floor
    kept it finite -- not numerically valid as a spectral decomposition.
    """
    U = random_unitary(N, seed=100 + N, device=DEVICE, dtype=cplx_dtype)
    if N == 3:
        lam = torch.tensor([1.0, 1.0, -2.0], device=DEVICE, dtype=cplx_dtype)
    else:
        lam = torch.tensor([1.0, 1.0, -0.5, -1.5], device=DEVICE, dtype=cplx_dtype)

    T = traceless_from_eigenvalues(lam, U)
    M, lam_out, _ = hamiltonian_spectral_projectors_traceless(T)

    residuals = spectral_projector_residuals(M, T, lam_out)
    for name, value in residuals.items():
        assert float(value) < tol, f"N={N} {name} residual too large: {float(value):.3e}"


@pytest.mark.parametrize(
    "real_dtype,cplx_dtype,tol",
    [
        (torch.float64, torch.complex128, 1.0e-10),
        (torch.float32, torch.complex64, 1.0e-4),
    ],
)
@pytest.mark.parametrize("N", [3, 4])
def test_fully_degenerate_zero_T_reconstructs_precisely(real_dtype, cplx_dtype, tol, N):
    """T=0 (all eigenvalues equal) is finite even without the fallback (see
    ``test_nearly_degenerate_hamiltonian_projectors_remain_finite``), but only
    the eigh fallback makes it a numerically valid spectral decomposition.
    """
    T = torch.zeros((N, N), device=DEVICE, dtype=cplx_dtype)
    M, lam, _ = hamiltonian_spectral_projectors_traceless(T)

    residuals = spectral_projector_residuals(M, T, lam)
    for name, value in residuals.items():
        assert float(value) < tol, f"N={N} {name} residual too large: {float(value):.3e}"


@pytest.mark.parametrize("N", [3, 4])
def test_degeneracy_mask_flags_only_truly_degenerate_spectra(N):
    scale = 1.0
    eps = torch.finfo(DTYPE).eps

    if N == 3:
        well_separated = torch.tensor([1.0, -0.3, -0.7], dtype=DTYPE)
        barely_separated = torch.tensor(
            [scale, scale - 1.0e7 * eps * scale, -(2 * scale - 1.0e7 * eps * scale)], dtype=DTYPE,
        )
        nearly_degenerate = torch.tensor(
            [scale, scale - 1.0 * eps * scale, -(2 * scale - 1.0 * eps * scale)], dtype=DTYPE,
        )
        exactly_degenerate = torch.tensor([scale, scale, -2.0 * scale], dtype=DTYPE)
    else:
        well_separated = torch.tensor([1.0, -0.3, -0.5, -0.2], dtype=DTYPE)
        barely_separated = torch.tensor([
            scale, scale - 1.0e7 * eps * scale, -0.5 * scale, -(0.5 * scale - 1.0e7 * eps * scale),
        ], dtype=DTYPE)
        nearly_degenerate = torch.tensor([
            scale, scale - 1.0 * eps * scale, -0.5 * scale, -(0.5 * scale - 1.0 * eps * scale),
        ], dtype=DTYPE)
        exactly_degenerate = torch.tensor([scale, scale, -0.5 * scale, -0.5 * scale], dtype=DTYPE)

    assert not bool(_spectral_degeneracy_mask(
        well_separated, relative_eps=1.0e3, absolute_eps=1.0e3,
    ))
    assert not bool(_spectral_degeneracy_mask(
        barely_separated, relative_eps=1.0e3, absolute_eps=1.0e3,
    ))
    assert bool(_spectral_degeneracy_mask(
        nearly_degenerate, relative_eps=1.0e3, absolute_eps=1.0e3,
    ))
    assert bool(_spectral_degeneracy_mask(
        exactly_degenerate, relative_eps=1.0e3, absolute_eps=1.0e3,
    ))


@pytest.mark.parametrize("N", [3, 4])
def test_degeneracy_fallback_not_triggered_for_well_separated_spectra_is_bit_identical(N):
    """For well-separated spectra the default call must match a call that has
    the fallback forced off (relative_eps=absolute_eps=0), i.e. the fallback
    must not perturb the pre-Fase-2 closed-form output.
    """
    H = make_hermitian() if N == 3 else make_hermitian4(seed=7)
    T, _ = hamiltonian_traceless(H)

    M_default, lam_default, c1_default = hamiltonian_spectral_projectors_traceless(T)
    M_forced_off, lam_off, c1_off = hamiltonian_spectral_projectors_traceless(
        T, relative_eps=0.0, absolute_eps=0.0,
    )

    assert torch.equal(M_default, M_forced_off)
    assert torch.equal(lam_default, lam_off)
    assert torch.equal(c1_default, c1_off)


@pytest.mark.parametrize("N", [3, 4])
def test_degeneracy_fallback_matches_eigh_ground_truth_when_forced(N):
    """Independent correctness check of the eigh-fallback branch itself,
    decoupled from whether it is naturally triggered: force it on a
    well-separated spectrum (where the closed form is known-good, see
    ``test_n4_spectral_projectors_match_eigh_eigenvectors`` and the N=3
    baseline tests) and confirm it still reconstructs T correctly.
    """
    H = make_hermitian() if N == 3 else make_hermitian4(seed=11)
    T, _ = hamiltonian_traceless(H)

    M, lam, _ = hamiltonian_spectral_projectors_traceless(
        T, relative_eps=1.0e30, absolute_eps=1.0e30,
    )

    residuals = spectral_projector_residuals(M, T, lam)
    tol = 1.0e-9
    for name, value in residuals.items():
        assert float(value) < tol, f"N={N} {name} residual too large: {float(value):.3e}"


@pytest.mark.parametrize("N", [3, 4])
def test_eigenvalue_crossing_sweep_projectors_remain_well_conditioned(N):
    """Sweep a Hamiltonian parameter through a controlled eigenvalue crossing
    and confirm the spectral decomposition stays numerically valid throughout
    -- including exactly at the crossing -- with no discontinuous blow-up.
    """
    U = random_unitary(N, seed=200 + N, device=DEVICE, dtype=CDTYPE)
    t = torch.linspace(-0.05, 0.05, 21, device=DEVICE, dtype=DTYPE).to(dtype=CDTYPE)

    if N == 3:
        lam = torch.stack([1.0 + t, 1.0 - t, torch.full_like(t, -2.0)], dim=-1)
    else:
        lam = torch.stack(
            [1.0 + t, 1.0 - t, torch.full_like(t, -0.5), torch.full_like(t, -1.5)], dim=-1,
        )

    T = traceless_from_eigenvalues(lam, U)
    M, lam_out, _ = hamiltonian_spectral_projectors_traceless(T)

    residuals = spectral_projector_residuals(M, T, lam_out)
    tol = 1.0e-9
    for name, value in residuals.items():
        assert float(value.amax()) < tol, f"N={N} {name} residual too large across sweep: {float(value.amax()):.3e}"

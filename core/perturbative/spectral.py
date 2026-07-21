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
Hamiltonian utilities for the Tpeanuts perturbative evolution scheme.

This module contains the spectral tools required by the peanuts perturbative
evolution scheme. The Hamiltonian is first decomposed into a trace part and a
traceless part,

    H = T + Tr(H) / N I,

where T is a traceless NxN matrix (N=3 for the 3-flavour Standard Model, N=4
for the 3+1 sterile-neutrino extension; both are supported with dedicated
closed-form spectral-projector formulas, see
``hamiltonian_spectral_projectors_traceless``).

The peanuts evolution method uses the eigenvalues of T (obtained from
``torch.linalg.eigvalsh`` -- there is no closed-form root solver in this
module, only a closed-form *projector* construction from already-known
eigenvalues) and the associated spectral projectors M_a. These projectors
allow the constant-density evolution operator to be written as

    U0(L) = sum_a exp[-i (lambda_a + Tr(H)/N) L] M_a.

The module functions are organized as follows:

    hamiltonian_traceless(...)
        Splits H into trace and traceless components.

    hamiltonian_traceless_c0(...)
        Computes the cubic invariant c0 = -tr(T^3)/3 of the traceless
        Hamiltonian (N=3 characteristic-polynomial convention).

    hamiltonian_traceless_c1(...)
        Computes the c1 invariant of the traceless Hamiltonian (= e2, the
        second elementary symmetric polynomial of the eigenvalues; valid for
        any N).

    hamiltonian_traceless_e3(...)
        Computes e3 = tr(T^3)/3, the third elementary symmetric polynomial
        of the eigenvalues, used by the N=4 spectral projector formula. Note
        the sign: e3 = -c0 (see the two docstrings for why they are kept as
        separate functions instead of reusing one with a sign flip inline).

    hamiltonian_traceless_eigenvalues(...)
        Computes the eigenvalues of the traceless Hamiltonian.

    hamiltonian_spectral_projectors_traceless(...)
        Builds the spectral projectors M_a from T and its eigenvalues, for
        N=3 or N=4. Falls back to eigh-derived projectors (via
        ``_spectral_degeneracy_mask``) for batch entries with a nearly
        degenerate spectrum, where the closed-form formula is ill-conditioned.

    hamiltonian_spectral_data(...)
        Computes and returns all spectral quantities required by the evolution
        module.

    spectral_projector_residuals(...)
        Diagnostic residual norms (completeness, idempotency, orthogonality,
        trace, reconstruction) for a computed projector decomposition, used
        both by tests and by degeneracy-handling instrumentation.

This module receives an already-built Hamiltonian and prepares the spectral
objects needed for evolution.

"""



from __future__ import annotations

import torch

# Minimum absolute value of the spectral projector denominator (3λ² + c1).
# This is a last-resort safety net against literal division by zero; the
# primary degeneracy handling is the relative detector + eigh fallback below
# (see ``_spectral_degeneracy_mask`` and ``hamiltonian_spectral_projectors_traceless``).
_DENOM_EPS: float = 1.0e-30

# Degeneracy-detection thresholds, expressed as multipliers of
# ``torch.finfo(dtype).eps`` so they scale automatically with precision
# (float32/complex64 vs float64/complex128). ``_DEGENERACY_RELATIVE_EPS``
# bounds the minimum pairwise eigenvalue gap relative to the spectrum scale;
# ``_DEGENERACY_ABSOLUTE_EPS`` is a floor on the spectrum scale itself, for
# the pathological T≈0 case where the relative test is meaningless (0/0).
_DEGENERACY_RELATIVE_EPS: float = 1.0e3
_DEGENERACY_ABSOLUTE_EPS: float = 1.0e3


def hamiltonian_traceless(
    H: torch.Tensor,
    trace_H: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Subtract one Nth of the trace from a Hamiltonian to obtain its traceless part.

    Formula: Uses T = H - tr(H) I / N, with N = H.shape[-1] (3 or 4).

    Args:
        H: Hamiltonian tensor shaped (..., N, N) in km^-1, N in {3, 4}.
        trace_H: Trace of the Hamiltonian shaped (...) and expressed in km^-1.

    Returns:
        Tuple containing the traceless Hamiltonian shaped (..., N, N) and its
        trace with the batch shape of H.
    """
    N = H.shape[-1]
    I_N = torch.eye(N, device=H.device, dtype=H.dtype)

    if trace_H is None:
        trace_H = torch.diagonal(H, dim1=-2, dim2=-1).sum(dim=-1)

    trace_H = trace_H.to(dtype=H.dtype)

    T = H - trace_H[..., None, None] * I_N / N

    return T, trace_H

def hamiltonian_traceless_c0(T: torch.Tensor) -> torch.Tensor:
    """
    Compute the cubic invariant c0 of a traceless 3x3 Hamiltonian.

    Formula: Uses c0 = -Tr(T^3) / 3. Together with the quadratic invariant
    c1 (see ``hamiltonian_traceless_c1``), c0 enters the characteristic
    polynomial of T, ``lambda^3 + c1*lambda + c0 = 0`` (no quadratic term
    since T is traceless; verified numerically -- this is the corrected sign
    convention, the previous docstring had both signs flipped), whose three
    real roots are the eigenvalues returned by
    ``hamiltonian_traceless_eigenvalues``.

    Args:
        T: Traceless Hermitian Hamiltonian tensor shaped (..., 3, 3).

    Returns:
        Real tensor c0 with the batch shape of T.
    """

    T3 = T @ T @ T

    trT3 = torch.diagonal(
        T3,
        dim1=-2,
        dim2=-1
    ).sum(dim=-1)

    return -trT3 / 3.0

def hamiltonian_traceless_c1(T: torch.Tensor, T2: torch.Tensor | None = None) -> torch.Tensor:
    """
    Compute the quadratic invariant c1 of a traceless 3x3 Hamiltonian.

    Formula: Uses c1 = -tr(T^2) / 2.

    Args:
        T: Traceless Hermitian Hamiltonian tensor shaped (..., 3, 3).
        T2: Optional precomputed T @ T shaped (..., 3, 3). When provided the
            matrix multiplication is skipped, avoiding a redundant bmm.

    Returns:
        Real tensor c1 with the batch shape of T.
    """
    if T2 is None:
        T2 = T @ T
    trT2 = torch.diagonal(
        T2,
        dim1=-2,
        dim2=-1
    ).sum(dim=-1)

    return -trT2 / 2.0


def hamiltonian_traceless_e3(
    T: torch.Tensor,
    T3: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the third elementary symmetric invariant e3 of a traceless Hamiltonian.

    Formula: Uses e3 = tr(T^3) / 3 (valid for any N, not just N=3 -- Newton's
    identity relating power sums to elementary symmetric polynomials does not
    depend on the total number of eigenvalues). Used, together with c1 (= e2,
    see ``hamiltonian_traceless_c1``), by the N=4 spectral projector formula
    in ``hamiltonian_spectral_projectors_traceless``.

    Note the sign relative to ``hamiltonian_traceless_c0``: e3 = -c0. This
    function is kept deliberately separate (not implemented as `-c0(T)`
    inlined at call sites) to avoid a sign-confusion bug where a caller
    reuses c0 as if it already were e3.

    Args:
        T: Traceless Hermitian Hamiltonian tensor shaped (..., N, N).
        T3: Optional precomputed T @ T @ T shaped (..., N, N). When provided
            the matrix multiplications are skipped, avoiding redundant bmm.

    Returns:
        Real tensor e3 with the batch shape of T.
    """
    if T3 is None:
        T3 = T @ T @ T

    trT3 = torch.diagonal(T3, dim1=-2, dim2=-1).sum(dim=-1)

    return trT3 / 3.0


def hamiltonian_traceless_eigenvalues(
    T: torch.Tensor,
    *,
    already_symmetric: bool = False,
) -> torch.Tensor:
    """
    Compute the eigenvalues of a traceless Hermitian Hamiltonian.

    Args:
        T: Traceless Hermitian Hamiltonian tensor shaped (..., 3, 3).
        already_symmetric: When True, skip the symmetrization step.  Set this
            flag when the caller (e.g. ``hamiltonian_spectral_data``) has
            already enforced Hermitian symmetry to avoid a redundant operation.

    Returns:
        Tensor shaped (..., 3) with the real eigenvalues represented in T.dtype.
    """
    T = T.contiguous()
    if not already_symmetric:
        T = 0.5 * (T + T.conj().transpose(-1, -2))

    if not torch.isfinite(T).all():
        raise FloatingPointError("T contains NaN or Inf before eigvalsh.")

    lam = torch.linalg.eigvalsh(T).to(dtype=T.dtype)

    return lam


def _spectral_degeneracy_mask(
    lam: torch.Tensor,
    *,
    relative_eps: float,
    absolute_eps: float,
) -> torch.Tensor:
    """Flag batch entries whose eigenvalue spectrum is (nearly) degenerate.

    Uses a *relative* threshold rather than a fixed absolute one, since the
    dynamic range of lambda spans vacuum splittings (~1e-5) to large matter
    potentials. For each eigenvalue lambda_a, ``gap_a = min_{b!=a} |lambda_a -
    lambda_b|`` is compared against ``relative_eps * eps * scale``, where
    ``scale = max_b |lambda_b|`` and ``eps = torch.finfo(dtype).eps``. A batch
    entry is flagged degenerate if any eigenvalue fails this test, or if
    ``scale`` itself is below ``absolute_eps * eps`` (the T≈0 case, where the
    relative test degenerates to 0/0).

    Args:
        lam: Eigenvalues shaped (..., N) (real- or complex-dtyped; only the
            real part is used).
        relative_eps: Multiplier of ``eps`` bounding the minimum pairwise gap
            relative to the spectrum scale.
        absolute_eps: Multiplier of ``eps`` giving the floor on the spectrum
            scale itself.

    Returns:
        Boolean tensor shaped (...,), True where the closed-form projector
        formula is not numerically trustworthy for that batch entry.
    """
    lam_real = lam.real if torch.is_complex(lam) else lam
    N = lam_real.shape[-1]
    eps = torch.finfo(lam_real.dtype).eps

    diff = (lam_real[..., :, None] - lam_real[..., None, :]).abs()
    eye_mask = torch.eye(N, dtype=torch.bool, device=lam_real.device)
    diff = diff.masked_fill(eye_mask, float("inf"))
    gap = diff.amin(dim=-1)

    scale = lam_real.abs().amax(dim=-1)
    threshold = relative_eps * eps * scale

    any_pair_degenerate = (gap < threshold[..., None]).any(dim=-1)
    scale_below_floor = scale < absolute_eps * eps

    return any_pair_degenerate | scale_below_floor


def hamiltonian_spectral_projectors_traceless(
    T: torch.Tensor,
    lam: torch.Tensor | None = None,
    c1: torch.Tensor | None = None,
    T2: torch.Tensor | None = None,
    e3: torch.Tensor | None = None,
    T3: torch.Tensor | None = None,
    *,
    relative_eps: float = _DEGENERACY_RELATIVE_EPS,
    absolute_eps: float = _DEGENERACY_ABSOLUTE_EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build spectral projectors for a traceless Hamiltonian from eigenvalues and invariants.

    Both formulas are the Lagrange/Sylvester interpolation projector
    ``M_a = prod_{b != a} (T - lambda_b I) / (lambda_a - lambda_b)``,
    algebraically reduced (via Cayley-Hamilton and the tracelessness of T,
    i.e. e1 = sum_a lambda_a = 0) to an explicit polynomial in T with
    coefficients depending only on T, T^2 (and, for N=4, T^3), lambda_a, and
    the trace-derived invariants c1 (= e2 = -tr(T^2)/2) and e3 (=
    tr(T^3)/3) -- never on the individual "other" eigenvalues. Both formulas
    were verified symbolically (sympy) and numerically (against direct
    Lagrange interpolation and against eigenvector outer products from
    ``torch.linalg.eigh``) to match to floating-point precision.

    N=3 (unchanged from the original formula):
        M_a = [(lambda_a^2 + c1) I + lambda_a T + T^2] / (3 lambda_a^2 + c1)

    N=4:
        M_a = [T^3 + lambda_a T^2 + (c1 + lambda_a^2) T
               + (lambda_a^3 + c1 lambda_a - e3) I]
              / (4 lambda_a^3 + 2 c1 lambda_a - e3)

    In both cases the denominator equals p'(lambda_a), the derivative of the
    characteristic polynomial at the root -- it vanishes exactly when
    lambda_a is a repeated eigenvalue (degenerate spectrum). Near such a
    degeneracy the closed-form formula above loses precision well before
    ``_DENOM_EPS`` (a last-resort floor against literal division by zero) is
    reached, so batch entries flagged by ``_spectral_degeneracy_mask`` (a
    *relative* gap-vs-scale test, see that function) have their projectors
    replaced wholesale by ``M_a = v_a v_a^dagger`` built from a single
    ``torch.linalg.eigh(T)`` call -- exact and stable by construction,
    independent of the closed-form formula's conditioning. This eigh call is
    only made when at least one batch entry needs it. The module runs under
    ``@torch.no_grad()`` (see ``core/perturbative/evolutor.py``), so the
    well-known gradient instability of degenerate eigenvectors does not
    apply here.

    Args:
        T: Traceless Hermitian Hamiltonian tensor shaped (..., N, N), N in {3, 4}.
        lam: Hamiltonian eigenvalues shaped (..., N).
        c1: Quadratic invariant c1 (= e2) of the traceless Hamiltonian.
        T2: Optional precomputed T @ T shaped (..., N, N). When provided the
            matrix multiplication is skipped, avoiding a redundant bmm.
        e3: N=4 only. Third elementary symmetric invariant (see
            ``hamiltonian_traceless_e3``). Ignored for N=3.
        T3: N=4 only. Optional precomputed T @ T @ T shaped (..., N, N).
            Ignored for N=3.
        relative_eps: Degeneracy-detector threshold, see
            ``_spectral_degeneracy_mask``.
        absolute_eps: Degeneracy-detector scale floor, see
            ``_spectral_degeneracy_mask``.

    Returns:
        Tuple containing projectors shaped (..., N, N, N), eigenvalues shaped
        (..., N), and c1 with the batch shape of T.

    Raises:
        ValueError: If ``T.shape[-1]`` is not 3 or 4.
    """
    N = T.shape[-1]
    I_N = torch.eye(N, device=T.device, dtype=T.dtype)

    if T2 is None:
        T2 = T @ T

    if lam is None:
        lam = hamiltonian_traceless_eigenvalues(T, already_symmetric=True)

    if c1 is None:
        c1 = hamiltonian_traceless_c1(T, T2=T2)

    if N == 3:
        denom = 3.0 * lam**2 + c1[..., None]

        # Guard against near-degenerate eigenvalues (e.g. vacuum or Δm²₂₁ → 0).
        safe_denom = torch.where(
            denom.abs() < _DENOM_EPS,
            denom.new_full((), _DENOM_EPS),
            denom,
        )

        M = (
            (lam**2 + c1[..., None])[..., :, None, None] * I_N
            + lam[..., :, None, None] * T[..., None, :, :]
            + T2[..., None, :, :]
        ) / safe_denom[..., :, None, None]

    elif N == 4:
        if T3 is None:
            T3 = T2 @ T

        if e3 is None:
            e3 = hamiltonian_traceless_e3(T, T3=T3)

        lam2 = lam * lam
        lam3 = lam2 * lam
        e2_lam = c1[..., None] * lam  # (..., N)

        denom = 4.0 * lam3 + 2.0 * e2_lam - e3[..., None]

        safe_denom = torch.where(
            denom.abs() < _DENOM_EPS,
            denom.new_full((), _DENOM_EPS),
            denom,
        )

        numerator_scalar = lam3 + e2_lam - e3[..., None]  # (..., N)

        M = (
            T3[..., None, :, :]
            + lam[..., :, None, None] * T2[..., None, :, :]
            + (c1[..., None] + lam2)[..., :, None, None] * T[..., None, :, :]
            + numerator_scalar[..., :, None, None] * I_N
        ) / safe_denom[..., :, None, None]

    else:
        raise ValueError(
            f"hamiltonian_spectral_projectors_traceless supports N in {{3, 4}}, got N={N}."
        )

    degenerate = _spectral_degeneracy_mask(
        lam, relative_eps=relative_eps, absolute_eps=absolute_eps,
    )
    if bool(torch.any(degenerate)):
        _, V = torch.linalg.eigh(T)
        M_eigh = torch.einsum("...ia,...ja->...aij", V, V.conj()).to(dtype=T.dtype)
        M = torch.where(degenerate[..., None, None, None], M_eigh, M)

    return M, lam, c1


def hamiltonian_spectral_data(
    H: torch.Tensor,
    trace_H: torch.Tensor | None = None,
    *,
    relative_eps: float = _DEGENERACY_RELATIVE_EPS,
    absolute_eps: float = _DEGENERACY_ABSOLUTE_EPS,
) -> dict[str, torch.Tensor]:
    """
    Return trace, traceless Hamiltonian, eigenvalues, and spectral projectors for H.

    Args:
        H: Hamiltonian tensor shaped (..., N, N) in km^-1, N in {3, 4}.
        trace_H: Trace of the Hamiltonian shaped (...) and expressed in km^-1.
        relative_eps: Degeneracy-detector threshold forwarded to
            ``hamiltonian_spectral_projectors_traceless``, see
            ``_spectral_degeneracy_mask``.
        absolute_eps: Degeneracy-detector scale floor forwarded to
            ``hamiltonian_spectral_projectors_traceless``, see
            ``_spectral_degeneracy_mask``.

    Returns:
        Dictionary containing the traceless Hamiltonian, trace, eigenvalues,
        c1 invariant, and spectral projectors.

    Notes:
        T @ T is computed once here and forwarded to both ``hamiltonian_traceless_c1``
        and ``hamiltonian_spectral_projectors_traceless`` to avoid redundant bmm.
        T @ T @ T is additionally computed once for N=4 (unused, and skipped,
        for N=3). The symmetrization of T is also done once; downstream
        helpers receive ``already_symmetric=True`` so they skip the redundant
        transpose.
    """
    T, trace_H = hamiltonian_traceless(H, trace_H=trace_H)
    # Enforce Hermitian symmetry once; pass the flag to avoid a second transpose.
    T = 0.5 * (T + T.conj().transpose(-1, -2))
    N = T.shape[-1]

    # Compute T² once and reuse for both c1 and the spectral projectors.
    T2 = T @ T
    T3 = T2 @ T if N == 4 else None

    lam = hamiltonian_traceless_eigenvalues(T, already_symmetric=True)
    c1 = hamiltonian_traceless_c1(T, T2=T2)
    e3 = hamiltonian_traceless_e3(T, T3=T3) if N == 4 else None
    M, lam, c1 = hamiltonian_spectral_projectors_traceless(
        T, lam=lam, c1=c1, T2=T2, e3=e3, T3=T3,
        relative_eps=relative_eps, absolute_eps=absolute_eps,
    )

    return {
        "T": T,
        "trace": trace_H,
        "lam": lam,
        "c1": c1,
        "M": M,
    }


def spectral_projector_residuals(
    M: torch.Tensor,
    T: torch.Tensor,
    lam: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute diagnostic residual norms for a spectral projector decomposition.

    Every returned quantity should be at or near floating-point precision for
    a numerically healthy decomposition ``T = sum_a lam_a * M_a``. Used both
    by tests (as an explicit, reusable set of invariant checks) and by
    degeneracy-handling instrumentation to detect when the closed-form
    projector formula has lost too much precision and a fallback is needed.

    Args:
        M: Spectral projectors shaped (..., N, N, N); the leading N indexes
            the projector (``M[..., a, :, :]`` is ``M_a``).
        T: Traceless Hermitian matrix shaped (..., N, N) that ``M`` decomposes.
        lam: Eigenvalues shaped (..., N), paired with the leading ``M`` index.

    Returns:
        Dict of real-valued residual norms, each shaped (...,) (the batch
        shape of ``T``):
            "completeness": ``||sum_a M_a - I||_F``.
            "idempotency": ``max_a ||M_a @ M_a - M_a||_F``.
            "orthogonality": ``max_{a != b} ||M_a @ M_b||_F``.
            "trace_one": ``max_a |tr(M_a) - 1|`` (each projector has rank 1).
            "reconstruction": ``||sum_a lam_a * M_a - T||_F``.
    """
    N = T.shape[-1]
    I = torch.eye(N, device=T.device, dtype=T.dtype).expand_as(T)

    completeness = (M.sum(dim=-3) - I).norm(dim=(-2, -1))

    M_squared = torch.einsum("...aik,...akj->...aij", M, M)
    idempotency = (M_squared - M).norm(dim=(-2, -1)).amax(dim=-1)

    MM_pairs = torch.einsum("...aik,...bkj->...abij", M, M)
    pair_norm = MM_pairs.norm(dim=(-2, -1))
    off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=T.device)
    pair_norm_offdiag = pair_norm.masked_fill(~off_diag_mask, 0.0)
    orthogonality = pair_norm_offdiag.amax(dim=(-2, -1))

    trace_M = torch.diagonal(M, dim1=-2, dim2=-1).sum(dim=-1)
    trace_one = (trace_M - 1.0).abs().amax(dim=-1)

    reconstruction = ((lam[..., :, None, None] * M).sum(dim=-3) - T).norm(dim=(-2, -1))

    return {
        "completeness": completeness,
        "idempotency": idempotency,
        "orthogonality": orthogonality,
        "trace_one": trace_one,
        "reconstruction": reconstruction,
    }


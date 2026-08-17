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

The peanuts evolution method uses the eigenvalues of T (by default obtained
from ``torch.linalg.eigvalsh``; a closed-form Cardano/Ferrari alternative is
also available, see ``Analytic (Cardano/Ferrari) eigenvalues`` below) and
the associated spectral projectors M_a. These projectors
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

    hamiltonian_traceless_e4(...)
        Computes e4 = det(T), the fourth elementary symmetric polynomial of
        the eigenvalues (N=4 only), used by the Ferrari eigenvalue solver.

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

Analytic (Cardano/Ferrari) eigenvalues
---------------------------------------
``hamiltonian_traceless_eigenvalues`` defaults to ``torch.linalg.eigvalsh``.
Passing ``analytic=True`` instead evaluates a closed-form solution:

    N=3: the trigonometric (Cardano) solution of the depressed cubic
        ``lambda^3 + c1*lambda + c0 = 0`` -- see
        ``_hamiltonian_traceless_eigenvalues_cardano``.
    N=4: the Ferrari solution of the depressed quartic
        ``lambda^4 + c1*lambda^2 - e3*lambda + e4 = 0`` (e4 = det(T), see
        ``hamiltonian_traceless_e4``), which factors into two real
        quadratics via one real root of a resolvent cubic -- itself solved
        with the same trigonometric formula as the N=3 case, since a
        real-rooted quartic (guaranteed here: T is Hermitian) always has a
        real-rooted resolvent cubic too. See
        ``_hamiltonian_traceless_eigenvalues_ferrari``.

The N=3 trigonometric formula computes accurate eigenvalues unconditionally
(only the *projector* formula built from them needs the eigh fallback near
degeneracy, see ``hamiltonian_spectral_projectors_traceless``). 

The N=4 Ferrari factorization is not quite as forgiving -- an unlucky 
resolvent-root pairing can make the eigenvalues themselves ill-conditioned, 
not just the projectors -- so ``_hamiltonian_traceless_eigenvalues_ferrari``
additionally falls back to ``eigvalsh`` per-batch-entry in that (rare, 
near-T=0 or specially-paired-spectrum) regime. 

"""



from __future__ import annotations

import math

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


def hamiltonian_traceless_e4(
    T: torch.Tensor,
    T2: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the fourth elementary symmetric invariant e4 (= det(T)) of a
    traceless N=4 Hamiltonian.

    Formula: Uses Newton's identity e4 = (2*c1^2 - tr(T^4)) / 4 (derivable
    from the power sums p1=tr(T)=0, p2=tr(T^2)=-2*c1, p3=tr(T^3)=3*e3,
    p4=tr(T^4), independently of e3). ``tr(T^4) = tr(T2 @ T2)`` is computed
    as ``||T2||_F^2 = sum_ij |T2_ij|^2`` rather than via an extra matmul: T2
    = T @ T is itself Hermitian (since T is), and for any Hermitian A,
    ``tr(A @ A) = tr(A @ A^dagger) = ||A||_F^2``. Verified against
    ``torch.linalg.det`` and against a hand-worked example (eigenvalues
    4, 1, -2, -3: c1=-15, e3=10, e4=24 = 4*1*-2*-3).

    Used by the N=4 Ferrari eigenvalue solver (see
    ``_hamiltonian_traceless_eigenvalues_ferrari``): e4 is the constant term
    of the depressed quartic characteristic polynomial
    ``lambda^4 + c1*lambda^2 - e3*lambda + e4 = 0`` (no cubic term since T
    is traceless).

    Args:
        T: Traceless Hermitian Hamiltonian tensor shaped (..., 4, 4).
        T2: Optional precomputed T @ T shaped (..., 4, 4). When provided the
            matrix multiplication is skipped, avoiding a redundant bmm.

    Returns:
        Real-valued tensor (in T's dtype) e4 with the batch shape of T.
    """
    if T2 is None:
        T2 = T @ T

    c1 = hamiltonian_traceless_c1(T, T2=T2)
    trT4 = (T2 * T2.conj()).real.sum(dim=(-2, -1)).to(dtype=T.dtype)

    return (2.0 * c1 * c1 - trT4) / 4.0


def _hamiltonian_traceless_eigenvalues_cardano(
    T: torch.Tensor,
    c1: torch.Tensor | None = None,
    c0: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Closed-form (Cardano/trigonometric) eigenvalues of a traceless N=3
    Hermitian Hamiltonian.

    Formula: The characteristic polynomial of a traceless T is the depressed
    cubic ``lambda^3 + c1*lambda + c0 = 0`` (c1, c0 as defined by
    ``hamiltonian_traceless_c1``/``hamiltonian_traceless_c0``). Since T is
    Hermitian, c1 = -tr(T^2)/2 <= 0 always (equality iff T = 0) and the three
    roots are real, given by the standard trigonometric solution

        lambda_k = 2*sqrt(-c1/3) * cos[ (1/3)*arccos(r) - 2*pi*k/3 ],
        r = -c0 / (2*(-c1/3)^{3/2}),  k = 0, 1, 2,

    equivalent to the textbook eigenvalue formula for a 3x3 real symmetric
    matrix (Smith 1961) specialised to the traceless case. ``r`` is clamped
    to [-1, 1] to absorb floating-point overshoot at (near-)degenerate
    spectra, where this formula loses precision in the same regime as
    ``hamiltonian_spectral_projectors_traceless``'s closed-form projectors
    (see ``_spectral_degeneracy_mask``).

    Args:
        T: Traceless Hermitian Hamiltonian tensor shaped (..., 3, 3).
        c1: Optional precomputed quadratic invariant (see
            ``hamiltonian_traceless_c1``). Computed from T when omitted.
        c0: Optional precomputed cubic invariant (see
            ``hamiltonian_traceless_c0``). Computed from T when omitted.

    Returns:
        Tensor shaped (..., 3) with the real eigenvalues, ascending (to
        match ``torch.linalg.eigvalsh``'s convention), represented in
        T.dtype.
    """
    if c1 is None:
        c1 = hamiltonian_traceless_c1(T)
    if c0 is None:
        c0 = hamiltonian_traceless_c0(T)

    c1_r = c1.real if torch.is_complex(c1) else c1
    c0_r = c0.real if torch.is_complex(c0) else c0
    eps = torch.finfo(c1_r.dtype).eps

    # c1 <= 0 for Hermitian T; clamp absorbs floating-point noise at c1 ~ 0.
    scale = torch.sqrt(torch.clamp(-c1_r, min=0.0) / 3.0)
    safe_scale = torch.clamp(scale, min=eps)

    r = torch.clamp(-c0_r / (2.0 * safe_scale**3), min=-1.0, max=1.0)
    phi = torch.acos(r) / 3.0

    two_pi_3 = 2.0 * math.pi / 3.0
    lam0 = 2.0 * scale * torch.cos(phi)
    lam1 = 2.0 * scale * torch.cos(phi + two_pi_3)
    lam2 = -lam0 - lam1  # exact sum-to-zero, independent of trig round-off

    lam = torch.sort(torch.stack([lam0, lam1, lam2], dim=-1), dim=-1).values

    return lam.to(dtype=T.dtype)


def _hamiltonian_traceless_eigenvalues_ferrari(
    T: torch.Tensor,
    c1: torch.Tensor | None = None,
    e3: torch.Tensor | None = None,
    e4: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Closed-form (Ferrari) eigenvalues of a traceless N=4 Hermitian Hamiltonian.

    Formula: The characteristic polynomial of a traceless T is the depressed
    quartic ``lambda^4 + p*lambda^2 + q*lambda + r = 0``, with ``p = c1``
    (= e2, see ``hamiltonian_traceless_c1``), ``q = -e3`` (see
    ``hamiltonian_traceless_e3``), and ``r = e4`` (see
    ``hamiltonian_traceless_e4``). Ferrari's method factors this into two
    real quadratics via one real root ``m`` of the resolvent cubic

        m^3 + p*m^2 + (p^2/4 - r)*m - q^2/8 = 0,

    solved here by depressing it (``m = z - p/3``) and reusing the same
    trigonometric approach as ``_hamiltonian_traceless_eigenvalues_cardano``:
    for a real-rooted quartic (guaranteed here since T is Hermitian) the
    resolvent cubic's three roots are exactly the real pairings
    ``m_{ij|kl} = -(lambda_i+lambda_j)*(lambda_k+lambda_l)/2`` of the four
    (real) quartic roots over the three ways to split them into two pairs,
    so it is real-rooted too and the same closed form applies
    unconditionally. Given the largest resolvent root m (the standard choice
    for conditioning -- it maximises ``2m``, keeping the next step's square
    root away from zero whenever *any* pairing is well separated).

        s = sqrt(2*m),
        lambda in { (s +- sqrt(disc1))/2 , (-s +- sqrt(disc2))/2 },
        disc1 = -2*(m + p + q/s), disc2 = -2*(m + p - q/s).

    (Both the resolvent cubic and the final formula above were verified
    against a hand-worked example -- eigenvalues 4, 1, -2, -3 give
    p=-15, q=-10, r=24, resolvent roots {12.5, 2, 0.5}, and the largest
    root m=12.5 reproduces all four eigenvalues exactly -- and numerically
    against ``torch.linalg.eigvalsh`` for random Hermitian traceless
    matrices, see ``core/perturbative/test/test1_perturbative_spectral.py``.)

    Unlike the N=3 trigonometric formula, this factorization is not
    unconditionally well-conditioned, in two independent ways: the chosen
    pairing's ``m`` can itself be close to singular (only possible, for a
    traceless Hermitian T, near T ~ 0), or the resolvent cubic's own three
    roots can nearly coincide even when the *chosen* ``m`` is perfectly
    well separated from zero -- e.g. lambda=(a,a,-a,-a) has a
    well-conditioned pairing (m=2a^2) whose resolvent cubic nonetheless has
    a repeated root elsewhere, which loses precision through
    ``acos``'s diverging derivative at +-1. Batch entries flagged by either
    check fall back to ``torch.linalg.eigvalsh`` directly -- unlike
    ``hamiltonian_spectral_projectors_traceless``'s degeneracy fallback,
    which only ever needs to replace the *projectors*, this closed form can
    lose the *eigenvalues* themselves in these regimes, so the fallback is
    applied here rather than left to the caller.

    Args:
        T: Traceless Hermitian Hamiltonian tensor shaped (..., 4, 4).
        c1: Optional precomputed quadratic invariant (= e2, see
            ``hamiltonian_traceless_c1``). Computed from T when omitted.
        e3: Optional precomputed cubic invariant (see
            ``hamiltonian_traceless_e3``). Computed from T when omitted.
        e4: Optional precomputed quartic invariant (see
            ``hamiltonian_traceless_e4``). Computed from T when omitted.

    Returns:
        Tensor shaped (..., 4) with the real eigenvalues, ascending (to
        match ``torch.linalg.eigvalsh``'s convention), represented in
        T.dtype.
    """
    if c1 is None:
        c1 = hamiltonian_traceless_c1(T)
    if e3 is None:
        e3 = hamiltonian_traceless_e3(T)
    if e4 is None:
        e4 = hamiltonian_traceless_e4(T)

    p = c1.real if torch.is_complex(c1) else c1
    q = -(e3.real if torch.is_complex(e3) else e3)
    r = e4.real if torch.is_complex(e4) else e4
    eps = torch.finfo(p.dtype).eps

    # Resolvent cubic m^3 + p*m^2 + (p^2/4 - r)*m - q^2/8 = 0, depressed via
    # m = z - p/3 into z^3 + A*z + B = 0, solved with the same trigonometric
    # formula as _hamiltonian_traceless_eigenvalues_cardano (guaranteed
    # real-rooted, see the docstring above).
    A = -(p * p / 12.0 + r)
    B = -(p ** 3) / 108.0 + (p * r) / 3.0 - (q * q) / 8.0

    scale = torch.sqrt(torch.clamp(-A, min=0.0) / 3.0)
    safe_scale = torch.clamp(scale, min=eps)
    r_arg = torch.clamp(-B / (2.0 * safe_scale**3), min=-1.0, max=1.0)
    phi = torch.acos(r_arg) / 3.0

    two_pi_3 = 2.0 * math.pi / 3.0
    z0 = 2.0 * scale * torch.cos(phi)
    z1 = 2.0 * scale * torch.cos(phi + two_pi_3)
    z2 = -z0 - z1
    m = torch.maximum(torch.maximum(z0, z1), z2) - p / 3.0

    s = torch.sqrt(torch.clamp(2.0 * m, min=0.0))
    safe_s = torch.clamp(s, min=eps)

    disc1 = torch.clamp(-2.0 * (m + p + q / safe_s), min=0.0)
    disc2 = torch.clamp(-2.0 * (m + p - q / safe_s), min=0.0)
    sq1 = torch.sqrt(disc1)
    sq2 = torch.sqrt(disc2)

    lam = torch.stack(
        [0.5 * (s + sq1), 0.5 * (s - sq1), 0.5 * (-s + sq2), 0.5 * (-s - sq2)],
        dim=-1,
    )
    lam = torch.sort(lam, dim=-1).values

    # Two independent failure modes, each checked with the same relative-gap
    # / absolute-floor structure (and calibrated thresholds) as
    # _spectral_degeneracy_mask:
    #
    # 1. m small relative to the natural (p, sqrt(|r|)) scale: the chosen
    #    pairing is itself close to singular (only possible, for a
    #    traceless Hermitian T, near T ~ 0).
    # 2. The resolvent cubic's own roots {z0, z1, z2} nearly coincide: even
    #    though the *chosen* m can be perfectly well separated from zero
    #    (failure mode 1 does not fire), acos'(r) = -1/sqrt(1-r^2) diverges
    #    as r_arg -> +-1 -- exactly the repeated-resolvent-root condition --
    #    so a tiny floating-point perturbation in r_arg is amplified into a
    #    much larger error in z0/z1/z2 and hence in m and lam. This is the
    #    dominant failure mode in practice: e.g. lambda=(a,a,-a,-a) has a
    #    perfectly well-separated pairing (m = 2a^2) but z1 == z2 exactly
    #    (r_arg clamps to 1), which loses ~8 digits on GPU trigonometric
    #    kernels despite failure mode 1 never firing.
    natural_scale = p.abs() + torch.sqrt(torch.clamp(r.abs(), min=0.0))
    pairing_singular = (
        (m < _DEGENERACY_RELATIVE_EPS * eps * natural_scale)
        | (natural_scale < _DEGENERACY_ABSOLUTE_EPS * eps)
    )

    z_gap = torch.minimum(torch.minimum((z0 - z1).abs(), (z1 - z2).abs()), (z0 - z2).abs())
    resolvent_degenerate = (
        (z_gap < _DEGENERACY_RELATIVE_EPS * eps * scale)
        | (scale < _DEGENERACY_ABSOLUTE_EPS * eps)
    )

    unreliable = pairing_singular | resolvent_degenerate
    if bool(torch.any(unreliable)):
        lam_eigh = torch.linalg.eigvalsh(T).to(dtype=p.dtype)
        lam = torch.where(unreliable[..., None], lam_eigh, lam)

    return lam.to(dtype=T.dtype)


def hamiltonian_traceless_eigenvalues(
    T: torch.Tensor,
    *,
    already_symmetric: bool = False,
    analytic: bool = False,
    c1: torch.Tensor | None = None,
    c0: torch.Tensor | None = None,
    e3: torch.Tensor | None = None,
    e4: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the eigenvalues of a traceless Hermitian Hamiltonian.

    Args:
        T: Traceless Hermitian Hamiltonian tensor shaped (..., N, N), N in {3, 4}.
        already_symmetric: When True, skip the symmetrization step.  Set this
            flag when the caller (e.g. ``hamiltonian_spectral_data``) has
            already enforced Hermitian symmetry to avoid a redundant operation.
        analytic: If True, evaluate a closed-form solution instead of
            ``torch.linalg.eigvalsh``: the Cardano solution
            (``_hamiltonian_traceless_eigenvalues_cardano``) for
            ``T.shape[-1] == 3``, or the Ferrari solution
            (``_hamiltonian_traceless_eigenvalues_ferrari``) for
            ``T.shape[-1] == 4``. Falls back to plain ``eigvalsh`` for any
            other N, since no closed form is defined there.
        c1: Optional precomputed quadratic invariant, forwarded to the
            Cardano/Ferrari path when ``analytic`` is used. Ignored otherwise.
        c0: Optional precomputed cubic invariant, forwarded to the Cardano
            path (N=3) when ``analytic`` is used. Ignored otherwise.
        e3: Optional precomputed cubic invariant, forwarded to the Ferrari
            path (N=4) when ``analytic`` is used. Ignored otherwise.
        e4: Optional precomputed quartic invariant, forwarded to the Ferrari
            path (N=4) when ``analytic`` is used. Ignored otherwise.

    Returns:
        Tensor shaped (..., N) with the real eigenvalues represented in T.dtype.
    """
    T = T.contiguous()
    if not already_symmetric:
        T = 0.5 * (T + T.conj().transpose(-1, -2))

    if not torch.isfinite(T).all():
        raise FloatingPointError("T contains NaN or Inf before eigenvalue computation.")

    if analytic and T.shape[-1] == 3:
        return _hamiltonian_traceless_eigenvalues_cardano(T, c1=c1, c0=c0)

    if analytic and T.shape[-1] == 4:
        return _hamiltonian_traceless_eigenvalues_ferrari(T, c1=c1, e3=e3, e4=e4)

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
    c0: torch.Tensor | None = None,
    e4: torch.Tensor | None = None,
    *,
    relative_eps: float = _DEGENERACY_RELATIVE_EPS,
    absolute_eps: float = _DEGENERACY_ABSOLUTE_EPS,
    analytic_eigenvalues: bool = False,
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
    only made when at least one batch entry needs it.

    **Gradient note (2026-08, no longer no_grad-protected):** this module
    and its callers (``core/perturbative/evolutor.py``) used to run entirely
    under ``@torch.no_grad()``. That decorator has since been removed (along
    with 10 others across ``medium.atmosphere``/``core.numerical.evolutor``/
    ``core.common.evolutor``) to enable direct-gradient fits through matter
    propagation. The non-degenerate closed-form ``M_a`` path above (a
    polynomial in ``T``/``lam``) has been verified end-to-end through
    ``medium.atmosphere.probability.atmosphere_probability_state`` (matter,
    method="analytical") against a central finite-difference derivative --
    autograd and finite differences agree to ~8 significant figures for
    d(P)/d(theta12)/d(theta13) at a representative (non-degenerate)
    benchmark point. (An earlier version of this note reported a spurious
    near-zero gradient; that was a test-methodology artifact -- summing a
    quantity that is analytically constant regardless of theta, such as
    ``lam.sum()`` (== tr(T) == 0 by construction) or a bare probability sum
    (== 1 by unitarity), rather than a genuine bug.) The
    ``torch.linalg.eigh`` degeneracy fallback just below, used only for
    batch entries with a nearly degenerate spectrum, has *not* been
    separately gradient-verified -- raw eigenvector-based projectors are the
    textbook case where autograd gradients become unstable near degenerate
    eigenvalues, and no test here currently exercises that branch under
    gradient tracking.

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
        c0: N=3 only, and only used when ``analytic_eigenvalues`` is True and
            ``lam`` is None. Optional precomputed cubic invariant, forwarded
            to ``hamiltonian_traceless_eigenvalues``'s Cardano path. Ignored
            otherwise.
        e4: N=4 only, and only used when ``analytic_eigenvalues`` is True and
            ``lam`` is None. Optional precomputed quartic invariant (see
            ``hamiltonian_traceless_e4``), forwarded to
            ``hamiltonian_traceless_eigenvalues``'s Ferrari path. Ignored
            otherwise.
        relative_eps: Degeneracy-detector threshold, see
            ``_spectral_degeneracy_mask``.
        absolute_eps: Degeneracy-detector scale floor, see
            ``_spectral_degeneracy_mask``.
        analytic_eigenvalues: If True and ``lam`` is None, compute ``lam``
            with the closed-form Cardano (N=3) or Ferrari (N=4) solution.
            Ignored when ``lam`` is supplied directly. See
            ``hamiltonian_traceless_eigenvalues``.

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

    if c1 is None:
        c1 = hamiltonian_traceless_c1(T, T2=T2)

    if N == 4 and T3 is None:
        T3 = T2 @ T

    if N == 4 and e3 is None:
        e3 = hamiltonian_traceless_e3(T, T3=T3)

    if lam is None:
        if analytic_eigenvalues and N == 4 and e4 is None:
            e4 = hamiltonian_traceless_e4(T, T2=T2)
        lam = hamiltonian_traceless_eigenvalues(
            T, already_symmetric=True, analytic=analytic_eigenvalues,
            c1=c1, c0=c0, e3=e3, e4=e4,
        )

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
        assert T3 is not None and e3 is not None  # guaranteed set above for N == 4

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
    analytic_eigenvalues: bool = False,
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
        analytic_eigenvalues: If True, compute the eigenvalues with the
            closed-form Cardano (N=3) or Ferrari (N=4) solution instead of
            ``torch.linalg.eigvalsh``. See ``hamiltonian_traceless_eigenvalues``.

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

    c1 = hamiltonian_traceless_c1(T, T2=T2)
    c0 = hamiltonian_traceless_c0(T) if (analytic_eigenvalues and N == 3) else None
    e3 = hamiltonian_traceless_e3(T, T3=T3) if N == 4 else None
    e4 = hamiltonian_traceless_e4(T, T2=T2) if (analytic_eigenvalues and N == 4) else None
    lam = hamiltonian_traceless_eigenvalues(
        T, already_symmetric=True, analytic=analytic_eigenvalues,
        c1=c1, c0=c0, e3=e3, e4=e4,
    )
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


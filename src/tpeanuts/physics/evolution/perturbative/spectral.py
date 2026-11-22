#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
#
# TPeanuts — Fast Simulation of Neutrino Oscillations in Matter.
#
# Developed as part of the Master's Thesis for the MSc in Advanced Physics,
# specialization in Theoretical Physics, at the Universitat de València.
#
# Author
# ------
# Juan Ramón Díaz Santos
# diazjuan@alumni.uv.es
#
# Supervisors
# -----------
# Roberto Ruiz de Austri Bazán
# rruiz@ific.uv.es
#
# Michele Lucente
# michele.lucente@unibo.it
#
# Universitat de València, Spain
# September 2026
# =============================================================================

"""Closed-form spectral decomposition of a traceless Hermitian Hamiltonian, N in {3, 4}.

Ported from ``v1_0/core/perturbative/spectral.py`` with the underlying
mathematics unchanged (verified there both symbolically and numerically
against ``torch.linalg.eigh``/``eigvalsh``) -- only the module header,
docstrings, and a handful of naming/typing conventions were adapted to this
project's own style. This module is the foundation
``tpeanuts.physics.evolution.perturbative.analytic`` builds its exact,
closed-form alternative to ``tpeanuts.physics.evolution.numerical.numerical``'s
``torch.linalg.matrix_exp`` on: any Hermitian generator ``H`` splits as

    H = T + Tr(H)/N * I,

with ``T`` traceless. For ``N in {3, 4}`` (the only sizes with a
polynomial-root closed form -- Abel-Ruffini rules out N >= 5), ``T``'s
eigenvalues have an exact trigonometric (Cardano, N=3) or quartic (Ferrari,
N=4) solution, and the corresponding spectral projectors ``M_a`` (built as
an explicit polynomial in ``T`` and the eigenvalues, not by an eigenvector
computation) let the constant-generator evolution operator for a path
length ``L`` be written directly as

    U(L) = sum_a exp[-i (lambda_a + Tr(H)/N) L] M_a

-- exact, not a series truncation, for any ``L``. Both the eigenvalue
formulas and the projector formulas degrade gracefully near a (near-)
degenerate spectrum, where they lose precision, by falling back to
``torch.linalg.eigh``/``eigvalsh`` for the affected batch entries only (see
``_spectral_degeneracy_mask`` and each closed-form eigenvalue function's own
fallback) -- never silently wrong, only silently slower for those entries.

Every function here operates on a bare ``torch.Tensor`` -- no
``RuntimeContext``, no ``FlavourSpace``, no mixing matrix -- exactly
mirroring ``tpeanuts.physics.evolution.numerical.numerical``'s own "knows
nothing about the physics that built H" scope.

Lives inside ``perturbative/`` rather than its own folder because it is
genuinely specific to the closed-form Cardano/Ferrari technique, not
general evolution infrastructure like ``numerical.numerical``: its only
two consumers, ``perturbative.analytic`` and ``perturbative.perturbative``,
are both colocated here too, unlike ``numerical.numerical``, which every
other strategy either specializes or falls back to.

Module contents:
    hamiltonian_traceless(...)
        Split H into its trace and traceless parts.
    hamiltonian_traceless_c0(...)
        Cubic invariant c0 = -tr(T^3)/3 (N=3 characteristic-polynomial
        convention).
    hamiltonian_traceless_c1(...)
        Quadratic invariant c1 = -tr(T^2)/2 = e2 (any N).
    hamiltonian_traceless_e3(...)
        Third elementary symmetric invariant e3 = tr(T^3)/3 = -c0 (any N;
        kept separate from c0 to avoid a sign-confusion bug at call sites).
    hamiltonian_traceless_e4(...)
        Fourth elementary symmetric invariant e4 = det(T) (N=4 only).
    hamiltonian_traceless_eigenvalues(...)
        Eigenvalues of the traceless Hamiltonian: ``torch.linalg.eigvalsh``
        by default, or the closed-form Cardano/Ferrari solution when
        ``analytic=True``.
    hamiltonian_spectral_projectors_traceless(...)
        Spectral projectors M_a from T and its eigenvalues, N in {3, 4}
        only (raises otherwise) -- with the eigh-based degeneracy fallback.
    hamiltonian_spectral_data(...)
        Convenience wrapper: trace, traceless Hamiltonian, eigenvalues,
        c1, and spectral projectors in one call.
    spectral_projector_residuals(...)
        Diagnostic residual norms (completeness, idempotency,
        orthogonality, trace, reconstruction) for a projector
        decomposition -- used by this module's own tests.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from tpeanuts.util.math import eigh_chunked, eigvalsh_chunked

# Minimum absolute value of the spectral projector denominator (3*lambda^2 +
# c1 for N=3, the N=4 analogue for N=4). A last-resort safety net against
# literal division by zero; the primary degeneracy handling is the relative
# detector + eigh fallback below (see ``_spectral_degeneracy_mask`` and
# ``hamiltonian_spectral_projectors_traceless``).
_DENOM_EPS: float = 1.0e-30

# Degeneracy-detection thresholds, expressed as multipliers of
# ``torch.finfo(dtype).eps`` so they scale automatically with precision
# (float32/complex64 vs float64/complex128). ``_DEGENERACY_RELATIVE_EPS``
# bounds the minimum pairwise eigenvalue gap relative to the spectrum scale;
# ``_DEGENERACY_ABSOLUTE_EPS`` is a floor on the spectrum scale itself, for
# the pathological T~0 case where the relative test is meaningless (0/0).
_DEGENERACY_RELATIVE_EPS: float = 1.0e3
_DEGENERACY_ABSOLUTE_EPS: float = 1.0e3


def hamiltonian_traceless(
    H: torch.Tensor,
    trace_H: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtract one Nth of the trace from a Hamiltonian to obtain its traceless part.

    Args:
        H: Hermitian generator shaped ``(..., N, N)``, ``N = H.shape[-1]``.
        trace_H: Optional precomputed trace, shaped ``(...)``. Computed
            from ``H`` when omitted.

    Returns:
        Tuple ``(T, trace_H)``: the traceless part ``T = H -
        trace_H/N * I``, shaped ``(..., N, N)``, and the trace itself,
        shaped ``(...)``.
    """
    N = H.shape[-1]
    I_N = torch.eye(N, device=H.device, dtype=H.dtype)

    if trace_H is None:
        trace_H = torch.diagonal(H, dim1=-2, dim2=-1).sum(dim=-1)
    trace_H = trace_H.to(dtype=H.dtype)

    T = H - trace_H[..., None, None] * I_N / N
    return T, trace_H


def hamiltonian_traceless_c0(T: torch.Tensor) -> torch.Tensor:
    """Compute the cubic invariant c0 of a traceless 3x3 Hamiltonian.

    ``c0 = -tr(T^3)/3``. Together with the quadratic invariant c1 (see
    ``hamiltonian_traceless_c1``), c0 enters the characteristic polynomial
    of T, ``lambda^3 + c1*lambda + c0 = 0`` (no quadratic term since T is
    traceless), whose three real roots are the eigenvalues returned by
    ``hamiltonian_traceless_eigenvalues``.

    Args:
        T: Traceless Hermitian Hamiltonian shaped ``(..., 3, 3)``.

    Returns:
        Real tensor c0 with the batch shape of T.
    """
    T3 = T @ T @ T
    trT3 = torch.diagonal(T3, dim1=-2, dim2=-1).sum(dim=-1)
    return -trT3 / 3.0


def hamiltonian_traceless_c1(
    T: torch.Tensor, T2: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute the quadratic invariant c1 of a traceless Hamiltonian (any N).

    ``c1 = -tr(T^2)/2``.

    Args:
        T: Traceless Hermitian Hamiltonian shaped ``(..., N, N)``.
        T2: Optional precomputed ``T @ T``, shaped ``(..., N, N)`` -- skips
            a redundant matmul when supplied.

    Returns:
        Real tensor c1 with the batch shape of T.
    """
    if T2 is None:
        T2 = T @ T
    trT2 = torch.diagonal(T2, dim1=-2, dim2=-1).sum(dim=-1)
    return -trT2 / 2.0


def hamiltonian_traceless_e3(
    T: torch.Tensor, T3: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute the third elementary symmetric invariant e3 of a traceless Hamiltonian (any N).

    ``e3 = tr(T^3)/3`` (Newton's identity relating power sums to elementary
    symmetric polynomials does not depend on N). Used, together with c1
    (= e2), by the N=4 spectral projector formula. Note the sign relative
    to ``hamiltonian_traceless_c0``: ``e3 = -c0`` -- kept as a separate
    function (not ``-c0(T)`` inlined at call sites) to avoid a
    sign-confusion bug where a caller reuses c0 as if it already were e3.

    Args:
        T: Traceless Hermitian Hamiltonian shaped ``(..., N, N)``.
        T3: Optional precomputed ``T @ T @ T``, shaped ``(..., N, N)``.

    Returns:
        Real tensor e3 with the batch shape of T.
    """
    if T3 is None:
        T3 = T @ T @ T
    trT3 = torch.diagonal(T3, dim1=-2, dim2=-1).sum(dim=-1)
    return trT3 / 3.0


def hamiltonian_traceless_e4(
    T: torch.Tensor, T2: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute the fourth elementary symmetric invariant e4 (= det(T)) of a traceless N=4 Hamiltonian.

    Uses Newton's identity ``e4 = (2*c1^2 - tr(T^4)) / 4`` (derivable from
    the power sums ``p1 = tr(T) = 0``, ``p2 = tr(T^2) = -2*c1``, ``p3 =
    tr(T^3) = 3*e3``, ``p4 = tr(T^4)``, independently of e3).
    ``tr(T^4) = tr(T2 @ T2)`` is computed as ``||T2||_F^2`` rather than via
    an extra matmul: ``T2 = T @ T`` is itself Hermitian (since T is), and
    for any Hermitian A, ``tr(A @ A) = tr(A @ A^dagger) = ||A||_F^2``. Used
    by the N=4 Ferrari eigenvalue solver: e4 is the constant term of the
    depressed quartic characteristic polynomial ``lambda^4 + c1*lambda^2 -
    e3*lambda + e4 = 0`` (no cubic term since T is traceless).

    Args:
        T: Traceless Hermitian Hamiltonian shaped ``(..., 4, 4)``.
        T2: Optional precomputed ``T @ T``, shaped ``(..., 4, 4)``.

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
    c1: Optional[torch.Tensor] = None,
    c0: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Closed-form (Cardano/trigonometric) eigenvalues of a traceless N=3 Hermitian Hamiltonian.

    The characteristic polynomial of a traceless T is the depressed cubic
    ``lambda^3 + c1*lambda + c0 = 0``. Since T is Hermitian, ``c1 =
    -tr(T^2)/2 <= 0`` always (equality iff T = 0) and the three roots are
    real, given by the standard trigonometric solution

        lambda_k = 2*sqrt(-c1/3) * cos[ (1/3)*arccos(r) - 2*pi*k/3 ],
        r = -c0 / (2*(-c1/3)^{3/2}),  k = 0, 1, 2,

    equivalent to the textbook eigenvalue formula for a 3x3 real symmetric
    matrix (Smith 1961) specialised to the traceless case. ``r`` is clamped
    to ``[-1, 1]`` to absorb floating-point overshoot at (near-)degenerate
    spectra, where this formula loses precision in the same regime as
    ``hamiltonian_spectral_projectors_traceless``'s closed-form projectors.

    Args:
        T: Traceless Hermitian Hamiltonian shaped ``(..., 3, 3)``.
        c1: Optional precomputed quadratic invariant. Computed from T when
            omitted.
        c0: Optional precomputed cubic invariant. Computed from T when
            omitted.

    Returns:
        Tensor shaped ``(..., 3)`` with the real eigenvalues, ascending
        (matching ``torch.linalg.eigvalsh``'s convention), in T's dtype.
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
    c1: Optional[torch.Tensor] = None,
    e3: Optional[torch.Tensor] = None,
    e4: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Closed-form (Ferrari) eigenvalues of a traceless N=4 Hermitian Hamiltonian.

    The characteristic polynomial of a traceless T is the depressed
    quartic ``lambda^4 + p*lambda^2 + q*lambda + r = 0``, with ``p = c1``
    (= e2), ``q = -e3``, and ``r = e4``. Ferrari's method factors this into
    two real quadratics via one real root ``m`` of the resolvent cubic

        m^3 + p*m^2 + (p^2/4 - r)*m - q^2/8 = 0,

    solved here by depressing it (``m = z - p/3``) and reusing the same
    trigonometric approach as
    ``_hamiltonian_traceless_eigenvalues_cardano``: for a real-rooted
    quartic (guaranteed here since T is Hermitian) the resolvent cubic's
    three roots are exactly the real pairings ``m_{ij|kl} =
    -(lambda_i+lambda_j)*(lambda_k+lambda_l)/2`` of the four (real) quartic
    roots over the three ways to split them into two pairs, so it is
    real-rooted too and the same closed form applies unconditionally.
    Given the largest resolvent root m (the standard choice for
    conditioning -- it maximises ``2m``, keeping the next step's square
    root away from zero whenever *any* pairing is well separated):

        s = sqrt(2*m),
        lambda in { (s +- sqrt(disc1))/2 , (-s +- sqrt(disc2))/2 },
        disc1 = -2*(m + p + q/s), disc2 = -2*(m + p - q/s).

    Unlike the N=3 trigonometric formula, this factorization is not
    unconditionally well-conditioned, in two independent ways: the chosen
    pairing's ``m`` can itself be close to singular (only possible, for a
    traceless Hermitian T, near T ~ 0), or the resolvent cubic's own three
    roots can nearly coincide even when the *chosen* ``m`` is perfectly
    well separated from zero -- e.g. ``lambda = (a,a,-a,-a)`` has a
    well-conditioned pairing (``m = 2a^2``) whose resolvent cubic
    nonetheless has a repeated root elsewhere, which loses precision
    through ``acos``'s diverging derivative at +-1. Batch entries flagged
    by either check fall back to ``torch.linalg.eigvalsh`` directly --
    unlike ``hamiltonian_spectral_projectors_traceless``'s degeneracy
    fallback, which only ever needs to replace the *projectors*, this
    closed form can lose the *eigenvalues* themselves in these regimes, so
    the fallback is applied here rather than left to the caller.

    Args:
        T: Traceless Hermitian Hamiltonian shaped ``(..., 4, 4)``.
        c1: Optional precomputed quadratic invariant (= e2).
        e3: Optional precomputed cubic invariant.
        e4: Optional precomputed quartic invariant.

    Returns:
        Tensor shaped ``(..., 4)`` with the real eigenvalues, ascending
        (matching ``torch.linalg.eigvalsh``'s convention), in T's dtype.
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
    B = -(p**3) / 108.0 + (p * r) / 3.0 - (q * q) / 8.0

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

    # Two independent failure modes, each checked with the same
    # relative-gap / absolute-floor structure (and calibrated thresholds)
    # as _spectral_degeneracy_mask:
    #
    # 1. m small relative to the natural (p, sqrt(|r|)) scale: the chosen
    #    pairing is itself close to singular (only possible, for a
    #    traceless Hermitian T, near T ~ 0).
    # 2. The resolvent cubic's own roots {z0, z1, z2} nearly coincide: even
    #    though the *chosen* m can be perfectly well separated from zero
    #    (failure mode 1 does not fire), acos'(r) = -1/sqrt(1-r^2)
    #    diverges as r_arg -> +-1 -- exactly the repeated-resolvent-root
    #    condition -- so a tiny floating-point perturbation in r_arg is
    #    amplified into a much larger error in z0/z1/z2 and hence in m and
    #    lam.
    natural_scale = p.abs() + torch.sqrt(torch.clamp(r.abs(), min=0.0))
    pairing_singular = (m < _DEGENERACY_RELATIVE_EPS * eps * natural_scale) | (
        natural_scale < _DEGENERACY_ABSOLUTE_EPS * eps
    )

    z_gap = torch.minimum(torch.minimum((z0 - z1).abs(), (z1 - z2).abs()), (z0 - z2).abs())
    resolvent_degenerate = (z_gap < _DEGENERACY_RELATIVE_EPS * eps * scale) | (
        scale < _DEGENERACY_ABSOLUTE_EPS * eps
    )

    unreliable = pairing_singular | resolvent_degenerate
    if bool(torch.any(unreliable)):
        lam_eigh = eigvalsh_chunked(T).to(dtype=p.dtype)
        lam = torch.where(unreliable[..., None], lam_eigh, lam)

    return lam.to(dtype=T.dtype)


def hamiltonian_traceless_eigenvalues(
    T: torch.Tensor,
    *,
    already_symmetric: bool = False,
    analytic: bool = False,
    c1: Optional[torch.Tensor] = None,
    c0: Optional[torch.Tensor] = None,
    e3: Optional[torch.Tensor] = None,
    e4: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the eigenvalues of a traceless Hermitian Hamiltonian.

    Args:
        T: Traceless Hermitian Hamiltonian shaped ``(..., N, N)``, ``N`` in
            ``{3, 4}`` for the ``analytic`` path (any ``N`` for the default
            ``eigvalsh`` path).
        already_symmetric: When True, skip the Hermitian-symmetrization
            step (set by callers, e.g. ``hamiltonian_spectral_data``, that
            already enforced it, to avoid a redundant transpose).
        analytic: If True, evaluate a closed-form solution instead of
            ``torch.linalg.eigvalsh``: the Cardano solution for
            ``T.shape[-1] == 3``, or the Ferrari solution for
            ``T.shape[-1] == 4``. Falls back to plain ``eigvalsh`` for any
            other N, since no closed form is defined there.
        c1: Optional precomputed quadratic invariant, forwarded to the
            Cardano/Ferrari path when ``analytic`` is used.
        c0: Optional precomputed cubic invariant, forwarded to the Cardano
            path (N=3).
        e3: Optional precomputed cubic invariant, forwarded to the Ferrari
            path (N=4).
        e4: Optional precomputed quartic invariant, forwarded to the
            Ferrari path (N=4).

    Returns:
        Tensor shaped ``(..., N)`` with the real eigenvalues, in T's dtype.
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

    return eigvalsh_chunked(T).to(dtype=T.dtype)


def _spectral_degeneracy_mask(
    lam: torch.Tensor,
    *,
    relative_eps: float,
    absolute_eps: float,
) -> torch.Tensor:
    """Flag batch entries whose eigenvalue spectrum is (nearly) degenerate.

    Uses a *relative* threshold rather than a fixed absolute one, since the
    dynamic range of lambda spans vacuum splittings (~1e-5) to large
    matter potentials. For each eigenvalue ``lambda_a``, ``gap_a = min_{b
    != a} |lambda_a - lambda_b|`` is compared against ``relative_eps * eps
    * scale``, where ``scale = max_b |lambda_b|`` and ``eps =
    torch.finfo(dtype).eps``. A batch entry is flagged degenerate if any
    eigenvalue fails this test, or if ``scale`` itself is below
    ``absolute_eps * eps`` (the T~0 case, where the relative test
    degenerates to 0/0).

    Args:
        lam: Eigenvalues shaped ``(..., N)`` (only the real part is used).
        relative_eps: Multiplier of ``eps`` bounding the minimum pairwise
            gap relative to the spectrum scale.
        absolute_eps: Multiplier of ``eps`` giving the floor on the
            spectrum scale itself.

    Returns:
        Boolean tensor shaped ``(...,)``, True where the closed-form
        projector formula is not numerically trustworthy for that batch
        entry.
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
    lam: Optional[torch.Tensor] = None,
    c1: Optional[torch.Tensor] = None,
    T2: Optional[torch.Tensor] = None,
    e3: Optional[torch.Tensor] = None,
    T3: Optional[torch.Tensor] = None,
    c0: Optional[torch.Tensor] = None,
    e4: Optional[torch.Tensor] = None,
    *,
    relative_eps: float = _DEGENERACY_RELATIVE_EPS,
    absolute_eps: float = _DEGENERACY_ABSOLUTE_EPS,
    analytic_eigenvalues: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build spectral projectors for a traceless Hamiltonian from eigenvalues and invariants.

    Both formulas are the Lagrange/Sylvester interpolation projector ``M_a
    = prod_{b != a} (T - lambda_b I) / (lambda_a - lambda_b)``,
    algebraically reduced (via Cayley-Hamilton and the tracelessness of T,
    i.e. ``e1 = sum_a lambda_a = 0``) to an explicit polynomial in T with
    coefficients depending only on T, T^2 (and, for N=4, T^3), lambda_a,
    and the trace-derived invariants c1 (= e2 = -tr(T^2)/2) and e3 (=
    tr(T^3)/3) -- never on the individual "other" eigenvalues.

    N=3::

        M_a = [(lambda_a^2 + c1) I + lambda_a T + T^2] / (3 lambda_a^2 + c1)

    N=4::

        M_a = [T^3 + lambda_a T^2 + (c1 + lambda_a^2) T
               + (lambda_a^3 + c1 lambda_a - e3) I]
              / (4 lambda_a^3 + 2 c1 lambda_a - e3)

    In both cases the denominator equals ``p'(lambda_a)``, the derivative
    of the characteristic polynomial at the root -- it vanishes exactly
    when ``lambda_a`` is a repeated eigenvalue (degenerate spectrum). Near
    such a degeneracy the closed-form formula above loses precision well
    before ``_DENOM_EPS`` (a last-resort floor against literal division by
    zero) is reached, so batch entries flagged by ``_spectral_degeneracy_mask``
    have their projectors replaced wholesale by ``M_a = v_a v_a^dagger``
    built from a single ``torch.linalg.eigh(T)`` call -- exact and stable
    by construction, independent of the closed-form formula's
    conditioning. This eigh call is only made when at least one batch
    entry needs it.

    Args:
        T: Traceless Hermitian Hamiltonian shaped ``(..., N, N)``, N in
            ``{3, 4}``.
        lam: Hamiltonian eigenvalues shaped ``(..., N)``.
        c1: Quadratic invariant c1 (= e2) of the traceless Hamiltonian.
        T2: Optional precomputed ``T @ T``, shaped ``(..., N, N)``.
        e3: N=4 only. Third elementary symmetric invariant. Ignored for
            N=3.
        T3: N=4 only. Optional precomputed ``T @ T @ T``. Ignored for N=3.
        c0: N=3 only, and only used when ``analytic_eigenvalues`` is True
            and ``lam`` is None. Optional precomputed cubic invariant,
            forwarded to ``hamiltonian_traceless_eigenvalues``'s Cardano
            path.
        e4: N=4 only, and only used when ``analytic_eigenvalues`` is True
            and ``lam`` is None. Optional precomputed quartic invariant,
            forwarded to ``hamiltonian_traceless_eigenvalues``'s Ferrari
            path.
        relative_eps: Degeneracy-detector threshold, see
            ``_spectral_degeneracy_mask``.
        absolute_eps: Degeneracy-detector scale floor, see
            ``_spectral_degeneracy_mask``.
        analytic_eigenvalues: If True and ``lam`` is None, compute ``lam``
            with the closed-form Cardano (N=3) or Ferrari (N=4) solution.
            Ignored when ``lam`` is supplied directly.

    Returns:
        Tuple ``(M, lam, c1)``: projectors shaped ``(..., N, N, N)``
        (``M[..., a, :, :]`` is ``M_a``), eigenvalues shaped ``(..., N)``,
        and c1 with the batch shape of T.

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
        safe_denom = torch.where(
            denom.abs() < _DENOM_EPS, denom.new_full((), _DENOM_EPS), denom,
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
            denom.abs() < _DENOM_EPS, denom.new_full((), _DENOM_EPS), denom,
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

    degenerate = _spectral_degeneracy_mask(lam, relative_eps=relative_eps, absolute_eps=absolute_eps)
    if bool(torch.any(degenerate)):
        _, V = eigh_chunked(T)
        M_eigh = torch.einsum("...ia,...ja->...aij", V, V.conj()).to(dtype=T.dtype)
        M = torch.where(degenerate[..., None, None, None], M_eigh, M)

    return M, lam, c1


def hamiltonian_spectral_data(
    H: torch.Tensor,
    trace_H: Optional[torch.Tensor] = None,
    *,
    relative_eps: float = _DEGENERACY_RELATIVE_EPS,
    absolute_eps: float = _DEGENERACY_ABSOLUTE_EPS,
    analytic_eigenvalues: bool = False,
) -> dict[str, torch.Tensor]:
    """Return trace, traceless Hamiltonian, eigenvalues, and spectral projectors for H.

    Args:
        H: Hermitian generator shaped ``(..., N, N)``, N in ``{3, 4}``.
        trace_H: Optional precomputed trace, shaped ``(...)``.
        relative_eps: Degeneracy-detector threshold forwarded to
            ``hamiltonian_spectral_projectors_traceless``.
        absolute_eps: Degeneracy-detector scale floor forwarded to
            ``hamiltonian_spectral_projectors_traceless``.
        analytic_eigenvalues: If True, compute the eigenvalues with the
            closed-form Cardano (N=3) or Ferrari (N=4) solution instead of
            ``torch.linalg.eigvalsh``.

    Returns:
        Dict with keys ``"T"`` (traceless Hamiltonian), ``"trace"``,
        ``"lam"`` (eigenvalues), ``"c1"`` (quadratic invariant), and
        ``"M"`` (spectral projectors).
    """
    T, trace_H = hamiltonian_traceless(H, trace_H=trace_H)
    # Enforce Hermitian symmetry once; downstream helpers get
    # already_symmetric=True to skip the redundant transpose.
    T = 0.5 * (T + T.conj().transpose(-1, -2))
    N = T.shape[-1]

    T2 = T @ T
    T3 = T2 @ T if N == 4 else None

    c1 = hamiltonian_traceless_c1(T, T2=T2)
    c0 = hamiltonian_traceless_c0(T) if (analytic_eigenvalues and N == 3) else None
    e3 = hamiltonian_traceless_e3(T, T3=T3) if N == 4 else None
    e4 = hamiltonian_traceless_e4(T, T2=T2) if (analytic_eigenvalues and N == 4) else None
    lam = hamiltonian_traceless_eigenvalues(
        T, already_symmetric=True, analytic=analytic_eigenvalues, c1=c1, c0=c0, e3=e3, e4=e4,
    )
    M, lam, c1 = hamiltonian_spectral_projectors_traceless(
        T, lam=lam, c1=c1, T2=T2, e3=e3, T3=T3, relative_eps=relative_eps, absolute_eps=absolute_eps,
    )

    return {"T": T, "trace": trace_H, "lam": lam, "c1": c1, "M": M}


def spectral_projector_residuals(
    M: torch.Tensor,
    T: torch.Tensor,
    lam: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute diagnostic residual norms for a spectral projector decomposition.

    Every returned quantity should be at or near floating-point precision
    for a numerically healthy decomposition ``T = sum_a lam_a * M_a``. Used
    by this module's own tests as an explicit, reusable set of invariant
    checks.

    Args:
        M: Spectral projectors shaped ``(..., N, N, N)`` (``M[..., a, :,
            :]`` is ``M_a``).
        T: Traceless Hermitian matrix shaped ``(..., N, N)`` that ``M``
            decomposes.
        lam: Eigenvalues shaped ``(..., N)``, paired with the leading M
            index.

    Returns:
        Dict of real-valued residual norms, each shaped ``(...,)``:
            ``"completeness"``: ``||sum_a M_a - I||_F``.
            ``"idempotency"``: ``max_a ||M_a @ M_a - M_a||_F``.
            ``"orthogonality"``: ``max_{a != b} ||M_a @ M_b||_F``.
            ``"trace_one"``: ``max_a |tr(M_a) - 1|`` (each projector has
                rank 1).
            ``"reconstruction"``: ``||sum_a lam_a * M_a - T||_F``.
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

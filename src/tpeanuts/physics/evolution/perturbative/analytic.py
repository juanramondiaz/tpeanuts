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

"""Closed-form (Cardano/Ferrari) evolution: an exact alternative to ``numerical``'s ``matrix_exp``.

Mirrors ``tpeanuts.physics.evolution.numerical.numerical`` function-for-function
(``segment_evolutor_numerical`` <-> ``segment_evolutor_analytic``,
``evolve_path_numerical`` <-> ``evolve_path_analytic``, same shape/signature
conventions), so a caller can swap one module for the other without
touching anything upstream of it -- both take the exact same ``H``/``dx``
this project's ``PropagationModel.generator_reduced``/``generator_flavour``
and ``MediumProfile.length_m``/``evolution_scale_m`` already produce.

The two are not merely two ways of computing the same approximation to the
same thing -- ``segment_evolutor_analytic`` is *exact* for any segment
length, not a series truncation, but only defined for ``H.shape[-1] in {3,
4}`` (see ``tpeanuts.physics.evolution.perturbative.spectral``'s own
module docstring for why: Abel-Ruffini rules out a closed form for degree
>= 5). Outside that domain ``tpeanuts.physics.evolution.numerical.numerical``
remains the only option -- this module does not silently fall back to it;
the underlying ``hamiltonian_spectral_projectors_traceless`` call raises
``ValueError`` immediately for an unsupported dimension.

Lives alongside ``perturbative.py``/``profile.py``/``spectral.py`` under
the same ``perturbative/`` folder rather than its own, since ``analytic``
and ``analytic_perturbative`` are two variants of the same closed-form
technique, not independent methods: this module is deliberately purely
exact/zero-order -- everything specific to the optional first-order
within-segment density-variation correction
(``tpeanuts.physics.evolution.perturbative.profile.EvenPowerProfile``-consuming code)
lives in the sibling ``tpeanuts.physics.evolution.perturbative.perturbative``
module instead, which imports ``_segment_evolutor_from_spectral`` from here to
reuse the exact same zero-order computation (and the single
``hamiltonian_spectral_data`` call both pieces need) without a second,
redundant spectral decomposition.

Module contents:
    segment_evolutor_analytic(...)
        Build ``sum_a exp[-i(lambda_a + Tr(H)/N)dx] M_a`` for one segment
        (or an entire single-segment path) -- the exact zero-order
        (constant-Hamiltonian) evolution operator.
    evolve_path_analytic(...)
        Build and compose the per-segment evolutors for a whole path into
        one total evolution operator, via
        ``tpeanuts.physics.common.composition.compose_segment_evolutors``,
        unchanged and shared with ``numerical.evolve_path_numerical``.
"""

from __future__ import annotations

import torch

from tpeanuts.physics.common.composition import compose_segment_evolutors
from tpeanuts.physics.evolution.perturbative.spectral import hamiltonian_spectral_data
from tpeanuts.util.tensor import TensorLike, as_tensor


def _segment_evolutor_from_spectral(
    spectral: dict,
    dx: TensorLike,
    *,
    device: torch.device,
    real_dtype: torch.dtype,
) -> torch.Tensor:
    """Shared zero-order evolutor assembly, given already-built spectral data.

    Factored out of ``segment_evolutor_analytic`` so
    ``tpeanuts.physics.evolution.perturbative.perturbative.
    segment_evolutor_analytic_perturbative`` can reuse the exact same U0
    computation (and the single ``hamiltonian_spectral_data`` call it
    needs anyway for the first-order correction's own ``M``/``lam``)
    without a second, redundant spectral decomposition. Imported directly
    by that sibling module -- still module-private by convention (leading
    underscore), not part of this module's own public contract, but shared
    within the ``tpeanuts.physics.evolution`` package rather than
    duplicated.

    Args:
        spectral: The dict ``hamiltonian_spectral_data`` returns.
        dx: Dimensionless path length(s).
        device: Device to build ``dx`` on.
        real_dtype: Real dtype to build ``dx`` on (before casting to the
            spectral data's own complex dtype).

    Returns:
        Complex unitary tensor shaped ``(..., N, N)``.
    """
    N = spectral["M"].shape[-1]
    lam = spectral["lam"]
    trace = spectral["trace"].to(dtype=lam.dtype)
    dx_t = as_tensor(dx, device=device, dtype=real_dtype).to(dtype=lam.dtype)
    phase = torch.exp(-1j * (lam + trace[..., None] / N) * dx_t[..., None])
    return (phase[..., :, None, None] * spectral["M"]).sum(dim=-3)


def segment_evolutor_analytic(
    H: torch.Tensor,
    dx: TensorLike,
    *,
    analytic_eigenvalues: bool = True,
) -> torch.Tensor:
    """Build the exact unitary evolution operator for one constant-Hamiltonian segment (or a whole path).

    Args:
        H: Hermitian generator, shaped ``(..., N, N)``, ``N in {3, 4}`` --
            e.g.
            ``tpeanuts.physics.model.PropagationModel.generator_flavour``'s
            output, already including whatever
            ``tpeanuts.physics.extensions`` terms are active (that are
            themselves compatible with this dimension restriction -- see
            ``tpeanuts.physics.evolution``'s own module docstring).
        dx: Dimensionless path length(s), broadcastable against ``H``'s
            batch shape -- see ``tpeanuts.physics.evolution.numerical.
            numerical.segment_evolutor_numerical``'s own docstring for what
            "dimensionless" requires; the convention is identical here.
        analytic_eigenvalues: If True (the default), also use the
            closed-form Cardano (N=3) / Ferrari (N=4) eigenvalue solution
            (see ``tpeanuts.physics.evolution.perturbative.spectral``) rather than
            ``torch.linalg.eigvalsh`` -- the fully closed-form path
            end to end, matching what selecting the analytic evolver is
            meant to give. The spectral *projectors* are always built from
            the closed-form polynomial-in-T formula regardless of this
            flag (that is what makes this exact for any ``dx`` without a
            ``matrix_exp`` call at all); this flag only decides where the
            eigenvalues plugged into that formula come from. Set to False
            to keep the projector formula's exactness while sourcing
            eigenvalues from ``eigvalsh`` instead (matching v1_0's own
            default), e.g. for a direct side-by-side precision comparison.

    Returns:
        Complex unitary tensor shaped ``(..., N, N)``.

    Raises:
        ValueError: If ``H.shape[-1]`` is not 3 or 4 (raised by
            ``tpeanuts.physics.evolution.perturbative.spectral.
            hamiltonian_spectral_projectors_traceless``).
    """
    spectral = hamiltonian_spectral_data(H, analytic_eigenvalues=analytic_eigenvalues)
    return _segment_evolutor_from_spectral(spectral, dx, device=H.device, real_dtype=H.real.dtype)


def evolve_path_analytic(
    H_segments: torch.Tensor,
    dx_segments: TensorLike,
    *,
    reverse: bool = False,
    analytic_eigenvalues: bool = True,
) -> torch.Tensor:
    """Build and compose the per-segment analytic evolutors for a whole path.

    Args:
        H_segments: Hermitian generators, one per segment, shaped
            ``(..., S, N, N)`` (``S`` segments).
        dx_segments: Dimensionless segment lengths, shaped ``(..., S)`` (or
            broadcastable against it).
        reverse: Forwarded to
            ``tpeanuts.physics.common.composition.compose_segment_evolutors``:
            if False (default), segment 0 is the first one a state passes
            through -- identical convention to
            ``tpeanuts.physics.evolution.numerical.numerical.evolve_path_numerical``.
        analytic_eigenvalues: Forwarded to ``segment_evolutor_analytic``.

    Returns:
        Complex unitary tensor shaped ``(..., N, N)``, the total evolution
        operator for the whole path.

    Raises:
        ValueError: If ``H_segments.shape[-1]`` is not 3 or 4.
    """
    S_segments = segment_evolutor_analytic(
        H_segments, dx_segments, analytic_eigenvalues=analytic_eigenvalues
    )
    return compose_segment_evolutors(S_segments, reverse=reverse)

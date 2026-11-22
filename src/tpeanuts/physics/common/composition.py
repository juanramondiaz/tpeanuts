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

"""Composition of per-segment evolution operators into one total operator.

Dimension-agnostic on purpose: this module has no idea whether the ``d x d``
operators it composes are unitary generators' evolutors (``d =
mixing.space.n_flavour``, or ``2 * n_flavour`` once a magnetic-moment
extension doubles the state space) or Liouville-space superoperators for
open-system/decoherence evolution (``d = n_flavour**2``) -- it is exactly
the same tree-reduction either way, so this lives in ``physics.common`` and
is shared, unchanged, by every entry in ``tpeanuts.physics.evolution``.

Module contents:
    compose_segment_evolutors(...)
        Compose ordered per-segment evolution operators into one total
        operator, via ``tpeanuts.util.math.tree_reduce_matmul``.
"""

from __future__ import annotations

import torch

from tpeanuts.util.math import tree_reduce_matmul


def compose_segment_evolutors(
    U_segments: torch.Tensor,
    *,
    segment_dim: int = -3,
    reverse: bool = False,
) -> torch.Tensor:
    """Compose ordered per-segment evolution operators into one total operator.

    Args:
        U_segments: Segment operators shaped ``(..., N, d, d)`` for any
            ``d`` -- a unitary generator's evolutor or a Liouville-space
            superoperator alike (see the module docstring).
        segment_dim: Axis enumerating segments in propagation order (index 0
            is the first segment a state passes through).
        reverse: If False (default), ``U_segments[0]`` is applied first to a
            state -- the standard convention for operators acting on kets
            (or vectorized density matrices) from the left:
            ``S_total = U[N-1] @ ... @ U[1] @ U[0]``. If True, the
            composition order is reversed (``U[0]`` applied last):
            ``S_total = U[0] @ U[1] @ ... @ U[N-1]``.

    Returns:
        Total evolution operator shaped ``(..., d, d)``.

    Raises:
        ValueError: If ``U_segments``'s final two dimensions are not square.
    """
    if U_segments.shape[-1] != U_segments.shape[-2]:
        raise ValueError("U_segments must have final shape (..., d, d) with a square matrix.")

    U_segments = torch.movedim(U_segments, segment_dim, -3)
    return tree_reduce_matmul(U_segments, left=reverse)

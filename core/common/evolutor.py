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

"""Generic evolution-operator utilities.

Module functions:
    apply_evolutor_to_state(...)
        Apply a scalar or batched evolution operator to a state vector.
    compose_segment_evolutors(...)
        Compose ordered segment operators into a total evolutor.
"""

from __future__ import annotations

import torch

from tpeanuts.util.math import tree_reduce_matmul


def apply_evolutor_to_state(
    evolutor: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    """Apply a batched evolution operator to a state with final dimension 3.

    Args:
        evolutor: Evolution operator shaped (..., 3, 3).
        state: State amplitudes shaped (..., 3).

    Returns:
        Evolved state amplitudes with final dimension 3.
    """
    if evolutor.shape[-2:] != (3, 3):
        raise ValueError("evolutor must have final dimensions (3, 3).")
    if state.shape[-1] != 3:
        raise ValueError("state must have last dimension 3.")

    while state.ndim < evolutor.ndim - 1:
        state = state.unsqueeze(-2)

    return torch.matmul(evolutor, state[..., None]).squeeze(-1)


@torch.no_grad()
def compose_segment_evolutors(
    U_segments: torch.Tensor,
    *,
    segment_dim: int = -3,
    multiply: str = "left",
) -> torch.Tensor:
    """Compose ordered segment evolutors into one evolution operator.

    The product is computed via a binary-tree reduction (see
    ``tpeanuts.util.math.tree_reduce_matmul``): matrices are paired and
    multiplied at each level, halving the stack per level.  This exposes
    O(log N) levels of data-parallel batched matmuls on GPU instead of the
    O(N) sequential kernel launches of a plain loop.

    Args:
        U_segments: Segment operators shaped (..., N, d, d) for any flavour
            count d (3 for the SM, 4 for 3+1 sterile extensions, ...).
        segment_dim: Axis enumerating segments in propagation order.
        multiply: ``"left"`` accumulates ``U_seg @ U_total`` (each new segment
            is applied on the left); ``"right"`` accumulates the reverse order.

    Returns:
        Total evolution operator shaped (..., d, d).
    """
    if U_segments.shape[-1] != U_segments.shape[-2]:
        raise ValueError("U_segments must have final shape (..., d, d) with a square matrix.")

    if multiply not in ("left", "right"):
        raise ValueError("multiply must be either 'left' or 'right'.")

    U_segments = torch.movedim(U_segments, segment_dim, -3)

    # Historical compose_segment_evolutors semantics:
    #   multiply="left"  -> U_total = U_seg @ U_total -> reversed product
    #   multiply="right" -> U_total = U_total @ U_seg -> left-to-right product
    # tree_reduce_matmul(left=True) evaluates the left-to-right product, so the
    # flag is intentionally opposite to the accumulation name.
    return tree_reduce_matmul(U_segments, left=(multiply == "right"))

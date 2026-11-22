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

"""Medium-independent numerical (matrix-exponential) evolution.

This module knows nothing about mixing angles, flavour counts, or matter
potentials: it takes an already-built generator ``H`` (Hermitian, any
``d x d``, from ``tpeanuts.physics.common.generator`` plus whatever
``tpeanuts.physics.extensions`` terms are active) and a dimensionless path
length, and produces the unitary evolution operator ``S = matrix_exp(-i H
dx)`` -- exact for any generator, not an approximation, unlike
``tpeanuts.physics.evolution.perturbative.analytic``/``.perturbative.perturbative``,
which are exact resp. approximate only for ``d in {3, 4}`` (see
``tpeanuts.physics.evolution``'s own module docstring). Lives in its own
``numerical/`` folder to match every other propagation strategy, even
though it holds only this one module -- see this package's own
``__init__.py`` docstring for why. The same two
functions compose a whole path from an arbitrary number of segments (each
with its own generator sampled at that point in the medium), via
``tpeanuts.physics.common.composition.compose_segment_evolutors``.

Applying the resulting operator to a state is not this module's job either:
``tpeanuts.physics.common.state.DensityMatrix.conjugate_by`` already does
exactly that, generally (working equally for a pure or mixed input state),
so nothing here duplicates it.

Module contents:
    segment_evolutor_numerical(...)
        Build ``matrix_exp(-i H dx)`` for one segment (or an entire single-
        segment path).
    evolve_path_numerical(...)
        Build and compose the per-segment evolutors for a whole path into
        one total evolution operator.
"""

from __future__ import annotations

import torch

from tpeanuts.physics.common.composition import compose_segment_evolutors
from tpeanuts.util.tensor import TensorLike, as_tensor


def segment_evolutor_numerical(H: torch.Tensor, dx: TensorLike) -> torch.Tensor:
    """Build the exact unitary evolution operator for one segment (or a whole path).

    Args:
        H: Hermitian generator, shaped ``(..., d, d)`` -- e.g.
            ``tpeanuts.physics.model.PropagationModel.generator_flavour``'s
            output, already including whatever
            ``tpeanuts.physics.extensions`` terms are active.
        dx: Dimensionless path length(s), broadcastable against ``H``'s
            batch shape. Dimensionless because ``H`` itself was built
            non-dimensionalized by some ``evolution_scale_m``
            (``tpeanuts.physics.common.generator``'s default is the Earth
            radius): ``dx`` must be the physical path length divided by
            that exact same scale, or the two will not correspond to the
            same physical propagation.

    Returns:
        Complex unitary tensor shaped ``(..., d, d)``:
        ``matrix_exp(-i * H * dx)``.
    """
    dx_t = as_tensor(dx, device=H.device, dtype=H.real.dtype)
    phase = (-1j * dx_t.to(dtype=H.dtype))[..., None, None] * H
    return torch.linalg.matrix_exp(phase)


def evolve_path_numerical(
    H_segments: torch.Tensor,
    dx_segments: TensorLike,
    *,
    reverse: bool = False,
) -> torch.Tensor:
    """Build and compose the per-segment evolutors for a whole path.

    Args:
        H_segments: Hermitian generators, one per segment, shaped
            ``(..., N, d, d)``.
        dx_segments: Dimensionless segment lengths, shaped ``(..., N)`` (or
            broadcastable against it) -- see ``segment_evolutor_numerical``
            for what "dimensionless" requires.
        reverse: Forwarded to
            ``tpeanuts.physics.common.composition.compose_segment_evolutors``:
            if False (default), segment 0 is the first one a state passes
            through.

    Returns:
        Complex unitary tensor shaped ``(..., d, d)``, the total evolution
        operator for the whole path.
    """
    S_segments = segment_evolutor_numerical(H_segments, dx_segments)
    return compose_segment_evolutors(S_segments, reverse=reverse)

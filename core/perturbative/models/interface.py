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

"""Common interface objects for perturbative density profiles.

The dataclasses defined here are intentionally model-neutral. Medium modules
use them to pass segment geometry around without inspecting the internal model
data used to build perturbative segment profiles.

Module classes:
    PerturbativeSegmentBatch
        Batch of ordered trajectory segments with opaque model data.
    PerturbativeOuterSegment
        Metadata for the outermost crossed trajectory segment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class PerturbativeSegmentBatch:
    """Ordered trajectory segments plus opaque model data.

    Attributes:
        x1: Initial coordinate of each segment in the coordinate system
            expected by the perturbative segment model. Shape is typically
            ``(..., n_segments)``.
        x2: Final coordinate of each segment, broadcast-compatible with
            ``x1``. The sign and ordering encode the physical path direction
            chosen by the medium evolutor.
        crossed: Boolean mask selecting the segments that are physically
            crossed by each trajectory. Non-crossed entries are kept so callers
            can use rectangular tensors and replace them with identity
            evolutors after model construction.
        model_data: Model-specific payload needed to build local perturbative
            segment profiles. Medium code must treat this as opaque. For the
            even-power model this contains coefficient vectors, but another
            model may store interpolation nodes, spline parameters, or any
            other internal representation.
    """

    x1: torch.Tensor
    x2: torch.Tensor
    crossed: torch.Tensor
    model_data: Any


@dataclass
class PerturbativeOuterSegment:
    """Outermost crossed segment metadata plus opaque model data.

    Attributes:
        x_start: Coordinate where the detector-side segment starts. For Earth
            propagation this is usually the previous crossed shell boundary, or
            zero when only one shell is crossed.
        model_data: Model-specific payload associated with the outermost
            crossed layer. Medium code passes it back to the perturbative model
            but does not inspect its structure.
        has_any: Boolean mask indicating whether each trajectory crosses at
            least one material layer. Evolutors use this to keep no-crossing
            entries as identity operators.
        has_two: Optional boolean mask indicating whether at least two crossed
            layers exist. This is useful when the start coordinate depends on
            the second-outermost crossed layer.
        last_pos: Optional index of the outermost crossed layer in the model's
            layer ordering.
        second_last_pos: Optional index of the crossed layer immediately before
            ``last_pos`` in the same ordering.
    """

    x_start: torch.Tensor
    model_data: Any
    has_any: torch.Tensor
    has_two: torch.Tensor | None = None
    last_pos: torch.Tensor | None = None
    second_last_pos: torch.Tensor | None = None

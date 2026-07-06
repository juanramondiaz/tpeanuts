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

"""Shared geometry containers for numerical propagation.

The numerical core only needs a path grid, dimensionless segment lengths, and
the points where each medium sampled its density. 

Medium-specific modules build these objects from Earth, atmosphere, solar, or 
custom geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import torch

OdeMethod = Literal["midpoint", "left", "right"]


@dataclass
class Trajectory:
    """Numerical path data passed from a medium profile to the core evolutor.

    A trajectory discretizes a neutrino path through a medium into N
    segments delimited by N+1 boundary points along a geometry coordinate
    (e.g. the dimensionless path length x = L / evolution_scale_m). The
    medium-independent numerical evolutor only needs the dimensionless
    segment increments and the points at which to sample the electron
    density on each segment; it does not need to know whether ``x`` is a
    chord length, a radius, or any other medium-specific coordinate.

    Attributes:
        x: Geometry coordinate at each segment boundary, shape ``(N+1,)``.
            Units depend on the medium (e.g. dimensionless path length).
        dx_evolution: Dimensionless evolution-coordinate length of each
            segment, shape ``(N,)``. This is the quantity integrated against
            the Hamiltonian, i.e. ``dx_evolution = dx / evolution_scale_m``.
        sample_x: Coordinate at which each segment's electron density (or
            other medium property) is sampled, shape ``(N,)``. Typically the
            segment midpoint, but may be the left or right endpoint depending
            on the sampling rule used to build the trajectory.
        meta: Medium-specific metadata (e.g. layer indices, crossing flags)
            that the numerical core does not interpret.
    """

    x: torch.Tensor              # Geometry coordinate, shape (N+1,)
    dx_evolution: torch.Tensor   # Evolution increments, shape (N,)
    sample_x: torch.Tensor       # Sampling points, shape (N,)
    meta: dict


def segment_sample_points(
    x: torch.Tensor,
    method: OdeMethod | None = "midpoint",
) -> torch.Tensor:
    """Return per-segment sample points along the last dimension of ``x``.

    Given N+1 segment boundary coordinates, this picks one representative
    point per segment at which to evaluate the electron density (or another
    medium property) for the numerical evolutor. This is a quadrature-rule
    choice: ``"midpoint"`` samples the segment centre (second-order accurate
    for a piecewise-constant-density approximation), while ``"left"``/
    ``"right"`` sample the segment endpoints (first-order, useful for
    matching legacy or directional sampling conventions).

    Args:
        x: Segment boundary coordinates shaped ``(..., N+1)``.
        method: Sampling rule. ``"midpoint"`` (default, same as ``None``)
            returns ``0.5*(x[i] + x[i+1])``; ``"left"`` returns ``x[i]``;
            ``"right"`` returns ``x[i+1]``.

    Returns:
        Per-segment sample coordinates shaped ``(..., N)``.

    Raises:
        ValueError: If ``method`` is not one of ``None``, ``"midpoint"``,
            ``"left"``, or ``"right"``.
    """
    if method in (None, "midpoint"):
        return 0.5 * (x[..., :-1] + x[..., 1:])
    if method == "left":
        return x[..., :-1]
    if method == "right":
        return x[..., 1:]

    raise ValueError("method must be None, 'midpoint', 'left' or 'right'.")

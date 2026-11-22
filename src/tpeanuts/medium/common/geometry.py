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

"""A straight chord between two concentric spheres, shared by Earth and atmosphere geometry.

``chord_length_to_radius_fraction`` is the one closed-form building block
underneath both ``medium.earth.geometry.earth_chord_length_m`` (a chord
from a detector at radius fraction ``r_d`` out to the Earth's own surface,
``r_target_fraction=1``) and ``medium.atmosphere.geometry.
atmosphere_chord_length_m`` (a chord from the surface, ``r_start_fraction=1``,
out to a production shell at ``r_target_fraction=1+h_m/evolution_scale_m``)
-- confirmed algebraically identical to v1_0's own two independent
formulas (``medium/earth/geometry.py``'s ``chord_length_case_b`` and
``medium/atmosphere/geometry.py``'s ``total_path_length``/
``underground_path_length``, the latter after substituting v1_0's
detector-zenith ``theta`` for this project's own nadir ``eta_rad`` via
``cos(theta) = -cos(eta_rad)``, ``sin(theta) = sin(eta_rad)``) -- one primitive,
not two coincidentally-similar ones.

This function carries no domain validation of its own: what range of
``eta_rad`` is physically meaningful depends on the two radii involved (e.g.
Earth's own case A/B split when ``r_start_fraction=1``, vs. atmosphere's
unrestricted ``[0, pi]`` when ``r_start_fraction=1`` and
``r_target_fraction>1``), so each caller validates against its own known
domain.

Module contents:
    chord_length_to_radius_fraction(...)
        Length of the straight chord from a point at radius fraction
        ``r_start_fraction``, in the direction ``eta_rad`` from that point's
        own local vertical, to where it crosses radius fraction
        ``r_target_fraction``.
"""

from __future__ import annotations

from typing import Optional

import torch

from tpeanuts.util.constant import R_E
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import TensorLike, as_tensor, infer_device_dtype


def chord_length_to_radius_fraction(
    eta_rad: TensorLike,
    r_start_fraction: TensorLike,
    r_target_fraction: TensorLike,
    *,
    evolution_scale_m: TensorLike = R_E,
    context: Optional[RuntimeContext] = None,
) -> torch.Tensor:
    """Length of the chord from ``r_start_fraction`` to ``r_target_fraction``, at nadir angle ``eta_rad``.

    ``L = evolution_scale_m * (r_start_fraction*cos(eta_rad) +
    sqrt(r_target_fraction**2 - r_start_fraction**2 * sin(eta_rad)**2))`` --
    the discriminant is clamped to zero rather than left to produce NaN
    from floating-point noise exactly at its own zero (a real boundary
    case for Earth's own ``depth_m=0`` chord, ``r_start_fraction=
    r_target_fraction=1``, ``eta_rad`` approaching ``pi/2``).

    Args:
        eta_rad: Nadir angle, in radians, measured from ``r_start_fraction``'s
            own local vertical.
        r_start_fraction: Radius fraction (``r / evolution_scale_m``) of
            the chord's own starting point.
        r_target_fraction: Radius fraction of the sphere the chord is
            measured out to.
        evolution_scale_m: Positive length scale in metres the two radius
            fractions are expressed relative to.
        context: Optional runtime device/dtype. When omitted, both are
            inferred from the tensor inputs.

    Returns:
        Chord length in metres, broadcast shape of ``eta_rad``/
        ``r_start_fraction``/``r_target_fraction``.
    """
    if context is not None:
        device, dtype = context.device, context.dtype
    else:
        device, dtype = infer_device_dtype(eta_rad, r_start_fraction, r_target_fraction, evolution_scale_m)

    eta_t = as_tensor(eta_rad, device=device, dtype=dtype)
    r_start = as_tensor(r_start_fraction, device=device, dtype=dtype)
    r_target = as_tensor(r_target_fraction, device=device, dtype=dtype)
    scale = as_tensor(evolution_scale_m, device=device, dtype=dtype)

    discriminant = r_target**2 - r_start**2 * torch.sin(eta_t) ** 2
    return scale * (r_start * torch.cos(eta_t) + torch.sqrt(discriminant.clamp_min(0.0)))

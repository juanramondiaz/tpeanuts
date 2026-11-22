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

"""The straight-line path from a detector (at any depth) up through the
atmosphere to a production altitude.

A neutrino produced at altitude ``h_m`` above the Earth's surface,
arriving at a detector at depth ``depth_m`` below the surface (``depth_m
= 0`` is the surface itself, this package's original scope, fully
backward compatible) at nadir angle ``eta_rad`` (the same convention
``tpeanuts.medium.earth.geometry`` uses: ``eta_rad=0`` is straight down
toward Earth's centre, ``eta_rad=pi`` is straight up), traveled a straight
chord from the detector's own radius out to the production shell (radius
``evolution_scale_m + h_m`` -- production height is always measured
above the *surface*, never above the detector). ``atmosphere_chord_length_m``
is the *atmosphere-only* portion of that chord's own length (excluding
the underground detector-to-surface portion, which
``tpeanuts.medium.earth.earth.EarthMedium`` covers separately -- see this
module's own docstring further down for why); ``altitude_along_path_m``
samples altitude at any point along that atmosphere-only portion.

Both build on ``tpeanuts.medium.common.geometry.
chord_length_to_radius_fraction`` -- ported from v1_0's own
``medium/atmosphere/geometry.py`` (``total_path_length``/
``underground_path_length``, in its own ``theta_deg`` convention),
substituting ``cos(theta) = -cos(eta_rad)``, ``sin(theta) = sin(eta_rad)`` (v1_0's
own ``eta_rad = pi - theta``) and confirming algebraically that the result is
exactly the shared chord primitive at two different target radii -- see
that module's own docstring. Unlike ``tpeanuts.medium.earth.geometry``,
this chord needs no case-A/case-B split: with ``h_m > 0`` the production
shell (radius fraction ``1 + h_m/evolution_scale_m``) is always strictly
larger than the detector's own radius fraction (at most 1, the surface)
it starts from, so the underlying discriminant is positive for every
``eta_rad`` in ``[0, pi]`` (confirmed algebraically, not assumed -- see the
module docstring of the implementation plan this was built from).

**Why a buried detector's own underground leg is not built here, even
though ``depth_m`` is accepted**: confirmed directly (reading v1_0's own
``medium/atmosphere/evolutor.py``) that v1_0's atmosphere evolutor
accepts ``depth_km`` for exactly one purpose -- correctly locating where
along the shared straight-line trajectory the atmosphere-only segments
sit (the same ``L_und`` offset ``altitude_along_path_m`` already applies
internally, generalized here to a non-surface detector) -- and *never*
uses it to sample density below the surface. The underground portion is
covered by an entirely separate ``tpeanuts.medium.earth.earth.EarthMedium``
evolutor, composed afterwards (``S_total = S_earth @ S_atmosphere``, see
``tpeanuts.pipeline.atmospheric_neutrino``), exactly mirroring v1_0's own
``pipeline/atmosphere_earth.py``.

The conversions between the surface zenith angle ``alpha`` (the one MCEq
solves at) and the detector zenith angle ``theta``
(``alpha_surface_to_theta_detector``/``theta_detector_to_alpha_surface``)
are shared with the atmospheric flux sources, so they live in
``tpeanuts.util.geometry``, not here: ``tpeanuts.source`` and
``tpeanuts.medium`` never import each other.

Module contents:
    atmosphere_chord_length_m(...)
        Length of the atmosphere-only chord, from the detector's own
        radius fraction (surface, when ``depth_m=0``) to the production
        altitude.
    altitude_along_path_m(...)
        Altitude above the surface at a given distance into that
        atmosphere-only chord.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from tpeanuts.medium.common.geometry import chord_length_to_radius_fraction
from tpeanuts.util.constant import R_E
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.geometry import detector_radius_fraction
from tpeanuts.util.tensor import TensorLike, as_tensor, infer_device_dtype


def _resolved_device_dtype(context: Optional[RuntimeContext], *tensors: TensorLike):
    if context is not None:
        return context.device, context.dtype
    return infer_device_dtype(*tensors)


def atmosphere_chord_length_m(
    eta_rad: TensorLike,
    h_m: TensorLike,
    *,
    depth_m: TensorLike = 0.0,
    evolution_scale_m: TensorLike = R_E,
    context: Optional[RuntimeContext] = None,
) -> torch.Tensor:
    """Length of the atmosphere-only chord, from the detector's own radius to a production altitude ``h_m``.

    ``L = chord_length_to_radius_fraction(eta_rad, r_d, 1 + h_m/evolution_scale_m)
    - chord_length_to_radius_fraction(eta_rad, r_d, 1)`` -- ``r_d =
    detector_radius_fraction(depth_m)`` (1 at the surface, ``depth_m=0``).
    The second term is exactly the underground detector-to-surface chord
    ``tpeanuts.medium.earth.geometry.earth_chord_length_m`` would give
    for this same ``eta_rad``/``depth_m`` -- not called directly here because
    ``earth_chord_length_m`` imposes its own domain restriction
    (``eta_rad<pi/2`` when ``depth_m=0``) that does not apply to this
    atmosphere-only quantity, which needs the full ``[0, pi]`` even for a
    surface detector (see this module's own docstring).

    Args:
        eta_rad: Nadir angle, in radians, at the detector. Valid over the
            full ``[0, pi]`` -- see this module's own docstring for why
            no case split is needed here.
        h_m: Production altitude above the surface, in metres; must be
            strictly positive.
        depth_m: Detector depth below the surface, in metres. Defaults
            to ``0.0`` (a surface detector).
        evolution_scale_m: Positive length scale in metres (the Earth
            radius, physically).
        context: Optional runtime device/dtype. When omitted, both are
            inferred from the tensor inputs.

    Returns:
        Chord length in metres, broadcast shape of ``eta_rad``/``h_m``/``depth_m``.

    Raises:
        ValueError: If any entry of ``eta_rad`` is outside ``[0, pi]``, of
            ``h_m`` is not strictly positive, or (raised by
            ``detector_radius_fraction``) of ``depth_m`` is invalid.
    """
    device, dtype = _resolved_device_dtype(context, eta_rad, h_m, depth_m, evolution_scale_m)
    eta_t = as_tensor(eta_rad, device=device, dtype=dtype)
    h_t = as_tensor(h_m, device=device, dtype=dtype)
    scale = as_tensor(evolution_scale_m, device=device, dtype=dtype)

    if torch.any(eta_t < 0) or torch.any(eta_t > math.pi):
        raise ValueError("eta_rad must satisfy 0 <= eta_rad <= pi.")
    if torch.any(h_t <= 0):
        raise ValueError(
            "h_m must be strictly positive -- a neutrino produced at or below the surface has no atmosphere leg."
        )

    resolved_context = RuntimeContext(device=device, dtype=dtype)
    r_d = detector_radius_fraction(depth_m, evolution_scale_m=scale, context=resolved_context)
    r_production = 1.0 + h_t / scale
    total = chord_length_to_radius_fraction(
        eta_t, r_d, r_production, evolution_scale_m=scale, context=resolved_context
    )
    surface = chord_length_to_radius_fraction(
        eta_t, r_d, 1.0, evolution_scale_m=scale, context=resolved_context
    )
    return total - surface


def altitude_along_path_m(
    s_m: TensorLike,
    eta_rad: TensorLike,
    *,
    depth_m: TensorLike = 0.0,
    evolution_scale_m: TensorLike = R_E,
    context: Optional[RuntimeContext] = None,
) -> torch.Tensor:
    """Altitude above the surface at distance ``s_m`` into the atmosphere-only chord.

    ``s_m=0`` is the point where the trajectory (extended backward from
    the production point) first reaches the surface; ``s_m =
    atmosphere_chord_length_m(eta_rad, h_m, depth_m=depth_m)`` is the
    production altitude itself. This is *not* simply ``r_s**2 = r_d**2 +
    s'**2 - 2*r_d*s'*cos(eta_rad)`` at face value: v1_0's own
    ``altitude_along_detector_path`` measures its own ``s`` from the
    chord's other, underground-facing endpoint (the detector, at radius
    fraction ``r_d = detector_radius_fraction(depth_m)``), and for
    ``eta_rad < pi/2`` that same straight line dips further underground
    first (an ``L_und = chord_length_to_radius_fraction(eta_rad, r_d, 1)`` --
    non-zero for ``eta_rad<pi/2``, exactly zero for ``eta_rad>=pi/2`` -- excursion)
    before ever reaching this atmosphere-only segment. Confirmed directly
    against v1_0's own reference values (running
    ``medium/atmosphere/geometry.py`` standalone): omitting this offset
    silently gives altitudes below the true surface crossing instead of
    raising, so it is applied internally here -- the public ``s_m``
    argument stays the clean
    ``[0, atmosphere_chord_length_m(...)]`` range every caller of this
    atmosphere-only medium actually wants, with v1_0's own detector-frame
    offset kept as this function's private implementation detail rather
    than leaked to every caller.

    Args:
        s_m: Distance into the atmosphere-only chord, in metres;
            ``s_m=0`` is the surface, increasing toward the production
            altitude.
        eta_rad: Nadir angle, in radians, at the detector.
        depth_m: Detector depth below the surface, in metres. Defaults
            to ``0.0`` (a surface detector).
        evolution_scale_m: Positive length scale in metres.
        context: Optional runtime device/dtype. When omitted, both are
            inferred from the tensor inputs.

    Returns:
        Altitude above the surface, in metres, broadcast shape of
        ``s_m``/``eta_rad``/``depth_m``.
    """
    device, dtype = _resolved_device_dtype(context, s_m, eta_rad, depth_m, evolution_scale_m)
    s_t = as_tensor(s_m, device=device, dtype=dtype)
    eta_t = as_tensor(eta_rad, device=device, dtype=dtype)
    scale = as_tensor(evolution_scale_m, device=device, dtype=dtype)
    resolved_context = RuntimeContext(device=device, dtype=dtype)

    r_d = detector_radius_fraction(depth_m, evolution_scale_m=scale, context=resolved_context)
    L_und = chord_length_to_radius_fraction(eta_t, r_d, 1.0, evolution_scale_m=scale, context=resolved_context)
    s_from_detector_prime = (L_und + s_t) / scale
    r_s_fraction = torch.sqrt(
        (r_d**2 + s_from_detector_prime**2 - 2.0 * r_d * s_from_detector_prime * torch.cos(eta_t)).clamp_min(0.0)
    )
    return scale * (r_s_fraction - 1.0)



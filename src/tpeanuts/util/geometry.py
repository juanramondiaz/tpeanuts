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

"""Geometry shared by the Earth and atmosphere media and by the atmospheric flux sources.

The detector radius fraction and the conversion between a detector zenith angle and the zenith angle at the Earth's
surface depend only on the straight-line path through concentric spheres, not on any medium or source. They live here
so that ``tpeanuts.medium`` (which builds paths) and ``tpeanuts.source`` (which tabulates fluxes at a surface zenith
angle) can both use them without importing each other.

Module contents:
    detector_radius_fraction(...)
        ``r_d = 1 - depth_m/evolution_scale_m``.
    alpha_surface_to_theta_detector(...)
        Convert a surface (MCEq) zenith angle to its corresponding detector zenith angle, via impact-parameter
        conservation.
    theta_detector_to_alpha_surface(...)
        The inverse conversion.
"""

from __future__ import annotations

from typing import Optional

import torch

from tpeanuts.util.constant import R_E
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import TensorLike, as_tensor, infer_device_dtype


def _resolved_device_dtype(context: Optional[RuntimeContext], *tensors: TensorLike):
    if context is not None:
        return context.device, context.dtype
    return infer_device_dtype(*tensors)


def detector_radius_fraction(
    depth_m: TensorLike,
    *,
    evolution_scale_m: TensorLike = R_E,
    context: Optional[RuntimeContext] = None,
) -> torch.Tensor:
    """``r_d = 1 - depth_m / evolution_scale_m``, the detector's own radius fraction.

    Args:
        depth_m: Detector depth below the surface, in metres; must
            satisfy ``0 <= depth_m < evolution_scale_m`` (a detector at
            or beyond Earth's own centre is not physical).
        evolution_scale_m: Positive length scale in metres -- the Earth
            radius, physically.
        context: Optional runtime device/dtype. When omitted, both are
            inferred from the tensor inputs.

    Returns:
        ``r_d``, dimensionless, same shape as ``depth_m``, in ``(0, 1]``.

    Raises:
        ValueError: If any entry of ``depth_m`` is outside
            ``[0, evolution_scale_m)``.
    """
    device, dtype = _resolved_device_dtype(context, depth_m, evolution_scale_m)
    depth_t = as_tensor(depth_m, device=device, dtype=dtype)
    scale = as_tensor(evolution_scale_m, device=device, dtype=dtype)

    if torch.any(depth_t < 0) or torch.any(depth_t >= scale):
        raise ValueError(
            "depth_m must satisfy 0 <= depth_m < evolution_scale_m (a detector at or beyond "
            "Earth's own centre is not physical)."
        )

    return 1.0 - depth_t / scale


def alpha_surface_to_theta_detector(
    alpha_deg: TensorLike,
    depth_m: TensorLike = 0.0,
    *,
    evolution_scale_m: TensorLike = R_E,
    context: Optional[RuntimeContext] = None,
) -> torch.Tensor:
    """Convert a surface (MCEq) zenith angle to its corresponding detector zenith angle.

    Impact-parameter conservation for the straight-line trajectory
    through concentric spheres: ``r_d sin(theta) = R sin(alpha)``, so
    ``theta = arcsin((R/r_d) sin(alpha))`` -- the ``R`` factor folds
    into the dimensionless ``r_d = detector_radius_fraction(depth_m)``
    already used elsewhere in this module, so the implementation below
    is ``arcsin(sin(alpha) / r_d)``. Ported from v1_0's own
    ``medium/atmosphere/geometry.py::alpha_surface_to_theta_detector``
    -- see this module's own docstring for why this angle pair
    (``alpha``/``theta``, both "0 deg = straight down") is distinct
    from the ``eta_rad`` nadir convention (``eta_rad=pi`` straight down) the
    rest of this module uses.

    A sufficiently grazing ``alpha_deg`` combined with a sufficiently
    deep ``depth_m`` has no solution at all -- the straight line from
    that surface crossing never reaches that deep a detector radius
    (``sin(alpha)/r_d > 1``); this is a genuine geometric impossibility
    (unlike floating-point edge cases at the domain boundary, which
    ``torch.clamp`` alone would silently paper over), so it is raised
    explicitly rather than clamped, deviating deliberately from v1_0's
    own silent-clamp behaviour there.

    Args:
        alpha_deg: Surface zenith angle in degrees, measured relative
            to the local radial direction at the surface crossing
            point; ``alpha=0`` is vertical.
        depth_m: Detector depth below the surface, in metres. Defaults
            to ``0.0`` (a surface detector, where ``theta == alpha``
            exactly since ``r_d == 1``).
        evolution_scale_m: Positive length scale in metres (the Earth
            radius, physically).
        context: Optional runtime device/dtype. When omitted, both are
            inferred from the tensor inputs.

    Returns:
        Detector zenith angle in degrees, broadcast shape of
        ``alpha_deg``/``depth_m``.

    Raises:
        ValueError: If any entry would require ``sin(alpha)/r_d > 1``
            (no straight line reaches that detector depth at that
            surface angle), or (raised by ``detector_radius_fraction``)
            if ``depth_m`` is invalid.
    """
    device, dtype = _resolved_device_dtype(context, alpha_deg, depth_m, evolution_scale_m)
    alpha_t = as_tensor(alpha_deg, device=device, dtype=dtype)
    scale = as_tensor(evolution_scale_m, device=device, dtype=dtype)
    resolved_context = RuntimeContext(device=device, dtype=dtype)

    r_d = detector_radius_fraction(depth_m, evolution_scale_m=scale, context=resolved_context)
    arg = torch.sin(torch.deg2rad(alpha_t)) / r_d
    if torch.any(torch.abs(arg) > 1.0):
        raise ValueError(
            "alpha_surface_to_theta_detector: no straight-line trajectory reaches a "
            f"detector at depth_m={depth_m!r} from a surface zenith angle alpha_deg="
            f"{alpha_deg!r} -- sin(alpha)/r_d exceeds 1."
        )
    return torch.rad2deg(torch.asin(arg))


def theta_detector_to_alpha_surface(
    theta_deg: TensorLike,
    depth_m: TensorLike = 0.0,
    *,
    evolution_scale_m: TensorLike = R_E,
    context: Optional[RuntimeContext] = None,
) -> torch.Tensor:
    """Convert a detector zenith angle to its corresponding surface (MCEq) zenith angle.

    The inverse of ``alpha_surface_to_theta_detector``:
    ``alpha = arcsin(r_d sin(theta))``. Unlike that inverse direction,
    this one always has a solution for any ``theta_deg`` in ``[0,
    180]``: ``r_d <= 1`` and ``sin(theta) <= 1``, so their product never
    exceeds 1 -- no domain check is needed (a tiny ``clamp`` is applied
    purely as floating-point-boundary insurance, not to mask a real
    impossibility). Ported from v1_0's own
    ``medium/atmosphere/geometry.py::theta_detector_to_alpha_surface``.

    Note the resulting ``alpha_deg`` is not automatically inside
    MCEq's own valid ``[0, 90)`` domain -- a detector-frame ``theta_deg``
    at or beyond the horizon can still convert to a mathematically
    valid ``alpha_deg`` that MCEq itself cannot simulate; callers that
    need that guarantee (e.g. ``tpeanuts.source.atmosphere.mceq.
    generate_mceq_flux``) check it themselves.

    Args:
        theta_deg: Detector zenith angle in degrees; ``theta=0`` is
            vertical.
        depth_m: Detector depth below the surface, in metres. Defaults
            to ``0.0``.
        evolution_scale_m: Positive length scale in metres.
        context: Optional runtime device/dtype. When omitted, both are
            inferred from the tensor inputs.

    Returns:
        Surface zenith angle in degrees, broadcast shape of
        ``theta_deg``/``depth_m``.
    """
    device, dtype = _resolved_device_dtype(context, theta_deg, depth_m, evolution_scale_m)
    theta_t = as_tensor(theta_deg, device=device, dtype=dtype)
    scale = as_tensor(evolution_scale_m, device=device, dtype=dtype)
    resolved_context = RuntimeContext(device=device, dtype=dtype)

    r_d = detector_radius_fraction(depth_m, evolution_scale_m=scale, context=resolved_context)
    arg = torch.clamp(r_d * torch.sin(torch.deg2rad(theta_t)), -1.0, 1.0)
    return torch.rad2deg(torch.asin(arg))

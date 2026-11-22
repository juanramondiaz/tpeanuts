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

"""Fit an atmosphere-only trajectory into a few local-polynomial segments.

``tpeanuts.physics.evolution.perturbative.profile.PolynomialProfile`` is pure physics:
given a segment's own local polynomial coefficients, it builds
``average``/``residual_integral``/etc. It knows nothing about how those
coefficients were obtained. This module is the atmosphere-specific
"sampling and fitting" step that produces them -- ported from v1_0's own
split between ``core/perturbative/models/atmosphere/profile_layered.py``
(fitting) and ``profile_segment.py`` (physics), the same split v2 already
draws between this file and ``PolynomialProfile`` itself.

``fit_local_polynomial_coefficients`` is the plain linear-algebra step
(a Vandermonde solve, no geometry): given density sampled at
``degree + 1`` equally-spaced nodes in the local coordinate ``q in
[-1, 1]``, it returns the unique polynomial's own coefficients exactly
reproducing those samples. ``atmosphere_perturbative_profile`` is the
geometry-aware orchestration: split the atmosphere-only chord (``s in
[0, atmosphere_chord_length_m(eta_rad, h_m)]``) into ``n_segments`` equal
pieces, sample electron/neutron density at each segment's own nodes (via
``tpeanuts.medium.atmosphere.geometry.altitude_along_path_m`` --
already atmosphere-local, ``s=0`` at the surface, no detector-frame
offset to re-derive here -- and
``tpeanuts.medium.atmosphere.density.atmosphere_density``), fit both, and
return the resulting ``PolynomialProfile``.

Unlike Earth's ``even_power``/shell-crossing machinery, the segment count
here is a fixed user choice, not derived from ``eta_rad``/``depth_m`` -- so,
unlike ``tpeanuts.medium.earth.geometry.crossed_shell_segments`` (and
unlike ``tpeanuts.medium.earth.earth.EarthMedium`` itself, which requires
scalar ``eta_rad``/``depth_m`` as soon as ``depth_m > 0``), nothing here
requires either to be scalar; a batch of nadir angles and/or detector
depths fits one independent set of segments per entry (confirmed by this
module's own tests, not just expected).

Module contents:
    fit_local_polynomial_coefficients(...)
        Solve the Vandermonde system mapping density samples at
        equally-spaced local nodes to polynomial coefficients.
    atmosphere_perturbative_profile(...)
        Build the fitted ``PolynomialProfile`` for an atmosphere-only
        trajectory.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from tpeanuts.medium.atmosphere.density import atmosphere_density
from tpeanuts.medium.atmosphere.geometry import altitude_along_path_m, atmosphere_chord_length_m
from tpeanuts.physics.evolution.perturbative.profile import PolynomialProfile
from tpeanuts.util.constant import R_E
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import TensorLike, as_tensor


def fit_local_polynomial_coefficients(density_nodes: torch.Tensor) -> torch.Tensor:
    """Solve for local polynomial coefficients exactly reproducing ``density_nodes`` at equally-spaced nodes.

    ``density_nodes[..., j]`` is taken as the density at ``q_j =
    linspace(-1, 1, n_nodes)[j]``; the returned coefficients ``c`` satisfy
    ``sum_k c[..., k] * q_j**k == density_nodes[..., j]`` exactly, for
    every node ``j``, via ``c = Vandermonde(q)^-1 @ density_nodes``.

    Args:
        density_nodes: Density samples, shaped ``(..., n_nodes)`` --
            ``n_nodes = degree + 1``.

    Returns:
        Local polynomial coefficients, shaped ``(..., n_nodes)``, ordered
        ``q**0, q**1, ..., q**(n_nodes-1)``.

    Raises:
        ValueError: If ``density_nodes``'s last dimension is empty.
    """
    n_nodes = density_nodes.shape[-1]
    if n_nodes < 1:
        raise ValueError("density_nodes must have at least one node (its own last dimension).")

    q = (
        torch.zeros(1, device=density_nodes.device, dtype=density_nodes.dtype)
        if n_nodes == 1
        else torch.linspace(-1.0, 1.0, n_nodes, device=density_nodes.device, dtype=density_nodes.dtype)
    )
    vandermonde = torch.vander(q, N=n_nodes, increasing=True)
    return density_nodes @ torch.linalg.inv(vandermonde).T


def atmosphere_perturbative_profile(
    eta_rad: TensorLike,
    h_m: TensorLike,
    *,
    n_segments: int,
    degree: int,
    density_source: str,
    density_kwargs: Optional[Dict[str, Any]] = None,
    depth_m: TensorLike = 0.0,
    evolution_scale_m: TensorLike = R_E,
    context: RuntimeContext,
) -> PolynomialProfile:
    """Build the fitted local-polynomial profile for an atmosphere-only trajectory.

    Args:
        eta_rad: Nadir angle, in radians, at the detector. Scalar or any
            batch shape -- see this module's own docstring for why no
            scalar restriction is needed here.
        h_m: Production altitude above the surface, in metres.
        n_segments: Number of equal-length segments the atmosphere-only
            chord is split into.
        degree: Local polynomial degree fitted within each segment
            (``degree + 1`` equally-spaced nodes in ``q in [-1, 1]``).
        density_source: Forwarded to ``atmosphere_density``.
        density_kwargs: Forwarded to ``atmosphere_density``.
        depth_m: Detector depth below the surface, in metres. Defaults to
            ``0.0`` (a surface detector). Scalar or any batch shape,
            broadcastable against ``eta_rad`` -- same "no scalar restriction"
            reasoning as ``eta_rad``.
        evolution_scale_m: Positive length scale in metres.
        context: Runtime device/dtype.

    Returns:
        A ``PolynomialProfile`` with ``n_segments`` segments (batched over
        ``eta_rad``/``depth_m``'s own broadcast shape, if any), carrying both
        electron- and neutron-density coefficients.

    Raises:
        ValueError: If ``n_segments < 1`` or ``degree < 0`` (raised
            here), or if ``eta_rad``/``h_m``/``depth_m`` is outside its valid
            domain (raised by ``atmosphere_chord_length_m``).
    """
    if n_segments < 1:
        raise ValueError(f"n_segments must be at least 1, got {n_segments!r}.")
    if degree < 0:
        raise ValueError(f"degree must be non-negative, got {degree!r}.")

    density_kwargs = density_kwargs or {}
    eta_t = as_tensor(eta_rad, device=context.device, dtype=context.dtype)
    depth_t = as_tensor(depth_m, device=context.device, dtype=context.dtype)
    scale = as_tensor(evolution_scale_m, device=context.device, dtype=context.dtype)

    chord_length_m = atmosphere_chord_length_m(
        eta_t, h_m, depth_m=depth_t, evolution_scale_m=scale, context=context
    )
    u = torch.linspace(0.0, 1.0, n_segments + 1, device=context.device, dtype=context.dtype)
    boundaries = (chord_length_m / scale).unsqueeze(-1) * u

    n_nodes = degree + 1
    q = (
        torch.zeros(1, device=context.device, dtype=context.dtype)
        if n_nodes == 1
        else torch.linspace(-1.0, 1.0, n_nodes, device=context.device, dtype=context.dtype)
    )
    centres = 0.5 * (boundaries[..., :-1] + boundaries[..., 1:])
    half_widths = 0.5 * (boundaries[..., 1:] - boundaries[..., :-1])
    x_nodes = centres.unsqueeze(-1) + half_widths.unsqueeze(-1) * q  # (..., n_segments, n_nodes)

    altitude_m = altitude_along_path_m(
        x_nodes * scale,
        eta_t.unsqueeze(-1).unsqueeze(-1),
        depth_m=depth_t.unsqueeze(-1).unsqueeze(-1) if depth_t.ndim > 0 else depth_t,
        context=context,
    )

    density_nodes = atmosphere_density(
        altitude_m, source=density_source, density_type="electron_density", context=context, **density_kwargs
    )
    density_nodes_n = atmosphere_density(
        altitude_m, source=density_source, density_type="neutron_density", context=context, **density_kwargs
    )

    coefficients = fit_local_polynomial_coefficients(density_nodes)
    coefficients_n = fit_local_polynomial_coefficients(density_nodes_n)

    return PolynomialProfile(
        x1=boundaries[..., :-1],
        x2=boundaries[..., 1:],
        coefficients=coefficients,
        coefficients_n=coefficients_n,
        profile_scale_m=scale,
        evolution_scale_m=scale,
        device=context.device,
        dtype=context.dtype,
    )

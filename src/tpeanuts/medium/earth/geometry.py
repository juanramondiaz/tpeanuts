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

"""Chord geometry through a spherical Earth, for a detector at any depth.

For a detector at depth ``depth_m`` below the surface (``depth_m = 0`` is
the surface-detector case this package started with), ``r_d = 1 -
depth_m/evolution_scale_m`` is the detector's own radius fraction, and
``eta_prime = asin(r_d * sin(eta_rad))`` (``eta_prime_from_eta``) is the
nadir angle the *same straight-line trajectory* would have if extended
to the surface -- the trajectory's true point of closest approach to
Earth's centre sits at radius ``sin(eta_prime)``, confirmed algebraically
(``detector_x_coordinate_m(eta_rad, r_d) == evolution_scale_m * sqrt(r_d**2 -
sin(eta_prime)**2)`` exactly), so it is ``eta_prime`` -- not ``eta_rad`` --
that must be used wherever a shell-crossing or ``r(x)`` calculation is
needed.

Two geometric cases, ported from v1_0's own ``medium/earth/geometry.py``:

    Case A (``eta_rad < pi/2``, "Earth-crossing")
        The chord runs from the far-side surface entry point (``x1 =
        -evolution_scale_m * cos(eta_prime)``, using ``eta_prime``
        because the outermost crossed shell is always the surface,
        ``rj=1``) to the detector itself (``x2 = detector_x_coordinate_m(
        eta_rad, r_d)``) -- it does *not* continue past the detector back
        out to the near-side surface. With ``depth_m=0`` (``r_d=1``,
        ``eta_prime=eta_rad``) this collapses exactly to the symmetric
        ``[-L/2, L/2]`` chord this package started with -- a special
        case, not a different formula.
    Case B (``eta_rad >= pi/2``, only reachable when ``depth_m > 0``)
        A short, shallow path between the surface and a buried
        detector -- v1_0 approximates its density as *constant*
        (sampled once at the path's own midpoint radius), not with a
        real ``r(x)`` relationship, since there is no meaningful "point
        of closest approach to the centre" for a path this shallow.
        Handled entirely in ``tpeanuts.medium.earth.earth.EarthMedium``
        (a single constant-density segment for every
        ``PropagationMethod``), not in this module.

**The two cases share one chord-length formula.** Confirmed
algebraically: case A's own length (``x2 - x1`` above) equals v1_0's
``chord_length_case_b`` term for term --

    L(eta_rad, r_d) = evolution_scale_m * (r_d*cos(eta_rad) + sqrt(1 - r_d**2 * sin(eta_rad)**2))

-- positive throughout ``eta_rad in [0, pi]`` whenever ``r_d < 1`` (checked
at both endpoints), and only over ``eta_rad in [0, pi/2)`` when ``r_d = 1``
(``depth_m=0``) -- the same domain restriction this package always had,
now derived rather than assumed. ``earth_chord_length_m`` implements
this single formula for both cases; case-specific handling (segment
boundaries, densities) lives elsewhere.

``crossed_shell_segments`` builds the few, PREM-shell-aligned segment
boundaries ``EarthMedium``'s ``PropagationMethod.ANALYTIC_PERTURBATIVE``
path needs for case A, in place of ``NUMERICAL``/``ANALYTIC``'s many
near-equal-length ones: for ``m`` shells crossed (using ``eta_prime``),
the chord's own point-of-closest-approach symmetry (``r(x) = r(-x)``)
gives the innermost crossed shell one segment spanning ``x=0``, and
every shell further out two (entry- and exit-side) -- ``2*m - 1``
segments, *unless* the detector itself sits inside one of those shells
(``depth_m > 0``), in which case the path stops at the detector rather
than at that shell's own natural outer boundary: the final (or, if the
detector is in the innermost shell, the *only* inner) segment is
clipped at ``detector_x_coordinate_m(eta_rad, r_d)`` instead. With
``depth_m=0`` the detector always sits exactly at the outermost crossed
shell's own boundary, so no clipping ever occurs -- bit-for-bit the same
``2*m-1`` unclipped segments as before.

Module contents:
    eta_prime_from_eta(...)
        The surface-equivalent nadir angle for a buried detector.
    earth_chord_length_m(...)
        Full chord length through the Earth, case A and case B alike.
    detector_x_coordinate_m(...)
        The detector's own chord coordinate.
    radius_fraction_along_chord(...)
        ``r / evolution_scale_m`` at a point along the chord.
    crossed_shell_segments(...)
        The (``<= 2*m - 1``) PREM-shell-aligned, detector-depth-aware
        segment boundaries for one (scalar) nadir angle -- kept as a
        simple reference oracle; ``EarthMedium`` itself uses the
        batched form below.
    crossed_shell_segments_batched(...)
        The batched counterpart: fixed ``2*n_shells - 1`` slots per
        element, padded with zero-length (physically inert) segments,
        so ``eta_rad`` can carry any batch shape.
    aligned_chord_segments(...)
        ``nsteps`` segments of a chord, cut at the radii of the density
        jumps of a table, for the methods that sample the density at
        segment midpoints.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

import tpeanuts.util.constant as constant
from tpeanuts.medium.common.geometry import chord_length_to_radius_fraction
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.geometry import detector_radius_fraction
from tpeanuts.util.math import searchsorted_contiguous
from tpeanuts.util.tensor import TensorLike, as_tensor, infer_device_dtype


def _resolved_device_dtype(context: Optional[RuntimeContext], *tensors: TensorLike):
    if context is not None:
        return context.device, context.dtype
    return infer_device_dtype(*tensors)


def eta_prime_from_eta(eta_rad: TensorLike, r_d: TensorLike) -> torch.Tensor:
    """The surface-equivalent nadir angle, ``eta_prime = asin(r_d * sin(eta_rad))``.

    Args:
        eta_rad: Detector nadir angle, in radians.
        r_d: Detector radius fraction (see ``detector_radius_fraction``),
            broadcastable against ``eta_rad``.

    Returns:
        ``eta_prime``, in radians, broadcast shape of ``eta_rad``/``r_d``.
    """
    eta_t = torch.as_tensor(eta_rad)
    return torch.asin(torch.as_tensor(r_d, dtype=eta_t.dtype, device=eta_t.device) * torch.sin(eta_t))


def earth_chord_length_m(
    eta_rad: TensorLike,
    *,
    depth_m: TensorLike = 0.0,
    evolution_scale_m: TensorLike = constant.R_E,
    context: Optional[RuntimeContext] = None,
) -> torch.Tensor:
    """Full chord length through the Earth, for a detector at any depth (case A and case B alike).

    ``L = evolution_scale_m * (r_d*cos(eta_rad) + sqrt(1 - r_d**2 * sin(eta_rad)**2))``
    -- see this module's own docstring for why one formula covers both
    the Earth-crossing (``eta_rad < pi/2``) and shallow-near-detector
    (``eta_rad >= pi/2``) cases. This is exactly
    ``tpeanuts.medium.common.geometry.chord_length_to_radius_fraction``
    at ``r_target_fraction=1`` (the Earth's own surface) -- delegated to
    it below, not reimplemented, since
    ``tpeanuts.medium.atmosphere.geometry`` needs the identical algebra
    for a different target radius (the production shell).

    Args:
        eta_rad: Nadir angle, in radians; scalar or tensor, broadcastable
            against ``depth_m``/``r_d``. Per element, the valid domain
            is ``[0, pi/2)`` where ``depth_m == 0`` (a surface detector
            has nothing to sample for ``eta_rad >= pi/2`` -- use
            ``tpeanuts.medium.vacuum.vacuum.VacuumMedium`` instead of
            reinterpreting it here), or ``[0, pi]`` where ``depth_m >
            0`` -- both may be batched together, mixing surface and
            buried entries in the same call.
        depth_m: Detector depth below the surface, in metres (see
            ``detector_radius_fraction``).
        evolution_scale_m: Positive length scale in metres.
        context: Optional runtime device/dtype. When omitted, both are
            inferred from the tensor inputs.

    Returns:
        Chord length in metres, broadcast shape of ``eta_rad``/``depth_m``.

    Raises:
        ValueError: If any entry of ``eta_rad`` is outside the valid domain
            for its own (broadcast) ``depth_m``/``r_d`` entry, or
            (raised by ``detector_radius_fraction``) if ``depth_m`` is
            invalid.
    """
    device, dtype = _resolved_device_dtype(context, eta_rad, depth_m, evolution_scale_m)
    eta_t = as_tensor(eta_rad, device=device, dtype=dtype)
    scale = as_tensor(evolution_scale_m, device=device, dtype=dtype)
    r_d = detector_radius_fraction(depth_m, evolution_scale_m=scale, context=context)

    # Per-element domain check, buried (r_d != 1) and surface (r_d == 1)
    # entries validated independently, then broadcast together -- eta_rad
    # and r_d need not share a shape, and either (or both) may carry a
    # batch axis. Confirmed by direct experiment (see
    # [[reference_batching_axis_collision]]'s own discipline) that the
    # underlying formula below, delegated unchanged to
    # chord_length_to_radius_fraction, already broadcasts eta_rad/r_d on
    # independent axes correctly -- this was previously gated by an
    # unconditional 0-dim requirement that was never actually required
    # by the formula itself, only by this validation.
    buried = torch.broadcast_to(r_d != 1.0, torch.broadcast_shapes(eta_t.shape, r_d.shape))
    eta_b = torch.broadcast_to(eta_t, buried.shape)
    valid_buried = (eta_b >= 0.0) & (eta_b <= math.pi)
    valid_surface = (eta_b >= 0.0) & (eta_b < math.pi / 2)
    if not torch.all(torch.where(buried, valid_buried, valid_surface)):
        raise ValueError(
            "eta_rad must satisfy 0 <= eta_rad <= pi where depth_m > 0, or 0 <= eta_rad < pi/2 where "
            "depth_m == 0 (for eta_rad >= pi/2 at depth_m == 0 the path never crosses the Earth at "
            "all, use tpeanuts.medium.vacuum.vacuum.VacuumMedium instead of reinterpreting it here)."
        )

    return chord_length_to_radius_fraction(
        eta_t, r_d, 1.0, evolution_scale_m=scale, context=RuntimeContext(device=device, dtype=dtype)
    )


def detector_x_coordinate_m(
    eta_rad: TensorLike,
    r_d: TensorLike,
    *,
    evolution_scale_m: TensorLike = constant.R_E,
    context: Optional[RuntimeContext] = None,
) -> torch.Tensor:
    """The detector's own chord coordinate, ``x_d = evolution_scale_m * r_d * cos(eta_rad)``.

    Case A's chord runs from the far-side surface entry point to exactly
    this coordinate (see this module's own docstring) -- not, in
    general, to the outermost crossed shell's own natural boundary.

    Args:
        eta_rad: Nadir angle, in radians.
        r_d: Detector radius fraction, broadcastable against ``eta_rad``.
        evolution_scale_m: Positive length scale in metres.
        context: Optional runtime device/dtype. When omitted, both are
            inferred from the tensor inputs.

    Returns:
        ``x_d`` in metres, broadcast shape of ``eta_rad``/``r_d``.
    """
    device, dtype = _resolved_device_dtype(context, eta_rad, r_d, evolution_scale_m)
    eta_t = as_tensor(eta_rad, device=device, dtype=dtype)
    r_d_t = as_tensor(r_d, device=device, dtype=dtype)
    scale = as_tensor(evolution_scale_m, device=device, dtype=dtype)

    return scale * r_d_t * torch.cos(eta_t)


def radius_fraction_along_chord(
    x_m: TensorLike,
    eta_rad: TensorLike,
    *,
    evolution_scale_m: TensorLike = constant.R_E,
    context: Optional[RuntimeContext] = None,
) -> torch.Tensor:
    """Radius fraction ``r / evolution_scale_m`` at chord coordinate ``x_m``.

    ``x_m`` is measured from the point of closest approach to the centre
    (``x_m = 0``, where ``r/evolution_scale_m = sin(eta_rad)``, the chord's own
    minimum radius) -- not from the entry point. For a buried detector
    (``depth_m > 0``, case A only -- case B has no ``r(x)`` relationship
    at all, see this module's own docstring), the caller must pass
    ``eta_prime`` (see ``eta_prime_from_eta``) here, not the raw
    detector-frame ``eta_rad``; for a surface detector they are identical.

    Args:
        x_m: Chord coordinate, in metres, any shape broadcastable against
            ``eta_rad``.
        eta_rad: The angle defining this chord's point of closest approach --
            ``eta_prime``, not the raw detector-frame ``eta_rad``, whenever
            ``depth_m > 0`` (identical to it when ``depth_m = 0``). Not
            revalidated here -- see ``earth_chord_length_m`` for the
            domain check, already applied by whatever built ``x_m``'s
            own range.
        evolution_scale_m: Positive length scale in metres.
        context: Optional runtime device/dtype. When omitted, both are
            inferred from the tensor inputs.

    Returns:
        ``r / evolution_scale_m``, dimensionless, broadcast shape of
        ``x_m``/``eta_rad``.
    """
    device, dtype = _resolved_device_dtype(context, x_m, eta_rad, evolution_scale_m)
    x_t = as_tensor(x_m, device=device, dtype=dtype)
    eta_t = as_tensor(eta_rad, device=device, dtype=dtype)
    scale = as_tensor(evolution_scale_m, device=device, dtype=dtype)

    return torch.sqrt((x_t / scale) ** 2 + torch.sin(eta_t) ** 2)


def crossed_shell_segments(
    eta_rad: TensorLike,
    rj_shells: torch.Tensor,
    *,
    depth_m: TensorLike = 0.0,
    evolution_scale_m: TensorLike = constant.R_E,
    context: Optional[RuntimeContext] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The PREM-shell-aligned, detector-depth-aware segment boundaries for one case-A nadir angle.

    Only shells with ``rj > sin(eta_prime)`` are crossed (``m`` of them,
    ``eta_prime = eta_prime_from_eta(eta_rad, r_d)``). Segments, in physical
    path order (entry at the far-side surface to exit at the detector):

        innermost crossed shell:  one segment, ``[-x_1, x_1]`` -- or, if
                                   the detector itself sits inside this
                                   shell, ``[-x_1, x_d]`` (clipped)
        every shell further out:  two segments, ``[-x_i, -x_{i-1}]``
                                   (entry side, unclipped) and
                                   ``[x_{i-1}, x_i]`` (exit side) -- or,
                                   for the shell the detector itself sits
                                   inside (if not the innermost one),
                                   ``[x_{i-1}, x_d]`` (clipped); no
                                   segments are built for shells further
                                   out than that one, since the path
                                   stops at the detector

    With ``depth_m=0`` the detector always sits exactly at the outermost
    crossed shell's own boundary, so no clipping ever occurs -- the same
    ``2*m-1`` unclipped segments as before this function supported depth.

    Args:
        eta_rad: Nadir angle, in radians -- must be a 0-dim (scalar) tensor:
            the segment *count* itself depends on how many shells are
            crossed (and where the detector sits among them), so a batch
            of different ``eta_rad``/``depth_m`` values cannot in general
            share one output shape.
        rj_shells: Shell outer-radius fractions, sorted ascending, shape
            ``(n_shells,)`` -- see
            ``tpeanuts.medium.earth.even_power.EvenPowerShellTable.rj``.
        depth_m: Detector depth below the surface, in metres.
        evolution_scale_m: Positive length scale in metres.
        context: Optional runtime device/dtype. When omitted, both are
            inferred from the tensor inputs.

    Returns:
        ``(x1, x2, shell_index)``, each shaped ``(n_segments,)``
        (``n_segments <= 2*m - 1``): segment bounds in metres (in
        physical path order, ``x1`` strictly ascending) and the row of
        ``rj_shells``/``EvenPowerShellTable`` each segment's density
        comes from.

    Raises:
        ValueError: If ``eta_rad`` is not a 0-dim tensor, or (raised by
            ``detector_radius_fraction``) if ``depth_m`` is invalid.
    """
    device, dtype = _resolved_device_dtype(context, eta_rad, rj_shells, depth_m, evolution_scale_m)
    eta_t = as_tensor(eta_rad, device=device, dtype=dtype)
    if eta_t.ndim != 0:
        raise ValueError(
            f"eta_rad must be a 0-dim (scalar) tensor for crossed_shell_segments -- got shape {tuple(eta_t.shape)} "
            "(the segment count itself depends on eta_rad, so a batch of trajectories cannot share one output shape)."
        )
    rj_t = as_tensor(rj_shells, device=device, dtype=dtype)
    scale = as_tensor(evolution_scale_m, device=device, dtype=dtype)

    r_d = detector_radius_fraction(depth_m, evolution_scale_m=scale, context=context)
    eta_prime = eta_prime_from_eta(eta_t, r_d)
    x_d = detector_x_coordinate_m(eta_t, r_d, evolution_scale_m=scale, context=context)

    sin_eta_prime_sq = torch.sin(eta_prime) ** 2
    crossed = torch.nonzero(rj_t > torch.sqrt(sin_eta_prime_sq), as_tuple=True)[0]
    boundaries = scale * torch.sqrt((rj_t[crossed] ** 2 - sin_eta_prime_sq).clamp_min(0.0))
    m = crossed.shape[0]

    # Which crossed shell the detector itself sits inside: the smallest
    # rj (original table index) with rj >= r_d, then its position k_d
    # within the (ascending) crossed list -- always present, since
    # rj_shell_d >= r_d >= sin(eta_prime) makes that shell crossed by
    # construction.
    shell_d = int(torch.nonzero(rj_t >= r_d, as_tuple=True)[0][0])
    k_d = int(torch.nonzero(crossed == shell_d, as_tuple=True)[0][0])

    x1_list, x2_list, shell_list = [], [], []
    for k in range(m - 1, 0, -1):
        x1_list.append(-boundaries[k])
        x2_list.append(-boundaries[k - 1])
        shell_list.append(crossed[k])

    if k_d == 0:
        x1_list.append(-boundaries[0])
        x2_list.append(x_d)
        shell_list.append(crossed[0])
    else:
        x1_list.append(-boundaries[0])
        x2_list.append(boundaries[0])
        shell_list.append(crossed[0])
        for k in range(1, k_d):
            x1_list.append(boundaries[k - 1])
            x2_list.append(boundaries[k])
            shell_list.append(crossed[k])
        x1_list.append(boundaries[k_d - 1])
        x2_list.append(x_d)
        shell_list.append(crossed[k_d])

    x1 = torch.stack(x1_list)
    x2 = torch.stack(x2_list)
    shell_index = torch.stack(shell_list)
    return x1, x2, shell_index


def crossed_shell_segments_batched(
    eta_rad: TensorLike,
    rj_shells: torch.Tensor,
    *,
    depth_m: TensorLike = 0.0,
    evolution_scale_m: TensorLike = constant.R_E,
    context: Optional[RuntimeContext] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The batched counterpart of ``crossed_shell_segments``: fixed-size, padded, mask-validated.

    ``crossed_shell_segments`` itself cannot batch ``eta_rad`` because the
    segment *count* varies per element -- this function sidesteps that
    by always allocating ``2*n_shells - 1`` slots (the true maximum,
    reached when every shell is crossed), filling the first
    ``n_segments`` of them with exactly what ``crossed_shell_segments``
    itself would return for that element, and marking the rest
    ``x1=x2=0`` (a zero-length, physically inert segment --
    ``matrix_exp(H*0) == I`` regardless of what ``shell_index`` a padded
    slot happens to carry; ``tpeanuts.physics.evolution.perturbative.
    profile.EvenPowerProfile`` itself already special-cases a
    zero-length segment to an exactly-zero average density, not a
    stray ``0/0``).

    Only possible because ``rj_shells`` is sorted ascending: "crossed"
    (``rj > sin(eta_prime)``) is then always a contiguous *suffix* of
    the shell table, so both the crossed-shell count and the detector's
    own position within it reduce to two ``torch.searchsorted`` calls,
    fully vectorized -- no Python-level loop over shells anywhere here.

    Args:
        eta_rad: Nadir angle, in radians -- scalar or any batch shape,
            broadcastable against ``depth_m``/``r_d``.
        rj_shells: Shell outer-radius fractions, sorted ascending, shape
            ``(n_shells,)``.
        depth_m: Detector depth below the surface, in metres.
        evolution_scale_m: Positive length scale in metres.
        context: Optional runtime device/dtype. When omitted, both are
            inferred from the tensor inputs.

    Returns:
        ``(x1, x2, shell_index, valid)``, each shaped ``(*batch_shape,
        2*n_shells - 1)``: segment bounds in metres and the row of
        ``rj_shells``/``EvenPowerShellTable`` each segment's density
        comes from, matching ``crossed_shell_segments`` element-for
        -element (same physical order) wherever ``valid`` is True; a
        zero-length no-op segment (``shell_index=0``) wherever it is
        False.

    Raises:
        ValueError: Via ``detector_radius_fraction``, if ``depth_m`` is
            invalid.
    """
    device, dtype = _resolved_device_dtype(context, eta_rad, rj_shells, depth_m, evolution_scale_m)
    eta_t = as_tensor(eta_rad, device=device, dtype=dtype)
    rj_t = as_tensor(rj_shells, device=device, dtype=dtype)
    scale = as_tensor(evolution_scale_m, device=device, dtype=dtype)
    resolved_context = RuntimeContext(device=device, dtype=dtype)
    n_shells = rj_t.shape[0]
    max_slots = 2 * n_shells - 1

    r_d = detector_radius_fraction(depth_m, evolution_scale_m=scale, context=resolved_context)
    batch_shape = torch.broadcast_shapes(eta_t.shape, r_d.shape)
    eta_b = torch.broadcast_to(eta_t, batch_shape)
    r_d_b = torch.broadcast_to(r_d, batch_shape)
    lead_ones = (1,) * len(batch_shape)

    eta_prime = eta_prime_from_eta(eta_b, r_d_b)
    x_d = detector_x_coordinate_m(eta_b, r_d_b, evolution_scale_m=scale, context=resolved_context)
    sin_eta_prime = torch.sin(eta_prime)
    sin_eta_prime_sq = sin_eta_prime**2

    # "crossed" (rj > sin(eta_prime)) is a contiguous suffix of rj_shells
    # (ascending): j0 is its own start index, m = n_shells - j0 its length.
    j0 = searchsorted_contiguous(rj_t, sin_eta_prime.reshape(-1), right=True).reshape(batch_shape)
    m = n_shells - j0

    # shell_d: smallest shell index j with rj_shells[j] >= r_d -- always
    # within [j0, n_shells-1] (the detector's own shell is always crossed,
    # by construction). k_d is its position within the crossed suffix.
    shell_d = searchsorted_contiguous(rj_t, r_d_b.reshape(-1), right=False).reshape(batch_shape)
    k_d = shell_d - j0

    boundaries_full = scale * torch.sqrt(
        (rj_t.reshape(*lead_ones, n_shells) ** 2 - sin_eta_prime_sq.unsqueeze(-1)).clamp_min(0.0)
    )  # (*batch_shape, n_shells)

    s = torch.arange(max_slots, device=device, dtype=torch.long).reshape(*lead_ones, max_slots)
    s = torch.broadcast_to(s, (*batch_shape, max_slots))
    m_b = m.unsqueeze(-1)
    k_d_b = k_d.unsqueeze(-1)
    j0_b = j0.unsqueeze(-1)

    total_segments = torch.where(k_d_b == 0, m_b, m_b + k_d_b)
    valid = s < total_segments

    is_entry = s < (m_b - 1)
    is_k0 = s == (m_b - 1)
    is_exit = (s >= m_b) & (s <= m_b + k_d_b - 2)
    is_final = (s == (m_b + k_d_b - 1)) & (k_d_b != 0)

    k = torch.where(is_entry, m_b - 1 - s, torch.zeros_like(s))
    k = torch.where(is_exit, s - m_b + 1, k)
    k = torch.where(is_final, k_d_b.expand_as(k), k)
    # is_k0 slots already carry k=0 from the zeros_like default above.

    global_idx = (j0_b + k).clamp(0, n_shells - 1)
    global_idx_minus1 = (j0_b + k - 1).clamp(0, n_shells - 1)

    boundaries_at_k = torch.gather(boundaries_full, dim=-1, index=global_idx)
    boundaries_at_k_minus1 = torch.gather(boundaries_full, dim=-1, index=global_idx_minus1)
    x_d_expanded = x_d.unsqueeze(-1).expand(*batch_shape, max_slots)

    x1 = torch.zeros(*batch_shape, max_slots, device=device, dtype=dtype)
    x2 = torch.zeros_like(x1)
    shell_index = torch.zeros(*batch_shape, max_slots, device=device, dtype=torch.long)

    x1 = torch.where(is_entry, -boundaries_at_k, x1)
    x2 = torch.where(is_entry, -boundaries_at_k_minus1, x2)
    shell_index = torch.where(is_entry, global_idx, shell_index)

    x1 = torch.where(is_k0, -boundaries_at_k, x1)
    x2 = torch.where(is_k0 & (k_d_b == 0), x_d_expanded, x2)
    x2 = torch.where(is_k0 & (k_d_b != 0), boundaries_at_k, x2)
    shell_index = torch.where(is_k0, global_idx, shell_index)

    x1 = torch.where(is_exit, boundaries_at_k_minus1, x1)
    x2 = torch.where(is_exit, boundaries_at_k, x2)
    shell_index = torch.where(is_exit, global_idx, shell_index)

    x1 = torch.where(is_final, boundaries_at_k_minus1, x1)
    x2 = torch.where(is_final, x_d_expanded, x2)
    shell_index = torch.where(is_final, global_idx, shell_index)

    x1 = torch.where(valid, x1, torch.zeros_like(x1))
    x2 = torch.where(valid, x2, torch.zeros_like(x2))
    shell_index = torch.where(valid, shell_index, torch.zeros_like(shell_index))

    return x1, x2, shell_index, valid


def aligned_chord_segments(
    x1_m: TensorLike,
    x_d_m: TensorLike,
    eta_prime_rad: TensorLike,
    discontinuity_radii: torch.Tensor,
    nsteps: int,
    *,
    evolution_scale_m: TensorLike = constant.R_E,
    context: Optional[RuntimeContext] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Midpoints and lengths of ``nsteps`` chord segments that never straddle a density jump.

    ``NUMERICAL`` and ``ANALYTIC`` sample the density of a table at the midpoint of each segment, so a segment that
    contains a jump takes one density for both sides of it; with ``nsteps`` equal segments this gives errors of up to
    ``2.5e-2`` in a probability for the PREM table. Here the chord ``[x1, x_d]`` is first cut at the coordinates where
    it crosses each radius of ``discontinuity_radii`` (``x = +-sqrt(r^2 - sin^2(eta')) R`` for the radii above its
    closest approach and inside ``[x1, x_d]``), and the ``nsteps`` segments are shared out among the pieces: one at
    least for each piece, the rest in proportion to its length (rounded running totals, which keep a symmetric chord
    symmetric), then equal inside each piece.
    The total number of segments is always ``nsteps`` and the lengths add up to ``x_d - x1``, so the output has the same
    shape as that of equal segments and can replace it. Pieces shorter than ``1e-9`` of the chord get no segment (their
    length is dropped).

    Args:
        x1_m: Chord coordinate of the entry point, in metres, shape ``(...,)``.
        x_d_m: Chord coordinate of the detector, same shape.
        eta_prime_rad: Nadir angle that defines the closest approach of the chord (``eta_prime`` for a buried
            detector), same shape.
        discontinuity_radii: Radii ``r / R`` of the jumps, ascending, shape ``(n_jumps,)``.
        nsteps: Number of segments; at least ``2 * n_jumps + 1``.
        evolution_scale_m: Length ``R`` of the radii, in metres.
        context: Optional runtime device/dtype; inferred from the tensors when omitted.

    Returns:
        ``(midpoints_x_m, length_m)``, both shaped ``(..., nsteps)``; the integer allocation is not differentiable,
        the positions and lengths are differentiable with respect to the inputs.

    Raises:
        ValueError: If ``nsteps < 2 * n_jumps + 1``.
    """
    device, dtype = _resolved_device_dtype(context, x1_m, x_d_m, eta_prime_rad, discontinuity_radii)
    x1 = as_tensor(x1_m, device=device, dtype=dtype)
    x_d = as_tensor(x_d_m, device=device, dtype=dtype)
    sin_eta = torch.sin(as_tensor(eta_prime_rad, device=device, dtype=dtype))
    radii = discontinuity_radii.to(device=device, dtype=dtype)
    scale = as_tensor(evolution_scale_m, device=device, dtype=dtype)
    n_jumps = radii.shape[-1]
    if nsteps < 2 * n_jumps + 1:
        raise ValueError(f"nsteps must be at least 2 * n_jumps + 1 = {2 * n_jumps + 1}, got {nsteps}.")

    batch_shape = torch.broadcast_shapes(x1.shape, x_d.shape, sin_eta.shape)
    x1, x_d, sin_eta = (t.expand(batch_shape).reshape(-1) for t in (x1, x_d, sin_eta))

    crosses = radii.unsqueeze(0) > sin_eta.unsqueeze(-1)
    half_chord = torch.sqrt((radii.unsqueeze(0) ** 2 - sin_eta.unsqueeze(-1) ** 2).clamp_min(0.0)) * scale
    candidates = torch.cat([-half_chord, half_chord], dim=-1)
    valid = torch.cat([crosses, crosses], dim=-1) & (candidates > x1.unsqueeze(-1)) & (candidates < x_d.unsqueeze(-1))
    positions = torch.where(valid, candidates, x_d.unsqueeze(-1).expand_as(candidates))
    bounds = torch.cat([x1.unsqueeze(-1), torch.sort(positions, dim=-1).values, x_d.unsqueeze(-1)], dim=-1)
    pieces = bounds[..., 1:] - bounds[..., :-1]

    total = pieces.sum(dim=-1, keepdim=True)
    present = pieces > 1.0e-9 * total
    kept = torch.where(present, pieces, torch.zeros_like(pieces)).detach()
    spare = (nsteps - present.sum(dim=-1, keepdim=True)).to(dtype)
    # One segment for every piece plus the spare ones in proportion to length, as a real number per piece; the integer
    # counts are the differences of the rounded running totals. Rounding the running totals, unlike handing out the
    # remainders one by one, treats the two sides of a symmetric chord alike, keeps one segment at least per piece
    # (each ideal count is at least 1) and adds up to nsteps exactly (the last running total is nsteps).
    ideal = present.to(dtype) + spare * kept / kept.sum(dim=-1, keepdim=True)
    running = torch.round(ideal.cumsum(dim=-1)).long()
    running[..., -1] = nsteps
    count = running - torch.cat([torch.zeros_like(running[..., :1]), running[..., :-1]], dim=-1)
    cumulative = count.cumsum(dim=-1)
    slot = torch.arange(nsteps, device=device).unsqueeze(0).expand(cumulative.shape[0], nsteps)
    piece = torch.searchsorted(cumulative.contiguous(), slot.contiguous(), right=True).clamp_max(count.shape[-1] - 1)
    first_slot = torch.gather(cumulative - count, -1, piece)
    within = (slot - first_slot).to(dtype)
    piece_length = torch.gather(pieces, -1, piece)
    segment_length = piece_length / torch.gather(count, -1, piece).to(dtype)
    midpoints = torch.gather(bounds[..., :-1], -1, piece) + (within + 0.5) * segment_length
    return midpoints.reshape(*batch_shape, nsteps), segment_length.reshape(*batch_shape, nsteps)

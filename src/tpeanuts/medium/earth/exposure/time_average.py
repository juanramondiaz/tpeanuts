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

"""Elliptic-integral nadir-exposure geometry, ported from v1_0's ``exposure_math.py``.

For a detector at geographic latitude ``lam``, the fraction of time it
observes a given nadir angle ``eta_rad`` depends on the Earth's daily rotation
and annual orbit (``tpeanuts.util.constant.EARTH_INCLINATION_RAD`` sets
how the Sun's apparent declination swings over the year). On any given
day the Sun traces a small circle on the celestial sphere at a fixed
declination; the detector sees nadir angle ``eta_rad`` for the fraction of
the day during which that circle intersects the cone of half-angle
``eta_rad`` around local nadir. ``indefinite_integral_day`` is the closed-form
antiderivative of that instantaneous rate (in incomplete elliptic
integrals of the first kind, since the underlying geometric integral has
no elementary closed form); ``integral_angle`` evaluates it over the
valid sub-interval of one year-angle window ``[a1, a2]``; ``integral_day``
sums the contributions of every such window inside a requested
day-of-year range ``[d1, d2]`` to give the total exposure weight
``W(eta_rad)`` for one nadir angle. ``make_eta_grid`` builds the uniform
``eta_rad`` grid ``table.build_nadir_exposure`` tabulates ``W`` over.

All intermediate algebra here is carried out in complex arithmetic
(``tpeanuts.util.math.complex_sqrt``/``ellipf_incomplete``) to stay
well-defined across the branch cuts that appear for different
``eta_rad``/``lam``/``T`` regimes -- ``_branch_stable_sqrt`` additionally
collapses a complex intermediate whose imaginary part is only rounding
noise (below 1e-12 of its magnitude, or of 1 when it nearly cancels) back
onto the real axis first, avoiding spurious branch-cut sign flips (v1_0's own
fix for exact zeros, widened here: CUDA's complex arithmetic leaves a ~1e-16
imaginary residue where the CPU gets exactly zero, which flipped the sign of
the whole antiderivative on the GPU). ``integral_angle``/``integral_day`` extract plain Python
floats from a handful of interval-endpoint tensors to drive their own
control flow (which of several possibly-empty sub-intervals exist, and
how many terms to sum) -- an intrinsic feature of this closed-form
method, not a shortcut: ``eta_rad``/``lam`` (a nadir angle and a detector
latitude) are never fit parameters, so nothing of physical interest is
lost by this boundary being non-differentiable.

Precision: the closed form is evaluated in float64 whatever the caller's
``context.dtype`` (``_evaluated_in_float64``), and only the result is cast
back. It is a handful of scalar evaluations per nadir angle, so float64 costs
nothing, whereas in float32 its complex square roots pick different branches
on CPU and CUDA (the real part of the antiderivative flipped sign for
``T`` above ~0.5, giving ``W(eta_rad) < 0`` on the GPU) and even on CPU the
result was off by ~0.6 % (``W(2.6) = 4.920`` against 4.949).

Module contents:
    indefinite_integral_day(...)
        Closed-form antiderivative (in the day-phase variable ``T``) of
        the instantaneous nadir-exposure rate.
    integral_angle(...)
        Definite exposure-rate integral over one year-angle window
        ``[a1, a2]``.
    integral_day(...)
        Total day-integrated exposure weight ``W(eta_rad)`` over a
        day-of-year range ``[d1, d2]``.
    HORIZON_EDGE_RAD
        How far the day and night grids stop short of the horizon.
    make_eta_grid(...)
        Uniform nadir-angle grid in ``[0, pi]`` (or its day/night half).
"""

from __future__ import annotations

import functools
import math
from typing import Callable, Literal, Optional, TypeVar

import torch

from tpeanuts.config import default
from tpeanuts.util.constant import EARTH_INCLINATION_RAD
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.math import complex_sqrt, csc, ellipf_incomplete, intersection, sec
from tpeanuts.util.tensor import TensorLike, as_tensor, cdtype_from_real

DayNight = Optional[Literal["day", "night"]]

# Distance, in radians, at which the day and night grids stop short of the horizontal direction (pi/2).
HORIZON_EDGE_RAD = 1.0e-6

_F = TypeVar("_F", bound=Callable[..., torch.Tensor])

# Size (relative to the value, and at least to 1: the operands are O(1), so a difference of them carries O(1e-16)
# absolute noise even where it nearly cancels) below which an imaginary part is rounding noise, not a complex value.
_REAL_TOLERANCE = 1.0e-12


def _evaluated_in_float64(function: _F) -> _F:
    """Run a keyword-``context`` function in float64 and cast its result back to the caller's dtype.

    Args:
        function: Function taking a keyword-only ``context`` and returning a
            real or complex tensor.

    Returns:
        A wrapper with the same signature. If ``context.dtype`` is already
        float64 it calls ``function`` unchanged; otherwise it evaluates with
        a float64 context on the same device and casts the result to the
        caller's real (or matching complex) dtype.
    """

    @functools.wraps(function)
    def wrapper(*args, context: RuntimeContext, **kwargs):
        if context.dtype == torch.float64:
            return function(*args, context=context, **kwargs)
        result = function(*args, context=RuntimeContext(device=context.device, dtype=torch.float64), **kwargs)
        return result.to(cdtype_from_real(context.dtype) if torch.is_complex(result) else context.dtype)

    return wrapper  # type: ignore[return-value]


def _branch_stable_sqrt(z: torch.Tensor) -> torch.Tensor:
    """Square root that collapses an already-real complex value back onto the real axis first.

    Args:
        z: Complex tensor.

    Returns:
        Tensor holding ``sqrt(z)``, with floating-point imaginary noise on
        exactly-real entries suppressed before the branch cut is chosen.
    """
    real_as_complex = z.real.to(dtype=z.dtype)
    z = torch.where(z.imag.abs() <= _REAL_TOLERANCE * z.abs().clamp_min(1.0), real_as_complex, z)
    return torch.sqrt(z)


@_evaluated_in_float64
def indefinite_integral_day(
    T: TensorLike,
    eta_rad: TensorLike,
    lam: TensorLike,
    *,
    inclination: float = EARTH_INCLINATION_RAD,
    elliptic_tol: float = default.EARTH_EXPOSURE_ELLIPTIC_TOL,
    context: RuntimeContext,
) -> torch.Tensor:
    """Closed-form antiderivative of the instantaneous nadir-exposure rate.

    Args:
        T: Dimensionless day-phase integration variable, in ``[-1, 1]``.
        eta_rad: Nadir angle in radians at which the exposure rate is
            evaluated.
        lam: Detector geographic latitude in radians.
        inclination: Earth's orbital-plane (rotation-axis) inclination in
            radians.
        elliptic_tol: Convergence tolerance passed to
            ``tpeanuts.util.math.ellipf_incomplete``.
        context: Runtime device/dtype.

    Returns:
        Complex tensor with the antiderivative value at ``T``, broadcast
        over ``T``, ``eta_rad``, and ``lam``. Callers take the real part of a
        difference of two evaluations to obtain a physical exposure
        contribution (see ``integral_angle``).
    """
    cdtype = cdtype_from_real(context.dtype)

    T_t = as_tensor(T, device=context.device, dtype=context.dtype)
    eta_t = as_tensor(eta_rad, device=context.device, dtype=context.dtype)
    lam_t = as_tensor(lam, device=context.device, dtype=context.dtype)

    Tc = T_t.to(dtype=cdtype)
    etac = eta_t.to(dtype=cdtype)
    lamc = lam_t.to(dtype=cdtype)

    incc = torch.tensor(inclination, device=context.device, dtype=cdtype)
    sqrt2 = complex_sqrt(torch.tensor(2.0, device=context.device, dtype=context.dtype))

    sin_i = torch.sin(incc)
    csc_i = csc(incc)

    sin_eta = torch.sin(etac)
    cos_eta = torch.cos(etac)

    sin_eta_minus_lam = torch.sin(etac - lamc)
    sin_eta_plus_lam = torch.sin(etac + lamc)

    sec_lam = sec(lamc)
    tan_lam = torch.tan(lamc)

    one = torch.ones_like(Tc)

    # Elliptic amplitude.
    phi_arg_1 = (one + Tc) * (sin_i - sin_eta_plus_lam)
    phi_arg_2 = (-one + Tc) * (sin_i + sin_eta_plus_lam)
    phi_arg = _branch_stable_sqrt(-(phi_arg_1 / phi_arg_2))
    phi = torch.asin(phi_arg)

    # Elliptic parameter.
    m_1 = (sin_i + sin_eta_minus_lam) * (sin_i + sin_eta_plus_lam)
    m_2 = (sin_i - sin_eta_minus_lam) * (sin_i - sin_eta_plus_lam)
    m = m_1 / m_2

    F = ellipf_incomplete(phi, m, tol=elliptic_tol)

    # Coefficients.
    A_1 = Tc * sin_i + sin_eta_minus_lam
    A_2 = (-one + Tc) * (sin_i - sin_eta_minus_lam)
    A = _branch_stable_sqrt(A_1 / A_2)

    B_1 = (one + Tc) * (sin_i - sin_eta_plus_lam)
    B_2 = (-one + Tc) * (sin_i + sin_eta_plus_lam)
    B = _branch_stable_sqrt(-(B_1 / B_2))

    C_1 = Tc * sin_i - sin_eta_plus_lam
    C_2 = (-one + Tc) * (sin_i + sin_eta_plus_lam)
    C = _branch_stable_sqrt(C_1 / C_2)

    D_1 = -one + cos_eta**2 * sec_lam**2
    D_2 = Tc**2 * sec_lam**2 * sin_i**2
    D_3 = -2.0 * Tc * cos_eta * sec_lam * sin_i * tan_lam
    D = _branch_stable_sqrt(D_1 + D_2 + D_3)

    E_1 = one + Tc**2 - Tc**2 * torch.cos(2.0 * incc) + torch.cos(2.0 * etac)
    E_2 = 4.0 * Tc * cos_eta * sec_lam * sin_i * tan_lam
    E = _branch_stable_sqrt(-2.0 + E_1 * sec_lam**2 - E_2)

    G_1 = -one + cos_eta**2 * sec_lam**2 + Tc**2 * sec_lam**2 * sin_i**2
    G_2 = -2.0 * Tc * cos_eta * sec_lam * sin_i * tan_lam
    G = _branch_stable_sqrt((-one + Tc**2) * (G_1 + G_2))

    N_1 = -2.0 * sqrt2 * (-one + Tc) ** 2 * csc_i * F * sec_lam
    N_2 = sin_eta * A * B * C * (sin_i + sin_eta_plus_lam) * D
    numerator = N_1 * N_2

    denominator = (-one + csc_i * sin_eta_plus_lam) * E * G

    return numerator / denominator


@_evaluated_in_float64
def integral_angle(
    eta_rad: TensorLike,
    lam: TensorLike,
    a1: float = 0.0,
    a2: float = math.pi,
    eps: float = default.EARTH_EXPOSURE_ANGLE_EPS,
    *,
    inclination: float = EARTH_INCLINATION_RAD,
    context: RuntimeContext,
) -> torch.Tensor:
    """Definite exposure-rate integral over one year-angle window ``[a1, a2]``.

    Intersects the day-phase domain ``T in [-1, 1]`` with the geometric
    constraints required for the detector at latitude ``lam`` to observe
    nadir angle ``eta_rad`` during that part of the year, then evaluates
    ``indefinite_integral_day`` at the resulting sub-interval endpoints.

    Args:
        eta_rad: Nadir angle in radians. Must be a scalar (0-dim) tensor or
            plain float -- this function (and ``integral_day``, which
            calls it) is evaluated once per grid point, not batched.
        lam: Detector geographic latitude in radians.
        a1: Lower bound of the year-angle window in radians, in
            ``[0, pi]``.
        a2: Upper bound of the year-angle window in radians, in
            ``[0, pi]``, with ``a2 >= a1``.
        eps: Small tolerance keeping the day-phase/year-angle
            intersection intervals strictly open, avoiding degenerate
            (zero-length) integration ranges from floating point boundary
            effects.
        inclination: Earth's orbital-plane (rotation-axis) inclination in
            radians.
        context: Runtime device/dtype.

    Returns:
        Scalar real tensor with the exposure contribution of this
        year-angle window, zero if it is geometrically inaccessible for
        this ``eta_rad``.

    Raises:
        ValueError: If ``a1``/``a2`` lie outside ``[0, pi]`` or
            ``a1 > a2``.
        RuntimeError: If the geometric intersection yields a disconnected
            set of valid sub-intervals (not handled by this closed-form
            method).
    """
    if not (0.0 <= a1 <= math.pi) or not (0.0 <= a2 <= math.pi) or a1 > a2:
        raise ValueError("a1 and a2 must be between 0 and pi, with a2 >= a1.")

    eta_t = as_tensor(eta_rad, device=context.device, dtype=context.dtype)
    lam_t = as_tensor(lam, device=context.device, dtype=context.dtype)
    inc = torch.tensor(inclination, device=context.device, dtype=context.dtype)

    int1 = torch.tensor([-1.0 + eps, 1.0 - eps], device=context.device, dtype=context.dtype)
    int2 = torch.stack(
        [
            torch.sin(lam_t - eta_t) / torch.sin(inc) + eps,
            torch.sin(lam_t + eta_t) / torch.sin(inc) - eps,
        ]
    )
    a1_t = torch.tensor(a1, device=context.device, dtype=context.dtype)
    a2_t = torch.tensor(a2, device=context.device, dtype=context.dtype)
    int3 = torch.stack([torch.cos(a2_t), torch.cos(a1_t)])

    int_full = intersection(int1, int2, int3)

    if int_full.numel() == 0:
        return torch.zeros((), device=context.device, dtype=context.dtype)
    if int_full.numel() != 2:
        raise RuntimeError("Unable to treat disconnected integration intervals.")

    low, up = int_full[0], int_full[1]
    if (up - low) <= 2.0 * eps:
        return torch.zeros((), device=context.device, dtype=context.dtype)

    value = indefinite_integral_day(
        up, eta_t, lam_t, inclination=inclination, context=context
    ) - indefinite_integral_day(low, eta_t, lam_t, inclination=inclination, context=context)

    return value.real


@_evaluated_in_float64
def integral_day(
    eta_rad: TensorLike,
    lam: TensorLike,
    d1: float = default.EARTH_EXPOSURE_D1,
    d2: float = default.EARTH_EXPOSURE_D2,
    *,
    inclination: float = EARTH_INCLINATION_RAD,
    context: RuntimeContext,
) -> torch.Tensor:
    """Total day-integrated exposure weight ``W(eta_rad)`` over a day-of-year range.

    Converts ``[d1, d2]`` into year-angle radians (``a = 2*pi*d/365``),
    splits it at the half-year boundaries ``pi``/``2*pi`` (the underlying
    day/night geometry is evaluated separately on each half via a sign
    reflection), and sums ``integral_angle``'s own contribution of each
    resulting sub-window.

    Args:
        eta_rad: Nadir angle in radians. Must be a scalar (0-dim) tensor or
            plain float -- see ``integral_angle``.
        lam: Detector geographic latitude in radians.
        d1: First day of year of the integration window, in
            ``[0, 365]``.
        d2: Last day of year of the integration window, in ``[0, 365]``,
            with ``d2 >= d1``.
        inclination: Earth's orbital-plane (rotation-axis) inclination in
            radians.
        context: Runtime device/dtype.

    Returns:
        Scalar real tensor with the total exposure weight for this
        ``eta_rad`` over the requested day-of-year range.

    Raises:
        ValueError: If ``d1``/``d2`` lie outside ``[0, 365]`` or
            ``d1 > d2``.
    """
    if not (0.0 <= d1 <= 365.0) or not (0.0 <= d2 <= 365.0) or d1 > d2:
        raise ValueError("d1 and d2 must be between 0 and 365, with d2 >= d1.")

    eta_t = as_tensor(eta_rad, device=context.device, dtype=context.dtype)
    lam_t = as_tensor(lam, device=context.device, dtype=context.dtype)

    a1 = 2.0 * math.pi * d1 / 365.0
    a2 = 2.0 * math.pi * d2 / 365.0
    year_interval = torch.tensor([a1, a2], device=context.device, dtype=context.dtype)

    int1 = intersection(
        year_interval, torch.tensor([0.0, math.pi], device=context.device, dtype=context.dtype)
    )
    int2 = intersection(
        year_interval,
        torch.tensor([math.pi, 2.0 * math.pi], device=context.device, dtype=context.dtype),
    )

    weight1 = torch.zeros((), device=context.device, dtype=context.dtype)
    weight2 = torch.zeros((), device=context.device, dtype=context.dtype)

    # Clamped into [0, pi] with the plain Python math.pi used by
    # integral_angle's own domain check: a context.dtype below float64
    # (float32) rounds pi to a value that can land a hair outside
    # [0, math.pi] here (both directly, and doubly so after the 2*pi
    # reflection below), which is a floating-point boundary artifact, not
    # a genuine domain violation -- eps-scale, well within
    # integral_angle's own eps-regularized tolerance for what its
    # geometry actually resolves.
    if int1.numel() == 2:
        weight1 = integral_angle(
            eta_t,
            lam_t,
            min(max(float(int1[0].detach().cpu()), 0.0), math.pi),
            min(max(float(int1[1].detach().cpu()), 0.0), math.pi),
            inclination=inclination,
            context=context,
        )

    if int2.numel() == 2:
        two_pi = torch.tensor(2.0 * math.pi, device=context.device, dtype=context.dtype)
        weight2 = integral_angle(
            eta_t,
            lam_t,
            min(max(float((two_pi - int2[1]).detach().cpu()), 0.0), math.pi),
            min(max(float((two_pi - int2[0]).detach().cpu()), 0.0), math.pi),
            inclination=inclination,
            context=context,
        )

    return weight1 + weight2


def make_eta_grid(
    ns: int,
    *,
    daynight: DayNight = default.EARTH_EXPOSURE_DAYNIGHT,
    context: RuntimeContext,
) -> torch.Tensor:
    """Build the uniform nadir-angle grid used to tabulate exposure weights.

    The full grid has ``ns`` points over ``[0, pi]``. The night grid covers the whole of ``[0, pi/2]`` and the day grid
    the whole of ``[pi/2, pi]``, each with ``(ns + 1) // 2`` points at about the spacing of the full grid, so the
    trapezoidal average over a half is an average over that half and not over a range that stops short of the
    horizon. (Slicing the full grid, as an earlier version did, left out the point at ``pi/2`` for odd ``ns`` and ended
    half a step before it for even ``ns``; the error of a night average was ``3e-3`` at 41 angles.) The horizontal
    direction itself is not a point of the grid: a surface detector has no Earth chord there, which the Earth medium
    rejects, so each half stops ``HORIZON_EDGE_RAD`` short of ``pi/2``.

    Args:
        ns: Number of grid points spanning the full ``[0, pi]`` nadir-angle
            range (before any day/night restriction).
        daynight: ``"day"`` for the grid of ``[pi/2, pi]``, ``"night"`` for that of ``[0, pi/2]``, or ``None`` for the
            full ``[0, pi]`` grid.
        context: Runtime device/dtype.

    Returns:
        One-dimensional tensor of nadir angles in radians, uniformly
        spaced over ``[0, pi]`` (or over the requested half, less ``HORIZON_EDGE_RAD`` at the horizon).

    Raises:
        ValueError: If ``ns < 2``, or if ``daynight`` is not ``None``, ``"day"`` or ``"night"``.
    """
    if ns < 2:
        raise ValueError(f"ns must be at least 2 to define a grid with a spacing; got {ns!r}.")
    if daynight not in (None, "day", "night"):
        raise ValueError("daynight must be None, 'day' or 'night'.")

    half = (int(ns) + 1) // 2
    if daynight == "night":
        return torch.linspace(0.0, math.pi / 2 - HORIZON_EDGE_RAD, half, device=context.device, dtype=context.dtype)
    if daynight == "day":
        return torch.linspace(math.pi / 2 + HORIZON_EDGE_RAD, math.pi, half, device=context.device, dtype=context.dtype)
    return torch.linspace(0.0, math.pi, ns, device=context.device, dtype=context.dtype)

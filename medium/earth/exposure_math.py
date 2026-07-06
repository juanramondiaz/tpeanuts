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

"""
Mathematical exposure utilities for peanuts-torch earth propagation.

This module contains the low-level mathematical functions required to compute
the nadir-angle exposure function W(eta) used in earth matter-regeneration
averages.

The main purpose of this module is to provide Torch/GPU-friendly versions of
the analytical ingredients appearing in the peanuts time-averaging machinery.

The core idea is that the time exposure for a detector at latitude lambda can
be written in terms of angular integrals involving incomplete elliptic
integrals. Since standard GPU libraries do not provide these special functions,
this module implements the required pieces directly in PyTorch.

The module contains:

    IndefiniteIntegralDayTorch(...)
        Torch implementation of the peanuts indefinite day-exposure primitive.

    IntegralAngleTorch(...)
        Computes the valid angular contribution for a fixed eta and latitude.

    IntegralDayTorch(...)
        Computes the day-integrated exposure weight for a tensor of eta values.

This module does not load exposure files, does not cache tables, and does not
compute neutrino oscillation probabilities. It only provides mathematical
building blocks for earth/exposure_table.py and earth/exposure_io.py.

Physical background:
    For a detector at geographic latitude ``lambda``, the fraction of time a
    given nadir angle ``eta`` is observed depends on the Earth's daily
    rotation and its annual orbital motion (the orbital-plane inclination
    ``inclination`` sets how the Sun's apparent declination varies over the
    year). On any given day the Sun's position traces a small circle on the
    celestial sphere at a fixed declination; the detector sees nadir angle
    ``eta`` for the fraction of the day during which that circle intersects
    the cone of half-angle ``eta`` around the local nadir direction. Each of
    those daily windows is an interval of "time" parametrized here by ``T``
    (or, after the day-angle change of variables, by a phase angle in
    ``[0, 2 pi)`` representing the day of the year). Integrating that
    instantaneous daily window over a chosen day-of-year range
    ``[d1, d2]`` (``d1, d2`` in days, ``0 <= d1 <= d2 <= 365``) gives the
    total exposure weight ``W(eta)`` used elsewhere in the Earth pipeline to
    time-average matter-regeneration probabilities. ``IndefiniteIntegralDay``
    is the closed-form antiderivative of the instantaneous exposure rate
    (expressed via incomplete elliptic integrals of the first kind);
    ``IntegralAngle`` evaluates the definite integral over the valid
    sub-interval of ``T`` for a single year-angle window ``[a1, a2]``; and
    ``IntegralDay`` sums the contributions of all such windows that fall
    inside the requested day-of-year range to give the total day-integrated
    weight for one nadir angle ``eta``.

Module functions:
    csqrt(...)
        Branch-safe complex square root used by the elliptic-integral
        algebra below (real inputs are promoted to complex only when
        negative).
    IndefiniteIntegralDay(...)
        Closed-form antiderivative (in the day-phase variable ``T``) of the
        instantaneous nadir-angle exposure rate for a detector at latitude
        ``lam``, expressed through incomplete elliptic integrals.
    IntegralAngle(...)
        Definite integral of the exposure rate over the sub-interval of
        ``T`` consistent with a single year-angle window ``[a1, a2]`` and
        with the geometric validity constraints of the day/night problem.
    IntegralDay(...)
        Total day-integrated exposure weight ``W(eta)`` for one nadir angle,
        obtained by summing ``IntegralAngle`` over the year-angle windows
        contained in the requested day-of-year range ``[d1, d2]``.
    _daynight_slice(...)
        Slice a full ``[0, pi]`` eta-indexed tensor down to its day-only or
        night-only half.
    make_eta_grid(...)
        Build the uniform nadir-angle grid in ``[0, pi]`` (or its day/night
        half) used to tabulate exposure weights.
"""



from __future__ import annotations

from typing import Optional, Union, Literal
import torch

import tpeanuts.util.default as default


from tpeanuts.util.math import(
    csin, ccos, ctan, casin, csqrt as _generic_csqrt,
    sec, csc, ellipf_incomplete, intersection
    )

from tpeanuts.util.type import TensorLike, as_tensor, as_complex_tensor, cdtype_from_real

DayNight = Optional[Literal["day", "night"]]


def csqrt(z: torch.Tensor) -> torch.Tensor:
    """Branch-stable square root used by the exposure elliptic-integral algebra.

    Real, non-negative inputs are returned as a plain real square root.
    Complex inputs with zero imaginary part are first collapsed back onto
    the real axis (avoiding spurious branch-cut artefacts from floating
    point noise) before taking the square root; otherwise the standard
    principal complex square root is used. Real inputs that are not already
    floating point are promoted to ``float64`` before being cast to the
    matching complex dtype.

    Args:
        z: Real or complex input tensor (or array-like convertible to a
            tensor). Dimensionless; this is a generic algebraic helper with
            no physical units of its own.

    Returns:
        Tensor of the same shape holding the (possibly complex) square root
        of ``z``.
    """
    if not torch.is_tensor(z):
        z = torch.as_tensor(z)

    if torch.is_complex(z):
        real_as_complex = z.real.to(dtype=z.dtype)
        z = torch.where(z.imag == 0.0, real_as_complex, z)
        return torch.sqrt(z)

    if not torch.is_floating_point(z):
        z = z.to(dtype=torch.float64)

    return torch.sqrt(z.to(dtype=cdtype_from_real(z.dtype)))


@torch.no_grad()
def IndefiniteIntegralDay(
    T: TensorLike,
    eta: TensorLike,
    lam: TensorLike,
    *,
    inclination: float = default.earth_inclination,    # earth rotation axis inclination
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = default.dtype,
    elliptic_tol: float = default.earth_elliptic_tol,
) -> torch.Tensor:
    """Closed-form antiderivative of the instantaneous nadir-exposure rate.

    Returns the indefinite integral (with respect to the day-phase variable
    ``T``) of the instantaneous rate at which a detector at latitude ``lam``
    observes nadir angle ``eta`` over the course of one day, for a Sun whose
    apparent declination varies sinusoidally with the Earth's orbital-plane
    ``inclination``. ``T`` parametrizes the time-of-day/time-of-year phase
    through the same change of variables used by ``IntegralAngle`` and
    ``IntegralDay`` (it is not itself a physical angle in radians, but an
    auxiliary integration variable in ``[-1, 1]``). The result is built from
    incomplete elliptic integrals of the first kind (``ellipf_incomplete``)
    because the underlying geometric integral has no elementary closed form.
    All intermediate algebra is carried out in complex arithmetic
    (``as_complex_tensor``, ``csqrt``) to remain well-defined across the
    branch cuts that appear for different ``eta``/``lam``/``T`` regimes; the
    physical exposure contribution recovered by callers is the real part of
    the difference of two evaluations of this primitive (see
    ``IntegralAngle``).

    Args:
        T: Dimensionless day-phase integration variable (the limits of
            integration used by ``IntegralAngle``/``IntegralDay``), valued in
            ``[-1, 1]``.
        eta: Nadir angle in radians at which the exposure rate is evaluated.
        lam: Detector geographic latitude in radians.
        inclination: Earth's orbital-plane (rotation-axis) inclination in
            radians, controlling the annual swing of the Sun's apparent
            declination.
        device: Device for intermediate and output tensors.
        dtype: Real dtype; complex intermediates use the matching complex
            dtype.
        elliptic_tol: Convergence tolerance passed to the incomplete
            elliptic integral evaluator ``ellipf_incomplete``.

    Returns:
        Complex tensor with the antiderivative value at ``T``, broadcast
        over ``T``, ``eta``, and ``lam``. Callers take the real part of a
        difference of two evaluations to obtain a physical exposure
        contribution.
    """
    T = as_tensor(T, device=device, dtype=dtype)
    eta = as_tensor(eta, device=T.device, dtype=dtype)
    lam = as_tensor(lam, device=T.device, dtype=dtype)

    Tc = as_complex_tensor(T)
    etac = as_complex_tensor(eta)
    lamc = as_complex_tensor(lam)

    # earth rotation axis inclination
    inc = torch.tensor(inclination, device=T.device, dtype=dtype)
    incc = as_complex_tensor(inc)

    sqrt2 = csqrt(torch.tensor(2.0, device=T.device, dtype=dtype))

    sin_i = csin(incc)
    csc_i = csc(incc)

    sin_eta = csin(etac)
    cos_eta = ccos(etac)

    sin_eta_minus_lam = csin(etac - lamc)
    sin_eta_plus_lam = csin(etac + lamc)

    sec_lam = sec(lamc)
    tan_lam = ctan(lamc)

    one = torch.ones_like(Tc)

    # Elliptic amplitude
    phi_arg_1 = (one + Tc) * (sin_i - sin_eta_plus_lam)
    phi_arg_2 = (-one + Tc) * (sin_i + sin_eta_plus_lam)
    phi_arg = csqrt( -(phi_arg_1 / phi_arg_2))
    phi = casin(phi_arg)

    # Elliptic parameter
    m_1 = (sin_i + sin_eta_minus_lam) * (sin_i + sin_eta_plus_lam) 
    m_2 = (sin_i - sin_eta_minus_lam) * (sin_i - sin_eta_plus_lam)
    m =  m_1  / m_2
        
    # Elliptic integral
    F = ellipf_incomplete(phi, m, tol=elliptic_tol)

    # Coeficients
    A_1 = (Tc * sin_i + sin_eta_minus_lam)
    A_2 = ((-one + Tc) * (sin_i - sin_eta_minus_lam))
    A = csqrt( A_1 / A_2)
    
    B_1 = (one + Tc) * (sin_i - sin_eta_plus_lam)
    B_2 = (-one + Tc) * (sin_i + sin_eta_plus_lam)
    B = csqrt( -(B_1/ B_2))

    C_1 = Tc * sin_i - sin_eta_plus_lam
    C_2 = (-one + Tc) * (sin_i + sin_eta_plus_lam)
    C = csqrt(C_1 / C_2)

    D_1 = -one + cos_eta**2 * sec_lam**2
    D_2 = Tc**2 * sec_lam**2 * sin_i**2 
    D_3 = - 2.0 * Tc * cos_eta * sec_lam * sin_i * tan_lam
    D = csqrt(D_1 + D_2 + D_3 )

    E_1 = one + Tc**2 - Tc**2 * ccos(2.0 * incc) + ccos(2.0 * etac)
    E_2 = 4.0 * Tc * cos_eta * sec_lam * sin_i * tan_lam
    E = csqrt(-2.0 + E_1 * sec_lam**2 - E_2)

    G_1 = -one + cos_eta**2 * sec_lam**2 + Tc**2 * sec_lam**2 * sin_i**2
    G_2 = - 2.0 * Tc * cos_eta * sec_lam * sin_i * tan_lam
    G = csqrt((-one + Tc**2) * ( G_1 + G_2))

    N_1 = -2.0 * sqrt2 * (-one + Tc) ** 2 * csc_i * F * sec_lam
    N_2 = sin_eta * A * B * C * (sin_i + sin_eta_plus_lam) * D 
    numerator = ( N_1 * N_2)
        
    denominator = ( (-one + csc_i * sin_eta_plus_lam)* E * G)

    return numerator / denominator


@torch.no_grad()
def IntegralAngle(
    eta: TensorLike,
    lam: TensorLike,
    a1: float = default.earth_angle_a1,
    a2: float = default.earth_angle_a2,
    eps: float = default.earth_angle_eps,
    *,
    inclination: float = default.earth_inclination,    # earth rotation axis inclination
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = default.dtype,
) -> torch.Tensor:
    """Definite exposure-rate integral over one year-angle window.

    Evaluates the contribution to the nadir-angle exposure weight ``W(eta)``
    coming from a single year-angle interval ``[a1, a2]`` (in radians, both
    within ``[0, pi]``), by intersecting the day-phase domain ``T`` in
    ``[-1, 1]`` with the geometric constraints required for the detector at
    latitude ``lam`` to observe nadir angle ``eta`` during that part of the
    year, then evaluating ``IndefiniteIntegralDay`` at the resulting
    sub-interval endpoints. Returns zero when the constraints leave no valid
    (or only a numerically negligible) sub-interval.

    Args:
        eta: Nadir angle in radians at which the exposure rate is evaluated.
        lam: Detector geographic latitude in radians.
        a1: Lower bound of the year-angle window in radians, in ``[0, pi]``.
        a2: Upper bound of the year-angle window in radians, in ``[0, pi]``,
            with ``a2 >= a1``.
        eps: Small tolerance used to keep the day-phase and year-angle
            intersection intervals strictly open, avoiding degenerate
            (zero-length) integration ranges from floating point boundary
            effects.
        inclination: Earth's orbital-plane (rotation-axis) inclination in
            radians.
        device: Device for intermediate and output tensors.
        dtype: Real dtype used for the computation.

    Returns:
        Scalar real tensor with the exposure contribution of this year-angle
        window for the given ``eta`` and ``lam``. Zero if the window is
        geometrically inaccessible for this ``eta``.

    Raises:
        ValueError: If ``a1``/``a2`` lie outside ``[0, pi]`` or ``a1 > a2``.
        RuntimeError: If the geometric intersection yields a disconnected set
            of valid sub-intervals (not currently handled).
    """
    if (not 0.0 <= a1 <= torch.pi) or (not 0.0 <= a2 <= torch.pi) or (a1 > a2):
        raise ValueError(
            "a1 and a2 must be between 0 and pi, with a2 >= a1."
        )

    eta = as_tensor(eta, device=device, dtype=dtype)
    lam = as_tensor(lam, device=eta.device, dtype=dtype)

    inc = torch.tensor(inclination, device=eta.device, dtype=dtype)

    int1 = torch.tensor(
        [-1.0 + eps, 1.0 - eps],
        device=eta.device,
        dtype=dtype,
    )

    int2 = torch.stack(
        [
            torch.sin(lam - eta) / torch.sin(inc) + eps,
            torch.sin(lam + eta) / torch.sin(inc) - eps,
        ]
    )

    a1_t = torch.tensor(a1, device=eta.device, dtype=dtype)
    a2_t = torch.tensor(a2, device=eta.device, dtype=dtype)

    int3 = torch.stack(
        [
            torch.cos(a2_t),
            torch.cos(a1_t),
        ]
    )

    int_full = intersection(int1, int2, int3)

    if int_full.numel() == 0:
        return torch.zeros((), device=eta.device, dtype=dtype)

    if int_full.numel() != 2:
        raise RuntimeError("Unable to treat disconnected integration intervals.")

    low = int_full[0]
    up = int_full[1]

    if (up - low) <= 2.0 * eps:
        return torch.zeros((), device=eta.device, dtype=dtype)

    value = (
        IndefiniteIntegralDay(up, eta, lam,
                              inclination=inclination,
                              device=eta.
                              device, dtype=dtype)
        - IndefiniteIntegralDay(low, eta, lam,
                                inclination=inclination,
                                device=eta.device, 
                                dtype=dtype)
    )

    return value.real

@torch.no_grad()
def IntegralDay(
    eta: TensorLike,
    lam: TensorLike,
    d1: float = default.earth_d1,
    d2: float = default.earth_d2,
    *,
    inclination: float = default.earth_inclination,    # earth rotation axis inclination
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = default.dtype,
) ->  tuple[torch.Tensor, torch.Tensor]:
    """Total day-integrated exposure weight ``W(eta)`` over a day-of-year range.

    Converts the requested day-of-year range ``[d1, d2]`` into year-angle
    radians (``a = 2 pi d / 365``), splits it at the half-year boundaries
    ``pi`` and ``2 pi`` (because the underlying day/night geometry is
    evaluated separately on each half via a sign reflection), and sums the
    ``IntegralAngle`` contribution of each resulting sub-window. This gives
    the exposure weight ``W(eta)`` for a single nadir angle ``eta``, i.e. the
    fraction of the requested day-of-year window during which the detector
    observes that nadir angle; the full exposure table tabulates this over a
    grid of ``eta`` values (see ``make_eta_grid`` and
    ``medium.earth.exposure_table``).

    Args:
        eta: Nadir angle in radians at which the exposure weight is
            evaluated.
        lam: Detector geographic latitude in radians.
        d1: First day of year of the integration window, in ``[0, 365]``.
        d2: Last day of year of the integration window, in ``[0, 365]``,
            with ``d2 >= d1``.
        inclination: Earth's orbital-plane (rotation-axis) inclination in
            radians.
        device: Device for intermediate and output tensors.
        dtype: Real dtype used for the computation.

    Returns:
        Scalar real tensor with the total exposure weight for this ``eta``
        over the requested day-of-year range.

    Raises:
        ValueError: If ``d1``/``d2`` lie outside ``[0, 365]`` or ``d1 > d2``.
    """
    if (not 0.0 <= d1 <= 365.0) or (not 0.0 <= d2 <= 365.0) or (d1 > d2):
        raise ValueError(
            "d1 and d2 must be between 0 and 365, with d2 >= d1."
        )

    eta = as_tensor(eta, device=device, dtype=dtype)
    lam = as_tensor(lam, device=eta.device, dtype=dtype)

    a1 = 2.0 * torch.pi * d1 / 365.0
    a2 = 2.0 * torch.pi * d2 / 365.0

    year_interval = torch.tensor([a1, a2], device=eta.device, dtype=dtype)

    int1 = intersection(
        year_interval,
        torch.tensor([0.0, torch.pi], device=eta.device, dtype=dtype),
    )

    int2 = intersection(
        year_interval,
        torch.tensor([torch.pi, 2.0 * torch.pi], device=eta.device, dtype=dtype),
    )

    weight1 = torch.zeros((), device=eta.device, dtype=dtype)
    weight2 = torch.zeros((), device=eta.device, dtype=dtype)

    if int1.numel() == 2:
        weight1 = IntegralAngle(
            eta,
            lam,
            float(int1[0].detach().cpu()),
            float(int1[1].detach().cpu()),
            inclination=inclination,
            device=eta.device,
            dtype=dtype,
        )

    if int2.numel() == 2:
        weight2 = IntegralAngle(
            eta,
            lam,
            2.0 * torch.pi - float(int2[1].detach().cpu()),
            2.0 * torch.pi - float(int2[0].detach().cpu()),
            inclination=inclination,
            device=eta.device,
            dtype=dtype,
        )

    return weight1 + weight2

def _daynight_slice(
    tensor: torch.Tensor,
    ns: int,
    daynight: DayNight,
) -> torch.Tensor:
    """Slice a full ``eta in [0, pi]`` tensor to its day-only or night-only half.

    The full nadir grid spans ``[0, pi]``, with the convention that smaller
    nadir angles (closer to 0, looking straight down) correspond to the
    "night" half and larger nadir angles (closer to ``pi``) correspond to
    the "day" half (the boundary follows the day-night asymmetry of the
    underlying exposure geometry: a neutrino crossing the Earth's core
    typically arrives at night, while shallow/no-crossing trajectories
    dominate during the day). ``daynight=None`` returns the tensor
    unchanged.

    Args:
        tensor: One-dimensional tensor indexed the same way as the full
            ``ns``-point eta grid (e.g. eta values themselves, or exposure
            weights defined on that grid).
        ns: Total number of points in the full (un-sliced) grid.
        daynight: ``"day"`` to keep the upper half, ``"night"`` to keep the
            lower half, or ``None`` to keep the full grid.

    Returns:
        The requested slice of ``tensor``.

    Raises:
        ValueError: If ``daynight`` is not one of ``None``, ``"day"``, or
            ``"night"``.
    """
    ns = int(ns)

    if daynight == "day":
        return tensor[(ns + 1) // 2:]

    if daynight == "night":
        return tensor[:ns // 2]
    
    if daynight is None:
       return tensor

    raise ValueError("daynight must be None, 'day' or 'night'.")


def make_eta_grid(
    ns: int,
    *,
    daynight: DayNight = default.earth_daynight,
    device: Union[str, torch.device] = default.earth_device,
    dtype: torch.dtype = default.dtype,
) -> torch.Tensor:
    """Build the uniform nadir-angle grid used to tabulate exposure weights.

    Args:
        ns: Number of grid points spanning the full ``[0, pi]`` nadir-angle
            range (before any day/night slicing).
        daynight: ``"day"`` to keep only the upper half of the grid,
            ``"night"`` to keep only the lower half, or ``None`` to keep the
            full ``[0, pi]`` grid (see ``_daynight_slice`` for the
            convention).
        device: Device for the returned tensor.
        dtype: Real dtype for the returned tensor.

    Returns:
        One-dimensional tensor of nadir angles in radians, uniformly spaced
        over ``[0, pi]`` (or the requested day/night half).
    """
    eta_full = torch.linspace(
        0.0,
        float(torch.pi),
        ns,
        device=device,
        dtype=dtype,
    )

    return _daynight_slice(
        eta_full,
        ns,
        daynight,
    )

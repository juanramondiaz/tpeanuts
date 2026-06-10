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
building blocks for earth/exposure.py and earth/exposure_io.py.
"""



from __future__ import annotations

from typing import Optional, Union, Literal
import torch

import tpeanuts.util.default as default

TensorLike = Union[float, int, torch.Tensor]

from tpeanuts.util.math import(
    csin, ccos, ctan, casin, csqrt as _generic_csqrt,
    sec, csc, ellipf_incomplete, intersection
    )

from tpeanuts.util.type import _as_tensor, _as_complex_tensor, _cdtype_from_real

DayNight = Optional[Literal["day", "night"]]


def csqrt(z: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(z):
        z = torch.as_tensor(z)

    if torch.is_complex(z):
        real_as_complex = z.real.to(dtype=z.dtype)
        z = torch.where(z.imag == 0.0, real_as_complex, z)
        return torch.sqrt(z)

    if not torch.is_floating_point(z):
        z = z.to(dtype=torch.float64)

    return torch.sqrt(z.to(dtype=_cdtype_from_real(z.dtype)))


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
    T = _as_tensor(T, device=device, dtype=dtype)
    eta = _as_tensor(eta, device=T.device, dtype=dtype)
    lam = _as_tensor(lam, device=T.device, dtype=dtype)

    Tc = _as_complex_tensor(T)
    etac = _as_complex_tensor(eta)
    lamc = _as_complex_tensor(lam)

    # earth rotation axis inclination
    inc = torch.tensor(inclination, device=T.device, dtype=dtype)
    incc = _as_complex_tensor(inc)

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
    if (not 0.0 <= a1 <= torch.pi) or (not 0.0 <= a2 <= torch.pi) or (a1 > a2):
        raise ValueError(
            "a1 and a2 must be between 0 and pi, with a2 >= a1."
        )

    eta = _as_tensor(eta, device=device, dtype=dtype)
    lam = _as_tensor(lam, device=eta.device, dtype=dtype)

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
    if (not 0.0 <= d1 <= 365.0) or (not 0.0 <= d2 <= 365.0) or (d1 > d2):
        raise ValueError(
            "d1 and d2 must be between 0 and 365, with d2 >= d1."
        )

    eta = _as_tensor(eta, device=device, dtype=dtype)
    lam = _as_tensor(lam, device=eta.device, dtype=dtype)

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

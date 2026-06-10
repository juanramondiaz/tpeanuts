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
Analytical integration utilities for peanuts-torch.

This module contains low-level analytical ingredients used by the peanuts
perturbative matter-evolution scheme.

The main object computed here is the integral

    I_ab =
    ∫_{x1}^{x2} dx
        exp[-i lambda_a (x2 - x)]
        delta_n_e(x)
        exp[-i lambda_b (x - x1)],

where the density perturbation is

    delta_n_e(x) = atilde + b x^2 + c x^4.

This integral appears in the first-order correction to the segment evolution
operator.

The module also provides the original peanuts analytical coefficients c0 and
c1 for the characteristic equation of the traceless Hamiltonian,

    lambda^3 + c1 lambda + c0 = 0.

The functions are organized as follows:

    c0(...)
        Computes the original peanuts c0 coefficient.

    c1(...)
        Computes the original peanuts c1 coefficient.

    lambdas_cardano(...)
        Computes eigenvalues using a Cardano-like complex formula.

    lambdas_eigvalsh(...)
        Computes eigenvalues using torch.linalg.eigvalsh.

    Iab(...)
        Computes the analytical first-order integral I_ab with a Taylor branch
        for nearly degenerate eigenvalues.

"""



from __future__ import annotations

from typing import Union

import torch

from tpeanuts.core.potential import matter_potential
from tpeanuts.util.type import _as_tensor

TensorLike = Union[float, int, torch.Tensor]


@torch.no_grad()
def c0(
    ki: torch.Tensor,
    th12: TensorLike,
    th13: TensorLike,
    n_e: TensorLike,
    antinu: Union[bool, torch.Tensor],
) -> torch.Tensor:
    """
    Compute the c0 invariant of the reduced traceless Hamiltonian used by the analytical eigenvalue solver.
    
    Args:
        ki: Kinetic eigenvalue vector shaped (..., 3) in km^-1.
        th12: Solar mixing angle theta12 in radians.
        th13: Reactor mixing angle theta13 in radians.
        n_e: Electron density in mol/cm^3.
        antinu: Bool or boolean tensor; True selects antineutrino sign or complex-conjugated mixing.
    
    Returns:
        Tensor containing c0 with the broadcasted input shape.
    """
    device = ki.device
    dtype = ki.dtype

    th12 = _as_tensor(th12, device=device, dtype=dtype)
    th13 = _as_tensor(th13, device=device, dtype=dtype)
    n_e = _as_tensor(n_e, device=device, dtype=dtype)

    k1, k2, k3 = ki[..., 0], ki[..., 1], ki[..., 2]
    V = matter_potential(n_e, antinu=antinu).to(dtype=dtype)

    c2th12 = torch.cos(2.0 * th12)
    c2th13 = torch.cos(2.0 * th13)
    cth13 = torch.cos(th13)

    term1 = -4.0 * (k1 + k2 - 2.0 * k3) * (2.0 * k1 - k2 - k3) * (k1 - 2.0 * k2 + k3)

    term2 = 3.0 * (
        k1**2 - 4.0 * k1 * k2 + k2**2 + 2.0 * (k1 + k2) * k3 - 2.0 * k3**2
    ) * V

    term3 = 3.0 * (k1 + k2 - 2.0 * k3) * V**2
    term4 = -8.0 * V**3

    term5 = -18.0 * (k1 - k2) * V * (k1 + k2 - 2.0 * k3 + V) * c2th12 * cth13**2

    term6 = -9.0 * V * (
        k1**2 + k2**2 - 2.0 * k3 * (k3 + V)
        + k2 * (2.0 * k3 + V)
        + k1 * (-4.0 * k2 + 2.0 * k3 + V)
    ) * c2th13

    return (term1 + term2 + term3 + term4 + term5 + term6) / 108.0


@torch.no_grad()
def c1(
    ki: torch.Tensor,
    th12: TensorLike,
    th13: TensorLike,
    n_e: TensorLike,
    antinu: Union[bool, torch.Tensor],
) -> torch.Tensor:
    """
    Compute the c1 invariant of the reduced traceless Hamiltonian used by the analytical eigenvalue solver.
    
    Args:
        ki: Kinetic eigenvalue vector shaped (..., 3) in km^-1.
        th12: Solar mixing angle theta12 in radians.
        th13: Reactor mixing angle theta13 in radians.
        n_e: Electron density in mol/cm^3.
        antinu: Bool or boolean tensor; True selects antineutrino sign or complex-conjugated mixing.
    
    Returns:
        Tensor containing c1 with the broadcasted input shape.
    """
    device = ki.device
    dtype = ki.dtype

    th12 = _as_tensor(th12, device=device, dtype=dtype)
    th13 = _as_tensor(th13, device=device, dtype=dtype)
    n_e = _as_tensor(n_e, device=device, dtype=dtype)

    k1, k2, k3 = ki[..., 0], ki[..., 1], ki[..., 2]
    V = matter_potential(n_e, antinu=antinu).to(dtype=dtype)

    c2th12 = torch.cos(2.0 * th12)
    c2th13 = torch.cos(2.0 * th13)
    cth13 = torch.cos(th13)

    term1 = -4.0 * (k1**2 - k1 * k2 + k2**2 - (k1 + k2) * k3 + k3**2)
    term2 = (k1 + k2 - 2.0 * k3) * V
    term3 = -4.0 * V**2
    term4 = 6.0 * (-k1 + k2) * V * c2th12 * cth13**2
    term5 = -3.0 * (k1 + k2 - 2.0 * k3) * V * c2th13

    return (term1 + term2 + term3 + term4 + term5) / 12.0


@torch.no_grad()
def lambdas_cardano(
    c0: torch.Tensor,
    c1: torch.Tensor,
) -> torch.Tensor:
    """
    Solve the cubic traceless-Hamiltonian eigenvalue equation with the Cardano trigonometric form.
    
    Args:
        c0: Cubic invariant c0 of the traceless Hamiltonian.
        c1: Quadratic invariant c1 of the traceless Hamiltonian.
    
    Returns:
        Tensor shaped (..., 3) with ordered traceless eigenvalues in km^-1.
    """
    device = c0.device
    rdtype = c0.dtype
    cdtype = torch.complex128 if rdtype == torch.float64 else torch.complex64

    c0c = c0.to(dtype=cdtype)
    c1c = c1.to(dtype=cdtype)

    D = torch.sqrt(81.0 * c0c**2 + 12.0 * c1c**3)
    A = -9.0 * c0c + D

    A13 = torch.pow(A, 1.0 / 3.0)
    A23 = torch.pow(A, 2.0 / 3.0)

    two13 = 2.0 ** (1.0 / 3.0)
    three13 = 3.0 ** (1.0 / 3.0)
    six23 = 6.0 ** (2.0 / 3.0)

    l1 = (-2.0 * three13 * c1c + two13 * A23) / (six23 * A13)

    minus1_13 = torch.pow(torch.tensor(-1.0, device=device, dtype=cdtype), 1.0 / 3.0)
    minus2_13 = torch.pow(torch.tensor(-2.0, device=device, dtype=cdtype), 1.0 / 3.0)
    minus3_13 = torch.pow(torch.tensor(-3.0, device=device, dtype=cdtype), 1.0 / 3.0)

    l2 = minus1_13 * (2.0 * three13 * c1c + minus2_13 * A23) / (six23 * A13)
    l3 = -minus1_13 * (2.0 * minus3_13 * c1c + two13 * A23) / (six23 * A13)

    return torch.stack([l1, l2, l3], dim=-1)


@torch.no_grad()
def lambdas_eigvalsh(
    T: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Hermitian traceless-Hamiltonian eigenvalues with torch.linalg.eigvalsh.
    
    Args:
        T: Traceless Hermitian Hamiltonian tensor shaped (..., 3, 3).
    
    Returns:
        Tensor shaped (..., 3) with Hermitian eigenvalues in km^-1.
    """
    T = 0.5 * (T + T.conj().transpose(-1, -2))
    evals = torch.linalg.eigvalsh(T)

    return evals.real


def _lift_to_matrix_batch(
    x: torch.Tensor,
    target_ndim: int,
) -> torch.Tensor:
    """
    Add singleton dimensions to a tensor so it broadcasts against matrix-batched quantities.
    
    Args:
        x: Tensor to expand with trailing singleton dimensions.
        target_ndim: Target number of tensor dimensions required for broadcasting.
    
    Returns:
        Tensor with enough trailing singleton dimensions for matrix-batch broadcasting.
    """
    while x.ndim < target_ndim - 2:
        x = x.unsqueeze(-1)

    return x.unsqueeze(-1).unsqueeze(-1)


@torch.no_grad()
def Iab(
    la: torch.Tensor,
    lb: torch.Tensor,
    atilde: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    x2: torch.Tensor,
    x1: torch.Tensor,
    *,
    small_ratio: float = 1.0e-2,
    dl_zero_eps: float = 0.0,
) -> torch.Tensor:
    """
    Evaluate the analytical first-order oscillatory integral I_ab for polynomial density corrections.
    
    Args:
        la: Initial eigenvalue or eigenvalue batch in km^-1.
        lb: Final eigenvalue or eigenvalue batch in km^-1.
        atilde: Constant perturbation coefficient after subtracting the segment average density.
        b: Quadratic coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        c: Quartic coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        x2: Final segment coordinate in Earth-radius units or polynomial coordinate.
        x1: Initial segment coordinate in Earth-radius units or polynomial coordinate.
        small_ratio: Threshold on |la-lb|/|la+lb| below which the Taylor branch is used.
        dl_zero_eps: Absolute degeneracy threshold for treating la-lb as zero; 0 uses exact equality.
    
    Returns:
        Complex tensor containing I_ab for each eigenvalue pair and broadcasted segment.
    """
    target_ndim = la.ndim

    x1 = _lift_to_matrix_batch(x1, target_ndim)
    x2 = _lift_to_matrix_batch(x2, target_ndim)
    atilde = _lift_to_matrix_batch(atilde, target_ndim)
    b = _lift_to_matrix_batch(b, target_ndim)
    c = _lift_to_matrix_batch(c, target_ndim)

    if not torch.is_complex(la):
        cdtype = torch.complex128 if la.dtype == torch.float64 else torch.complex64
        la = la.to(dtype=cdtype)

    if not torch.is_complex(lb):
        lb = lb.to(dtype=la.dtype)

    cdtype = la.dtype
    device = la.device

    atilde = atilde.to(device=device, dtype=cdtype)
    b = b.to(device=device, dtype=cdtype)
    c = c.to(device=device, dtype=cdtype)
    x2 = x2.to(device=device, dtype=cdtype)
    x1 = x1.to(device=device, dtype=cdtype)

    Dl = la - lb
    dx = x2 - x1

    phase = torch.exp(1j * lb * (-x2 + x1))

    if dl_zero_eps > 0.0:
        is_zero = torch.abs(Dl) < dl_zero_eps
    else:
        is_zero = Dl == 0

    denom = la + lb
    ratio = torch.where(
        torch.abs(denom) > 0,
        torch.abs(Dl / denom),
        torch.full_like(torch.abs(Dl), float("inf")),
    )

    is_small = ratio < small_ratio

    t1 = Dl * (
        (-0.5j) * atilde * dx**2
        - (1j / 12.0) * b * (x2**4 - 4.0 * x2 * x1**3 + 3.0 * x1**4)
        - (1j / 30.0) * c * (x2**6 - 6.0 * x2 * x1**5 + 5.0 * x1**6)
    )

    t2 = Dl**2 * (
        -(atilde * dx**3) / 6.0
        - b * (x2**5 - 10.0 * x2**2 * x1**3 + 15.0 * x2 * x1**4 - 6.0 * x1**5) / 60.0
        - c * (x2**7 - 21.0 * x2**2 * x1**5 + 35.0 * x2 * x1**6 - 15.0 * x1**7) / 210.0
    )

    I_taylor = phase * (t1 + t2)

    exp_d = torch.exp(1j * Dl * dx)

    full_a = atilde * (-1j + 1j / exp_d) / Dl

    full_b = b * (
        2j + 2.0 * Dl * x2 - 1j * Dl**2 * x2**2
        + 1j * (-2.0 + 2j * Dl * x1 + Dl**2 * x1**2) / exp_d
    ) / Dl**3

    poly_c_x2 = 24.0 + Dl * x2 * (
        -24j + Dl * x2 * (-12.0 + Dl * x2 * (4j + Dl * x2))
    )

    poly_c_x1 = 24.0 + Dl * x1 * (
        -24j + Dl * x1 * (-12.0 + Dl * x1 * (4j + Dl * x1))
    )

    full_c = -1j * c * (poly_c_x2 - poly_c_x1 / exp_d) / Dl**5

    I_full = phase * (full_a + full_b + full_c)

    out = torch.where(is_small, I_taylor, I_full)
    out = torch.where(is_zero, torch.zeros_like(out), out)

    return out

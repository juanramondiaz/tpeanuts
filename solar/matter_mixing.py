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
Torch-native solar matter mixing utilities.

These functions mirror the original peanuts matter_mixing.py formulas.
"""



from __future__ import annotations

from typing import Union

import torch


TensorLike = Union[float, int, torch.Tensor]


def _as_tensor_like(x: TensorLike, reference: torch.Tensor | None = None) -> torch.Tensor:
    if reference is not None:
        return torch.as_tensor(x, device=reference.device, dtype=reference.dtype)

    if torch.is_tensor(x):
        return x

    return torch.tensor(x, dtype=torch.float64)


def _first_tensor(*values: TensorLike) -> torch.Tensor | None:
    for value in values:
        if torch.is_tensor(value):
            return value

    return None


def Vk(Deltam2: TensorLike, E: TensorLike, ne: TensorLike) -> torch.Tensor:
    reference = _first_tensor(E, ne, Deltam2)
    E_t = _as_tensor_like(E, reference)
    dm_t = _as_tensor_like(Deltam2, E_t)
    ne_t = _as_tensor_like(ne, E_t)

    return (3.868e-7 / 2.533) * (ne_t * E_t / dm_t)


def DeltamSqee(th12: TensorLike, DeltamSq21: TensorLike, DeltamSq3l: TensorLike) -> torch.Tensor:
    reference = _first_tensor(th12, DeltamSq21, DeltamSq3l)
    th12_t = _as_tensor_like(th12, reference)
    dm21 = _as_tensor_like(DeltamSq21, th12_t)
    dm3l = _as_tensor_like(DeltamSq3l, th12_t)

    dm31 = torch.where(dm3l > 0.0, dm3l, dm3l + dm21)
    dm32 = torch.where(dm3l < 0.0, dm3l, dm3l - dm21)

    return torch.cos(th12_t) ** 2 * dm31 + torch.sin(th12_t) ** 2 * dm32


def th13_M(
    th12: TensorLike,
    th13: TensorLike,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    E: TensorLike,
    ne: TensorLike,
) -> torch.Tensor:
    reference = _first_tensor(E, ne, th13, th12, DeltamSq21, DeltamSq3l)
    th13_t = _as_tensor_like(th13, reference)
    th12_t = _as_tensor_like(th12, th13_t)
    E_t = _as_tensor_like(E, th13_t)
    ne_t = _as_tensor_like(ne, E_t)

    vk = Vk(DeltamSqee(th12_t, DeltamSq21, DeltamSq3l), E_t, ne_t)
    numerator = torch.cos(2.0 * th13_t) - vk
    denominator = torch.sqrt(numerator**2 + torch.sin(2.0 * th13_t) ** 2)
    arg = torch.clamp(numerator / denominator, min=-1.0, max=1.0)

    return torch.remainder(0.5 * torch.arccos(arg), torch.pi / 2.0)


def th12_M(
    th12: TensorLike,
    th13: TensorLike,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    E: TensorLike,
    ne: TensorLike,
) -> torch.Tensor:
    reference = _first_tensor(E, ne, th12, th13, DeltamSq21, DeltamSq3l)
    th12_t = _as_tensor_like(th12, reference)
    th13_t = _as_tensor_like(th13, th12_t)
    E_t = _as_tensor_like(E, th12_t)
    ne_t = _as_tensor_like(ne, E_t)

    th13m = th13_M(th12_t, th13_t, DeltamSq21, DeltamSq3l, E_t, ne_t)
    dm21 = _as_tensor_like(DeltamSq21, th12_t)
    dm_ee = DeltamSqee(th12_t, DeltamSq21, DeltamSq3l)

    vk_prime = (
        Vk(dm21, E_t, ne_t) * torch.cos(th13m) ** 2
        + dm_ee / dm21 * torch.sin(th13m - th13_t) ** 2
    )

    numerator = torch.cos(2.0 * th12_t) - vk_prime
    denominator = torch.sqrt(
        numerator**2 + torch.sin(2.0 * th12_t) ** 2 * torch.cos(th13m - th13_t) ** 2
    )
    arg = torch.clamp(numerator / denominator, min=-1.0, max=1.0)

    return torch.remainder(0.5 * torch.arccos(arg), torch.pi / 2.0)

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
earth layer utilities for peanuts-torch.

This module contains helper functions for handling polynomial earth-density
layers returned by a density object.

The expected density API is:

    coeffs_all, xj_all, crossed = density.parameters_abc(eta_prime)

where:

    coeffs_all:
        Tensor with shape (B, Ns, Nc), containing polynomial coefficients.

    xj_all:
        Tensor with shape (B, Ns), containing shell-boundary coordinates.

    crossed:
        Boolean tensor with shape (B, Ns), indicating crossed shells.

Only the first three coefficients are used by the core peanuts segment
evolutor:

    n_e(x) = a + b x^2 + c x^4.
"""



from __future__ import annotations

import torch


def normalize_density_layer_output(
    coeffs_all: torch.Tensor,
    xj_all: torch.Tensor,
    crossed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if coeffs_all.ndim == 4 and coeffs_all.shape[0] == 1:
        coeffs_all = coeffs_all.squeeze(0)
        xj_all = xj_all.squeeze(0)
        crossed = crossed.squeeze(0)

    return coeffs_all, xj_all, crossed


def extract_quartic_coefficients(
    coeffs_all: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a = coeffs_all[..., 0]
    b = coeffs_all[..., 1]
    c = coeffs_all[..., 2]

    return a, b, c


def gather_batch_vector(
    mat: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    _, ns = mat.shape

    idx_long = idx.to(torch.long).clamp(min=0, max=ns - 1)

    return mat.gather(
        dim=-1,
        index=idx_long.unsqueeze(-1),
    ).squeeze(-1)


def outermost_crossed_shell(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    xj_all: torch.Tensor,
    crossed: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch_size, ns = xj_all.shape

    pos = torch.arange(ns, device=device)
    pos2 = pos.unsqueeze(0).expand(batch_size, ns)

    neg_inf = torch.full(
        (batch_size, ns),
        -1.0e30,
        device=device,
        dtype=dtype,
    )

    pos_masked = torch.where(
        crossed,
        pos2.to(dtype),
        neg_inf,
    )

    last_pos = pos_masked.max(dim=-1).values
    has_any = last_pos > -1.0e20

    pos_masked2 = torch.where(
        pos2.to(dtype) == last_pos.unsqueeze(-1),
        neg_inf,
        pos_masked,
    )

    second_last_pos = pos_masked2.max(dim=-1).values
    has_two = second_last_pos > -1.0e20

    a_o = gather_batch_vector(a, last_pos)
    b_o = gather_batch_vector(b, last_pos)
    c_o = gather_batch_vector(c, last_pos)

    x_start = torch.zeros(
        (batch_size,),
        device=device,
        dtype=dtype,
    )

    x_start = torch.where(
        has_two,
        gather_batch_vector(xj_all, second_last_pos),
        x_start,
    )

    return {
        "a_o": a_o,
        "b_o": b_o,
        "c_o": c_o,
        "x_start": x_start,
        "has_any": has_any,
        "has_two": has_two,
        "last_pos": last_pos,
        "second_last_pos": second_last_pos,
    }


def build_flipped_shell_segments(
    xj_all: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    crossed: torch.Tensor,
) -> dict[str, torch.Tensor]:
    xs2 = torch.flip(xj_all, dims=(-1,))
    a2 = torch.flip(a, dims=(-1,))
    b2 = torch.flip(b, dims=(-1,))
    c2 = torch.flip(c, dims=(-1,))
    crossed2 = torch.flip(crossed, dims=(-1,))

    x_hi = xs2

    x_lo = torch.zeros_like(xs2)
    x_lo[..., :-1] = xs2[..., 1:]
    x_lo[..., -1] = 0.0

    return {
        "x_hi": x_hi,
        "x_lo": x_lo,
        "a2": a2,
        "b2": b2,
        "c2": c2,
        "crossed2": crossed2,
    }
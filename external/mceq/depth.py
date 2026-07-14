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
MCEq-backed atmosphere-depth utilities.

This module adapts MCEq density/overburden models to the generic
atmosphere-depth helpers in :mod:`tpeanuts.medium.atmosphere.depth`.
Generic tensor-only operations live there; this module only contains the
parts that need an MCEq object or MCEq configuration.

Definitions:
    h:
        Altitude above sea level, in km.
    X_vertical(h):
        Vertical atmospheric depth, in g/cm^2, obtained from the active
        MCEq density model through ``density_model.get_mass_overburden``.
    X_slant(h, alpha):
        Slant atmospheric depth for a surface zenith angle alpha,
        computed as ``X_vertical(h) / cos(alpha)`` by the generic
        :func:`tpeanuts.medium.atmosphere.depth.atmosphere_slant_depth`
        helper.
    alpha:
        Surface/MCEq zenith angle in degrees. alpha=0 is vertical
        downward, and values must satisfy 0 <= alpha < 90 when a new
        MCEq object is initialized.

Module functions:
    vertical_atmosphere_depth:
        Evaluate the MCEq vertical atmospheric depth at one altitude for
        an already initialized MCEq object.
    vertical_atmosphere_depth_mceq:
        Vectorize ``vertical_atmosphere_depth`` over a scalar/tensor
        altitude grid, initializing MCEq from ``MCEqModelConfig`` when
        no MCEq object is supplied.
    atmosphere_slant_depth_mceq:
        Compute ``X_slant(h, alpha)`` by combining
        ``vertical_atmosphere_depth_mceq`` with the generic
        plane-parallel slant-depth projection.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from tpeanuts.medium.atmosphere.depth import (
    atmosphere_slant_depth as _atmosphere_slant_depth,
)
from tpeanuts.util.type import as_tensor
from tpeanuts.util.torch_util import default_device

from tpeanuts.external.mceq.config import MCEqModelConfig
from tpeanuts.external.mceq.core import init_mceq
from tpeanuts.external.mceq.density import get_density_model


TensorLike = Union[float, int, torch.Tensor]


def vertical_atmosphere_depth(
    mceq,
    h_km: float,
) -> float:
    """
    Evaluate MCEq vertical atmospheric depth at a single altitude.

    Args:
        mceq: Initialized MCEq object whose density_model implements
            ``get_mass_overburden`` with altitude in cm.
        h_km: Altitude above sea level in km.

    Returns:
        Vertical atmospheric depth in g/cm^2.

    Raises:
        AttributeError: If the MCEq density model does not expose
            ``get_mass_overburden``.
    """
    density_model = get_density_model(mceq)

    if not hasattr(density_model, "get_mass_overburden"):
        raise AttributeError(
            "The mceq density model does not provide get_mass_overburden."
        )

    h_cm = float(h_km) * 1.0e5

    return float(density_model.get_mass_overburden(h_cm))


@torch.no_grad()
def vertical_atmosphere_depth_mceq(
    h_km: TensorLike,
    mceq=None,
    alpha_deg: TensorLike = 0.0,
    config: Optional[MCEqModelConfig] = None,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Build a tensor-valued vertical atmospheric-depth profile from MCEq.

    Args:
        h_km: Scalar or tensor-like altitude values in km.
        mceq: Optional initialized MCEq object. If omitted, a new object
            is created through ``init_mceq``.
        alpha_deg: Surface/MCEq zenith angle used only when a new MCEq
            object must be initialized.
        config: Optional MCEqModelConfig used by ``init_mceq``.
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for input conversion and output.

    Returns:
        Tensor of vertical atmospheric depth values in g/cm^2 with the
        same shape as h_km.
    """
    dev = default_device(device)

    h_t = as_tensor(h_km, device=dev, dtype=dtype)
    original_shape = h_t.shape

    if mceq is None:
        mceq = init_mceq(
            alpha_deg=alpha_deg,
            config=config,
            info=False,
        )

    h_flat_cpu = h_t.detach().cpu().reshape(-1)

    X_vals = [
        vertical_atmosphere_depth(
            mceq=mceq,
            h_km=float(h.item()),
        )
        for h in h_flat_cpu
    ]

    return torch.tensor(
        X_vals,
        device=dev,
        dtype=dtype,
    ).reshape(original_shape)


@torch.no_grad()
def atmosphere_slant_depth_mceq(
    h_km: TensorLike,
    alpha_deg: TensorLike,
    mceq=None,
    config: Optional[MCEqModelConfig] = None,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Compute slant atmospheric depth X(h, alpha) using an MCEq model.

    Args:
        h_km: Scalar or tensor-like altitude values in km.
        alpha_deg: Surface/MCEq zenith angle in degrees.
        mceq: Optional initialized MCEq object. If omitted, a new object
            is created through ``init_mceq``.
        config: Optional MCEqModelConfig used by ``init_mceq``.
        device: Output torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for input conversion and output.

    Returns:
        Tensor of slant atmospheric depth values in g/cm^2 with the same
        shape as h_km.
    """
    dev = default_device(device)

    X_vertical = vertical_atmosphere_depth_mceq(
        h_km=h_km,
        mceq=mceq,
        alpha_deg=alpha_deg,
        config=config,
        device=dev,
        dtype=dtype,
    )

    return _atmosphere_slant_depth(
        X_vertical_gcm2=X_vertical,
        alpha_deg=alpha_deg,
        device=dev,
        dtype=dtype,
    )

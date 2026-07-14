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
nuSQuIDS density helper functions.

This module does not call the external nuSQuIDS package; it is a
tpeanuts-native (torch) reimplementation of the analytic exponential
atmosphere mass-density formula hard-coded inside nuSQuIDS's ``EarthAtm``
body. It exists so that tpeanuts's own propagation code (see
``medium.atmosphere.density``) can use exactly the same atmosphere density
profile as the nuSQuIDS reference backend, for an apples-to-apples
cross-validation of the two oscillation codes.

Module functions:
    atmosphere_density_nusquids(...)
        Evaluate the nuSQuIDS EarthAtm exponential atmosphere mass-density
        formula on a torch altitude grid.
"""

from __future__ import annotations

import torch

from tpeanuts.external.nusquids.core import NuSQuIDSConfig
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import TensorLike, as_tensor


@torch.no_grad()
def atmosphere_density_nusquids(
    h_km: TensorLike,
    config: NuSQuIDSConfig,
    *,
    context: RuntimeContext,
) -> torch.Tensor:
    """
    Evaluate the nuSQuIDS EarthAtm atmosphere mass-density formula.

    The nuSQuIDS ``EarthAtm`` body uses ``rho(h) = 0.0012 exp(-h / 7.594)``
    for points above the Earth surface, with ``h`` in km and ``rho`` in
    g/cm^3. This is a simple isothermal-exponential approximation of the
    atmosphere's density falloff with altitude (matter density decreasing
    roughly exponentially with a fixed scale height), distinct from the
    multi-layer empirical density used by the PyMSIS or MCEq backends.

    Args:
        h_km: Altitude above the Earth's surface in km. Scalar or
            broadcastable tensor.
        config: NuSQuIDSConfig providing the reference density
            ``nusquids_rho0_gcm3`` (rho0, in g/cm^3) and scale height
            ``nusquids_scale_height_km`` (H, in km) used in the formula.
        context: Runtime device/dtype used for inputs and output.

    Returns:
        Tensor containing rho(h) = rho0 * exp(-h/H) in g/cm^3, with the
        broadcast shape of h_km.
    """
    h_km = as_tensor(h_km, device=context.device, dtype=context.dtype)
    rho0 = as_tensor(
        config.nusquids_rho0_gcm3,
        device=h_km.device,
        dtype=context.dtype,
    )
    scale_height = as_tensor(
        config.nusquids_scale_height_km,
        device=h_km.device,
        dtype=context.dtype,
    )
    return rho0 * torch.exp(-h_km / scale_height)

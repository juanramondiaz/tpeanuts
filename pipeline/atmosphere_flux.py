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
Flux propagation utilities for atmosphere neutrinos.

This module combines atmosphere propagation and earth propagation
to build flavour-transition probability matrices and propagate
atmosphere neutrino fluxes from production height to detector.

Flavour ordering:

    [nue, numu, nutau]

The propagated flux is:

    Phi_det_beta = sum_alpha P[beta, alpha] Phi_prod_alpha

Module functions:
    
    _check_flux_dict_shapes(...)
        Validates flavour-key presence and (n_E, n_h) grid consistency for
        height-differential atmosphere flux dictionaries.
    
    build_probability_matrix(...)
        Builds the combined atmosphere-plus-Earth transition matrix
        P=|S_earth S_atm|^2 for a production height, energy, and angle.
    
    propagate_flux_vector(...)
        Applies the transition matrix to a single flavour-flux vector ordered
        as [nue, numu, nutau].
    
    propagate_flux_E_h(...)
        Propagates full Phi(E,h) grids for each flavour and optionally caches
        per-grid-point probability matrices.
    
    integrate_flux_over_height(...), integrate_detector_flux_over_height(...)
        Integrate height-differential fluxes over the production-height grid.
"""





from __future__ import annotations

from typing import Optional
import torch

from tpeanuts.medium.earth.profile import EarthParameters, EarthProfile

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.medium.atmosphere.evolutor import atmosphere_evolutor
from tpeanuts.medium.atmosphere.profile import AtmosphereParameters
from tpeanuts.medium.earth.evolutor import earth_evolutor_from_zenith
from tpeanuts.core.common.probability import (
    probability_incoherent,
    probability_transition,
)

from tpeanuts.pipeline.pipeline_common import prepare_earth_profile
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import TensorLike, as_tensor


# ============================================================
# Helpers
# ============================================================

def _check_flux_dict_shapes(
    E_grid_GeV: torch.Tensor,
    h_grid_km: torch.Tensor,
    phi_E_h_flavours: dict,
    required=("nue", "numu", "nutau"),
):
    """
    Validate flavour-flux dictionary shapes on an (E, h) grid.

    Args:
        E_grid_GeV: 1D tensor of energy-grid points in GeV with shape (n_E,).
        h_grid_km: 1D tensor of height-grid points in km with shape (n_h,).
        phi_E_h_flavours: Dictionary mapping flavour names to tensors with
            shape (n_E, n_h).
        required: Iterable of required flavour keys. Defaults to nue, numu,
            and nutau.

    Returns:
        None. Raises KeyError or ValueError when required flavours or shapes
        are missing.
    """
    n_E = E_grid_GeV.numel()
    n_h = h_grid_km.numel()

    for key in required:
        if key not in phi_E_h_flavours:
            raise KeyError(f"Missing input flavour: {key}")

        arr = phi_E_h_flavours[key]

        if arr.shape != (n_E, n_h):
            raise ValueError(
                f"phi_E_h_flavours['{key}'] must have shape {(n_E, n_h)}, "
                f"got {tuple(arr.shape)}."
            )

# ============================================================
# Probability matrix
# ============================================================

@torch.no_grad()
def build_probability_matrix(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    h_km: TensorLike,
    theta_deg: TensorLike,
    detector_depth_m: TensorLike = 0.0,
    profile_earth: Optional[EarthProfile] = None,
    earth: EarthParameters = EarthParameters(),
    atmosphere: AtmosphereParameters = AtmosphereParameters(),
    reunitarize_earth: bool = False,
    *,
    context: RuntimeContext = RuntimeContext.resolve(None, torch.float64),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the full atmosphere-plus-Earth flavour probability matrix.

    Args:
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy in MeV. Scalar or tensor.
        h_km: Atmosphere production height in km.
        theta_deg: Atmosphere zenith angle in degrees.
        detector_depth_m: Detector depth in meters.
        profile_earth: Optional already-built EarthProfile object. When
            omitted, one is built from ``earth``.
        earth: Earth electron-density profile construction settings.
        atmosphere: Atmosphere density profile construction settings.
        reunitarize_earth: If True, reunitarize the Earth evolution operator.
        context: Runtime device/dtype for inputs.

    Returns:
        Pair (P, S_total), where P=|S_total|^2 is real and S_total is complex.
        Both have shape (..., 3, 3).
    """
    dev, dtype = context.device, context.dtype

    profile_earth, _ = prepare_earth_profile(
        profile_earth,
        earth=earth,
        context=context,
    )

    S_atm, _ = atmosphere_evolutor(
        oscillation,
        E_MeV,
        h_km,
        theta_deg,
        as_tensor(detector_depth_m, device=dev, dtype=dtype) / 1.0e3,
        atmosphere=atmosphere,
        context=context,
    )

    S_earth = earth_evolutor_from_zenith(
        profile_earth=profile_earth,
        oscillation=oscillation,
        E_MeV=E_MeV,
        theta_deg=theta_deg,
        depth_m=float(as_tensor(detector_depth_m, device="cpu", dtype=dtype).item()),
        reunitarize=reunitarize_earth,
    )

    S_total = torch.matmul(S_earth, S_atm)

    P = probability_transition(S_total, real_dtype=dtype)

    return P, S_total


# ============================================================
# flux-vector propagation
# ============================================================

@torch.no_grad()
def propagate_flux_vector(
    flux_flavour: TensorLike,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    h_km: TensorLike,
    theta_deg: TensorLike,
    detector_depth_m: TensorLike = 0.0,
    profile_earth: Optional[EarthProfile] = None,
    earth: EarthParameters = EarthParameters(),
    atmosphere: AtmosphereParameters = AtmosphereParameters(),
    reunitarize_earth: bool = False,
    *,
    context: RuntimeContext = RuntimeContext.resolve(None, torch.float64),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Propagate a single flavour-flux vector to the detector.

    Args:
        flux_flavour: Real tensor with last dimension 3 ordered as
            [nue, numu, nutau].
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy in MeV.
        h_km: Production height in km.
        theta_deg: Atmosphere zenith angle in degrees.
        detector_depth_m: Detector depth in meters.
        profile_earth: Optional already-built EarthProfile object.
        earth: Earth electron-density profile construction settings.
        atmosphere: Atmosphere density profile construction settings.
        reunitarize_earth: If True, reunitarize Earth evolution.
        context: Runtime device/dtype for flux and probability tensors.

    Returns:
        Pair (flux_detector, P). flux_detector has last dimension 3 and P has
        shape (..., 3, 3).
    """
    flux_flavour = as_tensor(
        flux_flavour,
        device=context.device,
        dtype=context.dtype,
    )

    if flux_flavour.shape[-1] != 3:
        raise ValueError("flux_flavour must have last dimension 3.")

    P, _ = build_probability_matrix(
        oscillation,
        E_MeV,
        h_km,
        theta_deg,
        detector_depth_m,
        profile_earth=profile_earth,
        earth=earth,
        atmosphere=atmosphere,
        reunitarize_earth=reunitarize_earth,
        context=context,
    )

    flux_detector = probability_incoherent(P, flux_flavour)

    return flux_detector, P


# ============================================================
# Grid propagation: Phi(E, h)
# ============================================================

@torch.no_grad()
def propagate_flux_E_h(
    E_grid_GeV: TensorLike,
    h_grid_km: TensorLike,
    phi_E_h_flavours: dict,
    theta_deg: TensorLike,
    oscillation: OscillationParameters,
    detector_depth_m: TensorLike = 0.0,
    profile_earth: Optional[EarthProfile] = None,
    earth: EarthParameters = EarthParameters(),
    atmosphere: AtmosphereParameters = AtmosphereParameters(),
    store_probabilities: bool = False,
    reunitarize_earth: bool = False,
    verbose: bool = True,
    *,
    context: RuntimeContext = RuntimeContext.resolve(None, torch.float64),
) -> tuple[dict, Optional[dict]]:
    """
    Propagate height-differential fluxes on an energy-height grid.

    Args:
        E_grid_GeV: 1D energy grid in GeV with shape (n_E,).
        h_grid_km: 1D production-height grid in km with shape (n_h,).
        phi_E_h_flavours: Dictionary with keys "nue", "numu", "nutau" and
            tensors of shape (n_E, n_h).
        theta_deg: Atmospheric zenith angle in degrees for this flux slice.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        detector_depth_m: Detector depth in meters.
        profile_earth: Optional already-built EarthProfile object.
        earth: Earth electron-density profile construction settings.
        atmosphere: Atmosphere density profile construction settings.
        store_probabilities: If True, cache each (E, h) probability matrix.
        reunitarize_earth: If True, reunitarize Earth evolution.
        verbose: If True, print grid progress information.
        context: Runtime device/dtype for grids and fluxes.

    Returns:
        Pair (phi_detector, P_cache). phi_detector maps each flavour to a
        tensor of shape (n_E, n_h). P_cache is None unless requested; then it
        maps (i_E, i_h) indices to 3x3 probability matrices.
    """
    dev, dtype = context.device, context.dtype

    profile_earth, _ = prepare_earth_profile(
        profile_earth,
        earth=earth,
        context=context,
    )

    E_grid_GeV = as_tensor(E_grid_GeV, device=dev, dtype=dtype)
    h_grid_km = as_tensor(h_grid_km, device=dev, dtype=dtype)

    phi_E_h_flavours = {
        key: as_tensor(value, device=dev, dtype=dtype)
        for key, value in phi_E_h_flavours.items()
    }

    required = ("nue", "numu", "nutau")

    _check_flux_dict_shapes(
        E_grid_GeV=E_grid_GeV,
        h_grid_km=h_grid_km,
        phi_E_h_flavours=phi_E_h_flavours,
        required=required,
    )

    n_E = E_grid_GeV.numel()
    n_h = h_grid_km.numel()

    if verbose:
        print(
            f"Propagating atmosphere grid: n_E={n_E}, n_h={n_h}, "
            f"device={dev}"
        )

    E_MeV_grid = (1.0e3 * E_grid_GeV)[:, None].expand(n_E, n_h)
    h_grid = h_grid_km[None, :].expand(n_E, n_h)

    flux_in = torch.stack(
        [
            phi_E_h_flavours["nue"],
            phi_E_h_flavours["numu"],
            phi_E_h_flavours["nutau"],
        ],
        dim=-1,
    )

    P, _ = build_probability_matrix(
        oscillation,
        E_MeV_grid,
        h_grid,
        theta_deg,
        detector_depth_m,
        profile_earth=profile_earth,
        earth=earth,
        atmosphere=atmosphere,
        reunitarize_earth=reunitarize_earth,
        context=context,
    )

    flux_out = probability_incoherent(P, flux_in)

    phi_detector = {
        "nue": flux_out[..., 0],
        "numu": flux_out[..., 1],
        "nutau": flux_out[..., 2],
    }

    P_cache = None
    if store_probabilities:
        P_cache = {
            (i_E, i_h): P[i_E, i_h].detach().clone()
            for i_E in range(n_E)
            for i_h in range(n_h)
        }

    return phi_detector, P_cache


# ============================================================
# Helper: integrated over height
# ============================================================

@torch.no_grad()
def integrate_flux_over_height(
    h_grid_km: TensorLike,
    phi_E_h: TensorLike,
    *,
    context: RuntimeContext = RuntimeContext.resolve(None, torch.float64),
) -> torch.Tensor:
    """
    Integrate a height-differential flux over production height.

    Args:
        h_grid_km: 1D height grid in km with shape (n_h,).
        phi_E_h: Tensor of flux values with shape (n_E, n_h), or any tensor
            whose height axis is dimension 1.
        context: Runtime device/dtype for integration.

    Returns:
        Tensor integrated over height using torch.trapezoid along dim=1.
    """
    h_grid_km = as_tensor(h_grid_km, device=context.device, dtype=context.dtype)
    phi_E_h = as_tensor(phi_E_h, device=h_grid_km.device, dtype=context.dtype)

    return torch.trapezoid(
        y=phi_E_h,
        x=h_grid_km,
        dim=1,
    )


@torch.no_grad()
def integrate_detector_flux_over_height(
    h_grid_km: TensorLike,
    phi_detector: dict,
    *,
    context: RuntimeContext = RuntimeContext.resolve(None, torch.float64),
) -> dict:
    """
    Integrate all detector-flavour fluxes over production height.

    Args:
        h_grid_km: 1D height grid in km with shape (n_h,).
        phi_detector: Dictionary mapping flavour names to tensors shaped
            (n_E, n_h).
        context: Runtime device/dtype for integration.

    Returns:
        Dictionary with the same keys as phi_detector and height-integrated
        tensors for each flavour.
    """
    return {
        key: integrate_flux_over_height(
            h_grid_km=h_grid_km,
            phi_E_h=value,
            context=context,
        )
        for key, value in phi_detector.items()
    }

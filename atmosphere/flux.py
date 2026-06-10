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
flux propagation utilities for atmospheric neutrinos.

This module combines atmospheric propagation and earth propagation
to build flavour-transition probability matrices and propagate
atmospheric neutrino fluxes from production height to detector.

Flavour ordering:

    [nue, numu, nutau]

The propagated flux is:

    Phi_det_beta = sum_alpha P[beta, alpha] Phi_prod_alpha

Module functions:
    
    _check_flux_dict_shapes(...)
        Validates flavour-key presence and (n_E, n_h) grid consistency for
        height-differential atmospheric flux dictionaries.
    
    build_probability_matrix(...)
        Builds the combined atmospheric-plus-Earth transition matrix
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

from typing import Optional, Union, Callable
import torch

from tpeanuts.earth.density import EarthDensity
from tpeanuts.io.io_earth import load_earth_density_from_csv

from tpeanuts.atmosphere.propagation import atmospheric_evolution_operator
from tpeanuts.atmosphere.earth import earth_evolution_operator

from tpeanuts.atmosphere.density import atmospheric_electron_density_profile

from tpeanuts.util.type import _as_tensor
from tpeanuts.util.torch_util import _default_device

TensorLike = Union[float, int, torch.Tensor]

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
    pmns,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    E_MeV: TensorLike,
    h_km: TensorLike,
    theta_deg: TensorLike,
    detector_depth_m: TensorLike = 0.0,
    density: Optional[EarthDensity] = None,
    density_file: Optional[str] = None,
    antinu: Union[bool, torch.Tensor] = False,
    atmosphere_matter: bool = True,
    atmosphere_ne_profile: Optional[Callable] = None,
    atmosphere_n_steps: int = 600,
    reunitarize_earth: bool = False,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the full atmosphere-plus-Earth flavour probability matrix.

    Args:
        pmns: PMNS object or compatible mixing container.
        DeltamSq21: Delta m^2_21 in eV^2.
        DeltamSq3l: Delta m^2_3l in eV^2.
        E_MeV: Neutrino energy in MeV. Scalar or tensor.
        h_km: Atmospheric production height in km.
        theta_deg: Atmospheric zenith angle in degrees.
        detector_depth_m: Detector depth in meters.
        density: Optional EarthDensity object.
        density_file: Earth-density CSV used when density is None.
        antinu: Bool or tensor mask for antineutrino propagation.
        atmosphere_matter: If False, disable atmospheric matter effects.
        atmosphere_ne_profile: Optional callable returning ne in mol/cm^3.
        atmosphere_n_steps: Number of atmospheric path grid points.
        reunitarize_earth: If True, reunitarize the Earth evolution operator.
        device: Optional torch device.
        dtype: Real dtype for inputs.

    Returns:
        Pair (P, S_total), where P=|S_total|^2 is real and S_total is complex.
        Both have shape (..., 3, 3).
    """
    dev = _default_device(device)

    if density is None:
        if density_file is None:
            raise ValueError(
                "Provide either density=EarthDensity(...) or density_file."
            )

        density = load_earth_density_from_csv(
            density_file,
            device=dev,
            dtype=dtype,
        )

    if atmosphere_ne_profile is None:
        atmosphere_ne_profile = atmospheric_electron_density_profile

    S_atm, _ = atmospheric_evolution_operator(
        pmns=pmns,
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=E_MeV,
        h_km=h_km,
        theta_deg=theta_deg,
        depth_km=_as_tensor(detector_depth_m, device=dev, dtype=dtype) / 1.0e3,
        antinu=antinu,
        ne_profile=atmosphere_ne_profile,
        n_steps=atmosphere_n_steps,
        matter=atmosphere_matter,
        device=dev,
        dtype=dtype,
    )

    S_earth = earth_evolution_operator(
        pmns=pmns,
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=E_MeV,
        theta_deg=theta_deg,
        detector_depth_m=detector_depth_m,
        density=density,
        density_file=None,
        antinu=antinu,
        reunitarize=reunitarize_earth,
        device=dev,
        dtype=dtype,
    )

    S_total = torch.matmul(S_earth, S_atm)

    P = torch.abs(S_total) ** 2

    return P, S_total


# ============================================================
# flux-vector propagation
# ============================================================

@torch.no_grad()
def propagate_flux_vector(
    flux_flavour: TensorLike,
    pmns,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    E_MeV: TensorLike,
    h_km: TensorLike,
    theta_deg: TensorLike,
    detector_depth_m: TensorLike = 0.0,
    density: Optional[EarthDensity] = None,
    density_file: Optional[str] = None,
    antinu: Union[bool, torch.Tensor] = False,
    atmosphere_matter: bool = True,
    atmosphere_ne_profile: Optional[Callable] = None,
    atmosphere_n_steps: int = 600,
    reunitarize_earth: bool = False,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Propagate a single flavour-flux vector to the detector.

    Args:
        flux_flavour: Real tensor with last dimension 3 ordered as
            [nue, numu, nutau].
        pmns: PMNS object or compatible mixing container.
        DeltamSq21: Delta m^2_21 in eV^2.
        DeltamSq3l: Delta m^2_3l in eV^2.
        E_MeV: Neutrino energy in MeV.
        h_km: Production height in km.
        theta_deg: Atmospheric zenith angle in degrees.
        detector_depth_m: Detector depth in meters.
        density: Optional EarthDensity object.
        density_file: Earth-density CSV used when density is None.
        antinu: Bool or tensor mask for antineutrino propagation.
        atmosphere_matter: If False, ignore atmospheric matter.
        atmosphere_ne_profile: Optional atmospheric electron-density callable.
        atmosphere_n_steps: Number of atmospheric grid points.
        reunitarize_earth: If True, reunitarize Earth evolution.
        device: Optional torch device.
        dtype: Real dtype for flux and probability tensors.

    Returns:
        Pair (flux_detector, P). flux_detector has last dimension 3 and P has
        shape (..., 3, 3).
    """
    dev = _default_device(device)

    flux_flavour = _as_tensor(
        flux_flavour,
        device=dev,
        dtype=dtype,
    )

    if flux_flavour.shape[-1] != 3:
        raise ValueError("flux_flavour must have last dimension 3.")

    P, _ = build_probability_matrix(
        pmns=pmns,
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=E_MeV,
        h_km=h_km,
        theta_deg=theta_deg,
        detector_depth_m=detector_depth_m,
        density=density,
        density_file=density_file,
        antinu=antinu,
        atmosphere_matter=atmosphere_matter,
        atmosphere_ne_profile=atmosphere_ne_profile,
        atmosphere_n_steps=atmosphere_n_steps,
        reunitarize_earth=reunitarize_earth,
        device=dev,
        dtype=dtype,
    )

    flux_detector = torch.matmul(
        P,
        flux_flavour[..., None],
    ).squeeze(-1)

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
    pmns,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    detector_depth_m: TensorLike = 0.0,
    density: Optional[EarthDensity] = None,
    density_file: Optional[str] = None,
    antinu: Union[bool, torch.Tensor] = False,
    atmosphere_matter: bool = True,
    atmosphere_ne_profile: Optional[Callable] = None,
    atmosphere_n_steps: int = 600,
    store_probabilities: bool = False,
    reunitarize_earth: bool = False,
    verbose: bool = True,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[dict, Optional[dict]]:
    """
    Propagate height-differential fluxes on an energy-height grid.

    Args:
        E_grid_GeV: 1D energy grid in GeV with shape (n_E,).
        h_grid_km: 1D production-height grid in km with shape (n_h,).
        phi_E_h_flavours: Dictionary with keys "nue", "numu", "nutau" and
            tensors of shape (n_E, n_h).
        theta_deg: Atmospheric zenith angle in degrees for this flux slice.
        pmns: PMNS object or compatible mixing container.
        DeltamSq21: Delta m^2_21 in eV^2.
        DeltamSq3l: Delta m^2_3l in eV^2.
        detector_depth_m: Detector depth in meters.
        density: Optional EarthDensity object.
        density_file: Earth-density CSV used when density is None.
        antinu: Bool or tensor mask for antineutrino propagation.
        atmosphere_matter: If False, ignore atmospheric matter.
        atmosphere_ne_profile: Optional electron-density callable.
        atmosphere_n_steps: Number of atmospheric path grid points.
        store_probabilities: If True, cache each (E, h) probability matrix.
        reunitarize_earth: If True, reunitarize Earth evolution.
        verbose: If True, print grid progress information.
        device: Optional torch device.
        dtype: Real dtype for grids and fluxes.

    Returns:
        Pair (phi_detector, P_cache). phi_detector maps each flavour to a
        tensor of shape (n_E, n_h). P_cache is None unless requested; then it
        maps (i_E, i_h) indices to 3x3 probability matrices.
    """
    dev = _default_device(device)

    if density is None:
        if density_file is None:
            raise ValueError(
                "Provide either density=EarthDensity(...) or density_file."
            )

        density = load_earth_density_from_csv(
            density_file,
            device=dev,
            dtype=dtype,
        )

    E_grid_GeV = _as_tensor(E_grid_GeV, device=dev, dtype=dtype)
    h_grid_km = _as_tensor(h_grid_km, device=dev, dtype=dtype)

    phi_E_h_flavours = {
        key: _as_tensor(value, device=dev, dtype=dtype)
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
            f"Propagating atmospheric grid: n_E={n_E}, n_h={n_h}, "
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
        pmns=pmns,
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=E_MeV_grid,
        h_km=h_grid,
        theta_deg=theta_deg,
        detector_depth_m=detector_depth_m,
        density=density,
        density_file=None,
        antinu=antinu,
        atmosphere_matter=atmosphere_matter,
        atmosphere_ne_profile=atmosphere_ne_profile,
        atmosphere_n_steps=atmosphere_n_steps,
        reunitarize_earth=reunitarize_earth,
        device=dev,
        dtype=dtype,
    )

    flux_out = torch.einsum(
        "...ba,...a->...b",
        P,
        flux_in,
    )

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
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Integrate a height-differential flux over production height.

    Args:
        h_grid_km: 1D height grid in km with shape (n_h,).
        phi_E_h: Tensor of flux values with shape (n_E, n_h), or any tensor
            whose height axis is dimension 1.
        device: Optional torch device.
        dtype: Real dtype for integration.

    Returns:
        Tensor integrated over height using torch.trapezoid along dim=1.
    """
    h_grid_km = _as_tensor(h_grid_km, device=device, dtype=dtype)
    phi_E_h = _as_tensor(phi_E_h, device=h_grid_km.device, dtype=dtype)

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
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> dict:
    """
    Integrate all detector-flavour fluxes over production height.

    Args:
        h_grid_km: 1D height grid in km with shape (n_h,).
        phi_detector: Dictionary mapping flavour names to tensors shaped
            (n_E, n_h).
        device: Optional torch device.
        dtype: Real dtype for integration.

    Returns:
        Dictionary with the same keys as phi_detector and height-integrated
        tensors for each flavour.
    """
    return {
        key: integrate_flux_over_height(
            h_grid_km=h_grid_km,
            phi_E_h=value,
            device=device,
            dtype=dtype,
        )
        for key, value in phi_detector.items()
    }

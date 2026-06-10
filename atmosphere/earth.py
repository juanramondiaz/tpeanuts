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
earth propagation utilities for atmospheric neutrinos.

This module propagates neutrino states from the earth surface to an
underground detector using peanuts.

The atmospheric zenith angle theta is converted into the peanuts nadir
angle eta using:

    eta = pi - theta

Convention
----------
Atmospheric detector zenith angle theta:

    theta = 0 deg   -> downward-going
    theta = 90 deg  -> horizontal
    theta = 180 deg -> upward-going

The earth propagation is handled by the torch-native EarthDensity model and
earth_evolutor machinery.

Module functions:
    
    earth_evolution_operator(...)
        Converts atmospheric zenith angle theta to peanuts nadir angle eta,
        loads or receives an EarthDensity object, and calls earth_evolutor to
        build the surface-to-detector matter evolution operator.
    
    propagate_surface_to_detector(...)
        Applies the Earth evolution operator to a coherent flavour state at
        the surface and returns the detector flavour amplitudes.
    
Notes:
    Probability calculations are intentionally delegated to
    tpeanuts.earth.probabilities.pearth(..., method=...) to avoid wrapper
    duplication in the atmospheric namespace.
"""




from __future__ import annotations

from typing import Optional, Union
import torch

from tpeanuts.earth.density import EarthDensity
from tpeanuts.earth.evolutor import earth_evolutor
from tpeanuts.io.io_earth import load_earth_density_from_csv

from tpeanuts.atmosphere.geometry import theta_to_eta

from tpeanuts.util.type import _as_tensor,_cdtype_from_real
from tpeanuts.util.torch_util import _default_device

TensorLike = Union[float, int, torch.Tensor]


# ============================================================
# earth evolutor
# ============================================================

@torch.no_grad()
def earth_evolution_operator(
    pmns,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    E_MeV: TensorLike,
    theta_deg: TensorLike,
    detector_depth_m: TensorLike = 0.0,
    density: Optional[EarthDensity] = None,
    density_file: Optional[str] = None,
    antinu: Union[bool, torch.Tensor] = False,
    *,
    reunitarize=False,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Build the Earth matter evolution operator for an atmospheric trajectory.

    Args:
        pmns: PMNS object or compatible mixing container.
        DeltamSq21: Delta m^2_21 in eV^2. Scalar or tensor.
        DeltamSq3l: Delta m^2_3l in eV^2. Scalar or tensor.
        E_MeV: Neutrino energy in MeV.
        theta_deg: Atmospheric zenith angle in degrees, converted internally
            to peanuts nadir angle eta.
        detector_depth_m: Detector depth below surface in meters.
        density: Optional EarthDensity object. If None, density_file is loaded.
        density_file: Path to Earth-density CSV used when density is None.
        antinu: Bool or tensor mask for antineutrino propagation.
        reunitarize: If True, request reunitarization in the Earth evolutor.
        device: Optional torch device.
        dtype: Real dtype for geometry and Hamiltonian inputs.

    Returns:
        Complex tensor S_earth with shape (..., 3, 3), mapping surface flavour
        amplitudes to detector flavour amplitudes.
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

    theta_deg = _as_tensor(theta_deg, device=dev, dtype=dtype)
    E_MeV = _as_tensor(E_MeV, device=dev, dtype=dtype)
    DeltamSq21 = _as_tensor(DeltamSq21, device=dev, dtype=dtype)
    DeltamSq3l = _as_tensor(DeltamSq3l, device=dev, dtype=dtype)

    detector_depth_m = _as_tensor(
        detector_depth_m,
        device=dev,
        dtype=dtype,
    )

    eta = theta_to_eta(
        theta_deg,
        device=dev,
        dtype=dtype,
    )

    S_earth = earth_evolutor(
        density=density,
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        pmns=pmns,
        E=E_MeV,
        eta=eta,
        depth_m=detector_depth_m,
        antinu=antinu,
        reunitarize=reunitarize,
    )

    return S_earth


# ============================================================
# State propagation
# ============================================================

@torch.no_grad()
def propagate_surface_to_detector(
    nustate_surface: TensorLike,
    pmns,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    E_MeV: TensorLike,
    theta_deg: TensorLike,
    detector_depth_m: TensorLike = 0.0,
    density: Optional[EarthDensity] = None,
    density_file: Optional[str] = None,
    antinu: Union[bool, torch.Tensor] = False,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Propagate a coherent flavour state from surface to detector.

    Args:
        nustate_surface: Complex or real flavour-amplitude tensor with last
            dimension 3.
        pmns: PMNS object or compatible mixing container.
        DeltamSq21: Delta m^2_21 in eV^2.
        DeltamSq3l: Delta m^2_3l in eV^2.
        E_MeV: Neutrino energy in MeV.
        theta_deg: Atmospheric zenith angle in degrees.
        detector_depth_m: Detector depth in meters.
        density: Optional EarthDensity object.
        density_file: Density CSV path used if density is None.
        antinu: Bool or tensor mask for antineutrino propagation.
        device: Optional torch device.
        dtype: Real dtype for inputs; complex dtype is inferred for state.

    Returns:
        Complex tensor of detector flavour amplitudes with last dimension 3.
    """
    dev = _default_device(device)
    cdtype = _cdtype_from_real(dtype)

    S_earth = earth_evolution_operator(
        pmns=pmns,
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=E_MeV,
        theta_deg=theta_deg,
        detector_depth_m=detector_depth_m,
        density=density,
        density_file=density_file,
        antinu=antinu,
        device=dev,
        dtype=dtype,
    )

    nustate_surface = _as_tensor(
        nustate_surface,
        device=dev,
        dtype=cdtype,
    )

    if nustate_surface.shape[-1] != 3:
        raise ValueError(
            "nustate_surface must have last dimension 3."
        )

    target_shape = S_earth.shape[:-1]

    while nustate_surface.ndim < len(target_shape):
        nustate_surface = nustate_surface.unsqueeze(0)

    nustate_surface = nustate_surface.expand(*S_earth.shape[:-1])

    nustate_detector = torch.matmul(
        S_earth,
        nustate_surface[..., None],
    ).squeeze(-1)

    return nustate_detector

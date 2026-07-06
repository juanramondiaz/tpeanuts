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
Atmosphere density profiles for neutrino propagation.

This module provides a common interface for evaluating atmosphere mass and
electron densities on scalar or tensor altitude grids. Density data can be
obtained from an exponential approximation, the external nuSQuIDS EarthAtm
atmosphere formula, a two-column data file, an MCEq atmosphere model, or the optional
PyMSIS backend.

The units used by the public functions are:

    h_km             : km above the Earth's surface
    mass_density     : g / cm^3
    electron_density : mol / cm^3

Electron density is obtained from mass density using the electron fraction Ye
and the shared mass-to-molar-density conversion constant. PyMSIS calculations
use the electron fraction stored in their backend configuration.

Module functions:

    atmosphere_mass_density_profile_exponential(...)
        Evaluate an exponential atmosphere mass-density approximation.

    atmosphere_mass_density_profile_from_file(...)
        Load and interpolate a two-column altitude-density profile.

    atmosphere_density(...)
        Select a density backend and return either mass or electron density.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch

import tpeanuts.util.default as default
from tpeanuts.util.io import load_datafile_2column
from tpeanuts.util.constant import GCM3_TO_NUCLEON_MOLCM3
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.math import interp1d_linear
from tpeanuts.util.type import TensorLike, as_tensor

from tpeanuts.external.nusquids.core import NuSQuIDSConfig
from tpeanuts.external.nusquids.density import (
    atmosphere_mass_density_profile_nusquids,
)
from tpeanuts.external.mceq.config import MCEqModelConfig
try:
    from tpeanuts.external.mceq import density as mceq_density

    atmosphere_mass_density_profile_from_mceq = getattr(
        mceq_density,
        "atmosphere_mass_density_profile_from_mceq",
    )
except (ImportError, AttributeError):
    atmosphere_mass_density_profile_from_mceq = None


try:
    from tpeanuts.external.pymsis.density import (
        PyMSISatmosphereConfig,
        atmosphere_density_pymsis,
    )
except ImportError:
    PyMSISatmosphereConfig = None
    atmosphere_density_pymsis = None


DensityType = Literal["electron_density", "mass_density"]


@torch.no_grad()
def atmosphere_mass_density_profile_exponential(
    h_km: TensorLike,
    rho0_gcm3: TensorLike = default.atmosphere_rho0_gcm3,
    scale_height_km: TensorLike = default.atmosphere_scale_height_km,
    *,
    context: RuntimeContext,
) -> torch.Tensor:
    """
    Evaluate an exponential atmosphere mass-density profile.

    Args:
        h_km: Altitude above the Earth's surface in km. Scalar or
            broadcastable tensor.
        rho0_gcm3: Reference mass density at zero altitude in g/cm^3.
        scale_height_km: Exponential scale height in km. Must be broadcastable
            with h_km.
        context: Runtime device/dtype used for inputs and output.

    Returns:
        Tensor containing rho(h) = rho0 exp(-h/H) in g/cm^3, with the
        broadcast shape of the input values.
    """
    h_km = as_tensor(h_km, device=context.device, dtype=context.dtype)
    rho0 = as_tensor(rho0_gcm3, device=h_km.device, dtype=context.dtype)
    scale_height = as_tensor(
        scale_height_km,
        device=h_km.device,
        dtype=context.dtype,
    )
    return rho0 * torch.exp(-h_km / scale_height)


@torch.no_grad()
def atmosphere_mass_density_profile_from_file(
    h_km: TensorLike,
    density_file: str,
    interpolation: str = "linear",
    *,
    context: RuntimeContext,
) -> torch.Tensor:
    """
    Interpolate an atmosphere mass-density profile from a data file.

    Args:
        h_km: Altitudes in km at which the density is requested. Scalar or
            tensor.
        density_file: Path to a text file whose first two numeric columns are
            altitude in km and mass density in g/cm^3.
        interpolation: Interpolation method. Currently only ``"linear"`` is
            supported.
        context: Runtime device/dtype used for the loaded data and output.

    Returns:
        Tensor of interpolated mass densities in g/cm^3 with the same shape
        as h_km. Values outside the file range use the nearest endpoint.

    Raises:
        ValueError: If interpolation is not ``"linear"`` or the file has no
            valid two-column numeric rows.
        FileNotFoundError: If density_file does not exist.
    """
    if interpolation.lower() != "linear":
        raise ValueError("Only linear interpolation is currently supported.")

    h_km = as_tensor(h_km, device=context.device, dtype=context.dtype)
    h_file, rho_file = load_datafile_2column(
        density_file,
        device=h_km.device,
        dtype=context.dtype,
    )

    return interp1d_linear(
        h_km,
        h_file,
        rho_file,
        left=rho_file[0],
        right=rho_file[-1],
        device=h_km.device,
        dtype=context.dtype,
    )


@torch.no_grad()
def atmosphere_density(
    h_km: TensorLike,
    source: str = default.atmosphere_source_density,
    density_type: DensityType = "mass_density",
    Ye: TensorLike = default.atmosphere_Ye,
    rho0_gcm3: TensorLike = default.atmosphere_rho0_gcm3,
    scale_height_km: TensorLike = default.atmosphere_scale_height_km,
    nusquids_config: Optional[NuSQuIDSConfig] = None,
    density_file: Optional[str] = None,
    mceq=None,
    theta_deg: TensorLike = default.mceq_theta_deg,
    mceq_config: Optional[MCEqModelConfig] = None,
    pymsis_config: Optional[PyMSISatmosphereConfig] = None,
    *,
    context: RuntimeContext,
) -> torch.Tensor:
    """
    Evaluate a mass- or electron-density atmosphere profile.

    Args:
        h_km: Altitudes above the Earth's surface in km. Scalar or tensor.
        source: Density backend. Accepted values are ``"exponential"``,
            ``"nusquids"``, ``"earthatm"``, ``"nusquids_earthatm"``,
            ``"file"``, ``"mceq"``, ``"msis"``, and ``"pymsis"``.
        density_type: Output quantity. ``"mass_density"`` returns g/cm^3 and
            ``"electron_density"`` returns mol/cm^3.
        Ye: Electron fraction used to convert non-PyMSIS mass densities into
            electron density.
        rho0_gcm3: Zero-altitude density for the exponential backend in
            g/cm^3.
        scale_height_km: Scale height for the exponential backend in km.
        nusquids_config: Configuration used by the external nuSQuIDS backend,
            including the EarthAtm density normalization, scale height, and
            electron fraction.
        density_file: Two-column altitude-density file required by the file
            backend.
        mceq: Optional initialized MCEq object.
        theta_deg: Zenith angle in degrees used to initialize MCEq.
        mceq_config: Configuration used when the MCEq backend must initialize
            a new MCEq object.
        pymsis_config: Configuration used by the external PyMSIS backend.
        device: Optional torch device used for inputs and output.
        dtype: Real torch dtype used for the returned tensor.

    Returns:
        Tensor with the same shape as h_km containing mass density in g/cm^3
        or electron density in mol/cm^3, according to density_type.

    Raises:
        ValueError: If density_type or source is invalid, or if the file
            backend is selected without density_file.
        ImportError: If a requested optional backend is unavailable.
    """
    source = str(source).lower().strip()
    density_type = str(density_type).lower().strip()

    if density_type not in {"electron_density", "mass_density"}:
        raise ValueError(
            "density_type must be 'electron_density' or 'mass_density'."
        )

    device, dtype = context.device, context.dtype
    h_km = as_tensor(h_km, device=device, dtype=dtype)
    h_context = RuntimeContext(device=h_km.device, dtype=dtype)

    if source == "exponential":
        rho_gcm3 = atmosphere_mass_density_profile_exponential(
            h_km,
            rho0_gcm3=rho0_gcm3,
            scale_height_km=scale_height_km,
            context=h_context,
        )
    elif source in {"nusquids", "earthatm", "nusquids_earthatm"}:
        if nusquids_config is None:
            nusquids_config = NuSQuIDSConfig()
        rho_gcm3 = atmosphere_mass_density_profile_nusquids(
            h_km,
            config=nusquids_config,
            context=h_context,
        )
    elif source == "file":
        if density_file is None:
            raise ValueError("density_file must be provided when source='file'.")
        rho_gcm3 = atmosphere_mass_density_profile_from_file(
            h_km,
            density_file=density_file,
            context=h_context,
        )
    elif source == "mceq":
        if atmosphere_mass_density_profile_from_mceq is None:
            raise ImportError("source='mceq' requires the optional mceq package.")
        if mceq is None and mceq_config is None:
            raise ValueError("mceq_config must be provided when source='mceq' and mceq is None.")
        rho_gcm3 = atmosphere_mass_density_profile_from_mceq(
            h_km=h_km,
            mceq=mceq,
            theta_deg=theta_deg,
            config=mceq_config,
            device=h_km.device,
            dtype=dtype,
        )
    elif source in {"msis", "pymsis"}:
        if PyMSISatmosphereConfig is None or atmosphere_density_pymsis is None:
            raise ImportError("source='pymsis' requires the optional pymsis package.")
        if pymsis_config is None:
            raise ValueError("pymsis_config must be provided when source='pymsis'.")
        profile = atmosphere_density_pymsis(
            alt_km=h_km,
            config=pymsis_config,
        )
        rho_gcm3 = profile["rho_g_cm3"]
    else:
        raise ValueError(
            "Unknown density source. Use 'exponential', 'nusquids', "
            "'earthatm', 'nusquids_earthatm', 'file', 'mceq', 'msis', "
            "or 'pymsis'."
        )

    if density_type == "mass_density":
        return rho_gcm3

    if source in {"msis", "pymsis"}:
        if pymsis_config is None:
            raise ValueError("pymsis_config must be provided when source='pymsis'.")
        electron_fraction = pymsis_config.Ye
    elif source in {"nusquids", "earthatm", "nusquids_earthatm"}:
        if nusquids_config is None:
            nusquids_config = NuSQuIDSConfig()
        electron_fraction = nusquids_config.nusquids_Ye
    else:
        electron_fraction = Ye
    electron_fraction = as_tensor(
        electron_fraction,
        device=rho_gcm3.device,
        dtype=dtype,
    )
    return electron_fraction * rho_gcm3 * GCM3_TO_NUCLEON_MOLCM3


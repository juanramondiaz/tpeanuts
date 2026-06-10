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
Atmospheric density profiles for tpeanuts.

Pure PyTorch implementation.

Units
-----
h_km      : km
rho       : g / cm^3
n_e       : mol / cm^3

Module functions:
    
    atmospheric_mass_density_profile_exponential(...)
        Evaluates rho(h)=rho0 exp(-h/H) in g/cm^3 for torch altitude grids.
    
    atmospheric_mass_density_profile_from_file(...)
        Loads a two-column altitude-density profile and interpolates it on
        requested altitudes.
    
    atmospheric_mass_density_profile_mceq(...)
        Wraps the MCEq atmospheric-density backend and returns torch tensors.
    
    atmospheric_mass_density_profile(...)
        Dispatches mass-density calculations for source="exponential",
        source="file", source="mceq", source="msis", or source="pymsis".
    
    atmospheric_electron_density_profile(...)
        Converts the selected mass-density source to electron density in
        mol/cm^3 using the shared mass-to-molar-density conversion.
"""




from __future__ import annotations

from typing import Optional, Union
import os
import torch

from tpeanuts.external.mceq.density import atmospheric_mass_density_profile_from_mceq
import tpeanuts.util.default as default
from tpeanuts.util.constant import GCM3_TO_NUCLEON_MOLCM3
from tpeanuts.util.type import _as_tensor
from tpeanuts.util.math import interp1d_linear
from tpeanuts.util.torch_util import _default_device

try:
    from tpeanuts.atmosphere.density_pymsis import(
        atmospheric_density_profile_pymsis,
        PyMSISatmosphereConfig
        )
except ImportError:
    atmospheric_density_profile_pymsis = None
    PyMSISatmosphereConfig = None
TensorLike = Union[float, int, torch.Tensor]


@torch.no_grad()
def _load_two_column_file(
    file: str,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not os.path.isfile(file):
        raise FileNotFoundError(f"File not found: {file}")

    col_1 = []
    col_2 = []

    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.replace(",", " ").split()

            if len(parts) < 2:
                continue

            col_1.append(float(parts[0]))
            col_2.append(float(parts[1]))

    if len(col_1) == 0:
        raise ValueError(
            "file does not contain valid numeric rows with "
            "at least two columns."
        )

    dev = _default_device(device)

    col_1_file = torch.tensor(col_1, device=dev, dtype=dtype)
    col_2_file = torch.tensor(col_2, device=dev, dtype=dtype)

    order = torch.argsort(col_1_file)

    return col_1_file[order], col_2_file[order]

# ============================================================
# Exponential atmosphere
# ============================================================

@torch.no_grad()
def atmospheric_mass_density_profile_exponential(
    h_km: TensorLike,
    rho0_gcm3: TensorLike = default.atmosphere_rho0_gcm3,
    scale_height_km: TensorLike = default.atmosphere_scale_height_km,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = default.dtype,
) -> torch.Tensor:
    """
    Evaluate an exponential atmospheric mass-density profile.

    Args:
        h_km: Altitude above surface in km. Scalar or tensor.
        rho0_gcm3: Sea-level mass density in g/cm^3. Scalar or tensor.
        scale_height_km: Exponential scale height in km. Scalar or tensor.
        device: Optional torch device.
        dtype: Real dtype for the output.

    Returns:
        Tensor rho(h)=rho0 exp(-h/H) in g/cm^3 with broadcast shape of inputs.
    """
    h_km = _as_tensor(h_km, device=device, dtype=dtype)
    rho0_gcm3 = _as_tensor(rho0_gcm3, device=h_km.device, dtype=dtype)
    scale_height_km = _as_tensor(scale_height_km, device=h_km.device, dtype=dtype)

    return rho0_gcm3 * torch.exp(-h_km / scale_height_km)


# ============================================================
# File-based atmosphere
# ============================================================

@torch.no_grad()
def atmospheric_mass_density_profile_from_file(
    h_km: TensorLike,
    density_file: str,
    interpolation: str = "linear",
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = default.dtype,
) -> torch.Tensor:
    """
    Interpolate an atmospheric mass-density profile from a two-column file.

    Args:
        h_km: Altitude values in km where density is requested. Scalar or tensor.
        density_file: Path to a file with two columns: altitude km and
            rho_gcm3.
        interpolation: Interpolation mode. Currently only "linear" is
            supported.
        device: Optional torch device.
        dtype: Real dtype for loaded and interpolated tensors.

    Returns:
        Tensor of interpolated mass density in g/cm^3 with shape matching h_km.
    """
    if interpolation.lower() != "linear":
        raise ValueError("Only linear interpolation is currently supported.")

    h_km = _as_tensor(h_km, device=device, dtype=dtype)

    h_file, rho_file = _load_two_column_file(
        density_file,
        device=h_km.device,
        dtype=dtype,
    )

    return interp1d_linear(
        h_km,
        h_file,
        rho_file,
        left=rho_file[0],
        right=rho_file[-1],
        device=h_km.device,
        dtype=dtype,
    )

# ============================================================
# mceq wrapper
# ============================================================

@torch.no_grad()
def atmospheric_mass_density_profile_mceq(
    h_km: TensorLike,
    *,
    mceq=None,
    theta_deg: TensorLike = default.mceq_theta_deg,
    interaction_model: str = default.mceq_interaction_model,
    primary_model=None,
    density_model: str = default.mceq_density_model,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = default.dtype,
) -> torch.Tensor:
    """
    Wrap the MCEq atmospheric-density model and return a torch tensor.

    Args:
        h_km: Altitude grid in km. Scalar or tensor.
        mceq: Optional preconfigured MCEqRun-like object.
        theta_deg: Zenith angle in degrees used by the MCEq density model.
        interaction_model: MCEq interaction-model name.
        primary_model: Optional MCEq primary-model specification.
        density_model: Atmospheric density-model name accepted by MCEq.
        device: Optional output torch device.
        dtype: Real dtype for returned density.

    Returns:
        Tensor of mass density in g/cm^3 with shape following h_km.
    """
    h_t = _as_tensor(h_km, device=device, dtype=dtype)
    theta_t = _as_tensor(theta_deg, device=h_t.device, dtype=dtype)

    # mceq routines are usually CPU/numpy based.
    h_cpu = h_t.detach().cpu()
    theta_cpu = theta_t.detach().cpu()

    rho = atmospheric_mass_density_profile_from_mceq(
        h_km=h_cpu.tolist() if h_cpu.ndim > 0 else float(h_cpu.item()),
        mceq=mceq,
        theta_deg=theta_cpu.tolist() if theta_cpu.ndim > 0 else float(theta_cpu.item()),
        interaction_model=interaction_model,
        primary_model=primary_model,
        density_model=density_model,
    )

    return torch.as_tensor(rho, device=h_t.device, dtype=dtype)


def _build_pymsis_config(
    *,
    pymsis_date=default.pymsis_date,
    pymsis_lon_deg: float = default.pymsis_lon_deg,
    pymsis_lat_deg: float = default.pymsis_lat_deg,
    pymsis_f107: float = default.pymsis_f107,
    pymsis_f107a: float = default.pymsis_f107a,
    pymsis_ap: float = default.pymsis_ap,
    pymsis_version=default.pymsis_version,
    pymsis_Ye: float = default.pymsis_Ye,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = default.dtype,
):
    """
    Build a PyMSISatmosphereConfig from dispatcher keyword arguments.

    Args:
        pymsis_date: UTC date/time accepted by np.datetime64.
        pymsis_lon_deg: Geographic longitude in degrees.
        pymsis_lat_deg: Geographic latitude in degrees.
        pymsis_f107: Daily solar radio flux index.
        pymsis_f107a: 81-day averaged solar radio flux index.
        pymsis_ap: Geomagnetic activity index.
        pymsis_version: MSIS version identifier accepted by pymsis.
        pymsis_Ye: Electron fraction used by pymsis-derived electron density.
        device: Torch device for returned tensors.
        dtype: Real torch dtype for returned tensors.

    Returns:
        PyMSISatmosphereConfig instance populated with the supplied values.
    """
    if PyMSISatmosphereConfig is None:
        raise ImportError(
            "pymsis configuration requires pymsis. Install pymsis or use "
            "source='exponential', 'file', or 'mceq'."
        )

    return PyMSISatmosphereConfig(
        date=pymsis_date,
        lon_deg=pymsis_lon_deg,
        lat_deg=pymsis_lat_deg,
        f107=pymsis_f107,
        f107a=pymsis_f107a,
        ap=pymsis_ap,
        version=pymsis_version,
        Ye=pymsis_Ye,
        device=device,
        dtype=dtype,
    )

# ============================================================
# Main dispatcher
# ============================================================

@torch.no_grad()
def atmospheric_mass_density_profile(
    h_km: TensorLike,
    source: str = default.atmosphere_source_density,
    rho0_gcm3: TensorLike = default.atmosphere_rho0_gcm3,
    scale_height_km: TensorLike = default.atmosphere_scale_height_km,
    density_file: Optional[str] = None,
    mceq=None,
    theta_deg: TensorLike = default.mceq_theta_deg,
    interaction_model: str = default.mceq_interaction_model,
    primary_model=None,
    density_model: str = default.mceq_density_model,
    pymsis_date=default.pymsis_date,
    pymsis_lon_deg: float = default.pymsis_lon_deg,
    pymsis_lat_deg: float = default.pymsis_lat_deg,
    pymsis_f107: float = default.pymsis_f107,
    pymsis_f107a: float = default.pymsis_f107a,
    pymsis_ap: float = default.pymsis_ap,
    pymsis_version=default.pymsis_version,
    pymsis_Ye: float = default.pymsis_Ye,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = default.dtype,
) -> torch.Tensor:
    """
    Dispatch atmospheric mass-density profiles by source name.

    Args:
        h_km: Altitude in km. Scalar or tensor.
        source: Density source. Supported options are "exponential", "file",
            "mceq", and "msis".
        rho0_gcm3: Sea-level density for source="exponential".
        scale_height_km: Scale height in km for source="exponential".
        density_file: Required path for source="file".
        mceq: Optional MCEq object for source="mceq".
        theta_deg: Zenith angle in degrees for MCEq-based density.
        interaction_model: MCEq interaction-model name.
        primary_model: Optional MCEq primary-model specification.
        density_model: MCEq density-model name.
        pymsis_date: UTC date/time for source="msis"/"pymsis"; accepts an
            ISO string or np.datetime64-compatible value.
        pymsis_lon_deg: Geographic longitude in degrees for pymsis.
        pymsis_lat_deg: Geographic latitude in degrees for pymsis.
        pymsis_f107: Daily solar radio flux index for pymsis.
        pymsis_f107a: 81-day averaged solar radio flux index for pymsis.
        pymsis_ap: Geomagnetic activity index for pymsis.
        pymsis_version: MSIS version identifier accepted by pymsis.
        pymsis_Ye: Electron fraction stored in PyMSISatmosphereConfig.
        device: Optional torch device.
        dtype: Real dtype for output.

    Returns:
        Tensor of atmospheric mass density in g/cm^3 with shape matching h_km.
    """
    source = str(source).lower().strip()

    h_km = _as_tensor(h_km, device=device, dtype=dtype)

    if source == "exponential":
        return atmospheric_mass_density_profile_exponential(
            h_km=h_km,
            rho0_gcm3=rho0_gcm3,
            scale_height_km=scale_height_km,
            device=h_km.device,
            dtype=dtype,
        )

    elif source == "file":
        if density_file is None:
            raise ValueError("density_file must be provided when source='file'.")

        return atmospheric_mass_density_profile_from_file(
            h_km=h_km,
            density_file=density_file,
            device=h_km.device,
            dtype=dtype,
        )

    elif source == "mceq":
        return atmospheric_mass_density_profile_mceq(
            h_km=h_km,
            mceq=mceq,
            theta_deg=theta_deg,
            interaction_model=interaction_model,
            primary_model=primary_model,
            density_model=density_model,
            device=h_km.device,
            dtype=dtype,
        )
    
    elif source in ("msis", "pymsis"):

        if atmospheric_density_profile_pymsis is None:
            raise ImportError(
                "source='msis' requires pymsis. Install pymsis or use "
                "source='exponential', 'file', or 'mceq'."
            )
        
        config_msis = _build_pymsis_config(
            pymsis_date=pymsis_date,
            pymsis_lon_deg=pymsis_lon_deg,
            pymsis_lat_deg=pymsis_lat_deg,
            pymsis_f107=pymsis_f107,
            pymsis_f107a=pymsis_f107a,
            pymsis_ap=pymsis_ap,
            pymsis_version=pymsis_version,
            pymsis_Ye=pymsis_Ye,
            device=h_km.device,
            dtype=dtype,
        )
        profile = atmospheric_density_profile_pymsis(
            alt_km=h_km,
            config=config_msis,
        )
        
        return profile['rho_g_cm3']

    else:
        raise ValueError(
            "Unknown density source. Use source='exponential', source='file', "
            "source='mceq', source='msis', or source='pymsis'."
        )


# ============================================================
# 5. Atmospheric Electron density Profile
# ============================================================

@torch.no_grad()
def atmospheric_electron_density_profile(
    h_km: TensorLike,
    Ye: TensorLike = default.atmosphere_Ye,
    source: str = default.atmosphere_source_density,
    rho0_gcm3: TensorLike = default.atmosphere_rho0_gcm3,
    scale_height_km: TensorLike = default.atmosphere_scale_height_km,
    density_file: Optional[str] = None,
    mceq=None,
    theta_deg: TensorLike = default.mceq_theta_deg,
    interaction_model: str = default.mceq_interaction_model,
    primary_model=None,
    density_model: str = default.mceq_density_model,
    pymsis_date=default.pymsis_date,
    pymsis_lon_deg: float = default.pymsis_lon_deg,
    pymsis_lat_deg: float = default.pymsis_lat_deg,
    pymsis_f107: float = default.pymsis_f107,
    pymsis_f107a: float = default.pymsis_f107a,
    pymsis_ap: float = default.pymsis_ap,
    pymsis_version=default.pymsis_version,
    pymsis_Ye: float = default.pymsis_Ye,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = default.dtype,
) -> torch.Tensor:
    """
    Convert atmospheric mass density into electron density.

    Args:
        h_km: Altitude in km. Scalar or tensor.
        Ye: Electron fraction, usually about 0.5. Scalar or broadcastable
            tensor.
        source: Mass-density source; see atmospheric_mass_density_profile.
        rho0_gcm3: Sea-level density for the exponential source.
        scale_height_km: Exponential scale height in km.
        density_file: Two-column density file for source="file".
        mceq: Optional MCEq object for source="mceq".
        theta_deg: Zenith angle in degrees for MCEq density.
        interaction_model: MCEq interaction-model name.
        primary_model: Optional MCEq primary-model specification.
        density_model: MCEq density-model name.
        pymsis_date: UTC date/time for source="msis"/"pymsis".
        pymsis_lon_deg: Geographic longitude in degrees for pymsis.
        pymsis_lat_deg: Geographic latitude in degrees for pymsis.
        pymsis_f107: Daily solar radio flux index for pymsis.
        pymsis_f107a: 81-day averaged solar radio flux index for pymsis.
        pymsis_ap: Geomagnetic activity index for pymsis.
        pymsis_version: MSIS version identifier accepted by pymsis.
        pymsis_Ye: Electron fraction passed to PyMSISatmosphereConfig for
            source="msis"/"pymsis". For other sources, Ye is used directly.
        device: Optional torch device.
        dtype: Real dtype for output.

    Returns:
        Tensor n_e in mol/cm^3, in the convention expected by the matter
        potential routines. For mass-density sources it uses
        n_e = Ye * rho_gcm3 * GCM3_TO_NUCLEON_MOLCM3.
    """
    h_km = _as_tensor(h_km, device=device, dtype=dtype)
    source_normalized = str(source).lower().strip()
    electron_fraction = pymsis_Ye if source_normalized in ("msis", "pymsis") else Ye
    electron_fraction = _as_tensor(electron_fraction, device=h_km.device, dtype=dtype)

    rho_gcm3 = atmospheric_mass_density_profile(
        h_km=h_km,
        source=source_normalized,
        rho0_gcm3=rho0_gcm3,
        scale_height_km=scale_height_km,
        density_file=density_file,
        mceq=mceq,
        theta_deg=theta_deg,
        interaction_model=interaction_model,
        primary_model=primary_model,
        density_model=density_model,
        pymsis_date=pymsis_date,
        pymsis_lon_deg=pymsis_lon_deg,
        pymsis_lat_deg=pymsis_lat_deg,
        pymsis_f107=pymsis_f107,
        pymsis_f107a=pymsis_f107a,
        pymsis_ap=pymsis_ap,
        pymsis_version=pymsis_version,
        pymsis_Ye=pymsis_Ye,
        device=h_km.device,
        dtype=dtype,
    )

    return electron_fraction * rho_gcm3 * GCM3_TO_NUCLEON_MOLCM3

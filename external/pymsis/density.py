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
External PyMSIS atmosphere density backend.

This module adapts the optional third-party ``pymsis`` package to the
tpeanuts atmosphere-density interface, for example:

    source = "pymsis"

``pymsis`` wraps NRLMSIS 2.0, NASA/NRL's empirical atmosphere model, fit to
decades of satellite and ground-based atmospheric measurements. The actual
mass-density calculation (the call to ``pymsis.calculate(...)`` below) is
delegated entirely to the external package; this module only supplies the
input configuration, converts pymsis's output units into the g/cm^3,
mol/cm^3, and g/cm^2 conventions used elsewhere in tpeanuts, and wraps the
result as torch tensors. No oscillation physics is implemented here.

MSIS variables
--------------
date  : UTC date/time
lon   : geographic longitude [deg]
lat   : geographic latitude [deg]
alt   : altitude [km]
f107  : daily solar radio flux
f107a : 81-day averaged solar radio flux
ap    : geomagnetic activity index

Module functions:
    
    PyMSISatmosphereConfig
        Stores the UTC date, geographic coordinates, solar and geomagnetic
        activity indices, MSIS version, electron fraction, dtype, and device
        used by pymsis calculations.
    
    atmosphere_density_pymsis(...)
        Evaluates pymsis on a scalar or 1D altitude grid, returns mass
        density, electron density, vertical depth, and metadata as torch
        tensors.
    
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union, Optional, Dict

import numpy as np
import torch
import pymsis

import tpeanuts.util.default as default
from tpeanuts.util.constant import GCM3_TO_NUCLEON_MOLCM3, N_A
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import as_datetime64, constant_float_row, ensure_1d


TensorLike = Union[float, int, np.ndarray, torch.Tensor]


# ============================================================
# Configuration container
# ============================================================

@dataclass
class PyMSISatmosphereConfig:
    """
    Configuration for pymsis atmosphere density calculations.

    Args:
        date: UTC date/time passed to pymsis. Accepts an ISO-8601 string or
            np.datetime64 scalar.
        lon_deg: Geographic longitude in degrees. Scalar.
        lat_deg: Geographic latitude in degrees. Scalar.
        f107: Daily solar radio flux index used by MSIS. Scalar.
        f107a: 81-day averaged solar radio flux index. Scalar.
        ap: Geomagnetic activity index. Scalar; expanded internally to the
            seven-column pymsis ap array.
        version: pymsis/MSIS version identifier accepted by pymsis.calculate.
        Ye: Electron fraction used to convert mass density to electron density.
        context: Runtime device/dtype for returned tensors.
    """

    date: Union[str, np.datetime64] = default.pymsis_date
    lon_deg: float = default.pymsis_lon_deg
    lat_deg: float = default.pymsis_lat_deg

    f107: float = default.pymsis_f107
    f107a: float = default.pymsis_f107a
    ap: float = default.pymsis_ap

    version: Union[float, str] = default.pymsis_version
    Ye: float = default.pymsis_Ye

    context: RuntimeContext = field(
        default_factory=lambda: RuntimeContext.resolve(None, default.dtype)
    )


# ============================================================
# Main pymsis density routine
# ============================================================

def atmosphere_density_pymsis(
    alt_km: TensorLike,
    config: PyMSISatmosphereConfig,
) -> Dict[str, torch.Tensor]:
    """
    Compute an atmosphere density profile with pymsis.

    Args:
        alt_km: Altitude grid in km. Accepts scalar, 1D NumPy array, 1D torch
            tensor, or list-like input. The routine evaluates pymsis at each
            altitude and requires non-negative values.
        config: PyMSISatmosphereConfig object defining date, geographic
            position, solar and geomagnetic activity, MSIS version, electron
            fraction, dtype, and device.

    Returns:
        Dictionary with torch tensors on config.device/default device, all
        with shape (n_alt,) matching alt_km:
            "alt_km": Altitude grid in km (echoes the input).
            "rho_kg_m3": Total mass density in kg/m^3, as returned by
                pymsis.
            "rho_g_cm3": Same mass density converted to g/cm^3.
            "ne_mol_cm3": Electron density in mol/cm^3, obtained as
                Ye * rho_g_cm3 * GCM3_TO_NUCLEON_MOLCM3, where Ye is the
                electron fraction (electrons per nucleon) from config.Ye.
            "ne_cm3": Electron number density in electrons/cm^3
                (ne_mol_cm3 * Avogadro's number).
            "ne_m3": Electron number density in electrons/m^3.
            "X_g_cm2": Vertical atmospheric depth (slant column mass per
                unit area) above each altitude, in g/cm^2 -- the amount of
                atmosphere a vertically downward-going particle would
                traverse from the top of the grid down to that altitude.
        Plus "source" ("pymsis") and "metadata" (dict echoing the date,
        location, solar/geomagnetic indices, MSIS version, and Ye used).

    Notes:
        The vertical depth is computed as X(h) = integral_h^top rho(h') dh'
        using torch.trapz over the supplied altitude grid (h' in cm), i.e.
        the standard atmospheric depth/column-density convention used for
        cosmic-ray shower development and neutrino production-height
        bookkeeping.
    """
    device = config.context.device
    dtype = config.context.dtype

    alt_np = ensure_1d(alt_km, "alt_km").astype(float)
    date_np = as_datetime64(config.date, "config.date")

    if np.any(alt_np < 0.0):
        raise ValueError("alt_km must contain non-negative altitudes.")

    # pymsis calculation
    output = pymsis.calculate(
        dates=date_np,
        lons=float(config.lon_deg),
        lats=float(config.lat_deg),
        alts=alt_np,
        f107s=float(config.f107),
        f107as=float(config.f107a),
        aps=constant_float_row(config.ap, 7, "config.ap"),
        version=config.version,
    )

    # Output shape for scalar date/lon/lat and vector altitude:
    # (1, 1, 1, n_alt, n_variables)
    mass_density_var = getattr(
        pymsis.Variable,
        "MASS_DENSITY",
        getattr(pymsis.Variable, "MASS_density", None),
    )
    if mass_density_var is None:
        raise AttributeError("pymsis.Variable does not define MASS_DENSITY.")

    rho_kg_m3_np = np.asarray(
        output[0, 0, 0, :, mass_density_var],
        dtype=float,
    )

    # Convert to torch
    alt_t = torch.as_tensor(alt_np, dtype=dtype, device=device)
    rho_kg_m3 = torch.as_tensor(rho_kg_m3_np, dtype=dtype, device=device)

    # Unit conversions
    rho_g_cm3 = rho_kg_m3 * 1.0e-3

    ne_mol_cm3 = config.Ye * rho_g_cm3 * GCM3_TO_NUCLEON_MOLCM3
    ne_cm3 = ne_mol_cm3 * N_A
    ne_m3 = ne_cm3 * 1.0e6

    # Atmospheric vertical depth X(h)
    # h[km] -> h[cm]
    h_cm = alt_t * 1.0e5

    X_g_cm2 = torch.zeros_like(alt_t)

    for i in range(alt_t.numel()):
        X_g_cm2[i] = torch.trapz(
            rho_g_cm3[i:],
            h_cm[i:],
        )

    return {
        "source": "pymsis",
        "alt_km": alt_t,
        "rho_kg_m3": rho_kg_m3,
        "rho_g_cm3": rho_g_cm3,
        "ne_m3": ne_m3,
        "ne_cm3": ne_cm3,
        "ne_mol_cm3": ne_mol_cm3,
        "X_g_cm2": X_g_cm2,
        "metadata": {
            "date": str(date_np),
            "lon_deg": float(config.lon_deg),
            "lat_deg": float(config.lat_deg),
            "f107": float(config.f107),
            "f107a": float(config.f107a),
            "ap": float(config.ap),
            "version": config.version,
            "Ye": float(config.Ye),
        },
    }

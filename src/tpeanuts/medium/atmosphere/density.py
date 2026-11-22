#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
#
# TPeanuts — Fast Simulation of Neutrino Oscillations in Matter.
#
# Developed as part of the Master's Thesis for the MSc in Advanced Physics,
# specialization in Theoretical Physics, at the Universitat de València.
#
# Author
# ------
# Juan Ramón Díaz Santos
# diazjuan@alumni.uv.es
#
# Supervisors
# -----------
# Roberto Ruiz de Austri Bazán
# rruiz@ific.uv.es
#
# Michele Lucente
# michele.lucente@unibo.it
#
# Universitat de València, Spain
# September 2026
# =============================================================================

"""Mass/electron/neutron density along an atmosphere path.

Three backends: ``"exponential"`` (``rho0 * exp(-h/H)``, v1_0's own
reference values) and ``"file"`` (linear interpolation of a
user-supplied two-column CSV, clamped at the endpoints -- v2's own
header/CSV convention, not v1_0's headerless text format, since there is
no existing file to stay byte-compatible with here), both closed-form or
plainly tabulated with no optional dependency; and ``"pymsis"``, wrapping
the optional third-party ``pymsis`` package (NRLMSIS 2.0, an empirical,
data-driven neutral-atmosphere model), a more realistic alternative given
a date/geographic location/space-weather indices. MCEq is a separate,
much heavier optional dependency handled by its own module,
``tpeanuts.source.atmosphere.mceq`` (flux generation, not a density
backend here). ``pymsis`` is imported lazily (only when
``source="pymsis"`` is actually requested), following the exact same
try/except-at-import-time-then-``_ensure_*_available``-at-call-time
pattern ``tpeanuts.source.atmosphere.mceq`` already established for its
own optional dependencies -- this module's own core three-dependency
(``torch``/``numpy``) footprint is unaffected by whether ``pymsis`` is
installed.

``atmosphere_density`` always returns mass density in g/cm^3 or
electron/neutron density in mol/cm^3 (via the electron fraction
``Ye``/``1-Ye`` and ``tpeanuts.util.constant.GCM3_TO_NUCLEON_MOLCM3`` --
this project's own PREM table ships electron/neutron density directly,
precomputed offline, so this conversion did not exist anywhere in v2
before this module needed it). Unlike v1_0's own ``atmosphere_density_pymsis``
(which also computed electron density directly and a vertical
atmospheric-depth column, ``X_g_cm2``), this backend returns only mass
density: the electron/neutron conversion is already generic in
``atmosphere_density`` itself (no need to duplicate it per backend), and
path-length/depth geometry is already ``tpeanuts.medium.atmosphere.
geometry``'s own job, not a density backend's.

Module contents:
    atmosphere_mass_density_exponential(...)
        ``rho0_gcm3 * exp(-h_m / scale_height_m)``.
    atmosphere_mass_density_from_file(...)
        Linear interpolation of a two-column ``altitude_m,mass_density_gcm3``
        CSV.
    atmosphere_mass_density_pymsis(...)
        NRLMSIS 2.0 mass density via the optional ``pymsis`` package, at
        a fixed date/location/space-weather configuration.
    atmosphere_density(...)
        Backend dispatcher, returning mass, electron, or neutron density.
"""

from __future__ import annotations

import csv
from typing import Literal, Union

import numpy as np
import torch

from tpeanuts.config import default
from tpeanuts.util.constant import GCM3_TO_NUCLEON_MOLCM3
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.math import searchsorted_contiguous
from tpeanuts.util.tensor import TensorLike, as_tensor

try:
    import pymsis as _pymsis
except ImportError:
    _pymsis = None

DensityType = Literal["mass_density", "electron_density", "neutron_density"]
_DENSITY_TYPES = ("mass_density", "electron_density", "neutron_density")

PYMSIS_IMPORT_ERROR = (
    "pymsis could not be imported. Install the optional pymsis dependency "
    "(e.g. `pip install pymsis`) in this Python environment before calling "
    "tpeanuts.medium.atmosphere.density.atmosphere_mass_density_pymsis() "
    "(or atmosphere_density(source='pymsis'))."
)


def _ensure_pymsis_available() -> None:
    """Raise an actionable ``ImportError`` if pymsis is not installed."""
    if _pymsis is None:
        raise ImportError(PYMSIS_IMPORT_ERROR)


def _as_float(value: TensorLike) -> float:
    """A Python float from a scalar tensor, array or number."""
    if isinstance(value, torch.Tensor):
        return float(value)
    return float(np.asarray(value))


def atmosphere_mass_density_exponential(
    h_m: TensorLike,
    rho0_gcm3: TensorLike = default.ATMOSPHERE_RHO0_GCM3,
    scale_height_m: TensorLike = default.ATMOSPHERE_SCALE_HEIGHT_M,
    *,
    context: RuntimeContext,
) -> torch.Tensor:
    """Exponential atmosphere mass-density profile, ``rho0 * exp(-h_m / scale_height_m)``.

    Args:
        h_m: Altitude above the surface, in metres; scalar or
            broadcastable tensor.
        rho0_gcm3: Reference mass density at zero altitude, in g/cm^3.
        scale_height_m: Exponential scale height, in metres.
        context: Runtime device/dtype.

    Returns:
        Mass density in g/cm^3, broadcast shape of ``h_m``.
    """
    h_t = as_tensor(h_m, device=context.device, dtype=context.dtype)
    rho0 = as_tensor(rho0_gcm3, device=context.device, dtype=context.dtype)
    scale_height = as_tensor(scale_height_m, device=context.device, dtype=context.dtype)
    return rho0 * torch.exp(-h_t / scale_height)


def atmosphere_mass_density_from_file(
    h_m: TensorLike,
    density_file: str,
    *,
    context: RuntimeContext,
) -> torch.Tensor:
    """Linearly interpolate a tabulated altitude/mass-density CSV.

    Args:
        h_m: Altitude above the surface, in metres, to query.
        density_file: Path to a CSV with columns ``altitude_m`` and
            ``mass_density_gcm3``, sorted ascending by altitude.
        context: Runtime device/dtype.

    Returns:
        Mass density in g/cm^3, broadcast shape of ``h_m``. Queries
        outside the file's own altitude range are clamped to the nearest
        endpoint value, not extrapolated.

    Raises:
        ValueError: If the file has fewer than two rows.
    """
    altitude_m = []
    mass_density_gcm3 = []
    with open(density_file, newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            altitude_m.append(float(row["altitude_m"]))
            mass_density_gcm3.append(float(row["mass_density_gcm3"]))

    if len(altitude_m) < 2:
        raise ValueError(f"{density_file!r} must have at least two rows to interpolate.")

    h_t = as_tensor(h_m, device=context.device, dtype=context.dtype)
    xp = as_tensor(altitude_m, device=context.device, dtype=context.dtype)
    fp = as_tensor(mass_density_gcm3, device=context.device, dtype=context.dtype)

    n_rows = xp.shape[0]
    idx_right = searchsorted_contiguous(xp, h_t, right=False).clamp(1, n_rows - 1)
    idx_left = idx_right - 1

    x0, x1 = xp[idx_left], xp[idx_right]
    y0, y1 = fp[idx_left], fp[idx_right]
    frac = (h_t - x0) / (x1 - x0).clamp_min(torch.finfo(context.dtype).eps)
    interpolated = y0 + frac * (y1 - y0)

    return torch.where(h_t < xp[0], fp[0], torch.where(h_t > xp[-1], fp[-1], interpolated))


def atmosphere_mass_density_pymsis(
    h_m: TensorLike,
    date: str = default.ATMOSPHERE_PYMSIS_DATE,
    lon_deg: TensorLike = default.ATMOSPHERE_PYMSIS_LON_DEG,
    lat_deg: TensorLike = default.ATMOSPHERE_PYMSIS_LAT_DEG,
    f107: TensorLike = default.ATMOSPHERE_PYMSIS_F107,
    f107a: TensorLike = default.ATMOSPHERE_PYMSIS_F107A,
    ap: TensorLike = default.ATMOSPHERE_PYMSIS_AP,
    version: Union[float, str] = default.ATMOSPHERE_PYMSIS_VERSION,
    *,
    context: RuntimeContext,
) -> torch.Tensor:
    """NRLMSIS 2.0 mass-density profile via the optional ``pymsis`` package.

    Ported from v1_0's ``external/pymsis/density.py::atmosphere_density_pymsis``,
    trimmed to this module's own backend contract: a single mass-density
    tensor, no electron-density conversion or atmospheric-depth column
    (both already generic elsewhere in v2 -- see this module's own
    docstring). ``h_m`` may carry any batch shape (not just scalar or
    1-D, unlike v1_0's own ``ensure_1d``-restricted input): it is
    flattened for the one ``pymsis.calculate`` call this function makes,
    then reshaped back to ``h_m``'s own shape.

    Args:
        h_m: Altitude above the surface, in metres; scalar or any batch
            shape. Must be non-negative.
        date: UTC date/time passed to pymsis, as an ISO-8601 string (or
            any value ``numpy.datetime64`` accepts).
        lon_deg: Geographic longitude, in degrees.
        lat_deg: Geographic latitude, in degrees.
        f107: Daily F10.7 solar radio flux index (solar flux units).
        f107a: 81-day averaged F10.7 solar radio flux index (solar flux
            units).
        ap: Geomagnetic Ap index -- expanded internally to pymsis's own
            seven-column ap array (one constant value applied to every
            sub-index, matching v1_0's own convention).
        version: pymsis/NRLMSIS version identifier, forwarded to
            ``pymsis.calculate`` unchanged.
        context: Runtime device/dtype.

    Returns:
        Mass density in g/cm^3, same shape as ``h_m``.

    Raises:
        ImportError: If pymsis is not installed (see
            ``PYMSIS_IMPORT_ERROR``).
        ValueError: If any entry of ``h_m`` is negative.
        AttributeError: If the installed pymsis version does not expose
            ``pymsis.Variable.MASS_DENSITY``.
    """
    _ensure_pymsis_available()

    h_t = as_tensor(h_m, device=context.device, dtype=context.dtype)
    alt_km_np = (h_t.detach().cpu().numpy().astype(float) * 1.0e-3).reshape(-1)
    if np.any(alt_km_np < 0.0):
        raise ValueError("h_m must be non-negative for the pymsis backend.")

    output = _pymsis.calculate(
        dates=np.datetime64(date),
        lons=_as_float(lon_deg),
        lats=_as_float(lat_deg),
        alts=alt_km_np,
        f107s=_as_float(f107),
        f107as=_as_float(f107a),
        aps=[[_as_float(ap)] * 7],
        version=version,
    )

    mass_density_var = getattr(_pymsis.Variable, "MASS_DENSITY", None)
    if mass_density_var is None:
        raise AttributeError("pymsis.Variable does not define MASS_DENSITY.")

    # pymsis squeezes its own leading (date, lon, lat) batch axes inconsistently
    # depending on n_alt (confirmed directly: n_alt=1 collapses to a bare
    # (1, n_variables) array, n_alt>1 keeps the full (1, 1, 1, n_alt,
    # n_variables) shape) -- flattening every axis except the trailing
    # variable one is robust to both, since every one of those leading axes
    # is always exactly 1 element here (scalar date/lon/lat), so the
    # flattened leading axis is always exactly n_alt long either way.
    output_flat = np.asarray(output, dtype=float).reshape(-1, output.shape[-1])
    rho_kg_m3 = output_flat[:, mass_density_var]
    rho_gcm3_np = rho_kg_m3 * 1.0e-3
    return as_tensor(rho_gcm3_np, device=context.device, dtype=context.dtype).reshape(h_t.shape)


def atmosphere_density(
    h_m: TensorLike,
    source: str = default.ATMOSPHERE_SOURCE_DENSITY,
    density_type: DensityType = "electron_density",
    Ye: TensorLike = default.ATMOSPHERE_YE,
    *,
    context: RuntimeContext,
    **backend_kwargs,
) -> torch.Tensor:
    """Evaluate a mass, electron, or neutron density profile at altitude ``h_m``.

    Args:
        h_m: Altitude above the surface, in metres.
        source: Density backend -- ``"exponential"``, ``"file"``, or
            ``"pymsis"`` (requires the optional ``pymsis`` package, see
            this module's own docstring).
        density_type: ``"mass_density"`` (g/cm^3, backend output
            unchanged), or ``"electron_density"``/``"neutron_density"``
            (mol/cm^3, via ``Ye``/``1-Ye`` and
            ``tpeanuts.util.constant.GCM3_TO_NUCLEON_MOLCM3``).
        Ye: Electron fraction used for the mass -> electron/neutron
            conversion.
        context: Runtime device/dtype.
        **backend_kwargs: Forwarded to the selected backend
            (``atmosphere_mass_density_exponential``'s
            ``rho0_gcm3``/``scale_height_m``,
            ``atmosphere_mass_density_from_file``'s ``density_file``, or
            ``atmosphere_mass_density_pymsis``'s ``date``/``lon_deg``/
            ``lat_deg``/``f107``/``f107a``/``ap``/``version``).

    Returns:
        Tensor with the same broadcast shape as ``h_m``.

    Raises:
        ValueError: If ``source``/``density_type`` is not recognized.
        ImportError: If ``source="pymsis"`` and pymsis is not installed.
    """
    source = str(source).lower().strip()
    density_kind = str(density_type).lower().strip()

    if density_kind not in _DENSITY_TYPES:
        raise ValueError(f"density_type must be one of {_DENSITY_TYPES}, got {density_kind!r}.")

    if source == "exponential":
        mass_density_gcm3 = atmosphere_mass_density_exponential(h_m, context=context, **backend_kwargs)
    elif source == "file":
        mass_density_gcm3 = atmosphere_mass_density_from_file(h_m, context=context, **backend_kwargs)
    elif source == "pymsis":
        mass_density_gcm3 = atmosphere_mass_density_pymsis(h_m, context=context, **backend_kwargs)
    else:
        raise ValueError(
            f"Unknown atmosphere density source {source!r} -- expected 'exponential', 'file', or 'pymsis'."
        )

    if density_kind == "mass_density":
        return mass_density_gcm3

    Ye_t = as_tensor(Ye, device=mass_density_gcm3.device, dtype=mass_density_gcm3.dtype)
    fraction = Ye_t if density_kind == "electron_density" else (1.0 - Ye_t)
    return mass_density_gcm3 * GCM3_TO_NUCLEON_MOLCM3 * fraction

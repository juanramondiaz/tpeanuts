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
Earth nadir-exposure table construction utilities.

This module loads or builds nadir-angle exposure tables used to compute
time-averaged earth matter-regeneration probabilities.

The exposure table represents a weight function

    W(eta)

defined on eta in [0, pi]. It is later used by earth/exposure_integration.py
to compute

    P_int(E) = integral d eta W(eta) P_earth(E, eta).

This module provides the main table builder:

    build_nadir_exposure(...)
        Load or construct a GPU-ready ``NadirExposureTable`` from a math,
        cache, CSV, or legacy source.
    prepare_nadir_exposure(...)
        Return an eta grid, normalized exposure vector, and metadata from an
        explicit grid or a configured exposure source.
    integrate_exposure(...)
        Integrate eta-dependent flavour probabilities against an exposure
        table or exposure vector.

The module is intentionally separated from medium/earth/exposure_math.py because it
depends on optional CPU-side libraries such as numpy, pandas, scipy, and the
legacy peanuts implementation.
"""



from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

import torch

from tpeanuts.util.type import TensorLike
from tpeanuts.util.context import RuntimeContext

import tpeanuts.util.default as default
from tpeanuts.util.math import interp1d_linear, normalize_trapz
from tpeanuts.util.torch_util import as_1d_tensor
from tpeanuts.medium.earth.exposure_math import (
    IntegralDay,
    make_eta_grid, _daynight_slice
    )
from tpeanuts.medium.earth.exposure_io import (
    nadir_exposure_from_cache,
    save_nadir_exposure_to_cache,
    nadir_exposure_from_csv,
    _cache_filename
    )

AngleMode = Literal["Nadir", "Zenith", "CosZenith"]
DayNight = Optional[Literal["day", "night"]]
ExposureSource = Literal["math", "cache", "csv", "legacy"]


@dataclass(frozen=True)
class ExposureParameters:
    """
    Nadir-angle exposure table construction settings for a detector site.

    Parameters
    ----------
    exposure_source:
        Exposure backend: "math" (direct integration), "cache" (read a
        previously cached table), "csv" (load a tabulated file), or
        "legacy" (delegate to the legacy peanuts implementation, used only
        for validation).

    exposure_csv_path:
        Path to a CSV exposure table. Required when
        ``exposure_source="csv"``.

    exposure_angle:
        Angle convention used by the CSV file: "Nadir", "Zenith", or
        "CosZenith". Only relevant for ``exposure_source="csv"``.

    exposure_daynight:
        Optional "day"/"night" filter applied to the exposure window. None
        keeps the full day-night cycle.

    exposure_d1:
        First day of year (0-365) of the integration window used to
        time-average the detector exposure.

    exposure_d2:
        Last day of year (0-365) of the integration window used to
        time-average the detector exposure.

    exposure_ns:
        Number of nadir-angle samples used to build the exposure table.

    exposure_cache_dir:
        Directory used to read/write cached exposure tables.

    exposure_use_cache:
        If True, reuse a cached table on disk instead of recomputing it.

    integrate_exposure:
        If True, the pipeline integrates eta-dependent probabilities
        against this exposure table down to a single time-averaged value.
        If False, the eta-resolved probabilities are kept instead.

    detector_latitude_rad:
        Geographic latitude of the detector in radians. Required to build
        the exposure table when ``exposure_source`` is "math", "cache", or
        "legacy" (not needed for "csv").

    inclination:
        Earth's orbital-plane inclination used by the "math"/"legacy"
        exposure integral, in radians.
    """

    exposure_source: ExposureSource = "math"
    exposure_csv_path: Optional[str] = None
    exposure_angle: AngleMode = "Nadir"
    exposure_daynight: DayNight = None
    exposure_d1: float = default.earth_d1
    exposure_d2: float = default.earth_d2
    exposure_ns: int = default.earth_exposure_ns
    exposure_cache_dir: str = default.earth_legacy_cache_dir
    exposure_use_cache: bool = default.earth_use_cache
    integrate_exposure: bool = True
    detector_latitude_rad: Optional[float] = None
    inclination: float = default.earth_inclination


@dataclass
class NadirExposureTable:
    """
    GPU-ready nadir-angle exposure table.

    Parameters
    ----------
    eta:
        Nadir-angle grid in radians. Shape: (Neta,).

    exposure:
        Exposure weights evaluated on eta. Shape: (Neta,).
    """

    eta: torch.Tensor
    exposure: torch.Tensor

    @property
    def device(self) -> torch.device:
        """Device on which ``eta`` (and therefore ``exposure``) is stored."""
        return self.eta.device

    @property
    def dtype(self) -> torch.dtype:
        """Real floating dtype of ``eta`` (and therefore ``exposure``)."""
        return self.eta.dtype

    def normalize_(
        self,
        eps: float = default.earth_normalize_eps,
    ) -> "NadirExposureTable":
        """Normalize the exposure weights to unit trapezoidal integral, in place.

        Rescales ``self.exposure`` so that
        ``integral d eta exposure(eta) = 1`` over the stored ``eta`` grid
        (using the trapezoidal rule), after clamping negative weights to
        zero. This turns ``W(eta)`` into a proper probability density over
        nadir angle so that ``pearth_integrated`` computes a weighted average
        rather than a weighted sum.

        Args:
            eps: Minimum value the trapezoidal-integral normalization
                denominator is clamped to, avoiding division by zero for a
                degenerate (near-empty) exposure table.

        Returns:
            ``self``, with ``exposure`` replaced by its normalized version.
        """
        self.exposure = normalize_trapz(
            self.exposure,
            x=self.eta,
            clamp_min=0.0,
            eps=eps,
        )

        return self

    @torch.no_grad()
    def interp(
        self,
        eta_query: torch.Tensor,
    ) -> torch.Tensor:
        """Linearly interpolate the exposure weight at arbitrary nadir angles.

        Args:
            eta_query: Nadir angles in radians at which to evaluate the
                exposure weight. Need not coincide with the stored grid
                points.

        Returns:
            Exposure weight tensor with the same shape as ``eta_query``,
            linearly interpolated from the stored ``(eta, exposure)`` table.
            Queries outside the stored grid are clamped to the boundary
            values (``exposure[0]`` / ``exposure[-1]``).
        """
        return interp1d_linear(
            x=eta_query,
            xp=self.eta,
            fp=self.exposure,
            left=self.exposure[0],
            right=self.exposure[-1],
            device=self.device,
            dtype=self.dtype,
        )


def integrate_exposure(
    probabilities_eta: torch.Tensor,
    eta: torch.Tensor,
    exposure: torch.Tensor,
) -> torch.Tensor:
    """Integrate eta-dependent probabilities against an exposure weight.

    Args:
        probabilities_eta: Probability tensor whose penultimate dimension is
            the nadir-angle axis and whose final dimension is flavour.
        eta: One-dimensional nadir-angle grid in radians.
        exposure: One-dimensional exposure weights defined on ``eta``.

    Returns:
        Probability tensor with the eta dimension integrated out.

    Raises:
        ValueError: If the eta axis or exposure shapes are inconsistent.
    """
    if eta.ndim != 1 or exposure.ndim != 1:
        raise ValueError("eta and exposure must be one-dimensional.")
    if eta.shape != exposure.shape:
        raise ValueError("eta and exposure must have the same shape.")
    if probabilities_eta.shape[-2] != eta.numel():
        raise ValueError(
            "probabilities_eta penultimate dimension must match eta length."
        )

    weights = exposure.reshape(
        *((1,) * (probabilities_eta.ndim - 2)),
        -1,
        1,
    )
    return torch.trapezoid(probabilities_eta * weights, x=eta, dim=-2)


def prepare_nadir_exposure(
    eta: Optional[TensorLike],
    *,
    exposure: ExposureParameters,
    context: RuntimeContext,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Prepare the nadir-angle grid and normalized exposure weights.

    Args:
        eta: Optional explicit eta grid. When provided, a uniform normalized
            exposure over that grid is used.
        exposure: Exposure-table construction settings.
        context: Runtime device/dtype.

    Returns:
        Tuple ``(eta_grid, exposure_weights, metadata)`` with normalized
        exposure.
    """
    device, dtype = context.device, context.dtype

    if eta is not None:
        eta_grid = as_1d_tensor(eta, name="eta", device=device, dtype=dtype)
        exposure_weights = torch.ones_like(eta_grid)
        exposure_weights = exposure_weights / torch.trapezoid(
            exposure_weights,
            x=eta_grid,
        ).clamp_min(torch.finfo(dtype).tiny)
        return eta_grid, exposure_weights, {
            "source": "user_eta_uniform",
            "normalized": True,
        }

    if exposure.detector_latitude_rad is None and exposure.exposure_source in ("math", "cache", "legacy"):
        raise ValueError(
            "detector_latitude_rad is required when eta is not provided "
            "and exposure_source is math/cache/legacy."
        )

    exposure_table = build_nadir_exposure(exposure=exposure, context=context)

    return exposure_table.eta, exposure_table.exposure, {
        "source": exposure.exposure_source,
        "d1": exposure.exposure_d1,
        "d2": exposure.exposure_d2,
        "ns": exposure.exposure_ns,
        "daynight": exposure.exposure_daynight,
        "normalized": True,
        "detector_latitude_rad": exposure.detector_latitude_rad,
    }


# ============================================================
# Source 1: direct mathematical computation
# ============================================================


@torch.no_grad()
def nadir_exposure_from_math(
    *,
    exposure: ExposureParameters,
    context: RuntimeContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a nadir-exposure table by direct mathematical integration.

    Evaluates ``medium.earth.exposure_math.IntegralDay`` pointwise on a
    uniform nadir-angle grid (``make_eta_grid``) to obtain unnormalized
    exposure weights ``W(eta)`` for a detector at
    ``exposure.detector_latitude_rad``, integrated over the day-of-year
    window ``[exposure.exposure_d1, exposure.exposure_d2]``. This is the
    "math" backend of ``build_nadir_exposure``: the most accurate but also
    the most expensive source, since it loops over the eta grid evaluating
    incomplete elliptic integrals at each point.

    Args:
        exposure: Exposure-table construction settings (latitude, day-of-year
            window, grid size, day/night filter, orbital inclination).
        context: Runtime device/dtype used for the computation.

    Returns:
        Tuple ``(eta, exposure_values)``: the nadir-angle grid in radians and
        the corresponding unnormalized exposure weights.
    """
    device = torch.device(context.device)
    dtype = context.dtype

    eta = make_eta_grid(
        exposure.exposure_ns,
        daynight=exposure.exposure_daynight,
        device=device,
        dtype=dtype,
    )

    lam = torch.as_tensor(
        exposure.detector_latitude_rad,
        device=device,
        dtype=dtype,
    )

    exposure_values = torch.stack(
        [
            IntegralDay(
                eta_i,
                lam,
                d1=exposure.exposure_d1,
                d2=exposure.exposure_d2,
                inclination=exposure.inclination,
                device=device,
                dtype=dtype,
            )
            for eta_i in eta
        ]
    )

    return eta, exposure_values


# ==============================================================
# Source 4: Compute from Legacy peanuts (for Validation purpose)
# ==============================================================


def nadir_exposure_from_legacy_peanuts(
    *,
    exposure: ExposureParameters,
    context: RuntimeContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a nadir-exposure table using the legacy peanuts implementation.

    Delegates to ``peanuts.time_average.NadirExposure`` (NumPy/SciPy-based)
    and converts its output to torch tensors on the requested device/dtype.
    This is the "legacy" backend of ``build_nadir_exposure``, used only to
    validate the torch-native "math" backend (``nadir_exposure_from_math``)
    against the original reference implementation.

    Args:
        exposure: Exposure-table construction settings (latitude, day-of-year
            window, grid size, day/night filter).
        context: Runtime device/dtype for the returned tensors.

    Returns:
        Tuple ``(eta, exposure_values)``: the nadir-angle grid in radians and
        the corresponding unnormalized exposure weights, as returned by the
        legacy implementation.
    """
    from peanuts.time_average import NadirExposure

    table = NadirExposure(
        lam=exposure.detector_latitude_rad,
        d1=exposure.exposure_d1,
        d2=exposure.exposure_d2,
        ns=exposure.exposure_ns,
        normalized=False,
        from_file=None,
        angle="Nadir",
        daynight=exposure.exposure_daynight,
    )

    eta_np = table[:, 0]
    exposure_np = table[:, 1]

    eta = torch.as_tensor(
        eta_np,
        device=context.device,
        dtype=context.dtype,
    )

    exposure_values = torch.as_tensor(
        exposure_np,
        device=context.device,
        dtype=context.dtype,
    )

    return eta, exposure_values


# ============================================================
# Main public builder
# ============================================================

@torch.no_grad()
def build_nadir_exposure(
    *,
    exposure: ExposureParameters,
    context: RuntimeContext,
    normalized: bool = default.earth_normalized_exposure,
) -> NadirExposureTable:
    """Load or construct a GPU-ready nadir-exposure table.

    Dispatches to one of four backends selected by
    ``exposure.exposure_source``:

        "csv": load and angle-convert a tabulated exposure file
            (``nadir_exposure_from_csv``).
        "cache": load a previously computed torch cache file
            (``nadir_exposure_from_cache``).
        "math": evaluate the exposure integral directly
            (``nadir_exposure_from_math``), optionally reading from
            (``exposure.exposure_use_cache``) or writing to a cache file to
            avoid recomputation.
        "legacy": delegate to the legacy peanuts implementation
            (``nadir_exposure_from_legacy_peanuts``), used for validation,
            with the same optional caching as "math".

    The resulting ``(eta, exposure)`` pair is wrapped into a
    ``NadirExposureTable`` and optionally normalized to a unit trapezoidal
    integral (``normalized=True``) so it represents a probability density
    over nadir angle rather than raw unnormalized weights.

    Args:
        exposure: Exposure-table construction settings, including the
            backend selector ``exposure_source`` and the parameters required
            by that backend (CSV path, detector latitude, day-of-year
            window, grid size, day/night filter, caching options).
        context: Runtime device/dtype for the returned table.
        normalized: If True, normalize the exposure weights to unit
            trapezoidal integral before returning.

    Returns:
        ``NadirExposureTable`` with the nadir-angle grid and corresponding
        exposure weights, on the requested device/dtype.

    Raises:
        ValueError: If required parameters are missing for the selected
            backend (e.g. ``exposure_csv_path`` for "csv",
            ``detector_latitude_rad`` for "cache"/"math"/"legacy"), or if
            ``exposure_source`` is not one of "math", "cache", "csv", or
            "legacy".
    """
    device, dtype = context.device, context.dtype
    source = exposure.exposure_source

    if source == "csv":

        if exposure.exposure_csv_path is None:
            raise ValueError("exposure_csv_path must be provided when exposure_source='csv'.")

        eta, exposure_values = nadir_exposure_from_csv(
            exposure.exposure_csv_path,
            angle=exposure.exposure_angle,
            daynight=exposure.exposure_daynight,
            device=device,
            dtype=dtype,
        )

    elif source == "cache":

        if exposure.detector_latitude_rad is None:
            raise ValueError("detector_latitude_rad must be provided when exposure_source='cache'.")

        eta, exposure_values = nadir_exposure_from_cache(
            cache_dir=exposure.exposure_cache_dir,
            lam_rad=exposure.detector_latitude_rad,
            d1=exposure.exposure_d1,
            d2=exposure.exposure_d2,
            ns=exposure.exposure_ns,
            daynight=exposure.exposure_daynight,
            device=device,
            dtype=dtype,
        )

    elif source == "math":

        if exposure.detector_latitude_rad is None:
            raise ValueError("detector_latitude_rad must be provided when exposure_source='math'.")

        path_cache = _cache_filename(
            exposure.exposure_cache_dir,
            exposure.detector_latitude_rad,
            exposure.exposure_d1,
            exposure.exposure_d2,
            exposure.exposure_ns,
            exposure.exposure_daynight,
        )

        if exposure.exposure_use_cache and os.path.exists(path_cache):
            eta, exposure_values = nadir_exposure_from_cache(
                cache_dir=exposure.exposure_cache_dir,
                lam_rad=exposure.detector_latitude_rad,
                d1=exposure.exposure_d1,
                d2=exposure.exposure_d2,
                ns=exposure.exposure_ns,
                daynight=exposure.exposure_daynight,
                device=device,
                dtype=dtype,
            )

        else:

            eta, exposure_values = nadir_exposure_from_math(
                exposure=exposure,
                context=context,
            )

    elif source == "legacy":

        if exposure.detector_latitude_rad is None:
            raise ValueError("detector_latitude_rad must be provided when exposure_source='legacy'.")

        eta, exposure_values = nadir_exposure_from_legacy_peanuts(
            exposure=exposure,
            context=context,
        )
    else:
        raise ValueError("exposure_source must be 'math', 'cache', 'csv' or 'legacy'.")

    table = NadirExposureTable(
        eta=eta,
        exposure=exposure_values,
    )

    if normalized:
        table.normalize_()

    if exposure.exposure_use_cache and source in {"math", "legacy"}:
        if exposure.detector_latitude_rad is None:
            raise ValueError(f"detector_latitude_rad must be provided to cache source={source!r}.")

        save_nadir_exposure_to_cache(
            eta=table.eta,
            exposure=table.exposure,
            lam_rad=exposure.detector_latitude_rad,
            d1=exposure.exposure_d1,
            d2=exposure.exposure_d2,
            ns=exposure.exposure_ns,
            daynight=exposure.exposure_daynight,
            cache_dir=exposure.exposure_cache_dir,
            )

    return table

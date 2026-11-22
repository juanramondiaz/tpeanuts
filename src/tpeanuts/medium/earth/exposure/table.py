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

"""The nadir-exposure table ``W(eta_rad)``, and the physics-agnostic reduction against it.

``build_nadir_exposure`` tabulates
``tpeanuts.medium.earth.exposure.time_average.integral_day`` over a full
``make_eta_grid`` (looping in Python: ``integral_day`` is evaluated one
scalar ``eta_rad`` at a time, see its own docstring for why), for a given
detector latitude -- the "math" exposure source, computed on the fly with
no data file (see this subpackage's own ``__init__`` docstring for why
CSV/cache sources are left out of this round).

``integrate_exposure`` is the other half: a plain trapezoidal reduction
of an already-computed probability tensor against ``W(eta_rad)``, with no
knowledge of how that tensor was produced -- this is what keeps this
subpackage decoupled from ``tpeanuts.physics.model.PropagationModel``
(see this subpackage's own ``__init__`` docstring).

Module contents:
    NadirExposureTable
        Frozen-shape dataclass: a nadir-angle grid and its exposure
        weight at each point, with a ``normalize_()`` in-place helper.
    build_nadir_exposure(...)
        Build a ``NadirExposureTable`` for a detector latitude, the
        "math" source.
    integrate_exposure(...)
        ``integral(d eta_rad * W(eta_rad) * probabilities_eta(eta_rad))``, reduced
        via the trapezoidal rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tpeanuts.config import default
from tpeanuts.medium.earth.exposure.time_average import DayNight, integral_day, make_eta_grid
from tpeanuts.util.constant import EARTH_INCLINATION_RAD
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import TensorLike, as_tensor


@dataclass
class NadirExposureTable:
    """A tabulated nadir-angle exposure weight, ``W(eta_rad)`` on a fixed grid.

    Attributes:
        eta_rad: Nadir-angle grid, in radians, shape ``(n_eta,)``.
        exposure: Exposure weight at each ``eta_rad``, same shape.

    Examples:
        >>> import math
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> from tpeanuts.medium import build_nadir_exposure
        >>> latitude = torch.tensor(math.radians(45.0), dtype=torch.float64)
        >>> table = build_nadir_exposure(latitude, ns=21, normalized=True, context=context)
        >>> table.eta_rad.shape, table.exposure.shape
        (torch.Size([21]), torch.Size([21]))
        >>> round(float(torch.trapezoid(table.exposure, table.eta_rad)), 6)   # normalised to unit total weight
        1.0
    """

    eta_rad: torch.Tensor
    exposure: torch.Tensor

    def normalize_(self) -> None:
        """Rescale ``exposure`` in place so its trapezoidal integral over ``eta_rad`` is one."""
        self.exposure = self.exposure / torch.trapezoid(self.exposure, self.eta_rad)


def build_nadir_exposure(
    detector_latitude_rad: TensorLike,
    *,
    d1: float = default.EARTH_EXPOSURE_D1,
    d2: float = default.EARTH_EXPOSURE_D2,
    ns: int = default.EARTH_EXPOSURE_NS,
    daynight: DayNight = default.EARTH_EXPOSURE_DAYNIGHT,
    inclination: float = EARTH_INCLINATION_RAD,
    normalized: bool = default.EARTH_EXPOSURE_NORMALIZED,
    context: RuntimeContext,
) -> NadirExposureTable:
    """Build the "math" nadir-exposure table for a detector at a given latitude.

    Args:
        detector_latitude_rad: Detector geographic latitude, in radians.
            No default -- specific to each detector, and a "reasonable"
            default would silently pick a latitude (e.g. the equator)
            that is misleading rather than helpful.
        d1: First day of year of the integration window, in ``[0, 365]``.
        d2: Last day of year of the integration window, in ``[0, 365]``.
        ns: Number of nadir-angle grid points spanning ``[0, pi]`` (before
            any ``daynight`` slicing).
        daynight: ``"day"``/``"night"`` to restrict the grid to one half,
            or ``None`` for the full ``[0, pi]`` range.
        inclination: Earth's orbital-plane (rotation-axis) inclination in
            radians.
        normalized: When True, rescale the returned table's own exposure
            so it integrates to one (see ``NadirExposureTable.normalize_``).
        context: Runtime device/dtype.

    Returns:
        NadirExposureTable holding the grid and its exposure weight at
        each point.

    Examples:
        >>> import math
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> from tpeanuts.medium import build_nadir_exposure
        >>> latitude = torch.tensor(math.radians(46.47), dtype=torch.float64)   # SNO
        >>> night = build_nadir_exposure(latitude, ns=21, daynight="night", normalized=True, context=context)
        >>> bool(torch.all(night.eta_rad <= math.pi / 2 + 1e-12))   # night: the Sun is below the horizon
        True
    """
    eta_rad = make_eta_grid(ns, daynight=daynight, context=context)
    lam = as_tensor(detector_latitude_rad, device=context.device, dtype=context.dtype)

    exposure = torch.stack(
        [
            integral_day(eta_i, lam, d1, d2, inclination=inclination, context=context)
            for eta_i in eta_rad
        ]
    )

    table = NadirExposureTable(eta_rad=eta_rad, exposure=exposure)
    if normalized:
        table.normalize_()
    return table


def integrate_exposure(
    probabilities_eta: torch.Tensor,
    eta_rad: torch.Tensor,
    exposure: torch.Tensor,
) -> torch.Tensor:
    """Exposure-weighted average of an already-computed probability tensor.

    ``integral(d eta_rad * exposure(eta_rad) * probabilities_eta(eta_rad)) / integral(d
    eta_rad * exposure(eta_rad))`` -- normalizes by ``exposure``'s own integral
    regardless of whether it was already normalized, so the result is
    always a genuine weighted average (not a raw weighted sum scaled by
    whatever ``exposure``'s own units happen to be).

    Knows nothing about how ``probabilities_eta`` was computed -- any of
    the three ``PropagationMethod`` values, any detector depth, any
    number of flavours -- only that its ``eta_rad``-indexed axis (axis 0)
    lines up with ``eta_rad``/``exposure``.

    Args:
        probabilities_eta: Tensor shaped ``(n_eta, ...)``, probabilities
            (or any other per-eta_rad quantity) evaluated at each point of
            ``eta_rad``.
        eta_rad: Nadir-angle grid, in radians, shape ``(n_eta,)`` -- typically
            a ``NadirExposureTable.eta_rad``.
        exposure: Exposure weight at each ``eta_rad``, shape ``(n_eta,)`` --
            typically a ``NadirExposureTable.exposure``.

    Returns:
        Tensor shaped like ``probabilities_eta`` with axis 0 removed.
    """
    weight = exposure / torch.trapezoid(exposure, eta_rad)
    shape = (-1,) + (1,) * (probabilities_eta.ndim - 1)
    return torch.trapezoid(weight.reshape(shape) * probabilities_eta, eta_rad, dim=0)

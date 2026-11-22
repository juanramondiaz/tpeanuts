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

"""The Earth, atmosphere and Sun that nuSQuIDS integrates, read from its tables and its bodies.

A comparison with nuSQuIDS on the same profile needs the profile the solver actually uses, not the table it reads: the
solver interpolates the 201 nodes of its Earth table with Akima splines applied to the density and to the electron
fraction separately (found in ``notebooks/3_validation/2_external/1_nusquids/02_earth``, where the linear interpolation
leaves a residual of ``3e-4`` and this one ``2e-6``), reads the Earth and the atmosphere of ``EarthAtm`` along a track,
and gives the Sun as the density and electron fraction of the BS05 table along ``SunASnu``. This module returns those
profiles as plain NumPy arrays and callables, with ``N_e = rho Y_e`` and ``N_n = rho (1 - Y_e)`` in mol/cm^3. It imports
nothing from ``tpeanuts``.

Module contents:
    EarthTable
        The Earth table of nuSQuIDS with its Akima and linear profiles.
    akima_density(...)
        Akima interpolation of a table of rows ``(radius, density, fraction)``.
    earthatm_radial_profile(...)
        Density and electron fraction of the Earth along the radius, read from ``EarthAtm``.
    sun_profile(...)
        Electron and neutron density of the Sun along the radius, read from ``SunASnu``.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy.interpolate import Akima1DInterpolator

from validation.nusquids.adapter import data_file_args, make_solar_track, require_nusquids

_EARTH_TABLE = "EARTH_MODEL_PREM.dat"
_SUN_TABLE = "bs05_agsop.dat"


class EarthTable:
    """The Earth table of nuSQuIDS and the profile that the solver builds from it.

    Attributes:
        table: Array shaped ``(201, 3)``: radius in units of the Earth radius, density in g/cm^3, electron fraction.
        nodes: The radii of the table.
    """

    def __init__(self, table: np.ndarray | None = None) -> None:
        self.table = np.loadtxt(data_file_args(_EARTH_TABLE)[0]) if table is None else np.asarray(table, dtype=float)
        self.nodes = self.table[:, 0]
        self._density = Akima1DInterpolator(self.nodes, self.table[:, 1])
        self._ye = Akima1DInterpolator(self.nodes, self.table[:, 2])

    def density(self, fraction: np.ndarray) -> np.ndarray:
        """Mass density in g/cm^3 at a radius in units of the Earth radius, Akima interpolation."""
        return self._density(fraction)

    def electron_fraction(self, fraction: np.ndarray) -> np.ndarray:
        """Electron fraction ``Y_e`` at a radius, Akima interpolation."""
        return self._ye(fraction)

    def electron_density(self, fraction: np.ndarray) -> np.ndarray:
        """``N_e = rho Y_e`` in mol/cm^3, the product of the two Akima interpolations."""
        return self._density(fraction) * self._ye(fraction)

    def neutron_density(self, fraction: np.ndarray) -> np.ndarray:
        """``N_n = rho (1 - Y_e)`` in mol/cm^3, with the two Akima interpolations."""
        return self._density(fraction) * (1.0 - self._ye(fraction))

    def linear_electron_density(self, fraction: np.ndarray) -> np.ndarray:
        """``N_e`` with the linear interpolation of the product ``rho Y_e``, which is NOT what nuSQuIDS integrates."""
        return np.interp(fraction, self.nodes, self.table[:, 1] * self.table[:, 2])


def akima_density(rows: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Akima interpolation of the second column of a table of rows ``(radius, density, fraction)``.

    Args:
        rows: Array shaped ``(n, 3)``; with a fraction of one, as for a table that carries ``N_e`` directly, the
            interpolation is that of nuSQuIDS.

    Returns:
        A function of the radius.
    """
    spline = Akima1DInterpolator(rows[:, 0], rows[:, 1])
    return lambda radius: spline(radius)


def earthatm_radial_profile(nsq: object | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Density and electron fraction of the Earth along the radius, as ``EarthAtm`` returns them along a vertical track.

    The radii are dense (6001 up to 0.99, 3001 up to 1, and every node of the table) so that the discontinuities are
    resolved.

    Args:
        nsq: The nuSQuIDS module; imported if None.

    Returns:
        ``(fractions, density_gcm3, electron_fraction)``; form ``N_e`` and ``N_n`` from them.
    """
    nsq = require_nusquids() if nsq is None else nsq
    earth_atm = nsq.EarthAtm(*data_file_args(_EARTH_TABLE))
    nodes = np.loadtxt(data_file_args(_EARTH_TABLE)[0])[:, 0]
    track = earth_atm.MakeTrackWithCosine(-1.0)
    km = nsq.Const().km
    centre = (earth_atm.GetAtmosphereHeight() + earth_atm.GetRadius()) * km
    fractions = np.unique(
        np.concatenate([np.linspace(0.0, 0.99, 6001), np.linspace(0.99, 1.0, 3001)[:-1], nodes[1:-1], [1.0 - 1.0e-9]])
    )
    density, ye = [], []
    for fraction in fractions:
        track.SetX(float(centre + fraction * earth_atm.GetRadius() * km))
        density.append(earth_atm.density(track))
        ye.append(earth_atm.ye(track))
    return fractions, np.array(density), np.array(ye)


def sun_profile(nsq: object | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Electron and neutron density of the Sun along the radius, from the BS05 table of ``SunASnu``.

    Args:
        nsq: The nuSQuIDS module; imported if None.

    Returns:
        ``(radius, electron_density, neutron_density)`` in units of the solar radius and mol/cm^3, from the centre
        to the surface (the centre takes the value of the first row, the surface the last).
    """
    nsq = require_nusquids() if nsq is None else nsq
    sun = nsq.SunASnu(*data_file_args(_SUN_TABLE))
    radii = np.loadtxt(data_file_args(_SUN_TABLE)[0])[:, 1]
    density = np.array([sun.density(make_solar_track(float(r))) for r in radii])
    ye = np.array([sun.ye(make_solar_track(float(r))) for r in radii])
    radius = np.concatenate([[0.0], radii[:-1], [1.0]])
    electron = density * ye
    neutron = density * (1.0 - ye)
    return radius, np.concatenate([[electron[0]], electron]), np.concatenate([[neutron[0]], neutron])

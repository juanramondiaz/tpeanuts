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

"""The scenario catalogue and the size grids that every backend is timed on.

A fair comparison needs the same physical case and the same grids in every package, so they are defined once, here, as
plain data and ``numpy`` grids (no physics and no reference-package import). The code that builds each backend's own
objects for a scenario belongs to ``benchmarks/<backend>/`` and reads its sizes from this catalogue.

The energy grids are those of the benchmarks of the original thesis: solar energies linear between 0.5 and 15 MeV and
atmospheric energies logarithmic between 100 MeV and 20 GeV. The nadir angles are linear between 0.05 and pi/2 - 0.05
rad: the hemisphere in which the neutrinos of a surface detector cross the Earth, the only one every backend can
represent (nuSQuIDS covers cos(zenith) < 0 there).

Module contents:
    OSCILLATION
        The oscillation parameters shared by every backend.
    Scenario
        A named physical case with the size axes it is swept over.
    SCENARIOS
        The catalogue, by name.
    METHOD_SCENARIOS
        The scenarios whose cost depends on the propagation method; the others have none.
    energy_grid_MeV(...), nadir_grid_rad(...)
        The standard grids.
    expand_sizes(...)
        The size combinations of a scenario, full or reduced to a smoke test.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import numpy as np

SOLAR_ENERGY_RANGE_MEV = (0.5, 15.0)
ATMOSPHERIC_ENERGY_RANGE_MEV = (100.0, 2.0e4)
NADIR_MARGIN_RAD = 0.05
NADIR_MAX_RAD = math.pi / 2 - NADIR_MARGIN_RAD

ENERGY_KINDS = ("solar", "atmospheric")

# The oscillation parameters every backend uses (NuFIT-like, normal ordering): angles and phase in degrees, mass
# splittings in eV^2. Defined once so that no package is timed on a slightly different physical case.
OSCILLATION = {
    "theta12_deg": 33.41,
    "theta13_deg": 8.58,
    "theta23_deg": 42.0,
    "delta13_deg": 232.0,
    "DeltamSq21_eV2": 7.41e-5,
    "DeltamSq3l_eV2": 2.507e-3,
}


@dataclass(frozen=True)
class Scenario:
    """A named physical case and the sizes it is swept over.

    Attributes:
        name: Identifier used in the result files.
        description: One line saying what is propagated.
        energy_kind: ``"solar"`` or ``"atmospheric"``, which fixes the energy grid.
        sizes: Axis name mapped to the sizes it takes, e.g. ``{"n_energy": (10, 100, 1000)}``.
        smoke: The reduced sizes used by quick runs and tests (same axes as ``sizes``).
    """

    name: str
    description: str
    energy_kind: str
    sizes: Dict[str, Tuple[int, ...]]
    smoke: Dict[str, Tuple[int, ...]]


_ENERGY_SIZES = (1, 10, 100, 1000, 10000)  # one point is the unit cost of the slowest backends
_NADIR_SIZES = (1, 10, 100, 1000)

# The scenarios whose cost depends on the propagation method; the others (vacuum, solar) have none, so a reference
# record of them is paired with the tpeanuts record that has no method.
METHOD_SCENARIOS = ("earth", "solar_earth", "atmosphere")

SCENARIOS: Dict[str, Scenario] = {
    scenario.name: scenario
    for scenario in (
        Scenario(
            name="vacuum",
            description="Three-flavour oscillation in vacuum over a fixed baseline, swept in energy.",
            energy_kind="solar",
            sizes={"n_energy": _ENERGY_SIZES},
            smoke={"n_energy": (1, 10, 100)},
        ),
        Scenario(
            name="solar",
            description="Adiabatic solar-interior propagation of the 8B flux, swept in energy.",
            energy_kind="solar",
            sizes={"n_energy": _ENERGY_SIZES},
            smoke={"n_energy": (1, 10, 100)},
        ),
        Scenario(
            name="earth",
            description="Propagation through the Earth along a chord, swept in energy and nadir angle.",
            energy_kind="solar",
            sizes={"n_energy": _ENERGY_SIZES, "n_nadir": _NADIR_SIZES},
            smoke={"n_energy": (1, 10, 100), "n_nadir": (1, 10)},
        ),
        Scenario(
            name="solar_earth",
            description="Solar propagation followed by the Earth leg, swept in energy and nadir angle.",
            energy_kind="solar",
            sizes={"n_energy": _ENERGY_SIZES, "n_nadir": _NADIR_SIZES},
            smoke={"n_energy": (1, 10, 100), "n_nadir": (1, 10)},
        ),
        Scenario(
            name="atmosphere",
            description="Atmospheric neutrinos from production to detector, swept in energy and nadir angle.",
            energy_kind="atmospheric",
            sizes={"n_energy": _ENERGY_SIZES, "n_nadir": _NADIR_SIZES},
            smoke={"n_energy": (1, 10, 100), "n_nadir": (1, 10)},
        ),
    )
}


def energy_grid_MeV(kind: str, n: int) -> np.ndarray:
    """The standard energy grid of a scenario kind.

    Args:
        kind: ``"solar"`` (linear, 0.5 to 15 MeV) or ``"atmospheric"`` (logarithmic, 100 MeV to 20 GeV).
        n: Number of points.

    Returns:
        Energies in MeV, shape ``(n,)``, float64.

    Raises:
        ValueError: If ``kind`` is unknown or ``n`` is not positive.
    """
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}.")
    if kind == "solar":
        return np.linspace(*SOLAR_ENERGY_RANGE_MEV, int(n))
    if kind == "atmospheric":
        low, high = ATMOSPHERIC_ENERGY_RANGE_MEV
        return np.logspace(math.log10(low), math.log10(high), int(n))
    raise ValueError(f"kind must be one of {ENERGY_KINDS}, got {kind!r}.")


def nadir_grid_rad(n: int) -> np.ndarray:
    """The standard nadir-angle grid.

    Args:
        n: Number of points.

    Returns:
        Angles in radians, shape ``(n,)``, float64, linear between 0.05 and pi/2 - 0.05 (the Earth-crossing hemisphere
        of a surface detector).

    Raises:
        ValueError: If ``n`` is not positive.
    """
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}.")
    return np.linspace(NADIR_MARGIN_RAD, NADIR_MAX_RAD, int(n))


def expand_sizes(scenario: Scenario, *, smoke: bool = False) -> Iterator[Dict[str, int]]:
    """The combinations of the size axes of a scenario.

    Args:
        scenario: The scenario.
        smoke: Use the reduced sizes of quick runs instead of the full ones.

    Yields:
        One ``{axis: size}`` dictionary per combination, the last axis varying fastest.
    """
    axes = scenario.smoke if smoke else scenario.sizes
    names = list(axes)
    for combination in itertools.product(*(axes[name] for name in names)):
        yield dict(zip(names, combination, strict=True))

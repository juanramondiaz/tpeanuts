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

"""Adapter to the original PEANUTS: its scenarios evaluated over the benchmark grids, one scalar call per point.

The original package (Lucente and Gonzalo's NumPy/Numba code, a separate git repository) is imported from its own
directory, never from this repository: ``v1_0/peanuts`` here is a modified fork (it imports the first tpeanuts), and
timing it would not be timing the original. The directory is found from an explicit argument, the ``PEANUTS_PATH``
environment variable, or a ``PEANUTS`` directory next to this repository, and its git commit is recorded with the
results. This module imports neither ``torch`` nor ``tpeanuts``.

PEANUTS evaluates one energy and one nadir angle per call, so a grid is a Python loop over scalar calls; that is how its
users run it, and what is timed. Its two Earth methods are paired with the tpeanuts ones that use the same algorithm:
``analytical`` (the perturbative evolutor, accurate to a few parts in 10^4 and the default of the package) with
``analytic_perturbative``, and ``numerical`` (an ODE integration) with ``numerical``. PEANUTS has no atmospheric
propagation, so that scenario is unavailable here.

Module contents:
    PAIRED_METHODS
        PEANUTS Earth mode mapped to the tpeanuts method it is paired with.
    UNAVAILABLE
        The scenarios PEANUTS cannot run, with the reason.
    locate_peanuts(...)
        The directory of the original package.
    git_state(...)
        Commit of the original and whether its package directory has local changes.
    Peanuts
        The loaded package with the shared oscillation parameters, and the scenarios over a grid.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from benchmarks.common.scenarios import OSCILLATION

PAIRED_METHODS = {"analytical": "analytic_perturbative", "numerical": "numerical"}
UNAVAILABLE = {"atmosphere": "the original PEANUTS has no atmospheric propagation (it starts at the Earth's surface)"}

_REPOSITORY = Path(__file__).resolve().parents[2]
_ENVIRONMENT_VARIABLE = "PEANUTS_PATH"
SOLAR_CHANNEL = "8B"
VACUUM_BASELINE_KM = 180.0  # benchmarks.tpeanuts.cases.VACUUM_BASELINE_M, in km, which is what PEANUTS takes


def locate_peanuts(path: Optional[Union[str, Path]] = None) -> Path:
    """The directory of the original PEANUTS (the one that contains the ``peanuts`` package and ``Data``).

    Args:
        path: Explicit location; otherwise the ``PEANUTS_PATH`` environment variable, then a ``PEANUTS`` directory
            next to this repository.

    Returns:
        The directory.

    Raises:
        FileNotFoundError: If no candidate contains ``peanuts/__init__.py`` and ``Data``.
    """
    candidates = [path, os.environ.get(_ENVIRONMENT_VARIABLE), _REPOSITORY.parent / "PEANUTS"]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate)
        if (root / "peanuts" / "__init__.py").is_file() and (root / "Data").is_dir():
            return root.resolve()
    raise FileNotFoundError(
        f"the original PEANUTS was not found; pass its directory or set {_ENVIRONMENT_VARIABLE} "
        f"(tried {[str(c) for c in candidates if c]})."
    )


def git_state(directory: Path) -> Dict[str, Any]:
    """The commit of the original PEANUTS and whether its ``peanuts`` package directory has local changes.

    Args:
        directory: The directory returned by ``locate_peanuts``.

    Returns:
        ``{"path", "commit", "package_dirty"}``; the last two are ``None`` when git is unavailable.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=directory, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", "peanuts"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"path": str(directory), "commit": None, "package_dirty": None}
    return {"path": str(directory), "commit": commit or None, "package_dirty": bool(status.strip())}


class Peanuts:
    """The original PEANUTS loaded with the shared oscillation parameters, evaluated over (nadir, energy) grids.

    Args:
        path: Location of the original (see ``locate_peanuts``).

    Attributes:
        directory: Where the original was loaded from.
        pmns: Its PMNS object, built from ``scenarios.OSCILLATION``.
        earth_density: Its Earth density object (the package default, ``Data/Earth_Density.csv``).
        solar_model: Its solar model (the package default, B16 with AGSS09 composition).
    """

    def __init__(self, path: Optional[Union[str, Path]] = None) -> None:
        self.directory = locate_peanuts(path)
        if str(self.directory) not in sys.path:
            sys.path.append(str(self.directory))  # appended, so that nothing of the original shadows another package
        with warnings.catch_warnings():  # the original reads its CSV files with a pandas call that now warns
            warnings.simplefilter("ignore", FutureWarning)
            from peanuts.earth import EarthDensity, Pearth
            from peanuts.pmns import PMNS
            from peanuts.solar import Psolar, SolarModel, solar_flux_mass
            from peanuts.vacuum import Pvacuum
        warnings.filterwarnings("ignore", category=FutureWarning, module="peanuts")
        warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

        self._pearth, self._pvacuum, self._psolar, self._solar_flux_mass = Pearth, Pvacuum, Psolar, solar_flux_mass
        self.theta12 = math.radians(OSCILLATION["theta12_deg"])
        self.theta13 = math.radians(OSCILLATION["theta13_deg"])
        self.dm21 = OSCILLATION["DeltamSq21_eV2"]
        self.dm3l = OSCILLATION["DeltamSq3l_eV2"]
        self.pmns = PMNS(
            self.theta12,
            self.theta13,
            math.radians(OSCILLATION["theta23_deg"]),
            math.radians(OSCILLATION["delta13_deg"]),
        )
        self.earth_density = EarthDensity()
        self.solar_model = SolarModel()
        self._radius = self.solar_model.radius()
        self._density = self.solar_model.density()
        self._fraction = self.solar_model.fraction(SOLAR_CHANNEL)

    def vacuum(self, energy_MeV: np.ndarray) -> np.ndarray:
        """Flavour probabilities of a ``nue`` after the vacuum baseline, shape ``(n_energy, 3)``."""
        nue = np.array([1.0, 0.0, 0.0])
        return np.stack(
            [
                self._pvacuum(nue, self.pmns, self.dm21, self.dm3l, float(e), VACUUM_BASELINE_KM, massbasis=False)
                for e in energy_MeV
            ]
        )

    def solar(self, energy_MeV: np.ndarray) -> np.ndarray:
        """Flavour probabilities at the solar surface of the 8B channel, shape ``(n_energy, 3)``."""
        return np.stack(
            [
                self._psolar(self.pmns, self.dm21, self.dm3l, float(e), self._radius, self._density, self._fraction)
                for e in energy_MeV
            ]
        )

    def earth(self, energy_MeV: np.ndarray, nadir_rad: np.ndarray, mode: str) -> np.ndarray:
        """Flavour probabilities of a ``nue`` through the Earth, shape ``(n_nadir, n_energy, 3)``.

        Args:
            energy_MeV: Energies in MeV.
            nadir_rad: Nadir angles in radians.
            mode: ``"analytical"`` or ``"numerical"`` (see ``PAIRED_METHODS``).
        """
        nue = np.array([1.0, 0.0, 0.0])
        return np.stack(
            [
                np.stack(
                    [
                        self._pearth(
                            nue,
                            self.earth_density,
                            self.pmns,
                            self.dm21,
                            self.dm3l,
                            float(e),
                            float(a),
                            0.0,
                            mode=mode,
                            massbasis=False,
                        )
                        for e in energy_MeV
                    ]
                )
                for a in nadir_rad
            ]
        )

    def solar_earth(self, energy_MeV: np.ndarray, nadir_rad: np.ndarray, mode: str) -> np.ndarray:
        """Flavour probabilities of the solar 8B flux through the Earth, shape ``(n_nadir, n_energy, 3)``.

        The solar mass-basis weights of each energy are computed once and propagated through every nadir angle, which
        is how the original is used for this chain (the Sun does not depend on the angle).
        """
        weights = [
            np.asarray(
                self._solar_flux_mass(
                    self.theta12,
                    self.theta13,
                    self.dm21,
                    self.dm3l,
                    float(e),
                    self._radius,
                    self._density,
                    self._fraction,
                )
            )
            for e in energy_MeV
        ]
        return np.stack(
            [
                np.stack(
                    [
                        self._pearth(
                            w,
                            self.earth_density,
                            self.pmns,
                            self.dm21,
                            self.dm3l,
                            float(e),
                            float(a),
                            0.0,
                            mode=mode,
                            massbasis=True,
                        )
                        for e, w in zip(energy_MeV, weights, strict=True)
                    ]
                )
                for a in nadir_rad
            ]
        )

    def evaluate(
        self, scenario: str, energy_MeV: np.ndarray, nadir_rad: Optional[np.ndarray], mode: Optional[str]
    ) -> np.ndarray:
        """The probabilities of a scenario over a grid.

        Args:
            scenario: ``"vacuum"``, ``"solar"``, ``"earth"`` or ``"solar_earth"``.
            energy_MeV: Energies in MeV.
            nadir_rad: Nadir angles in radians (the scenarios that cross the Earth).
            mode: PEANUTS Earth mode (those same scenarios).

        Returns:
            ``(n_nadir, n_energy, 3)`` or ``(n_energy, 3)``.

        Raises:
            ValueError: If the scenario is unavailable in PEANUTS or unknown.
        """
        if scenario in UNAVAILABLE:
            raise ValueError(f"{scenario}: {UNAVAILABLE[scenario]}")
        if scenario == "vacuum":
            return self.vacuum(energy_MeV)
        if scenario == "solar":
            return self.solar(energy_MeV)
        if scenario == "earth":
            return self.earth(energy_MeV, nadir_rad, mode)
        if scenario == "solar_earth":
            return self.solar_earth(energy_MeV, nadir_rad, mode)
        raise ValueError(f"unknown scenario {scenario!r}.")

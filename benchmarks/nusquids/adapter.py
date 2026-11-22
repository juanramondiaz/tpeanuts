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

"""Adapter to nuSQuIDS: the benchmark scenarios it can run, evaluated over the shared grids.

nuSQuIDS is a C++ solver reached through its Python bindings, and it runs in its own environment (the WSL virtual
environment): this module imports neither ``torch`` nor ``tpeanuts``, and reuses the helpers of
``validation.nusquids.adapter`` (solver set-up, bodies, tracks, read-out), which are pure nuSQuIDS as well.

A solver is used the way a user of the package uses it: the body is created once, and for every nadir angle a track is
set and for every energy the energy and the initial state are set and the state is evolved. Creating the solver and the
body is the ``build`` phase; setting tracks and energies and evolving is the ``evaluate`` phase. nuSQuIDS evolves one
energy at a time through this interface, so a grid is a Python loop over those calls.

Which scenarios have a nuSQuIDS equivalent:

* ``vacuum``: the same baseline and initial flavour (``nue``).
* ``earth``: the Earth body with the PREM table, ``nue``; the nadir angle ``eta`` is ``cos(zenith) = -cos(eta)``.
* ``atmosphere``: ``EarthAtm``, ``numu``. Its atmosphere is fixed (production height of 22 km), so it is compared with
  tpeanuts computed with that single height, not with the four-height average of the catalogue.
* ``solar`` and ``solar_earth`` are unavailable: tpeanuts integrates the 8B flux over its production distribution in the
  Sun, nuSQuIDS evolves from one production radius, and the two are not the same observable.

Module contents:
    UNAVAILABLE
        The scenarios nuSQuIDS has no equivalent for, with the reason.
    PAIRED_METHOD
        The tpeanuts method whose algorithm nuSQuIDS shares (a numerical integration of the evolution equation).
    configuration(...)
        The ``NuSQuIDSConfig`` that carries the shared oscillation parameters.
    NuSQuIDSBackend
        The solver and bodies, and the scenarios over a grid.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from benchmarks.common.scenarios import OSCILLATION
from validation.nusquids import adapter as nsq_adapter
from validation.nusquids.config import NuSQuIDSConfig

PAIRED_METHOD = "numerical"
UNAVAILABLE = {
    "solar": "tpeanuts integrates the 8B flux over the production distribution in the Sun; nuSQuIDS evolves from one "
    "production radius, which is a different observable",
    "solar_earth": "it needs the solar leg, which has no equivalent observable in nuSQuIDS (see the solar scenario)",
}
VACUUM_BASELINE_KM = 180.0
_EARTH_FILE = "EARTH_MODEL_PREM.dat"


def configuration() -> NuSQuIDSConfig:
    """The ``NuSQuIDSConfig`` with the oscillation parameters of ``benchmarks.common.scenarios.OSCILLATION``."""
    return NuSQuIDSConfig(
        theta12=math.radians(OSCILLATION["theta12_deg"]),
        theta13=math.radians(OSCILLATION["theta13_deg"]),
        theta23=math.radians(OSCILLATION["theta23_deg"]),
        delta_cp=math.radians(OSCILLATION["delta13_deg"]),
        DeltamSq21=OSCILLATION["DeltamSq21_eV2"],
        DeltamSq3l=OSCILLATION["DeltamSq3l_eV2"],
    )


class NuSQuIDSBackend:
    """A nuSQuIDS solver for one scenario, evaluated over (nadir, energy) grids.

    Args:
        scenario: ``"vacuum"``, ``"earth"`` or ``"atmosphere"``.
        config: Oscillation and solver configuration (default: ``configuration()``).

    Attributes:
        scenario: The scenario this backend was built for.
        config: The configuration used.

    Raises:
        ValueError: If the scenario is unavailable in nuSQuIDS or unknown.
        validation.nusquids.adapter.NuSQuIDSError: If nuSQuIDS cannot be imported.
    """

    def __init__(self, scenario: str, config: Optional[NuSQuIDSConfig] = None) -> None:
        if scenario in UNAVAILABLE:
            raise ValueError(f"{scenario}: {UNAVAILABLE[scenario]}")
        if scenario not in ("vacuum", "earth", "atmosphere"):
            raise ValueError(f"unknown scenario {scenario!r}.")
        self.scenario = scenario
        self.config = config or configuration()
        self._nsq = nsq_adapter.require_nusquids()
        self._units = nsq_adapter.units(self._nsq)
        self._solver = nsq_adapter.init_solver(antinu=False, config=self.config)
        if scenario == "vacuum":
            self._body: Any = self._nsq.Vacuum()
            self._flavour = "nue"
        elif scenario == "earth":
            self._body = self._nsq.Earth(*nsq_adapter.data_file_args(_EARTH_FILE))
            self._flavour = "nue"
        else:
            self._body = self._nsq.EarthAtm(*nsq_adapter.data_file_args(_EARTH_FILE))
            self._flavour = "numu"
        self._solver.Set_Body(self._body)
        self._state = nsq_adapter.initial_state(self._flavour, n_flavours=self.config.numneu)

    def _track(self, nadir_rad: Optional[float]) -> Any:
        if self.scenario == "vacuum":
            return self._nsq.Vacuum.Track(VACUUM_BASELINE_KM * self._units.km)
        cos_zenith = -math.cos(float(nadir_rad))  # type: ignore[arg-type]
        if self.scenario == "earth":
            return nsq_adapter._build_earth_track_from_cosine(self._nsq, self._body, cos_zenith)
        return self._body.MakeTrackWithCosine(cos_zenith)

    def _evolve(self, energy_MeV: float) -> np.ndarray:
        self._solver.Set_E(float(energy_MeV) * 1.0e-3 * self._units.GeV)
        self._solver.Set_initial_state(self._state, self._nsq.Basis.flavor)
        self._solver.EvolveState()
        return nsq_adapter.eval_probabilities(self._solver, numneu=self.config.numneu)

    def evaluate(self, energy_MeV: np.ndarray, nadir_rad: Optional[np.ndarray]) -> np.ndarray:
        """The flavour probabilities over a grid.

        Args:
            energy_MeV: Energies in MeV.
            nadir_rad: Nadir angles in radians (``None`` for vacuum).

        Returns:
            ``(n_nadir, n_energy, 3)``, or ``(n_energy, 3)`` for vacuum.
        """
        if self.scenario == "vacuum":
            self._solver.Set_Track(self._track(None))
            return np.stack([self._evolve(e) for e in energy_MeV])
        assert nadir_rad is not None
        out = []
        for angle in nadir_rad:
            self._solver.Set_Track(self._track(angle))
            out.append(np.stack([self._evolve(e) for e in energy_MeV]))
        return np.stack(out)


def describe(config: NuSQuIDSConfig) -> Dict[str, Any]:
    """The solver settings that affect the result, for the record of a run."""
    return {
        "rel_error": config.rel_error,
        "abs_error": config.abs_error,
        "h_max_km": config.h_max_km,
        "numneu": config.numneu,
    }

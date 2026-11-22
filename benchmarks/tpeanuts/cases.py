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

"""The tpeanuts side of every benchmark scenario: build the objects, evaluate them, chunk the grid to a memory budget.

Each scenario of ``benchmarks.common.scenarios`` is implemented here once, through the public API only (the media and
legs of ``tpeanuts.medium`` and ``tpeanuts.pipeline``), so that what is timed is what a user calls. A ``Case`` separates
the two phases the protocol times apart: ``build`` creates the model, the media and the legs (the media are lazy, so
their profiles are built at evaluation time, as in normal use), and ``evaluate`` propagates the whole grid and returns a
checksum, or the full probabilities with ``probabilities``.

The grid of (nadir angle, energy) points is evaluated in chunks whose size keeps ``points * work_per_point`` under a
budget, so that the largest sizes of the catalogue do not exhaust the memory; the chunking is part of what is timed,
as it would be for a user who needs those sizes. The probabilities do not depend on the chunking (it is tested).

Scenarios and what they propagate (the initial flavour is the one that source emits or, for the media alone, ``nue``;
the atmosphere one uses a synthetic flux, so that the benchmark needs no external data and times only the propagation):

* ``vacuum``: one baseline of 180 km, energies on the solar grid.
* ``solar``: the 8B channel from its production in the Sun to the surface.
* ``earth``: a surface detector, ``nue`` through the Earth along each nadir angle.
* ``solar_earth``: the 8B flux through the Sun and then the Earth along each nadir angle.
* ``atmosphere``: ``numu`` produced at four altitudes, through the atmosphere and then the Earth.

Module contents:
    METHODS
        The propagation methods the Earth and atmosphere scenarios can be timed with, by name.
    Case
        One scenario at one size: its ``build``, ``evaluate`` and ``probabilities``.
    build_case(...)
        Create the ``Case`` of a scenario, sizes, context, method and discretisation.
    make_context(...)
        The ``RuntimeContext`` of a device and precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from benchmarks.common.scenarios import OSCILLATION, SCENARIOS, energy_grid_MeV, nadir_grid_rad
from tpeanuts.interfaces.flux import Flux, ProductionDistribution
from tpeanuts.medium import EarthMedium, VacuumMedium
from tpeanuts.medium.solar.solar import SolarMedium
from tpeanuts.physics.common.mass_spectrum import MassSpectrum
from tpeanuts.physics.common.mixing import MixingMatrix, StandardMixingParameters
from tpeanuts.physics.common.nature import Dirac
from tpeanuts.physics.common.space import FlavourSpace
from tpeanuts.physics.common.state import DensityMatrix
from tpeanuts.physics.evolution.method import PropagationMethod
from tpeanuts.physics.model import PropagationModel
from tpeanuts.pipeline import AtmosphereLeg, EarthLeg, solar_flux
from tpeanuts.pipeline.chain import propagate_chain
from tpeanuts.source.solar.solar import SolarSource
from tpeanuts.util.context import RuntimeContext

METHODS = {
    "numerical": PropagationMethod.NUMERICAL,
    "analytic": PropagationMethod.ANALYTIC,
    "analytic_perturbative": PropagationMethod.ANALYTIC_PERTURBATIVE,
}

VACUUM_BASELINE_M = 1.8e5
ATMOSPHERE_HEIGHTS_M = (1.0e4, 2.0e4, 3.0e4, 4.0e4)
DEFAULT_NSTEPS = 100
DEFAULT_MAX_WORK = 2_000_000

_Runner = Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]


@dataclass
class Case:
    """One scenario at one size, ready to be timed.

    Attributes:
        scenario: Name in ``benchmarks.common.scenarios.SCENARIOS``.
        sizes: The sizes of this case (``{"n_energy": ..., "n_nadir": ...}``).
        method: The propagation method of the Earth and atmosphere legs.
        context: Device and precision.
        nsteps: Chord discretisation of those legs.
        build: Zero-argument callable that creates the objects (timed as the ``build`` phase); returns them.
        evaluate: Callable taking what ``build`` returned and propagating the whole grid, returning a scalar checksum
            (the sum of the probabilities, which is the number of points when the evolution is unitary).
        probabilities: Callable taking what ``build`` returned and returning every probability, shaped
            ``(n_nadir, n_energy, 3)`` (``(n_energy, 3)`` for the scenarios without a nadir axis).
    """

    scenario: str
    sizes: Dict[str, int]
    method: PropagationMethod
    context: RuntimeContext
    nsteps: int
    build: Callable[[], Any]
    evaluate: Callable[[Any], torch.Tensor]
    probabilities: Callable[[Any], torch.Tensor]


def make_context(device: str = "cpu", dtype: str = "float64") -> RuntimeContext:
    """The ``RuntimeContext`` of a device (``"cpu"``, ``"cuda"``) and precision (``"float32"``, ``"float64"``).

    Args:
        device: Device name as ``torch`` accepts it.
        dtype: ``"float32"`` or ``"float64"``.

    Returns:
        The resolved context.

    Raises:
        ValueError: If ``dtype`` is neither of the two.
    """
    if dtype not in ("float32", "float64"):
        raise ValueError(f"dtype must be 'float32' or 'float64', got {dtype!r}.")
    return RuntimeContext.resolve(device, torch.float32 if dtype == "float32" else torch.float64)


def build_case(
    scenario: str,
    sizes: Dict[str, int],
    *,
    context: RuntimeContext,
    method: PropagationMethod = PropagationMethod.NUMERICAL,
    nsteps: int = DEFAULT_NSTEPS,
    max_work: int = DEFAULT_MAX_WORK,
    heights_m: Optional[Sequence[float]] = None,
) -> Case:
    """Create the ``Case`` of a scenario at the given sizes.

    Args:
        scenario: One of ``benchmarks.common.scenarios.SCENARIOS``.
        sizes: The size of each axis of that scenario (``n_energy``, and ``n_nadir`` where it has one).
        context: Device and precision of the run.
        method: Propagation method of the Earth and atmosphere legs (ignored by ``vacuum`` and ``solar``).
        nsteps: Chord discretisation of those legs.
        max_work: Budget of one chunk, in (points x segments) evaluated at once; the grid is split to stay under it.
        heights_m: Production heights of the ``atmosphere`` scenario, in metres, weighted equally (default
            ``ATMOSPHERE_HEIGHTS_M``); ignored by the other scenarios. nuSQuIDS has one fixed height, so its reference
            is computed with a single one.

    Returns:
        The case.

    Raises:
        ValueError: If the scenario is unknown or ``sizes`` does not have exactly its axes.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {sorted(SCENARIOS)}.")
    spec = SCENARIOS[scenario]
    if set(sizes) != set(spec.sizes):
        raise ValueError(f"scenario {scenario!r} needs the sizes {sorted(spec.sizes)}, got {sorted(sizes)}.")

    on_device = {"dtype": context.dtype, "device": context.device}
    energy = torch.as_tensor(energy_grid_MeV(spec.energy_kind, sizes["n_energy"]), **on_device)
    nadir = None
    if "n_nadir" in sizes:
        nadir = torch.as_tensor(nadir_grid_rad(sizes["n_nadir"]), **on_device)
    work_per_point = _WORK[scenario](nsteps, method)

    def build() -> _Runner:
        model = _model(context)
        if scenario == "atmosphere":
            return _atmosphere(model, context, method, nsteps, heights_m=heights_m or ATMOSPHERE_HEIGHTS_M)
        return _RUNNERS[scenario](model, context, method, nsteps)

    def probabilities(runner: _Runner) -> torch.Tensor:
        return _evaluate_chunks(runner, energy, nadir, work_per_point, max_work, collect=True)

    def evaluate(runner: _Runner) -> torch.Tensor:
        return _evaluate_chunks(runner, energy, nadir, work_per_point, max_work, collect=False)

    return Case(scenario, dict(sizes), method, context, nsteps, build, evaluate, probabilities)


def _model(context: RuntimeContext) -> PropagationModel:
    """The three-flavour Dirac model every scenario propagates with."""
    space = FlavourSpace()
    standard = StandardMixingParameters(
        theta12=math.radians(OSCILLATION["theta12_deg"]),
        theta13=math.radians(OSCILLATION["theta13_deg"]),
        theta23=math.radians(OSCILLATION["theta23_deg"]),
        delta13=math.radians(OSCILLATION["delta13_deg"]),
        context=context,
    )
    mixing = MixingMatrix(space=space, standard=standard, nature=Dirac(), context=context)
    masses = MassSpectrum(
        space=space,
        DeltamSq21=OSCILLATION["DeltamSq21_eV2"],
        DeltamSq3l=OSCILLATION["DeltamSq3l_eV2"],
        context=context,
    )
    return PropagationModel(mixing=mixing, mass_spectrum=masses)


def _basis_flux(energy: torch.Tensor, flavour: int, context: RuntimeContext, **extra: Any) -> Flux:
    """A flux of one flavour eigenstate on ``energy``."""
    state = DensityMatrix.basis_state(flavour, FlavourSpace(), context)
    return Flux(energy_MeV=energy, state=state, antinu=False, **extra)


def _vacuum(model: PropagationModel, context: RuntimeContext, method: PropagationMethod, nsteps: int) -> _Runner:
    length = torch.tensor(VACUUM_BASELINE_M, dtype=context.dtype, device=context.device)
    medium = VacuumMedium(length_m=length, context=context)

    def run(energy: torch.Tensor, nadir: Optional[torch.Tensor]) -> torch.Tensor:
        return medium.propagate(_basis_flux(energy, 0, context), model).state.probabilities()

    return run


def _solar(model: PropagationModel, context: RuntimeContext, method: PropagationMethod, nsteps: int) -> _Runner:
    sun = SolarMedium.default(context=context)
    source = SolarSource.default(context=context)

    def run(energy: torch.Tensor, nadir: Optional[torch.Tensor]) -> torch.Tensor:
        return solar_flux(model, source, sun, "8B", energy, intensity=False)["8B"].state.probabilities()

    return run


def _earth(model: PropagationModel, context: RuntimeContext, method: PropagationMethod, nsteps: int) -> _Runner:
    def run(energy: torch.Tensor, nadir: Optional[torch.Tensor]) -> torch.Tensor:
        assert nadir is not None
        medium = EarthMedium(eta_rad=nadir.reshape(-1, 1), context=context, method=method, nsteps=nsteps)
        return medium.propagate(_basis_flux(energy, 0, context), model).state.probabilities()

    return run


def _solar_earth(model: PropagationModel, context: RuntimeContext, method: PropagationMethod, nsteps: int) -> _Runner:
    sun = SolarMedium.default(context=context)
    source = SolarSource.default(context=context)

    def run(energy: torch.Tensor, nadir: Optional[torch.Tensor]) -> torch.Tensor:
        assert nadir is not None
        earth = EarthLeg(eta_rad=nadir.reshape(-1, 1), method=method, nsteps=nsteps)
        return solar_flux(model, source, sun, "8B", energy, earth=earth, intensity=False)["8B"].state.probabilities()

    return run


def _atmosphere(
    model: PropagationModel,
    context: RuntimeContext,
    method: PropagationMethod,
    nsteps: int,
    heights_m: Sequence[float] = ATMOSPHERE_HEIGHTS_M,
) -> _Runner:
    heights = torch.tensor(tuple(heights_m), dtype=context.dtype, device=context.device)
    weight = torch.full_like(heights, 1.0 / heights.numel())
    production = ProductionDistribution(radius=heights, weight=weight, context=context, measure="shell_fraction")
    atmosphere = AtmosphereLeg(method=method, nsteps=nsteps)
    earth = EarthLeg(method=method, nsteps=nsteps)

    def run(energy: torch.Tensor, nadir: Optional[torch.Tensor]) -> torch.Tensor:
        assert nadir is not None
        flux = _basis_flux(
            energy, 1, context, differential_flux=_synthetic_flux(energy, nadir), production=production, direction=nadir
        )
        steps = [lambda current: atmosphere.medium(current, 0.0), earth]
        return propagate_chain(flux, steps, model).state.probabilities()

    return run


def _synthetic_flux(energy: torch.Tensor, nadir: torch.Tensor) -> torch.Tensor:
    """A smooth muon-neutrino-like differential flux, ``(n_nadir, n_energy)``: a power law with a cut-off."""
    x = energy / 1000.0
    spectrum = 250.0 * x.pow(-2.05) * torch.exp(-energy / 6.0e4)
    return spectrum.unsqueeze(0) * (1.0 + 0.5 * torch.cos(nadir)).unsqueeze(1)


_RUNNERS: Dict[str, Callable[..., _Runner]] = {
    "vacuum": _vacuum,
    "solar": _solar,
    "earth": _earth,
    "solar_earth": _solar_earth,
    "atmosphere": _atmosphere,
}

# Segments crossed per point, which sets how many points fit in one chunk (the methods that fit the density with a few
# polynomial segments or one step cost far less than ``nsteps``, but share the budget for a like-for-like grid).
_WORK: Dict[str, Callable[[int, PropagationMethod], int]] = {
    "vacuum": lambda nsteps, method: 1,
    "solar": lambda nsteps, method: 1,
    "earth": lambda nsteps, method: nsteps,
    "solar_earth": lambda nsteps, method: nsteps,
    "atmosphere": lambda nsteps, method: nsteps * (len(ATMOSPHERE_HEIGHTS_M) + 1),
}


def _chunk_sizes(n_energy: int, n_nadir: Optional[int], work_per_point: int, max_work: int) -> Tuple[int, int]:
    """Energy and nadir chunk sizes whose product times ``work_per_point`` stays under ``max_work`` (at least 1x1)."""
    points = max(1, max_work // max(1, work_per_point))
    energy_chunk = min(n_energy, points)
    nadir_chunk = 1 if n_nadir is None else max(1, min(n_nadir, points // energy_chunk))
    return energy_chunk, nadir_chunk


def _evaluate_chunks(
    runner: _Runner,
    energy: torch.Tensor,
    nadir: Optional[torch.Tensor],
    work_per_point: int,
    max_work: int,
    *,
    collect: bool,
) -> torch.Tensor:
    """Run ``runner`` over the (nadir, energy) grid in chunks; the probabilities, or just their sum."""
    n_energy = energy.shape[0]
    energy_chunk, nadir_chunk = _chunk_sizes(
        n_energy, None if nadir is None else nadir.shape[0], work_per_point, max_work
    )
    checksum = torch.zeros((), dtype=energy.dtype, device=energy.device)
    rows: List[torch.Tensor] = []
    nadir_starts = [0] if nadir is None else range(0, nadir.shape[0], nadir_chunk)
    for n0 in nadir_starts:
        nadir_part = None if nadir is None else nadir[n0 : n0 + nadir_chunk]
        columns: List[torch.Tensor] = []
        for e0 in range(0, n_energy, energy_chunk):
            part = runner(energy[e0 : e0 + energy_chunk], nadir_part)
            if collect:
                columns.append(part)
            else:
                checksum = checksum + part.sum()
        if collect:
            rows.append(torch.cat(columns, dim=-2))
    if collect:
        return torch.cat(rows, dim=0) if nadir is not None else rows[0]
    return checksum

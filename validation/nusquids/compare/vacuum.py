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

"""Vacuum-oscillation comparison: tpeanuts vs nuSQuIDS.

The first concrete ``validation.nusquids.compare`` script (Phase C, first
entry): the simplest possible case, no matter effects at all, so any
disagreement it finds is purely about the oscillation formula/PMNS
convention, not about matter potentials, densities, or path
discretisation.

This module is the one place allowed to import BOTH ``tpeanuts`` and
``validation.nusquids.adapter`` -- that is its entire purpose (see
``validation.nusquids``'s own package docstring for the independence
rule this bridging role is deliberately exempt from).

Both sides are built from the very same ``validation.nusquids.config.
NuSQuIDSConfig`` instance's oscillation fields (``theta12``,
``theta13``, ``theta23``, ``delta_cp``, ``DeltamSq21``,
``DeltamSq3l``), so a mismatch can never be traced to the two sides
having silently used different physical inputs.

Module contents:
    tpeanuts_probability_vacuum(...)
        Compute P(nu_i -> nu_alpha) for a fixed vacuum baseline using
        tpeanuts's own ``PropagationModel``/``evolve_path`` machinery.
    compare_vacuum(...)
        Run ``validation.nusquids.compare.grid.compare_grid`` between
        ``tpeanuts_probability_vacuum`` and
        ``validation.nusquids.adapter.probability_vacuum`` over an
        energy/baseline grid, once per requested initial flavour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch

from tpeanuts.medium.vacuum.vacuum import VacuumMedium
from tpeanuts.physics.common.state import DensityMatrix
from tpeanuts.physics.common.terms import PropagationEnvironment
from tpeanuts.physics.evolution import PropagationMethod
from tpeanuts.physics.evolution.dispatch import evolve_path
from tpeanuts.physics.model import PropagationModel
from tpeanuts.util.constant import R_E
from tpeanuts.util.context import RuntimeContext
from validation.nusquids.adapter import initial_state, probability_vacuum
from validation.nusquids.compare.grid import ComparisonResult, GridPoint, compare_grid, export_csv
from validation.nusquids.compare.oscillation_model import mixing_and_mass_spectrum
from validation.nusquids.config import NuSQuIDSConfig

_LABELS = ("P_nue", "P_numu", "P_nutau")


def tpeanuts_probability_vacuum(
    *, E_GeV: float, baseline_km: float, initial_flavour: str, antinu: bool = False,
    config: Optional[NuSQuIDSConfig] = None, context: Optional[RuntimeContext] = None,
) -> np.ndarray:
    """Compute vacuum flavour probabilities with tpeanuts's own propagation machinery.

    Builds a single-segment ``VacuumMedium`` at ``baseline_km`` and
    evolves a pure initial flavour eigenstate through it with
    ``PropagationMethod.ANALYTIC`` (exact for a single constant --
    here zero -- density segment), then reads off the final flavour
    populations. Mirrors
    ``validation.nusquids.adapter.probability_vacuum``'s call signature
    exactly, so both are valid ``validation.nusquids.compare.grid.
    ProbabilityFn`` implementations for the same grid.

    Args:
        E_GeV: Neutrino energy in GeV.
        baseline_km: Vacuum baseline in km.
        initial_flavour: Initial flavour label ("nue", "numu",
            "nutau" -- the three active flavours; "sterile" is not
            valid here, this plain FlavourSpace has no sterile state).
        antinu: If True, propagate the antineutrino state.
        config: Oscillation parameters, reused directly from
            ``validation.nusquids.config.NuSQuIDSConfig`` so both sides
            of the comparison share exactly the same physical input.
            Defaults to ``NuSQuIDSConfig()`` when None.
        context: RuntimeContext for the computation. Defaults to
            CPU/float64 (comparison precision) when None.

    Returns:
        1D float64 numpy array of length 3: P(nu_e), P(nu_mu),
        P(nu_tau) (or antineutrino equivalents), summing to 1.
    """
    cfg = NuSQuIDSConfig() if config is None else config
    ctx = RuntimeContext.resolve("cpu", torch.float64) if context is None else context

    mixing, mass_spectrum = mixing_and_mass_spectrum(cfg, ctx)
    model = PropagationModel(mixing=mixing, mass_spectrum=mass_spectrum, antinu=antinu)

    medium_profile = VacuumMedium(length_m=float(baseline_km) * 1.0e3, context=ctx).profile()
    E_MeV = torch.tensor(float(E_GeV) * 1.0e3, dtype=ctx.dtype, device=ctx.device)
    environment = PropagationEnvironment(
        E_MeV=E_MeV, n_e_mol_cm3=medium_profile.n_e_mol_cm3, n_n_mol_cm3=medium_profile.n_n_mol_cm3,
    )
    dx_segments = medium_profile.length_m / R_E

    evolutor = evolve_path(
        model, environment, dx_segments, method=PropagationMethod.ANALYTIC,
        perturbative_profile=medium_profile.perturbative_profile, evolution_scale_m=R_E,
    )

    flavour_idx = int(np.argmax(initial_state(initial_flavour, n_flavours=3)))
    initial = DensityMatrix.basis_state(flavour_idx, mixing.space, ctx)
    evolved = initial.conjugate_by(evolutor)

    return evolved.probabilities().detach().cpu().numpy().astype(np.float64).reshape(3)


def compare_vacuum(
    energies_GeV: Sequence[float], baselines_km: Sequence[float],
    *, initial_flavours: Sequence[str] = ("nue", "numu", "nutau"), antinu: bool = False,
    config: Optional[NuSQuIDSConfig] = None, context: Optional[RuntimeContext] = None,
) -> Dict[str, ComparisonResult]:
    """Compare tpeanuts and nuSQuIDS vacuum probabilities over an energy/baseline grid.

    One ``ComparisonResult`` is built per initial flavour, since
    ``validation.nusquids.compare.grid.compare_grid``'s grid points are
    float-only (the flavour label cannot itself be a grid coordinate).

    Args:
        energies_GeV: Neutrino energies in GeV.
        baselines_km: Vacuum baselines in km.
        initial_flavours: Initial flavour labels to compare, one
            ``ComparisonResult`` each.
        antinu: If True, compare the antineutrino channel.
        config: Shared oscillation configuration for both sides.
            Defaults to ``NuSQuIDSConfig()`` when None.
        context: tpeanuts RuntimeContext. Defaults to CPU/float64 when
            None.

    Returns:
        Dict mapping each initial flavour label to its own
        ``ComparisonResult`` (nuSQuIDS as ``reference``, tpeanuts as
        ``candidate``).

    Raises:
        validation.nusquids.adapter.NuSQuIDSError: If nuSQuIDS bindings
            are not installed (this function calls the real solver;
            check ``validation.nusquids.adapter.is_available()`` first).
    """
    cfg = NuSQuIDSConfig() if config is None else config
    grid_points: Sequence[GridPoint] = [
        {"E_GeV": e, "baseline_km": b} for e in energies_GeV for b in baselines_km
    ]

    def reference_fn(E_GeV: float, baseline_km: float, *, _flavour: str) -> np.ndarray:
        return probability_vacuum(
            E_GeV=E_GeV, baseline_km=baseline_km, initial_flavour=_flavour, antinu=antinu, config=cfg
        )

    def candidate_fn(E_GeV: float, baseline_km: float, *, _flavour: str) -> np.ndarray:
        return tpeanuts_probability_vacuum(
            E_GeV=E_GeV, baseline_km=baseline_km, initial_flavour=_flavour, antinu=antinu, config=cfg, context=context,
        )

    results: Dict[str, ComparisonResult] = {}
    for flavour in initial_flavours:
        results[flavour] = compare_grid(
            grid_points,
            lambda E_GeV, baseline_km, _flavour=flavour: reference_fn(E_GeV, baseline_km, _flavour=_flavour),
            lambda E_GeV, baseline_km, _flavour=flavour: candidate_fn(E_GeV, baseline_km, _flavour=_flavour),
            labels=_LABELS,
        )
    return results


def export_all_csv(results: Dict[str, ComparisonResult], output_dir: Union[str, Path]) -> None:
    """Export each flavour's ``ComparisonResult`` to its own CSV file.

    Args:
        results: Mapping from initial flavour label to its
            ``ComparisonResult``, as returned by ``compare_vacuum``.
        output_dir: Directory to write ``vacuum_{flavour}.csv`` files
            into (created if missing).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for flavour, result in results.items():
        export_csv(result, output_path / f"vacuum_{flavour}.csv")

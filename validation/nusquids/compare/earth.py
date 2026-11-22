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

"""Earth-crossing oscillation comparison: tpeanuts vs nuSQuIDS.

Second ``validation.nusquids.compare`` script (Phase C). Two physically
distinct cases, both present in ``validation.nusquids.adapter``:

    Pure flavour eigenstate through the Earth
        ``tpeanuts_probability_earth`` vs ``adapter.probability_earth``.
    Incoherent vacuum-mass mixture through the Earth
        (the solar-neutrino-after-Sun-Earth-decoherence case)
        ``tpeanuts_probability_earth_massbasis`` vs
        ``adapter.probability_earth_massbasis``, built from
        ``tpeanuts_transition_matrix_earth_mass_to_flavour`` vs
        ``adapter.transition_matrix_earth_mass_to_flavour``.

Geometry note -- nadir angle vs cos(zenith): tpeanuts's own
``tpeanuts.medium.earth.earth.EarthMedium`` is parameterized by a
detector nadir angle ``eta_rad`` (0 = straight through the Earth's centre,
pi/2 = grazing incidence, chord length 0), while
``validation.nusquids.adapter``'s functions take a detector
``cos_zenith`` (1 = straight down, no Earth crossing; -1 = straight up,
full diameter). For a surface detector (``depth_m=0``, the only case
this module covers), the two chord-length formulas
(``EarthMedium``'s exact PREM formula and the adapter's straight-line
``L = max(0, -2 R_earth cos_zenith)``) agree exactly at
``eta_rad = arccos(-cos_zenith)``: both reduce to ``L = 2 R_earth
cos(eta_rad)`` there. Confirmed algebraically, not just asserted. This
restricts the valid domain to ``cos_zenith < 0`` (genuine Earth
crossing) -- ``cos_zenith >= 0`` has no Earth-medium equivalent on the
tpeanuts side (a surface detector looking at the sky needs
``validation.nusquids.compare.vacuum`` instead, not this module).

Verification without nuSQuIDS installed: unlike ``compare/vacuum.py``
(checked against a closed-form textbook formula), a genuine PREM
Earth-crossing trajectory has no closed-form oracle -- this module's
own tests instead check physical invariants already established by
``tpeanuts/test/test_earth_benchmark.py`` (unitarity, a measurable
matter-effect difference from vacuum at the same chord length,
PropagationMethod.ANALYTIC matching NUMERICAL). Direct nuSQuIDS
agreement is deferred to whenever the bindings are installed (see
``is_available()``-gated tests).

Module contents:
    tpeanuts_probability_earth(...)
        Pure flavour eigenstate through EarthMedium.
    tpeanuts_transition_matrix_earth_mass_to_flavour(...)
        Per-vacuum-mass-eigenstate flavour transition matrix through
        EarthMedium.
    tpeanuts_probability_earth_massbasis(...)
        Incoherent vacuum-mass mixture through EarthMedium.
    compare_earth(...)
        Run ``validation.nusquids.compare.grid.compare_grid`` between
        each tpeanuts function above and its adapter counterpart, over
        an energy/cos(zenith) grid.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch

from tpeanuts.medium.earth.earth import EarthMedium
from tpeanuts.physics.common.state import DensityMatrix
from tpeanuts.physics.common.terms import PropagationEnvironment
from tpeanuts.physics.evolution import PropagationMethod
from tpeanuts.physics.evolution.dispatch import evolve_path
from tpeanuts.physics.model import PropagationModel
from tpeanuts.util.constant import R_E
from tpeanuts.util.context import RuntimeContext
from validation.nusquids.adapter import (
    initial_state,
    probability_earth,
    probability_earth_massbasis,
)
from validation.nusquids.compare.grid import ComparisonResult, GridPoint, compare_grid, export_csv
from validation.nusquids.compare.oscillation_model import mixing_and_mass_spectrum
from validation.nusquids.config import NuSQuIDSConfig

_LABELS = ("P_nue", "P_numu", "P_nutau")


def _nadir_angle_from_cos_zenith(cos_zenith: float) -> float:
    """Convert a detector cos(zenith) to tpeanuts's own nadir angle ``eta_rad``, surface-detector case only.

    Args:
        cos_zenith: Detector cos(zenith angle). Must be strictly
            negative (genuine Earth crossing) -- see this module's own
            docstring for why ``cos_zenith >= 0`` has no
            ``EarthMedium`` equivalent.

    Returns:
        ``eta_rad = arccos(-cos_zenith)``, in ``(0, pi/2)``.

    Raises:
        ValueError: If ``cos_zenith >= 0``.
    """
    if cos_zenith >= 0.0:
        raise ValueError(
            f"cos_zenith={cos_zenith} has no Earth-crossing trajectory for a surface detector "
            "(cos_zenith >= 0 means the neutrino never enters the Earth) -- use "
            "validation.nusquids.compare.vacuum instead."
        )
    return math.acos(-cos_zenith)


def _earth_evolutor(
    *, E_GeV: float, cos_zenith: float, antinu: bool, config: NuSQuIDSConfig, context: RuntimeContext, nsteps: int,
):
    """Build the model, environment-derived evolutor, and mixing for one Earth-crossing point."""
    eta_rad = _nadir_angle_from_cos_zenith(cos_zenith)
    mixing, mass_spectrum = mixing_and_mass_spectrum(config, context)
    model = PropagationModel(mixing=mixing, mass_spectrum=mass_spectrum, antinu=antinu)

    medium_profile = EarthMedium(
        eta_rad=eta_rad, context=context, method=PropagationMethod.ANALYTIC, nsteps=nsteps,
    ).profile()
    E_MeV = torch.tensor(float(E_GeV) * 1.0e3, dtype=context.dtype, device=context.device)
    environment = PropagationEnvironment(
        E_MeV=E_MeV, n_e_mol_cm3=medium_profile.n_e_mol_cm3, n_n_mol_cm3=medium_profile.n_n_mol_cm3,
    )
    dx_segments = medium_profile.length_m / R_E

    evolutor = evolve_path(
        model, environment, dx_segments, method=PropagationMethod.ANALYTIC,
        perturbative_profile=medium_profile.perturbative_profile, evolution_scale_m=R_E,
    )
    return evolutor, mixing


def tpeanuts_probability_earth(
    *, E_GeV: float, cos_zenith: float, initial_flavour: str, antinu: bool = False,
    config: Optional[NuSQuIDSConfig] = None, context: Optional[RuntimeContext] = None, nsteps: int = 200,
) -> np.ndarray:
    """Propagate a pure initial flavour eigenstate through tpeanuts's own EarthMedium.

    Same call signature as ``validation.nusquids.adapter.probability_earth``,
    so both are valid ``validation.nusquids.compare.grid.ProbabilityFn``
    implementations for the same grid.

    Args:
        E_GeV: Neutrino energy in GeV.
        cos_zenith: Detector cos(zenith angle); must be strictly
            negative (see this module's own docstring).
        initial_flavour: Initial flavour label ("nue", "numu",
            "nutau").
        antinu: If True, propagate the antineutrino state.
        config: Shared oscillation configuration. Defaults to
            ``NuSQuIDSConfig()`` when None.
        context: RuntimeContext. Defaults to CPU/float64 when None.
        nsteps: Number of equal-length chord segments (see
            ``EarthMedium``). 200 by default -- comfortably converged
            for ``PropagationMethod.ANALYTIC`` (exact per segment).

    Returns:
        1D float64 numpy array of length 3, summing to 1.
    """
    cfg = NuSQuIDSConfig() if config is None else config
    ctx = RuntimeContext.resolve("cpu", torch.float64) if context is None else context

    evolutor, mixing = _earth_evolutor(
        E_GeV=E_GeV, cos_zenith=cos_zenith, antinu=antinu, config=cfg, context=ctx, nsteps=nsteps,
    )
    flavour_idx = int(np.argmax(initial_state(initial_flavour, n_flavours=3)))
    initial = DensityMatrix.basis_state(flavour_idx, mixing.space, ctx)
    evolved = initial.conjugate_by(evolutor)

    return evolved.probabilities().detach().cpu().numpy().astype(np.float64).reshape(3)


def tpeanuts_transition_matrix_earth_mass_to_flavour(
    *, E_GeV: float, cos_zenith: float, antinu: bool = False,
    config: Optional[NuSQuIDSConfig] = None, context: Optional[RuntimeContext] = None, nsteps: int = 200,
) -> np.ndarray:
    """Build the Earth-crossing mass-to-flavour probability matrix with tpeanuts's own machinery.

    Each pure vacuum mass eigenstate ``|nu_i>`` is expressed in the
    flavour basis as the i-th column of the vacuum PMNS matrix
    (``mixing.pmns_matrix()[:, i]``), propagated coherently through the
    same Earth evolutor as ``tpeanuts_probability_earth``, and its
    final flavour populations become column ``i`` of the returned
    matrix -- the same construction
    ``validation.nusquids.adapter.transition_matrix_earth_mass_to_flavour``
    performs by propagating nuSQuIDS's own mass-basis initial state.

    Args:
        E_GeV: Neutrino energy in GeV.
        cos_zenith: Detector cos(zenith angle); must be strictly
            negative.
        antinu: If True, propagate antineutrinos.
        config: Shared oscillation configuration. Defaults to
            ``NuSQuIDSConfig()`` when None.
        context: RuntimeContext. Defaults to CPU/float64 when None.
        nsteps: See ``tpeanuts_probability_earth``.

    Returns:
        Real float64 array shaped ``(3, 3)``, flavour rows and initial
        -mass columns. Each column sums to 1 (unitary evolution of a
        pure state).
    """
    cfg = NuSQuIDSConfig() if config is None else config
    ctx = RuntimeContext.resolve("cpu", torch.float64) if context is None else context

    evolutor, mixing = _earth_evolutor(
        E_GeV=E_GeV, cos_zenith=cos_zenith, antinu=antinu, config=cfg, context=ctx, nsteps=nsteps,
    )
    U = mixing.pmns_matrix(antinu=antinu)

    matrix = np.zeros((3, 3), dtype=np.float64)
    for mass_idx in range(3):
        initial = DensityMatrix.pure(U[:, mass_idx], mixing.space, ctx)
        evolved = initial.conjugate_by(evolutor)
        matrix[:, mass_idx] = evolved.probabilities().detach().cpu().numpy().astype(np.float64)
    return matrix


def tpeanuts_probability_earth_massbasis(
    *, E_GeV: float, cos_zenith: float, mass_weights: Sequence[float], antinu: bool = False,
    config: Optional[NuSQuIDSConfig] = None, context: Optional[RuntimeContext] = None, nsteps: int = 200,
) -> np.ndarray:
    """Propagate an incoherent vacuum-mass mixture through tpeanuts's own EarthMedium.

    Args:
        E_GeV: Neutrino energy in GeV.
        cos_zenith: Detector cos(zenith angle); must be strictly
            negative.
        mass_weights: Incoherent mass-eigenstate weights ``w_i >= 0`` at
            Earth entry, length 3.
        antinu: If True, propagate antineutrinos.
        config: Shared oscillation configuration. Defaults to
            ``NuSQuIDSConfig()`` when None.
        context: RuntimeContext. Defaults to CPU/float64 when None.
        nsteps: See ``tpeanuts_probability_earth``.

    Returns:
        Final flavour-probability vector, length 3.

    Raises:
        ValueError: If ``mass_weights`` is not a length-3 vector.
    """
    weights = np.asarray(mass_weights, dtype=np.float64).reshape(-1)
    if weights.shape != (3,):
        raise ValueError("mass_weights must be a length-3 vector.")
    norm = weights.sum()
    if np.isfinite(norm) and norm > 0.0:
        weights = weights / norm

    matrix = tpeanuts_transition_matrix_earth_mass_to_flavour(
        E_GeV=E_GeV, cos_zenith=cos_zenith, antinu=antinu, config=config, context=context, nsteps=nsteps,
    )
    return matrix @ weights


def compare_earth(
    energies_GeV: Sequence[float], cos_zeniths: Sequence[float],
    *, initial_flavours: Sequence[str] = ("nue", "numu", "nutau"),
    mass_weights: Sequence[float] = (1.0, 0.0, 0.0), antinu: bool = False,
    config: Optional[NuSQuIDSConfig] = None, context: Optional[RuntimeContext] = None, nsteps: int = 200,
) -> Dict[str, ComparisonResult]:
    """Compare tpeanuts and nuSQuIDS Earth-crossing probabilities over an energy/cos(zenith) grid.

    Builds one ``ComparisonResult`` per initial flavour (flavour
    -eigenstate case) plus one keyed ``"massbasis"`` (incoherent
    mass-mixture case, at the fixed ``mass_weights``).

    Args:
        energies_GeV: Neutrino energies in GeV.
        cos_zeniths: Detector cos(zenith) values; every entry must be
            strictly negative (see this module's own docstring).
        initial_flavours: Initial flavour labels for the coherent case.
        mass_weights: Fixed incoherent mass weights for the
            ``"massbasis"`` entry.
        antinu: If True, compare the antineutrino channel.
        config: Shared oscillation configuration. Defaults to
            ``NuSQuIDSConfig()`` when None.
        context: tpeanuts RuntimeContext. Defaults to CPU/float64 when
            None.
        nsteps: See ``tpeanuts_probability_earth``.

    Returns:
        Dict mapping each initial flavour label, plus ``"massbasis"``,
        to its own ``ComparisonResult`` (nuSQuIDS as ``reference``,
        tpeanuts as ``candidate``).

    Raises:
        validation.nusquids.adapter.NuSQuIDSError: If nuSQuIDS bindings
            are not installed (this function calls the real solver).
    """
    cfg = NuSQuIDSConfig() if config is None else config
    grid_points: Sequence[GridPoint] = [
        {"E_GeV": e, "cos_zenith": c} for e in energies_GeV for c in cos_zeniths
    ]

    results: Dict[str, ComparisonResult] = {}
    for flavour in initial_flavours:
        results[flavour] = compare_grid(
            grid_points,
            lambda E_GeV, cos_zenith, _flavour=flavour: probability_earth(
                E_GeV=E_GeV, cos_zenith=cos_zenith, initial_flavour=_flavour, antinu=antinu, config=cfg,
            ),
            lambda E_GeV, cos_zenith, _flavour=flavour: tpeanuts_probability_earth(
                E_GeV=E_GeV, cos_zenith=cos_zenith, initial_flavour=_flavour, antinu=antinu,
                config=cfg, context=context, nsteps=nsteps,
            ),
            labels=_LABELS,
        )

    results["massbasis"] = compare_grid(
        grid_points,
        lambda E_GeV, cos_zenith: probability_earth_massbasis(
            E_GeV=E_GeV, cos_zenith=cos_zenith, mass_weights=mass_weights, antinu=antinu, config=cfg,
        ),
        lambda E_GeV, cos_zenith: tpeanuts_probability_earth_massbasis(
            E_GeV=E_GeV, cos_zenith=cos_zenith, mass_weights=mass_weights, antinu=antinu,
            config=cfg, context=context, nsteps=nsteps,
        ),
        labels=_LABELS,
    )
    return results


def export_all_csv(results: Dict[str, ComparisonResult], output_dir: Union[str, Path]) -> None:
    """Export each entry's ``ComparisonResult`` to its own CSV file.

    Args:
        results: Mapping from label (initial flavour, or
            ``"massbasis"``) to its ``ComparisonResult``, as returned
            by ``compare_earth``.
        output_dir: Directory to write ``earth_{label}.csv`` files into
            (created if missing).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for label, result in results.items():
        export_csv(result, output_path / f"earth_{label}.csv")

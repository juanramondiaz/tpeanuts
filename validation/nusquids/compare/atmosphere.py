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

"""Atmosphere + Earth oscillation comparison: tpeanuts vs nuSQuIDS.

Third ``validation.nusquids.compare`` script (Phase C). Compares
``tpeanuts_probability_atmosphere`` against
``validation.nusquids.adapter.probability_atmosphere`` -- the nuSQuIDS
reference counterpart to tpeanuts's own combined
``medium.atmosphere`` + ``medium.earth`` pipeline
(``tpeanuts.pipeline.atmospheric_flux``), rebuilt here at the
single-point level this comparison needs, rather than going through
that pipeline's own file-backed ``AtmosphericSource`` (which
loads pre-generated MCEq/Honda flux grids from disk -- irrelevant to
an arbitrary-point precision comparison).

No density-formula duplication (Phase A/B's own finding still holds):
tpeanuts's own ``tpeanuts.medium.atmosphere.atmosphere.AtmosphereMedium``
already generalizes an exponential density profile via
``density_source="exponential"`` + ``density_kwargs``, so the tpeanuts
side of this comparison simply passes nuSQuIDS's own hard-coded
``EarthAtm`` constants (``NuSQuIDSConfig.earthatm_rho0_gcm3``/
``earthatm_scale_height_km``/``earthatm_Ye``) as ``density_kwargs`` --
this project's own general-purpose atmosphere defaults
(``tpeanuts.config.default.ATMOSPHERE_RHO0_GCM3`` etc.) are a
*different*, unrelated reference atmosphere and must not be used here.

Geometry, two composed legs: an ``AtmosphereMedium`` leg (production
altitude ``h_m`` down to the detector, valid for the nadir angle's
full ``[0, pi]`` domain) composed with an ``EarthMedium`` leg (only
when the trajectory genuinely crosses the Earth, ``eta_rad < pi/2``, i.e.
``cos_zenith < 0`` -- see ``compare/earth.py``'s own docstring for the
same ``eta_rad = arccos(-cos_zenith)`` reconciliation, reused unchanged
here). For ``cos_zenith >= 0`` (downward-going, no Earth crossing at
all for a surface detector) the atmosphere leg alone is the complete
answer -- unlike ``compare/earth.py``, this module does not need to
reject that regime, since ``AtmosphereMedium`` (unlike ``EarthMedium``
at a surface detector) is valid across the whole nadir-angle domain.
The two legs' evolutors compose as ``S_total = S_earth @ S_atmosphere``
(atmosphere leg first, then the underground leg), exactly the order
``tpeanuts.pipeline.atmospheric_flux`` itself already uses.

Verification without nuSQuIDS installed: like ``compare/earth.py``,
no closed-form oracle exists for this composed geometry. This
module's own tests check: unitarity; the well-established physical
limit ``S_atmosphere -> I`` as ``h_m -> 0`` (already documented and
tested in ``tpeanuts.pipeline.atmospheric_flux``'s own module
docstring), meaning this module's own result must converge to
``compare/earth.py``'s pure-Earth-crossing result as ``h_m -> 0``; and
a measurable atmosphere-production-height effect (genuinely different
from the ``h_m -> 0`` limit at a realistic altitude). Direct nuSQuIDS
agreement is deferred to whenever the bindings are installed.

Module contents:
    tpeanuts_probability_atmosphere(...)
        Pure flavour eigenstate through the composed atmosphere+Earth
        path.
    compare_atmosphere(...)
        Run ``validation.nusquids.compare.grid.compare_grid`` between
        ``tpeanuts_probability_atmosphere`` and
        ``adapter.probability_atmosphere``, over an energy/cos(zenith)
        grid at a fixed production altitude.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch

from tpeanuts.medium.atmosphere.atmosphere import AtmosphereMedium
from tpeanuts.medium.earth.earth import EarthMedium
from tpeanuts.physics.common.state import DensityMatrix
from tpeanuts.physics.evolution import PropagationMethod
from tpeanuts.physics.model import PropagationModel
from tpeanuts.util.context import RuntimeContext
from validation.nusquids.adapter import initial_state, probability_atmosphere
from validation.nusquids.compare.grid import ComparisonResult, GridPoint, compare_grid, export_csv
from validation.nusquids.compare.oscillation_model import (
    atmosphere_density_kwargs,
    evolutor_from_profile,
    mixing_and_mass_spectrum,
)
from validation.nusquids.config import NuSQuIDSConfig

_LABELS = ("P_nue", "P_numu", "P_nutau")


def _composed_evolutor(
    *, E_GeV: float, cos_zenith: float, h_m: float, antinu: bool, config: NuSQuIDSConfig,
    context: RuntimeContext, nsteps: int,
):
    """Build the composed atmosphere(+Earth) evolutor and mixing, for one (E, cos_zenith, h_m) point."""
    eta_rad = math.acos(-float(cos_zenith))
    mixing, mass_spectrum = mixing_and_mass_spectrum(config, context)
    model = PropagationModel(mixing=mixing, mass_spectrum=mass_spectrum, antinu=antinu)

    atmosphere_profile = AtmosphereMedium(
        eta_rad=eta_rad, h_m=float(h_m), context=context, method=PropagationMethod.ANALYTIC, nsteps=nsteps,
        density_source="exponential", density_kwargs=atmosphere_density_kwargs(config),
    ).profile()
    S_atmosphere = evolutor_from_profile(atmosphere_profile, model, E_GeV, context, reverse=True)

    if cos_zenith >= 0.0:
        return S_atmosphere, mixing

    earth_profile = EarthMedium(
        eta_rad=eta_rad, context=context, method=PropagationMethod.ANALYTIC, nsteps=nsteps,
    ).profile()
    S_earth = evolutor_from_profile(earth_profile, model, E_GeV, context)
    return S_earth @ S_atmosphere, mixing


def tpeanuts_probability_atmosphere(
    *, E_GeV: float, cos_zenith: float, h_m: float, initial_flavour: str, antinu: bool = False,
    config: Optional[NuSQuIDSConfig] = None, context: Optional[RuntimeContext] = None, nsteps: int = 200,
) -> np.ndarray:
    """Propagate a pure initial flavour eigenstate through the composed atmosphere(+Earth) path.

    Same call signature as
    ``validation.nusquids.adapter.probability_atmosphere``, so both are
    valid ``validation.nusquids.compare.grid.ProbabilityFn``
    implementations for the same grid.

    Args:
        E_GeV: Neutrino energy in GeV.
        cos_zenith: Detector cos(zenith angle), in ``[-1, 1]``. Unlike
            ``compare/earth.py``, every value is valid here (see this
            module's own docstring).
        h_m: Production altitude above the surface, in metres; must be
            strictly positive (see ``AtmosphereMedium``'s own domain).
        initial_flavour: Initial flavour label ("nue", "numu",
            "nutau").
        antinu: If True, propagate the antineutrino state.
        config: Shared oscillation configuration (also supplies
            nuSQuIDS's own ``EarthAtm`` density constants). Defaults to
            ``NuSQuIDSConfig()`` when None.
        context: RuntimeContext. Defaults to CPU/float64 when None.
        nsteps: Chord discretization for both legs. 200 by default.

    Returns:
        1D float64 numpy array of length 3, summing to 1.
    """
    cfg = NuSQuIDSConfig() if config is None else config
    ctx = RuntimeContext.resolve("cpu", torch.float64) if context is None else context

    evolutor, mixing = _composed_evolutor(
        E_GeV=E_GeV, cos_zenith=cos_zenith, h_m=h_m, antinu=antinu, config=cfg, context=ctx, nsteps=nsteps,
    )
    flavour_idx = int(np.argmax(initial_state(initial_flavour, n_flavours=3)))
    initial = DensityMatrix.basis_state(flavour_idx, mixing.space, ctx)
    evolved = initial.conjugate_by(evolutor)

    return evolved.probabilities().detach().cpu().numpy().astype(np.float64).reshape(3)


def compare_atmosphere(
    energies_GeV: Sequence[float], cos_zeniths: Sequence[float], h_m: float,
    *, initial_flavours: Sequence[str] = ("nue", "numu", "nutau"), antinu: bool = False,
    config: Optional[NuSQuIDSConfig] = None, context: Optional[RuntimeContext] = None, nsteps: int = 200,
) -> Dict[str, ComparisonResult]:
    """Compare tpeanuts and nuSQuIDS atmosphere(+Earth) probabilities over an energy/cos(zenith) grid.

    Args:
        energies_GeV: Neutrino energies in GeV.
        cos_zeniths: Detector cos(zenith) values, each in ``[-1, 1]``.
        h_m: Fixed production altitude above the surface, in metres.
            nuSQuIDS's ``EarthAtm`` has a fixed atmosphere height and
            ignores this, so the comparison is only like-for-like at
            ``validation.nusquids.adapter.NUSQUIDS_EARTHATM_HEIGHT_M``.
        initial_flavours: Initial flavour labels.
        antinu: If True, compare the antineutrino channel.
        config: Shared oscillation configuration. Defaults to
            ``NuSQuIDSConfig()`` when None.
        context: tpeanuts RuntimeContext. Defaults to CPU/float64 when
            None.
        nsteps: See ``tpeanuts_probability_atmosphere``.

    Returns:
        Dict mapping each initial flavour label to its own
        ``ComparisonResult`` (nuSQuIDS as ``reference``, tpeanuts as
        ``candidate``).

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
            lambda E_GeV, cos_zenith, _flavour=flavour: probability_atmosphere(
                E_GeV=E_GeV, cos_zenith=cos_zenith, initial_flavour=_flavour, antinu=antinu, config=cfg,
            ),
            lambda E_GeV, cos_zenith, _flavour=flavour: tpeanuts_probability_atmosphere(
                E_GeV=E_GeV, cos_zenith=cos_zenith, h_m=h_m, initial_flavour=_flavour, antinu=antinu,
                config=cfg, context=context, nsteps=nsteps,
            ),
            labels=_LABELS,
        )
    return results


def export_all_csv(results: Dict[str, ComparisonResult], output_dir: Union[str, Path]) -> None:
    """Export each flavour's ``ComparisonResult`` to its own CSV file.

    Args:
        results: Mapping from initial flavour label to its
            ``ComparisonResult``, as returned by ``compare_atmosphere``.
        output_dir: Directory to write ``atmosphere_{flavour}.csv``
            files into (created if missing).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for flavour, result in results.items():
        export_csv(result, output_path / f"atmosphere_{flavour}.csv")

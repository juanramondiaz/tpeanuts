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

"""``reactor_flux``: the reactor scenario, from the source to the detector.

A reactor is a point source of electron antineutrinos a known distance from the detector, with nothing between them
but vacuum: the scenario is ``source.flux()`` followed by ``VacuumMedium(baseline).propagate`` (which, for an emission
rate, spreads the flux over the sphere the baseline reaches). It is written once here, as pure composition -- the
spectrum is the source's, the oscillation is the medium's and the model's -- so a notebook, a fit and an experiment
all reach the detector the same way::

    flux = reactor_flux(model, source, baseline_m)
    flux.state.probabilities()          # the survival/appearance probabilities at the detector
    detector.observe(flux)              # the event rate

The baseline batches like the energies of any medium: a scalar, or a *column* ``(n_L, 1)`` against the energy grid
``(n_E,)``, which gives a ``(n_L, n_E)`` flux (one row per baseline). A site with several cores gives each core its own
``relative_weight``/``snf_correction`` (see ``ReactorSource.flux``) and so calls this once per core. A flat
vector ``(n_L,)`` would be paired with the energy axis instead of crossed with it, so it is rejected with a message.

Module contents:
    reactor_flux(...)
        The reactor's antineutrino ``Flux`` at the detector, after a vacuum baseline.
"""

from __future__ import annotations

from typing import Any

from tpeanuts.interfaces.flux import Flux
from tpeanuts.medium.vacuum.vacuum import VacuumMedium
from tpeanuts.physics.model import PropagationModel
from tpeanuts.pipeline.chain import propagate_chain
from tpeanuts.source.reactor.reactor import COMPONENT, ReactorSource
from tpeanuts.util.tensor import TensorLike, as_tensor


def reactor_flux(
    model: PropagationModel,
    source: ReactorSource,
    baseline_m: TensorLike,
    *,
    dilute: bool = True,
    **source_options: Any,
) -> Flux:
    """Carry the reactor's antineutrinos over a vacuum baseline to the detector.

    ``source.flux(**source_options)["nue_bar"]`` followed by ``VacuumMedium(length_m=baseline_m).propagate``, through
    ``propagate_chain``.

    Args:
        model: Built PropagationModel; it must be the antineutrino one (``antinu=True``), as the source emits.
        source: The reactor.
        baseline_m: Source-to-detector distance, in metres: a scalar, or a column ``(n_L, 1)`` to cross several
            baselines with the energy grid.
        dilute: Spread the emission rate over ``4 pi L^2`` (``antinu s^-1 MeV^-1`` becomes
            ``antinu cm^-2 s^-1 MeV^-1``). It needs a source with a fission rate; pass False to keep the emission
            rate, or for a source without a spectrum (a unit-flux shape).
        **source_options: Forwarded to ``ReactorSource.flux``: ``energy_MeV``, ``curves``,
            ``fission_fractions``, ``fissions_per_second``, ``relative_weight``, ``snf_correction``.

    Returns:
        The antineutrino ``Flux`` at the detector (``location="detector"``, ``history=("vacuum",)``), whose state
        gives the probabilities and whose ``differential_flux`` (if the source has a spectrum) is diluted or not as
        asked. Its batch shape is that of ``baseline_m`` followed by the energy grid.

    Raises:
        ValueError: If ``baseline_m`` is a flat vector (not a scalar or a column), or from
            ``ReactorSource.flux`` / ``VacuumMedium.propagate`` (``model`` not an antineutrino model,
            ``dilute`` without an emission rate).

    Examples:
        >>> import math
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> space = tp.FlavourSpace()
        >>> params = tp.StandardMixingParameters(
        ...     theta12=math.radians(33.4), theta13=math.radians(8.6), theta23=math.radians(42.0),
        ...     delta13=math.radians(230.0), context=context)
        >>> mixing = tp.MixingMatrix(space=space, standard=params, nature=tp.Dirac(), context=context)
        >>> masses = tp.MassSpectrum(space=space, DeltamSq21=7.4e-5, DeltamSq3l=2.5e-3, context=context)
        >>> model = tp.PropagationModel(mixing=mixing, mass_spectrum=masses, antinu=True)   # antineutrinos
        >>> from tpeanuts.source import ReactorSource
        >>> source = ReactorSource.default(context=context)
        >>> arrived = tp.reactor_flux(model, source, 1650.0)   # a 1.65 km baseline, diluted over 4 pi L^2
        >>> arrived.state.probabilities().shape, arrived.flux_units
        (torch.Size([145, 3]), 'antinu cm^-2 s^-1 MeV^-1')
    """
    length = as_tensor(baseline_m, device=source.device, dtype=source.dtype)
    if length.ndim == 1:
        raise ValueError(
            f"baseline_m of shape {tuple(length.shape)} would be paired with the energy axis: pass a scalar, or a "
            f"column of shape ({length.shape[0]}, 1) to cross the baselines with the energies."
        )
    flux = source.flux(**source_options)[COMPONENT]
    medium = VacuumMedium(length_m=length, context=source.context)
    return propagate_chain(flux, [(medium, {"dilute": dilute})], model)

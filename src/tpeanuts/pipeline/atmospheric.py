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

"""``atmospheric_flux``: the atmospheric scenario, from production in the atmosphere to the detector.

A cosmic-ray shower produces neutrinos at some height; they cross the rest of the atmosphere to the surface and, for a
buried or upward-going detector, the Earth. The scenario is ``source.flux()`` followed by the atmosphere
(``AtmosphereLeg``) and the Earth (``EarthLeg``), one particle species at a time::

    arrived = atmospheric_flux(model, source, earth=EarthLeg(depth_m=2000.0))
    arrived["numu"].state.probabilities()      # (n_angle, n_E, n_flavour) at the detector
    detector.observe(arrived["numu"])          # the event rate, per direction

The source is passed already built (``AtmosphericSource.default`` reads the generated tables, and that
reading is the caller's decision). The production height is averaged out by the atmosphere step, weighted by the
source's production distribution, so the result has batch shape ``(n_angle, n_E)``. The nadir angles are the source's
own directions: an ``EarthLeg`` with its own ``eta_rad`` or an exposure table is rejected.

Each species is propagated with the model rebuilt for its own neutrino/antineutrino convention
(``dataclasses.replace(model, antinu=flux.antinu)``): a neutrino and an antineutrino cannot share one evolutor, so the
species axis is a loop over a ``SourceFlux``, not a tensor axis.

``ANALYTIC_PERTURBATIVE`` works in both legs, with float64: the perturbative solver loses accuracy in float32 at high
energy (errors of order 1 against float64 at 100 GeV, batched or not), so use ``NUMERICAL``/``ANALYTIC`` there.

Module contents:
    atmospheric_flux(...)
        ``source.flux()`` through the atmosphere and the Earth: a ``SourceFlux`` with one ``Flux`` per particle at the
        detector.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

from tpeanuts.interfaces.flux import SourceFlux
from tpeanuts.physics.model import PropagationModel
from tpeanuts.pipeline._legs import AtmosphereLeg, EarthLeg
from tpeanuts.pipeline.chain import propagate_chain
from tpeanuts.source.atmosphere.atmosphere import AtmosphericSource


def atmospheric_flux(
    model: PropagationModel,
    source: AtmosphericSource,
    particles: Optional[Sequence[str]] = None,
    *,
    atmosphere: Optional[AtmosphereLeg] = None,
    earth: Optional[EarthLeg] = None,
) -> SourceFlux:
    """Carry the atmospheric flux of every direction of ``source`` to a detector at any depth, per particle.

    ``source.flux(particles)`` followed, for each particle, by ``atmosphere.medium(...)`` and ``earth.propagate``
    through ``propagate_chain``, with ``model`` rebuilt for the particle's own ``antinu``.

    Args:
        model: Built PropagationModel; its own ``antinu`` is overridden per particle.
        source: The atmospheric source (its direction, energy and production-height grids).
        particles: Particle labels to carry (keys of the source's flux table); all of them if omitted.
        atmosphere: Method and discretisation of the atmosphere leg; defaults to ``AtmosphereLeg()``.
        earth: The Earth leg: detector depth, method and discretisation; defaults to ``EarthLeg()`` (a surface
            detector). It must have neither ``eta_rad`` nor ``exposure``.

    Returns:
        SourceFlux with one component per particle, ``location="detector"`` and ``history=("atmosphere", "earth")``,
        whose ``direction`` is ``source.eta_rad``, whose ``differential_flux`` is shaped ``(n_angle, n_E)`` and whose
        state has batch shape ``(n_angle, n_E)``.

    Raises:
        ValueError: If ``earth`` carries its own ``eta_rad`` or an exposure table, or from the media (an angle outside
            its domain at this depth, model and flux disagreeing), or for a particle the source does not have.

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
        >>> model = tp.PropagationModel(mixing=mixing, mass_spectrum=masses)
        >>> import tempfile
        >>> from pathlib import Path
        >>> from tpeanuts.pipeline import AtmosphereLeg, EarthLeg
        >>> from tpeanuts.source import AtmosphericSource
        >>> from tpeanuts.source.atmosphere.io import angle_output_path, write_angle_flux_csv
        >>> from tpeanuts.source.atmosphere.atmosphere import AtmosphericSourceParameters
        >>> eta_rad = 0.6   # nadir angle at the detector, radians
        >>> with tempfile.TemporaryDirectory() as directory:   # normally the tables come from generate_mceq_flux
        ...     write_angle_flux_csv(
        ...         angle_output_path(Path(directory), math.degrees(eta_rad)),
        ...         energy_GeV=torch.tensor([1.0, 2.0], dtype=torch.float64),
        ...         altitude_km=torch.tensor([10.0, 15.0], dtype=torch.float64),
        ...         per_particle_flux={"numu": torch.ones(2, 2, dtype=torch.float64)},
        ...         cos_zenith=math.cos(eta_rad))
        ...     source = AtmosphericSource.default(
        ...         params=AtmosphericSourceParameters(
        ...             theta_deg=torch.tensor(math.degrees(eta_rad), dtype=torch.float64), provider="mceq",
        ...             output_dir=Path(directory)),
        ...         context=context)
        >>> arrived = tp.atmospheric_flux(
        ...     model, source, atmosphere=AtmosphereLeg(nsteps=20), earth=EarthLeg(nsteps=20))
        >>> arrived["numu"].state.batch_shape   # (angle, energy): the production height is averaged out
        torch.Size([1, 2])
    """
    atmosphere = atmosphere or AtmosphereLeg()
    earth = earth or EarthLeg()
    if earth.eta_rad is not None or earth.exposure is not None:
        raise ValueError(
            "the nadir angles of an atmospheric flux are the source's own directions: pass an EarthLeg without "
            "eta_rad or exposure."
        )

    def carry(flux):
        particle_model = dataclasses.replace(model, antinu=flux.antinu)
        steps = [lambda current: atmosphere.medium(current, earth.depth_m), earth]
        return propagate_chain(flux, steps, particle_model)

    return source.flux(particles).map(carry)

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

"""The scenario pipelines: from a neutrino source to the detector, one function per kind of source.

Each function is pure composition, ``source.flux()`` followed by the media the neutrinos cross, and returns the flux in
the state it reaches the detector (``flux.state.probabilities()`` for the probabilities, ``detector.observe(flux)`` for
the rate)::

    arrived = solar_flux(model, source, sun, ["8B"], energies, earth=EarthLeg(eta_rad=0.5, depth_m=2070.0))
    arrived["8B"].state.probabilities()

The options of each leg travel in small immutable dataclasses (``SolarLeg``, ``EarthLeg``, ``AtmosphereLeg``), so the
choices for one leg never mix with another's keyword arguments. Unlike ``tpeanuts.medium``, which never depends on
``tpeanuts.physics.model.PropagationModel``, this package is precisely the layer allowed to depend on everything
(``medium``, ``physics``, ``interfaces``): the one thing it does that no medium can is average over a detector's
nadir-angle exposure, which needs the model.

Module contents:
    ``tpeanuts.pipeline.solar``
        ``solar_flux(model, source, sun, sources, energy)``: the chosen solar channels through the Sun and, with an
        ``EarthLeg``, the Earth (one nadir angle, or averaged over an exposure table).
    ``tpeanuts.pipeline.atmospheric``
        ``atmospheric_flux(model, source)``: every direction of an atmospheric source through the atmosphere and the
        Earth, one ``Flux`` per particle, with the production height averaged out.
    ``tpeanuts.pipeline.reactor``
        ``reactor_flux(model, source, baseline_m)``: a reactor's antineutrinos over a vacuum baseline, diluted by
        default.
    ``tpeanuts.pipeline._legs``
        ``SolarLeg``, ``EarthLeg`` and ``AtmosphereLeg``, and the exposure average they share.
    ``tpeanuts.pipeline.chain``
        ``propagate_chain(flux, steps, model)``: apply several media to one flux in order, each with its own options.
"""

from __future__ import annotations

from tpeanuts.util.lazy import lazy_exports

_EXPORTS = {
    "solar_flux": "tpeanuts.pipeline.solar",
    "atmospheric_flux": "tpeanuts.pipeline.atmospheric",
    "reactor_flux": "tpeanuts.pipeline.reactor",
    "SolarLeg": "tpeanuts.pipeline._legs",
    "EarthLeg": "tpeanuts.pipeline._legs",
    "AtmosphereLeg": "tpeanuts.pipeline._legs",
}
__all__ = list(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)

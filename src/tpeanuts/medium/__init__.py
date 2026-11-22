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

"""Concrete media: whatever a neutrino travels through between source and detector.

Each medium here builds a ``tpeanuts.interfaces.medium.MediumProfile`` --
that is the only thing that crosses this package's boundary.
``tpeanuts.medium`` never imports ``tpeanuts.source`` directly, nor
``tpeanuts.physics.model.PropagationModel`` (the physics composition
root -- see ``tpeanuts.medium.earth.exposure``'s own docstring for why),
and knows nothing about propagation itself (that is
``tpeanuts.physics``'s job); ``tpeanuts.pipeline`` is the layer allowed
to depend on everything, composing a medium's own ``MediumProfile`` with
a built ``PropagationModel``.

Three concrete media exist: ``tpeanuts.medium.vacuum.vacuum.VacuumMedium``
(a single zero-density segment), ``tpeanuts.medium.earth.earth.EarthMedium``
(a real PREM density chord, all three ``PropagationMethod`` values, a
detector at any depth, plus nadir-angle exposure-weighted time-averaging
-- see that subpackage's own module docstring), and
``tpeanuts.medium.atmosphere.atmosphere.AtmosphereMedium`` (the
production-to-detector atmosphere-only leg, same three methods and any
detector depth -- see that subpackage's own module docstring for why it
never builds an underground segment itself). ``tpeanuts.medium.common``
holds geometry genuinely shared by more than one medium (today: the
spherical-shell chord length both Earth and atmosphere are built from).
Still no shared base class or protocol across media -- nothing in this
project yet needs to accept "any medium" generically, so designing one
now would still be guessing at a surface no concrete caller has asked
for, the same reasoning already applied to
``tpeanuts.physics.common.terms.GeneratorTerm`` before a second
extension existed.
"""

from __future__ import annotations

from tpeanuts.util.lazy import lazy_exports

_EXPORTS = {
    "EarthMedium": "tpeanuts.medium.earth.earth",
    "VacuumMedium": "tpeanuts.medium.vacuum.vacuum",
    "AtmosphereMedium": "tpeanuts.medium.atmosphere.atmosphere",
    "SolarMedium": "tpeanuts.medium.solar.solar",
    "NadirExposureTable": "tpeanuts.medium.earth.exposure.table",
    "build_nadir_exposure": "tpeanuts.medium.earth.exposure.table",
}
__all__ = list(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)

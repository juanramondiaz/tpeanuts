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

"""A neutrino's path from its atmospheric production point down to the Earth's surface.

Deliberately still without: ``ANALYTIC_PERTURBATIVE`` (needing a
per-segment polynomial density fit, this medium's own analogue of
``tpeanuts.medium.earth.even_power``) and the optional heavy-dependency
density backends (MCEq, pymsis) -- both later, separate rounds.
``NUMERICAL``/``ANALYTIC`` (closed-form/tabulated density,
``"exponential"``/``"file"``) are both supported, identically sampled
(see ``tpeanuts.medium.atmosphere.atmosphere``'s own module docstring).

Mirrors ``tpeanuts.medium.earth`` in scope and in what it deliberately
does **not** do: this medium only ever propagates production -> surface,
exactly like v1_0's own ``medium.atmosphere`` -- it has no notion of a
detector or of a detector's own depth. Continuing a surface state through
the Earth to a buried detector is ``tpeanuts.pipeline.
atmospheric_neutrino``'s job, composing this medium's own evolutor with
``tpeanuts.medium.earth.earth.EarthMedium``/``tpeanuts.pipeline.earth``,
not something this package does itself.

Module contents:
    ``tpeanuts.medium.atmosphere.geometry``
        The straight-line path from the Earth's surface up through the
        atmosphere to a production altitude, and altitude sampling along
        it.
    ``tpeanuts.medium.atmosphere.density``
        Mass/electron/neutron density along that path
        (``"exponential"``/``"file"`` backends).
    ``tpeanuts.medium.atmosphere.atmosphere``
        ``AtmosphereMedium``, the ``MediumProfile`` producer.
"""

from __future__ import annotations

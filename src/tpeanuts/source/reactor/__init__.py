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

"""Reactor-neutrino production: the antineutrino spectrum a reactor emits, no propagation physics.

Mirrors ``tpeanuts.source.solar`` and ``tpeanuts.source.atmosphere`` (one package per neutrino source, the domain
model in ``model.py``, the data loaders in ``io.py``): the Huber-Mueller isotope spectra, the fission-fraction
weighting and ``ReactorSource``, whose ``flux()`` is a ``tpeanuts.interfaces.flux.SourceFlux`` with one
component, ``"nue_bar"``, of pure electron antineutrinos. A reactor is a point source with a single, known flavour
composition, so unlike the Sun or the atmosphere it carries no ``production`` distribution.

Module contents:
    ``tpeanuts.source.reactor.reactor``
        ``ReactorSource`` (``.default()``, ``.flux()``, the spectrum primitives), ``ReactorSourceParameters``,
        ``build_reactor_source``, ``ISOTOPES``, ``COMPONENT``.
    ``tpeanuts.source.reactor.io``
        Loaders of the packaged tables: ``load_reactor_parameters``, ``load_huber_mueller_spectra``,
        ``load_nonequilibrium_correction``.
"""

from __future__ import annotations

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

"""Concrete experiments: a source, the media in between, a detector and the data, wired together.

Each subpackage composes everything one experiment needs -- its own parameters and tabulated inputs (``parameters``,
``io``), the flux it sees, the propagation it applies (through ``tpeanuts.pipeline``), the response of its detector
(built from the generic blocks in ``tpeanuts.detector``) and the model that confronts it with data
(``inference_model``). They sit above ``tpeanuts.pipeline`` and below ``tpeanuts.inference`` in the import layers
(see ``tools/check_layers.py``): ``tpeanuts.source``, ``tpeanuts.medium`` and ``tpeanuts.detector`` never import each
other, and this is where they are brought together.

``tpeanuts.experiments.dayabay`` (the Daya Bay reactor experiment), ``tpeanuts.experiments.borexino`` (elastic
nu-e scattering in a liquid-scintillator target), ``tpeanuts.experiments.sno`` (SNO Phase I: CC + ES + NC on a
heavy-water target, with day/night Earth-regeneration averaging), ``tpeanuts.experiments.sno_ii`` (SNO Phase II, the
salt phase, predicting the 38 officially published pseudo-observables against a correlated-Gaussian covariance) and
``tpeanuts.experiments.icecube`` (the IceCube DeepCore atmospheric-neutrino experiment: event-by-event weighted Monte
Carlo reweighted by a full Earth flavour-transition matrix).
"""

from __future__ import annotations

__all__: list = []  # experimental in 1.0: no stability guarantee, import from the submodules

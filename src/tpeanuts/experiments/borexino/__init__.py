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

"""The Borexino solar-neutrino experiment: liquid-scintillator elastic scattering.

Ported from v1_0's ``experiments/borexino/`` package (see
[[project_tpeanuts_v2_rewrite]] for the round-by-round porting log),
built entirely on ``tpeanuts.detector.common`` (the generic response/
efficiency/event-rate/target machinery) and
``tpeanuts.detector.neutrino_electron``'s tree-level nu-e elastic
cross section -- this package supplies only what is specific to
Borexino itself: its pseudocumene target composition, its illustrative
sqrt(T) energy resolution, and its real Nature 2018 low-energy rate
spectrum. Unlike Daya Bay, there is no separate background module
here: v1_0's own ``backgrounds.py`` was always a zero placeholder, so
``background_counts=None`` (already the ``detector.common.event_rate``
convention) stands in for it directly (same round-1 decision as the
IBD detector).

Module contents:
    ``tpeanuts.experiments.borexino.io``
        Loader for the cached Nature 2018 observation CSV.
    ``tpeanuts.experiments.borexino.parameters``
        ``BorexinoParameters``, the loaded-once bundle every other
        module here reads from.
    ``tpeanuts.experiments.borexino.response``
        Illustrative Gaussian energy-response chain.
    ``tpeanuts.experiments.borexino.event_rate``
        Oscillated flux -> predicted counts per bin.
    ``tpeanuts.experiments.borexino.inference_model``
        ``BorexinoModel``, the differentiable oscillation x
        detector composition.
"""

from __future__ import annotations

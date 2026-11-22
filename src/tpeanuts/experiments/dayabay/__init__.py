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

"""The Daya Bay reactor antineutrino experiment: 6 reactor cores, 8 detectors, IBD.

Ported from v1_0's ``experiments/dayabay/`` package (see
[[project_tpeanuts_v2_rewrite]] for the round-by-round porting log),
built entirely on ``tpeanuts.detector.common`` (the generic response/
efficiency/event-rate/target machinery) and
``tpeanuts.detector.ibd_cross_section``'s order-1/M IBD cross section --
this package supplies only what is specific to Daya Bay itself: its own
geometry, tabulated data release, and detector-response corrections
(IAV/LSNL/energy resolution).

Module contents:
    ``tpeanuts.experiments.dayabay.io``
        Loaders for the cached Daya Bay data-release CSVs.
    ``tpeanuts.experiments.dayabay.parameters``
        ``DayaBayParameters``, the dense loaded-once bundle every other
        module here reads from.
"""

from __future__ import annotations

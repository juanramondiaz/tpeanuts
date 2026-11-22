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

"""The SNO (Sudbury Neutrino Observatory) solar-neutrino experiment: CC + ES + NC, day/night.

Ported from v1_0's ``experiments/sno/`` package (see
[[project_tpeanuts_v2_rewrite]] for the round-by-round porting log).
SNO measures solar neutrinos through three physically distinct
reactions on the same heavy-water (D2O) target -- CC (nu_e + d -> p +
p + e-), ES (nu + e- -> nu + e-, all flavours) and NC (nu_x + d -> p +
n + nu_x, flavour-blind, detected via the neutron-capture gamma) --
built on ``tpeanuts.detector.common`` (generic response/efficiency/
event-rate/target machinery), ``tpeanuts.detector.deuteron`` (CC/NC
cross sections) and ``tpeanuts.detector.neutrino_electron`` (ES cross
section), the same reusable pieces Daya Bay and Borexino are built
from.

Unlike Borexino (a single ES channel, no Earth crossing),
``tpeanuts.experiments.sno.inference_model.SNOModel`` also
averages over real day/night nadir-angle exposure, via
``tpeanuts.pipeline.solar.solar_flux`` and ``tpeanuts.pipeline._legs.EarthLeg``
(an exposure table) -- the first concrete detector in this rewrite to consume that pipeline.

Module contents:
    ``tpeanuts.experiments.sno.io``
        Loaders for the real SNO Phase-I day/night observation CSVs.
    ``tpeanuts.experiments.sno.parameters``
        ``SNOParameters``, the loaded-once bundle every other module
        here reads from.
    ``tpeanuts.experiments.sno.response``
        SNO's real Phase-I energy-response chain.
    ``tpeanuts.experiments.sno.event_rate``
        The three independent CC/ES/NC event-rate functions.
    ``tpeanuts.experiments.sno.inference_model``
        ``SNOModel``, the differentiable oscillation x Sun x
        Earth x detector composition.
"""

from __future__ import annotations

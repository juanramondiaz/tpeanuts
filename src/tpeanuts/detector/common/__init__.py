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

"""Detector-agnostic building blocks: response, efficiency, event-rate folding, observations, targets.

Every piece here is reusable by any concrete detector
(``tpeanuts.detector.ibd.IBDDetector`` today, a future Daya Bay/Borexino/
SNO-style detector next), with no experiment-specific physics of its own --
a concrete detector supplies its own response matrix, efficiency curve,
target composition, and published ``Observation``s, then composes them via
``event_rate``'s own folding chain.

Module contents:
    ``tpeanuts.detector.common.response``
        Gaussian energy-response matrices and mass-conserving grid
        redistribution.
    ``tpeanuts.detector.common.efficiency``
        Detection-efficiency curves (multiplicative, or a hard threshold).
    ``tpeanuts.detector.common.event_rate``
        The generic flux -> true spectrum -> response -> efficiency ->
        binned-counts folding chain.
    ``tpeanuts.detector.common.observation``
        ``Observation``, a published measurement's own data container.
    ``tpeanuts.detector.common.target``
        Target particle (electron/nucleus) counts from molecular
        composition and mass.
"""

from __future__ import annotations

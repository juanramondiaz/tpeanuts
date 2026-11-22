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

"""tpeanuts: a modular, torch-based neutrino propagation engine.

Supports the Standard Model, the 3+N sterile-neutrino extension, Non-Standard
Interactions (NSI), Dirac and Majorana neutrinos, magnetic-moment
(spin-flavor precession) interactions, and decoherence, all built on a shared
density-matrix state representation so that adding a new extension never
requires forking the propagation engine.

This is a ground-up rewrite; the previous implementation is preserved,
unmodified, under ``v1_0/`` at the repository root.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tpeanuts")  # single source of truth: the version in pyproject.toml
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"

from tpeanuts.util.lazy import lazy_exports

_EXPORTS = {
    "RuntimeContext": "tpeanuts.util.context",
    "FlavourSpace": "tpeanuts.physics.common.space",
    "StandardMixingParameters": "tpeanuts.physics.common.mixing",
    "MixingMatrix": "tpeanuts.physics.common.mixing",
    "MassSpectrum": "tpeanuts.physics.common.mass_spectrum",
    "Dirac": "tpeanuts.physics.common.nature",
    "Majorana": "tpeanuts.physics.common.nature",
    "PropagationModel": "tpeanuts.physics.model",
    "PropagationMethod": "tpeanuts.physics.evolution.method",
    "EarthMedium": "tpeanuts.medium.earth.earth",
    "SolarMedium": "tpeanuts.medium.solar.solar",
    "SolarSource": "tpeanuts.source.solar.solar",
    "solar_flux": "tpeanuts.pipeline.solar",
    "atmospheric_flux": "tpeanuts.pipeline.atmospheric",
    "reactor_flux": "tpeanuts.pipeline.reactor",
}
__all__ = list(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)

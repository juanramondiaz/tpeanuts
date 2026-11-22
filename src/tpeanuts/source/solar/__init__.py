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

"""Solar-neutrino production: where the Sun's neutrinos come from, no propagation physics.

Phase 1 of the solar migration (see ``tpeanuts.data.solar``'s own
``README.md`` for exactly which v1_0 tables were ported): radial
production distributions, total per-source fluxes, and production-energy
spectra, across three independent providers (zenodo, bahcall, legacy).
Nothing here depends on a future ``tpeanuts.medium.solar`` (the solar
propagation medium, not ported yet), and nothing in that future
package's own propagation path will depend on this one -- the only thing
that will ever join them is a future ``tpeanuts.pipeline.solar``, exactly
mirroring how ``tpeanuts.source.atmosphere``/``tpeanuts.medium.atmosphere``
stay decoupled until composed by ``tpeanuts.pipeline.atmosphere``.

Module contents:
    ``tpeanuts.source.solar.io``
        Provider-neutral CSV loaders: ``load_solar_production``,
        ``load_solar_flux``, ``load_spectrum_csv``,
        ``load_solar_spectrum``, ``available_solar_spectrum_sources``,
        ``solar_source_provider_path``, ``solar_spectrum_path``.
    ``tpeanuts.source.solar.solar``
        ``SolarSource`` (the torch-native production source),
        ``SolarContinuousSpectrum``/``SolarLineSpectrum``/``SolarSpectrum``,
        ``SolarSourceParameters``, ``build_solar_source``.
"""

from __future__ import annotations

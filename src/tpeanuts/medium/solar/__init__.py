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

"""The Sun as a propagation medium: ``SolarMedium`` and its interior density table.

``tpeanuts.medium.solar.solar`` holds ``SolarMedium`` -- the counterpart of ``EarthMedium``, ``AtmosphereMedium`` and
``VacuumMedium`` -- which validates and interpolates the radial density table, computes the mass-basis weights of a
neutrino born at each radius (``mass_weights``) and propagates a ``Flux`` from its production inside the Sun to the
solar surface (``propagate``). ``tpeanuts.medium.solar.density`` is the lookup table and its interpolation, the
analogue of ``tpeanuts.medium.earth.density``. See ``tpeanuts.data.solar``'s own ``README.md`` for which v1_0 tables
were ported.

The closed-form/diagonalised adiabatic matter-mixing physics (``matter_mixing``, ``adiabatic``, ``landau_zener``)
lives in ``tpeanuts.physics.evolution.adiabatic``, so that ``tpeanuts.medium`` holds the medium and its data while
every propagation algorithm, whatever the medium, lives under ``tpeanuts.physics.evolution``.

Module contents:
    ``tpeanuts.medium.solar.density``
        ``SolarDensityTable`` and its loader/interpolator,
        ``load_solar_density_table``/``interpolate_solar_density``.
    ``tpeanuts.medium.solar.solar``
        ``SolarMedium`` (the Sun as a medium: validated density profile,
        ``mass_weights``, ``propagate``), ``SolarMediumParameters``,
        ``build_solar_medium``.
"""

from __future__ import annotations

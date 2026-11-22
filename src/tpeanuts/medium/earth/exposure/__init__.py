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

"""Nadir-angle exposure (time-averaging): how long a detector sees each eta_rad.

A detector at fixed latitude sees its nadir angle ``eta_rad`` change
continuously as the Earth rotates daily and orbits annually -- so a
physical observable is not a probability at one ``eta_rad``, but a
*time-averaged* one: ``P_exp(E) = integral(d eta_rad * W(eta_rad) * P(E, eta_rad))``,
where ``W(eta_rad)`` is the fraction of the requested day-of-year window
during which the detector observes nadir angle ``eta_rad``. Ported from
v1_0's own ``medium/earth/exposure_math.py``/``exposure_table.py``
(``exposure_integration.py``'s ``earth_probability_exposure`` covered the
same physics, but coupled to v1_0's own ``probability.py`` end-to-end --
see below for why this package does not reproduce that coupling).

Deliberately **decoupled** from ``tpeanuts.physics.model.PropagationModel``:
this package only builds the exposure table ``W(eta_rad)`` and reduces an
already-computed probability tensor against it -- it knows nothing about
``EarthMedium``, generators, or evolutors. The caller loops over a nadir-
angle grid, computes ``P(eta_rad)`` however its own scenario requires (any of
the three ``PropagationMethod`` values, any detector depth -- exactly the
``EarthMedium(...).profile()`` -> ``PropagationEnvironment(...)`` ->
``tpeanuts.physics.evolution.dispatch.evolve_path(...)`` wiring already shown in
``tpeanuts.test.test_earth_benchmark``), and passes the resulting
``(n_eta, ...)`` tensor to ``table.integrate_exposure``. This keeps
``medium/`` from depending on ``physics.model`` (the composition root),
matching the one-directional dependency ``medium/`` already has on the
lower-level ``physics.evolution`` alone.

MVP scope, deliberately: the "math" exposure source only (``W(eta_rad)``
computed on the fly from a detector latitude via closed-form elliptic
integrals, no data file) -- CSV-tabulated and disk-cached sources exist in
v1_0 (``exposure_io.py``) but are left out until a concrete detector needs
one; ``build_nadir_exposure``'s own signature leaves room to add them
later without breaking anything already built here.

Module contents:
    ``tpeanuts.medium.earth.exposure.time_average``
        The elliptic-integral nadir-exposure geometry itself:
        ``indefinite_integral_day``/``integral_angle``/``integral_day``/
        ``make_eta_grid``.
    ``tpeanuts.medium.earth.exposure.table``
        ``NadirExposureTable``, ``build_nadir_exposure`` (the "math"
        source, looped over a full eta_rad grid), and the physics-agnostic
        ``integrate_exposure`` reduction.
"""

from __future__ import annotations

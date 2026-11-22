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

"""Both variants of the closed-form (Cardano/Ferrari) unitary evolver: exact zero-order, and first-order corrected.

Holds ``PropagationMethod.ANALYTIC`` and ``PropagationMethod.ANALYTIC_PERTURBATIVE``
together, not split into sibling folders: they are two variants of the
same closed-form technique, not independent propagation methods --
``ANALYTIC_PERTURBATIVE`` is defined directly in terms of ``ANALYTIC``'s
own exact zero-order result plus an optional first-order correction, and
reduces to it exactly when there is no density variation to correct for
(see ``perturbative.perturbative``'s own module docstring). Originally
``analytic``/``spectral`` lived in their own folder/flat module
respectively; regrouped here once that relationship was made explicit, so
that everything specific to this one technique -- exact and corrected
alike -- lives in one place.

Module contents:
    ``tpeanuts.physics.evolution.perturbative.analytic``
        ``segment_evolutor_analytic``/``evolve_path_analytic`` -- the
        exact, closed-form zero-order evolver ``PropagationMethod.ANALYTIC``
        selects, and the base this module's first-order correction builds
        on top of.
    ``tpeanuts.physics.evolution.perturbative.perturbative``
        ``segment_evolutor_analytic_perturbative``/
        ``evolve_path_analytic_perturbative`` -- the first-order
        correction on top of ``analytic``'s exact zero-order result.
    ``tpeanuts.physics.evolution.perturbative.profile``
        ``EvenPowerProfile``/``PolynomialProfile`` -- the within-segment
        density-profile models ``perturbative`` consumes.
    ``tpeanuts.physics.evolution.perturbative.spectral``
        Closed-form Cardano/Ferrari spectral decomposition (eigenvalues +
        projectors) of a traceless Hermitian Hamiltonian -- the shared
        foundation both ``analytic`` and ``perturbative`` build on, used
        by no other strategy in ``tpeanuts.physics.evolution``.
"""

from __future__ import annotations

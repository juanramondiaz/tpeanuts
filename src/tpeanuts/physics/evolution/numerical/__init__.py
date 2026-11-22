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

"""The ``PropagationMethod.NUMERICAL`` evolver: matrix-exponential, exact, unconditional.

Given its own folder to match every other propagation strategy
(``analytic``/``perturbative``, ``liouville``, ``adiabatic``), even though
it holds a single module -- unlike the shared infrastructure that was
deliberately kept flat (nothing else in ``tpeanuts.physics.evolution``
is infrastructure ``numerical`` itself depends on), this module *is* a
propagation strategy in its own right, the one every other unitary
strategy either specializes (``perturbative.analytic``/``.perturbative``,
restricted to ``dim in {3, 4}``) or falls back to (the sole option outside
that domain, and the only one with any Liouville-space counterpart at
all -- see ``tpeanuts.physics.evolution``'s own module docstring).

``shared_history`` also lives here, not under ``adiabatic/``: despite the
name similarity to the solar-neutrino physics that motivated it, it is
fully coherent/unitary propagation (the same ``matrix_exp`` machinery as
``numerical.numerical``) optimised for many production points sharing
one destination -- not the genuinely incoherent adiabatic approximation
``physics.evolution.adiabatic`` implements. Grouping it here instead
keeps every module in this package honestly coherent-and-unitary.

Module contents:
    ``tpeanuts.physics.evolution.numerical.numerical``
        ``segment_evolutor_numerical``/``evolve_path_numerical`` --
        ``torch.linalg.matrix_exp``-based, exact for any generator size,
        one path.
    ``tpeanuts.physics.evolution.numerical.shared_history``
        Numerical propagation from many origin points to one shared
        endpoint without re-propagating per origin -- the technique
        ``tpeanuts.pipeline.solar``'s own ``"numerical"`` method is built
        on.
"""

from __future__ import annotations

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

"""Adiabatic propagation: pointwise mass-eigenstate weights, not a path evolutor.

Structurally different from ``numerical``/``analytic``/``perturbative``
(which all build a coherent evolution *operator* for a path) and from
``liouville`` (a Liouville-space operator for a path): the adiabatic
family instead evaluates, at a single production point, the
instantaneous matter-mixing eigenbasis and returns *weights* (an
incoherent mass-eigenstate mixture) -- physically correct because
coherence between mass eigenstates is lost over astronomical
propagation distances, so there is no meaningful path evolutor to build
in the first place. Not a ``PropagationMethod`` value for the same
reason ``liouville`` is not one (see
``tpeanuts.physics.evolution``'s own module docstring): it answers a
structurally different question, reached through its own model-aware
entry point, ``tpeanuts.physics.evolution.adiabatic.adiabatic.mass_weights_adiabatic``,
not ``dispatch.evolve_path``.

Moved here from ``tpeanuts.medium.solar``/``tpeanuts.pipeline.solar``
(where this physics originally lived during the solar-neutrino
migration): none of it is solar-specific -- ``tpeanuts.medium.solar``
now holds only density/geometry data (``density.py``/``solar.py``,
mirroring ``tpeanuts.medium.earth``/``tpeanuts.medium.atmosphere``), and
``tpeanuts.pipeline.solar`` now only composes a source, a medium, and a
chosen propagation method, mirroring ``tpeanuts.pipeline.earth``/
``atmosphere``. Every actual algorithm lives here instead, alongside
every other propagation method.

Everything in this package is, without exception, about the genuinely
*incoherent* adiabatic approximation -- unlike the numerical shared
-origin technique the solar ``"numerical"`` method also uses (many
production points propagated to one shared endpoint), which is fully
coherent/unitary and therefore lives in
``tpeanuts.physics.evolution.numerical.shared_history`` instead, not
here.

Module contents:
    ``tpeanuts.physics.evolution.adiabatic.matter_mixing``
        Closed-form MSW matter-mixing angles: ``Vk``, ``DeltamSqee``,
        ``theta13_in_matter``, ``theta12_in_matter``.
    ``tpeanuts.physics.evolution.adiabatic.adiabatic``
        ``mass_weights_adiabatic_approximated`` (closed-form, plain SM
        only) and ``mass_weights_adiabatic_exact`` (pointwise
        diagonalisation, NSI/sterile-capable) -- pure functions, plus
        ``mass_weights_adiabatic``, the model-aware entry point every caller
        outside ``tpeanuts.physics.evolution`` itself should use.
    ``tpeanuts.physics.evolution.adiabatic.landau_zener``
        ``density_gradient``, ``resonance_radius``, ``landau_zener_probability``,
        ``landau_zener_spatial_correction`` -- the local 1-2 resonance
        crossing correction ``adiabatic``'s ``p_lz`` argument expects.
"""

from __future__ import annotations

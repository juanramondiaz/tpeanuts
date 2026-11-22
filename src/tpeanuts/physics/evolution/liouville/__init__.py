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

"""Liouville-space (open-system/decoherence) evolution -- structurally separate from the unitary methods.

Not a ``PropagationMethod`` value (see
``tpeanuts.physics.evolution``'s own module docstring for why): active
whenever a ``tpeanuts.physics.model.PropagationModel`` carries any
``dissipators``, orthogonal to which unitary method would otherwise
apply.

Module contents:
    ``tpeanuts.physics.evolution.liouville.open_system``
        ``commutator_superoperator``/``dissipator_superoperator``,
        ``DissipatorTerm`` (Protocol), ``segment_evolutor_liouville``/
        ``evolve_path_liouville``.
"""

from __future__ import annotations

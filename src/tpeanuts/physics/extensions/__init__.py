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

"""Every optional, off-by-default physics ingredient, in one flat registry.

Unlike ``tpeanuts.physics.common`` (needed unconditionally by every
scenario), nothing here is ever required: a propagation model with no
extension active is the plain Standard Model. There is no "SM" vs "BSM"
split -- each module here is a self-contained, independently optional piece.
Most (the sterile neutral-current term, NSI, a magnetic moment) satisfy
``tpeanuts.physics.common.terms.GeneratorTerm``: a
``contribute_reduced(mixing, ...)`` method returning a reduced-basis
*unitary* generator contribution, summed by
``tpeanuts.physics.model.PropagationModel`` alongside its own base kinetic
and charged-current terms before a single final rotation into the flavour
basis. ``decoherence`` is structurally different -- it satisfies
``tpeanuts.physics.evolution.liouville.open_system.DissipatorTerm`` instead
(``lindblad_operators(mixing, ...)``, no Hamiltonian-like contribution at
all), since genuine dissipation has no ``GeneratorTerm``-style additive
Hamiltonian piece to contribute in the first place.
"""

from __future__ import annotations

from tpeanuts.util.lazy import lazy_exports

_EXPORTS = {
    "NSIConfig": "tpeanuts.physics.extensions.nsi",
    "NSITerm": "tpeanuts.physics.extensions.nsi",
    "SterileNCTerm": "tpeanuts.physics.extensions.sterile",
    "MagneticMomentConfig": "tpeanuts.physics.extensions.magnetic_moment",
    "MagneticMomentTerm": "tpeanuts.physics.extensions.magnetic_moment",
    "DecoherenceConfig": "tpeanuts.physics.extensions.decoherence",
    "DecoherenceTerm": "tpeanuts.physics.extensions.decoherence",
}
__all__ = list(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)

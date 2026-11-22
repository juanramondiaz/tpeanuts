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

"""Generic, non-physics helpers shared across every tpeanuts subpackage.

Nothing under ``tpeanuts.util`` knows about neutrinos, flavours, or
Hamiltonians -- that model-agnostic-but-physics-aware layer lives in
``tpeanuts.physics.common`` instead. Anything reusable by more than one of
``source``/``medium``/``detector``/``physics`` that is *not* physics belongs
here, and only here, so it is never duplicated.
"""

from __future__ import annotations

from tpeanuts.util.lazy import lazy_exports

_EXPORTS = {
    "RuntimeContext": "tpeanuts.util.context",
    "TPeanutsDeprecationWarning": "tpeanuts.util.deprecation",
    "deprecated": "tpeanuts.util.deprecation",
}
__all__ = list(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)

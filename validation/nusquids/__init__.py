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

"""Independent nuSQuIDS cross-validation and benchmarking harness.

**Deliberately outside ``src/tpeanuts/``.** nuSQuIDS is an independent
external C++ (with Python bindings) neutrino-oscillation solver, used
here purely to cross-check tpeanuts's own torch propagation code and
to compare its runtime performance -- never as a dependency of any
tpeanuts computation. This package enforces that independence
structurally, not just by convention:

    ``validation.nusquids.config``/``validation.nusquids.adapter``
        Talk to the optional nuSQuIDS Python bindings. Import nothing
        from ``tpeanuts`` (confirmed by
        ``validation.nusquids.test.test_independence``) -- any small
        piece of physics they need (flavour-label normalization, the
        vacuum mass -> flavour projector) is reimplemented locally in
        plain numpy, not borrowed from ``tpeanuts``'s own PMNS code.
        This is not just decoupling for its own sake: reusing
        tpeanuts's own PMNS here would let a real bug in it propagate
        silently to *both* sides of an "independent" comparison,
        defeating the point of cross-validation (a real weakness in
        v1_0's own version of this adapter, not repeated here).
    ``validation.nusquids.compare``
        Comparison scripts (one per physics scenario: vacuum, Earth,
        atmosphere, solar, the combined solar-to-Earth-detector chain,
        the 3+1 sterile extension). These import *both* ``tpeanuts``
        and ``validation.nusquids.adapter`` -- that is their entire job,
        not a violation of the rule above.
    ``validation.nusquids.benchmark``
        Wall-clock performance comparisons between the two backends.

nuSQuIDS itself is optional and often not installed (no
``nuSQuIDS``/``nuSQUIDSpy``/``nusquids`` Python module found in this
environment as of this package's own creation) -- every function that
needs it degrades gracefully via
``validation.nusquids.adapter.is_available()``/``require_nusquids()``,
matching v1_0's own pattern.

Module contents:
    ``validation.nusquids.config``
        ``NuSQuIDSConfig``: oscillation parameters and solver
        tolerances passed to nuSQuIDS.
    ``validation.nusquids.adapter``
        The pure nuSQuIDS-facing wrapper: solver setup and
        vacuum/Earth/atmosphere/solar probability evaluation.
    ``validation.nusquids.profiles``
        The Earth, atmosphere and Sun that the solver integrates, as NumPy
        arrays (the Akima Earth profile, ``EarthAtm`` and ``SunASnu``).
    ``validation.nusquids.compare``
        tpeanuts-vs-nuSQuIDS comparison scripts, and the
        backend-agnostic grid/error-metric utilities they share.
    ``validation.nusquids.benchmark``
        tpeanuts-vs-nuSQuIDS timing comparisons.
"""

from __future__ import annotations

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

"""Concrete propagation-evolver implementations.

Four strategies live here, each in its own folder, all consuming the same
``PropagationModel``-built generator (``tpeanuts.physics.common.terms.
GeneratorTerm``'s summed contribution) or Liouvillian, never duplicating
how it is assembled:

    ``tpeanuts.physics.evolution.numerical.numerical``
        Matrix-exponential, exact, any generator size -- the only strategy
        that works unconditionally (no dimension restriction, and the only
        one that handles a helicity-doubled or N>=2-sterile-extended
        space). Given its own folder to match every other strategy here,
        even though it holds a single module -- see
        ``tpeanuts.physics.evolution.numerical``'s own package docstring.
    ``tpeanuts.physics.evolution.liouville.open_system``
        Liouville-space evolution for open-system/decoherence dynamics
        (``DissipatorTerm``-driven). Structurally distinct from the other
        three: it acts on a ``dim**2 x dim**2`` Liouvillian, not a ``dim x
        dim`` generator, and is deliberately *not* one of the
        ``PropagationMethod`` values ``tpeanuts.physics.evolution.dispatch.
        evolve_path_unitary`` selects between -- see that reasoning below.
    ``tpeanuts.physics.evolution.perturbative`` (folder, three modules)
        Both variants of one closed-form (Cardano/Ferrari) technique,
        grouped together rather than split into sibling folders, since
        ``analytic_perturbative`` is defined directly in terms of
        ``analytic``'s own result, not an independent method:

        ``tpeanuts.physics.evolution.perturbative.analytic``
            Exact closed-form alternative to ``numerical`` for a
            constant-per-segment generator. Only defined for ``dim in {3,
            4}`` (Abel-Ruffini rules out a closed form for degree >= 5)
            and for the purely unitary case (no active dissipators) --
            ``numerical`` remains the only universal fallback outside
            that domain.
        ``tpeanuts.physics.evolution.perturbative.perturbative``
            ``analytic``'s exact zero-order result, plus an optional
            first-order correction
            (``tpeanuts.physics.evolution.perturbative.profile``) for
            within-segment density variation -- an approximation, unlike
            the other three, and strictly additive: with no variation to
            correct for, it reproduces ``analytic``'s own result exactly.
        ``tpeanuts.physics.evolution.perturbative.spectral``
            Closed-form Cardano/Ferrari spectral decomposition (eigenvalues
            + projectors) both ``analytic`` and ``perturbative`` build on
            -- used by no other strategy here, so it lives alongside them
            rather than flat.

``PropagationMethod`` selects between ``numerical``, ``analytic``, and
``analytic_perturbative`` for the *unitary* generator -- it is data a
medium's own ``tpeanuts.interfaces.medium.MediumProfile`` carries
(``MediumProfile.method``), not something any of
``Source``/``Medium``/``PropagationModel`` branches on themselves. This
keeps the propagation method fully independent of which medium produced
the profile: any medium works with any of the three, and choosing one
outside its domain (dimension, active dissipators, a missing/unwanted
``perturbative_profile``) fails loudly rather than silently falling back
or producing a wrong result -- see
``tpeanuts.physics.evolution.dispatch.evolve_path_unitary`` and
``tpeanuts.physics.model.PropagationModel.check_propagation_method``.

The Liouville-space evolver (``liouville.open_system``) is deliberately
*not* a ``PropagationMethod`` value dispatched by
``dispatch.evolve_path_unitary``: its input (a Liouvillian, ``dim**2 x
dim**2``, no explicit ``-i`` factor -- already baked into
``commutator_superoperator``) is structurally different from the other
three's Hermitian generator, and only the ``NUMERICAL``-style strategy
has any Liouville-space counterpart at all (no closed form for a
Liouvillian's spectral decomposition is implemented, or even
researched, for either ``analytic`` or ``perturbative``) -- unifying the
two at the ``PropagationMethod`` level would mean generalizing a
mostly-empty combination space before a second concrete need exists,
which this project deliberately avoids doing. It is instead reached
through its own single entry point, ``dispatch.evolve_path_liouville``
(``liouville.open_system.evolve_path_liouville``, re-exported so
``PropagationModel`` reaches both representation families through
``dispatch`` alone).

That said, callers still want *one* place to ask for "the evolutor for
this path", regardless of whether dissipators are active. That is what
``EvolutionRepresentation`` is for: it names the two structurally distinct
spaces an evolution can happen in (``UNITARY``, ``dim x dim``, consumed by
``dispatch.evolve_path_unitary``; ``LIOUVILLE``, ``dim**2 x dim**2``,
consumed by ``dispatch.evolve_path_liouville``), one level up from
``PropagationMethod`` (which only ever selects among the three ``UNITARY``
strategies). ``dispatch.evolve_path(model, ...)`` is the single
model-aware entry point that decides ``EvolutionRepresentation`` from
``model.dissipators``, builds the matching tensor (generator or
Liouvillian, via the model's own assembly methods), and only then
delegates to ``dispatch.evolve_path_unitary`` or
``dispatch.evolve_path_liouville`` -- neither of those two lower-level
functions needs to know the other exists.

A no-formal-``Protocol`` policy still applies to the *unitary* evolvers'
own call signature (``segment_evolutor_numerical``/``evolve_path_numerical``
vs. ``segment_evolutor_analytic``/``evolve_path_analytic`` vs.
``segment_evolutor_analytic_perturbative``/``evolve_path_analytic_perturbative``):
each mirrors the others' signature by convention, not by a shared
``Protocol``, since nothing in this project yet needs to treat them
polymorphically beyond the one dispatch point
(``tpeanuts.physics.evolution.dispatch.evolve_path_unitary`` itself) --
a future generic ``Evolver`` interface would be designed once a
concrete need for that polymorphism *elsewhere* actually shows up.

Module contents:
    ``tpeanuts.physics.evolution.method``
        ``PropagationMethod`` (``NUMERICAL`` (default), ``ANALYTIC``, ``ANALYTIC_PERTURBATIVE`` -- which of the three
        *unitary* evolvers builds a segment's evolution operator) and ``EvolutionRepresentation`` (``UNITARY`` or
        ``LIOUVILLE`` -- which structural space an evolution happens in, one level up from ``PropagationMethod``).
        Re-exported here: ``from tpeanuts.physics.evolution import PropagationMethod``.
"""

from __future__ import annotations

from tpeanuts.util.lazy import lazy_exports

_EXPORTS = {
    "PropagationMethod": "tpeanuts.physics.evolution.method",
    "EvolutionRepresentation": "tpeanuts.physics.evolution.method",
}
__all__ = list(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)

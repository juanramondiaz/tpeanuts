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

"""The enums that name how a propagation is carried out: ``PropagationMethod`` and ``EvolutionRepresentation``.

Kept in their own module so that ``tpeanuts.physics.evolution`` (the package ``__init__``) only re-exports, like every
other package of the library. They are imported as ``from tpeanuts.physics.evolution import PropagationMethod`` -- see
that package's own docstring for what each evolver does and why the Liouville path is not a ``PropagationMethod``.

Module contents:
    PropagationMethod
        Enum: ``NUMERICAL`` (default), ``ANALYTIC``, or ``ANALYTIC_PERTURBATIVE`` -- which of the three *unitary*
        evolvers builds a segment's evolution operator.
    EvolutionRepresentation
        Enum: ``UNITARY`` or ``LIOUVILLE`` -- which structural space (a ``dim x dim`` Hermitian generator, or a
        ``dim**2 x dim**2`` Liouvillian) an evolution happens in. One level up from ``PropagationMethod``, which only
        discriminates among the three ``UNITARY`` strategies.
"""

from __future__ import annotations

import enum


class PropagationMethod(enum.Enum):
    """Which unitary evolver builds a segment's evolution operator.

    Attributes:
        NUMERICAL: ``tpeanuts.physics.evolution.numerical.numerical`` --
            ``torch.linalg.matrix_exp``, exact, unconditional. The default,
            reproducing every scenario built before this enum existed.
        ANALYTIC: ``tpeanuts.physics.evolution.perturbative.analytic`` --
            exact closed-form Cardano/Ferrari diagonalization, only
            defined for ``dim in {3, 4}`` and no active dissipators (see
            this module's own docstring). Ignores ``perturbative_profile``
            -- ``tpeanuts.physics.evolution.dispatch.evolve_path_unitary``
            raises if one is given together with this method, rather than
            silently ignoring it (use ``ANALYTIC_PERTURBATIVE`` instead).
        ANALYTIC_PERTURBATIVE: ``tpeanuts.physics.evolution.perturbative.perturbative``
            -- ``ANALYTIC``'s exact zero-order result plus a first-order
            correction for within-segment density variation. Requires a
            ``perturbative_profile`` (``evolve_path_unitary`` raises if
            none is given, rather than silently falling back to the
            zero-order-only result -- use ``ANALYTIC`` directly if that
            is what is actually wanted).

    Examples:
        >>> import tpeanuts as tp
        >>> [method.value for method in tp.PropagationMethod]
        ['numerical', 'analytic', 'analytic_perturbative']
    """

    NUMERICAL = "numerical"
    ANALYTIC = "analytic"
    ANALYTIC_PERTURBATIVE = "analytic_perturbative"


class EvolutionRepresentation(enum.Enum):
    """Which structural space an evolution's generator tensor lives in.

    One level up from ``PropagationMethod``: this discriminates between
    the two structurally incompatible tensor shapes/conventions an
    evolution can be built from, *before* ``PropagationMethod`` even comes
    into play (``PropagationMethod`` only selects among the three
    ``UNITARY`` strategies -- it has no ``LIOUVILLE`` counterpart, see
    this module's own docstring for why).

    Attributes:
        UNITARY: A ``dim x dim`` Hermitian generator (explicit ``-i``
            applied by whichever evolver consumes it), evolved via
            ``tpeanuts.physics.evolution.dispatch.evolve_path_unitary``
            under whichever ``PropagationMethod`` was selected.
        LIOUVILLE: A ``dim**2 x dim**2`` Liouvillian (no explicit ``-i`` --
            already baked into ``commutator_superoperator``), evolved via
            ``tpeanuts.physics.evolution.dispatch.evolve_path_liouville``.
            Used whenever active dissipators are present.
    """

    UNITARY = "unitary"
    LIOUVILLE = "liouville"

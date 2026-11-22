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

"""The single dispatch point between every propagation-representation family.

``evolve_path_unitary`` is the only place in this project that branches
on ``tpeanuts.physics.evolution.PropagationMethod`` among the three
unitary (Hermitian-generator) evolvers -- see that module's own
docstring for why this matters: neither
``tpeanuts.interfaces.medium.MediumProfile`` (which merely *carries* the
chosen method as data), nor ``tpeanuts.physics.model.PropagationModel``
(which builds the generator with no notion of how it will be evolved),
know that three strategies exist. Wiring code (a pipeline, or a test)
calls this one function with whatever ``MediumProfile.method``/
``perturbative_profile`` a medium supplied, and gets back the same
``(..., d, d)`` total path evolutor regardless of which method was
chosen.

``evolve_path_liouville`` is re-exported from
``tpeanuts.physics.evolution.liouville.open_system`` (the one implementation; it is imported, not wrapped, so the name
denotes a single function) so that ``tpeanuts.physics.model.PropagationModel`` reaches *both* representation families
(unitary and Liouville) through this one module, rather than importing the unitary family from ``dispatch`` and the
Liouville family from ``liouville.open_system`` directly. Unlike ``evolve_path_unitary``, there is no branching to do
yet -- Liouville-space evolution has exactly one implementation today (see ``tpeanuts.physics.evolution``'s own
docstring for why the Liouville path is deliberately not a ``PropagationMethod`` value and so is not branched on the
way the unitary family is). A future second Liouville-space strategy would turn the re-export into a dispatching
function of the same name and signature, without ``PropagationModel`` having to know which module implements it.

Each ``PropagationMethod`` value has its own independent branch in
``evolve_path_unitary``, deliberately not inferred from whether
``perturbative_profile`` happens to be given: ``ANALYTIC`` with a
``perturbative_profile`` and ``ANALYTIC_PERTURBATIVE`` with none are
both rejected explicitly rather than silently reinterpreted, since
either combination almost certainly means the caller meant the other
method.

``evolve_path`` is the model-aware entry point every caller outside
``tpeanuts.physics.evolution`` itself should use: it takes a built
``tpeanuts.physics.model.PropagationModel`` directly, decides
``tpeanuts.physics.evolution.EvolutionRepresentation`` from
``model.dissipators``, builds the matching generator/Liouvillian via the
model's own assembly methods, and delegates to ``evolve_path_unitary``/
``evolve_path_liouville``. It lives here, not on ``PropagationModel``
itself, so that ``PropagationModel`` stays exactly what its own module
docstring says it is -- a composition root that assembles Hamiltonians/
Liouvillians from ``extensions``/``dissipators`` -- with no knowledge of
"how to evolve" baked into it. This is legal without an import cycle
precisely because ``PropagationModel`` no longer needs anything from this
module: the dependency runs one way, ``dispatch -> physics.model``, never
back.

Module contents:
    evolve_path_unitary(...)
        Build and compose a whole path's per-segment unitary evolutors,
        via ``tpeanuts.physics.evolution.numerical.numerical.evolve_path_numerical``,
        ``tpeanuts.physics.evolution.perturbative.analytic.evolve_path_analytic``,
        or ``tpeanuts.physics.evolution.perturbative.perturbative.
        evolve_path_analytic_perturbative``, selected by ``method``.
    evolve_path_liouville(...)
        Re-exported from
        ``tpeanuts.physics.evolution.liouville.open_system.evolve_path_liouville``: composes a whole path's
        per-segment Liouville-space evolutors.
    evolve_path(...)
        Model-aware entry point: takes a ``PropagationModel`` directly,
        builds whichever tensor it needs, and delegates to
        ``evolve_path_unitary``/``evolve_path_liouville`` automatically.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

import torch

from tpeanuts.physics.common.terms import PropagationEnvironment
from tpeanuts.physics.evolution import EvolutionRepresentation, PropagationMethod
from tpeanuts.physics.evolution.liouville.open_system import evolve_path_liouville
from tpeanuts.physics.evolution.numerical.numerical import evolve_path_numerical
from tpeanuts.physics.evolution.perturbative.analytic import evolve_path_analytic
from tpeanuts.physics.evolution.perturbative.perturbative import (
    PerturbativeProfile,
    evolve_path_analytic_perturbative,
)
from tpeanuts.physics.model import PropagationModel
from tpeanuts.util.constant import R_E
from tpeanuts.util.tensor import TensorLike


def evolve_path_unitary(
    H_segments: torch.Tensor,
    dx_segments: TensorLike,
    *,
    method: PropagationMethod = PropagationMethod.NUMERICAL,
    reverse: bool = False,
    perturbative_profile: Optional[PerturbativeProfile] = None,
    antinu: Union[bool, torch.Tensor] = False,
    evolution_scale_m: TensorLike = R_E,
    P: Optional[torch.Tensor] = None,
    P_nc: Optional[torch.Tensor] = None,
    analytic_eigenvalues: bool = True,
) -> torch.Tensor:
    """Build and compose the per-segment unitary evolutors for a whole path, via whichever method is selected.

    ``method is PropagationMethod.NUMERICAL`` (the default) reproduces
    ``tpeanuts.physics.evolution.numerical.numerical.evolve_path_numerical`` exactly,
    for any generator size -- the only universal option (see
    ``tpeanuts.physics.evolution``'s own module docstring for the
    dimension/dissipator restrictions the other two are subject to).

    ``method is PropagationMethod.ANALYTIC`` reproduces
    ``tpeanuts.physics.evolution.perturbative.analytic.evolve_path_analytic``'s
    exact zero-order result exactly, and requires ``perturbative_profile
    is None``.

    ``method is PropagationMethod.ANALYTIC_PERTURBATIVE``
    additionally applies the first-order correction via
    ``tpeanuts.physics.evolution.perturbative.perturbative.
    evolve_path_analytic_perturbative``, and requires
    ``perturbative_profile`` to be given.

    Args:
        H_segments: Hermitian generators, one per segment, shaped
            ``(..., S, d, d)``.
        dx_segments: Dimensionless segment lengths, shaped ``(..., S)`` (or
            broadcastable against it).
        method: Which evolver builds each segment's own evolutor.
        reverse: Forwarded to whichever ``evolve_path_*`` function is
            selected.
        perturbative_profile: Density-profile object (see
            ``tpeanuts.physics.evolution.perturbative.profile.EvenPowerProfile``)
            for ``ANALYTIC_PERTURBATIVE``'s first-order correction. Must be
            None for ``NUMERICAL``/``ANALYTIC``, and non-None for
            ``ANALYTIC_PERTURBATIVE`` -- see Raises below.
        antinu: Forwarded to the perturbative correction (ignored by the
            other two methods).
        evolution_scale_m: Forwarded to the perturbative correction
            (ignored by the other two methods) -- must match whatever
            ``H_segments`` was itself built with.
        P: Optional matter-potential direction matrix forwarded to the
            perturbative correction (ignored by the other two methods) --
            see e.g. ``tpeanuts.physics.extensions.nsi.NSITerm.
            direction_reduced``.
        P_nc: Optional neutral-current direction matrix forwarded to the
            perturbative correction (ignored by the other two methods) --
            see ``tpeanuts.physics.extensions.sterile.SterileNCTerm.
            direction_reduced``.
        analytic_eigenvalues: Forwarded to whichever of the two analytic
            evolvers is selected (ignored for ``NUMERICAL``).

    Returns:
        Complex tensor shaped ``(..., d, d)``, the total evolution operator
        for the whole path.

    Raises:
        ValueError: If ``method`` is ``NUMERICAL`` or ``ANALYTIC`` and
            ``perturbative_profile`` is not None (it would be silently
            ignored -- almost certainly a caller meant
            ``ANALYTIC_PERTURBATIVE`` instead). If ``method`` is
            ``ANALYTIC_PERTURBATIVE`` and ``perturbative_profile`` is None
            (there is nothing to correct with -- almost certainly a caller
            meant ``ANALYTIC``, or forgot to pass the profile). If
            ``method`` is ``ANALYTIC``/``ANALYTIC_PERTURBATIVE`` and
            ``H_segments.shape[-1]`` is not 3 or 4. If ``method`` is not a
            recognised ``PropagationMethod`` value.
    """
    if method is PropagationMethod.NUMERICAL:
        if perturbative_profile is not None:
            raise ValueError(
                "perturbative_profile was given but method is "
                "PropagationMethod.NUMERICAL, which never uses it -- it "
                "would be silently ignored. Use "
                "PropagationMethod.ANALYTIC_PERTURBATIVE, or drop the profile."
            )
        return evolve_path_numerical(H_segments, dx_segments, reverse=reverse)

    if method is PropagationMethod.ANALYTIC:
        if perturbative_profile is not None:
            raise ValueError(
                "perturbative_profile was given but method is "
                "PropagationMethod.ANALYTIC, which never uses it -- it "
                "would be silently ignored. Use "
                "PropagationMethod.ANALYTIC_PERTURBATIVE instead, or drop "
                "the profile."
            )
        return evolve_path_analytic(
            H_segments, dx_segments, reverse=reverse, analytic_eigenvalues=analytic_eigenvalues
        )

    if method is PropagationMethod.ANALYTIC_PERTURBATIVE:
        if perturbative_profile is None:
            raise ValueError(
                "method is PropagationMethod.ANALYTIC_PERTURBATIVE but no "
                "perturbative_profile was given -- there would be nothing "
                "to correct with, reproducing PropagationMethod.ANALYTIC's "
                "own result exactly. Use PropagationMethod.ANALYTIC "
                "directly if that is what is actually wanted, or supply a "
                "perturbative_profile."
            )
        return evolve_path_analytic_perturbative(
            H_segments, dx_segments, perturbative_profile,
            reverse=reverse, antinu=antinu, evolution_scale_m=evolution_scale_m,
            P=P, P_nc=P_nc, analytic_eigenvalues=analytic_eigenvalues,
        )

    raise ValueError(f"Unknown PropagationMethod: {method!r}.")


def evolve_path(
    model: PropagationModel,
    environment: PropagationEnvironment,
    dx_segments: TensorLike,
    *,
    method: PropagationMethod = PropagationMethod.NUMERICAL,
    basis: Literal["reduced", "flavour"] = "flavour",
    reverse: bool = False,
    evolution_scale_m: TensorLike = R_E,
    perturbative_profile: Optional[PerturbativeProfile] = None,
    P: Optional[torch.Tensor] = None,
    P_nc: Optional[torch.Tensor] = None,
    analytic_eigenvalues: bool = True,
) -> torch.Tensor:
    """Build and compose the whole-path evolutor for a model, choosing UNITARY vs LIOUVILLE automatically.

    The model-aware entry point for evolving a path -- unlike
    ``evolve_path_unitary`` (which only ever sees a bare ``H_segments``
    tensor and knows nothing about dissipators), this function inspects
    ``model.dissipators`` itself, decides the
    ``tpeanuts.physics.evolution.EvolutionRepresentation`` that decision
    implies, builds the matching tensor (a Liouvillian via
    ``model.liouvillian_reduced``/``liouvillian_flavour``, or a generator
    via ``model.generator_reduced``/``generator_flavour``), and only then
    delegates to whichever of the two structurally-incompatible low-level
    evolvers actually applies:

        ``model.dissipators`` non-empty
            ``EvolutionRepresentation.LIOUVILLE`` -- ``evolve_path_liouville``
            on the built Liouvillian. ``method`` is otherwise unused here
            (only ``NUMERICAL`` has any Liouville-space counterpart at
            all -- see ``tpeanuts.physics.evolution``'s own module
            docstring).
        ``model.dissipators`` empty
            ``EvolutionRepresentation.UNITARY`` -- ``evolve_path_unitary``
            on the built generator, under whichever ``PropagationMethod``
            was selected.

    ``model.check_propagation_method(method)`` runs first in both cases,
    so an ``ANALYTIC``/``ANALYTIC_PERTURBATIVE`` request against a model
    with active dissipators fails loudly here rather than silently
    dropping the decoherence physics -- see that method's own docstring.

    Args:
        model: Built PropagationModel (mixing, mass spectrum, antinu, and
            any active extensions/dissipators) -- the only source this
            function reads its generator/Liouvillian assembly from.
        environment: Neutrino/medium data for the whole path. Its own
            fields carry whatever per-segment structure is wanted (e.g.
            ``n_e_mol_cm3`` shaped ``(..., S)``) -- ``generator_reduced``/
            ``liouvillian_reduced`` already broadcast through ``"..."``,
            so the resulting tensor comes out pre-stacked over segments,
            shaped ``(..., S, d, d)``/``(..., S, d**2, d**2)``, with no
            manual reshaping needed here.
        dx_segments: Dimensionless segment lengths, shaped ``(..., S)``
            (or broadcastable against it).
        method: Which unitary evolver builds each segment's evolutor,
            forwarded to ``evolve_path_unitary`` -- unused when
            ``model.dissipators`` is non-empty (see above).
        basis: ``"flavour"`` (default) uses ``generator_flavour``/
            ``liouvillian_flavour``; ``"reduced"`` uses
            ``generator_reduced``/``liouvillian_reduced``.
        reverse: Forwarded to whichever low-level evolver is selected.
        evolution_scale_m: Positive length scale in metres used to
            non-dimensionalize every term. Defaults to the Earth radius.
        perturbative_profile: Forwarded to ``evolve_path_unitary``
            (unitary case only) -- see its own docstring.
        P: Forwarded to ``evolve_path_unitary`` (unitary case only) --
            see e.g. ``tpeanuts.physics.extensions.nsi.NSITerm.
            direction_reduced``.
        P_nc: Forwarded to ``evolve_path_unitary`` (unitary case only) --
            see e.g. ``tpeanuts.physics.extensions.sterile.
            SterileNCTerm.direction_reduced``.
        analytic_eigenvalues: Forwarded to ``evolve_path_unitary``
            (unitary case only).

    Returns:
        Complex tensor, the total evolution operator for the whole
        path: shaped ``(..., d, d)`` (UNITARY) or ``(..., d**2, d**2)``
        (LIOUVILLE).

    Raises:
        ValueError: Via ``model.check_propagation_method``, or via
            ``evolve_path_unitary``'s own validation (mismatched
            ``method``/``perturbative_profile`` combinations, or an
            unrecognised ``method``).
    """
    model.check_propagation_method(method)

    build_liouvillian = model.liouvillian_flavour if basis == "flavour" else model.liouvillian_reduced
    build_generator = model.generator_flavour if basis == "flavour" else model.generator_reduced

    if model.dissipators:
        representation = EvolutionRepresentation.LIOUVILLE
    else:
        representation = EvolutionRepresentation.UNITARY

    if representation is EvolutionRepresentation.LIOUVILLE:
        L_segments = build_liouvillian(environment, evolution_scale_m=evolution_scale_m)
        return evolve_path_liouville(L_segments, dx_segments, reverse=reverse)

    if representation is EvolutionRepresentation.UNITARY:
        H_segments = build_generator(environment, evolution_scale_m=evolution_scale_m)
        return evolve_path_unitary(
            H_segments,
            dx_segments,
            method=method,
            reverse=reverse,
            perturbative_profile=perturbative_profile,
            antinu=model.antinu,
            evolution_scale_m=evolution_scale_m,
            P=P,
            P_nc=P_nc,
            analytic_eigenvalues=analytic_eigenvalues,
        )

    raise ValueError(f"Unknown EvolutionRepresentation: {representation!r}.")

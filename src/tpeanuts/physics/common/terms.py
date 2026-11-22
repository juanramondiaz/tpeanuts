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

"""The structural contract every pluggable generator contribution follows.

A single structural ``Protocol`` every extension satisfies,
plus one environment bundle that absorbs new environmental data as new
optional fields instead of new function parameters.

With both in place, ``PropagationModel.extensions`` is a plain tuple
of ``GeneratorTerm`` objects, and adding a brand new physics ingredient never
touches ``model.py`` again.

Scope note: this protocol is for contributions to a *Hamiltonian-like*
generator (unitary evolution) only. A future open-system/decoherence
evolver needs dissipator objects living in Liouville space, which are a
structurally different kind of thing (they do not contribute a `(dim, dim)`
term to `H` at all) -- designing a "one protocol to cover both" now, before
a concrete dissipator exists, would be guessing; that gets its own protocol
when ``tpeanuts.physics.evolution.liouville.open_system`` is built. Likewise,
``TermKind`` only has the two members concrete extensions need today
(``ADDITIVE``, ``REPLACES``); a third kind (e.g. a term that *modifies*
another rather than adding to or replacing it) gets added if and when a
concrete extension actually needs one.

Module contents:
    TermKind
        ADDITIVE (combines with everything else) or REPLACES (supersedes
        one specific base or extension term, by key).
    PropagationEnvironment
        Neutrino/medium data a term might need to evaluate its
        contribution. New extensions needing new data add an optional field
        here, never a new parameter anywhere else.
    GeneratorTerm
        The structural Protocol itself: ``kind``, ``replaces``,
        ``validate(...)``, ``contribute_reduced(...)``.
    embed_primary_block(...)
        Places an ``(n_flavour, n_flavour)`` matrix in the top-left
        "primary helicity" block of ``(dim, dim)`` -- the identity when
        ``space.n_helicity == 1`` (every scenario so far). Every concrete
        ``GeneratorTerm`` built from flavour-only physics (which is all of
        them except a future magnetic-moment term) calls this as its final
        step, so ``contribute_reduced``'s contract is genuinely
        ``(..., space.dim, space.dim)`` even though the physics itself
        never has to know ``dim`` exists.
    embed_companion_block(...)
        Places an ``(n_flavour, n_flavour)`` matrix in the bottom-right
        "companion helicity" block of ``(dim, dim)`` -- the mirror image of
        ``embed_primary_block``, used by
        ``tpeanuts.physics.model.PropagationModel`` to place the
        nature-dependent companion generator built via
        ``Nature.companion_helicity_terms``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional, Protocol, Union, runtime_checkable

import torch

from tpeanuts.physics.common.mixing import MixingMatrix
from tpeanuts.physics.common.space import FlavourSpace
from tpeanuts.util.tensor import TensorLike


class TermKind(Enum):
    """How a GeneratorTerm's contribution combines with the terms already active.

    Attributes:
        ADDITIVE: Summed with every other active term unconditionally (e.g.
            the sterile neutral-current term: it has no overlap with
            anything else).
        REPLACES: Supersedes exactly one other term, named by ``replaces``
            (e.g. NSI supersedes the plain charged-current term, since its
            own contribution already includes ``V_cc * diag(1,0,...,0)``
            plus ``epsilon`` -- summing both would double-count the CC
            piece).
    """

    ADDITIVE = auto()
    REPLACES = auto()


@dataclass(frozen=True)
class PropagationEnvironment:
    """Neutrino/medium data a GeneratorTerm might need to evaluate its contribution.

    A new extension needing new environmental data (a magnetic field
    sample, a scalar-field density, ...) adds an optional field here --
    never a new parameter to ``generator_reduced``/``generator_flavour`` or
    to any existing term's ``contribute_reduced``. A term simply reads
    whichever fields it needs and ignores the rest; ``validate`` is where a
    term checks that the field(s) it actually needs were supplied (see
    ``GeneratorTerm``).

    Attributes:
        E_MeV: Neutrino energy in MeV.
        n_e_mol_cm3: Electron density of the propagation medium, in
            mol/cm^3.
        n_n_mol_cm3: Optional neutron density, in mol/cm^3 -- required by
            the sterile neutral-current term, and by NSI only when its
            neutron-normalized coupling is non-zero.
        B_field_T: Optional transverse magnetic field magnitude sampled
            along the path, in tesla -- required by
            ``tpeanuts.physics.extensions.magnetic_moment.MagneticMomentTerm``,
            the only term that reads it.
    """

    E_MeV: torch.Tensor
    n_e_mol_cm3: torch.Tensor
    n_n_mol_cm3: Optional[torch.Tensor] = None
    B_field_T: Optional[torch.Tensor] = None

    if TYPE_CHECKING:  # the runtime __init__ is the dataclass one; tensor fields accept any TensorLike

        def __init__(
            self,
            E_MeV: TensorLike,
            n_e_mol_cm3: TensorLike,
            n_n_mol_cm3: Optional[TensorLike] = ...,
            B_field_T: Optional[TensorLike] = ...,
        ) -> None: ...


@runtime_checkable
class GeneratorTerm(Protocol):
    """Structural contract every pluggable generator contribution follows.

    No inheritance required -- any object exposing these members satisfies
    it (``isinstance(obj, GeneratorTerm)`` works, via ``runtime_checkable``,
    though it only checks that the members exist, not their exact
    signatures).

    Attributes:
        kind: TermKind.ADDITIVE or TermKind.REPLACES.
        replaces: The key (see ``tpeanuts.physics.model``'s base term keys,
            ``"kinetic"``/``"matter_cc"``, or another extension's own key)
            this term supersedes. Required (non-None) iff
            ``kind is TermKind.REPLACES``; ``None`` otherwise.
    """

    @property
    def kind(self) -> TermKind:
        """``TermKind.ADDITIVE`` or ``TermKind.REPLACES`` (read-only: implementations may use a class attribute)."""
        ...

    @property
    def replaces(self) -> Optional[str]:
        """The base-term key this term supersedes when ``kind`` is ``REPLACES``; ``None`` otherwise."""
        ...

    def validate(self, mixing: MixingMatrix, environment: PropagationEnvironment) -> None:
        """Raise ValueError if this term cannot be evaluated as configured.

        Called once per ``generator_reduced``/``generator_flavour`` call,
        before any term's ``contribute_reduced`` runs, so a missing
        environment field (or an incompatible ``mixing``) fails fast rather
        than partway through assembling the generator.

        Args:
            mixing: The MixingMatrix the generator is being built for.
            environment: The PropagationEnvironment this call was given.

        Returns:
            None.

        Raises:
            ValueError: If this term's requirements are not met.
        """
        ...

    def contribute_reduced(
        self,
        mixing: MixingMatrix,
        environment: PropagationEnvironment,
        *,
        antinu: Union[bool, torch.Tensor],
        evolution_scale_m: TensorLike,
    ) -> torch.Tensor:
        """Return this term's reduced-basis contribution to the generator.

        Args:
            mixing: The MixingMatrix the generator is being built for.
            environment: Neutrino/medium data (only the fields this term
                actually needs are read).
            antinu: Bool or boolean tensor selecting the antineutrino
                convention.
            evolution_scale_m: Positive length scale in metres, matching
                whatever every other active term uses.

        Returns:
            Complex tensor shaped ``(..., mixing.space.dim,
            mixing.space.dim)`` -- ``mixing.space.dim == mixing.space.
            n_flavour`` whenever ``mixing.space.n_helicity == 1`` (every
            scenario so far), so a term built from ordinary flavour physics
            (which is all of them except a future magnetic-moment term)
            simply builds its ``(n_flavour, n_flavour)`` contribution and
            calls ``embed_primary_block`` on it as the last step.
        """
        ...


def embed_primary_block(matrix: torch.Tensor, space: FlavourSpace) -> torch.Tensor:
    """Place an ``(n_flavour, n_flavour)`` matrix in the top-left "primary helicity" block of ``(dim, dim)``.

    The identity when ``space.n_helicity == 1`` (``dim == n_flavour``,
    every scenario built so far -- the plain Standard Model, sterile 3+N,
    NSI): ``matrix`` is returned unchanged, so this costs nothing until a
    magnetic-moment extension actually sets ``n_helicity == 2``.

    When ``space.n_helicity == 2``, a magnetic-moment extension couples this
    "primary helicity" block (built from ordinary flavour physics -- kinetic,
    matter potential, NSI, sterile NC, all unaware helicity exists at all)
    to a second, "companion helicity" block via its own off-diagonal
    contribution. This function is the mechanical step that lets a term with
    no concept of helicity still satisfy ``GeneratorTerm``'s
    ``(..., space.dim, space.dim)`` contract: it places ``matrix`` in the
    top-left corner of a ``(dim, dim)`` zero matrix.

    Args:
        matrix: Complex tensor shaped ``(..., space.n_flavour,
            space.n_flavour)``.
        space: FlavourSpace the term's mixing matrix was built for.

    Returns:
        Complex tensor shaped ``(..., space.dim, space.dim)``.
    """
    if not space.is_helicity_extended:
        return matrix

    n = space.n_flavour
    dim = space.dim
    out = torch.zeros(*matrix.shape[:-2], dim, dim, device=matrix.device, dtype=matrix.dtype)
    out[..., :n, :n] = matrix
    return out


def embed_companion_block(matrix: torch.Tensor, space: FlavourSpace) -> torch.Tensor:
    """Place an ``(n_flavour, n_flavour)`` matrix in the bottom-right "companion helicity" block of ``(dim, dim)``.

    The mirror image of ``embed_primary_block``. An ordinary flavour-only
    ``GeneratorTerm`` only ever populates the top-left "primary" block via
    ``embed_primary_block``, regardless of whether the space is
    helicity-extended -- this function exists for the handful of callers
    that deliberately give the companion block its own, independent
    contribution at the companion's own antinu convention (see
    ``tpeanuts.physics.common.nature.Nature.companion_helicity_terms``):
    ``tpeanuts.physics.model.PropagationModel`` itself, for the
    nature-dependent companion Hamiltonian, and
    ``tpeanuts.physics.extensions.decoherence.DecoherenceTerm``, for its
    optional ``gamma_companion`` Lindblad operators.

    Args:
        matrix: Complex tensor shaped ``(..., space.n_flavour,
            space.n_flavour)``.
        space: FlavourSpace the term's mixing matrix was built for. Must be
            helicity-extended -- a companion block does not exist otherwise.

    Returns:
        Complex tensor shaped ``(..., space.dim, space.dim)``.

    Raises:
        ValueError: If ``space.n_helicity != 2``.
    """
    if not space.is_helicity_extended:
        raise ValueError(
            "embed_companion_block requires a helicity-extended FlavourSpace "
            f"(n_helicity == 2), got n_helicity={space.n_helicity}."
        )

    n = space.n_flavour
    dim = space.dim
    out = torch.zeros(*matrix.shape[:-2], dim, dim, device=matrix.device, dtype=matrix.dtype)
    out[..., n:, n:] = matrix
    return out


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

"""The composition root of the physics engine: one object per scenario.

``PropagationModel`` is the single place in ``tpeanuts.physics`` allowed to
know about both ``tpeanuts.physics.common`` (unconditional infrastructure)
and ``tpeanuts.physics.extensions`` (optional, off-by-default ingredients)
at once -- exactly the role ``config/propagation.py`` played in v1, scoped
here to the physics engine alone (a named-preset registry building
``PropagationModel`` instances from strings like ``"_SM_NUFIT52_NO"`` is a
later, separate piece: ``tpeanuts.config``, not yet built).

Design note (revised after review): the first version of this module bundled
optional physics as named fields (``nsi: Optional[NSITerm]``,
``include_sterile_nc: bool``) with hand-written ``if``/``else`` branches
assembling them -- and a real bug (the charged-current term getting summed
twice whenever NSI was active) surfaced from exactly that hand-written
branching. It also meant every new extension would need a new import, a new
dataclass field, a new assembly branch, and a new keyword argument on
``generator_reduced``/``generator_flavour`` for whatever new environmental
data it needed -- open for modification, not open for extension. This
version fixes both: ``extensions`` is a plain tuple of
``tpeanuts.physics.common.terms.GeneratorTerm``-conforming objects, the
assembly loop is generic (no per-extension branching at all, see
``_assemble_terms``), and new environmental data is a new optional field on
``tpeanuts.physics.common.terms.PropagationEnvironment`` rather than a new
parameter here. Adding a brand new physics ingredient (a magnetic moment,
Lorentz violation, a scalar potential, ...) from here on never touches this
file: write the term class, satisfy the protocol, pass an instance in
``extensions=(...)``.

Helicity doubling (``space.n_helicity == 2``): every base/extension term in
``_assemble_terms`` only ever knows about the "primary" flavour physics,
embedded in the top-left corner of the ``(dim, dim)`` generator via
``embed_primary_block``. ``generator_reduced`` adds one further, disjoint
piece on top: the "companion helicity" block, built by re-running the exact
same assembly (``_assemble_generator``) at whatever ``antinu``/
``include_matter_terms`` combination ``self.nature.companion_helicity_terms``
prescribes, then re-embedded in the bottom-right corner via
``embed_companion_block``. The two diagonal blocks are summed independently;
the off-diagonal coupling between them is
``tpeanuts.physics.extensions.magnetic_moment.MagneticMomentTerm``'s own job
-- an ordinary ``ADDITIVE`` extension like any other, requiring no special
wiring here. ``generator_flavour`` dresses the resulting ``(dim, dim)``
generator with a *block-diagonal* outer rotation (see
``_outer_block_total``), not a single shared ``O``: the primary and
companion blocks were each built at their own ``antinu`` convention
(identical for Dirac, opposite for Majorana), so each needs its own block's
``O`` on rotation.

Open-system (decoherence) evolution: ``dissipators`` is the
``DissipatorTerm`` analogue of ``extensions`` -- a plain tuple, empty
(purely unitary) by default. ``liouvillian_reduced``/``liouvillian_flavour``
convert whatever ``generator_reduced``/``generator_flavour`` already built
into Liouville-space form (via
``tpeanuts.physics.evolution.liouville.open_system.commutator_superoperator``) and add
each active dissipator's own contribution on top -- they add no
helicity-specific logic of their own, reusing
``generator_reduced``/``_outer_block_total`` exactly as built. See
``tpeanuts.physics.evolution.liouville.open_system``'s own module docstring for why
decoherence needs this structurally different (Liouville-space) machinery
at all, rather than being just another ``GeneratorTerm``.

Module contents:
    PropagationModel
        Bundles a MixingMatrix, a MassSpectrum, the antineutrino selector, a
        tuple of active extension terms, and a tuple of active dissipator
        terms, into the four assembly methods every higher layer actually
        needs: ``generator_reduced``/``generator_flavour`` (unitary
        generator) and ``liouvillian_reduced``/``liouvillian_flavour``
        (open-system Liouvillian, unitary plus any active decoherence) --
        plus ``check_propagation_method`` for validating a chosen
        ``PropagationMethod`` against ``self.dissipators``. Deliberately
        does *not* know "how to evolve" or "which method": that logic
        lives entirely in ``tpeanuts.physics.evolution`` (e.g.
        ``tpeanuts.physics.evolution.dispatch.evolve_path(model, ...)``),
        as a model-aware function per family that takes a
        ``PropagationModel`` and calls back into these assembly methods --
        see ``tpeanuts.physics.evolution.dispatch``'s own module docstring
        for why this direction (``physics.evolution`` depending on
        ``physics.model``, not the reverse) is exactly backwards from how
        this file used to be structured, and why that is not a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import torch

from tpeanuts.physics.common.generator import (
    electron_projector,
    kinetic_term_reduced,
    matter_term_reduced,
    validate_consistency,
)
from tpeanuts.physics.common.mass_spectrum import MassSpectrum
from tpeanuts.physics.common.mixing import MixingMatrix
from tpeanuts.physics.common.nature import Nature
from tpeanuts.physics.common.space import FlavourSpace
from tpeanuts.physics.common.terms import (
    GeneratorTerm,
    PropagationEnvironment,
    TermKind,
    embed_companion_block,
    embed_primary_block,
)
from tpeanuts.physics.evolution import PropagationMethod
from tpeanuts.physics.evolution.liouville.open_system import (
    DissipatorTerm,
    commutator_superoperator,
    dissipator_superoperator,
)
from tpeanuts.util.constant import R_E
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import TensorLike, cdtype_from_real, kron_batched

# Keys for the two base terms every scenario has unconditionally. An
# extension supersedes one of these by setting its own `replaces` to the
# matching key (e.g. NSITerm.replaces == "matter_cc").
_KINETIC_KEY = "kinetic"
_MATTER_CC_KEY = "matter_cc"


class _KineticGeneratorTerm:
    """Adapter: wraps ``kinetic_term_reduced`` as a GeneratorTerm. Private to this module."""

    kind = TermKind.ADDITIVE
    replaces: Optional[str] = None

    def __init__(self, mass_spectrum: MassSpectrum) -> None:
        self._mass_spectrum = mass_spectrum

    def validate(self, mixing: MixingMatrix, environment: PropagationEnvironment) -> None:
        """Always valid: the kinetic term needs only E_MeV, always present."""

    def contribute_reduced(
        self,
        mixing: MixingMatrix,
        environment: PropagationEnvironment,
        *,
        antinu: Union[bool, torch.Tensor],
        evolution_scale_m: TensorLike,
    ) -> torch.Tensor:
        """Return the kinetic (mass-splitting) contribution to the generator, embedded to ``(dim, dim)``.

        Args:
            mixing: The MixingMatrix the generator is being built for.
            environment: Neutrino/medium data; only ``E_MeV`` is read.
            antinu: Bool or boolean tensor selecting the antineutrino convention.
            evolution_scale_m: Positive length scale in metres, shared by every active term.

        Returns:
            Complex tensor shaped ``(..., dim, dim)`` in the reduced basis.
        """
        H = kinetic_term_reduced(
            mixing,
            self._mass_spectrum,
            environment.E_MeV,
            antinu=antinu,
            evolution_scale_m=evolution_scale_m,
        )
        return embed_primary_block(H, mixing.space)


class _MatterCCGeneratorTerm:
    """Adapter: wraps ``matter_term_reduced`` as a GeneratorTerm. Private to this module."""

    kind = TermKind.ADDITIVE
    replaces: Optional[str] = None

    def validate(self, mixing: MixingMatrix, environment: PropagationEnvironment) -> None:
        """Always valid: the CC term needs only n_e_mol_cm3, always present."""

    def contribute_reduced(
        self,
        mixing: MixingMatrix,
        environment: PropagationEnvironment,
        *,
        antinu: Union[bool, torch.Tensor],
        evolution_scale_m: TensorLike,
    ) -> torch.Tensor:
        """Return the charged-current matter contribution to the generator, embedded to ``(dim, dim)``.

        Args:
            mixing: The MixingMatrix the generator is being built for.
            environment: Neutrino/medium data; only ``n_e_mol_cm3`` is read.
            antinu: Bool or boolean tensor selecting the antineutrino convention.
            evolution_scale_m: Positive length scale in metres, shared by every active term.

        Returns:
            Complex tensor shaped ``(..., dim, dim)`` in the reduced basis.
        """
        H = matter_term_reduced(
            mixing,
            environment.n_e_mol_cm3,
            antinu=antinu,
            evolution_scale_m=evolution_scale_m,
        )
        return embed_primary_block(H, mixing.space)

    def direction_reduced(
        self,
        mixing: MixingMatrix,
        environment: PropagationEnvironment,
        *,
        antinu: Union[bool, torch.Tensor] = False,
    ) -> torch.Tensor:
        """Return this term's V-independent direction matrix ``diag(1,0,...,0)``, embedded to ``(dim, dim)``.

        Optional ``GeneratorTerm`` extension consumed by
        ``tpeanuts.physics.evolution.perturbative.perturbative``'s first-order perturbative
        correction (not part of the required ``GeneratorTerm`` protocol
        surface -- a duck-typed extra, exactly like
        ``tpeanuts.physics.extensions.nsi.NSITerm``'s/``tpeanuts.physics.
        extensions.sterile.SterileNCTerm``'s own). Pairs with
        ``tpeanuts.physics.common.potential.matter_potential_cc(
        environment.n_e_mol_cm3, antinu, evolution_scale_m=...)``:
        multiplying the two together reproduces ``contribute_reduced``
        exactly (confirmed by a dedicated test, not just asserted).
        ``diag(1,0,...,0)`` is real and antinu-independent on its own --
        the antinu sign lives entirely in the potential -- so ``antinu`` is
        accepted only for signature parity with the other two terms that
        implement this, and is otherwise unused here.

        Args:
            mixing: Built MixingMatrix.
            environment: Unused -- this term's direction matrix needs no
                environment data (only the potential built separately does).
            antinu: Unused, see above.

        Returns:
            Complex tensor shaped ``(..., mixing.space.dim,
            mixing.space.dim)``.
        """
        context = mixing.context
        n = mixing.space.n_flavour
        cdtype = cdtype_from_real(context.dtype)
        P_cc = electron_projector(n, device=context.device, cdtype=cdtype)
        return embed_primary_block(P_cc, mixing.space)


@dataclass(frozen=True)
class PropagationModel:
    """One complete propagation scenario: mixing, mass spectrum, and active extensions.

    Attributes:
        mixing: Built MixingMatrix (its ``nature`` and ``space`` are this
            model's nature and flavour space too -- see the ``nature``/
            ``space``/``context`` properties below).
        mass_spectrum: Built MassSpectrum, sharing ``mixing``'s
            FlavourSpace and RuntimeContext.
        antinu: Bool or boolean tensor selecting the antineutrino
            convention, applied uniformly to every term.
        extensions: Tuple of ``GeneratorTerm``-conforming objects (e.g.
            ``tpeanuts.physics.extensions.sterile.SterileNCTerm()``,
            ``tpeanuts.physics.extensions.nsi.NSITerm(config)``). Empty
            (default) is the plain Standard Model matter potential; order
            does not affect the result (assembly is commutative -- see
            ``_assemble_terms``).
        dissipators: Tuple of ``tpeanuts.physics.evolution.liouville.open_system.
            DissipatorTerm``-conforming objects (e.g.
            ``tpeanuts.physics.extensions.decoherence.
            DecoherenceTerm(DecoherenceConfig(gamma, context))``). Empty (default) means
            purely unitary evolution -- ``liouvillian_reduced`` then reduces
            to exactly ``commutator_superoperator(generator_reduced(...))``,
            with no dissipative contribution at all. Independent of
            ``extensions``: a dissipator never replaces or is replaced by a
            ``GeneratorTerm``, since they contribute to structurally
            different things (a Hamiltonian term vs. a Lindblad operator).

    Raises:
        ValueError: If ``mixing``/``mass_spectrum`` disagree on space or
            context, if any entry of ``extensions`` does not satisfy
            ``GeneratorTerm``, if any entry of ``dissipators`` does not
            satisfy ``DissipatorTerm``, or if two ``extensions`` entries
            both declare ``replaces`` for the same key.

    Examples:
        >>> import math
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> params = tp.StandardMixingParameters(
        ...     theta12=math.radians(33.4), theta13=math.radians(8.6), theta23=math.radians(42.0),
        ...     delta13=math.radians(230.0), context=context)
        >>> space = tp.FlavourSpace()
        >>> mixing = tp.MixingMatrix(space=space, standard=params, nature=tp.Dirac(), context=context)
        >>> masses = tp.MassSpectrum(space=space, DeltamSq21=7.4e-5, DeltamSq3l=2.5e-3, context=context)
        >>> model = tp.PropagationModel(mixing=mixing, mass_spectrum=masses)   # no extensions: the Standard Model
        >>> from tpeanuts.physics.common.terms import PropagationEnvironment
        >>> environment = PropagationEnvironment(E_MeV=torch.tensor(1000.0, dtype=torch.float64), n_e_mol_cm3=1.0)
        >>> H = model.generator_reduced(environment)   # kinetic + matter generator, in the reduced basis
        >>> H.shape, bool(torch.allclose(H, H.conj().transpose(-2, -1)))
        (torch.Size([3, 3]), True)
        >>> model.antinu
        False
    """

    mixing: MixingMatrix
    mass_spectrum: MassSpectrum
    antinu: Union[bool, torch.Tensor] = False
    extensions: Tuple[GeneratorTerm, ...] = field(default_factory=tuple)
    dissipators: Tuple[DissipatorTerm, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        validate_consistency(self.mixing, self.mass_spectrum)
        for term in self.extensions:
            if not isinstance(term, GeneratorTerm):
                raise TypeError(
                    f"{term!r} does not satisfy tpeanuts.physics.common.terms."
                    "GeneratorTerm (needs kind, replaces, validate(...), "
                    "contribute_reduced(...))."
                )
            self._check_kind_replaces_consistency(term)
        self._check_no_conflicting_replaces()
        self._assemble_terms()  # discarded: run once here so a bad `replaces`
        # target (typo'd key, or a key removed by an earlier extension) is
        # rejected at construction time, not silently deferred to whichever
        # generator_reduced/generator_flavour call happens to run first.
        for dissipator in self.dissipators:
            if not isinstance(dissipator, DissipatorTerm):
                raise TypeError(
                    f"{dissipator!r} does not satisfy tpeanuts.physics.evolution."
                    "liouville.open_system.DissipatorTerm (needs "
                    "validate(...), lindblad_operators(...))."
                )

    @staticmethod
    def _check_kind_replaces_consistency(term: GeneratorTerm) -> None:
        """Raise if ``term``'s ``kind``/``replaces`` pair is internally inconsistent.

        Args:
            term: The extension term to check.

        Raises:
            ValueError: If ``kind is TermKind.REPLACES`` but ``replaces`` is
                None (REPLACES must name the key it supersedes), or if
                ``kind is TermKind.ADDITIVE`` but ``replaces`` is not None
                (an ADDITIVE term must not declare a target -- ADDITIVE and
                REPLACES are mutually exclusive by construction elsewhere in
                this module, so a mismatch here is always a caller bug in
                the term's own definition).
        """
        if term.kind is TermKind.REPLACES and term.replaces is None:
            raise ValueError(
                f"{term!r} has kind=TermKind.REPLACES but replaces=None; "
                "REPLACES must name the key of the term it supersedes."
            )
        if term.kind is TermKind.ADDITIVE and term.replaces is not None:
            raise ValueError(
                f"{term!r} has kind=TermKind.ADDITIVE but replaces={term.replaces!r}; "
                "an ADDITIVE term must not declare a replaces target."
            )

    def _check_no_conflicting_replaces(self) -> None:
        """Raise if two active extensions both declare ``replaces`` for the same key."""
        claimed: dict[str, GeneratorTerm] = {}
        for term in self.extensions:
            if term.kind is not TermKind.REPLACES:
                continue
            assert term.replaces is not None  # kind REPLACES requires a target (checked just before)
            if term.replaces in claimed:
                raise ValueError(
                    f"Both {claimed[term.replaces]!r} and {term!r} declare "
                    f"replaces={term.replaces!r}; at most one active extension "
                    "may supersede a given term."
                )
            claimed[term.replaces] = term

    def _assemble_terms(self) -> dict[str, GeneratorTerm]:
        """Build the active {key: term} mapping: base terms and extensions, minus anything superseded.

        Base terms (``"kinetic"``, ``"matter_cc"``) and every extension
        (keyed by its position in ``extensions``, ``"extension_0"``, ...)
        are all added *first*, so a later extension can genuinely replace an
        earlier one's key -- not just a base term's. Only once every key
        exists does any ``kind is TermKind.REPLACES`` extension remove the
        term it names; a ``replaces`` value that doesn't match any key at
        that point (a typo, or a key already removed by an earlier
        extension in this same call) is a hard error rather than a silently
        ignored no-op, since the alternative is the superseded term staying
        active *and* the replacing term being added -- silent double-counting.

        Returns:
            Ordered mapping from key to the GeneratorTerm still active.

        Raises:
            ValueError: If any REPLACES extension's ``replaces`` does not
                match an existing key at the time it is processed.
        """
        terms: dict[str, GeneratorTerm] = {
            _KINETIC_KEY: _KineticGeneratorTerm(self.mass_spectrum),
            _MATTER_CC_KEY: _MatterCCGeneratorTerm(),
        }
        for i, term in enumerate(self.extensions):
            terms[f"extension_{i}"] = term

        for term in self.extensions:
            if term.kind is not TermKind.REPLACES:
                continue
            if term.replaces not in terms:
                raise ValueError(
                    f"{term!r} declares replaces={term.replaces!r}, but no "
                    "active term uses that key at this point in assembly "
                    f"(active keys: {sorted(terms)!r}). Check for a typo, or "
                    "for a target already removed by an earlier extension."
                )
            terms.pop(term.replaces)
        return terms

    @property
    def nature(self) -> Nature:
        """This model's neutrino nature (Dirac or Majorana), read from ``mixing``."""
        return self.mixing.nature

    @property
    def space(self) -> FlavourSpace:
        """This model's FlavourSpace, read from ``mixing`` (equal to ``mass_spectrum.space``)."""
        return self.mixing.space

    @property
    def context(self) -> RuntimeContext:
        """This model's RuntimeContext, read from ``mixing`` (equal to ``mass_spectrum.context``)."""
        return self.mixing.context

    def check_propagation_method(self, method: PropagationMethod) -> None:
        """Raise if ``method`` is incompatible with this model's own active dissipators.

        Both analytic-family methods (``PropagationMethod.ANALYTIC`` and
        ``ANALYTIC_PERTURBATIVE``, see ``tpeanuts.physics.evolution.
        perturbative.analytic``/``.perturbative``) only ever build a unitary
        evolutor from ``generator_reduced``/``generator_flavour`` -- neither has any
        concept of a Liouvillian at all, so any active ``DissipatorTerm``
        would simply be dropped, not represented approximately. Rather than
        silently reproducing the purely-unitary result while the caller
        believes their decoherence model is included, this fails loudly --
        the same "fail fast rather than silently reproduce an unstated
        default" discipline every other genuine physics-scope limitation in
        this project already follows (e.g. ``tpeanuts.physics.extensions.
        sterile.SterileNCTerm.validate``, ``tpeanuts.physics.extensions.
        decoherence.DecoherenceTerm.validate``).

        ``tpeanuts.physics.evolution.dispatch.evolve_path(model, ...)`` calls
        this automatically, so wiring code that goes through it never needs
        to call this separately. Code that instead calls
        ``tpeanuts.physics.evolution.dispatch.evolve_path_unitary`` directly
        on a hand-built ``H_segments`` (bypassing ``dispatch.evolve_path``
        entirely, e.g. because it already has the generator and just wants
        the low-level dispatch) should call this method first, explicitly --
        that lower-level path has no ``PropagationModel`` of its own to
        check against.

        Args:
            method: The propagation method the caller intends to use.

        Raises:
            ValueError: If ``method`` is ``PropagationMethod.ANALYTIC`` or
                ``ANALYTIC_PERTURBATIVE`` and ``self.dissipators`` is
                non-empty.
        """
        analytic_family = (PropagationMethod.ANALYTIC, PropagationMethod.ANALYTIC_PERTURBATIVE)
        if method in analytic_family and self.dissipators:
            raise ValueError(
                f"{method!r} only builds a unitary evolutor (neither "
                "tpeanuts.physics.evolution.perturbative.analytic nor .perturbative.perturbative has "
                "any Liouvillian concept at all), but this model has active "
                f"dissipators ({self.dissipators!r}) -- using {method!r} "
                "would silently drop the decoherence physics rather than "
                "represent it approximately. Use PropagationMethod.NUMERICAL "
                "(with tpeanuts.physics.model.PropagationModel."
                "liouvillian_reduced/liouvillian_flavour) instead, or "
                "construct this model with dissipators=() if unitary-only "
                "propagation is genuinely intended."
            )

    def _assemble_generator(
        self,
        environment: PropagationEnvironment,
        antinu: Union[bool, torch.Tensor],
        evolution_scale_m: TensorLike,
        *,
        include_matter_terms: bool = True,
    ) -> torch.Tensor:
        """Assemble a reduced-basis generator from the active terms, at a given ``antinu``.

        Every active term's ``validate`` runs first,
        then every active term's ``contribute_reduced`` is summed. There is
        no per-extension branching here: which terms are "active" is
        entirely determined by ``_assemble_terms`` from ``self.extensions``'
        own ``kind``/``replaces`` declarations.

        Taking ``antinu`` as an explicit argument (rather than always reading
        ``self.antinu``) is what lets this same assembly serve a doubled
        (``n_helicity == 2``) generator's "companion helicity" block, which a
        Majorana nature evaluates at the *opposite* ``antinu`` from the
        primary block -- see
        ``tpeanuts.physics.common.nature.Nature.companion_helicity_terms``.
        Likewise, ``include_matter_terms=False`` restricts the sum to the
        kinetic term alone, for a Dirac companion block (a genuine
        gauge-singlet ``nu_R`` feels no matter potential, NSI, or sterile NC
        term).

        Args:
            environment: Neutrino/medium data (``E_MeV``, ``n_e_mol_cm3``,
                optionally ``n_n_mol_cm3``, and whatever else a given
                extension needs -- see
                ``tpeanuts.physics.common.terms.PropagationEnvironment``).
            antinu: Bool or boolean tensor selecting the antineutrino
                convention this particular block uses.
            evolution_scale_m: Positive length scale in metres used to
                non-dimensionalize every term.
            include_matter_terms: When False, only the kinetic term is
                assembled (every other base term and extension is skipped
                entirely, including its ``validate``). True (default)
                assembles every active term as usual.

        Returns:
            Complex Hermitian tensor shaped ``(..., space.dim, space.dim)``
            -- each summed term already calls ``embed_primary_block`` as its
            own last step, so this is genuinely ``(dim, dim)``, not
            ``(n_flavour, n_flavour)``, whenever ``space.n_helicity == 2``.

        Raises:
            ValueError: If any active term's ``validate`` fails (e.g. a
                required environment field is missing).
        """
        terms = self._assemble_terms()
        if not include_matter_terms:
            terms = {_KINETIC_KEY: terms[_KINETIC_KEY]}

        for term in terms.values():
            term.validate(self.mixing, environment)

        H: Optional[torch.Tensor] = None
        for term in terms.values():
            contribution = term.contribute_reduced(
                self.mixing, environment, antinu=antinu, evolution_scale_m=evolution_scale_m
            )
            H = contribution if H is None else H + contribution

        assert H is not None  # unreachable: terms always has at least "kinetic"
        return H

    def _companion_generator_reduced(
        self,
        environment: PropagationEnvironment,
        antinu: Union[bool, torch.Tensor],
        evolution_scale_m: TensorLike,
    ) -> torch.Tensor:
        """Build the companion helicity block's own contribution, embedded in the bottom-right corner.

        Deliberately not one of the entries ``_assemble_terms`` returns:
        that dict is what ``_assemble_generator`` re-derives every time it
        is called, so including this method's own output there would have
        it call itself recursively. Instead this is a distinct, final
        additive piece, wired in directly by ``generator_reduced`` only when
        ``self.space.is_helicity_extended``.

        The companion block's own physics (kinetic-only vs. full generator,
        and at which ``antinu``) is entirely delegated to
        ``self.nature.companion_helicity_terms`` -- see that method and this
        module's docstring for why it differs between Dirac and Majorana.
        Reusing ``_assemble_generator`` for it means the companion block
        already includes whichever extensions (NSI, sterile NC, ...) are
        active, at no extra cost: it is exactly the same assembly as the
        primary block, just at a different ``antinu`` and possibly
        restricted to the kinetic term alone.

        ``_assemble_generator`` returns its result already embedded in the
        top-left "primary" corner (every base/extension term calls
        ``embed_primary_block`` as its own last step) -- that corner is
        sliced back out here and re-embedded in the bottom-right "companion"
        corner via ``embed_companion_block``.

        Args:
            environment: See ``generator_reduced``.
            antinu: The *primary* block's antinu convention -- this method
                derives the companion's own antinu from it via
                ``Nature.companion_helicity_terms``.
            evolution_scale_m: Positive length scale in metres used to
                non-dimensionalize every term.

        Returns:
            Complex tensor shaped ``(..., space.dim, space.dim)``, zero
            outside the bottom-right ``(n_flavour, n_flavour)`` block.
        """
        companion_antinu, include_matter_terms = self.nature.companion_helicity_terms(antinu)
        H_companion = self._assemble_generator(
            environment, companion_antinu, evolution_scale_m, include_matter_terms=include_matter_terms
        )
        n = self.space.n_flavour
        return embed_companion_block(H_companion[..., :n, :n], self.space)

    def generator_reduced(
        self,
        environment: PropagationEnvironment,
        *,
        evolution_scale_m: TensorLike = R_E,
    ) -> torch.Tensor:
        """Assemble the reduced-basis generator from every active term.

        A thin wrapper over ``_assemble_generator`` at this model's own
        ``antinu``, with every active term included. When
        ``self.space.is_helicity_extended``, the nature-dependent companion
        helicity block (see ``_companion_generator_reduced``) is added as an
        extra, disjoint contribution in the bottom-right corner -- there is
        no off-diagonal coupling between the two blocks yet, that is a
        future magnetic-moment extension's job.

        Args:
            environment: Neutrino/medium data (``E_MeV``, ``n_e_mol_cm3``,
                optionally ``n_n_mol_cm3``, and whatever else a given
                extension needs -- see
                ``tpeanuts.physics.common.terms.PropagationEnvironment``).
            evolution_scale_m: Positive length scale in metres used to
                non-dimensionalize every term. Defaults to the Earth radius.

        Returns:
            Complex Hermitian tensor shaped ``(..., space.dim, space.dim)``.

        Raises:
            ValueError: If any active term's ``validate`` fails (e.g. a
                required environment field is missing).
        """
        H = self._assemble_generator(environment, self.antinu, evolution_scale_m)
        if self.space.is_helicity_extended:
            H = H + self._companion_generator_reduced(environment, self.antinu, evolution_scale_m)
        return H

    def _outer_block_total(self) -> torch.Tensor:
        """Build the full ``(dim, dim)`` outer rotation, block-diagonal when helicity-doubled.

        For ``space.n_helicity == 1`` this is exactly
        ``mixing.outer_block(antinu=self.antinu)`` (``dim == n_flavour``, no
        doubling). For ``space.n_helicity == 2`` the primary and companion
        blocks of ``generator_reduced``'s output were each built at their
        *own* antinu convention (identical for Dirac, opposite for Majorana
        -- see ``Nature.companion_helicity_terms``), so dressing the full
        generator into the flavour basis needs the matching block-diagonal
        rotation

            O_total = [[O_primary,   0        ],
                       [0,           O_companion]]

        rather than a single shared ``O`` -- each block is individually
        unitary (a product of unitary Givens rotations), so ``O_total`` is
        unitary too, and ``O_total @ H_reduced @ O_total^dagger`` correctly
        recovers the flavour-basis generator including any off-diagonal
        (magnetic-moment) coupling between the two blocks: a direct check
        confirms the coupling block rotates back to exactly the flavour-
        basis ``mu`` a ``MagneticMomentTerm`` was configured with, since
        ``O_companion @ (O_companion^dagger @ mu @ O_primary) @
        O_primary^dagger == mu``.

        Returns:
            Complex tensor shaped ``(..., space.dim, space.dim)``.
        """
        O_primary = self.mixing.outer_block(antinu=self.antinu)
        if not self.space.is_helicity_extended:
            return O_primary

        companion_antinu, _ = self.nature.companion_helicity_terms(self.antinu)
        O_companion = self.mixing.outer_block(antinu=companion_antinu)

        # Disjoint corners, so summing the two single-block embeddings is
        # exactly the block-diagonal combination -- reuses the same
        # primitives every GeneratorTerm's own contribute_reduced does,
        # rather than building the zero tensor by hand a second time.
        return embed_primary_block(O_primary, self.space) + embed_companion_block(O_companion, self.space)

    def generator_flavour(
        self,
        environment: PropagationEnvironment,
        *,
        evolution_scale_m: TensorLike = R_E,
    ) -> torch.Tensor:
        """Build the full flavour-basis generator from every active term.

        Dresses ``generator_reduced``'s output with the (block-diagonal,
        when helicity-doubled) outer rotation, ``O H_red O^dagger`` -- a
        single rotation applied once every active term has already been
        summed in the reduced basis. See ``_outer_block_total`` for why a
        helicity-doubled space needs a block-diagonal rotation rather than a
        single shared one.

        Args:
            environment: See ``generator_reduced``.
            evolution_scale_m: Positive length scale in metres used to
                non-dimensionalize every term. Defaults to the Earth radius.

        Returns:
            Complex Hermitian tensor shaped ``(..., space.dim, space.dim)``.
        """
        H_reduced = self.generator_reduced(environment, evolution_scale_m=evolution_scale_m)
        O = self._outer_block_total()
        return O @ H_reduced @ O.conj().transpose(-2, -1)

    def _assemble_lindblad_operators(
        self,
        environment: PropagationEnvironment,
        antinu: Union[bool, torch.Tensor],
        evolution_scale_m: TensorLike,
    ) -> Tuple[torch.Tensor, ...]:
        """Collect every active dissipator's Lindblad operators, already embedded to ``(dim, dim)``.

        Every active dissipator's ``validate`` runs first (mirrors
        ``_assemble_generator``'s own validate-before-contribute ordering),
        then every active dissipator's ``lindblad_operators`` is collected
        into one flat tuple -- there is no notion of one dissipator
        replacing another (unlike ``extensions``' ``REPLACES`` mechanism):
        every active ``DissipatorTerm`` simply contributes its own operators
        alongside every other one's.

        Args:
            environment: See ``liouvillian_reduced``.
            antinu: Bool or boolean tensor selecting the antineutrino
                convention this particular block uses.
            evolution_scale_m: Positive length scale in metres, matching
                whatever every other active term uses.

        Returns:
            Flat tuple of complex tensors, each shaped
            ``(..., space.dim, space.dim)``.

        Raises:
            ValueError: If any active dissipator's ``validate`` fails.
        """
        for term in self.dissipators:
            term.validate(self.mixing, environment)

        operators: list = []
        for term in self.dissipators:
            operators.extend(
                term.lindblad_operators(
                    self.mixing, environment, antinu=antinu, evolution_scale_m=evolution_scale_m
                )
            )
        return tuple(operators)

    def liouvillian_reduced(
        self,
        environment: PropagationEnvironment,
        *,
        evolution_scale_m: TensorLike = R_E,
    ) -> torch.Tensor:
        """Build the full reduced-basis Liouvillian: the unitary commutator plus every active dissipator.

        ``commutator_superoperator(self.generator_reduced(...))`` already
        includes the companion-helicity-block/off-diagonal-coupling physics
        ``generator_reduced`` itself assembles -- this method adds nothing
        helicity-specific of its own, it only converts whatever
        ``generator_reduced`` already built into its Liouville-space form
        and adds each active dissipator's own contribution
        (``tpeanuts.physics.evolution.liouville.open_system.dissipator_superoperator``)
        on top. With ``self.dissipators == ()`` (the default), this reduces
        to exactly ``commutator_superoperator(self.generator_reduced(...))``
        -- purely unitary evolution, recoverable bit-for-bit.

        Args:
            environment: Neutrino/medium data (``E_MeV``, ``n_e_mol_cm3``,
                optionally ``n_n_mol_cm3``/``B_field_T``, and whatever else
                a given extension or dissipator needs).
            evolution_scale_m: Positive length scale in metres used to
                non-dimensionalize every term. Defaults to the Earth radius.

        Returns:
            Complex tensor shaped ``(..., space.dim**2, space.dim**2)``.

        Raises:
            ValueError: If any active term's or dissipator's ``validate``
                fails.
        """
        H = self.generator_reduced(environment, evolution_scale_m=evolution_scale_m)
        liouvillian = commutator_superoperator(H)
        for L in self._assemble_lindblad_operators(environment, self.antinu, evolution_scale_m):
            liouvillian = liouvillian + dissipator_superoperator(L)
        return liouvillian

    def liouvillian_flavour(
        self,
        environment: PropagationEnvironment,
        *,
        evolution_scale_m: TensorLike = R_E,
    ) -> torch.Tensor:
        """Build the full flavour-basis Liouvillian.

        Dresses ``liouvillian_reduced``'s output with the Liouville-space
        form of the same block-diagonal outer rotation
        ``generator_flavour`` uses (``O_total kron O_total.conj()`` -- the
        general statement that a basis change ``O`` on the underlying
        Hilbert space induces ``O kron O.conj()`` on Liouville space,
        applying equally to the unitary and dissipative pieces alike, since
        every Lindblad operator here was built in the same reduced basis
        the Hamiltonian was).

        Args:
            environment: See ``liouvillian_reduced``.
            evolution_scale_m: Positive length scale in metres used to
                non-dimensionalize every term. Defaults to the Earth radius.

        Returns:
            Complex tensor shaped ``(..., space.dim**2, space.dim**2)``.
        """
        liouvillian_reduced_value = self.liouvillian_reduced(environment, evolution_scale_m=evolution_scale_m)
        O = self._outer_block_total()
        O_liouville = kron_batched(O, O.conj())
        return O_liouville @ liouvillian_reduced_value @ O_liouville.conj().transpose(-2, -1)


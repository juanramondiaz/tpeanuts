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

"""The sterile neutral-current matter term.

The neutral-current potential ``V_NC`` is common to every standard flavour
(lepton universality), so it is a pure overall phase whenever every flavour
in play is a standard (non-sterile) one -- ``tpeanuts.physics.common.
generator`` never adds it. Once sterile flavours are present, that is no
longer true: a sterile state is an SM gauge singlet and therefore does not
receive ``V_NC`` at all, so once the common phase shared by the standard
flavours is removed, a genuine relative potential ``-V_NC`` survives on
every sterile diagonal entry (all ``N`` of them alike -- lepton universality
makes them indistinguishable from the neutral-current potential's point of
view).

Unlike the charged-current term, this term's flavour-basis "direction"
matrix (a projector onto the sterile flavours) does not commute with the
mixing matrix's outer block ``O`` (the standard-sterile rotations rotate a
sterile basis vector into a combination that includes standard flavours),
so it is built directly in the flavour basis and rotated down into the
reduced basis via ``tpeanuts.physics.common.generator.rotate_to_reduced``,
rather than added as a reduced-basis fast path the way the charged-current
term is.

Module contents:
    SterileNCTerm
        Stateless ``tpeanuts.physics.common.terms.GeneratorTerm``,
        contributing ``-V_NC`` on every sterile diagonal entry, in the
        reduced basis. ``kind = TermKind.ADDITIVE``, ``replaces = None``: it
        never overlaps with anything else (nothing else touches the sterile
        diagonal), so it always simply adds.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from tpeanuts.physics.common.generator import rotate_to_reduced
from tpeanuts.physics.common.mixing import MixingMatrix
from tpeanuts.physics.common.potential import matter_potential_nc
from tpeanuts.physics.common.terms import PropagationEnvironment, TermKind, embed_primary_block
from tpeanuts.util.constant import R_E
from tpeanuts.util.tensor import TensorLike, cdtype_from_real


class SterileNCTerm:
    """The sterile neutral-current matter term, ``-V_NC`` on every sterile diagonal entry.

    Stateless (no construction parameters): unlike NSI, there is no
    per-scenario configuration here beyond "is a sterile extension present",
    which ``validate`` reads directly from ``mixing.space``.

    Attributes:
        kind: Always ``TermKind.ADDITIVE``.
        replaces: Always ``None``.

    Examples:
        >>> from tpeanuts.physics.extensions import SterileNCTerm
        >>> term = SterileNCTerm()   # stateless: whether it applies is read from the model's flavour space
        >>> term.kind.name, term.replaces
        ('ADDITIVE', None)
    """

    kind = TermKind.ADDITIVE
    replaces: Optional[str] = None

    @staticmethod
    def _sterile_projector(
        n_standard: int, n_sterile: int, *, device: torch.device, cdtype: torch.dtype
    ) -> torch.Tensor:
        """Build ``diag(0, ..., 0, 1, ..., 1)``: zero on every standard flavour, one on every sterile one.

        A staticmethod (not a module function): specifically the projector
        this class's own contribution needs, not a generic linear-algebra
        utility.

        Args:
            n_standard: Number of leading zero (standard-flavour) diagonal
                entries.
            n_sterile: Number of trailing one (sterile-flavour) diagonal
                entries.
            device: Device for the returned tensor.
            cdtype: Complex dtype for the returned tensor.

        Returns:
            Complex tensor shaped
            ``(n_standard + n_sterile, n_standard + n_sterile)``.
        """
        diag = torch.cat(
            [
                torch.zeros(n_standard, device=device, dtype=cdtype),
                torch.ones(n_sterile, device=device, dtype=cdtype),
            ]
        )
        return torch.diag_embed(diag)

    def validate(self, mixing: MixingMatrix, environment: PropagationEnvironment) -> None:
        """Check that ``mixing`` is sterile-extended and ``environment`` carries neutron density.

        Args:
            mixing: The MixingMatrix the generator is being built for.
            environment: The PropagationEnvironment this call was given.

        Raises:
            ValueError: If ``mixing.space.n_sterile == 0`` -- this term has
                no meaning (and no effect) without a sterile extension, so a
                caller reaching for it there almost certainly has a bug,
                rather than something safe to silently no-op. Or if
                ``environment.n_n_mol_cm3`` is None.
        """
        if not mixing.space.is_sterile_extended:
            raise ValueError(
                "SterileNCTerm requires a sterile-extended FlavourSpace "
                "(space.n_sterile > 0); it has no physical effect for the "
                "plain standard-flavour case (V_NC is an unobservable common "
                "phase there), so using it is almost certainly a caller "
                "bug rather than something to silently no-op."
            )
        if environment.n_n_mol_cm3 is None:
            raise ValueError(
                "SterileNCTerm requires environment.n_n_mol_cm3 (neutron "
                "density); see this class's module docstring."
            )

    def contribute_reduced(
        self,
        mixing: MixingMatrix,
        environment: PropagationEnvironment,
        *,
        antinu: Union[bool, torch.Tensor] = False,
        evolution_scale_m: TensorLike = R_E,
    ) -> torch.Tensor:
        """Build the reduced-basis sterile neutral-current term.

        Args:
            mixing: Built MixingMatrix; must be sterile-extended
                (``mixing.space.n_sterile > 0``, checked by ``validate``).
            environment: Neutrino/medium data; only ``n_n_mol_cm3`` is read.
            antinu: Bool or boolean tensor selecting the antineutrino
                convention (flips the sign of ``V_NC`` and complex-conjugates
                the outer block used to rotate the projector).
            evolution_scale_m: Positive length scale in metres, matching
                whatever every other active term uses.

        Returns:
            Complex Hermitian tensor shaped
            ``(..., mixing.space.n_flavour, mixing.space.n_flavour)``.
        """
        self.validate(mixing, environment)

        context = mixing.context
        cdtype = cdtype_from_real(context.dtype)

        assert environment.n_n_mol_cm3 is not None  # checked in validate()
        V_nc = matter_potential_nc(
            environment.n_n_mol_cm3, antinu, evolution_scale_m=evolution_scale_m, context=context
        )
        Q_flavour = self._sterile_projector(
            mixing.space.n_standard, mixing.space.n_sterile, device=context.device, cdtype=cdtype
        )
        O = mixing.outer_block(antinu=antinu)
        Q_reduced = rotate_to_reduced(Q_flavour, O)

        H = -V_nc.to(dtype=cdtype)[..., None, None] * Q_reduced
        return embed_primary_block(H, mixing.space)

    def direction_reduced(
        self,
        mixing: MixingMatrix,
        environment: PropagationEnvironment,
        *,
        antinu: Union[bool, torch.Tensor] = False,
    ) -> torch.Tensor:
        """Return this term's V-independent direction matrix ``-diag(0,...,0,1,...,1)``, rotated and embedded.

        Optional ``GeneratorTerm`` extension consumed by
        ``tpeanuts.physics.evolution.perturbative.perturbative``'s first-order perturbative
        correction (not part of the required ``GeneratorTerm`` protocol
        surface). Pairs with ``tpeanuts.physics.common.potential.
        matter_potential_nc(environment.n_n_mol_cm3, antinu,
        evolution_scale_m=...)``: multiplying the two together reproduces
        ``contribute_reduced`` exactly.

        Args:
            mixing: Built MixingMatrix; must be sterile-extended (checked
                by ``validate``).
            environment: Neutrino/medium data; only used by ``validate``.
            antinu: Bool or boolean tensor selecting the antineutrino
                convention (complex-conjugates the outer block used to
                rotate the projector), matching ``contribute_reduced``'s
                own convention.

        Returns:
            Complex tensor shaped ``(..., mixing.space.dim,
            mixing.space.dim)``.
        """
        self.validate(mixing, environment)

        context = mixing.context
        cdtype = cdtype_from_real(context.dtype)
        Q_flavour = self._sterile_projector(
            mixing.space.n_standard, mixing.space.n_sterile, device=context.device, cdtype=cdtype
        )
        O = mixing.outer_block(antinu=antinu)
        Q_reduced = rotate_to_reduced(Q_flavour, O)
        return embed_primary_block(-Q_reduced, mixing.space)

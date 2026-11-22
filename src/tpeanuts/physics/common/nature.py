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

"""Dirac vs Majorana neutrino nature.

- Every propagation model needs a ``Nature``: Dirac or Majorana

    -- ``Dirac`` is simply the *trivial* choice (it changes nothing relative
        to a  mixing matrix with no ``Nature`` concept at all), so it plays
        the same role that "off" plays for the optional extensions.

- ``Nature`` affects two things: whether an extra diagonal phase factor is
    appended to a mixing matrix's mass-eigenstate side (``phase_matrix``),
    and, once a doubled (``n_helicity == 2``) generator exists, what the
    "companion helicity block" looks like (``companion_helicity_terms``):

    -- for Dirac that companion is a genuine gauge-singlet ``nu_R``
        (kinetic term only, same ``antinu``).

    -- for Majorana it *is* the antiparticle sector: the full generator
        (kinetic, matter potential, and whatever extensions are active)
        evaluated at the opposite ``antinu``.

- It is also the natural extension point for a magnetic-moment extension,
    whose allowed transition-moment matrix structure depends on this same
    distinction:

    -- for Dirac, ``mu`` is a fully general complex matrix (``nu_L`` and
        ``nu_R`` are independent fields, so nothing relates ``mu_ij`` to
        ``mu_ji``) -- *not* Hermitian, there is no such constraint here.

    -- For Majorana, CPT forces ``mu`` to be purely antisymmetric
        (``mu_ij = -mu_ji``, zero diagonal -- a Majorana particle cannot
        have a moment coupling itself to itself, only transition moments
        between distinct massive states).

Module contents:
    Nature
        Abstract strategy distinguishing Dirac from Majorana neutrinos.
    Dirac
        Trivial nature: contributes no extra mixing-matrix phase factor;
        companion helicity block is kinetic-only, same antinu.
    Majorana
        Attaches ``n_flavour - 1`` physical CP-violating phases to the
        mixing matrix's mass-eigenstate side; companion helicity block is
        the full generator at the opposite antinu.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch

from tpeanuts.physics.common.space import FlavourSpace
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import TensorLike, as_tensor, cdtype_from_real, negate_antinu


class Nature(abc.ABC):
    """Strategy object distinguishing Dirac from Majorana neutrinos."""

    @property
    @abc.abstractmethod
    def is_majorana(self) -> bool:
        """Whether this nature is Majorana (True) or Dirac (False)."""
        raise NotImplementedError

    @abc.abstractmethod
    def phase_matrix(
        self,
        space: FlavourSpace,
        context: RuntimeContext,
    ) -> Optional[torch.Tensor]:
        """Return the extra mixing-matrix phase factor for this nature, or None.

        Args:
            space: FlavourSpace whose ``n_flavour`` sizes the returned
                matrix (helicity doubling, if any, is irrelevant here: this
                factor lives entirely on the flavour/mass side).
            context: RuntimeContext for the returned tensor.

        Returns:
            None for Dirac (no extra factor -- the mixing matrix is used
            exactly as built from its rotations alone). For Majorana, a
            complex diagonal tensor shaped ``(..., n_flavour, n_flavour)``,
            meant to be right-multiplied onto the reduced mixing matrix.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def companion_helicity_terms(
        self,
        primary_antinu: Union[bool, torch.Tensor],
    ) -> tuple[Union[bool, torch.Tensor], bool]:
        """Decide the "companion helicity" block's own generator, for a doubled (n_helicity=2) space.

        A magnetic-moment extension couples two helicity blocks: a
        "primary" one (ordinary flavour physics -- kinetic, matter
        potential, whatever extensions are active, evaluated at
        ``primary_antinu``) and a "companion" one, whose own internal
        physics (before the magnetic-moment off-diagonal coupling is even
        considered) depends entirely on nature -- see this module's
        docstring.

        Args:
            primary_antinu: The antinu convention the primary block uses
                (``PropagationModel.antinu``).

        Returns:
            ``(companion_antinu, include_matter_terms)``:
            ``companion_antinu`` is the antinu convention the companion
            block's own generator should use; ``include_matter_terms``
            is whether that generator includes anything beyond the kinetic
            term (matter potential, NSI, sterile NC, ...).
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Dirac(Nature):
    """Dirac neutrinos: no extra Majorana phase factor.

    The trivial/default nature: every mixing matrix built with ``Dirac()``
    is exactly what it would be with no ``Nature`` concept at all.

    Examples:
        >>> import tpeanuts as tp
        >>> tp.Dirac()
        Dirac()
    """

    @property
    def is_majorana(self) -> bool:
        """Always False."""
        return False

    def phase_matrix(
        self,
        space: FlavourSpace,
        context: RuntimeContext,
    ) -> Optional[torch.Tensor]:
        """Always None: Dirac neutrinos add no extra mixing-matrix phase."""
        return None

    def companion_helicity_terms(
        self,
        primary_antinu: Union[bool, torch.Tensor],
    ) -> tuple[Union[bool, torch.Tensor], bool]:
        """The companion block is a genuine gauge-singlet nu_R: kinetic-only, same antinu.

        nu_L and nu_R are the same Dirac particle, not conjugate fields, so
        the companion block uses the *same* antinu convention as the
        primary block (unlike Majorana) -- and, being a true gauge singlet,
        it feels no matter potential, NSI, or sterile neutral-current term
        at all.

        Args:
            primary_antinu: The antinu convention the primary block uses.

        Returns:
            ``(primary_antinu, False)``.
        """
        return primary_antinu, False


@dataclass(frozen=True)
class Majorana(Nature):
    """Majorana neutrinos: ``n_flavour - 1`` extra physical CP-violating phases.

    Attributes:
        phases: Real tensor shaped ``(..., n_flavour - 1)``: the physical
            Majorana phases carried by every mass eigenstate except the
            first (the phase-convention reference, as usual). Provably
            inert for every oscillation probability -- a rephasing-
            invariance argument: each phase contributed by ``U_bi`` cancels
            exactly against its complex conjugate contributed by ``U*_ai``,
            for every ``i``, independent of the propagation method. It
            matters only for observables that read the mixing matrix
            directly, such as the effective Majorana mass governing
            neutrinoless double-beta decay.
        context: RuntimeContext the phases are stored on.

    Examples:
        >>> import math
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> phases = torch.tensor([0.4, 1.1], dtype=torch.float64)   # the two Majorana phases, in radians
        >>> nature = tp.Majorana(phases=phases, context=context)
        >>> nature.phases.shape
        torch.Size([2])
    """

    phases: torch.Tensor
    context: RuntimeContext

    if TYPE_CHECKING:  # the runtime __init__ is the dataclass one; tensor fields accept any TensorLike

        def __init__(
            self,
            phases: TensorLike,
            context: RuntimeContext,
        ) -> None: ...

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "phases",
            as_tensor(self.phases, device=self.context.device, dtype=self.context.dtype),
        )

    @property
    def is_majorana(self) -> bool:
        """Always True."""
        return True

    def phase_matrix(
        self,
        space: FlavourSpace,
        context: RuntimeContext,
    ) -> torch.Tensor:
        """Build ``diag(1, exp(i phases[0]/2), ..., exp(i phases[-1]/2))``.

        Args:
            space: FlavourSpace whose ``n_flavour`` sizes the returned
                matrix.
            context: RuntimeContext for the returned tensor.

        Returns:
            Complex diagonal tensor shaped ``(..., n_flavour, n_flavour)``.

        Raises:
            ValueError: If ``phases``'s last dimension is not
                ``space.n_flavour - 1``.
        """
        n = space.n_flavour
        expected = n - 1
        if self.phases.shape[-1] != expected:
            raise ValueError(
                f"phases must have last dimension {expected} "
                f"(space.n_flavour - 1), got {self.phases.shape[-1]}."
            )

        cdtype = cdtype_from_real(context.dtype)
        phases = self.phases.to(device=context.device, dtype=context.dtype)
        batch_shape = phases.shape[:-1]

        reference = torch.ones(*batch_shape, 1, device=context.device, dtype=cdtype)
        half_phase_factor = torch.exp(1j * phases.to(dtype=cdtype) / 2)
        diag = torch.cat([reference, half_phase_factor], dim=-1)
        return torch.diag_embed(diag)

    def companion_helicity_terms(
        self,
        primary_antinu: Union[bool, torch.Tensor],
    ) -> tuple[Union[bool, torch.Tensor], bool]:
        """The companion block *is* the antiparticle sector: full generator, opposite antinu.

        A Majorana neutrino equals its own antiparticle (CPT), so the
        "other helicity" content of a given massive state is exactly what
        the antinu-flipped convention already computes -- and it interacts
        exactly as the antiparticle does: it feels the matter potential
        (opposite sign, already handled by every term's own antinu
        awareness), NSI, and the sterile neutral-current term alike,
        whichever are active.

        Args:
            primary_antinu: The antinu convention the primary block uses.

        Returns:
            ``(negate_antinu(primary_antinu), True)``.
        """
        return negate_antinu(primary_antinu), True

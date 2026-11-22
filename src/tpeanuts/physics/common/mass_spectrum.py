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

"""Mass-squared splittings, generalized to the 3+N sterile extension.

A single ``MassSpectrum`` class serves the plain Standard Model and every
3+N sterile extension alike, sized by a ``FlavourSpace`` instead of forking
into separate classes: ``space.n_sterile == 0`` simply means
``DeltamSq_sterile`` is absent, and ``difference_vector`` returns the
3-component standard vector unchanged.

Module contents:
    MassSpectrum
        Standard-sector splittings (``DeltamSq21``, ``DeltamSq3l``) plus an
        optional per-sterile-flavour splitting vector, with the reduced
        standard vector, the full (possibly sterile-extended) vector, and
        the absolute mass-squared vector needed by mass-scale-dependent
        observables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, cast

import torch

from tpeanuts.physics.common.space import FlavourSpace
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import TensorLike, as_tensor


@dataclass(frozen=True)
class MassSpectrum:
    """Mass-squared splittings for a given FlavourSpace.

    Attributes:
        space: FlavourSpace this spectrum is sized for.
            ``space.n_sterile > 0`` requires ``DeltamSq_sterile``;
            ``space.n_sterile == 0`` requires it to be omitted.
        DeltamSq21: Solar mass-squared splitting Delta m^2_21, in eV^2. Sets
            the oscillation frequency of the "slow", solar-driven term.
        DeltamSq3l: Atmospheric mass-squared splitting Delta m^2_3l, in
            eV^2 (Delta m^2_31 for normal ordering, Delta m^2_32 for
            inverted ordering, by sign convention). Sets the oscillation
            frequency of the "fast", atmospheric-driven term, and its sign
            selects the mass ordering.
        context: RuntimeContext the splittings are stored on.
        DeltamSq_sterile: Sterile mass-squared splittings
            Delta m^2_{4+k,1}, k = 0, ..., n_sterile - 1, in eV^2, shaped
            ``(..., n_sterile)`` -- always relative to ``m_1`` specifically,
            by the standard 3+N literature convention (see
            ``_sterile_relative_to_base_reference`` for how that is
            reconciled with ``difference_vector_base``'s own floating
            reference for inverted-ordering spectra). Like
            ``DeltamSq21``/``DeltamSq3l``, these do not enter any
            mixing-matrix rotation -- only the standard-sterile mixing angles
            do. None iff ``space.n_sterile == 0``.

    Raises:
        ValueError: If ``DeltamSq_sterile`` is missing when
            ``space.n_sterile > 0``, given when ``space.n_sterile == 0``, or
            its last dimension does not match ``space.n_sterile``.

    Examples:
        >>> import math
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> masses = tp.MassSpectrum(space=tp.FlavourSpace(), DeltamSq21=7.4e-5, DeltamSq3l=2.5e-3, context=context)
        >>> m_squared = masses.absolute_mass_squared_vector(0.01)   # eV^2, for a lightest mass of 0.01 eV
        >>> [f"{float(x):.3e}" for x in m_squared]
        ['1.000e-04', '1.740e-04', '2.600e-03']
    """

    space: FlavourSpace
    DeltamSq21: torch.Tensor
    DeltamSq3l: torch.Tensor
    context: RuntimeContext
    DeltamSq_sterile: Optional[torch.Tensor] = None

    if TYPE_CHECKING:  # the runtime __init__ is the dataclass one; tensor fields accept any TensorLike

        def __init__(
            self,
            space: FlavourSpace,
            DeltamSq21: TensorLike,
            DeltamSq3l: TensorLike,
            context: RuntimeContext,
            DeltamSq_sterile: Optional[TensorLike] = ...,
        ) -> None: ...

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "DeltamSq21",
            as_tensor(self.DeltamSq21, device=self.context.device, dtype=self.context.dtype),
        )
        object.__setattr__(
            self,
            "DeltamSq3l",
            as_tensor(self.DeltamSq3l, device=self.context.device, dtype=self.context.dtype),
        )

        if self.space.n_sterile > 0:
            if self.DeltamSq_sterile is None:
                raise ValueError(
                    f"space has {self.space.n_sterile} sterile flavour(s); "
                    "DeltamSq_sterile is required."
                )
            sterile = as_tensor(
                self.DeltamSq_sterile,
                device=self.context.device,
                dtype=self.context.dtype,
            )
            if sterile.shape[-1] != self.space.n_sterile:
                raise ValueError(
                    f"DeltamSq_sterile must have last dimension "
                    f"{self.space.n_sterile} (space.n_sterile), got "
                    f"{sterile.shape[-1]}."
                )
            object.__setattr__(self, "DeltamSq_sterile", sterile)
        elif self.DeltamSq_sterile is not None:
            raise ValueError(
                "space has no sterile flavours (n_sterile == 0); "
                "DeltamSq_sterile must be omitted."
            )

    def difference_vector_base(self) -> torch.Tensor:
        """Build the reduced 3-component standard-sector mass-squared vector.

        The overall additive constant (a common phase shared by every
        eigenvalue) is physically irrelevant to oscillation probabilities,
        so it is fixed by convention: whichever eigenstate each mass
        ordering treats as the reference is placed at zero.

        Returns:
            Tensor shaped ``(..., 3)``: ``(0, DeltamSq21, DeltamSq3l)`` for
            normal ordering (``DeltamSq3l > 0``), or
            ``(-DeltamSq21, 0, DeltamSq3l)`` for inverted ordering
            (``DeltamSq3l < 0``).
        """
        dm21 = self.DeltamSq21
        dm3l = self.DeltamSq3l
        zeros = torch.zeros_like(dm21)

        normal = torch.stack([zeros, dm21, dm3l], dim=-1)
        inverted = torch.stack([-dm21, zeros, dm3l], dim=-1)

        if dm3l.ndim == 0:
            return normal if dm3l.item() > 0 else inverted
        return torch.where((dm3l > 0)[..., None], normal, inverted)

    def _sterile_relative_to_base_reference(self) -> Optional[torch.Tensor]:
        """Re-anchor DeltamSq_sterile onto difference_vector_base's floating reference.

        ``DeltamSq_sterile`` is defined as ``Delta m^2_{4+k,1}`` -- always
        relative to ``m_1``, by the standard 3+N literature convention.
        ``difference_vector_base`` is *not* always relative to ``m_1``,
        though: its zero sits at ``m_1^2`` for normal ordering but at
        ``m_2^2`` for inverted ordering. Concatenating ``DeltamSq_sterile``
        onto it unchanged would silently mix two different references for
        inverted-ordering spectra, shifting every sterile splitting by
        ``DeltamSq21`` relative to what its name promises -- negligible
        next to typical eV-scale sterile splittings, but not zero. This
        method removes that offset by subtracting ``DeltamSq21`` whenever
        ``difference_vector_base``'s reference is ``m_2`` instead of
        ``m_1`` (inverted ordering), and leaves it unchanged for normal
        ordering, where the two references already coincide.

        Returns:
            None if ``space.n_sterile == 0``. Otherwise a tensor shaped
            ``(..., n_sterile)``, ``DeltamSq_sterile`` expressed relative to
            whichever reference ``difference_vector_base`` uses for each
            batch entry's mass ordering.
        """
        if self.space.n_sterile == 0:
            return None

        dm21 = self.DeltamSq21
        dm3l = self.DeltamSq3l
        if dm3l.ndim == 0:
            offset = torch.zeros_like(dm21) if dm3l.item() > 0 else dm21
        else:
            offset = torch.where(dm3l > 0, torch.zeros_like(dm21), dm21)

        sterile = cast(torch.Tensor, self.DeltamSq_sterile)
        return sterile - offset[..., None]

    def difference_vector(self) -> torch.Tensor:
        """Build the full mass-squared vector, sized to ``space.n_flavour``.

        Returns:
            Tensor shaped ``(..., space.n_flavour)``: the standard
            3-component vector from ``difference_vector_base``, extended
            with ``DeltamSq_sterile`` (re-anchored onto the same floating
            reference via ``_sterile_relative_to_base_reference``) when
            ``space.n_sterile > 0``.
        """
        base = self.difference_vector_base()
        if self.space.n_sterile == 0:
            return base

        sterile = cast(torch.Tensor, self._sterile_relative_to_base_reference())
        sterile = sterile.expand(*base.shape[:-1], self.space.n_sterile)
        return torch.cat([base, sterile], dim=-1)

    def absolute_mass_squared_vector(self, m_lightest_eV: TensorLike) -> torch.Tensor:
        """Build the absolute mass-squared vector, sized to ``space.n_flavour``.

        No oscillation probability computed anywhere in this project needs
        the absolute mass scale -- ``difference_vector`` (relative
        splittings) is the only input the kinetic generator ever consumes,
        by the same rephasing-invariance argument that makes Majorana
        phases unobservable (``tpeanuts.physics.common.nature.Majorana``).
        This method exists for observables that DO depend on the absolute
        mass scale, such as the effective Majorana mass or the sum of
        neutrino masses -- equally meaningful for Dirac neutrinos, so it
        stays generic here rather than living alongside Majorana-specific
        code. Includes the sterile masses (via ``difference_vector``, not
        ``difference_vector_base``) when ``space.n_sterile > 0``, correctly
        re-anchored onto the same reference as the standard block -- see
        ``_sterile_relative_to_base_reference``.

        Args:
            m_lightest_eV: Mass of the lightest standard eigenstate, in eV:
                ``m_1`` for normal ordering (``DeltamSq3l > 0``), ``m_3``
                for inverted ordering (``DeltamSq3l < 0``). Always the
                lightest *standard* state, even when ``space.n_sterile > 0``:
                every 3+N sterile scenario in this project has sterile
                states far heavier than the standard sector.

        Returns:
            Tensor shaped ``(..., space.n_flavour)``:
            ``(m_1^2, m_2^2, m_3^2)`` for the plain 3-flavour case, extended
            with ``(m_4^2, ..., m_{3+N}^2)`` when ``space.n_sterile == N``
            is positive, in eV^2.
        """
        dm3l = self.DeltamSq3l
        m_lightest_sq = (
            as_tensor(m_lightest_eV, device=self.context.device, dtype=self.context.dtype) ** 2
        )
        diff = self.difference_vector()

        # diff is relative to whichever eigenstate difference_vector_base
        # places at zero: m_1^2 for normal ordering, m_2^2 for inverted
        # ordering (the sterile block, if any, has already been re-anchored
        # onto that same reference by difference_vector). m_ref_sq is that
        # reference mass-squared, expressed in terms of the supplied
        # lightest mass for each ordering.
        if dm3l.ndim == 0:
            m_ref_sq = m_lightest_sq if dm3l.item() > 0 else m_lightest_sq - dm3l
        else:
            m_ref_sq = torch.where(dm3l > 0, m_lightest_sq, m_lightest_sq - dm3l)

        return m_ref_sq[..., None] + diff

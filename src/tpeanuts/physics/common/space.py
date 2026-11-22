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

"""Dimensional bookkeeping for the neutrino state space.

Every other physics module (mixing matrices, generators, evolvers,
probabilities) needs to know the size of the Hilbert space it is working in,
without needing to know *why* it has that size. ``FlavourSpace`` is the one
small, immutable value object that answers that question, so a new sterile
count or a magnetic-moment extension changes exactly one thing (this
object's fields) rather than a shape computed ad hoc in every consumer.

Three independent axes compose into the final dimension:
    - ``n_standard`` (fixed at 3: electron, muon, tau) -- the Standard Model
      sector, always present.
    - ``n_sterile`` (0, 1, ..., N) -- the 3+N sterile extension.
      ``n_sterile=0`` recovers the plain Standard Model exactly;
      ``n_sterile=1`` is the familiar 3+1 case.
    - ``n_helicity`` (1 or 2)
        -- 1 for every scenario without helicity-flipping physics (the
           ordinary oscillation/NSI/sterile/decoherence case);
        -- 2 only when a magnetic-moment/spin-flavor-precession extension is
            active and the state space must represent both helicities jointly,
            since that is exactly the physics that couples what would
            otherwise be two independent classical branches
            (a neutrino converting into an antineutrino).
            Plain SM/sterile/NSI/decoherence runs never pay for this doubled
            dimension.

Module contents:
    FlavourSpace
        Immutable (n_standard, n_sterile, n_helicity) triple, with the
        derived total dimension and a couple of convenience flags every
        consumer needs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlavourSpace:
    """Describes the dimension and structure of the neutrino Hilbert space.

    Attributes:
        n_standard: Number of Standard Model (weak-interacting) flavours.
            Fixed at exactly 3 for every scenario in this project (electron,
            muon, tau) -- kept as an explicit field rather than a hardcoded
            constant so that every downstream shape computation reads it
            from one place instead of assuming it, *not* because this
            project actually supports a different value. Every consumer
            that builds a mixing matrix (``tpeanuts.physics.common.mixing.
            MixingMatrix.R12/R13/R23/Delta``, and critically
            ``_sterile_rotation``, which hardcodes the first sterile column
            as index 3) assumes exactly 3 standard flavours, not merely "a
            positive number" of them -- see this field's own validation
            below.
        n_sterile: Number of sterile flavours added on top of the 3
            standard ones. 0 recovers the plain Standard Model exactly, 1 is
            the familiar 3+1 extension, and any non-negative N is the
            general 3+N extension -- nothing in this class hardcodes 3+1.
        n_helicity: 1 for the plain flavour-only space (every scenario
            without helicity-flipping physics); 2 when a magnetic-moment
            extension requires representing both helicities in one joint
            state (see the module docstring for why).

    Raises:
        ValueError: If ``n_standard`` is not exactly 3, ``n_sterile`` is
            negative, or ``n_helicity`` is not 1 or 2.

    Examples:
        >>> import tpeanuts as tp
        >>> tp.FlavourSpace().dim
        3
        >>> space = tp.FlavourSpace(n_sterile=1)
        >>> space.n_flavour, space.is_sterile_extended
        (4, True)
    """

    n_standard: int = 3
    n_sterile: int = 0
    n_helicity: int = 1

    def __post_init__(self) -> None:
        if self.n_standard != 3:
            raise ValueError(
                f"n_standard must be exactly 3, got {self.n_standard}. This field "
                "exists for downstream code to read rather than hardcode, not "
                "because a different standard-sector size is actually supported "
                "-- every mixing-matrix rotation (R12/R13/R23/Delta) and the "
                "first sterile column index (mixing.py's _sterile_rotation, "
                "hardcoded to 3 + sterile_index) assume exactly 3 standard "
                "flavours. A different n_standard would previously pass this "
                "check yet silently build a wrong, still-unitary mixing matrix "
                "(the would-be 4th+ standard flavour gets no rotation of its "
                "own and collides with the first sterile column) rather than "
                "fail -- rejected here instead."
            )
        if self.n_sterile < 0:
            raise ValueError(f"n_sterile must be a non-negative integer, got {self.n_sterile}.")
        if self.n_helicity not in (1, 2):
            raise ValueError(f"n_helicity must be 1 or 2, got {self.n_helicity}.")

    @property
    def n_flavour(self) -> int:
        """Total number of flavour states (standard + sterile), before helicity doubling.

        Returns:
            ``n_standard + n_sterile``.
        """
        return self.n_standard + self.n_sterile

    @property
    def dim(self) -> int:
        """Total Hilbert-space dimension consumed by state vectors/density matrices.

        Returns:
            ``n_flavour * n_helicity``.
        """
        return self.n_flavour * self.n_helicity

    @property
    def is_sterile_extended(self) -> bool:
        """Whether one or more sterile flavours are present.

        Returns:
            True iff ``n_sterile > 0``.
        """
        return self.n_sterile > 0

    @property
    def is_helicity_extended(self) -> bool:
        """Whether the space represents both helicities jointly.

        Returns:
            True iff ``n_helicity == 2``.
        """
        return self.n_helicity == 2

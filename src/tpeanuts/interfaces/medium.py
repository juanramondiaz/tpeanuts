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

"""What a medium does to a neutrino in transit: a sequence of homogeneous segments.

Shaped to match ``tpeanuts.physics.evolution.numerical.numerical.
evolve_path_numerical``'s own segment convention exactly (``H_segments:
(..., N, d, d)``, ``dx_segments: (..., N)``) -- a trailing segment axis,
not a scalar, even for the trivial single-segment case (e.g.
``tpeanuts.medium.vacuum``), so a future multi-segment medium (an Earth-
or Sun-density profile) needs no change to this contract, only to which
concrete class builds it.

``method``/``perturbative_profile`` (both optional, both default to
reproducing today's numerical-only behaviour exactly) are how a medium
opts a segment into ``tpeanuts.physics.evolution.perturbative.analytic``'s/
``.perturbative``'s closed-form evolution instead of ``numerical``'s
``matrix_exp`` -- see ``tpeanuts.physics.evolution``'s own module
docstring for why this lives here (data the medium carries) rather than
as a branch in any of ``Source``/``Medium``/``PropagationModel``.

Module contents:
    MediumProfile
        Frozen dataclass: per-segment ``length_m``, ``n_e_mol_cm3``, the
        optional ``n_n_mol_cm3``/``B_field_T`` only a sterile-NC/NSI-neutron
        or magnetic-moment extension needs, and the optional
        ``method``/``perturbative_profile`` evolver selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from tpeanuts.physics.evolution import PropagationMethod
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import TensorLike, as_tensor

if TYPE_CHECKING:
    from tpeanuts.physics.evolution.perturbative.perturbative import PerturbativeProfile


@dataclass(frozen=True)
class MediumProfile:
    """Per-segment medium samples along a propagation path.

    Attributes:
        length_m: Physical length of each segment, in metres, shaped
            ``(..., n_segments)``. A single-segment medium (e.g. vacuum) has
            ``n_segments == 1``, not a scalar with no segment axis at all.
        n_e_mol_cm3: Electron density per segment, in mol/cm^3, shaped
            ``(..., n_segments)`` (or broadcastable against ``length_m``).
        context: RuntimeContext this profile's tensors are stored on.
        n_n_mol_cm3: Optional neutron density per segment -- only needed by
            the sterile neutral-current term or NSI's neutron coupling.
            None (the default) reproduces every existing extension's own
            "neutron density not supplied" behaviour exactly.
        B_field_T: Optional magnetic field magnitude per segment -- only
            needed by the magnetic-moment extension. None by default, same
            reasoning as ``n_n_mol_cm3``.
        method: Which unitary evolver
            (``tpeanuts.physics.evolution.PropagationMethod``) this
            profile's segments should be evolved with.
            ``PropagationMethod.NUMERICAL`` (the default) reproduces every
            scenario built before this field existed, unconditionally.
            ``PropagationMethod.ANALYTIC`` opts into the exact closed-form
            zero-order evolver; ``ANALYTIC_PERTURBATIVE`` additionally
            applies ``perturbative_profile``'s first-order correction.
            Both are only valid for ``dim in {3, 4}`` and no active
            dissipators -- see ``tpeanuts.physics.evolution``'s own module
            docstring.
        perturbative_profile: Optional, opaque, duck-typed density-profile
            object (see ``tpeanuts.physics.evolution.perturbative.profile``) exposing
            within-segment density variation, consumed only by
            ``PropagationMethod.ANALYTIC_PERTURBATIVE`` (required there --
            ``tpeanuts.physics.evolution.dispatch.evolve_path_unitary`` raises
            if it is missing). None (the default) is required for every
            other ``method`` value -- passing one there would be silently
            ignored, so ``evolve_path`` rejects that combination
            explicitly instead.

    A ``length_m`` entry of exactly ``0`` is allowed (only a strictly
    *negative* one is rejected): a zero-length segment is a deliberate,
    physically inert no-op (``matrix_exp(H*0) == I``, exactly, regardless
    of that segment's own density/``perturbative_profile`` values --
    ``tpeanuts.physics.evolution.perturbative.profile.EvenPowerProfile``
    itself already special-cases a zero-length segment to an exactly
    -zero average density for the same reason, not a stray ``0/0``) --
    the pattern ``tpeanuts.medium.earth.earth.EarthMedium`` uses to pad
    a variable, per-trajectory segment count (``ANALYTIC_PERTURBATIVE``'s
    own PREM-shell crossings) up to one fixed shape shared across a
    batch of nadir angles, so genuinely different segment counts do not
    force ``eta_rad`` to stay scalar.

    Raises:
        ValueError: If ``length_m`` has any strictly negative entry.

    Examples:
        >>> import math
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> from tpeanuts.interfaces import MediumProfile
        >>> profile = MediumProfile(   # two segments of 1000 km and 500 km
        ...     length_m=[1.0e6, 5.0e5], n_e_mol_cm3=[1.0, 2.0], context=context)
        >>> profile.n_segments, float(profile.length_m.sum())
        (2, 1500000.0)
    """

    length_m: torch.Tensor
    n_e_mol_cm3: torch.Tensor
    context: RuntimeContext
    n_n_mol_cm3: Optional[torch.Tensor] = None
    B_field_T: Optional[torch.Tensor] = None
    method: PropagationMethod = PropagationMethod.NUMERICAL
    perturbative_profile: Optional[PerturbativeProfile] = None

    if TYPE_CHECKING:  # the runtime __init__ is the dataclass one; tensor fields accept any TensorLike

        def __init__(
            self,
            length_m: TensorLike,
            n_e_mol_cm3: TensorLike,
            context: RuntimeContext,
            n_n_mol_cm3: Optional[TensorLike] = ...,
            B_field_T: Optional[TensorLike] = ...,
            method: PropagationMethod = ...,
            perturbative_profile: Optional[PerturbativeProfile] = ...,
        ) -> None: ...

    def __post_init__(self) -> None:
        device, dtype = self.context.device, self.context.dtype

        length = as_tensor(self.length_m, device=device, dtype=dtype)
        if torch.any(length < 0):
            raise ValueError("length_m must be non-negative everywhere (exactly 0 is a deliberate no-op segment).")
        object.__setattr__(self, "length_m", length)

        object.__setattr__(
            self, "n_e_mol_cm3", as_tensor(self.n_e_mol_cm3, device=device, dtype=dtype)
        )
        if self.n_n_mol_cm3 is not None:
            object.__setattr__(
                self, "n_n_mol_cm3", as_tensor(self.n_n_mol_cm3, device=device, dtype=dtype)
            )
        if self.B_field_T is not None:
            object.__setattr__(
                self, "B_field_T", as_tensor(self.B_field_T, device=device, dtype=dtype)
            )

    @property
    def n_segments(self) -> int:
        """Number of segments this profile describes (``length_m``'s last dimension)."""
        return self.length_m.shape[-1]

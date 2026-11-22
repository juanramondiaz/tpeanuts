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

"""Data structure for pointwise or binned observations.

A pure data container for a published measurement -- no physics, no
folding of its own (see ``tpeanuts.detector.common.event_rate`` for that).
Unlike v1_0's own ``Observation`` (this class's origin), a ``context:
RuntimeContext`` field is added here, matching every other dataclass in
this project (``tpeanuts.interfaces.flux.Flux``,
``tpeanuts.interfaces.medium.MediumProfile``,
``tpeanuts.interfaces.observable.ObservableSpectrum``).

Module contents:
    Observation
        Labels, coordinates, measured values and one-sided uncertainties,
        with optional bin widths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, cast

import torch

from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import TensorLike, as_tensor


@dataclass(frozen=True)
class Observation:
    """A published measurement: value, one-sided uncertainty, and x-grid.

    Attributes:
        labels: One label per entry (e.g. solar source name "pp"/"7Be", or
            a plain bin index string) -- for annotating plots/tables, not
            consumed numerically.
        x_MeV: Energy (or bin-center energy) grid, MeV, shape ``(n,)``.
        value: Measured central value per entry, shape ``(n,)`` -- a
            probability, a rate, or a count, depending on the observable.
        sigma_minus: One-sided lower uncertainty (positive), shape
            ``(n,)``.
        sigma_plus: One-sided upper uncertainty (positive), shape
            ``(n,)``. Equal to ``sigma_minus`` for a symmetric
            uncertainty (see ``from_symmetric``).
        context: RuntimeContext every tensor here is built on.
        bin_width_MeV: Optional bin width per entry, shape ``(n,)``. None
            for a pointwise observation (e.g. four P_ee(E) points, each a
            single representative energy); set for a genuinely binned
            spectrum.

    Raises:
        ValueError: If ``labels``/``value``/``sigma_minus``/``sigma_plus``
            (or, when given, ``bin_width_MeV``) does not have length
            ``x_MeV.shape[0]``.
    """

    labels: Tuple[str, ...]
    x_MeV: TensorLike
    value: TensorLike
    sigma_minus: TensorLike
    sigma_plus: TensorLike
    context: RuntimeContext
    bin_width_MeV: Optional[TensorLike] = None

    def __post_init__(self) -> None:
        device, dtype = self.context.device, self.context.dtype
        x_MeV = as_tensor(self.x_MeV, device=device, dtype=dtype)
        value = as_tensor(self.value, device=device, dtype=dtype)
        sigma_minus = as_tensor(self.sigma_minus, device=device, dtype=dtype)
        sigma_plus = as_tensor(self.sigma_plus, device=device, dtype=dtype)

        n = x_MeV.shape[0]
        for name, length in (
            ("labels", len(self.labels)),
            ("value", value.shape[0]),
            ("sigma_minus", sigma_minus.shape[0]),
            ("sigma_plus", sigma_plus.shape[0]),
        ):
            if length != n:
                raise ValueError(f"Observation.{name} has length {length}, expected {n} (Observation.x_MeV's length).")

        bin_width_MeV = self.bin_width_MeV
        if bin_width_MeV is not None:
            bin_width_MeV = as_tensor(bin_width_MeV, device=device, dtype=dtype)
            if bin_width_MeV.shape[0] != n:
                raise ValueError(f"Observation.bin_width_MeV has length {bin_width_MeV.shape[0]}, expected {n}.")

        object.__setattr__(self, "x_MeV", x_MeV)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "sigma_minus", sigma_minus)
        object.__setattr__(self, "sigma_plus", sigma_plus)
        object.__setattr__(self, "bin_width_MeV", bin_width_MeV)

    @classmethod
    def from_symmetric(
        cls,
        labels: Sequence[str],
        x_MeV: TensorLike,
        value: TensorLike,
        sigma: TensorLike,
        *,
        context: RuntimeContext,
        bin_width_MeV: Optional[TensorLike] = None,
    ) -> "Observation":
        """Build an ``Observation`` from a single symmetric uncertainty column.

        Args:
            labels: One label per entry.
            x_MeV: Energy/bin-center grid, shape ``(n,)``.
            value: Measured central value, shape ``(n,)``.
            sigma: Symmetric uncertainty (positive), shape ``(n,)``; used
                as both ``sigma_minus`` and ``sigma_plus``.
            context: RuntimeContext the returned instance's tensors are
                built on.
            bin_width_MeV: Optional bin width per entry, shape ``(n,)``.

        Returns:
            Observation with ``sigma_minus is sigma_plus is sigma``.
        """
        return cls(
            labels=tuple(labels), x_MeV=x_MeV, value=value,
            sigma_minus=sigma, sigma_plus=sigma, context=context, bin_width_MeV=bin_width_MeV,
        )

    @property
    def bin_edges_MeV(self) -> torch.Tensor:
        """Bin edges, shape ``(n+1,)``, built from ``x_MeV`` and ``bin_width_MeV``.

        Assumes each bin is centered on its ``x_MeV`` entry, i.e.
        ``[x_i - width_i/2, x_i + width_i/2]``; adjacent bins are not
        required to be contiguous (a gap or overlap raises no error here,
        the responsibility of whoever built ``x_MeV``/``bin_width_MeV``).

        Returns:
            Tensor of bin edges shaped ``(n+1,)``.

        Raises:
            ValueError: If ``bin_width_MeV`` is None (this observation is
                pointwise, not binned).
        """
        if self.bin_width_MeV is None:
            raise ValueError(
                "bin_edges_MeV requires bin_width_MeV to be set; this "
                "Observation is pointwise (e.g. one representative energy "
                "per source), not a binned spectrum."
            )
        x_MeV = cast(torch.Tensor, self.x_MeV)  # a tensor once the Observation is validated
        half = 0.5 * cast(torch.Tensor, self.bin_width_MeV)
        return torch.cat([x_MeV - half, (x_MeV[-1:] + half[-1:])])

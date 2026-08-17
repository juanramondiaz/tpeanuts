#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
#  This module is part of the Master's Thesis (MSc Dissertation):
#  - Fast Simulation of Neutrino Oscillations in Matter
#
#  Author:
#      Juan Ramon Diaz Santos <diazjuan@alumni.uv.es>
#
#  Supervisors:
#      Roberto Ruiz de Austri Bazan <rruiz@ific.uv.es>
#      Michele Lucente <michele.lucente@unibo.it>
#
#  Date:
#      June 2026
# =============================================================================

"""
Observation: a detector-agnostic container for a published measurement.

Formalizes the ad hoc ``pandas.read_csv(...)`` + column selection repeated
across ``notebooks/inference/inference1_borexino.ipynb`` and
``inference2_borexino_nsi.ipynb`` into a single reusable type, and extends
it to binned spectra (bin width, symmetric uncertainty) alongside the
pointwise, asymmetric-uncertainty convention those two notebooks already
used.

Module contents:
    Observation
        Value, one-sided uncertainties, and an energy/bin-center grid (with
        an optional bin width for a binned spectrum), plus a ``label`` per
        entry (e.g. a solar source name or a bin index).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass(frozen=True)
class Observation:
    """A published measurement: value, one-sided uncertainty, and x-grid.

    Parameters
    ----------
    labels:
        One label per entry (e.g. solar source name "pp"/"7Be", or a plain
        bin index string) -- for annotating plots/tables, not consumed
        numerically.
    x_MeV:
        Energy (or bin-center energy) grid, MeV, shape ``(n,)``.
    value:
        Measured central value per entry, shape ``(n,)`` -- a probability,
        a rate, or a count, depending on the observable.
    sigma_minus:
        One-sided lower uncertainty (positive), shape ``(n,)``.
    sigma_plus:
        One-sided upper uncertainty (positive), shape ``(n,)``. Equal to
        ``sigma_minus`` for a symmetric uncertainty (see ``from_symmetric``).
    bin_width_MeV:
        Optional bin width per entry, shape ``(n,)``. None for a pointwise
        observation (e.g. the four Borexino P_ee(E) points, each a single
        representative energy); set for a genuinely binned spectrum (e.g.
        Borexino's low-energy rate spectrum).
    """

    labels: tuple[str, ...]
    x_MeV: torch.Tensor
    value: torch.Tensor
    sigma_minus: torch.Tensor
    sigma_plus: torch.Tensor
    bin_width_MeV: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        n = self.x_MeV.shape[0]
        for name, field in (
            ("labels", self.labels),
            ("value", self.value),
            ("sigma_minus", self.sigma_minus),
            ("sigma_plus", self.sigma_plus),
        ):
            length = len(field) if name == "labels" else field.shape[0]
            if length != n:
                raise ValueError(
                    f"Observation.{name} has length {length}, expected {n} "
                    "(Observation.x_MeV's length)."
                )
        if self.bin_width_MeV is not None and self.bin_width_MeV.shape[0] != n:
            raise ValueError(
                f"Observation.bin_width_MeV has length {self.bin_width_MeV.shape[0]}, "
                f"expected {n}."
            )

    @classmethod
    def from_symmetric(
        cls,
        labels: Sequence[str],
        x_MeV: torch.Tensor,
        value: torch.Tensor,
        sigma: torch.Tensor,
        *,
        bin_width_MeV: Optional[torch.Tensor] = None,
    ) -> "Observation":
        """Build an ``Observation`` from a single symmetric uncertainty column.

        Args:
            labels: One label per entry.
            x_MeV: Energy/bin-center grid, shape ``(n,)``.
            value: Measured central value, shape ``(n,)``.
            sigma: Symmetric uncertainty (positive), shape ``(n,)``; used as
                both ``sigma_minus`` and ``sigma_plus``.
            bin_width_MeV: Optional bin width per entry, shape ``(n,)``.

        Returns:
            Observation with ``sigma_minus is sigma_plus is sigma``.
        """
        return cls(
            labels=tuple(labels), x_MeV=x_MeV, value=value,
            sigma_minus=sigma, sigma_plus=sigma, bin_width_MeV=bin_width_MeV,
        )

    @property
    def bin_edges_MeV(self) -> torch.Tensor:
        """Bin edges, shape ``(n+1,)``, built from ``x_MeV`` and ``bin_width_MeV``.

        Assumes each bin is centered on its ``x_MeV`` entry, i.e.
        ``[x_i - width_i/2, x_i + width_i/2]``; adjacent bins are not
        required to be contiguous (a gap or overlap raises no error here,
        the responsibility of whoever built ``x_MeV``/``bin_width_MeV``).

        Raises:
            ValueError: If ``bin_width_MeV`` is None (this observation is
                pointwise, not binned).
        """
        if self.bin_width_MeV is None:
            raise ValueError(
                "bin_edges_MeV requires bin_width_MeV to be set; this "
                "Observation is pointwise (e.g. one representative energy "
                "per solar source), not a binned spectrum."
            )
        half = 0.5 * self.bin_width_MeV
        return torch.cat([self.x_MeV - half, (self.x_MeV[-1:] + half[-1:])])

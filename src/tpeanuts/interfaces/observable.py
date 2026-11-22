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

"""What a detector produces: a rate over true energy, or counts per bin of an observed energy.

Two shapes, because there are two stages of a detector. ``ObservableSpectrum`` is the event-rate *density* over the true
neutrino energy -- before any detector response, so it is indexed by the same energy grid as the ``Flux`` it came from.
``BinnedSpectrum`` is the expected *counts* per bin of an observed (visible or reconstructed) energy -- after the
detector's response (smearing, efficiency, threshold), the exposure and the background -- so it is indexed by bin edges
that have nothing to do with the neutrino grid. Their units are those of the flux the detector was given: a detector fed
a ``Flux`` with an absolute ``differential_flux`` returns an absolute rate (events per second per MeV) or absolute
counts. ``Detector`` is the contract for the last link of the chain, ``spectrum = detector.observe(flux)``.
``interfaces/`` staying the single crossing point between ``tpeanuts.source``/``tpeanuts.medium``/
``tpeanuts.detector`` (see this package's own module docstring) is the reason these contracts live here rather than
inside ``tpeanuts.detector`` itself.

Module contents:
    ObservableSpectrum
        Frozen dataclass: ``energy_MeV``, ``rate_per_MeV``, ``context``.
    BinnedSpectrum
        Frozen dataclass: ``bin_edges_MeV``, ``counts``, ``context``, ``observable``.
    Detector
        Protocol: ``observe(flux) -> ObservableSpectrum | BinnedSpectrum``, what any detector offers to a ``Flux`` that
        has reached it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

from tpeanuts.interfaces.flux import Flux
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import TensorLike, as_tensor


@dataclass(frozen=True)
class ObservableSpectrum:
    """An event-rate density sampled on a true-energy grid.

    Attributes:
        energy_MeV: True neutrino energy, in MeV, shaped ``(..., n_energy)``
            -- the same grid whatever produced this spectrum was evaluated
            on (e.g. ``tpeanuts.interfaces.flux.Flux.energy_MeV``).
        rate_per_MeV: Event-rate density at each energy, shaped like
            ``energy_MeV`` (or broadcastable against it). Its units/absolute
            normalization are entirely up to whatever built it -- see the
            concrete producer's own docstring.
        context: RuntimeContext this spectrum's tensors are stored on.

    Examples:
        >>> import math
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> from tpeanuts.interfaces import ObservableSpectrum
        >>> spectrum = ObservableSpectrum(
        ...     energy_MeV=[1.0, 2.0, 3.0], rate_per_MeV=[10.0, 5.0, 1.0], context=context)
        >>> spectrum.energy_MeV.shape, spectrum.rate_per_MeV.dtype
        (torch.Size([3]), torch.float64)
    """

    energy_MeV: torch.Tensor
    rate_per_MeV: torch.Tensor
    context: RuntimeContext

    if TYPE_CHECKING:  # the runtime __init__ is the dataclass one; tensor fields accept any TensorLike

        def __init__(
            self,
            energy_MeV: TensorLike,
            rate_per_MeV: TensorLike,
            context: RuntimeContext,
        ) -> None: ...

    def __post_init__(self) -> None:
        device, dtype = self.context.device, self.context.dtype
        object.__setattr__(self, "energy_MeV", as_tensor(self.energy_MeV, device=device, dtype=dtype))
        object.__setattr__(self, "rate_per_MeV", as_tensor(self.rate_per_MeV, device=device, dtype=dtype))


@dataclass(frozen=True)
class BinnedSpectrum:
    """Expected event counts per bin of an observed energy.

    The difference with ``ObservableSpectrum`` is which energy it is indexed by: that one is a density over the *true*
    neutrino energy (the grid of the ``Flux``), this one is a histogram over the *observed* energy -- the visible energy
    after the detector's response -- so the neutrino grid does not appear in it at all.

    Attributes:
        bin_edges_MeV: Strictly increasing bin edges of the observed energy, in MeV, shaped ``(n_bins + 1,)``.
        counts: Expected counts in each bin (events, exposure and background included), shaped ``(..., n_bins)``;
            leading axes are whatever batch the flux carried (scenarios, oscillation parameters, ...).
        context: RuntimeContext this spectrum's tensors are stored on.
        observable: What the bin edges measure, for labelling (``"visible energy"``, ``"electron kinetic energy"``...).

    Raises:
        ValueError: If the edges are not a strictly increasing vector, or ``counts`` does not have one entry per bin.

    Examples:
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> from tpeanuts.interfaces import BinnedSpectrum
        >>> spectrum = BinnedSpectrum(bin_edges_MeV=[1.0, 2.0, 4.0], counts=[10.0, 6.0], context=context)
        >>> spectrum.bin_centers_MeV.tolist(), spectrum.total().item()
        ([1.5, 3.0], 16.0)
    """

    bin_edges_MeV: torch.Tensor
    counts: torch.Tensor
    context: RuntimeContext
    observable: str = "visible energy"

    if TYPE_CHECKING:  # the runtime __init__ is the dataclass one; tensor fields accept any TensorLike

        def __init__(
            self,
            bin_edges_MeV: TensorLike,
            counts: TensorLike,
            context: RuntimeContext,
            observable: str = "visible energy",
        ) -> None: ...

    def __post_init__(self) -> None:
        device, dtype = self.context.device, self.context.dtype
        edges = as_tensor(self.bin_edges_MeV, device=device, dtype=dtype)
        counts = as_tensor(self.counts, device=device, dtype=dtype)
        if edges.ndim != 1 or edges.numel() < 2:
            raise ValueError(f"bin_edges_MeV must be a vector of at least two edges, got shape {tuple(edges.shape)}.")
        if bool(torch.any(torch.diff(edges) <= 0)):
            raise ValueError("bin_edges_MeV must be strictly increasing.")
        if counts.ndim == 0 or counts.shape[-1] != edges.numel() - 1:
            raise ValueError(
                f"counts must have one entry per bin on its last axis ({edges.numel() - 1}), "
                f"got shape {tuple(counts.shape)}."
            )
        object.__setattr__(self, "bin_edges_MeV", edges)
        object.__setattr__(self, "counts", counts)

    @property
    def bin_centers_MeV(self) -> torch.Tensor:
        """Bin centres, in MeV, shaped ``(n_bins,)``."""
        return 0.5 * (self.bin_edges_MeV[1:] + self.bin_edges_MeV[:-1])

    @property
    def bin_widths_MeV(self) -> torch.Tensor:
        """Bin widths, in MeV, shaped ``(n_bins,)``."""
        return torch.diff(self.bin_edges_MeV)

    def total(self) -> torch.Tensor:
        """Expected counts summed over the bins, shaped like ``counts`` without its last axis."""
        return self.counts.sum(dim=-1)


@runtime_checkable
class Detector(Protocol):
    """What a detector offers to a flux that has reached it.

    Any object with an ``observe`` method of this shape is a detector: the chain
    ``flux -> medium.propagate(...) -> detector.observe(flux)`` needs nothing else from it. A detector with no
    response (like ``IBDDetector``) returns the rate over the true energy, an ``ObservableSpectrum``; one that folds in
    a response, an exposure and a background returns the expected counts, a ``BinnedSpectrum``.

    Examples:
        >>> import torch
        >>> import tpeanuts as tp
        >>> from tpeanuts.detector.ibd import IBDDetector
        >>> from tpeanuts.interfaces import Detector
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> isinstance(IBDDetector(n_target_protons=1.0e30, context=context), Detector)
        True
    """

    def observe(self, flux: Flux) -> ObservableSpectrum | BinnedSpectrum:
        """What the detector sees from ``flux`` (the flux as it arrives at the detector)."""
        ...

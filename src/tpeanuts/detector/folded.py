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

"""Detectors that fold a flux through a cross section and a response into counts per bin.

The generic detectors behind Borexino, SNO and SNO-II: the experiment-specific part (target, cross sections, resolution,
threshold, exposure) is what a concrete experiment plugs in, the physics is the fold already in
``tpeanuts.detector.common.event_rate``::

    dR/dT(T) = n_target * integral dE [Phi_e(E) dsigma_e/dT + Phi_x(E) dsigma_x/dT]      (true spectrum)
    counts_i = exposure * integral_{bin i} dT' eps(T') integral dT R(T'|T) dR/dT(T)       (+ background)

Both are ``tpeanuts.interfaces.observable.Detector``: ``observe(flux)`` takes the ``Flux`` that has reached the detector
and returns a ``BinnedSpectrum``. The flavour fluxes are those of the evolved state, ``flux.per_flavour()``. A flux
whose ``flux_units`` carry ``MeV^-1`` is a continuous spectrum, integrated over the neutrino energy; any other is a set
of discrete lines (e.g. 7Be), summed with no quadrature. The lower-level ``counts`` takes bare flavour fluxes.

Module contents:
    FoldedDetector
        Frozen dataclass: a channel with a nu_e and a nu_mu/nu_tau cross section differential in the observable
        (elastic scattering, charged current).
    FixedEnergyDetector
        Frozen dataclass: flavour-blind interaction whose signal sits at one visible energy (e.g. neutron capture).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from tpeanuts.detector.common.event_rate import (
    bin_counts,
    true_observable_spectrum,
    true_observable_spectrum_discrete,
)
from tpeanuts.detector.common.response import DetectorResponse, gaussian_response_matrix
from tpeanuts.detector.common.target import reduced_target_count
from tpeanuts.interfaces.flux import Flux, SourceFlux
from tpeanuts.interfaces.observable import BinnedSpectrum
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import TensorLike, as_tensor

Tensor = torch.Tensor
CrossSection = Callable[[Tensor, Tensor], Tensor]


def _check_flux(flux: Flux, *, antinu: bool, detector: str) -> None:
    """Raise unless ``flux`` is something ``detector`` can count: right kind, absolute and per unit area."""
    if bool(torch.all(torch.as_tensor(flux.antinu))) != antinu:
        kind = "antineutrinos" if antinu else "neutrinos"
        raise ValueError(f"{detector} reacts to {kind}: the flux must have antinu={antinu}.")
    if flux.differential_flux is None:
        raise ValueError("this flux has no differential_flux, so an absolute event count cannot be computed.")
    if "cm^-2" not in flux.flux_units:
        raise ValueError(f"the flux must be per unit area (flux_units contains 'cm^-2'), got {flux.flux_units!r}.")


def _is_continuous(flux: Flux) -> bool:
    """Whether ``flux`` is a density in energy (``MeV^-1`` in its units) rather than a set of lines."""
    return "MeV^-1" in flux.flux_units


@dataclass(frozen=True)
class FoldedDetector:
    """A target with a nu_e and a nu_mu/nu_tau cross section, differential in an observable, and a detector response.

    Attributes:
        n_target: Number of target particles (electrons, deuterons, ...); scalar or batched.
        response: ``DetectorResponse`` of the detector (grid, response matrix, efficiency, analysis bins).
        exposure_s: Live time in seconds, or whatever factor turns a rate into counts.
        cross_section_e: ``(energy_MeV, T_MeV) -> dsigma_e/dT`` shaped ``(n_E, n_T)``, in units of ``SIGMA_UNIT_CM2``
            (see ``tpeanuts.detector.neutrino_electron``). Applies to the electron flavour.
        cross_section_x: The same for the muon and tau flavours; None for a channel blind to them (charged current).
        context: RuntimeContext the returned spectrum's tensors are built on.
        antinu: Whether the detector reacts to antineutrinos instead of neutrinos.
        background_counts: Optional background per bin, shaped ``(n_bins,)``, added to the signal.
        observable: What the analysis bins measure, for labelling the ``BinnedSpectrum``.
        signal_scale: Free overall normalisation of the signal (not of the background), 1 by default; a nuisance such
            as Daya Bay's ``global_normalization``. Scalar, may require grad.
    """

    n_target: TensorLike
    response: DetectorResponse
    exposure_s: TensorLike
    cross_section_e: CrossSection
    cross_section_x: Optional[CrossSection]
    context: RuntimeContext
    antinu: bool = False
    background_counts: Optional[Tensor] = None
    observable: str = "electron kinetic energy"
    signal_scale: TensorLike = 1.0

    def observe(self, flux: Flux | SourceFlux) -> BinnedSpectrum:
        """Expected counts per analysis bin from a flux that has reached the detector.

        Args:
            flux: Flux at the detector, per unit area (``cm^-2`` in its units), with a ``differential_flux``, one
                neutrino energy grid ``(n_E,)`` and at least three flavours; its state may carry leading batch axes.
                A ``SourceFlux`` is the sum of its components (e.g. the reactors seen by one detector): when they
                share the energy grid the fluxes are added and folded once, which the linearity of the fold allows.

        Returns:
            BinnedSpectrum shaped ``(..., n_bins)``, in events over the exposure.

        Raises:
            ValueError: If the flux is of the wrong kind (neutrino vs antineutrino), has no absolute intensity or is
                not per unit area, or has fewer than three flavours.
        """
        components = list(flux.components.values()) if isinstance(flux, SourceFlux) else [flux]
        parts = [self._flavour_fluxes(component) for component in components]
        first_energy, first_discrete = parts[0][0], parts[0][3]
        if all(torch.equal(p[0], first_energy) and p[3] == first_discrete for p in parts[1:]):
            flux_e = sum((p[1] for p in parts[1:]), parts[0][1])
            flux_x = sum((p[2] for p in parts[1:]), parts[0][2])
            return self.counts(first_energy, flux_e, flux_x, discrete=first_discrete)
        # different grids or kinds (continuous next to lines): fold each and add the counts
        spectra = [self.counts(e, fe, fx, discrete=d) for e, fe, fx, d in parts]
        return BinnedSpectrum(
            bin_edges_MeV=spectra[0].bin_edges_MeV, counts=sum((s.counts for s in spectra[1:]), spectra[0].counts),
            context=self.context, observable=self.observable,
        )

    def _flavour_fluxes(self, flux: Flux) -> tuple[Tensor, Tensor, Tensor, bool]:
        """``(energy, electron flux, mu+tau flux, discrete)`` of one flux, after checking the detector can count it."""
        _check_flux(flux, antinu=self.antinu, detector=type(self).__name__)
        per_flavour = flux.per_flavour()
        if per_flavour.shape[-1] < 3:
            raise ValueError(f"the flux must carry at least three flavours, got {per_flavour.shape[-1]}.")
        return flux.energy_MeV, per_flavour[..., 0], per_flavour[..., 1:3].sum(dim=-1), not _is_continuous(flux)

    def counts(self, energy_MeV: Tensor, flux_e: Tensor, flux_x: Tensor, *, discrete: bool = False) -> BinnedSpectrum:
        """Expected counts per analysis bin from bare flavour fluxes.

        Args:
            energy_MeV: Neutrino energies (the grid, or the line energies), shaped ``(n_E,)``.
            flux_e: Electron-flavour flux at those energies, shaped ``(..., n_E)``: a density per MeV, or the flux of
                each line when ``discrete``.
            flux_x: Muon-plus-tau flux, broadcastable with ``flux_e``.
            discrete: The fluxes are lines, summed with no energy quadrature.

        Returns:
            BinnedSpectrum shaped ``(..., n_bins)``.
        """
        grid = self.response.T_grid_MeV
        sigma_e = self.cross_section_e(energy_MeV, grid)
        sigma_x = torch.zeros_like(sigma_e) if self.cross_section_x is None else self.cross_section_x(energy_MeV, grid)
        n_target = as_tensor(self.signal_scale, device=flux_e.device, dtype=flux_e.dtype) * reduced_target_count(
            self.n_target, device=flux_e.device, dtype=flux_e.dtype,
        )
        if discrete:
            true_spectrum = true_observable_spectrum_discrete(flux_e, flux_x, sigma_e, sigma_x, n_target)
        else:
            true_spectrum = true_observable_spectrum(energy_MeV, flux_e, flux_x, sigma_e, sigma_x, n_target)
        counts = self.response.fold(true_spectrum, self.exposure_s)
        if self.background_counts is not None:
            counts = counts + self.background_counts
        return BinnedSpectrum(
            bin_edges_MeV=self.response.bin_edges_MeV, counts=counts, context=self.context, observable=self.observable,
        )


@dataclass(frozen=True)
class FixedEnergyDetector:
    """A flavour-blind interaction whose detected signal sits at a single visible energy.

    The interaction rate ``n_target * integral dE Phi_active(E) sigma(E)`` is multiplied by a detection efficiency and
    spread over the analysis bins by a Gaussian of width ``resolution_MeV`` around ``visible_energy_MeV`` -- the
    neutral-current deuteron breakup, where the neutron is captured and a gamma of fixed energy is what is seen,
    whatever the neutrino energy.

    Attributes:
        n_target: Number of target particles; scalar or batched.
        sigma_total: ``energy_MeV -> sigma(E)`` in units of ``SIGMA_UNIT_CM2``, the same for every active flavour.
        visible_energy_MeV: Energy at which the signal is seen.
        resolution_MeV: Standard deviation of the visible energy at ``visible_energy_MeV``.
        efficiency: Overall detection efficiency of the signal, in ``[0, 1]``.
        reco_grid_MeV: Grid on which the Gaussian is evaluated before binning, shaped ``(n_Tp,)``.
        bin_edges_MeV: Strictly increasing edges of the analysis bins, shaped ``(n_bins + 1,)``, inside the grid.
        exposure_s: Live time in seconds.
        context: RuntimeContext the returned spectrum's tensors are built on.
        background_counts: Optional background per bin, added to the signal.
        observable: What the analysis bins measure, for labelling the ``BinnedSpectrum``.
    """

    n_target: TensorLike
    sigma_total: Callable[[Tensor], Tensor]
    visible_energy_MeV: float
    resolution_MeV: TensorLike
    efficiency: float
    reco_grid_MeV: Tensor
    bin_edges_MeV: Tensor
    exposure_s: TensorLike
    context: RuntimeContext
    background_counts: Optional[Tensor] = None
    observable: str = "visible energy"

    def observe(self, flux: Flux) -> BinnedSpectrum:
        """Expected counts per analysis bin from a flux that has reached the detector.

        Only the active flavours (the first three) count, so a sterile admixture in the evolved state is lost.

        Args:
            flux: Neutrino flux at the detector, per unit area, with a ``differential_flux`` and one energy grid.

        Returns:
            BinnedSpectrum shaped ``(..., n_bins)``.

        Raises:
            ValueError: If the flux is not a neutrino flux with an absolute per-area intensity.
        """
        _check_flux(flux, antinu=False, detector=type(self).__name__)
        active = flux.per_flavour()[..., :3].sum(dim=-1)
        return self.counts(flux.energy_MeV, active, discrete=not _is_continuous(flux))

    def counts(self, energy_MeV: Tensor, active_flux: Tensor, *, discrete: bool = False) -> BinnedSpectrum:
        """Expected counts per analysis bin from a bare active flux.

        Args:
            energy_MeV: Neutrino energies, shaped ``(n_E,)``.
            active_flux: Flux summed over the active flavours, shaped ``(..., n_E)``: a density per MeV, or the flux
                of each line when ``discrete``.
            discrete: The flux is made of lines, summed with no energy quadrature.

        Returns:
            BinnedSpectrum shaped ``(..., n_bins)``.
        """
        sigma = self.sigma_total(energy_MeV)
        n_target = reduced_target_count(self.n_target, device=active_flux.device, dtype=active_flux.dtype)
        weighted = active_flux * sigma
        interactions_per_s = n_target * (weighted.sum(dim=-1) if discrete else torch.trapezoid(weighted, x=energy_MeV))
        detected_per_s = interactions_per_s * self.efficiency

        visible = torch.as_tensor(
            [self.visible_energy_MeV], dtype=self.reco_grid_MeV.dtype, device=self.reco_grid_MeV.device,
        )
        resolution = as_tensor(self.resolution_MeV, device=visible.device, dtype=visible.dtype)
        column = gaussian_response_matrix(visible, self.reco_grid_MeV, resolution)[:, 0]
        reco_spectrum = detected_per_s.unsqueeze(-1) * column  # (..., n_Tp), events / s / MeV
        counts = bin_counts(reco_spectrum, self.reco_grid_MeV, self.bin_edges_MeV, self.exposure_s)
        if self.background_counts is not None:
            counts = counts + self.background_counts
        return BinnedSpectrum(
            bin_edges_MeV=self.bin_edges_MeV, counts=counts, context=self.context, observable=self.observable,
        )

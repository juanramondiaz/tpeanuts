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

"""The trivial detector: inverse beta decay, no response, no efficiency, no background.

The simplest possible producer of a
``tpeanuts.interfaces.observable.ObservableSpectrum``: no energy smearing
(no resolution matrix), no efficiency curve (implicitly 1 everywhere), and
no background -- exactly the "theoretical floor" v1_0's own detector layer
documents as the first link in its event-rate chain (``true_observable_
spectrum``, before ``apply_response``/``apply_efficiency``/``bin_counts``).

It is a ``tpeanuts.interfaces.observable.Detector``: ``observe(flux)`` takes the ``Flux`` that has reached the detector
(an antineutrino flux per unit area, see ``VacuumMedium.propagate(..., dilute=True)``) and returns the absolute event
rate, ``n_target * ibd_cross_section(E) * Phi_e(E)`` with ``Phi_e`` the electron-antineutrino flux that survives. The
lower-level ``event_rate`` takes bare ``(energy_MeV, probabilities)`` tensors, optionally weighted by a flux spectrum;
without one it is the response to a *unit* flux, not a rate.


Module contents:
    IBDDetector
        Frozen dataclass: a target proton count and a RuntimeContext, with
        ``observe()`` (from a ``Flux``) and ``event_rate()`` (from bare
        tensors) building the corresponding ``ObservableSpectrum``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from tpeanuts.detector.common.target import reduced_target_count
from tpeanuts.detector.ibd_cross_section import ibd_cross_section
from tpeanuts.interfaces.flux import Flux
from tpeanuts.interfaces.observable import ObservableSpectrum
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import TensorLike, as_tensor


@dataclass(frozen=True)
class IBDDetector:
    """An inverse-beta-decay target, reacting to electron-antineutrino survival probability.

    Attributes:
        n_target_protons: Free-proton target count; scalar or batched over
            scenarios. No mass-to-target-count conversion is performed
            here -- pass whatever count is wanted directly, or compute it
            first via ``tpeanuts.detector.common.target.n_atoms``
            (``element="H"``).
        context: RuntimeContext the returned spectrum's tensors are built on.
    """

    n_target_protons: TensorLike
    context: RuntimeContext

    def observe(self, flux: Flux) -> ObservableSpectrum:
        """The IBD event rate of an antineutrino flux that has reached the detector.

        ``rate_per_MeV = n_target_protons * ibd_cross_section(E) * Phi_e(E)``, with
        ``Phi_e = flux.per_flavour()[..., 0]`` the flux of electron antineutrinos, i.e. the intensity times the
        survival probability of the evolved state -- no energy response, efficiency or background folded in. With
        ``Phi`` in ``cm^-2 s^-1 MeV^-1`` the result is in events per second per MeV.


        Args:
            flux: The flux as it arrives at the detector: antineutrinos (``antinu`` true), with a ``differential_flux``
                per unit area (``flux_units`` containing ``cm^-2``). A reactor flux is brought there by
                ``VacuumMedium.propagate(flux, model, dilute=True)``.

        Returns:
            ObservableSpectrum at the flux's energies.

        Raises:
            ValueError: If the flux is not an antineutrino flux, has no ``differential_flux``, or its units are not per
                unit area (an emission rate must be diluted over the baseline first).
        """
        if not bool(torch.all(torch.as_tensor(flux.antinu))):
            raise ValueError("inverse beta decay reacts to electron antineutrinos: the flux must have antinu=True.")
        if flux.differential_flux is None:
            raise ValueError("this flux has no differential_flux, so an absolute event rate cannot be computed.")
        if "cm^-2" not in flux.flux_units:
            raise ValueError(
                f"the flux must be per unit area (flux_units contains 'cm^-2'), got {flux.flux_units!r}: propagate an "
                "emission rate with VacuumMedium.propagate(..., dilute=True) first."
            )
        return self.event_rate(flux.energy_MeV, torch.ones_like(flux.energy_MeV),
                               differential_flux=flux.per_flavour()[..., 0])

    def event_rate(
        self,
        energy_MeV: TensorLike,
        probabilities: TensorLike,
        *,
        differential_flux: Optional[TensorLike] = None,
    ) -> ObservableSpectrum:
        """Build the IBD event-rate density from a survival-probability curve.

        ``rate_per_MeV = n_target_protons * ibd_cross_section(energy_MeV) * probabilities`` (times
        ``differential_flux``, if given) -- no energy response, efficiency or background folded in (see this module's
        own docstring). Without a ``differential_flux`` this is the response of the target to a **unit** flux of
        antineutrinos: proportional to a true rate only if the flux spectrum were flat, so it must not be read as one.
        Give the flux spectrum, or use ``observe`` with a ``Flux``, for an event rate.


        Args:
            energy_MeV: True antineutrino energy, in MeV, shaped ``(..., n_energy)`` (or broadcastable against
                ``probabilities``).
            probabilities: Electron-antineutrino survival probability at each energy (e.g.
                ``evolved.probabilities()[..., 0]``), shaped like ``energy_MeV``.
            differential_flux: Optional antineutrino flux per unit area and energy at the detector, shaped like
                ``energy_MeV``; multiplies the rate.

        Returns:
            ObservableSpectrum with this instance's own ``energy_MeV`` and the ``rate_per_MeV`` computed above.
        """
        device, dtype = self.context.device, self.context.dtype
        energy = as_tensor(energy_MeV, device=device, dtype=dtype)
        prob = as_tensor(probabilities, device=device, dtype=dtype)
        # ibd_cross_section is in units of SIGMA_UNIT_CM2, so the target count carries that factor.
        n_target = reduced_target_count(self.n_target_protons, device=device, dtype=dtype)

        rate_per_MeV = n_target * ibd_cross_section(energy) * prob
        if differential_flux is not None:
            rate_per_MeV = rate_per_MeV * as_tensor(differential_flux, device=device, dtype=dtype)
        return ObservableSpectrum(energy_MeV=energy, rate_per_MeV=rate_per_MeV, context=self.context)

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
Differentiable IceCube DeepCore event-count model: real weighted-MC x Earth matter x detector.

``IceCubeDetectorModel`` wraps ``tpeanuts.inference.atmospheric_model
.AtmosphericOscillationModel`` and ``tpeanuts.detector.icecube.event_rate``
into a ``fit_lbfgs``-compatible ``predict(theta)`` returning real per-bin
counts. Unlike every other detector model in this project,
``theta`` here is **not** oscillation parameters alone: this release's
``weight`` column has no independently known absolute livetime to convert
it into a physical rate (see ``detector.icecube.parameters``' module
docstring), so ``free`` also carries a neutrino and a muon-background
normalization scale, fit alongside theta23/DeltamSq3l rather than
pre-computed in closed form (unlike the official release's own example
notebook).

Module contents:
    IceCubeDetectorModel
        Wraps an AtmosphericOscillationModel and the real cached MC/data,
        predicting real per-bin counts from a 4-parameter free vector
        (theta23, DeltamSq3l, nu_norm, mu_norm).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from tpeanuts.detector.icecube.event_rate import (
    CHANNELS,
    ChannelEvents,
    HypersurfaceTable,
    muon_background_histogram,
    predicted_neutrino_counts,
    prepare_channel_events,
    prepare_hypersurface_table,
)
from tpeanuts.detector.icecube.parameters import DETECTOR_DEPTH_M
from tpeanuts.inference.atmospheric_model import AtmosphericOscillationModel

FREE_PARAM_KEYS: tuple[str, ...] = ("theta23", "DeltamSq3l", "nu_norm", "mu_norm")


@dataclass(frozen=True)
class IceCubeDetectorModel:
    """Predicted IceCube DeepCore counts per real analysis bin, as a function of ``theta``.

    Parameters
    ----------
    oscillation_model:
        An ``AtmosphericOscillationModel`` supplying the fixed
        theta12/theta13/DeltamSq21/delta13 inputs.
    earth_profile:
        Earth density profile (``tpeanuts.medium.earth.profile
        .build_earth_profile``'s return value).
    channels:
        ``{channel: ChannelEvents}`` for every entry of
        ``detector.icecube.event_rate.CHANNELS``, e.g. from
        ``IceCubeDetectorModel.from_real_data``.
    hypersurfaces:
        ``{channel: HypersurfaceTable}`` for every entry of ``CHANNELS``.
    muon_background:
        Real pre-binned atmospheric-muon background, shape ``(N_BINS,)``.
    detector_depth_m:
        Detector depth below the Earth's surface, metres (see
        ``detector.icecube.parameters.DETECTOR_DEPTH_M``).
    """

    oscillation_model: AtmosphericOscillationModel
    earth_profile: object
    channels: dict[str, ChannelEvents]
    hypersurfaces: dict[str, HypersurfaceTable]
    muon_background: torch.Tensor
    detector_depth_m: float = DETECTOR_DEPTH_M

    @property
    def free(self) -> tuple[str, ...]:
        """Free parameter names: theta23, DeltamSq3l, and the two normalization scales."""
        return FREE_PARAM_KEYS

    def predict(self, theta: torch.Tensor) -> torch.Tensor:
        """Predict real per-bin counts from ``theta = (theta23, DeltamSq3l, nu_norm, mu_norm)``.

        Args:
            theta: 1-D tensor of length 4, in ``FREE_PARAM_KEYS`` order.

        Returns:
            Predicted counts per real analysis bin, shape ``(N_BINS,)``.
        """
        theta23, deltam_sq_3l, nu_norm, mu_norm = theta.unbind()
        neutrino_counts = predicted_neutrino_counts(
            theta23, deltam_sq_3l,
            oscillation_model=self.oscillation_model,
            earth_profile=self.earth_profile,
            detector_depth_m=self.detector_depth_m,
            channels=self.channels,
            hypersurfaces=self.hypersurfaces,
        )
        return nu_norm * neutrino_counts + mu_norm * self.muon_background

    @classmethod
    def from_real_data(
        cls,
        oscillation_model: AtmosphericOscillationModel,
        earth_profile: object,
        *,
        downsample: int = 10,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
        detector_depth_m: float = DETECTOR_DEPTH_M,
    ) -> "IceCubeDetectorModel":
        """Build a model from the real cached IceCube DeepCore release.

        Args:
            oscillation_model: An ``AtmosphericOscillationModel``.
            earth_profile: Earth density profile.
            downsample: Event-level MC downsampling factor (weights
                rescaled to compensate), see ``detector.icecube.event_rate
                .DEFAULT_DOWNSAMPLE``.
            device, dtype: Target tensor device/dtype.
            detector_depth_m: Detector depth below the Earth's surface,
                metres.

        Returns:
            A configured ``IceCubeDetectorModel``.
        """
        channels = {
            channel: prepare_channel_events(channel, downsample=downsample, device=device, dtype=dtype)
            for channel in CHANNELS
        }
        hypersurfaces = {
            channel: prepare_hypersurface_table(channel, device=device, dtype=dtype)
            for channel in CHANNELS
        }
        muon_background = muon_background_histogram(device=device, dtype=dtype)

        return cls(
            oscillation_model=oscillation_model,
            earth_profile=earth_profile,
            channels=channels,
            hypersurfaces=hypersurfaces,
            muon_background=muon_background,
            detector_depth_m=detector_depth_m,
        )

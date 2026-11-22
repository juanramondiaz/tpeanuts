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

"""Differentiable IceCube DeepCore event-count model: real weighted-MC x Earth matter x detector.

Ported from v1_0's ``detector/icecube/inference_model.py``.
``IceCubeModel`` wraps
``tpeanuts.experiments.icecube.event_rate`` into a ``predict(theta)``
returning real per-bin counts. Unlike v1_0's own version (which wraps
a separate ``inference.model_atmosphere.AtmosphericOscillationModel``,
never ported to v2 -- see below), and following
``tpeanuts.experiments.borexino.inference_model.BorexinoModel``/
``tpeanuts.experiments.sno.inference_model.SNOModel``'s own
precedent, the fixed solar/reactor oscillation inputs
(theta12/theta13/delta13/DeltamSq21) are plain fields here, forwarded
directly to ``tpeanuts.experiments.icecube.event_rate
.predicted_neutrino_counts`` -- there is no separate oscillation-model
class to port, since that model added no behaviour beyond building a
``PropagationModel``, which ``predicted_neutrino_counts`` already does
inline.

Unlike every other detector model in this project, ``theta`` here is
**not** oscillation parameters alone: this release's ``weight`` column
has no independently known absolute livetime to convert it into a
physical rate (see
``tpeanuts.experiments.icecube.parameters``'s own module docstring), so
``free`` also carries a neutrino and a muon-background normalization
scale, fit alongside theta23/DeltamSq3l rather than pre-computed in
closed form (unlike the official release's own example notebook).

Module contents:
    FREE_PARAM_KEYS
        The four free parameter names, in ``theta`` order.
    IceCubeModel
        Wraps the real cached MC/data, predicting real per-bin counts
        from a 4-parameter free vector (theta23, DeltamSq3l, nu_norm,
        mu_norm).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch

from tpeanuts.config import default
from tpeanuts.experiments.icecube.event_rate import (
    CHANNELS,
    ChannelEvents,
    HypersurfaceTable,
    build_channel_events,
    build_hypersurface_table,
    muon_background_histogram,
    predicted_neutrino_counts,
)
from tpeanuts.experiments.icecube.parameters import DETECTOR_DEPTH_M, IceCubeParameters
from tpeanuts.physics.evolution import PropagationMethod
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import TensorLike

Tensor = torch.Tensor

FREE_PARAM_KEYS: Tuple[str, ...] = ("theta23", "DeltamSq3l", "nu_norm", "mu_norm")


@dataclass(frozen=True)
class IceCubeModel:
    """Predicted IceCube DeepCore counts per real analysis bin, as a function of ``theta``.

    Attributes:
        channels: ``{channel: ChannelEvents}`` for every entry of
            ``tpeanuts.experiments.icecube.event_rate.CHANNELS``, e.g.
            from ``IceCubeModel.from_real_data``.
        hypersurfaces: ``{channel: HypersurfaceTable}`` for every entry
            of ``CHANNELS``.
        muon_background: Real pre-binned atmospheric-muon background,
            shape ``(200,)``.
        theta12, theta13, delta13, DeltamSq21: Fixed solar/reactor
            oscillation inputs (radians/eV^2), not re-fit from
            atmospheric-only data -- see module docstring.
        context: RuntimeContext every tensor this model builds is on.
        detector_depth_m: Detector depth below the Earth's surface,
            metres.
        earth_method: Which ``PropagationMethod`` builds each event's
            own Earth transition matrix. Defaults to
            ``PropagationMethod.ANALYTIC`` (exact closed-form; safe for
            a large batched energy/eta_rad axis, unlike
            ``ANALYTIC_PERTURBATIVE`` -- confirmed directly, see
            ``tpeanuts.medium.earth.earth.EarthMedium.transition_matrix``'s own
            docstring and
            [[reference_earth_analytic_perturbative_batching]]).
        earth_nsteps: Chord discretization forwarded to
            ``EarthMedium`` (ignored by ``ANALYTIC``).
        extensions: Optional propagation extensions (for example
            ``tpeanuts.physics.extensions.nsi.NSITerm``), forwarded to the
            ``PropagationModel`` this model builds, so they act in the Earth. Empty (the
            default) is the Standard Model and changes nothing.

    Raises:
        ValueError: If ``channels``/``hypersurfaces`` is missing an
            entry for one of ``CHANNELS``.
    """

    channels: Dict[str, ChannelEvents]
    hypersurfaces: Dict[str, HypersurfaceTable]
    muon_background: Tensor
    theta12: TensorLike
    theta13: TensorLike
    delta13: TensorLike
    DeltamSq21: TensorLike
    context: RuntimeContext
    detector_depth_m: float = DETECTOR_DEPTH_M
    earth_method: PropagationMethod = PropagationMethod.ANALYTIC
    earth_nsteps: int = default.EARTH_NSTEPS
    extensions: Tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        missing_channels = set(CHANNELS) - set(self.channels)
        if missing_channels:
            raise ValueError(f"channels is missing entry(ies) for {sorted(missing_channels)}.")
        missing_hs = set(CHANNELS) - set(self.hypersurfaces)
        if missing_hs:
            raise ValueError(f"hypersurfaces is missing entry(ies) for {sorted(missing_hs)}.")

    @property
    def free(self) -> Tuple[str, ...]:
        """Free parameter names: theta23, DeltamSq3l, and the two normalization scales."""
        return FREE_PARAM_KEYS

    def predict(self, theta: Tensor) -> Tensor:
        """Predict real per-bin counts from ``theta = (theta23, DeltamSq3l, nu_norm, mu_norm)``.

        Args:
            theta: 1-D tensor of length 4, in ``FREE_PARAM_KEYS`` order.

        Returns:
            Predicted counts per real analysis bin, shape ``(200,)``.
        """
        theta23, deltam_sq_3l, nu_norm, mu_norm = theta.unbind()
        neutrino_counts = predicted_neutrino_counts(
            theta23, deltam_sq_3l,
            theta12=self.theta12, theta13=self.theta13, delta13=self.delta13, deltam_sq_21=self.DeltamSq21,
            context=self.context, detector_depth_m=self.detector_depth_m,
            channels=self.channels, hypersurfaces=self.hypersurfaces,
            earth_method=self.earth_method, earth_nsteps=self.earth_nsteps, extensions=self.extensions,
        )
        return nu_norm * neutrino_counts + mu_norm * self.muon_background

    @classmethod
    def from_real_data(
        cls,
        *,
        downsample: int = 10,
        context: RuntimeContext,
        detector_depth_m: float = DETECTOR_DEPTH_M,
        earth_method: PropagationMethod = PropagationMethod.ANALYTIC,
        earth_nsteps: int = default.EARTH_NSTEPS,
        extensions: Tuple[Any, ...] = (),
    ) -> "IceCubeModel":
        """Build a model from the real cached IceCube DeepCore release.

        Args:
            downsample: Event-level MC downsampling factor (weights
                rescaled to compensate), see
                ``tpeanuts.experiments.icecube.event_rate.DEFAULT_DOWNSAMPLE``.
            context: RuntimeContext every tensor this model builds is
                on.
            detector_depth_m: Detector depth below the Earth's
                surface, metres.
            earth_method, earth_nsteps, extensions: See the class docstring.

        Returns:
            A configured ``IceCubeModel``, with
            theta12/theta13/delta13/DeltamSq21 fixed at the real NuFit
            values ``tpeanuts.experiments.icecube.parameters`` documents.
        """
        channels = {
            channel: build_channel_events(channel, downsample=downsample, context=context)
            for channel in CHANNELS
        }
        hypersurfaces = {
            channel: build_hypersurface_table(channel, context=context) for channel in CHANNELS
        }
        muon_background = muon_background_histogram(context=context)

        params = IceCubeParameters.default(context=context)
        return cls(
            channels=channels,
            hypersurfaces=hypersurfaces,
            muon_background=muon_background,
            theta12=math.radians(params.theta12_deg),
            theta13=math.radians(params.theta13_deg),
            delta13=math.radians(params.delta13_deg),
            DeltamSq21=params.deltamsq21,
            context=context,
            detector_depth_m=detector_depth_m,
            earth_method=earth_method,
            earth_nsteps=earth_nsteps,
            extensions=extensions,
        )

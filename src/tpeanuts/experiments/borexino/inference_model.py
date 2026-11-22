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

"""Differentiable Borexino event-count model: composes oscillation x detector.

Ported from v1_0's ``experiments/borexino/inference_model.py``
(``BorexinoModel``). v1_0 composes a separate
``inference.model_solar.SolarSMOscillationModel`` object and calls its
``.oscillation(theta)`` method; this project's own
``tpeanuts.inference.model_solar`` collapsed that oscillation
-parameter step into a private ``_build_model`` on the point
-prediction model itself (``SolarPointModel``/``SolarNSIPointModel``,
see that module's own docstring), so there is no public
oscillation-only class left to compose over. Following
``tpeanuts.experiments.dayabay.inference_model``'s own precedent (its
``_vacuum_p_ee`` builds mixing/mass_spectrum/``PropagationModel``
inline rather than depending on another model's private method),
``BorexinoModel`` below builds its own ``PropagationModel``
inline too, instead of reaching into ``SolarPointModel``'s internals.

Only ``theta12``, ``theta13``, ``DeltamSq21`` and ``DeltamSq3l``
affect P_ee; ``theta23`` and the CP phase drop out of the survival
probability and are held fixed (matching v1_0's own documented
reasoning), not exposed as free parameters.

v1_0's own ``predict`` always adds a zero background via
``experiments.borexino.backgrounds.backgrounds_MeV``; that module was
never ported to v2 (see
``tpeanuts.experiments.borexino.event_rate``'s own module docstring --
``background_counts=None`` already means "no background" without a
placeholder tensor, the same choice already made for the IBD detector
in round 1).

Module contents:
    BorexinoModel
        Wraps oscillation parameters and a solar profile/source,
        predicting Borexino counts per bin from a free-parameter
        vector through ``source.flux -> medium.propagate ->
        detector.observe``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from tpeanuts.experiments.borexino.event_rate import es_detector
from tpeanuts.experiments.borexino.parameters import BorexinoParameters
from tpeanuts.medium.solar.solar import SolarMedium
from tpeanuts.physics.common.mass_spectrum import MassSpectrum
from tpeanuts.physics.common.mixing import MixingMatrix, StandardMixingParameters
from tpeanuts.physics.common.nature import Dirac
from tpeanuts.physics.common.space import FlavourSpace
from tpeanuts.physics.model import PropagationModel
from tpeanuts.pipeline._legs import SolarLeg
from tpeanuts.pipeline.solar import solar_flux
from tpeanuts.source.solar.solar import SolarSource
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import TensorLike

Tensor = torch.Tensor

_OSC_PARAM_KEYS: Tuple[str, ...] = ("theta12", "theta13", "DeltamSq21", "DeltamSq3l")


@dataclass(frozen=True)
class BorexinoModel:
    """Predicted Borexino counts per bin, as a function of oscillation parameters.

    Attributes:
        params: Loaded Borexino parameters.
        free: Names of the parameters to fit, a subset of
            ``_OSC_PARAM_KEYS`` (``"theta12"``, ``"theta13"``,
            ``"DeltamSq21"``, ``"DeltamSq3l"``), in the order
            ``predict``'s ``theta`` argument is unpacked.
        fixed: Values (radians for angles, eV^2 for the splittings)
            for every key of ``_OSC_PARAM_KEYS`` not in ``free``.
        theta23, delta13: Fixed atmospheric angle and CP phase, in
            radians -- do not affect P_ee, only needed to build a
            well-formed PMNS matrix.
        medium: Solar density profile (independent of ``theta``).
        source: Solar production source (radius/flux/spectrum tables,
            independent of ``theta``).
        source_names: Solar source keys whose signal contributions are
            summed before the (optional) background is added once.
        bin_edges_MeV: Observed-spectrum bin edges passed to
            ``tpeanuts.experiments.borexino.event_rate.es_detector``.
        context: RuntimeContext every tensor this model builds is on.
        background_counts: Optional background per bin; None applies
            no background (see module docstring).
        extensions: Optional propagation extensions (for example
            ``tpeanuts.physics.extensions.nsi.NSITerm``), forwarded to the
            ``PropagationModel`` this model builds, so they act in the Sun
            (through ``SolarLeg(method="adiabatic_exact")``, the only solar
            method that supports an extension). Empty (the
            default) is the Standard Model and changes nothing.

    Raises:
        ValueError: If ``free``/``fixed`` do not together cover exactly
            ``_OSC_PARAM_KEYS`` with no overlap and no gap, or if
            ``source_names`` is empty.
    """

    params: BorexinoParameters
    free: Tuple[str, ...]
    fixed: Mapping[str, TensorLike]
    theta23: TensorLike
    delta13: TensorLike
    medium: SolarMedium
    source: SolarSource
    source_names: Tuple[str, ...]
    bin_edges_MeV: Tensor
    context: RuntimeContext
    background_counts: Optional[Tensor] = None
    extensions: Tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        free_set = set(self.free)
        fixed_set = set(self.fixed)
        if len(free_set) != len(self.free):
            raise ValueError(f"free contains a duplicate parameter name: {self.free!r}.")
        if free_set & fixed_set:
            raise ValueError(
                f"free and fixed both name {sorted(free_set & fixed_set)} -- "
                "each parameter must be in exactly one of them."
            )
        covered = free_set | fixed_set
        if covered != set(_OSC_PARAM_KEYS):
            missing = set(_OSC_PARAM_KEYS) - covered
            unknown = covered - set(_OSC_PARAM_KEYS)
            raise ValueError(
                f"free/fixed together must cover exactly {_OSC_PARAM_KEYS}; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}."
            )
        if not self.source_names:
            raise ValueError("source_names must contain at least one solar source")

    def _build_model(self, theta: Tensor) -> PropagationModel:
        params = dict(self.fixed)
        params.update(zip(self.free, theta.unbind(), strict=True))

        space = FlavourSpace()
        mixing = MixingMatrix(
            space=space,
            standard=StandardMixingParameters(
                theta12=params["theta12"], theta13=params["theta13"],
                theta23=self.theta23, delta13=self.delta13,
                context=self.context,
            ),
            nature=Dirac(),
            context=self.context,
        )
        mass_spectrum = MassSpectrum(
            space=space, DeltamSq21=params["DeltamSq21"], DeltamSq3l=params["DeltamSq3l"], context=self.context,
        )
        return PropagationModel(mixing=mixing, mass_spectrum=mass_spectrum, extensions=self.extensions)

    def _solar_options(self) -> Dict[str, Any]:
        """Keyword arguments of ``solar_flux``: the exact adiabatic leg if an extension is active, none otherwise."""
        return {"solar_leg": SolarLeg(method="adiabatic_exact")} if self.extensions else {}

    def predict(self, theta: Tensor) -> Tensor:
        """Predict Borexino counts per bin from ``theta``.

        Args:
            theta: 1-D free-parameter tensor, see ``self.free``.

        Returns:
            Predicted counts per bin, shape ``(n_bins,)``.
        """
        model = self._build_model(theta)
        folded = es_detector(self.params, self.bin_edges_MeV)
        counts = torch.zeros(self.bin_edges_MeV.shape[0] - 1, device=theta.device, dtype=theta.dtype)

        for source_name in self.source_names:
            arrived = solar_flux(
                model, self.source, self.medium, source_name, self.params.e_nu_grid_MeV, **self._solar_options(),
            )
            counts = counts + folded.observe(arrived).counts

        if self.background_counts is not None:
            counts = counts + self.background_counts.to(device=theta.device, dtype=theta.dtype)
        return counts

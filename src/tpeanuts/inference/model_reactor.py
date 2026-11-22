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

"""The first concrete DifferentiableModel: reactor IBD event rate vs. oscillation parameters.

Wraps exactly the source->medium->propagation->detector chain
``tpeanuts.test.test_reactor_vacuum_benchmark`` already validates
(``ReactorSource`` -> ``VacuumMedium`` -> ``PropagationModel`` ->
``tpeanuts.detector.ibd.IBDDetector``, the first three being
``tpeanuts.pipeline.reactor.reactor_flux``) behind
``tpeanuts.inference.model.DifferentiableModel``'s ``free``/``predict(theta)``
surface -- no new physics, only a differentiable re-composition of pieces
that already exist and are already tested individually.

Recognizes exactly six parameter keys (``_PARAM_KEYS``): the four standard
mixing angles/phase and the two mass-squared splittings
``PropagationModel``'s plain-SM path needs (matching
``tpeanuts.physics.common.space.FlavourSpace``'s own ``n_standard == 3``
requirement -- no sterile/NSI/magnetic-moment/decoherence extension is
wired in here, since none of those add a free parameter this first model
needs to fit). Every key must appear in exactly one of ``free``/``fixed``.

Module contents:
    ReactorEventRateModel
        Frozen dataclass: which parameters are free vs. fixed, the energy
        grid, baseline, and target proton count, with a ``predict(theta)``
        method returning ``rate_per_MeV``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import torch

from tpeanuts.detector.ibd import IBDDetector
from tpeanuts.physics.common.mass_spectrum import MassSpectrum
from tpeanuts.physics.common.mixing import MixingMatrix, StandardMixingParameters
from tpeanuts.physics.common.nature import Dirac
from tpeanuts.physics.common.space import FlavourSpace
from tpeanuts.physics.model import PropagationModel
from tpeanuts.pipeline.reactor import reactor_flux
from tpeanuts.source.reactor.reactor import ReactorSource
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import TensorLike

_PARAM_KEYS = ("theta12", "theta13", "theta23", "delta13", "DeltamSq21", "DeltamSq3l")


@dataclass(frozen=True)
class ReactorEventRateModel:
    """IBD event-rate density vs. energy, differentiable w.r.t. a chosen subset of oscillation parameters.

    Attributes:
        free: Names of the parameters to fit, a subset of ``_PARAM_KEYS``
            (``"theta12"``, ``"theta13"``, ``"theta23"``, ``"delta13"``,
            ``"DeltamSq21"``, ``"DeltamSq3l"``), in the order ``predict``'s
            ``theta`` argument is unpacked.
        fixed: Values (radians for angles/phase, eV^2 for the splittings)
            for every key in ``_PARAM_KEYS`` not in ``free``.
        energy_MeV: True antineutrino energy grid, shaped ``(n_energy,)``.
        baseline_m: Source-to-detector baseline, in metres (vacuum, one
            segment -- see ``tpeanuts.medium.vacuum.vacuum.VacuumMedium``). Also
            used as ``evolution_scale_m``, matching
            ``test_reactor_vacuum_benchmark.py``'s own convention.
        n_target_protons: Forwarded to ``IBDDetector``.
        context: RuntimeContext every tensor this model builds is on.
        flux_curves: Optional tabulated per-isotope antineutrino spectra,
            ``{isotope: (E_tab_MeV, N_per_fission_per_MeV)}`` for every isotope of
            ``tpeanuts.source.reactor.reactor.ISOTOPES``.
        fission_fractions: Optional ``{isotope: fraction}``; required together with ``flux_curves``.
        fissions_per_second: Optional fission rate of the core; required together with ``flux_curves``.

    Without ``flux_curves`` the prediction is the response of the target to a *unit* flux, ``n sigma(E) P_ee(E)``: a
    shape with no reactor spectrum in it, enough to fit oscillation parameters from the distortion it implies but not
    an event rate. With them (and the fission fractions and rate) the source emits the Huber-Mueller spectrum, the
    baseline dilutes it over ``4 pi L^2``, and the prediction is the absolute rate in events per second per MeV,
    ``IBDDetector.observe`` of the arriving flux.

    Raises:
        ValueError: If ``free``/``fixed`` do not together cover exactly
            ``_PARAM_KEYS`` with no overlap and no gap, or if ``flux_curves``, ``fission_fractions`` and
            ``fissions_per_second`` are not given all together or not at all.
    """

    free: Tuple[str, ...]
    fixed: Mapping[str, TensorLike]
    energy_MeV: TensorLike
    baseline_m: TensorLike
    n_target_protons: TensorLike
    context: RuntimeContext
    flux_curves: Optional[Mapping[str, Tuple[torch.Tensor, torch.Tensor]]] = None
    fission_fractions: Optional[Mapping[str, TensorLike]] = None
    fissions_per_second: Optional[TensorLike] = None

    def __post_init__(self) -> None:
        spectrum = (self.flux_curves, self.fission_fractions, self.fissions_per_second)
        if any(item is not None for item in spectrum) and any(item is None for item in spectrum):
            raise ValueError(
                "flux_curves, fission_fractions and fissions_per_second must be given all together or not at all."
            )
        free_set = set(self.free)
        fixed_set = set(self.fixed)
        if len(free_set) != len(self.free):
            raise ValueError(f"free contains a duplicate parameter name: {self.free!r}.")
        if free_set & fixed_set:
            raise ValueError(
                f"free and fixed both name {sorted(free_set & fixed_set)} -- "
                "each of _PARAM_KEYS must be in exactly one of them."
            )
        covered = free_set | fixed_set
        if covered != set(_PARAM_KEYS):
            missing = set(_PARAM_KEYS) - covered
            unknown = covered - set(_PARAM_KEYS)
            raise ValueError(
                f"free/fixed together must cover exactly {_PARAM_KEYS}; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}."
            )

    def predict(self, theta: torch.Tensor) -> torch.Tensor:
        """Build the IBD event-rate density at the given free-parameter vector.

        Args:
            theta: Free-parameter vector, shaped ``(len(self.free),)``,
                same order as ``self.free``.

        Returns:
            ``rate_per_MeV``, shaped like ``self.energy_MeV``, differentiable
            w.r.t. ``theta``.
        """
        params = dict(self.fixed)
        params.update(zip(self.free, theta.unbind(), strict=True))

        space = FlavourSpace()
        mixing = MixingMatrix(
            space=space,
            standard=StandardMixingParameters(
                theta12=params["theta12"], theta13=params["theta13"],
                theta23=params["theta23"], delta13=params["delta13"],
                context=self.context,
            ),
            nature=Dirac(),
            context=self.context,
        )
        mass_spectrum = MassSpectrum(
            space=space, DeltamSq21=params["DeltamSq21"], DeltamSq3l=params["DeltamSq3l"],
            context=self.context,
        )

        source = ReactorSource(energy_MeV=self.energy_MeV, context=self.context)
        detector = IBDDetector(n_target_protons=self.n_target_protons, context=self.context)
        model = PropagationModel(mixing=mixing, mass_spectrum=mass_spectrum, antinu=True)

        if self.flux_curves is None:
            flux = reactor_flux(model, source, self.baseline_m, dilute=False)
            P_ee = flux.state.probabilities()[..., 0]
            return detector.event_rate(flux.energy_MeV, P_ee).rate_per_MeV

        flux = reactor_flux(
            model, source, self.baseline_m, curves=self.flux_curves, fission_fractions=self.fission_fractions,
            fissions_per_second=self.fissions_per_second,
        )
        return detector.observe(flux).rate_per_MeV

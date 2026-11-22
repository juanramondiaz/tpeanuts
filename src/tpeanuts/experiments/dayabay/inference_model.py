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

"""Differentiable Daya Bay event-count model: 6-reactor vacuum oscillation x detector.

``DayaBayDetectorModel`` wraps one of the 8 Daya Bay detectors: it
evaluates the vacuum nu_e_bar survival probability once per reactor (6
calls to the private ``_vacuum_p_ee``, each at that reactor's own
baseline), then folds the resulting flux sum through the IBD cross
section, response and background via
``tpeanuts.experiments.dayabay.event_rate.ibd_event_rate``.
``DayaBayJointModel`` concatenates all 8 detectors' predictions for a
combined fit.

Unlike v1_0's own version of this module (which wraps a separate
``inference.model_vacuum.VacuumOscillationModel``), ``_vacuum_p_ee``
here builds the mixing/mass-spectrum/vacuum-medium chain inline, the
same wiring
``tpeanuts.inference.model_reactor.ReactorEventRateModel.predict``
already uses and this project's own tests already validate -- there is
no separate oscillation-model class to port. Only ``theta12``,
``theta13``, ``DeltamSq21`` and ``DeltamSq3l`` affect P_ee; ``theta23``
and the CP phase drop out of the survival probability and are held
fixed per-model, not exposed as free parameters.

**Why the 6 reactors are still a Python loop here, unlike
``tpeanuts.experiments.dayabay.flux``'s own batched
``reactor_flux_at_face``**: confirmed directly (see this module's own
development notes) that passing a batched ``(n_reactor,)`` baseline as
``evolution_scale_m`` alongside a batched ``(n_E,)`` energy grid into
``tpeanuts.physics.model.PropagationModel.generator_flavour`` collides
with the mass-spectrum's own flavour axis inside
``tpeanuts.physics.common.potential.kinetic_potential`` (both are
plain, un-labelled tensor axes there, and ``kinetic_potential`` was
never designed to keep three independent batch axes -- energy,
reactor, flavour -- apart at once): with ``n_reactor=6`` and
``n_flavour=3``, this raises a shape-mismatch error immediately rather
than silently mixing the two, so the failure is safe, but it does mean
this specific batching would need a change to shared, already-tested
core physics machinery, out of scope for this port. A per-reactor loop
(exactly what v1_0 itself does too) is the correct, verified choice
here instead.

Module contents:
    DayaBayDetectorModel
        One detector's predicted counts per analysis bin, as a function
        of oscillation parameters (+ global_normalization, + background
        -rate nuisances, + LSNL pull-curve nuisances).
    DayaBayJointModel
        Concatenates several DayaBayDetectorModel predictions sharing
        one theta -- pass all 8 detectors for the full combined fit.
    DayaBayReparametrizedModel
        Reparametrizes a Daya Bay model's free vector as
        (sin^2(2 theta13), Delta m^2_ee) instead of the bare PMNS
        (theta13, Delta m^2_3l) -- Daya Bay's own reported observables.
    DayaBayNearFarRatioModel
        Predicts the far-hall/near-hall per-bin count ratio instead of
        absolute counts, cancelling systematics common to every
        detector the way Daya Bay's own original near/far measurement
        strategy does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple, Union, cast

import torch

from tpeanuts.experiments.dayabay.event_rate import ibd_detector, real_observed_counts
from tpeanuts.experiments.dayabay.flux import reactor_fluxes
from tpeanuts.experiments.dayabay.parameters import (
    BG_CATEGORIES,
    FAR_DETECTORS,
    NEAR_DETECTORS,
    DayaBayParameters,
)
from tpeanuts.interfaces.flux import SourceFlux
from tpeanuts.medium.vacuum.vacuum import VacuumMedium
from tpeanuts.physics.common.mass_spectrum import MassSpectrum
from tpeanuts.physics.common.mixing import MixingMatrix, StandardMixingParameters
from tpeanuts.physics.common.nature import Dirac
from tpeanuts.physics.common.space import FlavourSpace
from tpeanuts.physics.common.state import DensityMatrix
from tpeanuts.physics.model import PropagationModel
from tpeanuts.util.tensor import TensorLike

Tensor = torch.Tensor

_OSC_PARAM_KEYS: Tuple[str, ...] = ("theta12", "theta13", "DeltamSq21", "DeltamSq3l")
_ELECTRON_INDEX = 0

NORMALIZATION_PARAM: str = "global_normalization"
BACKGROUND_NUISANCE_PARAMS: Tuple[str, ...] = tuple(f"bg_scale_{category}" for category in BG_CATEGORIES)
LSNL_PULL_PARAMS: Tuple[str, ...] = tuple(f"lsnl_pull_{k}" for k in range(4))


def _vacuum_p_ee(
    mixing: MixingMatrix, mass_spectrum: MassSpectrum, E_grid_MeV: Tensor, baseline_m: TensorLike,
) -> Tensor:
    """Vacuum nu_e_bar survival probability P_ee(E) at one baseline.

    Exactly ``tpeanuts.inference.model_reactor.ReactorEventRateModel.
    predict``'s own wiring (mixing/mass_spectrum -> ``VacuumMedium`` ->
    ``PropagationModel.generator_flavour`` ->
    ``segment_evolutor_numerical`` -> conjugate a pure nu_e state),
    extracted as a reusable function since ``DayaBayDetectorModel``
    calls it once per reactor with the same ``mixing``/``mass_spectrum``
    but a different baseline.

    Args:
        mixing: Built MixingMatrix (``antinu=True`` is applied via
            ``PropagationModel`` below, not here).
        mass_spectrum: Built MassSpectrum, sharing ``mixing``'s space
            and context.
        E_grid_MeV: True antineutrino energy grid, shape ``(n_E,)``.
        baseline_m: Source-to-detector baseline, in metres (scalar).

    Returns:
        Real tensor shaped ``(n_E,)``, differentiable w.r.t. ``mixing``/
        ``mass_spectrum``'s own parameters.
    """
    context = mixing.context
    model = PropagationModel(mixing=mixing, mass_spectrum=mass_spectrum, antinu=True)
    evolutor = VacuumMedium(length_m=baseline_m, context=context).evolutor(model, E_grid_MeV)
    state = DensityMatrix.basis_state(_ELECTRON_INDEX, mixing.space, context)
    evolved = state.conjugate_by(evolutor)
    return evolved.probabilities()[..., 0]


@dataclass(frozen=True)
class DayaBayDetectorModel:
    """Predicted Daya Bay IBD counts per analysis bin for one detector.

    Attributes:
        params: Loaded Daya Bay parameters.
        detector: Detector name, e.g. ``"AD11"``.
        oscillation_free: Names (subset of ``("theta12", "theta13",
            "DeltamSq21", "DeltamSq3l")``, in order) read from
            ``theta``'s own leading entries.
        oscillation_fixed: Values (radians for angles, eV^2 for the
            splittings) for every key of that 4-tuple not in
            ``oscillation_free``.
        theta23, delta13: Fixed angle/phase, radians -- do not affect
            P_ee (see module docstring), never exposed as free
            parameters.
        n_target, exposure_seconds, background_counts: Optional
            overrides forwarded to ``ibd_event_rate``; None uses that
            function's own defaults.
        normalization_free: If True (default), ``theta`` carries the
            ``global_normalization`` nuisance right after the
            oscillation parameters; if False, the signal is left
            unscaled (``signal_scale=1.0``).
        background_free: If True (default False), ``theta``'s next 5
            entries (after ``global_normalization``, if present) are
            the per-category background-rate nuisances,
            ``BACKGROUND_NUISANCE_PARAMS``; if False, every category is
            left at its nominal rate (scale 1.0).
        lsnl_free: If True (default False), ``theta``'s final 4 entries
            are the LSNL pull-curve nuisances, ``LSNL_PULL_PARAMS``
            (official prior: independent, zero-mean, unit-sigma); if
            False, the nominal LSNL curve is used unchanged.

    Raises:
        ValueError: If ``oscillation_free``/``oscillation_fixed`` do
            not together cover exactly ``("theta12", "theta13",
            "DeltamSq21", "DeltamSq3l")`` with no overlap and no gap.
    """

    params: DayaBayParameters
    detector: str
    oscillation_free: Tuple[str, ...]
    oscillation_fixed: Mapping[str, TensorLike]
    theta23: TensorLike
    delta13: TensorLike
    n_target: Optional[Tensor] = None
    exposure_seconds: Optional[Tensor] = None
    background_counts: Optional[Tensor] = None
    normalization_free: bool = True
    background_free: bool = False
    lsnl_free: bool = False

    def __post_init__(self) -> None:
        free_set = set(self.oscillation_free)
        fixed_set = set(self.oscillation_fixed)
        if len(free_set) != len(self.oscillation_free):
            raise ValueError(f"oscillation_free contains a duplicate parameter name: {self.oscillation_free!r}.")
        if free_set & fixed_set:
            raise ValueError(
                f"oscillation_free and oscillation_fixed both name {sorted(free_set & fixed_set)} -- "
                "each of _OSC_PARAM_KEYS must be in exactly one of them."
            )
        covered = free_set | fixed_set
        if covered != set(_OSC_PARAM_KEYS):
            missing = set(_OSC_PARAM_KEYS) - covered
            unknown = covered - set(_OSC_PARAM_KEYS)
            raise ValueError(
                f"oscillation_free/oscillation_fixed together must cover exactly {_OSC_PARAM_KEYS}; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}."
            )

    @property
    def free(self) -> Tuple[str, ...]:
        """Free parameter names: oscillation (+ normalization) (+ background) (+ LSNL pulls)."""
        free = self.oscillation_free
        if self.normalization_free:
            free = free + (NORMALIZATION_PARAM,)
        if self.background_free:
            free = free + BACKGROUND_NUISANCE_PARAMS
        if self.lsnl_free:
            free = free + LSNL_PULL_PARAMS
        return free

    def predict(self, theta: Tensor) -> Tensor:
        """Predict this detector's analysis-bin counts from ``theta``.

        Args:
            theta: 1-D free-parameter tensor in ``self.free`` order.

        Returns:
            Predicted counts per analysis bin, shape ``(n_bins,)``.
        """
        context = self.params.context
        values = dict(self.oscillation_fixed)
        n_osc = len(self.oscillation_free)
        values.update(zip(self.oscillation_free, theta[:n_osc].unbind(), strict=True))
        rest = theta[n_osc:]

        if self.normalization_free:
            signal_scale, rest = rest[0], rest[1:]
        else:
            signal_scale = torch.ones((), dtype=theta.dtype, device=theta.device)

        if self.background_free:
            category_scale = dict(zip(BG_CATEGORIES, rest[: len(BACKGROUND_NUISANCE_PARAMS)].unbind(), strict=True))
            rest = rest[len(BACKGROUND_NUISANCE_PARAMS):]
        else:
            category_scale = None

        lsnl_pulls = rest if self.lsnl_free else None

        space = FlavourSpace()
        mixing = MixingMatrix(
            space=space,
            standard=StandardMixingParameters(
                theta12=values["theta12"], theta13=values["theta13"],
                theta23=self.theta23, delta13=self.delta13,
                context=context,
            ),
            nature=Dirac(),
            context=context,
        )
        mass_spectrum = MassSpectrum(
            space=space, DeltamSq21=values["DeltamSq21"], DeltamSq3l=values["DeltamSq3l"], context=context,
        )

        # source -> vacuum (one leg per core, at its own baseline) -> detector
        model = PropagationModel(mixing=mixing, mass_spectrum=mass_spectrum, antinu=True)
        emitted = reactor_fluxes(self.params, self.params.e_nu_grid_MeV)
        arrived = SourceFlux({
            reactor: VacuumMedium(
                length_m=self.params.baselines_km[self.detector][reactor] * 1000.0, context=context,
            ).propagate(flux, model, dilute=True)
            for reactor, flux in emitted.items()
        })
        folded = ibd_detector(
            self.params, self.detector,
            n_target=self.n_target, exposure_seconds=self.exposure_seconds,
            background_counts=self.background_counts,
            signal_scale=signal_scale, background_category_scale=category_scale, lsnl_pulls=lsnl_pulls,
        )
        return folded.observe(arrived).counts


@dataclass(frozen=True)
class DayaBayJointModel:
    """Concatenates several ``DayaBayDetectorModel`` predictions sharing one theta.

    Attributes:
        models: ``DayaBayDetectorModel`` instances (typically all 8
            detectors), evaluated at the same ``theta`` and
            concatenated in the given order.
    """

    models: Tuple[DayaBayDetectorModel, ...]

    @property
    def free(self) -> Tuple[str, ...]:
        """Free oscillation-parameter names, forwarded from the first model."""
        return self.models[0].free

    def predict(self, theta: Tensor) -> Tensor:
        """Predict concatenated counts per bin across every wrapped detector.

        Args:
            theta: 1-D free-parameter tensor, shared by every wrapped
                model.

        Returns:
            Predicted counts, shape ``(sum of each model's n_bins,)``,
            in ``self.models`` order.
        """
        return torch.cat([model.predict(theta) for model in self.models])


EXPERIMENTAL_PARAM_KEYS: Tuple[str, ...] = ("SinSq2Theta13", "DeltamSqEE")


@dataclass(frozen=True)
class DayaBayReparametrizedModel:
    """Reparametrizes a Daya Bay model's first two free parameters as (sin^2(2 theta13), Delta m^2_ee).

    Daya Bay's own combined analyses report their oscillation result as
    ``sin^2(2 theta13)`` and the effective splitting

        ``Delta m^2_ee = cos^2(theta12) Delta m^2_31 + sin^2(theta12) Delta m^2_32``,

    not the bare PMNS ``(theta13, Delta m^2_3l)`` a ``DayaBayDetectorModel``
    natively exposes. With ``theta12``/``Delta m^2_21`` fixed, the two
    parametrizations carry identical information -- ``Delta m^2_ee`` and
    ``Delta m^2_31`` differ only by the fixed offset

        ``Delta m^2_31 = Delta m^2_ee + sin^2(theta12) Delta m^2_21``

    so this wrapper is a pure change of variables, not a different fit.

    Attributes:
        model: A ``DayaBayDetectorModel`` or ``DayaBayJointModel`` whose
            ``free`` starts with ``("theta13", "DeltamSq3l")`` (any
            further entries, e.g. ``global_normalization``/background
            nuisances, are passed through unchanged).
        theta12, dm21: The same fixed values used to build ``model``'s
            own oscillation parameters (radians, eV^2).

    Raises:
        ValueError: If ``model.free`` does not start with
            ``("theta13", "DeltamSq3l")``.
    """

    model: Union[DayaBayDetectorModel, DayaBayJointModel]
    theta12: Tensor
    dm21: Tensor

    def __post_init__(self) -> None:
        if self.model.free[:2] != ("theta13", "DeltamSq3l"):
            raise ValueError(
                "DayaBayReparametrizedModel requires model.free to start with "
                f"('theta13', 'DeltamSq3l'), got {self.model.free!r}."
            )

    @property
    def free(self) -> Tuple[str, ...]:
        """Free parameter names: (SinSq2Theta13, DeltamSqEE) + model.free[2:]."""
        return EXPERIMENTAL_PARAM_KEYS + self.model.free[2:]

    def predict(self, theta: Tensor) -> Tensor:
        """Convert ``theta``'s first two entries to (theta13, DeltamSq3l) and delegate.

        Args:
            theta: 1-D free-parameter tensor in ``self.free`` order.

        Returns:
            ``self.model.predict(...)``'s own return shape.
        """
        sinsq2theta13, dm2_ee = theta[0], theta[1]
        theta13 = 0.5 * torch.asin(torch.sqrt(sinsq2theta13))
        dm31 = dm2_ee + torch.sin(self.theta12) ** 2 * self.dm21
        theta_native = torch.cat([theta13[None], dm31[None], theta[2:]])
        return self.model.predict(theta_native)


@dataclass(frozen=True)
class DayaBayNearFarRatioModel:
    """Predicts the far-hall/near-hall per-analysis-bin count ratio.

    Daya Bay's original 2012 discovery of theta13 was a near/far
    comparison: EH1 (AD11/AD12) and EH2 (AD21/AD22) sit at baselines
    short enough (~560/600 m) to be only weakly oscillated, while EH3
    (AD31-AD34, ~1640 m) sees the full oscillation deficit -- so a
    far/near ratio isolates the oscillation signal while cancelling, by
    construction, every systematic common to every detector:

        ``R_k(theta) = N_k^far(theta) / N_k^near(theta)``,
        ``N_k^{near,far}(theta) = sum_{d in NEAR_DETECTORS,FAR_DETECTORS} N_{k,d}(theta)``,

    with ``N_{k,d}`` each detector's own full per-bin prediction
    (``DayaBayDetectorModel.predict``). Because every detector shares
    the same response/cross-section/efficiency functions, a shared
    multiplicative factor -- in particular ``global_normalization`` --
    cancels exactly in the ratio. This is a simplified version of Daya
    Bay's own original near/far strategy, not a full multi-baseline
    global fit.

    Attributes:
        near_models, far_models: ``DayaBayDetectorModel`` instances for
            ``NEAR_DETECTORS``/``FAR_DETECTORS`` respectively, all
            sharing the same oscillation-parameter setup (hence the
            same ``free``).
    """

    near_models: Tuple[DayaBayDetectorModel, ...]
    far_models: Tuple[DayaBayDetectorModel, ...]

    @property
    def free(self) -> Tuple[str, ...]:
        """Free parameter names, forwarded from the first near-hall model."""
        return self.near_models[0].free

    def predict(self, theta: Tensor) -> Tensor:
        """Predict the far/near per-bin count ratio from ``theta``.

        Args:
            theta: 1-D free-parameter tensor, shared by every wrapped
                model.

        Returns:
            Tensor shaped ``(n_bins,)``, ``N_far(theta) / N_near(theta)``.
        """
        near_total = cast(Tensor, sum(model.predict(theta) for model in self.near_models))
        far_total = cast(Tensor, sum(model.predict(theta) for model in self.far_models))
        return far_total / near_total

    @staticmethod
    def real_observed_ratio(params: DayaBayParameters) -> Tuple[Tensor, Tensor]:
        """Observed far/near per-bin count ratio and its propagated 1-sigma uncertainty.

        Args:
            params: Loaded Daya Bay parameters.

        Returns:
            ``(ratio, sigma)``, each shape ``(n_bins,)``. ``sigma`` is
            the standard independent-Poisson-numerator/denominator
            propagation, ``ratio * sqrt(1/N_far + 1/N_near)``.
        """
        near_total = cast(Tensor, sum(real_observed_counts(params, detector) for detector in NEAR_DETECTORS))
        far_total = cast(Tensor, sum(real_observed_counts(params, detector) for detector in FAR_DETECTORS))
        ratio = far_total / near_total
        sigma = ratio * torch.sqrt(1.0 / far_total + 1.0 / near_total)
        return ratio, sigma

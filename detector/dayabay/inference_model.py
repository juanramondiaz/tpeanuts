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
Differentiable Daya Bay event-count model: real 6-reactor vacuum oscillation x real detector.

``DayaBayDetectorModel`` wraps one of the 8 real Daya Bay detectors around a
``tpeanuts.medium.vacuum.oscillation_model.VacuumOscillationModel``: it
evaluates ``VacuumOscillationModel.predict_pee`` once per reactor (6 cheap
vacuum-probability calls, each at that reactor's own real baseline), then
calls ``tpeanuts.detector.dayabay.event_rate.ibd_event_rate`` to fold the
real 6-reactor flux sum through the real IBD cross section, real response,
and real background. ``JointDayaBayModel`` concatenates all 8 detectors'
predictions for a single combined fit.

When ``normalization_free=True`` (the default), ``DayaBayDetectorModel``
appends one extra free parameter, ``global_normalization``, to
``oscillation_model.free``: Daya Bay's own real, official free-normalization
nuisance (``parameters/detector_normalization.yaml``, nominal value
``detector.dayabay.parameters.GLOBAL_NORMALIZATION_NOMINAL`` = 1.0), fit
alongside the oscillation parameters rather than fixed. Without it, a
rate-only (or even a full-shape) fit with millions of real events has
essentially zero Poisson error on the total rate and is forced to distort
the oscillation parameters themselves to absorb the real, documented ~20%
gap between this project's absolute prediction and the observed total (see
``detector.dayabay.event_rate`` module docstring) -- exactly the failure
mode ``normalization_free`` exists to avoid.

When ``background_free=True`` (default False), ``DayaBayDetectorModel``
further appends 5 real per-category background-rate nuisances,
``BACKGROUND_NUISANCE_PARAMS`` (one real correlated scale per
``detector.dayabay.parameters.BG_CATEGORIES`` entry, nominal 1.0), meant to
be fit with a Gaussian prior of width
``detector.dayabay.parameters.BACKGROUND_CATEGORY_SIGMA`` (see
``tpeanuts.inference.likelihood.gaussian_prior_penalty`` and
``tpeanuts.inference.fit.fit_lbfgs``'s ``penalty_fn``) -- a real, externally
measured constraint, not a free-floating extra degree of freedom. Even after
``normalization_free`` fixes the rate-only fit, a full-shape fit still runs
to unphysical oscillation parameters at Daya Bay's real ~3-million-event
statistics, because a single flat ``global_normalization`` cannot absorb
energy-dependent residuals; letting the background-rate
nuisances float (within their real uncertainties) gives the fit real degrees
of freedom to absorb some of that residual shape mismatch instead of
distorting theta13/DeltamSq31.

Module contents:
    DayaBayDetectorModel
        One real detector's predicted counts per analysis bin, as a
        function of oscillation parameters (+ global_normalization,
        + background-rate nuisances, + LSNL pull-curve nuisances).
    DayaBayExperimentalModel
        Reparametrizes a Daya Bay model's free vector as
        (sin^2(2 theta13), Delta m^2_ee) instead of the bare PMNS
        (theta13, Delta m^2_3l) -- Daya Bay's own reported observables.
    JointDayaBayModel
        Concatenates several DayaBayDetectorModel predictions sharing one
        theta -- pass all 8 real detectors for the full combined fit.
    NearFarRatioDayaBayModel
        Predicts the real far-hall/near-hall per-bin count ratio instead of
        absolute counts -- cancels systematics common to every detector
        (``global_normalization``, and much of any residual IAV/LSNL/cross-
        section *shape* mismatch, exactly by construction) the way Daya
        Bay's own original near/far measurement strategy does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from tpeanuts.detector.dayabay.event_rate import ibd_event_rate, real_observed_counts
from tpeanuts.detector.dayabay.parameters import (
    BASELINES_KM,
    BG_CATEGORIES,
    E_NU_GRID_MEV,
    FAR_DETECTORS,
    FINAL_EREC_BIN_EDGES_MEV,
    NEAR_DETECTORS,
    N_PROTONS,
    REACTORS,
)
from tpeanuts.medium.vacuum.oscillation_model import VacuumOscillationModel

NORMALIZATION_PARAM: str = "global_normalization"
BACKGROUND_NUISANCE_PARAMS: tuple[str, ...] = tuple(f"bg_scale_{category}" for category in BG_CATEGORIES)
LSNL_PULL_PARAMS: tuple[str, ...] = tuple(f"lsnl_pull_{k}" for k in range(4))


@dataclass(frozen=True)
class DayaBayDetectorModel:
    """Predicted Daya Bay IBD counts per analysis bin for one real detector.

    Parameters
    ----------
    oscillation_model:
        A ``VacuumOscillationModel`` (vacuum, antinu=True) supplying
        ``free``/``predict_pee(theta, L_km, E_grid)``, shared by every
        detector and every one of its 6 reactors in a joint fit.
    detector:
        Detector name, e.g. "AD11" (see
        ``detector.dayabay.parameters.DETECTORS``).
    n_target, exposure_seconds, background_counts:
        Optional overrides forwarded to
        ``detector.dayabay.event_rate.ibd_event_rate``; None uses that
        function's own real-data defaults (real target protons, real
        8AD-period exposure, real background).
    normalization_free:
        If True (default), ``theta`` carries Daya Bay's own real
        ``global_normalization`` nuisance (see module docstring) right
        after the oscillation parameters; if False, it is omitted and the
        signal is left unscaled (``signal_scale=1.0``).
    background_free:
        If True (default False), ``theta``'s next 5 entries (after
        ``global_normalization``, if present) are the real per-category
        background-rate nuisances, ``BACKGROUND_NUISANCE_PARAMS`` (see
        module docstring); if False, every category is left at its real
        nominal rate (scale 1.0).
    lsnl_free:
        If True (default False), ``theta``'s final 4 entries are the real
        LSNL pull-curve nuisances, ``LSNL_PULL_PARAMS`` (official prior:
        independent, zero-mean, unit-sigma -- see
        ``detector.dayabay.response._lsnl_warp_matrix``); if False, the
        real nominal LSNL curve is used unchanged.
    """

    oscillation_model: VacuumOscillationModel
    detector: str
    n_target: Optional[torch.Tensor] = None
    exposure_seconds: Optional[torch.Tensor] = None
    background_counts: Optional[torch.Tensor] = None
    normalization_free: bool = True
    background_free: bool = False
    lsnl_free: bool = False

    @property
    def free(self) -> tuple[str, ...]:
        """Free parameter names: oscillation (+ normalization) (+ background) (+ LSNL pulls)."""
        free = self.oscillation_model.free
        if self.normalization_free:
            free = free + (NORMALIZATION_PARAM,)
        if self.background_free:
            free = free + BACKGROUND_NUISANCE_PARAMS
        if self.lsnl_free:
            free = free + LSNL_PULL_PARAMS
        return free

    def predict(self, theta: torch.Tensor) -> torch.Tensor:
        """Predict this detector's real analysis-bin counts from ``theta``.

        Args:
            theta: 1-D free-parameter tensor in ``self.free`` order (see
                ``self.oscillation_model.oscillation`` for the oscillation-
                parameter entries).

        Returns:
            Predicted counts per analysis bin, shape ``(n_bins,)``.
        """
        n_osc = len(self.oscillation_model.free)
        theta_osc = theta[:n_osc]
        rest = theta[n_osc:]

        if self.normalization_free:
            signal_scale, rest = rest[0], rest[1:]
        else:
            signal_scale = torch.ones((), dtype=theta.dtype, device=theta.device)

        if self.background_free:
            category_scale = dict(zip(BG_CATEGORIES, rest[: len(BACKGROUND_NUISANCE_PARAMS)].unbind()))
            rest = rest[len(BACKGROUND_NUISANCE_PARAMS):]
        else:
            category_scale = None

        lsnl_pulls = rest if self.lsnl_free else None

        baselines = BASELINES_KM[self.detector]
        p_ee_per_reactor = {
            reactor: self.oscillation_model.predict_pee(theta_osc, float(baselines[reactor]), E_NU_GRID_MEV)
            for reactor in REACTORS
        }
        return ibd_event_rate(
            self.detector, p_ee_per_reactor,
            bin_edges_MeV=FINAL_EREC_BIN_EDGES_MEV,
            n_target=self.n_target if self.n_target is not None else N_PROTONS[self.detector],
            exposure_seconds=self.exposure_seconds,
            background_counts=self.background_counts,
            signal_scale=signal_scale,
            background_category_scale=category_scale,
            lsnl_pulls=lsnl_pulls,
        )


@dataclass(frozen=True)
class JointDayaBayModel:
    """Concatenates several ``DayaBayDetectorModel`` predictions sharing one theta.

    Parameters
    ----------
    models:
        ``DayaBayDetectorModel`` instances (typically all 8 real
        detectors), evaluated at the same ``theta`` and concatenated in
        the given order.
    """

    models: tuple[DayaBayDetectorModel, ...]

    @property
    def free(self) -> tuple[str, ...]:
        """Free oscillation-parameter names, forwarded from the first model."""
        return self.models[0].free

    def predict(self, theta: torch.Tensor) -> torch.Tensor:
        """Predict concatenated counts per bin across every wrapped detector.

        Args:
            theta: 1-D free-parameter tensor, shared by every wrapped model.

        Returns:
            Predicted counts, shape ``(sum of each model's n_bins,)``, in
            ``self.models`` order.
        """
        return torch.cat([model.predict(theta) for model in self.models])


EXPERIMENTAL_PARAM_KEYS: tuple[str, ...] = ("SinSq2Theta13", "DeltamSqEE")


@dataclass(frozen=True)
class DayaBayExperimentalModel:
    """Reparametrizes a Daya Bay model's first two free parameters as (sin^2(2 theta13), Delta m^2_ee).

    Daya Bay's own combined analyses report (F. P. An et al., Phys. Rev.
    Lett. 130, 161802 (2023), the notebook's reference [2]; the ``Delta
    m^2_ee`` construction itself is due to H. Nunokawa, S. Parke, R.
    Zukanovich Funchal, Phys. Rev. D 72, 013009 (2005)) their oscillation
    result as ``sin^2(2 theta13)`` and the effective splitting

        Delta m^2_ee = cos^2(theta12) Delta m^2_31 + sin^2(theta12) Delta m^2_32,

    not the bare PMNS ``(theta13, Delta m^2_3l)`` this package's
    ``VacuumOscillationModel`` natively exposes. With ``theta12``/``Delta
    m^2_21`` fixed (this package's convention throughout, see
    ``detector.dayabay.inference_model`` module docstring), the two
    parametrizations carry identical information -- ``Delta m^2_ee`` and
    ``Delta m^2_31`` differ only by the fixed offset

        Delta m^2_31 = Delta m^2_ee + sin^2(theta12) Delta m^2_21

    (substituting ``Delta m^2_32 = Delta m^2_31 - Delta m^2_21`` into the
    ``Delta m^2_ee`` definition above and solving) -- so this wrapper is a
    pure change of variables, not a different fit; it exists so a fit's
    free-parameter vector and any resulting contour are directly comparable
    to Daya Bay's own published numbers/plots without a manual conversion
    step, and so LBFGS sees two parameters of comparable, well-conditioned
    magnitude (both real Daya Bay ``theta13`` and ``Delta m^2_3l`` already
    differ by ~4 orders of magnitude in radians/eV^2; ``Delta m^2_ee`` has
    the same magnitude as ``Delta m^2_3l`` so this wrapper does not change
    that particular conditioning, but ``sin^2(2 theta13)`` -- an amplitude
    in [0, 1] -- is often better conditioned than an angle in radians for a
    fit started far from its optimum).

    Parameters
    ----------
    model:
        A ``DayaBayDetectorModel`` or ``JointDayaBayModel`` whose ``free``
        starts with ``("theta13", "DeltamSq3l")`` (any further entries,
        e.g. ``global_normalization``/background nuisances, are passed
        through unchanged).
    theta12, dm21:
        The same fixed values used to build ``model``'s own
        ``VacuumOscillationModel`` (radians, eV^2).
    """

    model: "DayaBayDetectorModel | JointDayaBayModel"
    theta12: torch.Tensor
    dm21: torch.Tensor

    def __post_init__(self) -> None:
        if self.model.free[:2] != ("theta13", "DeltamSq3l"):
            raise ValueError(
                "DayaBayExperimentalModel requires model.free to start with "
                f"('theta13', 'DeltamSq3l'), got {self.model.free!r}."
            )

    @property
    def free(self) -> tuple[str, ...]:
        """Free parameter names: (SinSq2Theta13, DeltamSqEE) + model.free[2:]."""
        return EXPERIMENTAL_PARAM_KEYS + self.model.free[2:]

    def predict(self, theta: torch.Tensor) -> torch.Tensor:
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
class NearFarRatioDayaBayModel:
    """Predicts the real far-hall/near-hall per-analysis-bin count ratio.

    Daya Bay's original 2012 discovery of theta13 (F. P. An et al., Phys.
    Rev. Lett. 108, 171803 (2012), this package's notebook reference [3])
    was a near/far *comparison*: EH1 (AD11/AD12) and EH2 (AD21/AD22) sit at
    real flux-weighted baselines short enough (~560/600 m) that they are
    only weakly oscillated, while EH3 (AD31-AD34, ~1640 m) sees the full
    real oscillation deficit -- so a far/near ratio isolates the
    oscillation signal while cancelling, by construction, every systematic
    common to all 8 real detectors:

        R_k(theta) = N_k^far(theta) / N_k^near(theta),
        N_k^{near,far}(theta) = sum_{d in NEAR_DETECTORS,FAR_DETECTORS} N_{k,d}(theta),

    with ``N_{k,d}`` each real detector's own full per-bin prediction
    (``DayaBayDetectorModel.predict``, same real 6-reactor flux, real
    order-1/M cross section, real IAV+LSNL+Gaussian response). Because
    every detector shares the identical response/cross-section/efficiency
    functions in this project's model, a shared multiplicative factor --
    in particular ``global_normalization`` -- cancels *exactly* in the
    ratio regardless of its value, and any *residual* mismatch between this
    project's simplified response and the true one (the LSNL/IAV pull-curve
    freedom Sections 5.2-5.3 showed is not otherwise absorbable) cancels
    *approximately*, since it multiplies both the near and far prediction
    the same way. This is a real, if simplified, version of Daya Bay's own
    original near/far strategy -- not a full multi-baseline global fit
    (which would additionally weight each near detector by its own
    baseline-dependent oscillation deficit), but a substantial
    simplification relative to a bare absolute-count fit.

    Parameters
    ----------
    near_models, far_models:
        ``DayaBayDetectorModel`` instances for
        ``detector.dayabay.parameters.NEAR_DETECTORS``/``FAR_DETECTORS``
        respectively, all sharing the same ``oscillation_model`` (hence the
        same ``free``).
    """

    near_models: tuple[DayaBayDetectorModel, ...]
    far_models: tuple[DayaBayDetectorModel, ...]

    @property
    def free(self) -> tuple[str, ...]:
        """Free parameter names, forwarded from the first near-hall model."""
        return self.near_models[0].free

    def predict(self, theta: torch.Tensor) -> torch.Tensor:
        """Predict the real far/near per-bin count ratio from ``theta``.

        Args:
            theta: 1-D free-parameter tensor, shared by every wrapped model.

        Returns:
            Real tensor shaped ``(n_bins,)``, ``N_far(theta) / N_near(theta)``.
        """
        near_total = sum(model.predict(theta) for model in self.near_models)
        far_total = sum(model.predict(theta) for model in self.far_models)
        return far_total / near_total

    @staticmethod
    def real_observed_ratio(
        *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Real observed far/near per-bin count ratio and its propagated 1-sigma uncertainty.

        Args:
            device, dtype: Target tensor device/dtype.

        Returns:
            ``(ratio, sigma)``, each shape ``(n_bins,)``. ``sigma`` is the
            standard independent-Poisson-numerator/denominator propagation,
            ``ratio * sqrt(1/N_far + 1/N_near)``.
        """
        near_total = sum(
            real_observed_counts(det, device=device, dtype=dtype) for det in NEAR_DETECTORS
        )
        far_total = sum(
            real_observed_counts(det, device=device, dtype=dtype) for det in FAR_DETECTORS
        )
        ratio = far_total / near_total
        sigma = ratio * torch.sqrt(1.0 / far_total + 1.0 / near_total)
        return ratio, sigma

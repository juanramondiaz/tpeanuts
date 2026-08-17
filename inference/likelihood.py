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
-2 ln L statistics for comparing predictions to observed data points.

Both functions here return a scalar ``-2 ln L`` (up to a likelihood-family-
dependent constant), differentiable w.r.t. ``prediction``, so
``tpeanuts.inference.fit.fit_lbfgs`` can minimize either one by gradient
descent and reuse the same Laplace/Fisher covariance construction (Fisher
information ``= 0.5 * Hessian(-2 ln L)`` regardless of the likelihood family).

    chi2_asymmetric
        Gaussian likelihood with a one-sided (asymmetric) uncertainty.
        Appropriate for published *measurements with quoted +/- errors*, e.g.
        Borexino's Nature 2018 P_ee(E) points
        (``data/detector/borexino/probability/nature2018_pee_points.csv``).
    poisson_nll
        Poisson likelihood ratio (Baker & Cousins 1984) for *event counts*:
        ``prediction``/``value`` are expected/observed counts per bin, not
        probabilities. This is the statistic most neutrino event-rate
        analyses actually minimize; nothing in this package builds event
        counts yet (see ``tpeanuts.inference.model``'s P_ee-only forward
        models), so it is provided ready for that extension.
    correlated_gaussian_nll
        Gaussian likelihood with a full covariance matrix between
        observables (not just per-bin variances) -- needed whenever a
        published result reports statistical/systematic *correlations*
        between its own observables, e.g.
        ``tpeanuts.detector.sno_ii``'s 38-observable SNO salt-phase vector
        (Eq. 19-21 of its primary source). Not registered in
        ``LIKELIHOODS`` (unlike the two above): it needs a mandatory
        Cholesky-factor argument with no sensible default, so a bare
        string lookup would immediately fail; bind the factor first via
        ``functools.partial(correlated_gaussian_nll, cholesky_L=L)`` and
        pass the *result* as ``tpeanuts.inference.fit.fit_lbfgs``'s
        ``likelihood`` argument, which accepts such a callable directly.

Every likelihood function above shares the call signature ``(prediction,
value, sigma_minus=None, sigma_plus=None)`` -- ``poisson_nll`` and
``correlated_gaussian_nll`` accept but reject/ignore a non-None
``sigma_minus``/``sigma_plus`` (Poisson variance is fixed by the mean;
the correlated-Gaussian case gets its uncertainty from the bound
Cholesky factor instead) -- purely so ``tpeanuts.inference.fit.fit_lbfgs``
can dispatch on ``likelihood`` uniformly, whether it is a registry string
or a directly-supplied callable, without a family-specific call signature.

Module contents:
    LIKELIHOODS
        Registry mapping a likelihood name to its function, consumed by
        ``fit_lbfgs``'s ``likelihood`` argument.
    chi2_asymmetric(...)
        Sum of squared residuals normalized by the one-sided uncertainty.
    poisson_nll(...)
        Baker-Cousins Poisson likelihood ratio statistic.
    correlated_gaussian_nll(...)
        Gaussian likelihood with a full covariance matrix, Cholesky-solved.
    cholesky_from_covariance(...)
        ``torch.linalg.cholesky`` wrapper, meant to be called *once* outside
        any fit/scan loop and its result reused across every evaluation.
    gaussian_prior_penalty(...)
        ``-2 ln L`` contribution from independent Gaussian priors on a
        subset of the free-parameter vector itself (not the prediction) --
        the standard way to add real, externally measured nuisance-
        parameter constraints (e.g. a background-rate uncertainty) on top
        of either likelihood above; see ``tpeanuts.inference.fit.fit_lbfgs``'s
        ``penalty_fn`` argument.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch


def chi2_asymmetric(
    prediction: torch.Tensor,
    value: torch.Tensor,
    sigma_minus: Optional[torch.Tensor] = None,
    sigma_plus: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sum of squared residuals normalized by a one-sided uncertainty.

    For each entry, ``sigma_plus`` is used where the prediction overshoots
    the observed value (``prediction >= value``) and ``sigma_minus``
    otherwise:

        chi2 = sum_i ((prediction_i - value_i) / sigma_i)^2,
        sigma_i = sigma_plus_i  if prediction_i >= value_i else sigma_minus_i.

    Args:
        prediction: Model-predicted values, any shape.
        value: Observed central values, broadcastable to ``prediction``.
        sigma_minus: One-sided lower uncertainty (positive), broadcastable
            to ``prediction``. Required (kept optional only to share this
            function's signature with ``poisson_nll``).
        sigma_plus: One-sided upper uncertainty (positive), broadcastable
            to ``prediction``. Required, see ``sigma_minus``.

    Returns:
        Scalar tensor, differentiable w.r.t. ``prediction``.

    Raises:
        ValueError: If ``sigma_minus`` or ``sigma_plus`` is None.
    """
    if sigma_minus is None or sigma_plus is None:
        raise ValueError(
            "chi2_asymmetric requires both sigma_minus and sigma_plus "
            "(the Gaussian one-sided uncertainties); got "
            f"sigma_minus={sigma_minus!r}, sigma_plus={sigma_plus!r}."
        )
    residual = prediction - value
    sigma = torch.where(residual >= 0, sigma_plus, sigma_minus)
    return torch.sum((residual / sigma) ** 2)


def poisson_nll(
    prediction: torch.Tensor,
    value: torch.Tensor,
    sigma_minus: Optional[torch.Tensor] = None,
    sigma_plus: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Baker-Cousins Poisson likelihood ratio statistic, ``-2 ln L``.

    The statistic most neutrino event-rate analyses minimize (Baker &
    Cousins, Nucl. Instrum. Meth. 221 (1984) 437):

        -2 ln L = 2 sum_i [ N_i^pred - N_i^obs + N_i^obs ln(N_i^obs / N_i^pred) ],

    with ``N^pred = prediction`` (expected counts per bin) and
    ``N^obs = value`` (observed counts per bin). Unlike a plain Poisson
    ``-2 ln L`` (``2 sum_i [N_i^pred - N_i^obs ln N_i^pred]``), the extra
    ``-N_i^obs + N_i^obs ln N_i^obs`` terms subtract off the saturated
    model's likelihood, so this statistic is exactly 0 when
    ``prediction == value`` bin by bin and behaves like an ordinary
    chi-square asymptotically (Wilks' theorem), the same property
    ``chi2_asymmetric`` already has by construction.

    The ``N^obs ln(N^obs/N^pred)`` term is built as
    ``xlogy(value, value) - xlogy(value, prediction)`` rather than the
    algebraically equal ``xlogy(value, value / prediction)``: both give the
    same *forward* value (``xlogy`` special-cases its first argument being
    exactly 0 to return 0 regardless of the second, matching the analytic
    limit ``lim_{x->0+} x ln x = 0``, so an empty bin, ``value == 0``, never
    produces a ``nan`` loss either way) -- but the single-``xlogy`` form
    additionally has a broken *gradient* at an empty bin: with
    ``y = value / prediction``, ``xlogy``'s backward w.r.t. ``y`` is
    ``value / y``, which becomes the literal ``0 / 0 -> nan`` whenever
    ``value == 0`` (``y`` is then 0 too), even though the true derivative of
    that vanishing term is 0. Splitting the term keeps ``prediction``
    (always strictly positive, validated below) as ``xlogy``'s sole ``y``
    argument, whose backward ``value / prediction`` is then a clean
    ``0 / prediction = 0`` for an empty bin; the other term,
    ``xlogy(value, value)``, does not depend on ``prediction`` at all, so it
    contributes no gradient by construction.

    Args:
        prediction: Predicted (expected) counts per bin, any shape, must be
            strictly positive.
        value: Observed counts per bin, broadcastable to ``prediction``,
            must be non-negative.
        sigma_minus: Must be None -- accepted only so this function shares
            ``chi2_asymmetric``'s call signature (see module docstring).
        sigma_plus: Must be None, see ``sigma_minus``.

    Returns:
        Scalar tensor, differentiable w.r.t. ``prediction``.

    Raises:
        ValueError: If ``sigma_minus``/``sigma_plus`` is not None, if any
            ``prediction`` entry is not strictly positive, or if any
            ``value`` entry is negative.
    """
    if sigma_minus is not None or sigma_plus is not None:
        raise ValueError(
            "poisson_nll takes no uncertainty: Poisson variance is fixed by "
            "the predicted mean itself, not a separate sigma. Pass "
            "sigma_minus=None, sigma_plus=None (the default) -- passing "
            "either here would otherwise be silently ignored."
        )
    if torch.any(prediction <= 0):
        raise ValueError(
            "poisson_nll requires strictly positive predicted counts "
            "(N_pred > 0): the log(N_obs/N_pred) term is undefined "
            "otherwise."
        )
    if torch.any(value < 0):
        raise ValueError("poisson_nll requires non-negative observed counts (N_obs >= 0).")

    log_ratio_term = torch.xlogy(value, value) - torch.xlogy(value, prediction)
    return 2.0 * torch.sum(prediction - value + log_ratio_term)


def cholesky_from_covariance(covariance: torch.Tensor) -> torch.Tensor:
    """Lower-triangular Cholesky factor ``L`` of ``covariance``, ``covariance = L @ L.T``.

    Meant to be called once, outside any fit/scan loop, whenever
    ``covariance`` does not itself depend on the fit parameters (e.g. a
    published statistical+systematic covariance matrix): the resulting
    ``L`` is then reused across every ``correlated_gaussian_nll`` call
    instead of re-factorizing the same fixed matrix at every LBFGS
    iteration or grid point.

    Args:
        covariance: Symmetric positive-definite matrix, shape ``(n, n)``.

    Returns:
        Lower-triangular tensor, shape ``(n, n)``.

    Raises:
        torch.linalg.LinAlgError: If ``covariance`` is not positive-definite.
    """
    return torch.linalg.cholesky(covariance)


def correlated_gaussian_nll(
    prediction: torch.Tensor,
    value: torch.Tensor,
    sigma_minus: Optional[torch.Tensor] = None,
    sigma_plus: Optional[torch.Tensor] = None,
    *,
    cholesky_L: torch.Tensor,
) -> torch.Tensor:
    """Gaussian ``-2 ln L`` with a full covariance matrix, solved via its Cholesky factor.

    ``chi2 = (prediction - value)^T @ covariance^-1 @ (prediction - value)``,
    computed as ``||L^-1 @ (prediction - value)||^2`` (``covariance = L @
    L.T``) rather than by explicitly inverting ``covariance`` -- the
    numerically stable, standard construction (see e.g.
    ``tpeanuts.detector.sno_ii``'s primary source, Eq. 19, whose own
    ``sigma_ij^2(tot) = sigma_ij^2(stat) + sigma_ij^2(syst)`` is exactly the
    ``covariance`` this function expects, pre-factorized via
    ``cholesky_from_covariance``).

    With a diagonal ``covariance`` (``cholesky_L`` itself then diagonal,
    entries ``sqrt(covariance[i,i])``), this reduces exactly to
    ``chi2_asymmetric`` with ``sigma_minus == sigma_plus ==
    sqrt(diag(covariance))`` -- the two-sided special case of the one-sided
    statistic.

    Args:
        prediction: Model-predicted values, shape ``(n,)``.
        value: Observed central values, shape ``(n,)``.
        sigma_minus: Must be None -- accepted only so this function shares
            ``chi2_asymmetric``/``poisson_nll``'s call signature (see
            module docstring); the uncertainty comes from ``cholesky_L``
            instead.
        sigma_plus: Must be None, see ``sigma_minus``.
        cholesky_L: Lower-triangular Cholesky factor of the observables'
            full covariance matrix, shape ``(n, n)`` -- see
            ``cholesky_from_covariance``. Keyword-only, with no default:
            bind it via ``functools.partial`` before passing this function
            as ``tpeanuts.inference.fit.fit_lbfgs``'s ``likelihood``
            argument.

    Returns:
        Scalar tensor, differentiable w.r.t. ``prediction``.

    Raises:
        ValueError: If ``sigma_minus``/``sigma_plus`` is not None.
    """
    if sigma_minus is not None or sigma_plus is not None:
        raise ValueError(
            "correlated_gaussian_nll takes its uncertainty from cholesky_L, "
            "not sigma_minus/sigma_plus. Pass sigma_minus=None, "
            "sigma_plus=None (the default) -- passing either here would "
            "otherwise be silently ignored."
        )
    residual = (prediction - value).unsqueeze(-1)
    z = torch.linalg.solve_triangular(cholesky_L, residual, upper=False)
    return torch.sum(z.squeeze(-1) ** 2)


def gaussian_prior_penalty(
    values: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_sigma: torch.Tensor,
) -> torch.Tensor:
    """``-2 ln L`` penalty from independent Gaussian priors, ``sum(((values-mean)/sigma)^2)``.

    Unlike ``chi2_asymmetric``/``poisson_nll``, which compare a *prediction*
    to observed *data*, this constrains free *parameters* directly (e.g. a
    background-rate nuisance whose real published uncertainty is known
    externally, independent of what the fit's own data bins say about it).
    Pass the result (or a sum of several such calls) as
    ``tpeanuts.inference.fit.fit_lbfgs``'s ``penalty_fn(theta)``, sliced to
    the nuisance sub-vector of ``theta``.

    Args:
        values: Current parameter values, any shape.
        prior_mean: Prior central value(s), broadcastable to ``values``.
        prior_sigma: Prior standard deviation(s), broadcastable to
            ``values``, strictly positive.

    Returns:
        Scalar tensor, differentiable w.r.t. ``values``.
    """
    return torch.sum(((values - prior_mean) / prior_sigma) ** 2)


LIKELIHOODS: dict[str, Callable[..., torch.Tensor]] = {
    "chi2_asymmetric": chi2_asymmetric,
    "poisson": poisson_nll,
}

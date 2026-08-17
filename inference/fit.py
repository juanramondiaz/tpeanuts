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
Gradient-based point estimate and Laplace/Fisher uncertainty for any DifferentiableModel.

``fit_lbfgs`` minimizes a ``-2 ln L`` statistic -- selected by the
``likelihood`` string, looked up in ``tpeanuts.inference.likelihood
.LIKELIHOODS`` -- over a model's free parameters with ``torch.optim.LBFGS``,
using autograd for the gradient (no finite differences). ``FitResult
.covariance`` is the Laplace approximation built from the autograd Hessian
of that statistic at the optimum: for any of these likelihood families,
``-2 ln L = chi2 + const`` in the usual sense (exactly for
``chi2_asymmetric``, asymptotically for ``poisson_nll`` via Wilks' theorem),
so the Fisher information is ``0.5 * Hessian(-2 ln L)`` and the covariance is
its inverse, ``2 * Hessian(-2 ln L)^-1`` -- the same construction regardless
of which likelihood was selected.

Medium/detector-agnostic: every model-specific input (solar profile,
detector bin edges, reactor baselines, ...) is bound as a field on the
concrete model itself (see ``tpeanuts.inference.model.DifferentiableModel``);
this module only ever calls ``model.predict(theta)``.

Module contents:
    FitResult
        Best-fit parameter vector, loss convergence history, and covariance.
    fit_lbfgs(...)
        Run the LBFGS point estimate and the Hessian-based uncertainty.
    minimize_lbfgs(...)
        The point-estimate-only half of ``fit_lbfgs`` (no Hessian/
        covariance), factored out so
        ``tpeanuts.inference.scan.loglik_grid``'s ``profile_others=True``
        mode can re-minimize nuisance parameters at every grid point without
        paying for a Hessian it would immediately discard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from tpeanuts.inference.likelihood import LIKELIHOODS
from tpeanuts.inference.model import DifferentiableModel


def _resolve_likelihood(likelihood: "str | Callable[..., torch.Tensor]") -> Callable[..., torch.Tensor]:
    """Look up a ``LIKELIHOODS`` name, or pass a directly-supplied callable through.

    A callable must share ``chi2_asymmetric``/``poisson_nll``'s call shape
    ``(prediction, value, sigma_minus, sigma_plus) -> loss`` -- for a
    likelihood that instead needs a full covariance (e.g.
    ``tpeanuts.inference.likelihood.correlated_gaussian_nll``), bind the
    covariance-specific argument first via ``functools.partial`` and pass
    the result here; ``sigma_minus``/``sigma_plus`` are then simply
    threaded through unused (see that function's own signature).

    Raises:
        ValueError: If ``likelihood`` is a string not in ``LIKELIHOODS``.
    """
    if isinstance(likelihood, str):
        if likelihood not in LIKELIHOODS:
            raise ValueError(
                f"likelihood must be one of {sorted(LIKELIHOODS)} or a callable, got {likelihood!r}."
            )
        return LIKELIHOODS[likelihood]
    return likelihood


@dataclass(frozen=True)
class FitResult:
    """Outcome of a gradient-based fit.

    Parameters
    ----------
    theta_hat:
        Best-fit free-parameter vector (detached), same order as
        ``param_names``.
    param_names:
        Names of the fitted parameters, i.e. the model's ``free`` tuple.
    chi2_history:
        ``-2 ln L`` value (named for the ``chi2_asymmetric`` case; still the
        quantity actually minimized for ``likelihood="poisson"``) recorded
        at every LBFGS closure evaluation (includes internal line-search
        evaluations, not just accepted steps), for convergence diagnostics.
    covariance:
        Laplace/Fisher covariance matrix of ``theta_hat``, shaped
        ``(len(param_names), len(param_names))``.
    """

    theta_hat: torch.Tensor
    param_names: tuple[str, ...]
    chi2_history: list[float]
    covariance: torch.Tensor

    @property
    def sigma(self) -> torch.Tensor:
        """1-D tensor of marginal 1-sigma uncertainties, ``sqrt(diag(covariance))``."""
        return torch.sqrt(torch.diag(self.covariance))


def minimize_lbfgs(
    model: DifferentiableModel,
    theta0: torch.Tensor,
    value: torch.Tensor,
    sigma_minus: Optional[torch.Tensor] = None,
    sigma_plus: Optional[torch.Tensor] = None,
    *,
    likelihood: "str | Callable[..., torch.Tensor]" = "chi2_asymmetric",
    penalty_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    max_iter: int = 200,
    tolerance_grad: float = 1.0e-10,
    tolerance_change: float = 1.0e-12,
) -> tuple[torch.Tensor, list[float]]:
    """Point-estimate-only LBFGS minimization (no Hessian/covariance).

    Args, Raises: see ``fit_lbfgs`` -- identical, minus the arguments
    ``fit_lbfgs`` alone needs for the Hessian step.

    Returns:
        ``(theta_hat, loss_history)``: the optimized free-parameter vector
        (detached, physical units) and the ``-2 ln L`` value at every LBFGS
        closure evaluation.
    """
    loss_fn = _resolve_likelihood(likelihood)

    def total_loss(prediction: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        loss = loss_fn(prediction, value, sigma_minus, sigma_plus)
        if penalty_fn is not None:
            loss = loss + penalty_fn(theta)
        return loss

    # LBFGS's identity-initialized inverse-Hessian approximation assumes
    # roughly comparable parameter scales; theta12 (~O(1) rad) and
    # DeltamSq21/DeltamSq3l (~O(1e-4)-O(1e-3) eV^2) differ by 3-4 orders of
    # magnitude, which starves the line search of a usable step size. Fit a
    # dimensionless u = theta / scale instead, and undo the change of
    # variables only in the returned theta_hat -- model.predict always
    # sees physical theta. scale is set from the loss gradient at theta0
    # (not theta0's own magnitude): a natural parameter starting value of
    # exactly 0 -- eps_ee0=0.0, the obvious SM-limit starting guess for
    # SolarNSIOscillationModel -- would otherwise floor scale at machine
    # tiny and freeze that direction, since |theta0| is 0 regardless of how
    # sensitive the loss actually is to it there.
    theta0_grad_point = theta0.detach().clone().requires_grad_(True)
    loss0 = total_loss(model.predict(theta0_grad_point), theta0_grad_point)
    grad0 = torch.autograd.grad(loss0, theta0_grad_point)[0]
    scale = grad0.abs().clamp_min(torch.finfo(theta0.dtype).tiny).reciprocal()
    # u0 = theta0 / scale, not torch.ones_like(theta0): scale is an unsigned
    # magnitude, so seeding u at exactly 1 would silently flip the sign of
    # any negative starting value (e.g. eps_ee0=-2.0, the LMA-Dark starting
    # guess) to its positive counterpart before the very first LBFGS step.
    u = (theta0.detach() / scale).clone().requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [u],
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        line_search_fn="strong_wolfe",
    )

    history: list[float] = []

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        theta = u * scale
        loss = total_loss(model.predict(theta), theta)
        loss.backward()
        history.append(float(loss.detach()))
        return loss

    optimizer.step(closure)
    theta = (u * scale).detach()

    return theta, history


def fit_lbfgs(
    model: DifferentiableModel,
    theta0: torch.Tensor,
    value: torch.Tensor,
    sigma_minus: Optional[torch.Tensor] = None,
    sigma_plus: Optional[torch.Tensor] = None,
    *,
    likelihood: "str | Callable[..., torch.Tensor]" = "chi2_asymmetric",
    penalty_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    max_iter: int = 200,
    tolerance_grad: float = 1.0e-10,
    tolerance_change: float = 1.0e-12,
) -> FitResult:
    """Fit ``model``'s free parameters to observed data points via LBFGS.

    Args:
        model: Differentiable model (see
            ``tpeanuts.inference.model.DifferentiableModel``); every
            model-specific input (solar profile, detector bin edges,
            reactor baselines, ...) is already bound as a field on
            ``model`` itself.
        theta0: Starting free-parameter vector (any ``requires_grad``
            state; a detached copy is optimized, ``theta0`` itself is not
            mutated).
        value: Observed central values, same length as ``model.predict``'s
            output -- P_ee for ``likelihood="chi2_asymmetric"``, event
            counts for ``likelihood="poisson"``.
        sigma_minus: One-sided lower uncertainties, same length as
            ``value``. Required for ``likelihood="chi2_asymmetric"``, must
            be None for ``likelihood="poisson"`` (see
            ``tpeanuts.inference.likelihood.poisson_nll``) or a callable
            that does not use them (e.g.
            ``tpeanuts.inference.likelihood.correlated_gaussian_nll``).
        sigma_plus: One-sided upper uncertainties, same length as
            ``value``. Same requirement as ``sigma_minus``.
        likelihood: Either a name in
            ``tpeanuts.inference.likelihood.LIKELIHOODS`` (``"chi2_asymmetric"``,
            the default -- Gaussian with one-sided uncertainty; or
            ``"poisson"`` -- Baker-Cousins likelihood ratio for event
            counts), or a directly-supplied callable sharing that same
            ``(prediction, value, sigma_minus, sigma_plus) -> loss`` shape
            -- see ``tpeanuts.inference.likelihood.correlated_gaussian_nll``
            for a correlated-covariance example, bound via
            ``functools.partial`` before being passed here.
        penalty_fn: Optional additional ``-2 ln L`` term as a function of
            the full ``theta`` vector (physical units, same order as
            ``model.free``), added to the selected likelihood before
            minimizing -- e.g.
            ``tpeanuts.inference.likelihood.gaussian_prior_penalty`` applied
            to a nuisance-parameter sub-vector, for a real externally
            measured constraint independent of ``value``. None (default)
            reproduces the previous, unconstrained-fit behaviour exactly.
        max_iter: Maximum LBFGS iterations (forwarded to
            ``torch.optim.LBFGS``).
        tolerance_grad: LBFGS gradient-norm convergence tolerance.
        tolerance_change: LBFGS parameter/loss-change convergence tolerance.

    Returns:
        FitResult with the optimized parameters, convergence history, and
        Laplace covariance (of the *penalized* ``-2 ln L``, so a tight prior
        correctly shrinks its own parameter's reported uncertainty).

    Raises:
        ValueError: If ``likelihood`` is a string not in
            ``tpeanuts.inference.likelihood.LIKELIHOODS``, or (raised by the
            selected likelihood function itself) if ``sigma_minus``/
            ``sigma_plus`` are incompatible with it.
    """
    theta, history = minimize_lbfgs(
        model, theta0, value, sigma_minus, sigma_plus,
        likelihood=likelihood, penalty_fn=penalty_fn, max_iter=max_iter,
        tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
    )
    loss_fn = _resolve_likelihood(likelihood)

    def loss_of_theta(t: torch.Tensor) -> torch.Tensor:
        prediction = model.predict(t)
        loss = loss_fn(prediction, value, sigma_minus, sigma_plus)
        if penalty_fn is not None:
            loss = loss + penalty_fn(t)
        return loss

    # A direction the data barely constrains (e.g. DeltamSq21 from a handful
    # of fixed-energy solar points) gives Hessian eigenvalues near zero, and
    # occasionally slightly negative from numerical noise around the LBFGS
    # optimum -- plain torch.linalg.inv then returns nan/inf or a negative
    # "variance". Symmetrize away asymmetric round-off and floor small/
    # negative eigenvalues at a tiny fraction of the largest one, so an
    # unconstrained direction gets a large but finite sigma instead of nan.
    hessian = torch.autograd.functional.hessian(loss_of_theta, theta.detach())
    hessian = 0.5 * (hessian + hessian.transpose(-2, -1))
    eigvals, eigvecs = torch.linalg.eigh(hessian)
    floor = eigvals.abs().amax().clamp_min(torch.finfo(hessian.dtype).tiny) * 1.0e-8
    eigvals = torch.clamp(eigvals, min=floor)
    covariance = 2.0 * (eigvecs * (1.0 / eigvals)) @ eigvecs.transpose(-2, -1)

    return FitResult(
        theta_hat=theta.detach().clone(),
        param_names=model.free,
        chi2_history=history,
        covariance=covariance,
    )

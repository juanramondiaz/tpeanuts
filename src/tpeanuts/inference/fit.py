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

"""Gradient-based point estimate and Laplace/Fisher uncertainty for any DifferentiableModel.

``fit`` minimizes a ``-2 ln L`` statistic -- selected by the ``likelihood``
string, looked up in ``tpeanuts.inference.likelihood.LIKELIHOODS`` -- over
a model's free parameters with a caller-selected optimizer, configured by
an ``OptimizerConfig`` (``optimizer_config``, ``"lbfgs"`` by default,
``"adam"`` also validated, or any ``OptimizerConfig``-conforming dataclass
instance for a custom optimizer), using autograd for the gradient.
``FitResult.covariance`` is the Laplace approximation built from the
autograd Hessian of that statistic at the optimum, the same construction
(covariance ``= 2 * Hessian(-2 ln L)^-1``) regardless of which likelihood
or optimizer was selected.

``OptimizerConfig`` is deliberately one small dataclass *per optimizer*
(``LBFGSConfig``, ``AdamConfig``), not one generic dataclass with the union
of every optimizer's fields: LBFGS (``lr``, ``tolerance_grad``,
``tolerance_change``, ``line_search_fn``) and Adam (``lr``, ``betas``,
``eps``, ``weight_decay``) share almost no real fields, so a single
generic dataclass would degenerate into the same untyped kwargs bag this
replaces, just with more ceremony -- the same reasoning that already
keeps ``tpeanuts.physics.common.mixing.StandardMixingParameters``/
``SterileMixingParameters`` separate rather than one dataclass with a pile of
``Optional`` fields. No forced base class either: each concrete config
just declares its own ``optimizer_cls: ClassVar[Type[torch.optim.Optimizer]]``,
matching this project's existing preference for structural ``Protocol``s
(``DifferentiableModel``, ``tpeanuts.physics.common.terms.GeneratorTerm``)
over inheritance hierarchies.

Two genuinely different control-flow paths exist under the same ``fit``/
``minimize`` surface, because ``torch.optim.LBFGS`` and every other
``torch.optim.Optimizer`` (Adam, SGD, ...) are used structurally
differently in PyTorch: LBFGS takes a ``closure`` and internally iterates
up to ``max_iter`` times with its own line search inside a single
``optimizer.step(closure)`` call; every other optimizer has no closure or
line search at all and is driven by an explicit
``for _ in range(max_iter): zero_grad(); backward(); step()`` loop here,
with no early stopping (unlike LBFGS's own ``tolerance_grad``/
``tolerance_change``-based stop) -- a deliberate simplification, not an
attempt to reproduce LBFGS's convergence criterion for an optimizer that
has no comparable per-step line-search tolerance concept. ``max_iter``
itself stays a top-level ``minimize``/``fit`` argument rather than an
``OptimizerConfig`` field, since both paths need the same "how much work
to do" concept regardless of which optimizer was chosen.

Medium/detector-agnostic: every model-specific input (source, medium,
detector, energy grid, ...) is bound as a field on the concrete model
itself; this module only ever calls ``model.predict(theta)``.

``tpeanuts.inference.scan`` builds on ``minimize`` (not ``fit`` --
a grid scan never needs the single-point Laplace covariance ``fit``
also computes) for a profile-likelihood grid scan and a
parametric-bootstrap threshold calibration, applying to any
``DifferentiableModel`` unchanged.

Module contents:
    OptimizerConfig
        Structural type every optimizer-settings dataclass must satisfy
        (a single ``optimizer_cls`` ClassVar).
    LBFGSConfig, AdamConfig
        Typed, defaulted construction hyperparameters for
        ``torch.optim.LBFGS``/``torch.optim.Adam``.
    FitResult
        Best-fit parameter vector, loss convergence history, covariance,
        and which optimizer produced it.
    fit(...)
        Run the point estimate and the Hessian-based uncertainty.
    minimize(...)
        The point-estimate-only half of ``fit`` (no Hessian/covariance).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable, ClassVar, Dict, List, Optional, Protocol, Tuple, Type, Union, cast, runtime_checkable

import torch

from tpeanuts.config import default
from tpeanuts.inference.likelihood import LIKELIHOODS
from tpeanuts.inference.model import DifferentiableModel


@runtime_checkable
class OptimizerConfig(Protocol):
    """Structural type every optimizer-settings dataclass must satisfy.

    No inheritance required -- any dataclass exposing this single
    ClassVar satisfies it. ``_resolve_optimizer_config`` reads
    ``optimizer_cls`` to know which ``torch.optim.Optimizer`` to build,
    and ``dataclasses.fields(config)`` to extract every other field as
    that optimizer's own constructor kwargs -- so writing a config for a
    new optimizer (e.g. SGD) needs no change to this module at all, only
    a new dataclass with matching field names.

    Attributes:
        optimizer_cls: The ``torch.optim.Optimizer`` subclass this
            config's fields are constructor kwargs for.
    """

    optimizer_cls: ClassVar[Type[torch.optim.Optimizer]]


@dataclass(frozen=True)
class LBFGSConfig:
    """``torch.optim.LBFGS``'s own construction hyperparameters, typed and defaulted.

    Field names match ``torch.optim.LBFGS``'s constructor kwargs exactly,
    so ``LBFGSConfig()``'s fields can be forwarded directly as ``**kwargs``.

    Attributes:
        lr: Learning rate (LBFGS's own step-size scale, not the same
            sense as a first-order optimizer's).
        tolerance_grad: Gradient-norm convergence tolerance.
        tolerance_change: Parameter/loss-change convergence tolerance.
        line_search_fn: ``"strong_wolfe"`` (the default here) or None
            (plain fixed-step LBFGS, not validated by any test in this
            project).
    """

    lr: float = default.LBFGS_LR
    tolerance_grad: float = default.LBFGS_TOLERANCE_GRAD
    tolerance_change: float = default.LBFGS_TOLERANCE_CHANGE
    line_search_fn: Optional[str] = default.LBFGS_LINE_SEARCH_FN
    optimizer_cls: ClassVar[Type[torch.optim.Optimizer]] = torch.optim.LBFGS


@dataclass(frozen=True)
class AdamConfig:
    """``torch.optim.Adam``'s own construction hyperparameters, typed and defaulted.

    Field names match ``torch.optim.Adam``'s constructor kwargs exactly,
    so ``AdamConfig()``'s fields can be forwarded directly as ``**kwargs``.

    Attributes:
        lr: Learning rate.
        betas: Coefficients for the running mean/variance estimates.
        eps: Term added to the denominator for numerical stability.
        weight_decay: L2 penalty coefficient.
    """

    lr: float = default.ADAM_LR
    betas: Tuple[float, float] = default.ADAM_BETAS
    eps: float = default.ADAM_EPS
    weight_decay: float = default.ADAM_WEIGHT_DECAY
    optimizer_cls: ClassVar[Type[torch.optim.Optimizer]] = torch.optim.Adam


_OPTIMIZER_CONFIGS: Dict[str, Type[OptimizerConfig]] = {
    "lbfgs": LBFGSConfig,
    "adam": AdamConfig,
}


def _resolve_likelihood(likelihood: Union[str, Callable[..., torch.Tensor]]) -> Callable[..., torch.Tensor]:
    """Look up a ``LIKELIHOODS`` name, or pass a directly-supplied callable through.

    A callable must share ``chi2_asymmetric``/``poisson_nll``'s call shape
    ``(prediction, value, sigma_minus, sigma_plus) -> loss``.

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


def _resolve_optimizer_config(optimizer_config: Union[str, OptimizerConfig]) -> OptimizerConfig:
    """Look up an ``_OPTIMIZER_CONFIGS`` name (built with its own defaults), or pass a config instance through.

    Mirrors ``_resolve_likelihood``'s "name or direct object" dispatch. A
    custom optimizer (anything outside ``_OPTIMIZER_CONFIGS``) needs no
    registration at all -- just construct its own ``OptimizerConfig``-
    conforming dataclass instance and pass it here directly.

    Raises:
        ValueError: If ``optimizer_config`` is a string not in
            ``_OPTIMIZER_CONFIGS``.
    """
    if isinstance(optimizer_config, str):
        if optimizer_config not in _OPTIMIZER_CONFIGS:
            raise ValueError(
                f"optimizer_config must be one of {sorted(_OPTIMIZER_CONFIGS)} or an "
                f"OptimizerConfig instance, got {optimizer_config!r}."
            )
        return _OPTIMIZER_CONFIGS[optimizer_config]()
    return optimizer_config


@dataclass(frozen=True)
class FitResult:
    """Outcome of a gradient-based fit.

    Attributes:
        theta_hat: Best-fit free-parameter vector (detached), same order as
            ``param_names``.
        param_names: Names of the fitted parameters, i.e. the model's
            ``free`` tuple.
        chi2_history: ``-2 ln L`` value (named for the ``chi2_asymmetric``
            case; still the quantity actually minimized for
            ``likelihood="poisson"``) recorded at every optimizer step
            (for LBFGS, includes internal line-search evaluations, not
            just accepted steps), for convergence diagnostics.
        covariance: Laplace/Fisher covariance matrix of ``theta_hat``,
            shaped ``(len(param_names), len(param_names))``.
        optimizer_name: Name of the optimizer class that produced
            ``theta_hat`` (e.g. ``"LBFGS"``, ``"Adam"``), for provenance.
    """

    theta_hat: torch.Tensor
    param_names: Tuple[str, ...]
    chi2_history: List[float]
    covariance: torch.Tensor
    optimizer_name: str

    @property
    def sigma(self) -> torch.Tensor:
        """1-D tensor of marginal 1-sigma uncertainties, ``sqrt(diag(covariance))``."""
        return torch.sqrt(torch.diag(self.covariance))


def minimize(
    model: DifferentiableModel,
    theta0: torch.Tensor,
    value: torch.Tensor,
    sigma_minus: Optional[torch.Tensor] = None,
    sigma_plus: Optional[torch.Tensor] = None,
    *,
    likelihood: Union[str, Callable[..., torch.Tensor]] = "chi2_asymmetric",
    penalty_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    optimizer_config: Union[str, OptimizerConfig] = "lbfgs",
    max_iter: int = default.FIT_MAX_ITER,
) -> Tuple[torch.Tensor, List[float]]:
    """Point-estimate-only minimization (no Hessian/covariance).

    Args, Raises: see ``fit`` -- identical, minus the arguments ``fit``
    alone needs for the Hessian step.

    Returns:
        ``(theta_hat, loss_history)``: the optimized free-parameter vector
        (detached, physical units) and the ``-2 ln L`` value at every
        optimizer step.
    """
    loss_fn = _resolve_likelihood(likelihood)
    config = _resolve_optimizer_config(optimizer_config)
    optimizer_cls = config.optimizer_cls
    optimizer_field_kwargs = {f.name: getattr(config, f.name) for f in fields(config)}  # type: ignore[arg-type]
    is_lbfgs = issubclass(optimizer_cls, torch.optim.LBFGS)

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
    # exactly 0 would otherwise floor scale at machine tiny and freeze that
    # direction, since |theta0| is 0 regardless of how sensitive the loss
    # actually is to it there. This rescaling is not LBFGS-specific -- it is
    # a generic change of variables applied identically for every
    # optimizer; a per-parameter-adaptive optimizer like Adam needs it less
    # (its own second-moment normalization already compensates differing
    # scales to some degree), but it does not hurt there either.
    theta0_grad_point = theta0.detach().clone().requires_grad_(True)
    loss0 = total_loss(model.predict(theta0_grad_point), theta0_grad_point)
    grad0 = torch.autograd.grad(loss0, theta0_grad_point)[0]
    scale = grad0.abs().clamp_min(torch.finfo(theta0.dtype).tiny).reciprocal()
    # A parameter that starts at its optimum has a vanishing gradient, so 1/|grad0| would give it an enormous scale
    # and the first optimizer step would throw it away (the two normalisations of a shape fit that are profiled
    # before the fit, with gradients of 1e-11, were sent to 3e10 and the propagation then failed). The scale is
    # therefore bounded by a multiple of the parameter's own magnitude, or of 1 if it starts at exactly 0.
    reference = torch.where(theta0.detach().abs() > 0, theta0.detach().abs(), torch.ones_like(theta0.detach()))
    scale = torch.minimum(scale, default.FIT_SCALE_MAX_RELATIVE * reference)
    # u0 = theta0 / scale, not torch.ones_like(theta0): scale is an unsigned
    # magnitude, so seeding u at exactly 1 would silently flip the sign of
    # any negative starting value to its positive counterpart before the
    # very first optimizer step.
    u = (theta0.detach() / scale).clone().requires_grad_(True)

    optimizer_instance: Any
    if is_lbfgs:
        optimizer_instance = cast(Any, optimizer_cls)([u], max_iter=max_iter, **optimizer_field_kwargs)
    else:
        optimizer_instance = optimizer_cls([u], **optimizer_field_kwargs)

    history: List[float] = []

    if is_lbfgs:
        def closure() -> torch.Tensor:
            optimizer_instance.zero_grad()
            theta = u * scale
            loss = total_loss(model.predict(theta), theta)
            loss.backward()
            history.append(float(loss.detach()))
            return loss

        optimizer_instance.step(closure)
    else:
        # No closure/line search for a plain first-order optimizer -- an
        # explicit Python loop of exactly max_iter steps, with no early
        # stopping (see this module's own docstring for why).
        for _ in range(max_iter):
            optimizer_instance.zero_grad()
            theta = u * scale
            loss = total_loss(model.predict(theta), theta)
            loss.backward()
            history.append(float(loss.detach()))
            optimizer_instance.step()

    theta = (u * scale).detach()

    return theta, history


def fit(
    model: DifferentiableModel,
    theta0: torch.Tensor,
    value: torch.Tensor,
    sigma_minus: Optional[torch.Tensor] = None,
    sigma_plus: Optional[torch.Tensor] = None,
    *,
    likelihood: Union[str, Callable[..., torch.Tensor]] = "chi2_asymmetric",
    penalty_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    optimizer_config: Union[str, OptimizerConfig] = "lbfgs",
    max_iter: int = default.FIT_MAX_ITER,
) -> FitResult:
    """Fit ``model``'s free parameters to observed data points.

    Args:
        model: Differentiable model (see
            ``tpeanuts.inference.model.DifferentiableModel``); every
            model-specific input (source, medium, detector, energy grid,
            ...) is already bound as a field on ``model`` itself.
        theta0: Starting free-parameter vector (any ``requires_grad``
            state; a detached copy is optimized, ``theta0`` itself is not
            mutated).
        value: Observed central values, same length as ``model.predict``'s
            output -- e.g. a probability curve for
            ``likelihood="chi2_asymmetric"``, event counts/rate for
            ``likelihood="poisson"``.
        sigma_minus: One-sided lower uncertainties, same length as
            ``value``. Required for ``likelihood="chi2_asymmetric"``, must
            be None for ``likelihood="poisson"`` (see
            ``tpeanuts.inference.likelihood.poisson_nll``).
        sigma_plus: One-sided upper uncertainties, same length as
            ``value``. Same requirement as ``sigma_minus``.
        likelihood: Either a name in
            ``tpeanuts.inference.likelihood.LIKELIHOODS`` (``"chi2_asymmetric"``,
            the default -- Gaussian with one-sided uncertainty; or
            ``"poisson"`` -- Baker-Cousins likelihood ratio for event
            counts), or a directly-supplied callable sharing that same
            ``(prediction, value, sigma_minus, sigma_plus) -> loss`` shape.
        penalty_fn: Optional additional ``-2 ln L`` term as a function of
            the full ``theta`` vector (physical units, same order as
            ``model.free``), added to the selected likelihood before
            minimizing. None (default) reproduces the unconstrained-fit
            behaviour exactly.
        optimizer_config: Either a name in ``_OPTIMIZER_CONFIGS``
            (``"lbfgs"``, the default; or ``"adam"``), built with its own
            validated defaults from ``tpeanuts.config.default``, or a
            directly-supplied ``OptimizerConfig``-conforming dataclass
            instance (``LBFGSConfig``/``AdamConfig`` with overridden
            fields, or a custom one for a different
            ``torch.optim.Optimizer`` entirely -- see this module's own
            docstring).
        max_iter: For LBFGS, the maximum internal line-search iterations
            (forwarded to ``torch.optim.LBFGS``). For any other
            optimizer, the number of explicit ``optimizer.step()`` calls
            run (no early stopping -- see this module's own docstring).

    Returns:
        FitResult with the optimized parameters, convergence history,
        Laplace covariance (of the *penalized* ``-2 ln L``, so a tight
        prior correctly shrinks its own parameter's reported uncertainty),
        and the resolved optimizer's class name.

    Raises:
        ValueError: If ``likelihood`` is a string not in
            ``tpeanuts.inference.likelihood.LIKELIHOODS``, if
            ``optimizer_config`` is a string not in
            ``_OPTIMIZER_CONFIGS``, or (raised by the selected likelihood
            function itself) if ``sigma_minus``/``sigma_plus`` are
            incompatible with it.
    """
    config = _resolve_optimizer_config(optimizer_config)
    theta, history = minimize(
        model, theta0, value, sigma_minus, sigma_plus,
        likelihood=likelihood, penalty_fn=penalty_fn,
        optimizer_config=config, max_iter=max_iter,
    )
    loss_fn = _resolve_likelihood(likelihood)

    def loss_of_theta(t: torch.Tensor) -> torch.Tensor:
        prediction = model.predict(t)
        loss = loss_fn(prediction, value, sigma_minus, sigma_plus)
        if penalty_fn is not None:
            loss = loss + penalty_fn(t)
        return loss

    # A direction the data barely constrains gives Hessian eigenvalues near
    # zero, and occasionally slightly negative from numerical noise around
    # the optimum -- plain torch.linalg.inv then returns nan/inf or a
    # negative "variance". Symmetrize away asymmetric round-off and floor
    # small/negative eigenvalues at a tiny fraction of the largest one, so
    # an unconstrained direction gets a large but finite sigma instead of
    # nan.
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
        optimizer_name=config.optimizer_cls.__name__,
    )

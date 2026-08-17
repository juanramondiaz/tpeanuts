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
2-D -2 ln L confidence maps: grid scans and Monte-Carlo threshold calibration.

``loglik_grid`` is a reusable version of the by-hand grid loop
``notebooks/inference/inference2_borexino_nsi.ipynb`` used for its chi2
landscape: fix (or, with ``profile_others=True``, re-minimize) every
parameter except two, evaluate the selected ``-2 ln L`` statistic on a
``(param_x, param_y)`` grid, and draw confidence contours at
``CHI2_LEVELS_2D``'s Delta(-2 ln L) thresholds (Wilks' theorem, 2 degrees of
freedom).

``calibrate_delta_threshold`` checks that Wilks' theorem actually holds at
a given exposure/statistics level instead of assuming it: it simulates many
toy datasets at a known truth, refits each, and returns the *empirical*
quantile of Delta(-2 ln L) between the truth and each toy's own best fit --
the parametric-bootstrap alternative to ``CHI2_LEVELS_2D``'s asymptotic
table values, useful whenever Poisson-limited (low-count) statistics put
the Laplace/Fisher sigma's asymptotic-normality assumption in doubt; the
same assumption underlies both the Laplace sigma and the chi-square table
used for grid contours.

Module contents:
    CHI2_LEVELS_2D
        Delta(-2 ln L) thresholds for standard confidence levels at 2 dof
        (Numerical Recipes Table 15.6.1 / PDG Statistics Review).
    loglik_grid(...)
        The reusable grid-scan function.
    calibrate_delta_threshold(...)
        Monte-Carlo (parametric-bootstrap) calibration of one CL's
        Delta(-2 ln L) threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from tpeanuts.inference.fit import _resolve_likelihood, minimize_lbfgs
from tpeanuts.inference.likelihood import LIKELIHOODS
from tpeanuts.inference.model import DifferentiableModel

# Delta(-2 ln L) thresholds for a joint 2-parameter confidence region
# (chi-square distribution, 2 degrees of freedom). Numerical Recipes,
# 3rd ed., Table 15.6.1; equivalently PDG Review of Particle Physics,
# Statistics chapter, Table "Delta chi^2 as a function of C.L. and dof".
CHI2_LEVELS_2D: dict[str, float] = {
    "68%": 2.30,
    "90%": 4.61,
    "95%": 6.18,
    "99%": 9.21,
}


@dataclass(frozen=True)
class _FixedSliceModel:
    """Adapter: (param_x, param_y) fixed, every other free parameter is ``free``.

    Lets ``profile_others=True`` reuse ``tpeanuts.inference.fit
    .minimize_lbfgs`` unchanged to re-minimize the nuisance parameters at
    one (x, y) grid point, instead of a bespoke sub-optimizer.
    """

    model: DifferentiableModel
    full_free: tuple[str, ...]
    fixed_positions: tuple[int, int]
    fixed_values: torch.Tensor
    nuisance_positions: tuple[int, ...]

    @property
    def free(self) -> tuple[str, ...]:
        return tuple(self.full_free[i] for i in self.nuisance_positions)

    def _assemble(self, nuisance_theta: torch.Tensor) -> torch.Tensor:
        pieces: list[Optional[torch.Tensor]] = [None] * len(self.full_free)
        pieces[self.fixed_positions[0]] = self.fixed_values[0]
        pieces[self.fixed_positions[1]] = self.fixed_values[1]
        for k, pos in enumerate(self.nuisance_positions):
            pieces[pos] = nuisance_theta[k]
        return torch.stack(pieces)  # type: ignore[arg-type]

    def predict(self, theta: torch.Tensor) -> torch.Tensor:
        return self.model.predict(self._assemble(theta))


def loglik_grid(
    model: DifferentiableModel,
    theta_baseline: torch.Tensor,
    param_x: str,
    x_grid: torch.Tensor,
    param_y: str,
    y_grid: torch.Tensor,
    value: torch.Tensor,
    sigma_minus: Optional[torch.Tensor] = None,
    sigma_plus: Optional[torch.Tensor] = None,
    *,
    likelihood: "str | Callable[..., torch.Tensor]" = "chi2_asymmetric",
    profile_others: bool = False,
    max_iter: int = 100,
) -> torch.Tensor:
    """Evaluate ``-2 ln L`` on a ``(param_x, param_y)`` grid.

    Every entry of ``model.free`` other than ``param_x``/``param_y`` is
    held fixed at ``theta_baseline`` (``profile_others=False``, cheap -- a
    plain slice through the likelihood surface) or re-minimized at each
    grid point (``profile_others=True``, the statistically correct
    construction when other parameters are also uncertain, at the cost of
    one LBFGS run per grid point). With exactly two free parameters in
    ``model.free`` (nothing to profile), both modes coincide and
    ``profile_others`` has no effect.

    Args:
        model: Differentiable model (see
            ``tpeanuts.inference.model.DifferentiableModel``); every
            model-specific input is already bound as a field on ``model``.
        theta_baseline: Values for every ``model.free`` entry not being
            scanned; must have the same length/order as ``model.free``.
            Only its ``param_x``/``param_y`` entries are ignored
            (overwritten by the grid).
        param_x: Name of the first scanned parameter, must be in
            ``model.free``.
        x_grid: Values to scan ``param_x`` over, shape ``(nx,)``.
        param_y: Name of the second scanned parameter, must be in
            ``model.free`` and different from ``param_x``.
        y_grid: Values to scan ``param_y`` over, shape ``(ny,)``.
        value: Observed central values.
        sigma_minus: One-sided lower uncertainties, see
            ``tpeanuts.inference.fit.fit_lbfgs``.
        sigma_plus: One-sided upper uncertainties, see
            ``tpeanuts.inference.fit.fit_lbfgs``.
        likelihood: A ``LIKELIHOODS`` name or a directly-supplied callable,
            see ``tpeanuts.inference.fit.fit_lbfgs``'s own ``likelihood``
            argument.
        profile_others: See above.
        max_iter: Maximum LBFGS iterations per grid point, only used when
            ``profile_others=True`` and ``model.free`` has more than two
            entries.

    Returns:
        Real tensor shaped ``(nx, ny)``, ``-2 ln L`` at each grid point.
        Subtract its minimum and compare against ``CHI2_LEVELS_2D`` to draw
        confidence contours.

    Raises:
        ValueError: If ``likelihood`` is unknown, or ``param_x``/
            ``param_y`` are not both (distinct) entries of ``model.free``.
    """
    loss_fn = _resolve_likelihood(likelihood)
    if param_x == param_y:
        raise ValueError(f"param_x and param_y must differ, both got {param_x!r}.")
    if param_x not in model.free or param_y not in model.free:
        raise ValueError(
            f"param_x/param_y must both be in model.free={model.free}; "
            f"got param_x={param_x!r}, param_y={param_y!r}."
        )

    ix = model.free.index(param_x)
    iy = model.free.index(param_y)
    other_idx = tuple(i for i in range(len(model.free)) if i not in (ix, iy))

    grid = torch.zeros(x_grid.shape[0], y_grid.shape[0], dtype=theta_baseline.dtype)

    with torch.no_grad():
        for i, x in enumerate(x_grid):
            for j, y in enumerate(y_grid):
                if profile_others and other_idx:
                    fixed_values = torch.stack([x, y])
                    sub_model = _FixedSliceModel(
                        model=model, full_free=model.free,
                        fixed_positions=(ix, iy), fixed_values=fixed_values,
                        nuisance_positions=other_idx,
                    )
                    nu0 = theta_baseline[list(other_idx)]
                    with torch.enable_grad():
                        nu_hat, _ = minimize_lbfgs(
                            sub_model, nu0, value, sigma_minus, sigma_plus,
                            likelihood=likelihood, max_iter=max_iter,
                        )
                    theta = sub_model._assemble(nu_hat)
                else:
                    theta = theta_baseline.clone()
                    theta[ix] = x
                    theta[iy] = y

                prediction = model.predict(theta)
                grid[i, j] = loss_fn(prediction, value, sigma_minus, sigma_plus)

    return grid


def _sample_toy_value(
    prediction: torch.Tensor,
    likelihood: str,
    sigma_minus: Optional[torch.Tensor],
    sigma_plus: Optional[torch.Tensor],
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Draw one toy ``value`` around ``prediction``, matching ``likelihood``'s noise model."""
    if likelihood == "poisson":
        return torch.poisson(prediction, generator=generator)
    z = torch.randn(prediction.shape, dtype=prediction.dtype, generator=generator)
    sigma = torch.where(z >= 0, sigma_plus, sigma_minus)
    return prediction + z * sigma


def calibrate_delta_threshold(
    model: DifferentiableModel,
    theta_truth: torch.Tensor,
    sigma_minus: Optional[torch.Tensor] = None,
    sigma_plus: Optional[torch.Tensor] = None,
    *,
    likelihood: str = "chi2_asymmetric",
    n_toys: int = 100,
    cl: float = 0.68,
    max_iter: int = 200,
    generator: Optional[torch.Generator] = None,
) -> float:
    """Monte-Carlo (parametric-bootstrap) calibration of a confidence-region threshold.

    Simulates ``n_toys`` datasets at ``theta_truth``, refits each with
    ``tpeanuts.inference.fit.minimize_lbfgs``, and returns the empirical
    ``cl``-quantile of

        Delta_toy = (-2 ln L)(theta_truth; toy_data) - (-2 ln L)(theta_hat_toy; toy_data),

    the parametric-bootstrap alternative to ``CHI2_LEVELS_2D``'s asymptotic
    (Wilks' theorem) table values -- use this directly as the contour level
    passed to ``matplotlib``'s ``contour(..., levels=[...])`` on a
    ``loglik_grid`` output (after subtracting the grid's own minimum)
    instead of ``CHI2_LEVELS_2D[cl]`` when Wilks' theorem is not trusted at
    the exposure/statistics in question (see module docstring).

    Args:
        model: Differentiable model; ``theta_truth`` is over its full
            ``model.free`` (this calibrates the whole joint fit's
            threshold, not a 2-parameter slice specifically). Every
            model-specific input is already bound as a field on ``model``.
        theta_truth: True/assumed parameter vector to simulate toys around.
        sigma_minus: One-sided lower uncertainties (``chi2_asymmetric``) or
            None (``poisson``).
        sigma_plus: One-sided upper uncertainties, see ``sigma_minus``.
        likelihood: Name in ``tpeanuts.inference.likelihood.LIKELIHOODS``;
            also selects the toy-sampling model (asymmetric Gaussian or
            Poisson, see ``_sample_toy_value``).
        n_toys: Number of toy datasets. The quantile's own Monte-Carlo
            uncertainty shrinks slowly (~1/sqrt(n_toys)); a few hundred is
            typical for a publication-grade calibration, a few dozen for a
            quick check.
        cl: Confidence level in (0, 1) whose threshold to return, e.g. 0.68.
        max_iter: Maximum LBFGS iterations per toy refit.
        generator: Optional torch.Generator for reproducible toy draws.

    Returns:
        The empirical Delta(-2 ln L) threshold at confidence level ``cl``.

    Raises:
        ValueError: If ``likelihood`` is unknown, or ``cl`` is not in (0, 1).
    """
    if likelihood not in LIKELIHOODS:
        raise ValueError(f"likelihood must be one of {sorted(LIKELIHOODS)}, got {likelihood!r}.")
    if not 0.0 < cl < 1.0:
        raise ValueError(f"cl must be in (0, 1), got {cl}.")
    loss_fn = LIKELIHOODS[likelihood]

    prediction_truth = model.predict(theta_truth).detach()

    deltas = torch.empty(n_toys, dtype=theta_truth.dtype)
    for i in range(n_toys):
        toy_value = _sample_toy_value(prediction_truth, likelihood, sigma_minus, sigma_plus, generator)
        loss_at_truth = float(loss_fn(prediction_truth, toy_value, sigma_minus, sigma_plus))

        theta_hat, _ = minimize_lbfgs(
            model, theta_truth, toy_value, sigma_minus, sigma_plus,
            likelihood=likelihood, max_iter=max_iter,
        )
        prediction_hat = model.predict(theta_hat).detach()
        loss_at_hat = float(loss_fn(prediction_hat, toy_value, sigma_minus, sigma_plus))

        deltas[i] = loss_at_truth - loss_at_hat

    return float(torch.quantile(deltas, cl))

"""Tests for the correlated-Gaussian likelihood and its str|Callable dispatch
through fit_lbfgs/loglik_grid."""

import functools
from dataclasses import dataclass

import torch

from tpeanuts.inference.fit import fit_lbfgs
from tpeanuts.inference.likelihood import (
    chi2_asymmetric,
    cholesky_from_covariance,
    correlated_gaussian_nll,
)
from tpeanuts.inference.scan import loglik_grid


def test_cholesky_from_covariance_reconstructs_the_matrix():
    covariance = torch.tensor([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]], dtype=torch.float64)
    L = cholesky_from_covariance(covariance)
    assert torch.allclose(L @ L.T, covariance, atol=1.0e-10)
    assert torch.allclose(torch.triu(L, diagonal=1), torch.zeros_like(L))


def test_diagonal_covariance_reproduces_chi2_asymmetric():
    prediction = torch.tensor([1.5, 2.5, -0.5], dtype=torch.float64)
    value = torch.tensor([1.0, 3.0, 0.0], dtype=torch.float64)
    sigma = torch.tensor([0.5, 1.0, 0.2], dtype=torch.float64)

    covariance = torch.diag(sigma ** 2)
    L = cholesky_from_covariance(covariance)
    correlated = correlated_gaussian_nll(prediction, value, cholesky_L=L)
    diagonal = chi2_asymmetric(prediction, value, sigma, sigma)
    assert torch.allclose(correlated, diagonal, atol=1.0e-10)


def test_correlated_gaussian_matches_explicit_inverse_form():
    """On a small dense matrix, the Cholesky-solved chi2 must equal the
    textbook r^T V^-1 r form exactly (up to floating-point roundoff) --
    this is the numerically stable form the same computation, not an
    approximation."""
    torch.manual_seed(0)
    n = 5
    A = torch.randn(n, n, dtype=torch.float64)
    covariance = A @ A.T + n * torch.eye(n, dtype=torch.float64)  # guaranteed PD
    prediction = torch.randn(n, dtype=torch.float64)
    value = torch.randn(n, dtype=torch.float64)

    L = cholesky_from_covariance(covariance)
    via_cholesky = correlated_gaussian_nll(prediction, value, cholesky_L=L)

    residual = prediction - value
    via_inverse = residual @ torch.linalg.inv(covariance) @ residual

    assert abs(float(via_cholesky) - float(via_inverse)) < 1.0e-8


def test_correlated_gaussian_nll_rejects_sigma_arguments():
    prediction = torch.zeros(2, dtype=torch.float64)
    value = torch.zeros(2, dtype=torch.float64)
    L = torch.eye(2, dtype=torch.float64)
    try:
        correlated_gaussian_nll(prediction, value, torch.ones(2), None, cholesky_L=L)
        assert False, "expected ValueError"
    except ValueError:
        pass


@dataclass(frozen=True)
class _LinearModel:
    """predict(theta) = A @ theta, the simplest possible DifferentiableModel."""

    A: torch.Tensor
    free: tuple[str, ...]

    def predict(self, theta: torch.Tensor) -> torch.Tensor:
        return self.A @ theta


def test_fit_lbfgs_accepts_a_bound_correlated_gaussian_callable():
    """Integration test for the str|Callable extension to fit_lbfgs: a
    functools.partial-bound correlated_gaussian_nll must work exactly like
    a registered LIKELIHOODS string, recovering a known linear truth."""
    torch.manual_seed(1)
    A = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    model = _LinearModel(A=A, free=("a", "b"))

    theta_truth = torch.tensor([2.0, -1.0], dtype=torch.float64)
    covariance = torch.tensor(
        [[0.04, 0.01, 0.0], [0.01, 0.09, 0.0], [0.0, 0.0, 0.05]], dtype=torch.float64,
    )
    value = model.predict(theta_truth)  # noiseless "data": the fit should land exactly on truth.
    L = cholesky_from_covariance(covariance)
    likelihood_fn = functools.partial(correlated_gaussian_nll, cholesky_L=L)

    theta0 = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)
    result = fit_lbfgs(model, theta0, value, likelihood=likelihood_fn, max_iter=50)

    assert torch.allclose(result.theta_hat, theta_truth, atol=1.0e-6)
    assert torch.isfinite(result.covariance).all()


def test_loglik_grid_accepts_a_bound_correlated_gaussian_callable():
    """Same str|Callable extension, exercised through loglik_grid: the
    minimum of the grid must sit at the known truth."""
    A = torch.eye(2, dtype=torch.float64)
    model = _LinearModel(A=A, free=("a", "b"))
    theta_truth = torch.tensor([1.0, -0.5], dtype=torch.float64)
    covariance = torch.tensor([[0.01, 0.0], [0.0, 0.01]], dtype=torch.float64)
    value = model.predict(theta_truth)
    L = cholesky_from_covariance(covariance)
    likelihood_fn = functools.partial(correlated_gaussian_nll, cholesky_L=L)

    x_grid = torch.linspace(0.8, 1.2, 9, dtype=torch.float64)
    y_grid = torch.linspace(-0.7, -0.3, 9, dtype=torch.float64)
    grid = loglik_grid(
        model, theta_truth, "a", x_grid, "b", y_grid, value, likelihood=likelihood_fn,
    )
    assert grid.shape == (9, 9)
    min_idx = torch.argmin(grid)
    ix, iy = divmod(int(min_idx), grid.shape[1])
    assert abs(float(x_grid[ix]) - 1.0) < 0.06
    assert abs(float(y_grid[iy]) - (-0.5)) < 0.06

"""Tests for automatically fitted perturbative atmosphere profiles."""

import torch

from tpeanuts.core.perturbative.models.atmosphere import (
    AtmospherePolynomialProfile,
    AtmospherePolynomialSegment,
)


DTYPE = torch.float64


def test_atmosphere_polynomial_profile_recovers_node_polynomial():
    boundaries = torch.tensor([0.0, 0.25, 0.5], dtype=DTYPE)
    q = torch.linspace(-1.0, 1.0, 4, dtype=DTYPE)
    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, -1.0, 0.5, 0.25]], dtype=DTYPE)
    powers = torch.arange(4, dtype=DTYPE)
    samples = torch.sum(expected[:, None, :] * q[None, :, None] ** powers, dim=-1)

    profile = AtmospherePolynomialProfile(boundaries, samples)

    torch.testing.assert_close(profile.coefficients, expected, atol=2.0e-14, rtol=2.0e-14)


def test_atmosphere_polynomial_segment_average_includes_only_even_terms():
    segment = AtmospherePolynomialSegment(
        x1=torch.tensor(0.0, dtype=DTYPE),
        x2=torch.tensor(2.0, dtype=DTYPE),
        coefficients=torch.tensor([3.0, 7.0, 6.0, -2.0, 5.0], dtype=DTYPE),
        profile_scale_m=1.0,
        evolution_scale_m=1.0,
    )
    expected = torch.tensor(3.0 + 6.0 / 3.0 + 5.0 / 5.0, dtype=DTYPE)
    torch.testing.assert_close(segment.average, expected, atol=1.0e-14, rtol=1.0e-14)


def test_constant_atmosphere_polynomial_has_zero_residual_integral():
    segment = AtmospherePolynomialSegment(
        x1=torch.tensor(0.0, dtype=DTYPE),
        x2=torch.tensor(0.2, dtype=DTYPE),
        coefficients=torch.tensor([1.2], dtype=DTYPE),
    )
    eigenvalues = torch.tensor([0.2, 0.7, 1.4], dtype=DTYPE)
    residual = segment.residual_integral(eigenvalues[:, None], eigenvalues[None, :])

    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1.0e-14, rtol=1.0e-14)
    assert segment.any_perturbation is False


def test_atmosphere_polynomial_residual_matches_direct_quadrature():
    coefficients = torch.tensor([1.2, -0.4, 0.7, 0.1], dtype=DTYPE)
    segment = AtmospherePolynomialSegment(
        x1=torch.tensor(0.1, dtype=DTYPE),
        x2=torch.tensor(0.6, dtype=DTYPE),
        coefficients=coefficients,
        profile_scale_m=1.0,
        evolution_scale_m=1.0,
    )
    la = torch.tensor([[0.8]], dtype=DTYPE)
    lb = torch.tensor([[0.2]], dtype=DTYPE)
    result = segment.residual_integral(la, lb).squeeze()

    x = torch.linspace(0.1, 0.6, 20001, dtype=DTYPE)
    q = (x - segment.centre) / segment.half_width
    powers = torch.arange(coefficients.numel(), dtype=DTYPE)
    density = torch.sum(coefficients * q[:, None] ** powers, dim=-1)
    residual = density - segment.average
    phase_integral = torch.trapezoid(residual * torch.exp(1j * (la - lb).squeeze() * x), x)
    expected = torch.exp(-1j * la.squeeze() * segment.x2 + 1j * lb.squeeze() * segment.x1) * phase_integral

    # The model result is analytic; the looser tolerance accounts for the
    # finite trapezoidal reference grid used here.
    torch.testing.assert_close(result, expected, atol=1.0e-9, rtol=1.0e-7)

"""Numerical accuracy tests for detector-independent folding."""

import torch

from tpeanuts.detector.common.event_rate import bin_counts
from tpeanuts.detector.common.response import gaussian_response_matrix


DTYPE = torch.float64


def test_gaussian_response_has_full_line_normalization_away_from_boundaries():
    true = torch.tensor([5.0], dtype=DTYPE)
    reconstructed = torch.linspace(0.0, 10.0, 10001, dtype=DTYPE)
    response = gaussian_response_matrix(true, reconstructed, torch.tensor([0.5], dtype=DTYPE))
    integral = torch.trapezoid(response[:, 0], x=reconstructed)
    torch.testing.assert_close(integral, torch.tensor(1.0, dtype=DTYPE), atol=1.0e-10, rtol=0.0)


def test_gaussian_response_does_not_renormalize_truncated_tail():
    true = torch.tensor([0.0], dtype=DTYPE)
    reconstructed = torch.linspace(0.0, 5.0, 5001, dtype=DTYPE)
    response = gaussian_response_matrix(true, reconstructed, torch.tensor([0.5], dtype=DTYPE))
    integral = torch.trapezoid(response[:, 0], x=reconstructed)
    torch.testing.assert_close(integral, torch.tensor(0.5, dtype=DTYPE), atol=1.0e-7, rtol=0.0)


def test_bin_counts_interpolates_exact_bin_edges():
    grid = torch.tensor([0.0, 0.7, 1.4, 2.0], dtype=DTYPE)
    spectrum = 2.0 * grid + 1.0
    edges = torch.tensor([0.2, 0.9, 1.8], dtype=DTYPE)
    counts = bin_counts(spectrum, grid, edges, exposure=1.0)
    primitive = lambda x: x ** 2 + x
    expected = torch.stack((primitive(edges[1]) - primitive(edges[0]), primitive(edges[2]) - primitive(edges[1])))
    torch.testing.assert_close(counts, expected, atol=1.0e-12, rtol=0.0)

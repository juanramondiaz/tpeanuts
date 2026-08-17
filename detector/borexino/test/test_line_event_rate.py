"""Regression tests for physically discrete solar-neutrino lines."""

import torch

from tpeanuts.detector.borexino.event_rate import line_event_rate


def test_line_event_rate_is_finite_and_linear_in_normalized_line_flux():
    dtype = torch.float64
    energies = torch.tensor([0.3843, 0.8618], dtype=dtype)
    weights = torch.tensor([0.103, 0.897], dtype=dtype)
    p_ee = torch.tensor([0.55, 0.53], dtype=dtype)
    probabilities = torch.stack((p_ee, (1.0 - p_ee) / 2, (1.0 - p_ee) / 2), dim=-1)
    bins = torch.tensor([0.05, 0.25, 0.50, 0.80], dtype=dtype)
    one = line_event_rate(probabilities, torch.tensor(1.0e9, dtype=dtype), weights, energies, bins)
    two = line_event_rate(probabilities, torch.tensor(2.0e9, dtype=dtype), weights, energies, bins)
    assert torch.isfinite(one).all()
    assert torch.all(one >= 0)
    torch.testing.assert_close(two, 2.0 * one)


def test_sterile_probability_is_not_reassigned_to_active_mu_tau_flux():
    dtype = torch.float64
    energies = torch.tensor([0.3843, 0.8618], dtype=dtype)
    weights = torch.tensor([0.103, 0.897], dtype=dtype)
    bins = torch.tensor([0.05, 0.25, 0.50, 0.80], dtype=dtype)
    active = torch.tensor([[0.5, 0.25, 0.25], [0.5, 0.25, 0.25]], dtype=dtype)
    sterile = torch.tensor([[0.5, 0.05, 0.05, 0.40], [0.5, 0.05, 0.05, 0.40]], dtype=dtype)

    active_counts = line_event_rate(active, torch.tensor(1.0e9, dtype=dtype), weights, energies, bins)
    sterile_counts = line_event_rate(sterile, torch.tensor(1.0e9, dtype=dtype), weights, energies, bins)

    assert torch.all(sterile_counts <= active_counts)
    assert torch.any(sterile_counts < active_counts)

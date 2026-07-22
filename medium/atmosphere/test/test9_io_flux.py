"""Canonical Honda/Bartol long-form flux loader tests."""

import torch

from tpeanuts.medium.atmosphere.io import load_atmospheric_flux


def test_default_atmospheric_flux_is_honda_long_form():
    table = load_atmospheric_flux(device="cpu", dtype=torch.float64)
    assert table.energy_GeV.shape == table.flux.shape
    assert table.cos_zenith.shape == table.flux.shape
    assert table.azimuth_deg is not None
    assert set(table.particle) == {"numu", "numubar", "nue", "nuebar"}
    assert torch.all(table.energy_GeV > 0)
    assert torch.all(table.flux >= 0)

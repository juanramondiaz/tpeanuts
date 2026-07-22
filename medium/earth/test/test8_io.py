"""Canonical provider-neutral Earth-table loader tests."""

import torch

from tpeanuts.medium.earth.io import load_earth_density


def test_default_earth_density_is_prem_and_has_composition_extension():
    table = load_earth_density(device="cpu", dtype=torch.float64)
    assert table.radius_km.ndim == 1
    assert torch.all(torch.diff(table.radius_km) >= 0)
    assert table.radius_km[0] == 0
    assert table.radius_km[-1] == 6371
    assert table.mass_density_g_cm3.shape == table.radius_km.shape
    assert table.electron_density_mol_cm3 is not None
    assert table.neutron_density_mol_cm3 is not None
    assert torch.all(table.mass_density_g_cm3 > 0)


def test_legacy_earth_provider_has_radial_and_composition_tables():
    table = load_earth_density(provider="legacy", device="cpu", dtype=torch.float64)
    assert table.radius_km[0] == 0
    assert table.radius_km[-1] == 6371
    assert table.electron_density_mol_cm3 is not None
    assert table.neutron_density_mol_cm3 is not None
    assert torch.all(table.mass_density_g_cm3 > 0)

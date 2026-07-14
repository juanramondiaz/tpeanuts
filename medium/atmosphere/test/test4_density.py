#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pytest-compatible tests for tpeanuts.medium.atmosphere.density."""

from __future__ import annotations

import pytest
import torch

from tpeanuts.external.nusquids.core import NuSQuIDSConfig
from tpeanuts.medium.atmosphere.density import (
    atmosphere_density,
    atmosphere_mass_density_profile_exponential,
    atmosphere_mass_density_profile_from_file,
)
from tpeanuts.util.constant import GCM3_TO_NUCLEON_MOLCM3
from tpeanuts.util.context import RuntimeContext


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_context(dtype: torch.dtype = DTYPE) -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, dtype)


def test_exponential_density_matches_closed_form_and_broadcasts():
    context = make_context()
    h = torch.tensor([[0.0], [7.5], [15.0]], device=DEVICE, dtype=DTYPE)
    rho0 = torch.tensor([1.2e-3, 1.0e-3], device=DEVICE, dtype=DTYPE)
    scale = torch.tensor(7.5, device=DEVICE, dtype=DTYPE)

    rho = atmosphere_mass_density_profile_exponential(h, rho0_gcm3=rho0, scale_height_km=scale, context=context)
    expected = rho0 * torch.exp(-h / scale)

    assert rho.shape == (3, 2)
    torch.testing.assert_close(rho, expected, rtol=1.0e-14, atol=1.0e-18)


def test_density_exponential_mass_and_electron_conversion():
    context = make_context()
    h = torch.tensor([0.0, 10.0], device=DEVICE, dtype=DTYPE)
    Ye = torch.tensor(0.5, device=DEVICE, dtype=DTYPE)

    rho = atmosphere_density(h, source="exponential", density_type="mass_density", Ye=Ye, rho0_gcm3=1.2e-3, scale_height_km=7.5, context=context)
    ne = atmosphere_density(h, source="exponential", density_type="electron_density", Ye=Ye, rho0_gcm3=1.2e-3, scale_height_km=7.5, context=context)

    torch.testing.assert_close(ne, Ye * rho * GCM3_TO_NUCLEON_MOLCM3, rtol=1.0e-14, atol=1.0e-18)


def test_file_density_interpolation_and_endpoint_clamping(tmp_path):
    context = make_context()
    density_file = tmp_path / "density_profile.txt"
    density_file.write_text(
        "# h_km rho_gcm3\n"
        "0.0 1.0e-3\n"
        "10.0 2.0e-4\n"
        "20.0 1.0e-5\n",
        encoding="utf-8",
    )
    h = torch.tensor([-5.0, 0.0, 5.0, 10.0, 25.0], device=DEVICE, dtype=DTYPE)

    rho = atmosphere_mass_density_profile_from_file(h, str(density_file), context=context)
    expected = torch.tensor([1.0e-3, 1.0e-3, 6.0e-4, 2.0e-4, 1.0e-5], device=DEVICE, dtype=DTYPE)

    torch.testing.assert_close(rho, expected, rtol=1.0e-14, atol=1.0e-18)


def test_density_file_backend_requires_density_file_and_uses_electron_conversion(tmp_path):
    context = make_context()
    h = torch.tensor([0.0, 10.0], device=DEVICE, dtype=DTYPE)
    density_file = tmp_path / "density_profile.txt"
    density_file.write_text("0.0 1.0e-3\n10.0 2.0e-4\n", encoding="utf-8")

    with pytest.raises(ValueError):
        atmosphere_density(h, source="file", density_type="mass_density", context=context)

    ne = atmosphere_density(h, source="file", density_type="electron_density", density_file=str(density_file), Ye=0.5, context=context)
    expected_mass = torch.tensor([1.0e-3, 2.0e-4], device=DEVICE, dtype=DTYPE)
    torch.testing.assert_close(ne, 0.5 * expected_mass * GCM3_TO_NUCLEON_MOLCM3, rtol=1.0e-14, atol=1.0e-18)


def test_nusquids_density_source_matches_configured_exponential_formula():
    context = make_context()
    h = torch.tensor([0.0, 7.594], device=DEVICE, dtype=DTYPE)
    config = NuSQuIDSConfig(nusquids_rho0_gcm3=1.2e-3, nusquids_scale_height_km=7.594, nusquids_Ye=0.5)

    rho = atmosphere_density(h, source="nusquids", density_type="mass_density", nusquids_config=config, context=context)
    ne = atmosphere_density(h, source="nusquids", density_type="electron_density", nusquids_config=config, context=context)
    expected_rho = torch.tensor([1.2e-3, 1.2e-3 / torch.e], device=DEVICE, dtype=DTYPE)

    torch.testing.assert_close(rho, expected_rho, rtol=1.0e-14, atol=1.0e-18)
    torch.testing.assert_close(ne, 0.5 * rho * GCM3_TO_NUCLEON_MOLCM3, rtol=1.0e-14, atol=1.0e-18)


def test_density_rejects_invalid_source_density_type_and_interpolation(tmp_path):
    context = make_context()
    h = torch.tensor([0.0, 1.0], device=DEVICE, dtype=DTYPE)
    density_file = tmp_path / "density_profile.txt"
    density_file.write_text("0.0 1.0e-3\n1.0 9.0e-4\n", encoding="utf-8")

    with pytest.raises(ValueError):
        atmosphere_density(h, source="exponential", density_type="not_a_density", context=context)
    with pytest.raises(ValueError):
        atmosphere_density(h, source="not_a_source", density_type="mass_density", context=context)
    with pytest.raises(ValueError):
        atmosphere_mass_density_profile_from_file(h, str(density_file), interpolation="nearest", context=context)


def test_density_preserves_context_device_dtype_and_shape():
    context = make_context(torch.float32)
    h = torch.tensor([[0.0, 1.0], [2.0, 3.0]], device=DEVICE, dtype=torch.float32)

    rho = atmosphere_density(h, source="exponential", density_type="mass_density", context=context)

    assert rho.shape == h.shape
    assert rho.device.type == DEVICE.type
    assert rho.dtype == torch.float32
    assert torch.isfinite(rho).all()

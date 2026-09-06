"""Consistency tests for neutrino interaction cross sections.

Module contents:
    DTYPE
        Floating-point type used by the numerical tests.
    test_ibd_kernel_is_in_prompt_not_positron_kinetic_energy(...)
        Check the energy represented by the approximate IBD kernel.
    test_ibd_differential_kernel_integrates_to_total_cross_section(...)
        Check the normalization of the approximate IBD kernel.
    test_deuteron_cc_differential_table_reproduces_total_table(...)
        Compare differential and total charged-current tables.
    test_neutrino_electron_cross_section_respects_kinematic_endpoint(...)
        Check the elastic-scattering kinematic endpoint.
"""

import torch

from tpeanuts.detector.interaction.deuteron import cc_cross_section_grid, sigma_cc_total
from tpeanuts.detector.interaction.inverse_beta_decay import ibd_cross_section_grid, sigma_ibd
from tpeanuts.detector.interaction.neutrino_electron import dsigma_dT, NUE_COUPLINGS
from tpeanuts.util.constant import DELTA_NP_MEV, M_ELECTRON_MEV


DTYPE = torch.float64


def test_ibd_kernel_is_in_prompt_not_positron_kinetic_energy():
    """Check that the approximate IBD kernel peaks at prompt energy.

    Args:
        None.

    Returns:
        None.
    """
    energy = torch.tensor([4.0, 6.0, 8.0], dtype=DTYPE)
    prompt_grid = torch.linspace(0.0, 10.0, 2001, dtype=DTYPE)
    kernel = ibd_cross_section_grid(energy, prompt_grid)
    peak = prompt_grid[kernel.argmax(dim=-1)]
    expected = energy - DELTA_NP_MEV + M_ELECTRON_MEV
    torch.testing.assert_close(peak, expected, atol=0.005, rtol=0.0)


def test_ibd_differential_kernel_integrates_to_total_cross_section():
    """Check that the approximate IBD kernel integrates to its total rate.

    Args:
        None.

    Returns:
        None.
    """
    energy = torch.tensor([2.0, 4.0, 6.0, 8.0], dtype=DTYPE)
    prompt_grid = torch.linspace(0.0, 10.0, 2001, dtype=DTYPE)
    integrated = torch.trapezoid(ibd_cross_section_grid(energy, prompt_grid), x=prompt_grid, dim=-1)
    torch.testing.assert_close(integrated, sigma_ibd(energy), rtol=2.0e-3, atol=0.0)


def test_deuteron_cc_differential_table_reproduces_total_table():
    """Check consistency of the differential and total deuteron CC data.

    Args:
        None.

    Returns:
        None.
    """
    energy = torch.tensor([2.0, 5.0, 8.0, 10.0, 15.0, 20.0], dtype=DTYPE)
    recoil = torch.linspace(0.0, 20.0, 4001, dtype=DTYPE)
    integrated = torch.trapezoid(cc_cross_section_grid(energy, recoil), x=recoil, dim=-1)
    torch.testing.assert_close(integrated, sigma_cc_total(energy), rtol=2.0e-3, atol=0.0)


def test_neutrino_electron_cross_section_respects_kinematic_endpoint():
    """Check that elastic scattering vanishes above its recoil endpoint.

    Args:
        None.

    Returns:
        None.
    """
    energy = torch.tensor(1.0, dtype=DTYPE)
    recoil = torch.tensor([0.0, 0.5, 2.0], dtype=DTYPE)
    cross_section = dsigma_dT(energy, recoil, *NUE_COUPLINGS)
    assert torch.all(cross_section[:2] >= 0.0)
    assert cross_section[-1] == 0.0

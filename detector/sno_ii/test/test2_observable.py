"""Physics closure tests for the SNO salt-phase equivalent-flux observables."""

import pytest
import torch

from tpeanuts.detector.sno_ii.observable import (
    cc_equivalent_flux_spectrum,
    es_equivalent_flux,
    nc_equivalent_flux,
)
from tpeanuts.detector.sno_ii.parameters import CC_BIN_EDGES_MEV, E_NU_GRID_MEV
from tpeanuts.source.solar import SolarNeutrinoSource
from tpeanuts.util.context import RuntimeContext


@pytest.fixture(scope="module")
def flux_tot_MeV():
    """Real 8B differential flux on E_NU_GRID_MEV, cm^-2 s^-1 MeV^-1."""
    context = RuntimeContext.resolve("cpu", torch.float64)
    source = SolarNeutrinoSource.default(context=context)
    return source.total_flux("8B") * source.spectrum("8B", E_NU_GRID_MEV)


def _constant_probabilities(pee: float, n_E: int) -> torch.Tensor:
    """(n_E, 3) probabilities with Pee == pee (constant in energy), split
    equally between nu_mu/nu_tau for the remainder."""
    p = torch.zeros(n_E, 3, dtype=torch.float64)
    p[:, 0] = pee
    p[:, 1] = 0.5 * (1.0 - pee)
    p[:, 2] = 0.5 * (1.0 - pee)
    return p


def test_cc_pee_one_recovers_total_flux(flux_tot_MeV):
    """Summed over every bin (5.5-20.0 MeV, exactly the denominator's own
    integration range) at Pee=1, the CC spectrum must recover Phi_tot exactly."""
    probabilities = _constant_probabilities(1.0, E_NU_GRID_MEV.shape[0])
    spectrum = cc_equivalent_flux_spectrum(probabilities, flux_tot_MeV, CC_BIN_EDGES_MEV)
    assert spectrum.shape == (17,)
    assert torch.all(spectrum >= 0)

    phi_tot = torch.trapezoid(flux_tot_MeV, x=E_NU_GRID_MEV)
    assert abs(float(spectrum.sum()) - float(phi_tot)) < 1.0e-6 * float(phi_tot)


def test_cc_is_exactly_linear_in_a_constant_pee(flux_tot_MeV):
    """For an energy-independent Pee, both numerator and denominator are
    linear in Pee (the fold is linear in flux_e), so the whole spectrum
    must scale by exactly Pee relative to the Pee=1 spectrum -- not just
    approximately, an exact closure property of Eq. A1's construction."""
    n_E = E_NU_GRID_MEV.shape[0]
    spectrum_full = cc_equivalent_flux_spectrum(
        _constant_probabilities(1.0, n_E), flux_tot_MeV, CC_BIN_EDGES_MEV,
    )
    for pee in (0.3, 0.55, 0.9):
        spectrum = cc_equivalent_flux_spectrum(
            _constant_probabilities(pee, n_E), flux_tot_MeV, CC_BIN_EDGES_MEV,
        )
        assert torch.allclose(spectrum, pee * spectrum_full, rtol=1.0e-6)


def test_nc_equals_total_flux_regardless_of_oscillation(flux_tot_MeV):
    """Eq. 10: the NC equivalent flux is the active-flavour integral, exactly
    Phi_tot whenever the three active probabilities sum to 1 (Standard
    Model), for any Pee(E) shape -- not just a constant one."""
    n_E = E_NU_GRID_MEV.shape[0]
    pee = 0.3 + 0.4 * torch.sin(torch.linspace(0.0, 3.0, n_E, dtype=torch.float64)) ** 2
    probabilities = torch.stack([pee, 0.5 * (1 - pee), 0.5 * (1 - pee)], dim=-1)

    nc_flux = nc_equivalent_flux(probabilities, flux_tot_MeV, E_nu_grid_MeV=E_NU_GRID_MEV)
    phi_tot = torch.trapezoid(flux_tot_MeV, x=E_NU_GRID_MEV)
    assert abs(float(nc_flux) - float(phi_tot)) < 1.0e-9 * float(phi_tot)

    nc_flux_no_prob = nc_equivalent_flux(None, flux_tot_MeV, E_nu_grid_MeV=E_NU_GRID_MEV)
    assert abs(float(nc_flux_no_prob) - float(phi_tot)) < 1.0e-9 * float(phi_tot)


def test_es_is_exactly_linear_in_a_constant_pee(flux_tot_MeV):
    """Same exact-linearity argument as CC, applied to ES: at constant Pee,
    ES(Pee) == Pee*ES(1) + (1-Pee)*ES(0) exactly (both bracketing values
    being the pure-nu_e and pure-nu_x equivalent fluxes)."""
    n_E = E_NU_GRID_MEV.shape[0]
    es_pure_e = es_equivalent_flux(_constant_probabilities(1.0, n_E), flux_tot_MeV)
    es_pure_x = es_equivalent_flux(_constant_probabilities(0.0, n_E), flux_tot_MeV)
    assert float(es_pure_e) > float(es_pure_x) >= 0.0  # nu_e ES cross section is larger.

    for pee in (0.2, 0.5, 0.8):
        es = es_equivalent_flux(_constant_probabilities(pee, n_E), flux_tot_MeV)
        expected = pee * es_pure_e + (1.0 - pee) * es_pure_x
        assert abs(float(es) - float(expected)) < 1.0e-6 * float(es_pure_e)


def test_gradient_flows_through_cc_and_es(flux_tot_MeV):
    """Autograd must reach a free Pee parameter through CC and ES (needed
    for a real oscillation fit) -- NC is deliberately excluded here, since
    Eq. 10 makes it exactly Pee-independent in the Standard Model (see
    test_nc_equals_total_flux_regardless_of_oscillation and the dedicated
    zero-gradient check below)."""
    n_E = E_NU_GRID_MEV.shape[0]
    pee_param = torch.tensor(0.55, dtype=torch.float64, requires_grad=True)
    probabilities = torch.stack([
        pee_param.expand(n_E),
        0.5 * (1 - pee_param).expand(n_E),
        0.5 * (1 - pee_param).expand(n_E),
    ], dim=-1)

    cc = cc_equivalent_flux_spectrum(probabilities, flux_tot_MeV, CC_BIN_EDGES_MEV).sum()
    es = es_equivalent_flux(probabilities, flux_tot_MeV)

    for observable in (cc, es):
        grad = torch.autograd.grad(observable, pee_param, retain_graph=True)[0]
        assert torch.isfinite(grad)
        assert float(grad) != 0.0


def test_nc_gradient_is_exactly_zero_in_the_standard_model(flux_tot_MeV):
    """The flip side of test_nc_equals_total_flux_regardless_of_oscillation:
    with the three active probabilities summing to 1 by construction, NC's
    gradient w.r.t. Pee must be exactly (not just approximately) zero."""
    n_E = E_NU_GRID_MEV.shape[0]
    pee_param = torch.tensor(0.55, dtype=torch.float64, requires_grad=True)
    probabilities = torch.stack([
        pee_param.expand(n_E),
        0.5 * (1 - pee_param).expand(n_E),
        0.5 * (1 - pee_param).expand(n_E),
    ], dim=-1)
    nc = nc_equivalent_flux(probabilities, flux_tot_MeV, E_nu_grid_MeV=E_NU_GRID_MEV)
    grad = torch.autograd.grad(nc, pee_param)[0]
    assert float(grad) == 0.0

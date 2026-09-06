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
Pytest-compatible checks specific to Non-Standard Interactions (NSI):
``tpeanuts.core.BSM.bsm_nsi.NSIConfig`` and its integration with the
common Hamiltonian builders and numerical evolutor.

Generic Hamiltonian-builder machinery is covered in
``core/common/test/test3_hamiltonian.py``; 3+1 sterile-neutrino checks live
in ``test3_bsm_sterile.py``.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from tpeanuts.core.common.hamiltonian import hamiltonian_flavour, hamiltonian_reduced
from tpeanuts.core.BSM.bsm_nsi import NSIConfig
from tpeanuts.core.common.oscillation import OscillationParameters, oscillation_needs_neutron_composition
from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.config.presets import NSI_PRESETS, OSCILLATION_PRESETS, list_presets
from tpeanuts.core.numerical.evolutor import evolutor_numerical
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.test_utils import assert_close


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CDTYPE = torch.complex128


def make_context() -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, DTYPE)


def make_oscillation(*, antinu=False, NSI_extension: str | None = None) -> OscillationParameters:
    return PropagationConfig.oscillation_parameters_from_preset(
        "_SM_NUFIT52_NO", antinu=antinu, NSI_extension=NSI_extension, context=make_context(),
    )


def eye_like(matrix: torch.Tensor) -> torch.Tensor:
    return torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype).expand(matrix.shape)


# ---------------------------------------------------------------------------
# NSIConfig dataclass — defaults, properties, preset construction
# ---------------------------------------------------------------------------

def test_default_config_is_sm_limit_without_cp_violation():
    cfg = NSIConfig()
    assert cfg.is_sm_limit
    assert not cfg.has_cp_violation


def test_has_cp_violation_reflects_only_offdiagonal_imaginary_parts():
    assert not NSIConfig(eps_ee=0.3).has_cp_violation
    assert not NSIConfig(eps_emu_re=0.1).has_cp_violation
    assert NSIConfig(eps_emu_im=0.05).has_cp_violation
    assert NSIConfig(eps_etau_im=-0.01).has_cp_violation
    assert NSIConfig(eps_mutau_im=0.02).has_cp_violation


def test_complex_properties_match_re_im_fields():
    cfg = NSIConfig(
        eps_emu_re=0.1, eps_emu_im=0.2,
        eps_etau_re=-0.3, eps_etau_im=0.05,
        eps_mutau_re=0.01, eps_mutau_im=-0.02,
    )
    assert cfg.eps_emu == complex(0.1, 0.2)
    assert cfg.eps_etau == complex(-0.3, 0.05)
    assert cfg.eps_mutau == complex(0.01, -0.02)


def test_epsilon_tensor_base_is_hermitian_with_correct_entries():
    cfg = NSIConfig(
        eps_ee=0.30, eps_mumu=0.0, eps_tautau=0.15,
        eps_emu_re=0.02, eps_emu_im=0.01,
        eps_etau_re=-0.05, eps_etau_im=0.0,
        eps_mutau_re=0.005, eps_mutau_im=-0.003,
    )
    eps = cfg.epsilon_tensor_base(device=DEVICE, real_dtype=DTYPE)

    assert eps.shape == (3, 3)
    assert eps.dtype == CDTYPE
    assert_close(eps, eps.conj().transpose(-2, -1), name="epsilon is Hermitian")

    assert_close(eps[0, 0].real, torch.tensor(0.30, dtype=DTYPE), name="eps_ee")
    assert_close(eps[1, 1].real, torch.tensor(0.0, dtype=DTYPE), name="eps_mumu")
    assert_close(eps[2, 2].real, torch.tensor(0.15, dtype=DTYPE), name="eps_tautau")
    assert_close(eps[0, 1], torch.tensor(complex(0.02, 0.01), dtype=CDTYPE), name="eps_emu")
    assert_close(eps[0, 2], torch.tensor(complex(-0.05, 0.0), dtype=CDTYPE), name="eps_etau")
    assert_close(eps[1, 2], torch.tensor(complex(0.005, -0.003), dtype=CDTYPE), name="eps_mutau")


def test_epsilon_tensor_base_dtype_follows_real_dtype():
    cfg = NSIConfig(eps_ee=0.1)
    eps64 = cfg.epsilon_tensor_base(device=DEVICE, real_dtype=torch.float64)
    eps32 = cfg.epsilon_tensor_base(device=DEVICE, real_dtype=torch.float32)

    assert eps64.dtype == torch.complex128
    assert eps32.dtype == torch.complex64


# ---------------------------------------------------------------------------
# Autograd: eps_*/eps_*_n accept differentiable tensors (_hermitian_3x3)
# ---------------------------------------------------------------------------

def test_diagonal_eps_ee_tensor_field_keeps_gradient_connected():
    x = torch.tensor(0.2, dtype=DTYPE, requires_grad=True)
    cfg = NSIConfig(eps_ee=x, device=DEVICE, real_dtype=DTYPE)

    assert cfg.epsilon.requires_grad
    loss = cfg.epsilon[0, 0].real
    loss.backward()
    assert x.grad is not None
    assert_close(x.grad, torch.tensor(1.0, dtype=DTYPE), name="d(epsilon_ee)/d(eps_ee) == 1")


def test_offdiagonal_eps_emu_tensor_fields_keep_gradient_connected():
    re = torch.tensor(0.1, dtype=DTYPE, requires_grad=True)
    im = torch.tensor(-0.05, dtype=DTYPE, requires_grad=True)
    cfg = NSIConfig(eps_emu_re=re, eps_emu_im=im, device=DEVICE, real_dtype=DTYPE)

    assert cfg.epsilon.requires_grad
    # Sum of |eps_emu|^2 contributions from both symmetric entries.
    loss = (cfg.epsilon[0, 1].abs() ** 2 + cfg.epsilon[1, 0].abs() ** 2).real
    loss.backward()
    assert re.grad is not None and im.grad is not None
    assert torch.isfinite(re.grad) and torch.isfinite(im.grad)
    assert re.grad.item() != 0.0
    assert im.grad.item() != 0.0


def test_epsilon_n_tensor_field_keeps_gradient_connected():
    x = torch.tensor(0.3, dtype=DTYPE, requires_grad=True)
    cfg = NSIConfig(eps_ee_n=x, device=DEVICE, real_dtype=DTYPE)

    assert cfg.epsilon_n.requires_grad
    cfg.epsilon_n[0, 0].real.backward()
    assert x.grad is not None
    assert_close(x.grad, torch.tensor(1.0, dtype=DTYPE), name="d(epsilon_n_ee)/d(eps_ee_n) == 1")


def test_eps_ee_gradient_flows_through_hamiltonian_reduced():
    """End-to-end check: a differentiable eps_ee must produce a finite,
    non-zero gradient on the assembled Hamiltonian, matching the pattern
    inference.model_solar.SolarNSIOscillationModel relies on."""
    ctx = make_context()
    x = torch.tensor(0.1, dtype=DTYPE, requires_grad=True)
    osc = dataclasses.replace(
        make_oscillation(), nsi=NSIConfig(eps_ee=x, device=DEVICE, real_dtype=DTYPE),
    )
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)

    H = hamiltonian_reduced(osc, 1000.0, n_e, context=ctx)
    H[0, 0].real.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad)
    assert x.grad.item() != 0.0


def test_float_eps_ee_still_produces_a_plain_leaf_tensor():
    """A plain Python float (the common case) must keep producing a
    non-grad-tracking tensor, exactly as before this refactor."""
    cfg = NSIConfig(eps_ee=0.3, device=DEVICE, real_dtype=DTYPE)
    assert not cfg.epsilon.requires_grad
    assert_close(cfg.epsilon[0, 0].real, torch.tensor(0.3, dtype=DTYPE), name="eps_ee value")


def test_eps_emu_property_detaches_gradient_tensor_fields():
    """eps_emu/etc. are documented as non-differentiable convenience
    accessors (_py_float detaches); they must not raise even when the
    underlying fields are grad-tracking tensors."""
    re = torch.tensor(0.1, dtype=DTYPE, requires_grad=True)
    im = torch.tensor(-0.02, dtype=DTYPE, requires_grad=True)
    cfg = NSIConfig(eps_emu_re=re, eps_emu_im=im, device=DEVICE, real_dtype=DTYPE)

    value = cfg.eps_emu
    assert isinstance(value, complex)
    assert value == pytest.approx(complex(0.1, -0.02))


def test_str_does_not_raise_with_gradient_tensor_fields():
    x = torch.tensor(0.2, dtype=DTYPE, requires_grad=True)
    cfg = NSIConfig(eps_ee=x, eps_ee_n=0.1, device=DEVICE, real_dtype=DTYPE)
    text = str(cfg)
    assert "0.2000" in text


# ---------------------------------------------------------------------------
# Identity-based equality/hashing (eq=False) -- see NSIConfig's class
# docstring for why a field-tuple-based __eq__ would be unsound now that
# eps_* can hold tensors and from_raw_epsilon leaves them at 0.0 defaults.
# ---------------------------------------------------------------------------

def test_default_configs_are_not_equal_by_value_only_by_identity():
    a = NSIConfig()
    b = NSIConfig()
    assert a != b
    assert a == a


def test_raw_epsilon_config_does_not_spuriously_equal_sm_limit():
    """Regression test: before eq=False, a config built via from_raw_epsilon
    (whose eps_* scalar fields stay at their 0.0 defaults) compared equal to
    the plain SM-limit NSIConfig() despite carrying a non-zero epsilon."""
    eps3 = torch.zeros((3, 3), device=DEVICE, dtype=CDTYPE)
    eps3[0, 0] = 0.5 + 0j
    raw_cfg = NSIConfig.from_raw_epsilon(eps3)
    sm_cfg = NSIConfig()

    assert raw_cfg != sm_cfg
    assert not raw_cfg.is_sm_limit
    assert sm_cfg.is_sm_limit


def test_nsiconfig_is_hashable_with_tensor_fields():
    x = torch.tensor(0.2, dtype=DTYPE, requires_grad=True)
    cfg = NSIConfig(eps_ee=x, device=DEVICE, real_dtype=DTYPE)
    hash(cfg)  # must not raise


def test_from_preset_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown NSI preset"):
        NSIConfig.from_preset("does_not_exist")


@pytest.mark.parametrize("name", list_presets(NSI_PRESETS))
def test_all_registered_nsi_presets_build_hermitian_epsilon(name):
    cfg = NSIConfig.from_preset(name)
    eps = cfg.epsilon_tensor_base(device=DEVICE, real_dtype=DTYPE)

    assert eps.shape == (3, 3)
    assert torch.isfinite(eps.real).all() and torch.isfinite(eps.imag).all()
    assert_close(eps, eps.conj().transpose(-2, -1), name=f"epsilon Hermitian [{name}]")


def test_only_sm_no_nsi_preset_is_the_sm_limit():
    for name in list_presets(NSI_PRESETS):
        cfg = NSIConfig.from_preset(name)
        expected = (name == "sm_no_nsi")
        assert cfg.is_sm_limit == expected, f"is_sm_limit mismatch for preset {name!r}"


# ---------------------------------------------------------------------------
# Integration with the BSM Hamiltonian builders
# ---------------------------------------------------------------------------

def test_epsilon_all_zero_matches_epsilon_none_in_reduced_hamiltonian():
    ctx = make_context()
    osc_nsi = make_oscillation(NSI_extension="sm_no_nsi")
    osc_sm = make_oscillation()
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)

    H_eps = hamiltonian_reduced(osc_nsi, E, n_e, context=ctx)
    H_none = hamiltonian_reduced(osc_sm, E, n_e, context=ctx)

    assert_close(H_eps, H_none, atol=1.0e-12, rtol=1.0e-12, name="epsilon=0 matches epsilon=None")


def test_epsilon_all_zero_matches_epsilon_none_in_flavour_hamiltonian():
    ctx = make_context()
    osc_nsi = make_oscillation(NSI_extension="sm_no_nsi")
    osc_sm = make_oscillation()
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)

    H_eps = hamiltonian_flavour(osc_nsi, E, n_e, context=ctx)
    H_none = hamiltonian_flavour(osc_sm, E, n_e, context=ctx)

    assert_close(H_eps, H_none, atol=1.0e-12, rtol=1.0e-12, name="epsilon=0 matches epsilon=None")


def test_positive_eps_ee_strengthens_and_negative_weakens_matter_potential():
    ctx = make_context()
    osc_sm = make_oscillation()
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)

    H_sm = hamiltonian_reduced(osc_sm, E, n_e, context=ctx)
    V_ee_sm = H_sm[0, 0].real

    osc_pos = dataclasses.replace(osc_sm, nsi=NSIConfig(eps_ee=0.30, device=DEVICE, real_dtype=DTYPE))
    osc_neg = make_oscillation(NSI_extension="nsi_lma_dark_esteban2018")

    H_pos = hamiltonian_reduced(osc_pos, E, n_e, context=ctx)
    H_neg = hamiltonian_reduced(osc_neg, E, n_e, context=ctx)

    assert H_pos[0, 0].real > V_ee_sm, "eps_ee > 0 must strengthen the (0,0) matter entry"
    assert H_neg[0, 0].real < V_ee_sm, "eps_ee = -2.0 must flip/weaken the (0,0) matter entry"


@pytest.mark.parametrize("name", list_presets(NSI_PRESETS))
def test_nsi_hamiltonian_hermitian_with_real_eigenvalues_for_all_presets(name):
    ctx = make_context()
    osc = make_oscillation(NSI_extension=name)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)

    H = hamiltonian_flavour(osc, E, n_e, context=ctx)
    assert_close(H, H.conj().transpose(-2, -1), name=f"H Hermitian [{name}]")

    eigvals = torch.linalg.eigvalsh(H)
    assert torch.isfinite(eigvals).all(), f"non-finite eigenvalues for preset {name!r}"


def test_nsi_vacuum_evolution_is_unitary_and_conserves_probability():
    osc = make_oscillation(NSI_extension="nsi_lma_dark_esteban2018")
    n_e = torch.tensor([1.0, 1.2, 1.4], device=DEVICE, dtype=DTYPE)
    dx = torch.tensor([0.02, 0.03, 0.04], device=DEVICE, dtype=DTYPE)

    S = evolutor_numerical(osc, 1000.0, n_e, dx, device=DEVICE, dtype=DTYPE)

    identity = eye_like(S)
    assert_close(S.conj().transpose(-2, -1) @ S, identity, atol=1.0e-10, rtol=1.0e-10, name="NSI evolutor unitarity")

    P = S.abs() ** 2
    assert_close(P.sum(dim=-1), torch.ones(3, device=DEVICE, dtype=DTYPE), atol=1.0e-10, rtol=1.0e-10, name="row probability sums to 1")


# ---------------------------------------------------------------------------
# LMA-Dark degeneracy (Esteban et al. 2018, arXiv:1805.04530)
# ---------------------------------------------------------------------------

def test_lma_dark_preset_uses_the_canonical_eps_ee_minus_two():
    cfg = NSIConfig.from_preset("nsi_lma_dark_esteban2018")
    assert cfg.eps_ee == pytest.approx(-2.0)
    assert cfg.eps_mumu == 0.0
    assert cfg.eps_tautau == 0.0


def test_lma_dark_angular_degeneracy_sin2_theta12_equals_cos2_theta12_dark():
    theta12_lma = math.radians(OSCILLATION_PRESETS["_SM_NUFIT52_NO"]["theta12_deg"])
    theta12_dark = math.radians(OSCILLATION_PRESETS["_LMA_DARK_NUFIT52_NO"]["theta12_deg"])

    sin2_lma = math.sin(theta12_lma) ** 2
    cos2_dark = math.cos(theta12_dark) ** 2

    assert sin2_lma == pytest.approx(cos2_dark, abs=1.0e-12)
    assert theta12_dark == pytest.approx(math.pi / 2.0 - theta12_lma, abs=1.0e-12)


# ---------------------------------------------------------------------------
# NSIConfig.epsilon_tensor (embeds self.epsilon for an arbitrary flavour count)
# ---------------------------------------------------------------------------

def test_epsilon_tensor_passthrough_when_shape_matches():
    ctx = make_context()
    eps4 = torch.zeros((4, 4), device=DEVICE, dtype=CDTYPE)
    eps4[0, 1] = 0.1 + 0.05j
    eps4[1, 0] = eps4[0, 1].conj()
    cfg = NSIConfig.from_raw_epsilon(eps4)
    out = cfg.epsilon_tensor(n_flavours=4, context=ctx)
    assert_close(out, eps4, name="4x4 epsilon passthrough")


def test_epsilon_tensor_embeds_3x3_for_larger_n_flavours():
    ctx = make_context()
    eps3 = torch.zeros((3, 3), device=DEVICE, dtype=CDTYPE)
    eps3[0, 0] = 0.3
    cfg = NSIConfig.from_raw_epsilon(eps3)
    out = cfg.epsilon_tensor(n_flavours=4, context=ctx)

    assert out.shape == (4, 4)
    assert_close(out[:3, :3], eps3, name="embedded active block")
    assert_close(out[3, :], torch.zeros(4, device=DEVICE, dtype=CDTYPE))


def test_epsilon_tensor_incompatible_shape_raises():
    ctx = make_context()
    cfg = NSIConfig.from_raw_epsilon(torch.zeros((2, 2), device=DEVICE, dtype=CDTYPE))
    with pytest.raises(ValueError, match="epsilon must have final dimensions"):
        cfg.epsilon_tensor(n_flavours=4, context=ctx)


def test_epsilon_tensor_uses_self_epsilon():
    ctx = make_context()
    cfg = NSIConfig.from_preset("nsi_dune_etau", device=DEVICE, real_dtype=DTYPE)
    out = cfg.epsilon_tensor(n_flavours=3, context=ctx)
    assert_close(out, cfg.epsilon, name="embeds self.epsilon")


def test_epsilon_tensor_missing_epsilon_raises():
    """epsilon is always auto-populated by __post_init__ now; force it back
    to None to exercise epsilon_tensor's defensive check."""
    cfg = NSIConfig()
    object.__setattr__(cfg, "epsilon", None)
    with pytest.raises(ValueError, match="No epsilon matrix available"):
        cfg.epsilon_tensor(n_flavours=3, context=make_context())


# ---------------------------------------------------------------------------
# Composition dependence: eps_*_n / epsilon_n / epsilon_n_tensor
# ---------------------------------------------------------------------------

def test_default_config_has_zero_epsilon_n_and_no_neutron_coupling():
    cfg = NSIConfig()
    assert cfg.epsilon_n is not None
    assert torch.all(cfg.epsilon_n == 0)
    assert not cfg.has_neutron_coupling
    assert cfg.is_sm_limit


def test_eps_ee_n_alone_breaks_sm_limit_and_sets_neutron_coupling():
    cfg = NSIConfig(eps_ee_n=0.05)
    assert cfg.has_neutron_coupling
    assert not cfg.is_sm_limit
    assert torch.all(cfg.epsilon == 0), "eps_ee_n must not leak into the electron-block epsilon"


def test_epsilon_n_tensor_base_is_hermitian_with_correct_entries():
    cfg = NSIConfig(
        eps_ee_n=0.10, eps_mumu_n=0.0, eps_tautau_n=-0.20,
        eps_emu_n_re=0.01, eps_emu_n_im=-0.02,
        eps_etau_n_re=0.03, eps_etau_n_im=0.0,
        eps_mutau_n_re=-0.004, eps_mutau_n_im=0.006,
    )
    eps_n = cfg.epsilon_n_tensor_base(device=DEVICE, real_dtype=DTYPE)

    assert eps_n.shape == (3, 3)
    assert eps_n.dtype == CDTYPE
    assert_close(eps_n, eps_n.conj().transpose(-2, -1), name="epsilon_n is Hermitian")
    assert_close(eps_n[0, 0].real, torch.tensor(0.10, dtype=DTYPE), name="eps_ee_n")
    assert_close(eps_n[2, 2].real, torch.tensor(-0.20, dtype=DTYPE), name="eps_tautau_n")
    assert_close(eps_n[0, 1], torch.tensor(complex(0.01, -0.02), dtype=CDTYPE), name="eps_emu_n")


def test_has_cp_violation_reflects_offdiagonal_imaginary_parts_of_epsilon_n_too():
    assert not NSIConfig(eps_ee_n=0.1).has_cp_violation
    assert not NSIConfig(eps_emu_n_re=0.1).has_cp_violation
    assert NSIConfig(eps_emu_n_im=0.05).has_cp_violation
    assert NSIConfig(eps_etau_n_im=-0.01).has_cp_violation
    assert NSIConfig(eps_mutau_n_im=0.02).has_cp_violation


def test_complex_properties_n_match_re_im_fields():
    cfg = NSIConfig(
        eps_emu_n_re=0.1, eps_emu_n_im=0.2,
        eps_etau_n_re=-0.3, eps_etau_n_im=0.05,
        eps_mutau_n_re=0.01, eps_mutau_n_im=-0.02,
    )
    assert cfg.eps_emu_n == complex(0.1, 0.2)
    assert cfg.eps_etau_n == complex(-0.3, 0.05)
    assert cfg.eps_mutau_n == complex(0.01, -0.02)


def test_epsilon_n_tensor_embeds_3x3_for_larger_n_flavours():
    ctx = make_context()
    cfg = NSIConfig(eps_ee_n=0.2, device=DEVICE, real_dtype=DTYPE)
    out = cfg.epsilon_n_tensor(n_flavours=4, context=ctx)

    assert out.shape == (4, 4)
    assert_close(out[:3, :3], cfg.epsilon_n, name="embedded epsilon_n active block")
    assert_close(out[3, :], torch.zeros(4, device=DEVICE, dtype=CDTYPE))


def test_epsilon_n_tensor_missing_raises():
    cfg = NSIConfig()
    object.__setattr__(cfg, "epsilon_n", None)
    with pytest.raises(ValueError, match="No epsilon_n matrix available"):
        cfg.epsilon_n_tensor(n_flavours=3, context=make_context())


def test_from_raw_epsilon_n_is_validated_and_stored():
    eps3 = torch.zeros((3, 3), device=DEVICE, dtype=CDTYPE)
    eps3[0, 0] = 0.1
    eps3_n = torch.zeros((3, 3), device=DEVICE, dtype=CDTYPE)
    eps3_n[0, 1] = 0.02 + 0.01j
    eps3_n[1, 0] = eps3_n[0, 1].conj()

    cfg = NSIConfig.from_raw_epsilon(eps3, epsilon_n=eps3_n)

    assert_close(cfg.epsilon, eps3, name="from_raw_epsilon stores epsilon as-is")
    assert_close(cfg.epsilon_n, eps3_n, name="from_raw_epsilon stores epsilon_n as-is")
    assert cfg.has_neutron_coupling


def test_from_raw_epsilon_n_non_hermitian_raises():
    eps3 = torch.zeros((3, 3), device=DEVICE, dtype=CDTYPE)
    non_hermitian_n = torch.zeros((3, 3), device=DEVICE, dtype=CDTYPE)
    non_hermitian_n[0, 1] = 1.0 + 0.0j  # no conjugate on [1, 0]: breaks Hermiticity

    with pytest.raises(ValueError, match="epsilon_n must be Hermitian"):
        NSIConfig.from_raw_epsilon(eps3, epsilon_n=non_hermitian_n)


def test_from_raw_epsilon_without_epsilon_n_keeps_default_zero_epsilon_n():
    eps3 = torch.zeros((3, 3), device=DEVICE, dtype=CDTYPE)
    eps3[0, 0] = 0.3
    cfg = NSIConfig.from_raw_epsilon(eps3)

    assert torch.all(cfg.epsilon_n == 0)
    assert not cfg.has_neutron_coupling


def test_from_raw_epsilon_mismatched_epsilon_n_block_size_raises():
    eps3 = torch.zeros((3, 3), device=DEVICE, dtype=CDTYPE)
    eps4 = torch.zeros((4, 4), device=DEVICE, dtype=CDTYPE)

    with pytest.raises(ValueError, match="same active-flavour block size"):
        NSIConfig.from_raw_epsilon(eps3, epsilon_n=eps4)


@pytest.mark.parametrize("bad_epsilon", [
    torch.tensor(1.0 + 0.0j, dtype=CDTYPE),
    torch.zeros(3, dtype=CDTYPE),
])
def test_from_raw_epsilon_rejects_fewer_than_2_dims(bad_epsilon):
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        NSIConfig.from_raw_epsilon(bad_epsilon.to(device=DEVICE))


# ---------------------------------------------------------------------------
# Composition dependence: integration with hamiltonian_matter_reduced
# ---------------------------------------------------------------------------

def test_epsilon_n_zero_leaves_hamiltonian_unaffected_by_n_n_mol_cm3():
    """Default epsilon_n=0 must reproduce the pre-existing behaviour exactly,
    for any flavour count: n_n_mol_cm3 stays inert unless epsilon_n is set.
    """
    ctx = make_context()
    osc = make_oscillation(NSI_extension="nsi_diagonal_biggio2009")
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)
    n_n = torch.tensor(1.3, device=DEVICE, dtype=DTYPE)

    H_with_n_n = hamiltonian_reduced(osc, 1000.0, n_e, n_n_mol_cm3=n_n, context=ctx)
    H_without_n_n = hamiltonian_reduced(osc, 1000.0, n_e, context=ctx)

    assert_close(H_with_n_n, H_without_n_n, name="epsilon_n=0 makes n_n_mol_cm3 inert")


def test_epsilon_n_nonzero_changes_hamiltonian_when_n_n_supplied():
    ctx = make_context()
    osc_sm = make_oscillation()
    nsi = NSIConfig(eps_ee=0.1, eps_ee_n=0.05, device=DEVICE, real_dtype=DTYPE)
    osc = dataclasses.replace(osc_sm, nsi=nsi)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)
    n_n = torch.tensor(1.3, device=DEVICE, dtype=DTYPE)
    osc_zero_n = dataclasses.replace(osc_sm, nsi=NSIConfig(eps_ee=0.1, device=DEVICE, real_dtype=DTYPE))

    H_zero_n = hamiltonian_reduced(osc_zero_n, 1000.0, n_e, context=ctx)
    H_supplied = hamiltonian_reduced(osc, 1000.0, n_e, n_n_mol_cm3=n_n, context=ctx)

    assert torch.max(torch.abs(H_zero_n - H_supplied)) > 1.0e-8, (
        "a non-zero epsilon_n must change H relative to the epsilon_n=0 baseline"
    )


def test_epsilon_n_nonzero_without_n_n_mol_cm3_raises():
    """A non-zero eps_*_n is an explicit user request for the composition
    term; omitting n_n_mol_cm3 must raise rather than silently fall back to
    the epsilon-only Hamiltonian (unlike the sterile NC term, this is not an
    optional refinement -- see hamiltonian_matter_reduced's docstring)."""
    ctx = make_context()
    osc_sm = make_oscillation()
    nsi = NSIConfig(eps_ee=0.1, eps_ee_n=0.05, device=DEVICE, real_dtype=DTYPE)
    osc = dataclasses.replace(osc_sm, nsi=nsi)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="eps_\\*_n"):
        hamiltonian_reduced(osc, 1000.0, n_e, context=ctx)
    with pytest.raises(ValueError, match="eps_\\*_n"):
        hamiltonian_flavour(osc, 1000.0, n_e, context=ctx)


def test_epsilon_n_matches_manual_v_cc_on_neutron_density():
    """epsilon_eff = epsilon + (n_n/n_e)*epsilon_n reformulated as
    V_CC(n_e)*(diag(1,0,0)+epsilon) + V_CC(n_n)*epsilon_n -- check this
    matches an explicit division-based construction for a 3-flavour pmns.
    """
    ctx = make_context()
    osc_sm = make_oscillation()
    nsi = NSIConfig(
        eps_ee=0.10, eps_etau_re=0.02,
        eps_ee_n=0.30, eps_mutau_n_re=0.01, eps_mutau_n_im=-0.005,
        device=DEVICE, real_dtype=DTYPE,
    )
    osc = dataclasses.replace(osc_sm, nsi=nsi)
    n_e = torch.tensor(2.1, device=DEVICE, dtype=DTYPE)
    n_n = torch.tensor(1.7, device=DEVICE, dtype=DTYPE)

    H = hamiltonian_reduced(osc, 1000.0, n_e, n_n_mol_cm3=n_n, context=ctx)

    from tpeanuts.core.common.potential import matter_potential_cc as _V_cc
    V_e = _V_cc(n_e, antinu=osc.antinu, context=ctx)
    ratio = (n_n / n_e).to(dtype=CDTYPE)
    eps_eff = nsi.epsilon + ratio * nsi.epsilon_n
    O = osc.pmns.outer_block(osc.antinu)
    D_flavour = torch.zeros(3, 3, device=DEVICE, dtype=CDTYPE)
    D_flavour[0, 0] = 1.0
    D_flavour = D_flavour + eps_eff
    Hmat_expected = V_e.to(dtype=CDTYPE) * (O.conj().transpose(-2, -1) @ D_flavour @ O)

    from tpeanuts.core.common.hamiltonian import hamiltonian_kinetic_reduced
    Hkin = hamiltonian_kinetic_reduced(osc, 1000.0, osc.pmns.reduced(antinu=osc.antinu))

    assert_close(H, Hkin + Hmat_expected, atol=1.0e-9, rtol=1.0e-9, name="V_CC(n_n)*epsilon_n matches division-based epsilon_eff")


def test_oscillation_needs_neutron_composition_reflects_has_neutron_coupling():
    osc_sm = make_oscillation()
    assert not oscillation_needs_neutron_composition(osc_sm)

    osc_eps = make_oscillation(NSI_extension="nsi_diagonal_biggio2009")
    assert not oscillation_needs_neutron_composition(osc_eps), (
        "eps_*_n all zero (no preset sets them) must not trigger the composition need"
    )

    nsi_n = NSIConfig(eps_ee_n=0.1, device=DEVICE, real_dtype=DTYPE)
    osc_eps_n = dataclasses.replace(osc_sm, nsi=nsi_n)
    assert oscillation_needs_neutron_composition(osc_eps_n)


def test_epsilon_n_hamiltonian_stays_hermitian():
    ctx = make_context()
    osc_sm = make_oscillation()
    nsi = NSIConfig(eps_ee_n=0.2, eps_emu_n_re=0.03, eps_emu_n_im=0.01, device=DEVICE, real_dtype=DTYPE)
    osc = dataclasses.replace(osc_sm, nsi=nsi)
    n_e = torch.tensor(1.8, device=DEVICE, dtype=DTYPE)
    n_n = torch.tensor(1.4, device=DEVICE, dtype=DTYPE)

    H = hamiltonian_flavour(osc, 1000.0, n_e, n_n_mol_cm3=n_n, context=ctx)
    assert_close(H, H.conj().transpose(-2, -1), name="composition-dependent NSI Hamiltonian stays Hermitian")

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
from tpeanuts.core.common.oscillation import OscillationParameters
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


def with_epsilon(cfg: NSIConfig) -> NSIConfig:
    """Populate a directly-constructed NSIConfig's epsilon field (only
    from_preset does this automatically)."""
    return dataclasses.replace(cfg, epsilon=cfg.epsilon_tensor_base(device=DEVICE, real_dtype=DTYPE))


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

    osc_pos = dataclasses.replace(osc_sm, nsi=with_epsilon(NSIConfig(eps_ee=0.30)))
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
    cfg = dataclasses.replace(NSIConfig(), epsilon=eps4)
    out = cfg.epsilon_tensor(n_flavours=4, context=ctx)
    assert_close(out, eps4, name="4x4 epsilon passthrough")


def test_epsilon_tensor_embeds_3x3_for_larger_n_flavours():
    ctx = make_context()
    eps3 = torch.zeros((3, 3), device=DEVICE, dtype=CDTYPE)
    eps3[0, 0] = 0.3
    cfg = dataclasses.replace(NSIConfig(), epsilon=eps3)
    out = cfg.epsilon_tensor(n_flavours=4, context=ctx)

    assert out.shape == (4, 4)
    assert_close(out[:3, :3], eps3, name="embedded active block")
    assert_close(out[3, :], torch.zeros(4, device=DEVICE, dtype=CDTYPE))


def test_epsilon_tensor_incompatible_shape_raises():
    ctx = make_context()
    cfg = dataclasses.replace(NSIConfig(), epsilon=torch.zeros((2, 2), device=DEVICE, dtype=CDTYPE))
    with pytest.raises(ValueError, match="epsilon must have final dimensions"):
        cfg.epsilon_tensor(n_flavours=4, context=ctx)


def test_epsilon_tensor_uses_self_epsilon():
    ctx = make_context()
    cfg = NSIConfig.from_preset("nsi_dune_etau", device=DEVICE, real_dtype=DTYPE)
    out = cfg.epsilon_tensor(n_flavours=3, context=ctx)
    assert_close(out, cfg.epsilon, name="embeds self.epsilon")


def test_epsilon_tensor_missing_epsilon_raises():
    with pytest.raises(ValueError, match="No epsilon matrix available"):
        NSIConfig().epsilon_tensor(n_flavours=3, context=make_context())

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
Spyder-compatible tests for core.py.

No pytest required.
"""



import torch

from tpeanuts.external.mceq.config import (
    MCEqModelConfig,
    DEFAULT_PRIMARY_MODEL,
    DEFAULT_DENSITY_MODEL,
)

from tpeanuts.external.mceq.core import (
    resolve_primary_model,
    resolve_density_model,
    theta_to_float,
    set_mceq_logging,
    init_mceq,
)

from tpeanuts.util.test_utils import (
    assert_true,
    assert_close,
    assert_raises,
    run_test_suite,
)


DEVICE = "cpu"
DTYPE = torch.float64


MODEL_CONFIG = MCEqModelConfig(
    interaction_model="SIBYLL23D",
    primary_model="HillasGaisser H3a",
    density_model="CORSIKA",
    info=False,
)


# ============================================================
# Resolver tests
# ============================================================

def test_resolve_primary_model_default():
    primary = resolve_primary_model(DEFAULT_PRIMARY_MODEL)

    print("resolved primary model:", primary)

    assert_true(primary is not None)


def test_resolve_primary_model_from_string():
    primary = resolve_primary_model("HillasGaisser H3a")

    print("resolved primary model:", primary)

    assert_true(primary is not None)


def test_resolve_primary_model_accepts_tuple():
    primary_default = resolve_primary_model("HillasGaisser H3a")

    primary = resolve_primary_model(primary_default)

    print("input tuple :", primary_default)
    print("output tuple:", primary)

    assert_true(primary == primary_default)


def test_resolve_primary_model_invalid_raises():
    assert_raises(
        ValueError,
        resolve_primary_model,
        "INVALID_PRIMARY_MODEL",
    )


def test_resolve_density_model_default():
    density = resolve_density_model(DEFAULT_DENSITY_MODEL)

    print("resolved density model:", density)

    assert_true(density is not None)


def test_resolve_density_model_from_string():
    density = resolve_density_model("CORSIKA")

    print("resolved density model:", density)

    assert_true(density is not None)


def test_resolve_density_model_invalid_raises():
    assert_raises(
        ValueError,
        resolve_density_model,
        "INVALID_density_MODEL",
    )


# ============================================================
# Theta conversion tests
# ============================================================

def test_theta_to_float_from_float():
    theta = theta_to_float(45.0)

    print("theta:", theta)

    assert_close(theta, 45.0)


def test_theta_to_float_from_int():
    theta = theta_to_float(30)

    print("theta:", theta)

    assert_close(theta, 30.0)


def test_theta_to_float_from_tensor_scalar():
    theta = theta_to_float(
        torch.tensor(60.0, dtype=DTYPE, device=DEVICE)
    )

    print("theta:", theta)

    assert_close(theta, 60.0)


def test_theta_to_float_from_tensor_vector_uses_first_value():
    theta = theta_to_float(
        torch.tensor([15.0, 30.0], dtype=DTYPE, device=DEVICE)
    )

    print("theta:", theta)

    assert_close(theta, 15.0)


def test_theta_to_float_rejects_negative():
    assert_raises(
        ValueError,
        theta_to_float,
        -1.0,
    )


def test_theta_to_float_rejects_90_deg():
    assert_raises(
        ValueError,
        theta_to_float,
        90.0,
    )


def test_theta_to_float_rejects_above_90_deg():
    assert_raises(
        ValueError,
        theta_to_float,
        91.0,
    )


# ============================================================
# Logging test
# ============================================================

def test_set_mceq_logging_runs():
    set_mceq_logging(info=False)
    set_mceq_logging(info=True)
    set_mceq_logging(info=False)

    print("set_mceq_logging executed without error.")

    assert_true(True)


# ============================================================
# mceq initialization tests
# ============================================================

def test_init_mceq_returns_object():
    mceq = init_mceq(
        theta_deg=0.0,
        config=MODEL_CONFIG,
    )

    print("mceq object:", mceq)

    assert_true(mceq is not None)


def test_init_mceq_has_energy_grid():
    mceq = init_mceq(
        theta_deg=0.0,
        config=MODEL_CONFIG,
    )

    print("has e_grid:", hasattr(mceq, "e_grid"))

    assert_true(hasattr(mceq, "e_grid"))
    assert_true(len(mceq.e_grid) > 0)


def test_init_mceq_has_density_model():
    mceq = init_mceq(
        theta_deg=0.0,
        config=MODEL_CONFIG,
    )

    print("has density_model:", hasattr(mceq, "density_model"))

    assert_true(hasattr(mceq, "density_model"))
    assert_true(mceq.density_model is not None)


def test_init_mceq_theta_45():
    mceq = init_mceq(
        theta_deg=45.0,
        config=MODEL_CONFIG,
    )

    print("mceq theta 45 object:", mceq)

    assert_true(mceq is not None)
    assert_true(hasattr(mceq, "e_grid"))


def test_init_mceq_accepts_overrides():
    mceq = init_mceq(
        theta_deg=0.0,
        config=MODEL_CONFIG,
        interaction_model="SIBYLL23D",
        primary_model="HillasGaisser H3a",
        density_model="CORSIKA",
        info=False,
    )

    print("mceq with overrides:", mceq)

    assert_true(mceq is not None)
    assert_true(hasattr(mceq, "e_grid"))


def test_init_mceq_rejects_invalid_theta():
    assert_raises(
        ValueError,
        init_mceq,
        90.0,
        config=MODEL_CONFIG,
    )


def test_init_mceq_rejects_invalid_interaction_model():
    bad_config = MCEqModelConfig(
        interaction_model="INVALID_MODEL",
        primary_model="HillasGaisser H3a",
        density_model="CORSIKA",
    )

    assert_raises(
        ValueError,
        init_mceq,
        0.0,
        config=bad_config,
    )


def test_init_mceq_rejects_invalid_primary_model():
    bad_config = MCEqModelConfig(
        interaction_model="SIBYLL23D",
        primary_model="INVALID_PRIMARY_MODEL",
        density_model="CORSIKA",
    )

    assert_raises(
        ValueError,
        init_mceq,
        0.0,
        config=bad_config,
    )


def test_init_mceq_rejects_invalid_density_model():
    bad_config = MCEqModelConfig(
        interaction_model="SIBYLL23D",
        primary_model="HillasGaisser H3a",
        density_model="INVALID_density_MODEL",
    )

    assert_raises(
        ValueError,
        init_mceq,
        0.0,
        config=bad_config,
    )


# ============================================================
# Runner
# ============================================================

def run_core_tests(verbose_traceback=False):
    tests = [
        test_resolve_primary_model_default,
        test_resolve_primary_model_from_string,
        test_resolve_primary_model_accepts_tuple,
        test_resolve_primary_model_invalid_raises,
        test_resolve_density_model_default,
        test_resolve_density_model_from_string,
        test_resolve_density_model_invalid_raises,
        test_theta_to_float_from_float,
        test_theta_to_float_from_int,
        test_theta_to_float_from_tensor_scalar,
        test_theta_to_float_from_tensor_vector_uses_first_value,
        test_theta_to_float_rejects_negative,
        test_theta_to_float_rejects_90_deg,
        test_theta_to_float_rejects_above_90_deg,
        test_set_mceq_logging_runs,
        test_init_mceq_returns_object,
        test_init_mceq_has_energy_grid,
        test_init_mceq_has_density_model,
        test_init_mceq_theta_45,
        test_init_mceq_accepts_overrides,
        test_init_mceq_rejects_invalid_theta,
        test_init_mceq_rejects_invalid_interaction_model,
# %%
        test_init_mceq_rejects_invalid_primary_model,

        test_init_mceq_rejects_invalid_density_model,
    ]

    return run_test_suite(
        tests,
        suite_name="core mceq tests",
        verbose_traceback=verbose_traceback,
    )


if __name__ == "__main__":
    run_core_tests(verbose_traceback=True)
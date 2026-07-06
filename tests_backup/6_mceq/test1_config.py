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
Spyder-compatible tests for tpeanuts.external.mceq.config.

Run directly in Spyder or with:

    from tpeanuts.external.mceq.tests.test_config_spyder import run_config_tests
    run_config_tests()
"""



import os
from pathlib import Path

import torch

from tpeanuts.external.mceq.config import (
    RunConfig,
    MCEqModelConfig,
    GridConfig,
    SmoothingConfig,
    make_config,
    default_config,
)
from tpeanuts.io.io_atmosphere import OutputConfig
from tpeanuts.util.parallel import ParallelConfig

from tpeanuts.util.test_utils import (
    assert_true,
    assert_raises,
    run_test_suite,
)

NOTEBOOK_STEM = Path(__file__).stem
DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_TEST_ROOT = Path(OUTPUT_ROOT / "test")
OUTPUT_DIR = OUTPUT_TEST_ROOT / "mceq" / NOTEBOOK_STEM


# ============================================================
# tests
# ============================================================

def test_default_config_is_valid():
    config = default_config()
    config.validate()

    print("Default RunConfig:")
    print(config)

    assert_true(isinstance(config, RunConfig))


def test_make_config_is_valid():
    config = make_config(
        theta_grid_deg=torch.Tensor([0.0, 15.0, 30.0, 60.0]),
        X_grid_gcm2=torch.linspace(1.0, 1030.0, 20),
        h_grid_km=torch.linspace(0.0, 80.0, 50),
    )

    config.validate()

    print("theta_grid_deg:", config.grid.theta_grid_deg)
    print("X_grid shape:", config.grid.X_grid_gcm2.shape)
    print("h_grid shape:", config.grid.h_grid_km.shape)

    assert_true(isinstance(config, RunConfig))
    assert_true(config.grid.theta_grid_deg.shape == (4,))
    assert_true(config.grid.X_grid_gcm2.shape == (20,))
    assert_true(config.grid.h_grid_km.shape == (50,))


def test_invalid_interaction_model_raises():
    config = MCEqModelConfig(
        interaction_model="INVALID_MODEL"
    )

    assert_raises(ValueError, config.validate)


def test_invalid_density_model_raises():
    config = MCEqModelConfig(
        density_model="INVALID_density"
    )

    assert_raises(ValueError, config.validate)


def test_invalid_primary_model_raises():
    config = MCEqModelConfig(
        primary_model="INVALID_PRIMARY"
    )

    assert_raises(ValueError, config.validate)


def test_theta_grid_must_be_1d():
    grid = GridConfig(
        theta_grid_deg=torch.Tensor([[0.0, 10.0], [20.0, 30.0]])
    )

    assert_raises(ValueError, grid.validate)


def test_theta_grid_must_be_below_90_deg():
    grid = GridConfig(
        theta_grid_deg=torch.Tensor([0.0, 45.0, 90.0])
    )

    assert_raises(ValueError, grid.validate)


def test_theta_grid_cannot_be_negative():
    grid = GridConfig(
        theta_grid_deg=torch.Tensor([-1.0, 0.0, 45.0])
    )

    assert_raises(ValueError, grid.validate)


def test_X_grid_must_be_strictly_increasing():
    grid = GridConfig(
        X_grid_gcm2=torch.Tensor([1.0, 10.0, 5.0, 1030.0])
    )

    assert_raises(ValueError, grid.validate)


def test_h_grid_must_be_strictly_increasing():
    grid = GridConfig(
        h_grid_km=torch.Tensor([0.0, 10.0, 5.0, 80.0])
    )

    assert_raises(ValueError, grid.validate)


def test_X_obs_must_be_inside_X_grid():
    grid = GridConfig(
        X_grid_gcm2=torch.linspace(1.0, 100.0, 20),
        X_obs_gcm2=1030.0,
    )

    assert_raises(ValueError, grid.validate)


def test_smoothing_config_valid():
    config = SmoothingConfig(
        method="spline",
        smoothing=1.0e-4,
        gaussian_sigma=2.0,
        positive_only=True,
    )

    config.validate()

    print("SmoothingConfig:")
    print(config)


def test_invalid_smoothing_method_raises():
    config = SmoothingConfig(
        method="invalid"
    )

    assert_raises(ValueError, config.validate)


def test_negative_smoothing_raises():
    config = SmoothingConfig(
        smoothing=-1.0
    )

    assert_raises(ValueError, config.validate)


def test_negative_gaussian_sigma_raises():
    config = SmoothingConfig(
        gaussian_sigma=-1.0
    )

    assert_raises(ValueError, config.validate)


def test_output_config_valid():
    config = OutputConfig(
        output_dir=OUTPUT_DIR,
        filename="test_output.npz",
        dtype=torch.float32,
        compressed=True,
    )

    config.validate()

    print("OutputConfig:")
    print(config)


def test_output_invalid_dtype_raises():
    config = OutputConfig(
        dtype="not_a_dtype"
    )

    assert_raises(ValueError, config.validate)


def test_parallel_config_valid():
    config = ParallelConfig(
        parallel=True,
        n_jobs=2,
        backend="loky",
    )

    config.validate()

    print("ParallelConfig:")
    print(config)


def test_parallel_n_jobs_zero_raises():
    config = ParallelConfig(
        n_jobs=0
    )

    assert_raises(ValueError, config.validate)


def test_parallel_invalid_backend_raises():
    config = ParallelConfig(
        backend="invalid_backend"
    )

    assert_raises(ValueError, config.validate)


def test_run_config_requires_non_empty_flavours():
    config = RunConfig(
        flavours={}
    )

    assert_raises(ValueError, config.validate)


def test_run_config_requires_string_flavour_names():
    config = RunConfig(
        flavours={1: "numu"}
    )

    assert_raises(ValueError, config.validate)


def test_run_config_requires_string_particle_names():
    config = RunConfig(
        flavours={"numu": 1}
    )

    assert_raises(ValueError, config.validate)


# ============================================================
# Runner
# ============================================================

def run_config_tests(verbose_traceback=False):
    tests = [
        test_default_config_is_valid,
        test_make_config_is_valid,
        test_invalid_interaction_model_raises,
        test_invalid_density_model_raises,
        test_invalid_primary_model_raises,
        test_theta_grid_must_be_1d,
        test_theta_grid_must_be_below_90_deg,
        test_theta_grid_cannot_be_negative,
        test_X_grid_must_be_strictly_increasing,
        test_h_grid_must_be_strictly_increasing,
        test_X_obs_must_be_inside_X_grid,
        test_smoothing_config_valid,
        test_invalid_smoothing_method_raises,
        test_negative_smoothing_raises,
        test_negative_gaussian_sigma_raises,
        test_output_config_valid,
        test_output_invalid_dtype_raises,
        test_parallel_config_valid,
        test_parallel_n_jobs_zero_raises,
        test_parallel_invalid_backend_raises,
        test_run_config_requires_non_empty_flavours,
        test_run_config_requires_string_flavour_names,
        test_run_config_requires_string_particle_names,
    ]

    return run_test_suite(
        tests,
        suite_name="CONFIG tests",
        verbose_traceback=verbose_traceback,
    )


if __name__ == "__main__":
    run_config_tests(verbose_traceback=True)

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
Spyder-compatible tests for tpeanuts.external.mceq.generator.

The expensive mceq production-profile routine is patched with a synthetic
torch result so these tests only validate orchestration and I/O.
"""



import os
from pathlib import Path
import shutil

import torch

from tpeanuts.atmosphere.geometry import detector_alpha_to_surface_theta
from tpeanuts.external.mceq import generator
from tpeanuts.external.mceq.config import (
    GridConfig,
    MCEqModelConfig,
    SmoothingConfig,
)
from tpeanuts.io.io_atmosphere import OutputConfig
from tpeanuts.util.parallel import ParallelConfig
from tpeanuts.io.io_atmosphere import load_directory
from tpeanuts.util.torch_util import _default_device

from tpeanuts.util.test_utils import (
    assert_true,
    assert_close,
    run_test_suite,
)


DEVICE = "cpu"
DTYPE = torch.float64

NOTEBOOK_STEM = Path(__file__).stem
DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_TEST_ROOT = Path(OUTPUT_ROOT / "test")
OUTPUT_DIR = OUTPUT_TEST_ROOT / "mceq" / NOTEBOOK_STEM


def fake_profile_result(
    theta_deg,
    particle,
    model_config=None,
    grid_config=None,
    smoothing_config=None,
    *,
    device=None,
    dtype=torch.float64,
):
    device = DEVICE if device is None else device

    E = torch.logspace(0.0, 2.0, 4, device=device, dtype=dtype)
    h = grid_config.h_grid_km.to(device=device, dtype=dtype)
    X = grid_config.X_grid_gcm2.to(device=device, dtype=dtype)

    f_Eh = torch.ones((E.numel(), h.numel()), device=device, dtype=dtype)
    f_Eh = f_Eh / torch.trapezoid(f_Eh, x=h, dim=1)[:, None]

    phi_E = E ** (-2.0)
    phi_Eh = phi_E[:, None] * f_Eh
    flux_XE = torch.ones((X.numel(), E.numel()), device=device, dtype=dtype)

    return {
        "theta_deg": torch.as_tensor(theta_deg, device=device, dtype=dtype),
        "E_grid_GeV": E,
        "X_grid_gcm2": X,
        "h_grid_km": h,
        "flux_XE": flux_XE,
        "flux_smooth_XE": flux_XE.clone(),
        "dPhi_dX_XE": torch.zeros_like(flux_XE),
        "X_of_h_gcm2": torch.linspace(
            1030.0,
            0.0,
            h.numel(),
            device=device,
            dtype=dtype,
        ),
        "dXdh_gcm2_per_km": torch.ones(h.numel(), device=device, dtype=dtype),
        "source_Eh": f_Eh.clone(),
        "f_Eh": f_Eh,
        "phi_E_obs": phi_E,
        "phi_Eh": phi_Eh,
    }


def make_configs(tmpdir):
    model_config = MCEqModelConfig()
    grid_config = GridConfig(
        theta_grid_deg=torch.tensor([0.0, 20.0], dtype=DTYPE),
        X_grid_gcm2=torch.linspace(1.0, 1030.0, 6, dtype=DTYPE),
        h_grid_km=torch.linspace(0.0, 20.0, 5, dtype=DTYPE),
        X_obs_gcm2=1030.0,
    )
    smoothing_config = SmoothingConfig(
        method="none",
        smoothing=0.0,
        positive_only=True,
    )
    output_config = OutputConfig(
        output_dir=tmpdir,
        filename="flux.pt",
        dtype=torch.float32,
        overwrite=True,
    )
    parallel_config = ParallelConfig(parallel=False)

    return (
        model_config,
        grid_config,
        smoothing_config,
        output_config,
        parallel_config,
    )


def test_detector_alpha_to_surface_theta_surface_identity():
    theta = detector_alpha_to_surface_theta(
        30.0,
        detector_depth_m=0.0,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("theta:", theta)

    assert_close(float(theta.item()), 30.0)


def test_detector_alpha_accepts_default_device_function():
    theta = detector_alpha_to_surface_theta(
        30.0,
        detector_depth_m=0.0,
        device=_default_device,
        dtype=DTYPE,
    )

    print("theta with callable device:", theta)

    assert_close(float(theta.item()), 30.0)


def test_generate_one_particle_angle_saves_file():
    original = generator.production_profiles_all_energies_from_flux_gradient
    tmpdir = OUTPUT_DIR

    try:
        generator.production_profiles_all_energies_from_flux_gradient = fake_profile_result

        (
            model_config,
            grid_config,
            smoothing_config,
            output_config,
            _,
        ) = make_configs(tmpdir)

        result = generator.generate_flux_for_particle_angle(
            particle="numu",
            theta_deg=20.0,
            flavour_name="numu",
            model_config=model_config,
            grid_config=grid_config,
            smoothing_config=smoothing_config,
            output_config=output_config,
            device=_default_device,
            debug=True,
        )

        print("output path:", result["output_path"])

        assert_true(os.path.exists(result["output_path"]))
        assert_true(result["phi_Eh"].shape == (4, 5))

    finally:
        generator.production_profiles_all_energies_from_flux_gradient = original
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_generate_particles_angle_grid_and_load_directory():
    original = generator.production_profiles_all_energies_from_flux_gradient
    tmpdir = OUTPUT_DIR

    try:
        generator.production_profiles_all_energies_from_flux_gradient = fake_profile_result

        (
            model_config,
            grid_config,
            smoothing_config,
            output_config,
            parallel_config,
        ) = make_configs(tmpdir)

        results = generator.generate_flux_for_particles_angle_grid(
            particles={"numu": "numu", "nue": "nue"},
            theta_grid_deg=torch.tensor([0.0, 20.0], dtype=DTYPE),
            model_config=model_config,
            grid_config=grid_config,
            smoothing_config=smoothing_config,
            output_config=output_config,
            parallel_config=parallel_config,
            debug=True,
        )

        loaded = load_directory(
            tmpdir,
            map_location="cpu",
            dtype=DTYPE,
            device=DEVICE,
        )

        print("result groups:", results.keys())
        print("loaded groups:", loaded.keys())

        assert_true(set(results.keys()) == {"numu", "nue"})
        assert_true(set(loaded.keys()) == {"numu", "nue"})
        assert_true(loaded["numu"]["phi_E_theta_h"].shape == (2, 4, 5))
        assert_true(loaded["nue"]["phi_E_theta"].shape == (2, 4))

    finally:
        generator.production_profiles_all_energies_from_flux_gradient = original
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_generator_tests(verbose_traceback=False):
    tests = [
        test_detector_alpha_to_surface_theta_surface_identity,
        test_detector_alpha_accepts_default_device_function,
        test_generate_one_particle_angle_saves_file,
        test_generate_particles_angle_grid_and_load_directory,
    ]

    return run_test_suite(
        tests,
        suite_name="mceq GENERATOR tests",
        verbose_traceback=verbose_traceback,
    )


if __name__ == "__main__":
    run_generator_tests(verbose_traceback=True)

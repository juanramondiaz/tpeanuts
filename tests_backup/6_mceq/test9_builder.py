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
Spyder-compatible tests and visual diagnostics for builder.py.

No pytest required.

These tests are integration-style tests because builder.py calls the
complete mceq profile reconstruction pipeline.
"""



import os
from pathlib import Path
import shutil

import torch
import matplotlib.pyplot as plt

from tpeanuts.external.mceq.config import (
    RunConfig,
    MCEqModelConfig,
    GridConfig,
    SmoothingConfig,
)
from tpeanuts.io.io_atmosphere import OutputConfig
from tpeanuts.util.parallel import ParallelConfig

from tpeanuts.external.mceq.builder import (
    split_run_config,
    validate_particle_name,
    build_theta_result,
    build_phi_E_theta_h_for_particle,
    build_phi_E_theta_h_for_particle_parallel,
    build_all_flavours,
)

from tpeanuts.external.mceq.diagnostics import (
    diagnose_result,
    compute_profile_normalization,
    reconstruct_flux_from_profile,
)

from tpeanuts.util.test_utils import (
    assert_true,
    assert_raises,
    run_test_suite,
)


DEVICE = "cpu"
DTYPE = torch.float64

PARTICLE = "numu"

NOTEBOOK_STEM = Path(__file__).stem
DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_TEST_ROOT = Path(OUTPUT_ROOT / "test")
OUTPUT_DIR = OUTPUT_TEST_ROOT / "mceq" / NOTEBOOK_STEM


# ============================================================
# Minimal configs
# ============================================================

def make_minimal_configs(
    output_dir=None,
    overwrite=True,
    parallel=False,
    n_jobs=2,
):
    if output_dir is None:
        output_dir = OUTPUT_DIR

    model_config = MCEqModelConfig(
        interaction_model="SIBYLL23D",
        primary_model="HillasGaisser H3a",
        density_model="CORSIKA",
        info=False,
    )

    grid_config = GridConfig(
        theta_grid_deg=torch.tensor(
            [0.0],
            dtype=DTYPE,
        ).numpy(),
        X_grid_gcm2=torch.linspace(
            10.0,
            1030.0,
            8,
            dtype=DTYPE,
        ).numpy(),
        h_grid_km=torch.linspace(
            0.0,
            80.0,
            40,
            dtype=DTYPE,
        ).numpy(),
        X_obs_gcm2=1030.0,
    )

    smoothing_config = SmoothingConfig(
        method="gaussian",
        gaussian_sigma=1.0,
        positive_only=True,
    )

    output_config = OutputConfig(
        output_dir=output_dir,
        filename="builder_test_output.pt",
        dtype=torch.float64,
        compressed=True,
        overwrite=overwrite,
        save_intermediate=True,
    )

    parallel_config = ParallelConfig(
        parallel=parallel,
        n_jobs=n_jobs,
        backend="loky",
    )

    return (
        model_config,
        grid_config,
        smoothing_config,
        output_config,
        parallel_config,
        output_dir,
    )


def make_minimal_run_config(
    output_dir=None,
    overwrite=True,
    parallel=False,
    n_jobs=2,
):
    (
        model_config,
        grid_config,
        smoothing_config,
        output_config,
        parallel_config,
        output_dir,
    ) = make_minimal_configs(
        output_dir=output_dir,
        overwrite=overwrite,
        parallel=parallel,
        n_jobs=n_jobs,
    )

    run_config = RunConfig(
        model=model_config,
        grid=grid_config,
        smoothing=smoothing_config,
        output=output_config,
        parallel=parallel_config,
        flavours={
            "numu": "numu",
        },
    )

    return run_config, output_dir


# ============================================================
# Basic tests
# ============================================================

def test_split_run_config():
    run_config, output_dir = make_minimal_run_config()

    try:
        model, grid, smoothing, output, parallel = split_run_config(run_config)

        print("model    :", model)
        print("grid     :", grid)
        print("smoothing:", smoothing)
        print("output   :", output)
        print("parallel :", parallel)

        assert_true(isinstance(model, MCEqModelConfig))
        assert_true(isinstance(grid, GridConfig))
        assert_true(isinstance(smoothing, SmoothingConfig))
        assert_true(isinstance(output, OutputConfig))
        assert_true(isinstance(parallel, ParallelConfig))

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_validate_particle_name_accepts_valid_string():
    validate_particle_name("numu")
    validate_particle_name("antinumu")
    validate_particle_name("mu+")

    print("valid particle names accepted.")

    assert_true(True)


def test_validate_particle_name_rejects_non_string():
    assert_raises(
        TypeError,
        validate_particle_name,
        123,
    )


def test_validate_particle_name_rejects_empty_string():
    assert_raises(
        ValueError,
        validate_particle_name,
        "",
    )


# ============================================================
# build_theta_result tests
# ============================================================

def test_build_theta_result_without_saving():
    (
        model_config,
        grid_config,
        smoothing_config,
        output_config,
        parallel_config,
        output_dir,
    ) = make_minimal_configs()

    try:
        result = build_theta_result(
            theta_deg=0.0,
            particle=PARTICLE,
            model_config=model_config,
            grid_config=grid_config,
            smoothing_config=smoothing_config,
            save=False,
            flavour_name="numu",
            output_config=output_config,
            device=DEVICE,
            dtype=DTYPE,
        )

        print("result keys:", result.keys())

        required_keys = [
            "theta_deg",
            "E_grid_GeV",
            "X_grid_gcm2",
            "h_grid_km",
            "flux_XE",
            "flux_smooth_XE",
            "dPhi_dX_XE",
            "X_of_h_gcm2",
            "dXdh_gcm2_per_km",
            "source_Eh",
            "f_Eh",
            "phi_E_obs",
            "phi_Eh",
            "particle",
            "flavour_name",
            "build_time_sec",
        ]

        for key in required_keys:
            assert_true(key in result)

        assert_true(result["particle"] == PARTICLE)
        assert_true(result["flavour_name"] == "numu")

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_build_theta_result_shapes():
    (
        model_config,
        grid_config,
        smoothing_config,
        output_config,
        parallel_config,
        output_dir,
    ) = make_minimal_configs()

    try:
        result = build_theta_result(
            theta_deg=0.0,
            particle=PARTICLE,
            model_config=model_config,
            grid_config=grid_config,
            smoothing_config=smoothing_config,
            save=False,
            flavour_name="numu",
            output_config=output_config,
            device=DEVICE,
            dtype=DTYPE,
        )

        n_X = result["X_grid_gcm2"].numel()
        n_E = result["E_grid_GeV"].numel()
        n_h = result["h_grid_km"].numel()

        print("n_X:", n_X)
        print("n_E:", n_E)
        print("n_h:", n_h)

        print("flux_XE shape:", result["flux_XE"].shape)
        print("f_Eh shape   :", result["f_Eh"].shape)
        print("phi_Eh shape :", result["phi_Eh"].shape)

        assert_true(result["flux_XE"].shape == (n_X, n_E))
        assert_true(result["f_Eh"].shape == (n_E, n_h))
        assert_true(result["phi_Eh"].shape == (n_E, n_h))
        assert_true(result["phi_E_obs"].shape == (n_E,))

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_build_theta_result_positive_and_finite():
    (
        model_config,
        grid_config,
        smoothing_config,
        output_config,
        parallel_config,
        output_dir,
    ) = make_minimal_configs()

    try:
        result = build_theta_result(
            theta_deg=0.0,
            particle=PARTICLE,
            model_config=model_config,
            grid_config=grid_config,
            smoothing_config=smoothing_config,
            save=False,
            flavour_name="numu",
            output_config=output_config,
            device=DEVICE,
            dtype=DTYPE,
        )

        for key in ["flux_XE", "f_Eh", "phi_E_obs", "phi_Eh"]:
            x = result[key]

            print(key, "min:", float(x.min().item()), "max:", float(x.max().item()))

            assert_true(not torch.isnan(x).any().item())
            assert_true(not torch.isinf(x).any().item())
            assert_true(torch.all(x >= 0.0).item())

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_build_theta_result_profile_normalization():
    (
        model_config,
        grid_config,
        smoothing_config,
        output_config,
        parallel_config,
        output_dir,
    ) = make_minimal_configs()

    try:
        result = build_theta_result(
            theta_deg=0.0,
            particle=PARTICLE,
            model_config=model_config,
            grid_config=grid_config,
            smoothing_config=smoothing_config,
            save=False,
            flavour_name="numu",
            output_config=output_config,
            device=DEVICE,
            dtype=DTYPE,
        )

        norm_E = compute_profile_normalization(
            h_grid_km=result["h_grid_km"],
            f_Eh=result["f_Eh"],
            device=DEVICE,
            dtype=DTYPE,
        )

        max_dev = torch.max(torch.abs(norm_E - 1.0)).item()

        print("normalization min:", float(norm_E.min().item()))
        print("normalization max:", float(norm_E.max().item()))
        print("max |norm - 1|   :", max_dev)

        assert_true(max_dev < 5.0e-3)

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_build_theta_result_flux_reconstruction():
    (
        model_config,
        grid_config,
        smoothing_config,
        output_config,
        parallel_config,
        output_dir,
    ) = make_minimal_configs()

    try:
        result = build_theta_result(
            theta_deg=0.0,
            particle=PARTICLE,
            model_config=model_config,
            grid_config=grid_config,
            smoothing_config=smoothing_config,
            save=False,
            flavour_name="numu",
            output_config=output_config,
            device=DEVICE,
            dtype=DTYPE,
        )

        phi_rec = reconstruct_flux_from_profile(
            h_grid_km=result["h_grid_km"],
            phi_Eh=result["phi_Eh"],
            device=DEVICE,
            dtype=DTYPE,
        )

        rel_err = torch.abs(phi_rec - result["phi_E_obs"]) / torch.clamp(
            torch.abs(result["phi_E_obs"]),
            min=1.0e-30,
        )

        max_rel = float(torch.max(rel_err).item())

        print("max relative reconstruction error:", max_rel)

        assert_true(max_rel < 5.0e-3)

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_build_theta_result_with_saving_creates_file():
    (
        model_config,
        grid_config,
        smoothing_config,
        output_config,
        parallel_config,
        output_dir,
    ) = make_minimal_configs(overwrite=True)

    try:
        result = build_theta_result(
            theta_deg=0.0,
            particle=PARTICLE,
            model_config=model_config,
            grid_config=grid_config,
            smoothing_config=smoothing_config,
            save=True,
            flavour_name="numu",
            output_config=output_config,
            device=DEVICE,
            dtype=DTYPE,
        )

        print("output path:", result.get("output_path"))

        assert_true("output_path" in result)
        assert_true(os.path.exists(result["output_path"]))
        assert_true(result["output_path"].endswith(".pt"))

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_build_theta_result_save_true_requires_output_config():
    (
        model_config,
        grid_config,
        smoothing_config,
        output_config,
        parallel_config,
        output_dir,
    ) = make_minimal_configs()

    try:
        assert_raises(
            ValueError,
            build_theta_result,
            0.0,
            PARTICLE,
            model_config,
            grid_config,
            smoothing_config,
            save=True,
            flavour_name="numu",
            output_config=None,
            device=DEVICE,
            dtype=DTYPE,
        )

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


# ============================================================
# Particle-level builder tests
# ============================================================

def test_build_phi_E_theta_h_for_particle_serial():
    (
        model_config,
        grid_config,
        smoothing_config,
        output_config,
        parallel_config,
        output_dir,
    ) = make_minimal_configs()

    try:
        result = build_phi_E_theta_h_for_particle(
            theta_grid_deg=grid_config.theta_grid_deg,
            particle=PARTICLE,
            model_config=model_config,
            grid_config=grid_config,
            smoothing_config=smoothing_config,
            output_config=output_config,
            flavour_name="numu",
            save=False,
            show_progress=False,
            device=DEVICE,
            dtype=DTYPE,
        )

        print("result keys:", result.keys())
        print("theta result keys:", result["results"].keys())

        assert_true(result["particle"] == PARTICLE)
        assert_true(result["flavour_name"] == "numu")
        assert_true(len(result["results"]) == len(grid_config.theta_grid_deg))

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_build_phi_E_theta_h_for_particle_serial_with_saving():
    (
        model_config,
        grid_config,
        smoothing_config,
        output_config,
        parallel_config,
        output_dir,
    ) = make_minimal_configs(overwrite=True)

    try:
        result = build_phi_E_theta_h_for_particle(
            theta_grid_deg=grid_config.theta_grid_deg,
            particle=PARTICLE,
            model_config=model_config,
            grid_config=grid_config,
            smoothing_config=smoothing_config,
            output_config=output_config,
            flavour_name="numu",
            save=True,
            show_progress=False,
            device=DEVICE,
            dtype=DTYPE,
        )

        theta0 = list(result["results"].keys())[0]
        theta_result = result["results"][theta0]

        print("theta:", theta0)
        print("output path:", theta_result.get("output_path"))

        assert_true("output_path" in theta_result)
        assert_true(os.path.exists(theta_result["output_path"]))

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


# ============================================================
# RunConfig / all flavours tests
# ============================================================

def test_build_all_flavours_serial_without_saving():
    run_config, output_dir = make_minimal_run_config(
        parallel=False,
        overwrite=True,
    )

    try:
        result = build_all_flavours(
            config=run_config,
            save=False,
            show_progress=False,
            device=DEVICE,
            dtype=DTYPE,
        )

        print("top-level keys:", result.keys())
        print("flavour keys  :", result["results"].keys())

        assert_true("config" in result)
        assert_true("results" in result)
        assert_true("numu" in result["results"])

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_build_all_flavours_serial_with_saving():
    run_config, output_dir = make_minimal_run_config(
        parallel=False,
        overwrite=True,
    )

    try:
        result = build_all_flavours(
            config=run_config,
            save=True,
            show_progress=False,
            device=DEVICE,
            dtype=DTYPE,
        )

        files = [
            f for f in os.listdir(output_dir)
            if f.endswith(".pt") or f.endswith(".pth")
        ]

        print("saved files:", files)

        assert_true(len(files) >= 1)
        assert_true("numu" in result["results"])

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


# ============================================================
# Optional parallel test
# ============================================================

def test_build_phi_E_theta_h_for_particle_parallel_optional():
    (
        model_config,
        grid_config,
        smoothing_config,
        output_config,
        parallel_config,
        output_dir,
    ) = make_minimal_configs(
        parallel=True,
        n_jobs=2,
        overwrite=True,
    )

    try:
        # Use two theta values to actually test task splitting.
        grid_config.theta_grid_deg = torch.tensor(
            [0.0, 30.0],
            dtype=DTYPE,
        ).numpy()

        result = build_phi_E_theta_h_for_particle_parallel(
            theta_grid_deg=grid_config.theta_grid_deg,
            particle=PARTICLE,
            model_config=model_config,
            grid_config=grid_config,
            smoothing_config=smoothing_config,
            output_config=output_config,
            parallel_config=parallel_config,
            flavour_name="numu",
            save=False,
            dtype=DTYPE,
        )

        print("parallel theta keys:", result["results"].keys())

        assert_true(len(result["results"]) == 2)

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


# ============================================================
# Visual diagnostics
# ============================================================

def build_visual_result():
    (
        model_config,
        grid_config,
        smoothing_config,
        output_config,
        parallel_config,
        output_dir,
    ) = make_minimal_configs()

    result = build_theta_result(
        theta_deg=0.0,
        particle=PARTICLE,
        model_config=model_config,
        grid_config=grid_config,
        smoothing_config=smoothing_config,
        save=False,
        flavour_name="numu",
        output_config=output_config,
        device=DEVICE,
        dtype=DTYPE,
    )

    return result, output_dir


def plot_builder_profile_for_selected_energies():
    result, output_dir = build_visual_result()

    try:
        E = result["E_grid_GeV"]
        h = result["h_grid_km"]
        f_Eh = result["f_Eh"]

        selected = [
            0,
            E.numel() // 4,
            E.numel() // 2,
            3 * E.numel() // 4,
            E.numel() - 1,
        ]

        plt.figure(figsize=(9, 6))

        for idx in selected:
            plt.plot(
                h.cpu().numpy(),
                f_Eh[idx].cpu().numpy(),
                lw=2,
                label=f"E={float(E[idx].item()):.3g} GeV",
            )

        plt.xlabel(r"Altitude $h$ [km]")
        plt.ylabel(r"$f(h|E,\theta)$ [km$^{-1}$]")
        plt.title(r"Builder output: normalized production profiles")

        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def plot_builder_phi_Eh_for_selected_energies():
    result, output_dir = build_visual_result()

    try:
        E = result["E_grid_GeV"]
        h = result["h_grid_km"]
        phi_Eh = result["phi_Eh"]

        selected = [
            0,
            E.numel() // 4,
            E.numel() // 2,
            3 * E.numel() // 4,
            E.numel() - 1,
        ]

        plt.figure(figsize=(9, 6))

        for idx in selected:
            plt.plot(
                h.cpu().numpy(),
                phi_Eh[idx].cpu().numpy(),
                lw=2,
                label=f"E={float(E[idx].item()):.3g} GeV",
            )

        plt.xlabel(r"Altitude $h$ [km]")
        plt.ylabel(r"$\Phi(E,h)$")
        plt.title(r"Builder output: height-differential flux")

        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def plot_builder_flux_reconstruction():
    result, output_dir = build_visual_result()

    try:
        E = result["E_grid_GeV"]

        phi_rec = reconstruct_flux_from_profile(
            h_grid_km=result["h_grid_km"],
            phi_Eh=result["phi_Eh"],
            device=DEVICE,
            dtype=DTYPE,
        )

        phi_obs = result["phi_E_obs"]

        rel_err = torch.abs(phi_rec - phi_obs) / torch.clamp(
            torch.abs(phi_obs),
            min=1.0e-30,
        )

        plt.figure(figsize=(9, 6))

        plt.loglog(
            E.cpu().numpy(),
            phi_obs.cpu().numpy(),
            lw=2,
            label=r"$\Phi_{\rm obs}(E)$",
        )

        plt.loglog(
            E.cpu().numpy(),
            phi_rec.cpu().numpy(),
            lw=2,
            ls="--",
            label=r"$\int \Phi(E,h)dh$",
        )

        plt.xlabel(r"Energy $E$ [GeV]")
        plt.ylabel(r"flux")
        plt.title("Builder reconstruction check")

        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(9, 6))

        plt.semilogx(
            E.cpu().numpy(),
            rel_err.cpu().numpy(),
            lw=2,
            marker="o",
        )

        plt.xlabel(r"Energy $E$ [GeV]")
        plt.ylabel("Relative error")
        plt.title(r"Relative error: $\int\Phi(E,h)dh = \Phi_{\rm obs}(E)$")

        plt.grid(True, which="both")
        plt.tight_layout()
        plt.show()

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def run_builder_visual_tests():
    print("\n" + "=" * 80)
    print("BUILDER VISUAL tests")
    print("=" * 80)

    plot_builder_profile_for_selected_energies()
    plot_builder_phi_Eh_for_selected_energies()
    plot_builder_flux_reconstruction()

    print("\nFinished builder visual diagnostics.")


# ============================================================
# Runner
# ============================================================

def run_builder_tests(
    verbose_traceback=False,
    make_plots=True,
    run_parallel_test=False,
):
    tests = [
        test_split_run_config,
        test_validate_particle_name_accepts_valid_string,
        test_validate_particle_name_rejects_non_string,
        test_validate_particle_name_rejects_empty_string,
        test_build_theta_result_without_saving,
        test_build_theta_result_shapes,
        test_build_theta_result_positive_and_finite,
        test_build_theta_result_profile_normalization,
        test_build_theta_result_flux_reconstruction,
        test_build_theta_result_with_saving_creates_file,
        test_build_theta_result_save_true_requires_output_config,
        test_build_phi_E_theta_h_for_particle_serial,
        test_build_phi_E_theta_h_for_particle_serial_with_saving,
        test_build_all_flavours_serial_without_saving,
        test_build_all_flavours_serial_with_saving,
    ]

    if run_parallel_test:
        tests.append(test_build_phi_E_theta_h_for_particle_parallel_optional)

    ok = run_test_suite(
        tests,
        suite_name="BUILDER tests",
        verbose_traceback=verbose_traceback,
    )

    if make_plots:
        run_builder_visual_tests()

    return ok


if __name__ == "__main__":
    run_builder_tests(
        verbose_traceback=True,
        make_plots=True,
        run_parallel_test=False,
    )
    

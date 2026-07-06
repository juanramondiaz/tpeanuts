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
Pipeline test0:

1. Obtain mceq atmospheric density profile and plot it.
2. Solve mceq on h-grid mapped to X(h, theta) for theta = 0, 45.
3. Plot numu energy spectra.
4. Plot flux evolution over h.
5. Compute production profiles f(h|E,theta) and plot them.
6. Compute Phi(E,h,theta), save to ../data/flux/test0, reload and plot all.
"""



import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt

from tpeanuts.util.test_utils import run_test_suite

from tpeanuts.external.mceq.config import (
    MCEqModelConfig,
    GridConfig,
    SmoothingConfig,
)
from tpeanuts.io.io_atmosphere import (
    load_phi_Eh_theta_result,
    OutputConfig,
    save_phi_Eh_theta_result,
)

from tpeanuts.external.mceq.core import init_mceq

from tpeanuts.external.mceq.density import (
    atmospheric_mass_density_profile_from_mceq,
)

from tpeanuts.external.mceq.depth import (
    compute_slant_depth_from_mceq,
)

from tpeanuts.external.mceq.solver import (
    solve_flux_vs_depth_grid,
    interpolate_flux_at_Xobs,
)

from tpeanuts.external.mceq.profiles import (
    production_profiles_all_energies_from_flux_gradient,
)

# ============================================================
# Configuration
# ============================================================

DEVICE = "cpu"
DTYPE = torch.float64

PARTICLE = "numu"
THETA_LIST = [0.0, 45.0]

N_H = 101
H_MIN_KM = 0.0
H_MAX_KM = 80.0

DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_TEST_ROOT = Path(OUTPUT_ROOT / "test")
OUTPUT_DIR = Path(OUTPUT_TEST_ROOT / "mceq" / Path(__file__).stem)

MODEL_CONFIG = MCEqModelConfig(
    interaction_model="SIBYLL23D",
    primary_model="HillasGaisser H3a",
    density_model="CORSIKA",
    info=False,
)

SMOOTHING_CONFIG = SmoothingConfig(
    method="gaussian",
    gaussian_sigma=1.0,
    positive_only=True,
)

OUTPUT_CONFIG = OutputConfig(
    output_dir=OUTPUT_DIR,
    filename="phi_E_theta_h_test0.pt",
    dtype=torch.float64,
    compressed=True,
    overwrite=True,
    save_intermediate=True,
)


h_grid = torch.linspace(
    H_MIN_KM,
    H_MAX_KM,
    N_H,
    device=DEVICE,
    dtype=DTYPE,
)


# ============================================================
# Helpers
# ============================================================

def select_energy_indices(E_grid, n=5):
    return torch.linspace(
        0,
        E_grid.numel() - 1,
        n,
        dtype=torch.long,
    )


def make_grid_config_for_theta(theta_deg):
    mceq = init_mceq(
        theta_deg=theta_deg,
        config=MODEL_CONFIG,
    )

    X_of_h = compute_slant_depth_from_mceq(
        h_km=h_grid,
        theta_deg=theta_deg,
        mceq=mceq,
        config=MODEL_CONFIG,
        device=DEVICE,
        dtype=DTYPE,
    )

    # mceq evolves from small X to large X.
    # X(h) decreases with h, so sort it.
    X_grid_sorted = torch.sort(X_of_h).values

    X_obs = float(X_of_h[0].item())

    return GridConfig(
        theta_grid_deg=torch.tensor([theta_deg], dtype=DTYPE).numpy(),
        X_grid_gcm2=X_grid_sorted.cpu().numpy(),
        h_grid_km=h_grid.cpu().numpy(),
        X_obs_gcm2=X_obs,
    )


# ============================================================
# Step 1
# Atmospheric density profile
# ============================================================

def step1_density_profile():
    print("\n[1] Computing mceq atmospheric density profile...")

    mceq = init_mceq(
        theta_deg=0.0,
        config=MODEL_CONFIG,
    )

    rho = atmospheric_mass_density_profile_from_mceq(
        h_km=h_grid,
        mceq=mceq,
        device=DEVICE,
        dtype=DTYPE,
    )

    plt.figure(figsize=(7, 6))

    plt.semilogx(
        rho.cpu().numpy(),
        h_grid.cpu().numpy(),
        lw=2,
    )

    plt.xlabel(r"$\rho(h)$ [g/cm$^3$]")
    plt.ylabel(r"$h$ [km]")
    plt.title("mceq atmospheric density profile")

    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()

    return rho


# ============================================================
# Step 2 and 3
# Solve mceq on h-derived X grid and plot spectra / flux over h
# ============================================================

def step2_solve_mceq_on_h_grid():
    print("\n[2] Solving mceq for theta = 0 and 45 deg...")

    solutions = {}

    for theta_deg in THETA_LIST:
        print(f"\nSolving theta = {theta_deg} deg")

        grid_config = make_grid_config_for_theta(theta_deg)

        X_grid = torch.as_tensor(
            grid_config.X_grid_gcm2,
            device=DEVICE,
            dtype=DTYPE,
        )

        X_out, E_grid, flux_XE = solve_flux_vs_depth_grid(
            theta_deg=theta_deg,
            particle=PARTICLE,
            X_grid_gcm2=X_grid,
            config=MODEL_CONFIG,
            device=DEVICE,
            dtype=DTYPE,
        )

        # Map the solved Phi(E,X) back to h-grid through X(h).
        mceq = init_mceq(
            theta_deg=theta_deg,
            config=MODEL_CONFIG,
        )

        X_of_h = compute_slant_depth_from_mceq(
            h_km=h_grid,
            theta_deg=theta_deg,
            mceq=mceq,
            config=MODEL_CONFIG,
            device=DEVICE,
            dtype=DTYPE,
        )

        flux_Eh_from_solver = torch.zeros(
            (E_grid.numel(), h_grid.numel()),
            device=DEVICE,
            dtype=DTYPE,
        )

        for ih in range(h_grid.numel()):
            flux_Eh_from_solver[:, ih] = interpolate_flux_at_Xobs(
                X_grid_gcm2=X_out,
                flux_XE=flux_XE,
                X_obs_gcm2=X_of_h[ih],
                log_interp=True,
                device=DEVICE,
                dtype=DTYPE,
            )

        phi_E_obs = interpolate_flux_at_Xobs(
            X_grid_gcm2=X_out,
            flux_XE=flux_XE,
            X_obs_gcm2=grid_config.X_obs_gcm2,
            log_interp=True,
            device=DEVICE,
            dtype=DTYPE,
        )

        solutions[theta_deg] = {
            "grid_config": grid_config,
            "X_grid": X_out,
            "E_grid": E_grid,
            "flux_XE": flux_XE,
            "X_of_h": X_of_h,
            "flux_Eh_from_solver": flux_Eh_from_solver,
            "phi_E_obs": phi_E_obs,
        }

    return solutions


def step2_plot_energy_spectra(solutions):
    print("\n[2 plot] Plotting numu energy spectra...")

    plt.figure(figsize=(8, 6))

    for theta_deg, sol in solutions.items():
        E = sol["E_grid"]
        phi = sol["phi_E_obs"]

        plt.loglog(
            E.cpu().numpy(),
            phi.cpu().numpy(),
            lw=2,
            label=rf"$\theta={theta_deg:.0f}^\circ$",
        )

    plt.xlabel(r"$E$ [GeV]")
    plt.ylabel(r"$\Phi_{\nu_\mu}(E; X_{\rm obs}, \theta)$")
    plt.title(r"mceq $\nu_\mu$ energy spectrum at observer depth")

    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()


def step3_plot_flux_on_h_grid(solutions):
    print("\n[3] Plotting mceq flux mapped onto h-grid...")

    for theta_deg, sol in solutions.items():
        E = sol["E_grid"]
        flux_Eh = sol["flux_Eh_from_solver"]

        idxs = select_energy_indices(E, n=5)

        plt.figure(figsize=(8, 6))

        for idx in idxs:
            plt.plot(
                h_grid.cpu().numpy(),
                flux_Eh[idx].cpu().numpy(),
                lw=2,
                label=f"E={float(E[idx].item()):.3g} GeV",
            )

        plt.xlabel(r"$h$ [km]")
        plt.ylabel(r"$\Phi_{\nu_\mu}(E,h,\theta)$ from mceq depth grid")
        plt.title(rf"mceq flux on h-grid, $\theta={theta_deg:.0f}^\circ$")

        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


# ============================================================
# Step 4 and 5
# Production profile and differential flux Phi(E,h,theta)
# ============================================================

def step4_5_compute_profiles_and_phi():
    print("\n[4-5] Computing production profiles and Phi(E,h,theta)...")

    profile_results = {}

    for theta_deg in THETA_LIST:
        print(f"\nComputing production profile for theta = {theta_deg} deg")

        grid_config = make_grid_config_for_theta(theta_deg)

        result = production_profiles_all_energies_from_flux_gradient(
            theta_deg=theta_deg,
            particle=PARTICLE,
            model_config=MODEL_CONFIG,
            grid_config=grid_config,
            smoothing_config=SMOOTHING_CONFIG,
            device=DEVICE,
            dtype=DTYPE,
        )

        result["particle"] = PARTICLE
        result["flavour_name"] = PARTICLE

        profile_results[theta_deg] = result

    return profile_results


def step4_plot_production_profiles(profile_results):
    print("\n[4 plot] Plotting production profiles...")

    for theta_deg, result in profile_results.items():
        E = result["E_grid_GeV"]
        h = result["h_grid_km"]
        f_Eh = result["f_Eh"]

        idxs = select_energy_indices(E, n=5)

        plt.figure(figsize=(8, 6))

        for idx in idxs:
            plt.plot(
                h.cpu().numpy(),
                f_Eh[idx].cpu().numpy(),
                lw=2,
                label=f"E={float(E[idx].item()):.3g} GeV",
            )

        plt.xlabel(r"$h$ [km]")
        plt.ylabel(r"$f(h|E,\theta)$ [km$^{-1}$]")
        plt.title(rf"Production profile, $\theta={theta_deg:.0f}^\circ$")

        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


def step5_plot_phi_Eh(profile_results):
    print("\n[5 plot] Plotting differential flux Phi(E,h,theta)...")

    for theta_deg, result in profile_results.items():
        E = result["E_grid_GeV"]
        h = result["h_grid_km"]
        phi_Eh = result["phi_Eh"]

        idxs = select_energy_indices(E, n=5)

        plt.figure(figsize=(8, 6))

        for idx in idxs:
            plt.plot(
                h.cpu().numpy(),
                phi_Eh[idx].cpu().numpy(),
                lw=2,
                label=f"E={float(E[idx].item()):.3g} GeV",
            )

        plt.xlabel(r"$h$ [km]")
        plt.ylabel(r"$\Phi_{\nu_\mu}(E,h,\theta)$")
        plt.title(rf"Height-differential flux, $\theta={theta_deg:.0f}^\circ$")

        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


# ============================================================
# Step 6
# Save, reload, and grid plot
# ============================================================

def step6_save_and_reload(profile_results):
    print("\n[6] Saving and reloading results...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    saved_paths = {}
    loaded_results = {}

    for theta_deg, result in profile_results.items():
        path = save_phi_Eh_theta_result(
            result=result,
            output_config=OUTPUT_CONFIG,
            flavour_name=PARTICLE,
        )

        print(f"Saved theta={theta_deg} deg -> {path}")

        loaded = load_phi_Eh_theta_result(
            input_path=path,
            map_location="cpu",
            dtype=DTYPE,
            device=DEVICE,
        )

        saved_paths[theta_deg] = path
        loaded_results[theta_deg] = loaded

    return saved_paths, loaded_results


def step6_plot_loaded_grid(loaded_results):
    print("\n[6 plot] Plotting loaded content in axes grid...")

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(18, 10),
    )

    for row, theta_deg in enumerate(THETA_LIST):
        result = loaded_results[theta_deg]

        E = result["E_grid_GeV"]
        h = result["h_grid_km"]
        X = result["X_of_h_gcm2"]
        f_Eh = result["f_Eh"]
        phi_Eh = result["phi_Eh"]
        phi_E_obs = result["phi_E_obs"]

        idxs = select_energy_indices(E, n=4)

        ax = axes[row, 0]
        ax.plot(
            X.cpu().numpy(),
            h.cpu().numpy(),
            lw=2,
        )
        ax.set_xlabel(r"$X(h,\theta)$ [g/cm$^2$]")
        ax.set_ylabel(r"$h$ [km]")
        ax.set_title(rf"$X(h,\theta)$, $\theta={theta_deg:.0f}^\circ$")
        ax.grid(True)

        ax = axes[row, 1]
        for idx in idxs:
            ax.plot(
                h.cpu().numpy(),
                f_Eh[idx].cpu().numpy(),
                lw=2,
                label=f"E={float(E[idx].item()):.2g}"
            )
        ax.set_xlabel(r"$h$ [km]")
        ax.set_ylabel(r"$f(h|E,\theta)$")
        ax.set_title(rf"Loaded production profile, $\theta={theta_deg:.0f}^\circ$")
        ax.grid(True)
        ax.legend(fontsize=8)

        ax = axes[row, 2]
        for idx in idxs:
            ax.plot(
                h.cpu().numpy(),
                phi_Eh[idx].cpu().numpy(),
                lw=2,
                label=f"E={float(E[idx].item()):.2g}"
            )
        ax.set_xlabel(r"$h$ [km]")
        ax.set_ylabel(r"$\Phi(E,h,\theta)$")
        ax.set_title(rf"Loaded differential flux, $\theta={theta_deg:.0f}^\circ$")
        ax.grid(True)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))

    for theta_deg in THETA_LIST:
        result = loaded_results[theta_deg]

        E = result["E_grid_GeV"]
        phi_E_obs = result["phi_E_obs"]

        plt.loglog(
            E.cpu().numpy(),
            phi_E_obs.cpu().numpy(),
            lw=2,
            label=rf"$\theta={theta_deg:.0f}^\circ$",
        )

    plt.xlabel(r"$E$ [GeV]")
    plt.ylabel(r"$\Phi_{\nu_\mu}(E;X_{\rm obs},\theta)$")
    plt.title("Reloaded observer spectra")

    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Full pipeline
# ============================================================

def run_pipeline_test0():
    print("\n" + "=" * 80)
    print("PIPELINE TEST0: mceq atmospheric numu flux")
    print("=" * 80)

    rho = step1_density_profile()

    solutions = step2_solve_mceq_on_h_grid()
    step2_plot_energy_spectra(solutions)

    step3_plot_flux_on_h_grid(solutions)

    profile_results = step4_5_compute_profiles_and_phi()
    step4_plot_production_profiles(profile_results)
    step5_plot_phi_Eh(profile_results)

    saved_paths, loaded_results = step6_save_and_reload(profile_results)
    step6_plot_loaded_grid(loaded_results)

    print("\nSaved files:")
    for theta_deg, path in saved_paths.items():
        print(f"theta={theta_deg:.0f} deg -> {path}")

    print("\nPipeline finished.")

    return {
        "rho": rho,
        "solutions": solutions,
        "profile_results": profile_results,
        "saved_paths": saved_paths,
        "loaded_results": loaded_results,
    }


def test_pipeline_test0():
    run_pipeline_test0()


def run_mceq_pipeline_tests(verbose_traceback=False):
    return run_test_suite(
        [test_pipeline_test0],
        suite_name="mceq PIPELINE tests",
        verbose_traceback=verbose_traceback,
    )


if __name__ == "__main__":
    run_mceq_pipeline_tests(verbose_traceback=True)

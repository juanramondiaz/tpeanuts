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
Spyder-compatible tests and visual diagnostics for solver.py.

No pytest required.
"""



import torch
import matplotlib.pyplot as plt

from tpeanuts.external.mceq.config import (
    MCEqModelConfig,
    GridConfig,
)

from tpeanuts.external.mceq.solver import (
    solve_flux_vs_depth_grid,
    get_mceq_flux_at_Xobs,
    interpolate_flux_at_Xobs,
)

from tpeanuts.util.test_utils import (
    assert_true,
    assert_close,
    assert_raises,
    run_test_suite,
)


DEVICE = "cpu"
DTYPE = torch.float64

PARTICLE = "numu"

MODEL_CONFIG = MCEqModelConfig(
    interaction_model="SIBYLL23D",
    primary_model="HillasGaisser H3a",
    density_model="CORSIKA",
    info=False,
)

GRID_CONFIG = GridConfig(
    theta_grid_deg=torch.tensor([0.0, 30.0, 60.0], dtype=DTYPE).numpy(),
    X_grid_gcm2=torch.linspace(10.0, 1030.0, 12, dtype=DTYPE).numpy(),
    h_grid_km=torch.linspace(0.0, 80.0, 60, dtype=DTYPE).numpy(),
    X_obs_gcm2=1030.0,
)


# ============================================================
# Synthetic interpolation tests
# ============================================================

def make_synthetic_flux():
    X = torch.linspace(
        10.0,
        1030.0,
        50,
        device=DEVICE,
        dtype=DTYPE,
    )

    E = torch.tensor(
        [1.0, 10.0, 100.0],
        device=DEVICE,
        dtype=DTYPE,
    )

    flux_XE = (1.0 - torch.exp(-X[:, None] / 250.0)) * E[None, :] ** (-2.0)

    return X, E, flux_XE


def test_interpolate_flux_at_Xobs_shape():
    X, E, flux_XE = make_synthetic_flux()

    phi = interpolate_flux_at_Xobs(
        X_grid_gcm2=X,
        flux_XE=flux_XE,
        X_obs_gcm2=500.0,
        log_interp=True,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("flux_XE shape:", flux_XE.shape)
    print("phi shape    :", phi.shape)

    assert_true(phi.shape == E.shape)


def test_interpolate_flux_at_grid_point_matches_value():
    X, E, flux_XE = make_synthetic_flux()

    idx = 20
    X_obs = X[idx]

    phi = interpolate_flux_at_Xobs(
        X_grid_gcm2=X,
        flux_XE=flux_XE,
        X_obs_gcm2=X_obs,
        log_interp=False,
        device=DEVICE,
        dtype=DTYPE,
    )

    max_diff = torch.max(torch.abs(phi - flux_XE[idx])).item()

    print("X_obs:", float(X_obs.item()))
    print("max difference:", max_diff)

    assert_close(max_diff, 0.0, atol=1.0e-14, rtol=0.0)


def test_interpolate_flux_at_Xobs_rejects_outside_range():
    X, E, flux_XE = make_synthetic_flux()

    assert_raises(
        ValueError,
        interpolate_flux_at_Xobs,
        X,
        flux_XE,
        2000.0,
        device=DEVICE,
        dtype=DTYPE,
    )


def test_interpolate_flux_at_Xobs_rejects_bad_flux_shape():
    X, E, flux_XE = make_synthetic_flux()

    bad_flux = torch.ones(
        X.numel(),
        device=DEVICE,
        dtype=DTYPE,
    )

    assert_raises(
        ValueError,
        interpolate_flux_at_Xobs,
        X,
        bad_flux,
        500.0,
        device=DEVICE,
        dtype=DTYPE,
    )


def test_interpolate_flux_at_Xobs_rejects_non_monotonic_X():
    X, E, flux_XE = make_synthetic_flux()

    bad_X = X.clone()
    bad_X[10] = bad_X[5]

    assert_raises(
        ValueError,
        interpolate_flux_at_Xobs,
        bad_X,
        flux_XE,
        500.0,
        device=DEVICE,
        dtype=DTYPE,
    )


# ============================================================
# mceq solver tests
# ============================================================

def test_get_mceq_flux_at_Xobs_shapes():
    E_grid, phi_E = get_mceq_flux_at_Xobs(
        theta_deg=0.0,
        particle=PARTICLE,
        X_obs_gcm2=1030.0,
        config=MODEL_CONFIG,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("E_grid shape:", E_grid.shape)
    print("phi_E shape :", phi_E.shape)
    print("E min/max   :", float(E_grid.min().item()), float(E_grid.max().item()))
    print("phi min/max :", float(phi_E.min().item()), float(phi_E.max().item()))

    assert_true(E_grid.ndim == 1)
    assert_true(phi_E.ndim == 1)
    assert_true(E_grid.shape == phi_E.shape)


def test_get_mceq_flux_at_Xobs_positive_and_finite():
    E_grid, phi_E = get_mceq_flux_at_Xobs(
        theta_deg=0.0,
        particle=PARTICLE,
        X_obs_gcm2=1030.0,
        config=MODEL_CONFIG,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("has nan:", bool(torch.isnan(phi_E).any().item()))
    print("has inf:", bool(torch.isinf(phi_E).any().item()))
    print("min phi:", float(phi_E.min().item()))

    assert_true(not torch.isnan(phi_E).any().item())
    assert_true(not torch.isinf(phi_E).any().item())
    assert_true(torch.all(phi_E >= 0.0).item())


def test_solve_flux_vs_depth_grid_shapes():
    X_grid = torch.linspace(
        10.0,
        1030.0,
        8,
        device=DEVICE,
        dtype=DTYPE,
    )

    X_out, E_grid, flux_XE = solve_flux_vs_depth_grid(
        theta_deg=0.0,
        particle=PARTICLE,
        X_grid_gcm2=X_grid,
        config=MODEL_CONFIG,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("X_out shape :", X_out.shape)
    print("E_grid shape:", E_grid.shape)
    print("flux shape  :", flux_XE.shape)

    assert_true(X_out.shape == X_grid.shape)
    assert_true(E_grid.ndim == 1)
    assert_true(flux_XE.shape == (X_grid.numel(), E_grid.numel()))


def test_solve_flux_vs_depth_grid_positive_and_finite():
    X_grid = torch.linspace(
        10.0,
        1030.0,
        8,
        device=DEVICE,
        dtype=DTYPE,
    )

    X_out, E_grid, flux_XE = solve_flux_vs_depth_grid(
        theta_deg=0.0,
        particle=PARTICLE,
        X_grid_gcm2=X_grid,
        config=MODEL_CONFIG,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("flux min:", float(flux_XE.min().item()))
    print("flux max:", float(flux_XE.max().item()))

    assert_true(not torch.isnan(flux_XE).any().item())
    assert_true(not torch.isinf(flux_XE).any().item())
    assert_true(torch.all(flux_XE >= 0.0).item())


def test_solve_flux_vs_depth_grid_rejects_non_monotonic_X():
    X_grid = torch.tensor(
        [10.0, 100.0, 50.0, 1030.0],
        device=DEVICE,
        dtype=DTYPE,
    )

    assert_raises(
        ValueError,
        solve_flux_vs_depth_grid,
        0.0,
        PARTICLE,
        X_grid,
        config=MODEL_CONFIG,
        device=DEVICE,
        dtype=DTYPE,
    )


def test_interpolated_Xobs_matches_direct_solution_reasonably():
    X_grid = torch.linspace(
        10.0,
        1030.0,
        12,
        device=DEVICE,
        dtype=DTYPE,
    )

    X_out, E_grid_1, flux_XE = solve_flux_vs_depth_grid(
        theta_deg=0.0,
        particle=PARTICLE,
        X_grid_gcm2=X_grid,
        config=MODEL_CONFIG,
        device=DEVICE,
        dtype=DTYPE,
    )

    phi_interp = interpolate_flux_at_Xobs(
        X_grid_gcm2=X_out,
        flux_XE=flux_XE,
        X_obs_gcm2=1030.0,
        log_interp=True,
        device=DEVICE,
        dtype=DTYPE,
    )

    E_grid_2, phi_direct = get_mceq_flux_at_Xobs(
        theta_deg=0.0,
        particle=PARTICLE,
        X_obs_gcm2=1030.0,
        config=MODEL_CONFIG,
        device=DEVICE,
        dtype=DTYPE,
    )

    rel_err = torch.abs(phi_interp - phi_direct) / torch.clamp(
        torch.abs(phi_direct),
        min=1.0e-30,
    )

    max_rel = float(torch.max(rel_err).item())

    print("max relative difference:", max_rel)

    assert_true(E_grid_1.shape == E_grid_2.shape)
    assert_true(max_rel < 5.0e-1)


# ============================================================
# Visual diagnostics
# ============================================================

def plot_flux_vs_depth_for_selected_energies(theta_deg=0.0):
    X_grid = torch.linspace(
        10.0,
        1030.0,
        20,
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

    selected_indices = [
        0,
        E_grid.numel() // 4,
        E_grid.numel() // 2,
        3 * E_grid.numel() // 4,
        E_grid.numel() - 1,
    ]

    plt.figure(figsize=(9, 6))

    for idx in selected_indices:
        E_val = float(E_grid[idx].item())

        plt.plot(
            X_out.cpu().numpy(),
            flux_XE[:, idx].cpu().numpy(),
            lw=2,
            marker="o",
            label=f"E={E_val:.3g} GeV",
        )

    plt.xlabel(r"Atmospheric depth $X$ [g/cm$^2$]")
    plt.ylabel(r"$\Phi(E,X,\theta)$")
    plt.title(rf"mceq flux vs depth, $\theta={theta_deg:.0f}^\circ$, {PARTICLE}")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_observer_flux_spectrum(theta_list=(0.0, 30.0, 60.0)):
    plt.figure(figsize=(9, 6))

    for theta_deg in theta_list:
        E_grid, phi_E = get_mceq_flux_at_Xobs(
            theta_deg=theta_deg,
            particle=PARTICLE,
            X_obs_gcm2=1030.0,
            config=MODEL_CONFIG,
            device=DEVICE,
            dtype=DTYPE,
        )

        plt.loglog(
            E_grid.cpu().numpy(),
            phi_E.cpu().numpy(),
            lw=2,
            label=rf"$\theta={theta_deg:.0f}^\circ$",
        )

    plt.xlabel(r"Energy $E$ [GeV]")
    plt.ylabel(r"$\Phi(E;\,X_{\rm obs},\theta)$")
    plt.title(rf"Observer-depth mceq spectrum, {PARTICLE}")

    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_direct_vs_interpolated_Xobs(theta_deg=0.0):
    X_grid = torch.linspace(
        10.0,
        1030.0,
        12,
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

    phi_interp = interpolate_flux_at_Xobs(
        X_grid_gcm2=X_out,
        flux_XE=flux_XE,
        X_obs_gcm2=1030.0,
        log_interp=True,
        device=DEVICE,
        dtype=DTYPE,
    )

    E_direct, phi_direct = get_mceq_flux_at_Xobs(
        theta_deg=theta_deg,
        particle=PARTICLE,
        X_obs_gcm2=1030.0,
        config=MODEL_CONFIG,
        device=DEVICE,
        dtype=DTYPE,
    )

    rel_err = torch.abs(phi_interp - phi_direct) / torch.clamp(
        torch.abs(phi_direct),
        min=1.0e-30,
    )

    plt.figure(figsize=(9, 6))

    plt.loglog(
        E_grid.cpu().numpy(),
        phi_direct.cpu().numpy(),
        lw=2,
        label="direct Xobs solution",
    )

    plt.loglog(
        E_grid.cpu().numpy(),
        phi_interp.cpu().numpy(),
        lw=2,
        ls="--",
        label="interpolated from depth grid",
    )

    plt.xlabel(r"Energy $E$ [GeV]")
    plt.ylabel(r"$\Phi(E)$")
    plt.title(rf"Direct vs interpolated observer flux, $\theta={theta_deg:.0f}^\circ$")

    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 6))

    plt.semilogx(
        E_grid.cpu().numpy(),
        rel_err.cpu().numpy(),
        lw=2,
        marker="o",
    )

    plt.xlabel(r"Energy $E$ [GeV]")
    plt.ylabel("Relative difference")
    plt.title("Relative difference: direct vs interpolated")

    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()


def run_solver_visual_tests():
    print("\n" + "=" * 80)
    print("SOLVER VISUAL tests")
    print("=" * 80)

    plot_flux_vs_depth_for_selected_energies(theta_deg=0.0)
    plot_observer_flux_spectrum(theta_list=(0.0, 30.0, 60.0))
    plot_direct_vs_interpolated_Xobs(theta_deg=0.0)

    print("\nFinished solver visual diagnostics.")


# ============================================================
# Runner
# ============================================================

def run_solver_tests(
    verbose_traceback=False,
    make_plots=True,
):
    tests = [
        test_interpolate_flux_at_Xobs_shape,
        test_interpolate_flux_at_grid_point_matches_value,
        test_interpolate_flux_at_Xobs_rejects_outside_range,
        test_interpolate_flux_at_Xobs_rejects_bad_flux_shape,
        test_interpolate_flux_at_Xobs_rejects_non_monotonic_X,
        test_get_mceq_flux_at_Xobs_shapes,
        test_get_mceq_flux_at_Xobs_positive_and_finite,
        test_solve_flux_vs_depth_grid_shapes,
        test_solve_flux_vs_depth_grid_positive_and_finite,
        test_solve_flux_vs_depth_grid_rejects_non_monotonic_X,
        test_interpolated_Xobs_matches_direct_solution_reasonably,
    ]

    ok = run_test_suite(
        tests,
        suite_name="SOLVER tests",
        verbose_traceback=verbose_traceback,
    )

    if make_plots:
        run_solver_visual_tests()

    return ok


if __name__ == "__main__":
    run_solver_tests(
        verbose_traceback=True,
        make_plots=True,
    )
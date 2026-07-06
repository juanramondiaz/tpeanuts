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
Spyder-compatible synthetic tests and visual diagnostics for smoothing.py.

No pytest required.
"""



import torch
import matplotlib.pyplot as plt

from tpeanuts.external.mceq.smoothing import (
    gaussian_kernel1d,
    smooth_flux_gaussian,
    smooth_flux_log_moving_average,
    smooth_flux_in_depth,
    compute_depth_derivative,
    smooth_and_differentiate_flux,
)

from tpeanuts.external.mceq.config import SmoothingConfig

from tpeanuts.util.test_utils import (
    assert_true,
    assert_close,
    assert_raises,
    run_test_suite,
)


# ============================================================
# Configuration
# ============================================================

DEVICE = "cpu"
DTYPE = torch.float64

N_X = 300
N_E = 5

X_MIN = 1.0
X_MAX = 1030.0

ENERGIES = torch.tensor(
    [0.5, 1.0, 5.0, 10.0, 50.0],
    device=DEVICE,
    dtype=DTYPE,
)


# ============================================================
# Synthetic flux
# ============================================================

def make_synthetic_flux(
    n_X=N_X,
    energies=ENERGIES,
    lambda_gcm2=250.0,
    noise_level=0.04,
    seed=1234,
):
    X = torch.linspace(
        X_MIN,
        X_MAX,
        n_X,
        device=DEVICE,
        dtype=DTYPE,
    )

    E = energies.to(device=DEVICE, dtype=DTYPE)

    base_X = 1.0 - torch.exp(-X[:, None] / lambda_gcm2)
    base_E = E[None, :] ** (-2.0)

    flux_clean = base_X * base_E

    # Deterministic pseudo-noise, useful for plots and tests.
    noise_X = torch.sin(0.09 * X[:, None]) + 0.5 * torch.sin(0.31 * X[:, None])
    noise_E = torch.linspace(0.5, 1.5, E.numel(), device=DEVICE, dtype=DTYPE)[None, :]

    flux_noisy = flux_clean * (1.0 + noise_level * noise_X * noise_E)
    flux_noisy = torch.clamp(flux_noisy, min=0.0)

    return X, E, flux_clean, flux_noisy


def analytic_dphi_dX(
    X,
    E,
    lambda_gcm2=250.0,
):
    return (
        torch.exp(-X[:, None] / lambda_gcm2)
        * E[None, :] ** (-2.0)
        / lambda_gcm2
    )


# ============================================================
# Unit-like tests
# ============================================================

def test_gaussian_kernel_is_normalized():
    kernel = gaussian_kernel1d(
        sigma=2.0,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("kernel shape:", kernel.shape)
    print("kernel sum:", float(kernel.sum().item()))

    assert_close(
        float(kernel.sum().item()),
        1.0,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_gaussian_kernel_rejects_non_positive_sigma():
    assert_raises(
        ValueError,
        gaussian_kernel1d,
        0.0,
        device=DEVICE,
        dtype=DTYPE,
    )


def test_smooth_flux_gaussian_preserves_shape():
    X, E, flux_clean, flux_noisy = make_synthetic_flux()

    flux_smooth = smooth_flux_gaussian(
        flux_XE=flux_noisy,
        sigma=2.0,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("flux_noisy shape :", flux_noisy.shape)
    print("flux_smooth shape:", flux_smooth.shape)

    assert_true(flux_smooth.shape == flux_noisy.shape)


def test_smooth_log_moving_average_preserves_shape():
    X, E, flux_clean, flux_noisy = make_synthetic_flux()

    flux_smooth = smooth_flux_log_moving_average(
        flux_XE=flux_noisy,
        window=9,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("flux_noisy shape :", flux_noisy.shape)
    print("flux_smooth shape:", flux_smooth.shape)

    assert_true(flux_smooth.shape == flux_noisy.shape)


def test_smooth_flux_none_returns_same_values():
    X, E, flux_clean, flux_noisy = make_synthetic_flux()

    flux_out = smooth_flux_in_depth(
        X_grid_gcm2=X,
        flux_XE=flux_noisy,
        method="none",
        device=DEVICE,
        dtype=DTYPE,
    )

    max_diff = float(torch.max(torch.abs(flux_out - flux_noisy)).item())

    print("max difference:", max_diff)

    assert_close(max_diff, 0.0, atol=1.0e-15, rtol=0.0)


def test_smooth_flux_gaussian_wrapper():
    X, E, flux_clean, flux_noisy = make_synthetic_flux()

    config = SmoothingConfig(
        method="gaussian",
        gaussian_sigma=2.0,
    )

    flux_smooth = smooth_flux_in_depth(
        X_grid_gcm2=X,
        flux_XE=flux_noisy,
        config=config,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("flux_smooth shape:", flux_smooth.shape)

    assert_true(flux_smooth.shape == flux_noisy.shape)


def test_smooth_flux_spline_alias_wrapper():
    X, E, flux_clean, flux_noisy = make_synthetic_flux()

    config = SmoothingConfig(
        method="spline",
        smoothing=1.0e-4,
    )

    flux_smooth = smooth_flux_in_depth(
        X_grid_gcm2=X,
        flux_XE=flux_noisy,
        config=config,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("flux_smooth shape:", flux_smooth.shape)

    assert_true(flux_smooth.shape == flux_noisy.shape)


def test_compute_depth_derivative_shape():
    X, E, flux_clean, flux_noisy = make_synthetic_flux()

    dPhi_dX = compute_depth_derivative(
        X_grid_gcm2=X,
        flux_XE=flux_clean,
        positive_only=True,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("dPhi_dX shape:", dPhi_dX.shape)

    assert_true(dPhi_dX.shape == flux_clean.shape)


def test_compute_depth_derivative_positive_for_clean_flux():
    X, E, flux_clean, flux_noisy = make_synthetic_flux()

    dPhi_dX = compute_depth_derivative(
        X_grid_gcm2=X,
        flux_XE=flux_clean,
        positive_only=True,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("min dPhi_dX:", float(dPhi_dX.min().item()))
    print("max dPhi_dX:", float(dPhi_dX.max().item()))

    assert_true(torch.all(dPhi_dX >= 0.0).item())


def test_compute_depth_derivative_matches_analytic_clean_flux():
    X, E, flux_clean, flux_noisy = make_synthetic_flux(
        n_X=2000,
        noise_level=0.0,
    )

    dPhi_num = compute_depth_derivative(
        X_grid_gcm2=X,
        flux_XE=flux_clean,
        positive_only=False,
        device=DEVICE,
        dtype=DTYPE,
    )

    dPhi_exact = analytic_dphi_dX(X, E)

    rel_err = torch.abs(dPhi_num - dPhi_exact) / torch.clamp(
        torch.abs(dPhi_exact),
        min=1.0e-30,
    )

    max_rel_err = float(torch.max(rel_err[5:-5]).item())

    print("max relative error derivative:", max_rel_err)

    assert_true(max_rel_err < 2.0e-3)


def test_positive_only_clips_negative_derivative():
    X = torch.linspace(
        1.0,
        100.0,
        100,
        device=DEVICE,
        dtype=DTYPE,
    )

    flux = -X[:, None] * torch.ones(
        (1, 3),
        device=DEVICE,
        dtype=DTYPE,
    )

    dPhi_dX = compute_depth_derivative(
        X_grid_gcm2=X,
        flux_XE=flux,
        positive_only=True,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("min clipped derivative:", float(dPhi_dX.min().item()))
    print("max clipped derivative:", float(dPhi_dX.max().item()))

    assert_true(torch.all(dPhi_dX == 0.0).item())


def test_smooth_and_differentiate_flux_shapes():
    X, E, flux_clean, flux_noisy = make_synthetic_flux()

    config = SmoothingConfig(
        method="gaussian",
        gaussian_sigma=2.0,
        positive_only=True,
    )

    flux_smooth, dPhi_dX = smooth_and_differentiate_flux(
        X_grid_gcm2=X,
        flux_XE=flux_noisy,
        config=config,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("flux_smooth shape:", flux_smooth.shape)
    print("dPhi_dX shape    :", dPhi_dX.shape)

    assert_true(flux_smooth.shape == flux_noisy.shape)
    assert_true(dPhi_dX.shape == flux_noisy.shape)


def test_invalid_smoothing_method_raises():
    X, E, flux_clean, flux_noisy = make_synthetic_flux()

    assert_raises(
        ValueError,
        smooth_flux_in_depth,
        X,
        flux_noisy,
        method="invalid",
        device=DEVICE,
        dtype=DTYPE,
    )


def test_invalid_flux_shape_raises():
    X = torch.linspace(
        1.0,
        100.0,
        100,
        device=DEVICE,
        dtype=DTYPE,
    )

    bad_flux = torch.ones(
        100,
        device=DEVICE,
        dtype=DTYPE,
    )

    assert_raises(
        ValueError,
        compute_depth_derivative,
        X,
        bad_flux,
        device=DEVICE,
        dtype=DTYPE,
    )


# ============================================================
# Visual tests
# ============================================================

def plot_smoothing_one_energy(i_E=2):
    X, E, flux_clean, flux_noisy = make_synthetic_flux(
        noise_level=0.06,
    )

    flux_gaussian = smooth_flux_in_depth(
        X_grid_gcm2=X,
        flux_XE=flux_noisy,
        method="gaussian",
        gaussian_sigma=3.0,
        device=DEVICE,
        dtype=DTYPE,
    )

    flux_logma = smooth_flux_in_depth(
        X_grid_gcm2=X,
        flux_XE=flux_noisy,
        method="log_moving_average",
        smoothing=5.0e-4,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("\nPlot smoothing for E =", float(E[i_E].item()), "GeV")

    plt.figure(figsize=(9, 6))

    plt.plot(
        X.cpu().numpy(),
        flux_clean[:, i_E].cpu().numpy(),
        lw=2,
        label="clean",
    )

    plt.plot(
        X.cpu().numpy(),
        flux_noisy[:, i_E].cpu().numpy(),
        lw=1,
        alpha=0.7,
        label="noisy",
    )

    plt.plot(
        X.cpu().numpy(),
        flux_gaussian[:, i_E].cpu().numpy(),
        lw=2,
        ls="--",
        label="gaussian",
    )

    plt.plot(
        X.cpu().numpy(),
        flux_logma[:, i_E].cpu().numpy(),
        lw=2,
        ls=":",
        label="log moving average",
    )

    plt.xlabel(r"Atmospheric depth $X$ [g/cm$^2$]")
    plt.ylabel(r"$\Phi(E,X)$")

    plt.title("flux smoothing along depth")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_derivative_one_energy(i_E=2):
    X, E, flux_clean, flux_noisy = make_synthetic_flux(
        noise_level=0.06,
    )

    flux_smooth = smooth_flux_in_depth(
        X_grid_gcm2=X,
        flux_XE=flux_noisy,
        method="gaussian",
        gaussian_sigma=3.0,
        device=DEVICE,
        dtype=DTYPE,
    )

    dPhi_noisy = compute_depth_derivative(
        X_grid_gcm2=X,
        flux_XE=flux_noisy,
        positive_only=False,
        device=DEVICE,
        dtype=DTYPE,
    )

    dPhi_smooth = compute_depth_derivative(
        X_grid_gcm2=X,
        flux_XE=flux_smooth,
        positive_only=False,
        device=DEVICE,
        dtype=DTYPE,
    )

    dPhi_exact = analytic_dphi_dX(X, E)

    print("\nPlot derivative for E =", float(E[i_E].item()), "GeV")

    plt.figure(figsize=(9, 6))

    plt.plot(
        X.cpu().numpy(),
        dPhi_exact[:, i_E].cpu().numpy(),
        lw=2,
        label="analytic clean derivative",
    )

    plt.plot(
        X.cpu().numpy(),
        dPhi_noisy[:, i_E].cpu().numpy(),
        lw=1,
        alpha=0.6,
        label="noisy numerical derivative",
    )

    plt.plot(
        X.cpu().numpy(),
        dPhi_smooth[:, i_E].cpu().numpy(),
        lw=2,
        ls="--",
        label="smoothed numerical derivative",
    )

    plt.xlabel(r"Atmospheric depth $X$ [g/cm$^2$]")
    plt.ylabel(r"$d\Phi/dX$")

    plt.title("Depth derivative before/after smoothing")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_positive_only_effect(i_E=2):
    X, E, flux_clean, flux_noisy = make_synthetic_flux(
        noise_level=0.12,
    )

    dPhi_raw = compute_depth_derivative(
        X_grid_gcm2=X,
        flux_XE=flux_noisy,
        positive_only=False,
        device=DEVICE,
        dtype=DTYPE,
    )

    dPhi_positive = compute_depth_derivative(
        X_grid_gcm2=X,
        flux_XE=flux_noisy,
        positive_only=True,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("\nPositive-only effect for E =", float(E[i_E].item()), "GeV")
    print("raw min:", float(dPhi_raw[:, i_E].min().item()))
    print("positive-only min:", float(dPhi_positive[:, i_E].min().item()))

    plt.figure(figsize=(9, 6))

    plt.plot(
        X.cpu().numpy(),
        dPhi_raw[:, i_E].cpu().numpy(),
        lw=1.5,
        label="raw derivative",
    )

    plt.plot(
        X.cpu().numpy(),
        dPhi_positive[:, i_E].cpu().numpy(),
        lw=2,
        ls="--",
        label="positive only",
    )

    plt.axhline(
        0.0,
        lw=1,
        ls=":",
    )

    plt.xlabel(r"Atmospheric depth $X$ [g/cm$^2$]")
    plt.ylabel(r"$d\Phi/dX$")

    plt.title("Effect of positive_only=True")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_all_energies_smoothing_ratio():
    X, E, flux_clean, flux_noisy = make_synthetic_flux(
        noise_level=0.06,
    )

    flux_smooth = smooth_flux_in_depth(
        X_grid_gcm2=X,
        flux_XE=flux_noisy,
        method="gaussian",
        gaussian_sigma=3.0,
        device=DEVICE,
        dtype=DTYPE,
    )

    ratio = flux_smooth / torch.clamp(flux_clean, min=1.0e-30)

    plt.figure(figsize=(9, 6))

    for i in range(E.numel()):
        plt.plot(
            X.cpu().numpy(),
            ratio[:, i].cpu().numpy(),
            lw=2,
            label=f"E={float(E[i].item()):g} GeV",
        )

    plt.xlabel(r"Atmospheric depth $X$ [g/cm$^2$]")
    plt.ylabel(r"$\Phi_{\rm smooth}/\Phi_{\rm clean}$")

    plt.title("Smoothing ratio for all energies")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_smoothing_visual_tests():
    print("\n" + "=" * 80)
    print("SMOOTHING VISUAL tests")
    print("=" * 80)

    plot_smoothing_one_energy(i_E=2)
    plot_derivative_one_energy(i_E=2)
    plot_positive_only_effect(i_E=2)
    plot_all_energies_smoothing_ratio()

    print("\nFinished smoothing visual diagnostics.")


# ============================================================
# Runner
# ============================================================

def run_smoothing_tests(verbose_traceback=False, make_plots=True):
    tests = [
        test_gaussian_kernel_is_normalized,
        test_gaussian_kernel_rejects_non_positive_sigma,
        test_smooth_flux_gaussian_preserves_shape,
        test_smooth_log_moving_average_preserves_shape,
        test_smooth_flux_none_returns_same_values,
        test_smooth_flux_gaussian_wrapper,
        test_smooth_flux_spline_alias_wrapper,
        test_compute_depth_derivative_shape,
        test_compute_depth_derivative_positive_for_clean_flux,
        test_compute_depth_derivative_matches_analytic_clean_flux,
        test_positive_only_clips_negative_derivative,
        test_smooth_and_differentiate_flux_shapes,
        test_invalid_smoothing_method_raises,
        test_invalid_flux_shape_raises,
    ]

    ok = run_test_suite(
        tests,
        suite_name="SMOOTHING SYNTHETIC tests",
        verbose_traceback=verbose_traceback,
    )

    if make_plots:
        run_smoothing_visual_tests()

    return ok


if __name__ == "__main__":
    run_smoothing_tests(
        verbose_traceback=True,
        make_plots=True,
    )
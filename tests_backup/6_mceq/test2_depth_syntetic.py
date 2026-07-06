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
Spyder-compatible synthetic tests for tpeanuts.external.mceq.depth.

No pytest required.
"""



import torch

from tpeanuts.external.mceq.depth import (
    theta_deg_to_cos,
    compute_vertical_depth_from_density,
    compute_slant_depth_from_vertical_depth,
    compute_dXdh,
)

from tpeanuts.util.test_utils import (
    assert_true,
    assert_close,
    assert_raises,
    run_test_suite,
)

from tpeanuts.util.torch_util import _default_device

# ============================================================
# Synthetic atmosphere
# ============================================================

def make_exponential_atmosphere(
    n_h=500,
    h_min_km=0.0,
    h_max_km=100.0,
    rho0_gcm3=1.225e-3,
    H_km=8.0,
    dtype=torch.float64,
):
    h = torch.linspace(
        h_min_km,
        h_max_km,
        n_h,
        dtype=dtype,
    )

    rho = rho0_gcm3 * torch.exp(-h / H_km)
    device = _default_device()
    return h.to(device), rho.to(device)


def analytic_vertical_depth_exponential(
    h_km,
    h_max_km=100.0,
    rho0_gcm3=1.225e-3,
    H_km=8.0,
):
    hmax_t = torch.tensor(
        h_max_km,
        dtype=h_km.dtype,
        device=h_km.device,
    )
    
    X= 1.0e5 * rho0_gcm3 * H_km * (torch.exp(-h_km / H_km)- torch.exp(-hmax_t / H_km))
    device = _default_device()
    return X.to(device)
       
    

# ============================================================
# tests
# ============================================================

def test_theta_deg_to_cos_basic_values():
    c0 = theta_deg_to_cos(0.0)
    c60 = theta_deg_to_cos(60.0)

    print("cos(0 deg)  =", c0)
    print("cos(60 deg) =", c60)

    assert_close(c0, 1.0, atol=1.0e-12, rtol=1.0e-12)
    assert_close(c60, 0.5, atol=1.0e-12, rtol=1.0e-12)


def test_vertical_depth_from_density_shape():
    h, rho = make_exponential_atmosphere()

    X = compute_vertical_depth_from_density(
        h_km=h,
        rho_gcm3=rho,
    )

    print("h shape:", h.shape)
    print("rho shape:", rho.shape)
    print("X shape:", X.shape)

    assert_true(X.shape == h.shape)


def test_vertical_depth_from_density_positive():
    h, rho = make_exponential_atmosphere()

    X = compute_vertical_depth_from_density(
        h_km=h,
        rho_gcm3=rho,
    )

    print("X min:", float(X.min().item()))
    print("X max:", float(X.max().item()))

    assert_true(torch.all(X >= 0.0).item())


def test_vertical_depth_decreases_with_height():
    h, rho = make_exponential_atmosphere()

    X = compute_vertical_depth_from_density(
        h_km=h,
        rho_gcm3=rho,
    )

    dX = torch.diff(X)

    print("max diff X:", float(dX.max().item()))
    print("min diff X:", float(dX.min().item()))

    assert_true(torch.all(dX <= 0.0).item())


def test_vertical_depth_top_is_zero():
    h, rho = make_exponential_atmosphere()

    X = compute_vertical_depth_from_density(
        h_km=h,
        rho_gcm3=rho,
    )

    X_top = float(X[-1].item())

    print("X(h_max) =", X_top)

    assert_close(X_top, 0.0, atol=1.0e-12, rtol=0.0)


def test_vertical_depth_matches_analytic_exponential():
    h_max = 100.0
    rho0 = 1.225e-3
    H = 8.0

    h, rho = make_exponential_atmosphere(
        n_h=2000,
        h_max_km=h_max,
        rho0_gcm3=rho0,
        H_km=H,
    )

    X_num = compute_vertical_depth_from_density(
        h_km=h,
        rho_gcm3=rho,
    )

    X_exact = analytic_vertical_depth_exponential(
        h,
        h_max_km=h_max,
        rho0_gcm3=rho0,
        H_km=H,
    )

    rel_err = torch.abs(X_num - X_exact) / torch.clamp(
        torch.abs(X_exact),
        min=1.0e-30,
    )

    mask = X_exact > 1.0e-6

    max_rel_err = float(torch.max(rel_err[mask]).item())

    print("X_num[0]   =", float(X_num[0].item()))
    print("X_exact[0] =", float(X_exact[0].item()))
    print("max relative error =", max_rel_err)

    assert_true(max_rel_err < 1.0e-3)


def test_slant_depth_theta_zero_equals_vertical():
    h, rho = make_exponential_atmosphere()

    X_vertical = compute_vertical_depth_from_density(
        h_km=h,
        rho_gcm3=rho,
    )

    X_slant = compute_slant_depth_from_vertical_depth(
        X_vertical_gcm2=X_vertical,
        theta_deg=0.0,
    )

    max_abs_diff = float(torch.max(torch.abs(X_slant - X_vertical)).item())

    print("max |X_slant - X_vertical| =", max_abs_diff)

    assert_true(
        torch.allclose(
            X_slant,
            X_vertical,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    )


def test_slant_depth_theta_60_is_twice_vertical():
    h, rho = make_exponential_atmosphere()

    X_vertical = compute_vertical_depth_from_density(
        h_km=h,
        rho_gcm3=rho,
    )

    X_slant = compute_slant_depth_from_vertical_depth(
        X_vertical_gcm2=X_vertical,
        theta_deg=60.0,
    )

    expected = 2.0 * X_vertical

    max_abs_diff = float(torch.max(torch.abs(X_slant - expected)).item())

    print("max |X_slant - 2 X_vertical| =", max_abs_diff)

    assert_true(
        torch.allclose(
            X_slant,
            expected,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    )


def test_dXdh_is_negative():
    h, rho = make_exponential_atmosphere()

    X = compute_vertical_depth_from_density(
        h_km=h,
        rho_gcm3=rho,
    )

    dXdh = compute_dXdh(
        X_gcm2=X,
        h_km=h,
    )

    print("dXdh min:", float(dXdh.min().item()))
    print("dXdh max:", float(dXdh.max().item()))

    assert_true(torch.all(dXdh[1:-1] < 0.0).item())


def test_dXdh_matches_expected_density_relation():
    h, rho = make_exponential_atmosphere(
        n_h=2000,
    )

    X = compute_vertical_depth_from_density(
        h_km=h,
        rho_gcm3=rho,
    )

    dXdh = compute_dXdh(
        X_gcm2=X,
        h_km=h,
    )

    expected = -1.0e5 * rho

    rel_err = torch.abs(dXdh - expected) / torch.clamp(
        torch.abs(expected),
        min=1.0e-30,
    )

    max_rel_err = float(torch.max(rel_err[5:-5]).item())

    print("max relative error dXdh =", max_rel_err)
    print("dXdh[0] expected approx:", float(expected[0].item()))
    print("dXdh[0] numerical      :", float(dXdh[0].item()))

    assert_true(max_rel_err < 1.0e-3)


def test_vertical_depth_rejects_non_1d_h():
    h = torch.zeros((2, 3), dtype=torch.float64)
    rho = torch.ones((2, 3), dtype=torch.float64)

    assert_raises(
        ValueError,
        compute_vertical_depth_from_density,
        h,
        rho,
    )


def test_vertical_depth_rejects_shape_mismatch():
    h = torch.linspace(0.0, 10.0, 10)
    rho = torch.ones(9)

    assert_raises(
        ValueError,
        compute_vertical_depth_from_density,
        h,
        rho,
    )


def test_vertical_depth_rejects_non_monotonic_h():
    h = torch.tensor(
        [0.0, 10.0, 5.0, 20.0],
        dtype=torch.float64,
    )

    rho = torch.ones_like(h)

    assert_raises(
        ValueError,
        compute_vertical_depth_from_density,
        h,
        rho,
    )


# ============================================================
# Runner
# ============================================================

def run_depth_synthetic_tests(verbose_traceback=False):
    tests = [
        test_theta_deg_to_cos_basic_values,
        test_vertical_depth_from_density_shape,
        test_vertical_depth_from_density_positive,
        test_vertical_depth_decreases_with_height,
        test_vertical_depth_top_is_zero,
        test_vertical_depth_matches_analytic_exponential,
        test_slant_depth_theta_zero_equals_vertical,
        test_slant_depth_theta_60_is_twice_vertical,
        test_dXdh_is_negative,
        test_dXdh_matches_expected_density_relation,
        test_vertical_depth_rejects_non_1d_h,
        test_vertical_depth_rejects_shape_mismatch,
        test_vertical_depth_rejects_non_monotonic_h,
    ]

    return run_test_suite(
        tests,
        suite_name="DEPTH SYNTHETIC tests",
        verbose_traceback=verbose_traceback,
    )


if __name__ == "__main__":
    run_depth_synthetic_tests(verbose_traceback=True)
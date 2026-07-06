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
Spyder-compatible synthetic tests and visual diagnostics for profiles.py.

No pytest required.
"""



import torch
import matplotlib.pyplot as plt

from tpeanuts.external.mceq.profiles import (
    interpolate_source_X_to_h,
    convert_depth_source_to_height_source,
    normalize_height_profiles,
    build_phi_Eh_from_profile,
)

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

N_X = 500
N_H = 400

X_MIN = 1.0
X_MAX = 1030.0

H_MIN = 0.0
H_MAX = 80.0

E_GRID = torch.tensor(
    [0.5, 1.0, 5.0, 10.0, 50.0],
    device=DEVICE,
    dtype=DTYPE,
)


# ============================================================
# Synthetic data
# ============================================================

def make_synthetic_depth_source():
    X_grid = torch.linspace(
        X_MIN,
        X_MAX,
        N_X,
        device=DEVICE,
        dtype=DTYPE,
    )

    h_grid = torch.linspace(
        H_MIN,
        H_MAX,
        N_H,
        device=DEVICE,
        dtype=DTYPE,
    )

    E = E_GRID.clone()

    X0 = 250.0
    H = 8.0
    X_surface = 1030.0

    A_E = E ** (-2.0)

    source_XE = torch.exp(-X_grid[:, None] / X0) * A_E[None, :]

    X_of_h = X_surface * torch.exp(-h_grid / H)

    dXdh = -X_of_h / H

    phi_E_obs = E ** (-2.7)

    return {
        "X_grid": X_grid,
        "h_grid": h_grid,
        "E_grid": E,
        "source_XE": source_XE,
        "X_of_h": X_of_h,
        "dXdh": dXdh,
        "phi_E_obs": phi_E_obs,
    }


# ============================================================
# tests
# ============================================================

def test_interpolate_source_X_to_h_shape():
    data = make_synthetic_depth_source()

    source_Eh = interpolate_source_X_to_h(
        X_grid_gcm2=data["X_grid"],
        source_XE=data["source_XE"],
        X_of_h_gcm2=data["X_of_h"],
        device=DEVICE,
        dtype=DTYPE,
    )

    print("source_XE shape:", data["source_XE"].shape)
    print("source_Eh shape:", source_Eh.shape)

    assert_true(source_Eh.shape == (data["E_grid"].numel(), data["h_grid"].numel()))


def test_interpolate_source_X_to_h_positive():
    data = make_synthetic_depth_source()

    source_Eh = interpolate_source_X_to_h(
        X_grid_gcm2=data["X_grid"],
        source_XE=data["source_XE"],
        X_of_h_gcm2=data["X_of_h"],
        device=DEVICE,
        dtype=DTYPE,
    )

    print("source_Eh min:", float(source_Eh.min().item()))
    print("source_Eh max:", float(source_Eh.max().item()))

    assert_true(torch.all(source_Eh >= 0.0).item())


def test_convert_depth_source_to_height_source_shape():
    data = make_synthetic_depth_source()

    source_Eh = convert_depth_source_to_height_source(
        X_grid_gcm2=data["X_grid"],
        source_XE=data["source_XE"],
        X_of_h_gcm2=data["X_of_h"],
        dXdh_gcm2_per_km=data["dXdh"],
        device=DEVICE,
        dtype=DTYPE,
    )

    print("height source shape:", source_Eh.shape)

    assert_true(source_Eh.shape == (data["E_grid"].numel(), data["h_grid"].numel()))


def test_convert_depth_source_to_height_source_positive():
    data = make_synthetic_depth_source()

    source_Eh = convert_depth_source_to_height_source(
        X_grid_gcm2=data["X_grid"],
        source_XE=data["source_XE"],
        X_of_h_gcm2=data["X_of_h"],
        dXdh_gcm2_per_km=data["dXdh"],
        device=DEVICE,
        dtype=DTYPE,
    )

    print("height source min:", float(source_Eh.min().item()))
    print("height source max:", float(source_Eh.max().item()))

    assert_true(torch.all(source_Eh >= 0.0).item())


def test_normalize_height_profiles_shape():
    data = make_synthetic_depth_source()

    source_Eh = convert_depth_source_to_height_source(
        X_grid_gcm2=data["X_grid"],
        source_XE=data["source_XE"],
        X_of_h_gcm2=data["X_of_h"],
        dXdh_gcm2_per_km=data["dXdh"],
        device=DEVICE,
        dtype=DTYPE,
    )

    f_Eh = normalize_height_profiles(
        h_grid_km=data["h_grid"],
        source_Eh=source_Eh,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("f_Eh shape:", f_Eh.shape)

    assert_true(f_Eh.shape == source_Eh.shape)


def test_normalize_height_profiles_integral_is_one():
    data = make_synthetic_depth_source()

    source_Eh = convert_depth_source_to_height_source(
        X_grid_gcm2=data["X_grid"],
        source_XE=data["source_XE"],
        X_of_h_gcm2=data["X_of_h"],
        dXdh_gcm2_per_km=data["dXdh"],
        device=DEVICE,
        dtype=DTYPE,
    )

    f_Eh = normalize_height_profiles(
        h_grid_km=data["h_grid"],
        source_Eh=source_Eh,
        device=DEVICE,
        dtype=DTYPE,
    )

    norm_E = torch.trapezoid(
        f_Eh,
        x=data["h_grid"],
        dim=1,
    )

    print("normalizations:", norm_E)

    max_abs_diff = float(torch.max(torch.abs(norm_E - 1.0)).item())

    print("max |norm - 1|:", max_abs_diff)

    assert_true(max_abs_diff < 1.0e-10)


def test_build_phi_Eh_from_profile_shape():
    data = make_synthetic_depth_source()

    source_Eh = convert_depth_source_to_height_source(
        X_grid_gcm2=data["X_grid"],
        source_XE=data["source_XE"],
        X_of_h_gcm2=data["X_of_h"],
        dXdh_gcm2_per_km=data["dXdh"],
        device=DEVICE,
        dtype=DTYPE,
    )

    f_Eh = normalize_height_profiles(
        h_grid_km=data["h_grid"],
        source_Eh=source_Eh,
        device=DEVICE,
        dtype=DTYPE,
    )

    phi_Eh = build_phi_Eh_from_profile(
        phi_E_obs=data["phi_E_obs"],
        f_Eh=f_Eh,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("phi_E_obs shape:", data["phi_E_obs"].shape)
    print("phi_Eh shape:", phi_Eh.shape)

    assert_true(phi_Eh.shape == f_Eh.shape)


def test_build_phi_Eh_reconstructs_phi_E_obs():
    data = make_synthetic_depth_source()

    source_Eh = convert_depth_source_to_height_source(
        X_grid_gcm2=data["X_grid"],
        source_XE=data["source_XE"],
        X_of_h_gcm2=data["X_of_h"],
        dXdh_gcm2_per_km=data["dXdh"],
        device=DEVICE,
        dtype=DTYPE,
    )

    f_Eh = normalize_height_profiles(
        h_grid_km=data["h_grid"],
        source_Eh=source_Eh,
        device=DEVICE,
        dtype=DTYPE,
    )

    phi_Eh = build_phi_Eh_from_profile(
        phi_E_obs=data["phi_E_obs"],
        f_Eh=f_Eh,
        device=DEVICE,
        dtype=DTYPE,
    )

    phi_rec = torch.trapezoid(
        phi_Eh,
        x=data["h_grid"],
        dim=1,
    )

    rel_err = torch.abs(phi_rec - data["phi_E_obs"]) / torch.clamp(
        torch.abs(data["phi_E_obs"]),
        min=1.0e-30,
    )

    max_rel_err = float(torch.max(rel_err).item())

    print("phi_E_obs:", data["phi_E_obs"])
    print("phi_rec  :", phi_rec)
    print("max relative error:", max_rel_err)

    assert_true(max_rel_err < 1.0e-10)


def test_interpolate_rejects_bad_source_shape():
    data = make_synthetic_depth_source()

    bad_source = torch.ones(
        data["X_grid"].numel(),
        device=DEVICE,
        dtype=DTYPE,
    )

    assert_raises(
        ValueError,
        interpolate_source_X_to_h,
        data["X_grid"],
        bad_source,
        data["X_of_h"],
        device=DEVICE,
        dtype=DTYPE,
    )


def test_interpolate_rejects_non_monotonic_X_grid():
    data = make_synthetic_depth_source()

    bad_X = data["X_grid"].clone()
    bad_X[20] = bad_X[10]

    assert_raises(
        ValueError,
        interpolate_source_X_to_h,
        bad_X,
        data["source_XE"],
        data["X_of_h"],
        device=DEVICE,
        dtype=DTYPE,
    )


def test_normalize_rejects_bad_shape():
    data = make_synthetic_depth_source()

    bad_source = torch.ones(
        data["h_grid"].numel(),
        device=DEVICE,
        dtype=DTYPE,
    )

    assert_raises(
        ValueError,
        normalize_height_profiles,
        data["h_grid"],
        bad_source,
        device=DEVICE,
        dtype=DTYPE,
    )


def test_build_phi_Eh_rejects_energy_mismatch():
    data = make_synthetic_depth_source()

    f_Eh = torch.ones(
        (3, data["h_grid"].numel()),
        device=DEVICE,
        dtype=DTYPE,
    )

    assert_raises(
        ValueError,
        build_phi_Eh_from_profile,
        data["phi_E_obs"],
        f_Eh,
        device=DEVICE,
        dtype=DTYPE,
    )


# ============================================================
# Visual diagnostics
# ============================================================

def build_full_synthetic_profile():
    data = make_synthetic_depth_source()

    source_interp_Eh = interpolate_source_X_to_h(
        X_grid_gcm2=data["X_grid"],
        source_XE=data["source_XE"],
        X_of_h_gcm2=data["X_of_h"],
        device=DEVICE,
        dtype=DTYPE,
    )

    source_Eh = convert_depth_source_to_height_source(
        X_grid_gcm2=data["X_grid"],
        source_XE=data["source_XE"],
        X_of_h_gcm2=data["X_of_h"],
        dXdh_gcm2_per_km=data["dXdh"],
        device=DEVICE,
        dtype=DTYPE,
    )

    f_Eh = normalize_height_profiles(
        h_grid_km=data["h_grid"],
        source_Eh=source_Eh,
        device=DEVICE,
        dtype=DTYPE,
    )

    phi_Eh = build_phi_Eh_from_profile(
        phi_E_obs=data["phi_E_obs"],
        f_Eh=f_Eh,
        device=DEVICE,
        dtype=DTYPE,
    )

    data["source_interp_Eh"] = source_interp_Eh
    data["source_Eh"] = source_Eh
    data["f_Eh"] = f_Eh
    data["phi_Eh"] = phi_Eh

    return data


def plot_depth_to_height_mapping():
    data = make_synthetic_depth_source()

    plt.figure(figsize=(8, 6))

    plt.plot(
        data["X_of_h"].cpu().numpy(),
        data["h_grid"].cpu().numpy(),
        lw=2,
        label=r"$X(h)$",
    )

    plt.xlabel(r"Depth $X(h)$ [g/cm$^2$]")
    plt.ylabel(r"Altitude $h$ [km]")
    plt.title("Synthetic depth-altitude mapping")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_source_X_and_h(i_E=2):
    data = build_full_synthetic_profile()

    E_val = float(data["E_grid"][i_E].item())

    plt.figure(figsize=(9, 6))

    plt.plot(
        data["X_grid"].cpu().numpy(),
        data["source_XE"][:, i_E].cpu().numpy(),
        lw=2,
        label=rf"$Q(E,X)$, E={E_val:g} GeV",
    )

    plt.xlabel(r"Depth $X$ [g/cm$^2$]")
    plt.ylabel(r"$Q(E,X)$")
    plt.title("Synthetic source in depth")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 6))

    plt.plot(
        data["h_grid"].cpu().numpy(),
        data["source_interp_Eh"][i_E, :].cpu().numpy(),
        lw=2,
        label=r"$Q(E,X(h))$",
    )

    plt.plot(
        data["h_grid"].cpu().numpy(),
        data["source_Eh"][i_E, :].cpu().numpy(),
        lw=2,
        ls="--",
        label=r"$Q(E,X(h)) |dX/dh|$",
    )

    plt.xlabel(r"Altitude $h$ [km]")
    plt.ylabel(r"Source")
    plt.title(r"Depth source mapped to height")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_normalized_profiles():
    data = build_full_synthetic_profile()

    plt.figure(figsize=(9, 6))

    for i in range(data["E_grid"].numel()):
        E_val = float(data["E_grid"][i].item())

        plt.plot(
            data["h_grid"].cpu().numpy(),
            data["f_Eh"][i, :].cpu().numpy(),
            lw=2,
            label=f"E={E_val:g} GeV",
        )

    plt.xlabel(r"Altitude $h$ [km]")
    plt.ylabel(r"$f(h|E,\theta)$ [km$^{-1}$]")
    plt.title("Normalized synthetic height profiles")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_phi_Eh_profiles():
    data = build_full_synthetic_profile()

    plt.figure(figsize=(9, 6))

    for i in range(data["E_grid"].numel()):
        E_val = float(data["E_grid"][i].item())

        plt.plot(
            data["h_grid"].cpu().numpy(),
            data["phi_Eh"][i, :].cpu().numpy(),
            lw=2,
            label=f"E={E_val:g} GeV",
        )

    plt.xlabel(r"Altitude $h$ [km]")
    plt.ylabel(r"$\Phi(E,h)$")
    plt.title(r"Height-differential flux $\Phi(E,h)$")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_reconstruction_check():
    data = build_full_synthetic_profile()

    phi_rec = torch.trapezoid(
        data["phi_Eh"],
        x=data["h_grid"],
        dim=1,
    )

    rel_err = torch.abs(phi_rec - data["phi_E_obs"]) / torch.clamp(
        torch.abs(data["phi_E_obs"]),
        min=1.0e-30,
    )

    plt.figure(figsize=(8, 6))

    plt.loglog(
        data["E_grid"].cpu().numpy(),
        data["phi_E_obs"].cpu().numpy(),
        marker="o",
        lw=2,
        label=r"Input $\Phi(E)$",
    )

    plt.loglog(
        data["E_grid"].cpu().numpy(),
        phi_rec.cpu().numpy(),
        marker="s",
        lw=2,
        ls="--",
        label=r"Reconstructed $\int \Phi(E,h) dh$",
    )

    plt.xlabel(r"Energy $E$ [GeV]")
    plt.ylabel(r"flux")
    plt.title("flux reconstruction from height profile")

    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))

    plt.semilogx(
        data["E_grid"].cpu().numpy(),
        rel_err.cpu().numpy(),
        marker="o",
        lw=2,
    )

    plt.xlabel(r"Energy $E$ [GeV]")
    plt.ylabel("Relative error")
    plt.title(r"Relative error in $\int \Phi(E,h)dh = \Phi(E)$")

    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()


def run_profiles_visual_tests():
    print("\n" + "=" * 80)
    print("PROFILES VISUAL tests")
    print("=" * 80)

    plot_depth_to_height_mapping()
    plot_source_X_and_h(i_E=2)
    plot_normalized_profiles()
    plot_phi_Eh_profiles()
    plot_reconstruction_check()

    print("\nFinished profile visual diagnostics.")


# ============================================================
# Runner
# ============================================================

def run_profiles_synthetic_tests(
    verbose_traceback=False,
    make_plots=True,
):
    tests = [
        test_interpolate_source_X_to_h_shape,
        test_interpolate_source_X_to_h_positive,
        test_convert_depth_source_to_height_source_shape,
        test_convert_depth_source_to_height_source_positive,
        test_normalize_height_profiles_shape,
        test_normalize_height_profiles_integral_is_one,
        test_build_phi_Eh_from_profile_shape,
        test_build_phi_Eh_reconstructs_phi_E_obs,
        test_interpolate_rejects_bad_source_shape,
        test_interpolate_rejects_non_monotonic_X_grid,
        test_normalize_rejects_bad_shape,
        test_build_phi_Eh_rejects_energy_mismatch,
    ]

    ok = run_test_suite(
        tests,
        suite_name="PROFILES SYNTHETIC tests",
        verbose_traceback=verbose_traceback,
    )

    if make_plots:
        run_profiles_visual_tests()

    return ok


if __name__ == "__main__":
    run_profiles_synthetic_tests(
        verbose_traceback=True,
        make_plots=True,
    )
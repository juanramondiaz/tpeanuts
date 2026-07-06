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
Spyder-friendly tests and visual diagnostics for tpeanuts.earth.evolutor.

This script checks:

    1. Import of the canonical earth evolutor.
    2. Energy/eta broadcasting.
    3. Above-horizon identity output at zero detector depth.
    4. Case-A and Case-B earth evolution matrices.
    5. Batched energy-angle grids.
    6. Antineutrino execution path.
    7. Invalid eta range handling.
    8. Visual diagnostics for unitarity and matrix structure.

Run directly in Spyder or from a terminal:

    python tpeanuts/tests/earth/test3_evolutor.py
"""



from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch


# ============================================================
# Import bootstrap
# ============================================================

THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parents[3]



from tpeanuts.core.pmns import PMNS  # noqa: E402
from tpeanuts.earth.evolutor import (  # noqa: E402
    _broadcast_energy_and_eta,
    earth_evolutor,
)
from tpeanuts.io.io_earth import load_earth_density_from_csv  # noqa: E402
from tpeanuts.util.test_utils import (  # noqa: E402
    assert_raises,
    assert_true,
    run_test_suite,
)


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device("cpu")
DTYPE = torch.float64
CDTYPE = torch.complex128

density_FILE = PACKAGE_DIR / "data" / "density" / "earth_density.csv"
TESTS_DIR = THIS_FILE.parents[1]

DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_TEST_ROOT = Path(OUTPUT_ROOT / "test")
OUTPUT_DIR = Path(OUTPUT_TEST_ROOT / "earth" / Path(__file__).stem)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DM21_EV2 = 7.42e-5
DM3L_EV2 = 2.517e-3
DEPTH_SURFACE_M = 0.0
DEPTH_UNDERGROUND_M = 1000.0

torch.set_printoptions(
    precision=10,
    sci_mode=True,
    linewidth=160,
)


# ============================================================
# Shared fixtures and helpers
# ============================================================

def build_pmns():
    return PMNS(
        theta12=0.59,
        theta13=0.15,
        theta23=0.78,
        delta=1.20,
        device=DEVICE,
        real_dtype=DTYPE,
    )


def load_density():
    return load_earth_density_from_csv(
        str(density_FILE),
        tabulated_density=False,
        device=DEVICE,
        dtype=DTYPE,
    )


def identity_like(U):
    return torch.eye(
        3,
        device=U.device,
        dtype=U.dtype,
    )


def unitarity_error(U):
    identity = torch.eye(3, device=U.device, dtype=U.dtype)
    left = U.conj().transpose(-1, -2) @ U
    return torch.amax(torch.abs(left - identity), dim=(-2, -1)).detach().cpu()


def distance_from_identity(U):
    identity = torch.eye(3, device=U.device, dtype=U.dtype)
    return torch.linalg.norm(U - identity, dim=(-2, -1)).detach().cpu()


def assert_tensor_close(actual, expected, message, atol=1.0e-10, rtol=1.0e-8):
    actual_t = torch.as_tensor(actual)
    expected_t = torch.as_tensor(expected, dtype=actual_t.dtype, device=actual_t.device)

    max_diff = torch.max(torch.abs(actual_t - expected_t)).item()

    print(f"Checking: {message}")
    print("  actual shape  :", tuple(actual_t.shape))
    print("  expected shape:", tuple(expected_t.shape))
    print("  max abs diff  :", f"{max_diff:.6e}")
    print("  atol / rtol   :", f"{atol:.1e} / {rtol:.1e}")

    assert_true(
        torch.allclose(actual_t, expected_t, atol=atol, rtol=rtol),
        message,
    )


def assert_unitary(U, message, atol=5.0e-10):
    err = unitarity_error(U)

    print(f"Checking: {message}")
    print("  U shape             :", tuple(U.shape))
    print("  max unitarity error :", f"{torch.max(err).item():.6e}")

    assert_true(
        torch.max(err).item() < atol,
        message,
    )


# ============================================================
# tests
# ============================================================

def test_import_earth_evolutor():
    print("\nImported public earth evolutor:")
    print("earth_evolutor :", earth_evolutor)

    assert_true(callable(earth_evolutor), "earth_evolutor must be callable")


def test_broadcast_energy_and_eta_helper():
    E = torch.tensor([500.0, 1000.0, 5000.0], device=DEVICE, dtype=DTYPE)
    eta = torch.tensor([0.20, 0.80], device=DEVICE, dtype=DTYPE)

    E_b, eta_b, device, rdtype, cdtype = _broadcast_energy_and_eta(E, eta)

    print("\nBroadcasted energy and eta:")
    print("E_b shape   :", tuple(E_b.shape))
    print("eta_b shape :", tuple(eta_b.shape))
    print("device      :", device)
    print("real dtype  :", rdtype)
    print("complex dtype:", cdtype)
    print("E_b:")
    print(E_b)
    print("eta_b:")
    print(eta_b)

    assert_true(E_b.shape == (3, 2), "Energy vector and eta vector must broadcast to a grid")
    assert_true(eta_b.shape == (3, 2), "Eta vector and energy vector must broadcast to a grid")
    assert_true(device == DEVICE, "Broadcast helper must keep the energy tensor device")
    assert_true(rdtype == DTYPE, "Broadcast helper must keep the energy tensor dtype")
    assert_true(cdtype == CDTYPE, "Broadcast helper must infer complex128 from float64")


def test_above_horizon_is_identity_at_surface():
    density = load_density()
    pmns = build_pmns()

    eta = torch.tensor([torch.pi / 2.0, 2.00, 3.00], device=DEVICE, dtype=DTYPE)
    E = torch.tensor([1000.0, 2000.0, 3000.0], device=DEVICE, dtype=DTYPE)

    U = earth_evolutor(
        density=density,
        DeltamSq21=DM21_EV2,
        DeltamSq3l=DM3L_EV2,
        pmns=pmns,
        E=E,
        eta=eta,
        depth_m=DEPTH_SURFACE_M,
        reunitarize=True,
    )

    expected = identity_like(U).expand(3, 3, 3)

    print("\nAbove-horizon surface output:")
    print("eta:", eta)
    print("U shape:", tuple(U.shape))
    print("U[0]:")
    print(U[0])

    assert_tensor_close(U, expected, "Surface above-horizon trajectories return identity", atol=1.0e-12, rtol=1.0e-12)


def test_case_a_scalar_evolutor_is_finite_and_unitary():
    density = load_density()
    pmns = build_pmns()

    U = earth_evolutor(
        density=density,
        DeltamSq21=DM21_EV2,
        DeltamSq3l=DM3L_EV2,
        pmns=pmns,
        E=torch.tensor(1000.0, device=DEVICE, dtype=DTYPE),
        eta=torch.tensor(0.40, device=DEVICE, dtype=DTYPE),
        depth_m=DEPTH_SURFACE_M,
        reunitarize=True,
    )

    print("\nCase-A scalar evolutor:")
    print("U shape:", tuple(U.shape))
    print("U:")
    print(U)

    assert_true(U.shape == (3, 3), "Scalar Case-A output must have shape (3, 3)")
    assert_true(torch.isfinite(U.real).all().item(), "Case-A real part must be finite")
    assert_true(torch.isfinite(U.imag).all().item(), "Case-A imaginary part must be finite")
    assert_unitary(U, "Case-A scalar output is unitary")


def test_case_b_underground_evolutor_is_finite_and_unitary():
    density = load_density()
    pmns = build_pmns()

    U = earth_evolutor(
        density=density,
        DeltamSq21=DM21_EV2,
        DeltamSq3l=DM3L_EV2,
        pmns=pmns,
        E=torch.tensor(2500.0, device=DEVICE, dtype=DTYPE),
        eta=torch.tensor(2.40, device=DEVICE, dtype=DTYPE),
        depth_m=DEPTH_UNDERGROUND_M,
        reunitarize=True,
    )

    print("\nCase-B underground evolutor:")
    print("U shape:", tuple(U.shape))
    print("max |U|:", torch.max(torch.abs(U)).item())
    print("distance from identity:", distance_from_identity(U).item())

    assert_true(U.shape == (3, 3), "Scalar Case-B output must have shape (3, 3)")
    assert_true(torch.isfinite(U.real).all().item(), "Case-B real part must be finite")
    assert_true(torch.isfinite(U.imag).all().item(), "Case-B imaginary part must be finite")
    assert_unitary(U, "Case-B scalar output is unitary")


def test_energy_eta_grid_output_shape_and_identity_region():
    density = load_density()
    pmns = build_pmns()

    E = torch.tensor([800.0, 2000.0, 6000.0], device=DEVICE, dtype=DTYPE)
    eta = torch.tensor([0.15, 2.20], device=DEVICE, dtype=DTYPE)

    U = earth_evolutor(
        density=density,
        DeltamSq21=DM21_EV2,
        DeltamSq3l=DM3L_EV2,
        pmns=pmns,
        E=E,
        eta=eta,
        depth_m=DEPTH_SURFACE_M,
        reunitarize=True,
    )

    identity = identity_like(U)

    print("\nEnergy-eta grid evolutor:")
    print("E shape  :", tuple(E.shape))
    print("eta shape:", tuple(eta.shape))
    print("U shape  :", tuple(U.shape))
    print("unitarity errors:")
    print(unitarity_error(U))

    assert_true(U.shape == (3, 2, 3, 3), "Energy vector and eta vector output must have shape (NE, Neta, 3, 3)")
    assert_unitary(U, "Energy-eta grid output is unitary")
    assert_tensor_close(U[:, 1], identity.expand(3, 3, 3), "Above-horizon eta column is identity", atol=1.0e-12, rtol=1.0e-12)


def test_antineutrino_path_runs_and_is_unitary():
    density = load_density()
    pmns = build_pmns()

    eta = torch.tensor([0.25, 1.10], device=DEVICE, dtype=DTYPE)
    E = torch.tensor([1000.0, 4000.0], device=DEVICE, dtype=DTYPE)

    U_nu = earth_evolutor(
        density=density,
        DeltamSq21=DM21_EV2,
        DeltamSq3l=DM3L_EV2,
        pmns=pmns,
        E=E,
        eta=eta,
        depth_m=DEPTH_SURFACE_M,
        antinu=False,
        reunitarize=True,
    )

    U_antinu = earth_evolutor(
        density=density,
        DeltamSq21=DM21_EV2,
        DeltamSq3l=DM3L_EV2,
        pmns=pmns,
        E=E,
        eta=eta,
        depth_m=DEPTH_SURFACE_M,
        antinu=True,
        reunitarize=True,
    )

    diff = torch.max(torch.abs(U_nu - U_antinu)).item()

    print("\nNeutrino vs antineutrino:")
    print("U_nu shape     :", tuple(U_nu.shape))
    print("U_antinu shape :", tuple(U_antinu.shape))
    print("max |difference|:", f"{diff:.6e}")

    assert_true(U_antinu.shape == (2, 3, 3), "Antineutrino batch output must have shape (2, 3, 3)")
    assert_unitary(U_antinu, "Antineutrino output is unitary")
    assert_true(diff > 0.0, "Neutrino and antineutrino paths should not be exactly identical for these inputs")


def test_invalid_eta_raises_value_error():
    density = load_density()
    pmns = build_pmns()

    def call_with_bad_eta():
        earth_evolutor(
            density=density,
            DeltamSq21=DM21_EV2,
            DeltamSq3l=DM3L_EV2,
            pmns=pmns,
            E=torch.tensor(1000.0, device=DEVICE, dtype=DTYPE),
            eta=torch.tensor(-0.01, device=DEVICE, dtype=DTYPE),
            depth_m=DEPTH_SURFACE_M,
        )

    print("\nInvalid eta test:")
    print("eta = -0.01 should raise ValueError")

    assert_raises(ValueError, call_with_bad_eta)


# ============================================================
# Visualization
# ============================================================

def plot_unitarity_error_vs_eta(savefig=False):
    density = load_density()
    pmns = build_pmns()

    eta = torch.linspace(0.0, torch.pi, 31, device=DEVICE, dtype=DTYPE)
    E = torch.full_like(eta, 2000.0)

    U_raw = earth_evolutor(
        density=density,
        DeltamSq21=DM21_EV2,
        DeltamSq3l=DM3L_EV2,
        pmns=pmns,
        E=E,
        eta=eta,
        depth_m=DEPTH_SURFACE_M,
        reunitarize=False,
    )

    U_projected = earth_evolutor(
        density=density,
        DeltamSq21=DM21_EV2,
        DeltamSq3l=DM3L_EV2,
        pmns=pmns,
        E=E,
        eta=eta,
        depth_m=DEPTH_SURFACE_M,
        reunitarize=True,
    )

    eta_cpu = eta.detach().cpu()
    raw_err = unitarity_error(U_raw)
    projected_err = unitarity_error(U_projected)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(eta_cpu, raw_err, marker="o", label="Raw evolutor")
    ax.semilogy(eta_cpu, projected_err, marker="s", label="Reunitarized evolutor")
    ax.axvline(float(torch.pi / 2.0), color="black", ls="--", lw=1.0, label=r"$\pi/2$")
    ax.set_xlabel(r"Nadir angle $\eta$ [rad]")
    ax.set_ylabel(r"max $|U^\dagger U - I|$")
    ax.set_title("earth evolutor unitarity diagnostic")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = OUTPUT_DIR / "earth_evolutor_unitarity_vs_eta.png"
    if savefig:
        fig.savefig(path, dpi=200)
    plt.show()

    if savefig:
        print(f"\nSaved plot: {path}")


def plot_distance_from_identity_map(savefig=False):
    density = load_density()
    pmns = build_pmns()

    E = torch.tensor([500.0, 1000.0, 3000.0, 10000.0], device=DEVICE, dtype=DTYPE)
    eta = torch.linspace(0.0, torch.pi, 25, device=DEVICE, dtype=DTYPE)

    U = earth_evolutor(
        density=density,
        DeltamSq21=DM21_EV2,
        DeltamSq3l=DM3L_EV2,
        pmns=pmns,
        E=E,
        eta=eta,
        depth_m=DEPTH_SURFACE_M,
        reunitarize=True,
    )

    dist = distance_from_identity(U).numpy()

    fig, ax = plt.subplots(figsize=(8, 4.8))
    image = ax.imshow(
        dist,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[
            float(eta[0].item()),
            float(eta[-1].item()),
            float(E[0].item()),
            float(E[-1].item()),
        ],
        cmap="magma",
    )
    ax.axvline(float(torch.pi / 2.0), color="white", ls="--", lw=1.0)
    ax.set_xlabel(r"Nadir angle $\eta$ [rad]")
    ax.set_ylabel("Energy [MeV]")
    ax.set_title(r"Distance from identity: $\|U-I\|_F$")
    fig.colorbar(image, ax=ax, label=r"$\|U-I\|_F$")
    fig.tight_layout()

    path = OUTPUT_DIR / "earth_evolutor_distance_from_identity.png"
    if savefig:
        fig.savefig(path, dpi=200)
    plt.show()

    if savefig:
        print(f"Saved plot: {path}")


def plot_matrix_magnitude_examples(savefig=False):
    density = load_density()
    pmns = build_pmns()

    labels = [
        ("Case A: through earth", 1000.0, 0.35, DEPTH_SURFACE_M),
        ("Case B: underground shallow chord", 1000.0, 2.40, DEPTH_UNDERGROUND_M),
    ]

    matrices = []

    for _, energy, eta, depth in labels:
        U = earth_evolutor(
            density=density,
            DeltamSq21=DM21_EV2,
            DeltamSq3l=DM3L_EV2,
            pmns=pmns,
            E=torch.tensor(energy, device=DEVICE, dtype=DTYPE),
            eta=torch.tensor(eta, device=DEVICE, dtype=DTYPE),
            depth_m=depth,
            reunitarize=True,
        )
        matrices.append(torch.abs(U).detach().cpu().numpy())

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), constrained_layout=True)

    for ax, matrix, (title, energy, eta, depth) in zip(axes, matrices, labels):
        for i in range(matrix.shape[1]):
            matrix[i,:] = matrix[i,:]/matrix[i,:].sum()
        image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(f"{title}\nE={energy:.0f} MeV, eta={eta:.2f} rad, depth={depth:.0f} m")
        ax.set_xlabel("Final flavour index")
        ax.set_ylabel("Initial flavour index")
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])

        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white")

    fig.colorbar(image, ax=axes, label=r"$|U_{\alpha\beta}|$")

    path = OUTPUT_DIR / "earth_evolutor_matrix_magnitudes.png"
    if savefig:
        fig.savefig(path, dpi=200)
    plt.show()

    if savefig:
        print(f"Saved plot: {path}")


def test_visualization_outputs(savefig=False):
    plot_unitarity_error_vs_eta(savefig=savefig)
    plot_distance_from_identity_map(savefig=savefig)
    plot_matrix_magnitude_examples(savefig=savefig)

    expected_files = [
        OUTPUT_DIR / "earth_evolutor_unitarity_vs_eta.png",
        OUTPUT_DIR / "earth_evolutor_distance_from_identity.png",
        OUTPUT_DIR / "earth_evolutor_matrix_magnitudes.png",
    ]

    for path in expected_files:
        print(f"Checking plot file: {path}")
        if savefig:
            assert_true(path.is_file(), f"Plot was not created: {path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    tests = [
        test_import_earth_evolutor,
        test_broadcast_energy_and_eta_helper,
        test_above_horizon_is_identity_at_surface,
        test_case_a_scalar_evolutor_is_finite_and_unitary,
        test_case_b_underground_evolutor_is_finite_and_unitary,
        test_energy_eta_grid_output_shape_and_identity_region,
        test_antineutrino_path_runs_and_is_unitary,
        test_invalid_eta_raises_value_error,
        test_visualization_outputs,
    ]

    run_test_suite(
        tests,
        suite_name="earth EVOLUTOR tests",
        verbose_traceback=True,
    )

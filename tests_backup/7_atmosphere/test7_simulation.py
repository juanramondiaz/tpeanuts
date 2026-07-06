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
Scientific validation checks for atmospheric coherent propagation.

This file is intentionally Spyder-friendly and does not require pytest.  It
checks controlled limits before producing visual diagnostics:

1. Zero atmospheric baseline gives the identity operator.
2. vacuum propagation matches the exact constant-Hamiltonian matrix exponential.
3. Matter propagation is unitary and conserves state norm.
4. Flavour probabilities are finite, bounded, and normalized.
5. vacuum convergence improves with integration steps.
6. Diagnostic plots show probability behaviour versus energy and zenith angle.
"""



from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch


THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parents[3]


from tpeanuts.core.pmns import PMNS  # noqa: E402
from tpeanuts.atmosphere.geometry import atmospheric_path_length  # noqa: E402
from tpeanuts.atmosphere.propagation import (  # noqa: E402
    atmospheric_evolution_operator,
    atmospheric_hamiltonian,
    propagate_atmosphere,
)
from tpeanuts.util.test_utils import (  # noqa: E402
    assert_close,
    assert_true,
    run_test_suite,
)


# ============================================================
# Configuration
# ============================================================

DEVICE = "cpu"
DTYPE = torch.float64
CDTYPE = torch.complex128

DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_TEST_ROOT = Path(OUTPUT_ROOT / "test")
OUTPUT_DIR = Path(OUTPUT_TEST_ROOT / "atmosphere" / Path(__file__).stem)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHOW_PLOTS = True

DM21_EV2 = torch.tensor(7.42e-5, device=DEVICE, dtype=DTYPE)
DM3L_EV2 = torch.tensor(2.517e-3, device=DEVICE, dtype=DTYPE)

THETA12 = torch.deg2rad(torch.tensor(33.44, device=DEVICE, dtype=DTYPE))
THETA13 = torch.deg2rad(torch.tensor(8.57, device=DEVICE, dtype=DTYPE))
THETA23 = torch.deg2rad(torch.tensor(49.2, device=DEVICE, dtype=DTYPE))
DELTA_CP = torch.deg2rad(torch.tensor(195.0, device=DEVICE, dtype=DTYPE))

E_MEV = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
E_GEV = E_MEV / 1000.0
H_KM = torch.tensor(20.0, device=DEVICE, dtype=DTYPE)
THETA_DEG = torch.tensor(45.0, device=DEVICE, dtype=DTYPE)
DEPTH_KM = torch.tensor(1.0, device=DEVICE, dtype=DTYPE)

N_STEPS_REFERENCE = 180
N_STEPS_FAST = 48

FLAVOUR_LABELS = [r"$\nu_e$", r"$\nu_\mu$", r"$\nu_\tau$"]


torch.set_printoptions(
    precision=10,
    sci_mode=True,
    linewidth=160,
)


def build_pmns() -> PMNS:
    return PMNS(
        THETA12,
        THETA13,
        THETA23,
        DELTA_CP,
        device=DEVICE,
        real_dtype=DTYPE,
    )


def flavour_state(index: int) -> torch.Tensor:
    state = torch.zeros(3, device=DEVICE, dtype=CDTYPE)
    state[int(index)] = 1.0 + 0.0j
    return state


def exact_vacuum_operator(
    *,
    pmns: PMNS,
    E_MeV: torch.Tensor,
    h_km: torch.Tensor,
    theta_deg: torch.Tensor,
    depth_km: torch.Tensor,
) -> torch.Tensor:
    H_vac = atmospheric_hamiltonian(
        pmns=pmns,
        DeltamSq21=DM21_EV2,
        DeltamSq3l=DM3L_EV2,
        E_MeV=E_MeV,
        ne_molcm3=torch.zeros((), device=DEVICE, dtype=DTYPE),
        antinu=False,
        device=DEVICE,
        dtype=DTYPE,
    )

    _, x_grid = atmospheric_evolution_operator(
        pmns=pmns,
        DeltamSq21=DM21_EV2,
        DeltamSq3l=DM3L_EV2,
        E_MeV=E_MeV,
        h_km=h_km,
        theta_deg=theta_deg,
        depth_km=depth_km,
        antinu=False,
        n_steps=8,
        matter=False,
        device=DEVICE,
        dtype=DTYPE,
    )
    length_x = (x_grid[..., -1] - x_grid[..., 0]).to(dtype=CDTYPE)

    return torch.linalg.matrix_exp(-1j * H_vac * length_x[..., None, None])


def probability_matrix(S: torch.Tensor) -> torch.Tensor:
    return torch.abs(S) ** 2


def max_unitarity_error(S: torch.Tensor) -> float:
    identity = torch.eye(3, device=S.device, dtype=S.dtype)
    err = S.conj().transpose(-1, -2) @ S - identity
    return float(torch.amax(torch.abs(err)).detach().cpu().item())


def save_plot(fig, filename: str, savefig=False) -> Path:
    path = OUTPUT_DIR / filename
    if savefig:
        fig.savefig(path, dpi=180, bbox_inches="tight")
    return path


def finish_plot(fig, filename: str, savefig=False) -> Path:
    path = save_plot(fig, filename, savefig=savefig)
    if SHOW_PLOTS:
        plt.show(block=False)
        plt.pause(0.1)
    else:
        plt.close(fig)
    return path


# ============================================================
# Numerical tests
# ============================================================

def test_zero_height_identity():
    pmns = build_pmns()

    S, x_grid = atmospheric_evolution_operator(
        pmns=pmns,
        DeltamSq21=DM21_EV2,
        DeltamSq3l=DM3L_EV2,
        E_MeV=E_MEV,
        h_km=torch.tensor(0.0, device=DEVICE, dtype=DTYPE),
        theta_deg=THETA_DEG,
        depth_km=DEPTH_KM,
        antinu=False,
        n_steps=N_STEPS_FAST,
        matter=True,
        device=DEVICE,
        dtype=DTYPE,
    )

    identity = torch.eye(3, device=DEVICE, dtype=CDTYPE)
    max_err = torch.max(torch.abs(S - identity)).item()

    print("\nZero-height identity check")
    print("x_grid span:", float((x_grid[-1] - x_grid[0]).detach().cpu().item()))
    print("max |S-I| :", f"{max_err:.3e}")

    assert_true(max_err < 1.0e-12, "Zero production height must return identity.")


def test_vacuum_limit_matches_exact_exponential():
    pmns = build_pmns()

    S_num, x_grid = atmospheric_evolution_operator(
        pmns=pmns,
        DeltamSq21=DM21_EV2,
        DeltamSq3l=DM3L_EV2,
        E_MeV=E_MEV,
        h_km=H_KM,
        theta_deg=THETA_DEG,
        depth_km=DEPTH_KM,
        antinu=False,
        n_steps=N_STEPS_REFERENCE,
        matter=False,
        device=DEVICE,
        dtype=DTYPE,
    )

    S_exact = exact_vacuum_operator(
        pmns=pmns,
        E_MeV=E_MEV,
        h_km=H_KM,
        theta_deg=THETA_DEG,
        depth_km=DEPTH_KM,
    )

    diff = torch.linalg.norm(S_num - S_exact).item()
    length_km = atmospheric_path_length(
        h_km=H_KM,
        theta_deg=THETA_DEG,
        depth_km=DEPTH_KM,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("\nvacuum exact-limit check")
    print("E [GeV]       :", float(E_GEV.detach().cpu().item()))
    print("h [km]        :", float(H_KM.detach().cpu().item()))
    print("theta [deg]   :", float(THETA_DEG.detach().cpu().item()))
    print("L_atm [km]    :", float(length_km.detach().cpu().item()))
    print("x span        :", float((x_grid[-1] - x_grid[0]).detach().cpu().item()))
    print("||S-S_exact|| :", f"{diff:.3e}")

    assert_true(diff < 1.0e-10, "vacuum numerical propagator must match exact exponential.")


def test_matter_unitarity_and_norm_conservation():
    pmns = build_pmns()
    psi0 = flavour_state(1)

    psi_surface, S = propagate_atmosphere(
        nustate=psi0,
        pmns=pmns,
        DeltamSq21=DM21_EV2,
        DeltamSq3l=DM3L_EV2,
        E_MeV=E_MEV,
        h_km=H_KM,
        theta_deg=THETA_DEG,
        depth_km=DEPTH_KM,
        antinu=False,
        n_steps=N_STEPS_REFERENCE,
        matter=True,
        device=DEVICE,
        dtype=DTYPE,
    )

    unit_err = max_unitarity_error(S)
    norm0 = torch.sum(torch.abs(psi0) ** 2)
    norm1 = torch.sum(torch.abs(psi_surface) ** 2)
    norm_err = torch.abs(norm1 - norm0).item()

    print("\nMatter unitarity and norm check")
    print("max |S^dag S-I|:", f"{unit_err:.3e}")
    print("initial norm   :", f"{float(norm0):.12f}")
    print("surface norm   :", f"{float(norm1):.12f}")
    print("norm error     :", f"{norm_err:.3e}")

    assert_true(unit_err < 1.0e-10, "Matter atmospheric operator must be unitary.")
    assert_true(norm_err < 1.0e-10, "coherent propagation must conserve state norm.")


def test_probability_matrix_is_physical():
    pmns = build_pmns()

    S, _ = atmospheric_evolution_operator(
        pmns=pmns,
        DeltamSq21=DM21_EV2,
        DeltamSq3l=DM3L_EV2,
        E_MeV=E_MEV,
        h_km=H_KM,
        theta_deg=THETA_DEG,
        depth_km=DEPTH_KM,
        antinu=False,
        n_steps=N_STEPS_REFERENCE,
        matter=True,
        device=DEVICE,
        dtype=DTYPE,
    )

    P = probability_matrix(S)
    row_sums = P.sum(dim=-1)
    col_sums = P.sum(dim=-2)

    print("\nProbability matrix physicality check")
    print("P(beta -> alpha):")
    print(P.detach().cpu())
    print("row sums:", row_sums.detach().cpu())
    print("col sums:", col_sums.detach().cpu())

    assert_true(torch.isfinite(P).all().item(), "Probability matrix contains NaN or Inf.")
    assert_true((P >= -1.0e-12).all().item(), "Probability matrix has negative values.")
    assert_true((P <= 1.0 + 1.0e-12).all().item(), "Probability matrix has values above one.")
    assert_close(row_sums, torch.ones_like(row_sums), atol=1.0e-10, rtol=1.0e-10)
    assert_close(col_sums, torch.ones_like(col_sums), atol=1.0e-10, rtol=1.0e-10)


def test_vacuum_convergence_is_stable():
    pmns = build_pmns()
    S_exact = exact_vacuum_operator(
        pmns=pmns,
        E_MeV=E_MEV,
        h_km=H_KM,
        theta_deg=THETA_DEG,
        depth_km=DEPTH_KM,
    )

    steps = [8, 16, 32, 64, 128, 256]
    errors = []

    for n_steps in steps:
        S, _ = atmospheric_evolution_operator(
            pmns=pmns,
            DeltamSq21=DM21_EV2,
            DeltamSq3l=DM3L_EV2,
            E_MeV=E_MEV,
            h_km=H_KM,
            theta_deg=THETA_DEG,
            depth_km=DEPTH_KM,
            antinu=False,
            n_steps=n_steps,
            matter=False,
            device=DEVICE,
            dtype=DTYPE,
        )
        errors.append(float(torch.linalg.norm(S - S_exact).detach().cpu().item()))

    print("\nvacuum convergence stability")
    for n_steps, error in zip(steps, errors):
        print(f"n_steps={n_steps:4d} | error={error:.3e}")

    assert_true(errors[-1] < 1.0e-10, "High-step vacuum error must be small.")


def test_matter_effect_is_small_but_finite_for_atmosphere():
    pmns = build_pmns()

    S_vac, _ = atmospheric_evolution_operator(
        pmns=pmns,
        DeltamSq21=DM21_EV2,
        DeltamSq3l=DM3L_EV2,
        E_MeV=E_MEV,
        h_km=H_KM,
        theta_deg=THETA_DEG,
        depth_km=DEPTH_KM,
        antinu=False,
        n_steps=N_STEPS_REFERENCE,
        matter=False,
        device=DEVICE,
        dtype=DTYPE,
    )
    S_mat, _ = atmospheric_evolution_operator(
        pmns=pmns,
        DeltamSq21=DM21_EV2,
        DeltamSq3l=DM3L_EV2,
        E_MeV=E_MEV,
        h_km=H_KM,
        theta_deg=THETA_DEG,
        depth_km=DEPTH_KM,
        antinu=False,
        n_steps=N_STEPS_REFERENCE,
        matter=True,
        device=DEVICE,
        dtype=DTYPE,
    )

    delta_p = torch.max(torch.abs(probability_matrix(S_mat) - probability_matrix(S_vac))).item()

    print("\nAtmospheric matter-vacuum difference")
    print("max |P_matter-P_vacuum|:", f"{delta_p:.3e}")

    assert_true(torch.isfinite(torch.as_tensor(delta_p)).item(), "Matter-vacuum difference must be finite.")
    assert_true(delta_p >= 0.0, "Matter-vacuum difference must be non-negative.")
    assert_true(delta_p < 1.0e-2, "Atmospheric matter effect should remain modest for this reference point.")


# ============================================================
# Visual diagnostics
# ============================================================

def plot_vacuum_convergence(savefig=False) -> Path:
    pmns = build_pmns()
    S_exact = exact_vacuum_operator(
        pmns=pmns,
        E_MeV=E_MEV,
        h_km=H_KM,
        theta_deg=THETA_DEG,
        depth_km=DEPTH_KM,
    )

    steps = torch.tensor([8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256])
    errors = []

    for n_steps in steps.tolist():
        S, _ = atmospheric_evolution_operator(
            pmns=pmns,
            DeltamSq21=DM21_EV2,
            DeltamSq3l=DM3L_EV2,
            E_MeV=E_MEV,
            h_km=H_KM,
            theta_deg=THETA_DEG,
            depth_km=DEPTH_KM,
            antinu=False,
            n_steps=int(n_steps),
            matter=False,
            device=DEVICE,
            dtype=DTYPE,
        )
        errors.append(float(torch.linalg.norm(S - S_exact).detach().cpu().item()))

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.loglog(steps.detach().cpu(), errors, marker="o", linewidth=1.8)
    ax.set_xlabel("Atmospheric integration steps")
    ax.set_ylabel(r"$||S_{num}-S_{exact}^{vac}||$")
    ax.set_title("vacuum-limit convergence")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return finish_plot(fig, "test7_vacuum_convergence.png", savefig=savefig)


def plot_probability_vs_energy(savefig=False) -> Path:
    pmns = build_pmns()
    energies_GeV = torch.logspace(-1, 3, 18, device=DEVICE, dtype=DTYPE)
    psi0 = flavour_state(1)
    probs = []

    for E_GeV_i in energies_GeV:
        psi, _ = propagate_atmosphere(
            nustate=psi0,
            pmns=pmns,
            DeltamSq21=DM21_EV2,
            DeltamSq3l=DM3L_EV2,
            E_MeV=1000.0 * E_GeV_i,
            h_km=H_KM,
            theta_deg=THETA_DEG,
            depth_km=DEPTH_KM,
            antinu=False,
            n_steps=N_STEPS_FAST,
            matter=True,
            device=DEVICE,
            dtype=DTYPE,
        )
        probs.append(torch.abs(psi) ** 2)

    probs = torch.stack(probs, dim=0)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for i, label in enumerate(FLAVOUR_LABELS):
        ax.semilogx(
            energies_GeV.detach().cpu(),
            probs[:, i].detach().cpu(),
            linewidth=1.8,
            label=rf"$\nu_\mu\to${label}",
        )
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel("Probability at surface")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(r"Atmospheric propagation: $P(\nu_\mu\to\nu_i)$ vs energy")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return finish_plot(fig, "test7_probability_vs_energy.png", savefig=savefig)


def plot_probability_vs_theta(savefig=False) -> Path:
    pmns = build_pmns()
    theta_values = torch.linspace(0.0, 88.0, 18, device=DEVICE, dtype=DTYPE)
    psi0 = flavour_state(1)
    probs = []

    for theta_i in theta_values:
        psi, _ = propagate_atmosphere(
            nustate=psi0,
            pmns=pmns,
            DeltamSq21=DM21_EV2,
            DeltamSq3l=DM3L_EV2,
            E_MeV=E_MEV,
            h_km=H_KM,
            theta_deg=theta_i,
            depth_km=DEPTH_KM,
            antinu=False,
            n_steps=N_STEPS_FAST,
            matter=True,
            device=DEVICE,
            dtype=DTYPE,
        )
        probs.append(torch.abs(psi) ** 2)

    probs = torch.stack(probs, dim=0)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for i, label in enumerate(FLAVOUR_LABELS):
        ax.plot(
            theta_values.detach().cpu(),
            probs[:, i].detach().cpu(),
            linewidth=1.8,
            label=rf"$\nu_\mu\to${label}",
        )
    ax.set_xlabel("Surface zenith angle theta [deg]")
    ax.set_ylabel("Probability at surface")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(r"Atmospheric propagation: $P(\nu_\mu\to\nu_i)$ vs theta")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    return finish_plot(fig, "test7_probability_vs_theta.png", savefig=savefig)


def plot_matter_vacuum_probability_difference(savefig=False) -> Path:
    pmns = build_pmns()
    energies_GeV = torch.logspace(-1, 3, 9, device=DEVICE, dtype=DTYPE)
    theta_values = torch.linspace(0.0, 88.0, 7, device=DEVICE, dtype=DTYPE)
    delta = torch.empty(
        (theta_values.numel(), energies_GeV.numel()),
        device=DEVICE,
        dtype=DTYPE,
    )

    for i_theta, theta_i in enumerate(theta_values):
        for i_E, E_GeV_i in enumerate(energies_GeV):
            kwargs = dict(
                pmns=pmns,
                DeltamSq21=DM21_EV2,
                DeltamSq3l=DM3L_EV2,
                E_MeV=1000.0 * E_GeV_i,
                h_km=H_KM,
                theta_deg=theta_i,
                depth_km=DEPTH_KM,
                antinu=False,
                n_steps=32,
                device=DEVICE,
                dtype=DTYPE,
            )
            S_vac, _ = atmospheric_evolution_operator(matter=False, **kwargs)
            S_mat, _ = atmospheric_evolution_operator(matter=True, **kwargs)
            delta[i_theta, i_E] = torch.max(
                torch.abs(probability_matrix(S_mat) - probability_matrix(S_vac))
            )

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    mesh = ax.pcolormesh(
        energies_GeV.detach().cpu(),
        theta_values.detach().cpu(),
        delta.detach().cpu(),
        shading="auto",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel("Surface zenith angle theta [deg]")
    ax.set_title(r"Max flavour-probability difference: matter minus vacuum")
    fig.colorbar(mesh, ax=ax, label=r"$\max_{\alpha,\beta}|P_{mat}-P_{vac}|$")
    fig.tight_layout()
    
    return finish_plot(fig, "test7_matter_vacuum_difference.png", savefig=savefig)


def test_visualization_outputs(savefig=False):
    paths = [
        plot_vacuum_convergence(savefig=savefig),
        plot_probability_vs_energy(savefig=savefig),
        plot_probability_vs_theta(savefig=savefig),
        plot_matter_vacuum_probability_difference(savefig=savefig),
    ]

    print("\nVisualization outputs")
    for path in paths:
        print(path)
        assert_true(path.exists(), f"Expected plot output missing: {path}")
        assert_true(path.stat().st_size > 0, f"Plot output is empty: {path}")


def run_atmosphere_simulation_tests(verbose_traceback=False):
    tests = [
        test_zero_height_identity,
        test_vacuum_limit_matches_exact_exponential,
        test_matter_unitarity_and_norm_conservation,
        test_probability_matrix_is_physical,
        test_vacuum_convergence_is_stable,
        test_matter_effect_is_small_but_finite_for_atmosphere,
        test_visualization_outputs,
    ]

    return run_test_suite(
        tests,
        suite_name="atmosphere SIMULATION VALIDATION tests",
        verbose_traceback=verbose_traceback,
    )


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    run_atmosphere_simulation_tests(verbose_traceback=True)

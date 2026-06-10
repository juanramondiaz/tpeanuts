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
State evolution test through a toy matter medium.

Run with:

    python tests/core/test_state_evolution_medium_plot.py

This script:

    1. Defines an initial flavour state.
    2. Defines a toy polynomial matter medium.
    3. Evolves the state segment by segment.
    4. Stores probabilities P_e, P_mu, P_tau along the path.
    5. Plots:
        - flavour probabilities
        - norm conservation
        - medium density profile
"""



from __future__ import annotations

import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt

from tpeanuts.core.pmns import PMNS
from tpeanuts.core.segment_evolution import perturbative_segment_evolutor


from tpeanuts.util.test_utils import (
    ATOL, RTOL, printoptions,
    banner, section, print_ok, print_fail, step,
    max_abs_error, assert_close, assert_true,
    default_inputs, build_pmns, run_test_suite
    )
printoptions()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REAL_DTYPE = torch.float64
COMPLEX_DTYPE = torch.complex128

DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_TEST_ROOT = Path(OUTPUT_ROOT / "test")
OUTPUT_DIR = Path(OUTPUT_TEST_ROOT / "core" / Path(__file__).stem)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Utilities
# ============================================================


def flavour_state(
    flavour: str,
    *,
    device=DEVICE,
    dtype=COMPLEX_DTYPE,
) -> torch.Tensor:
    psi = torch.zeros(3, dtype=dtype, device=device)

    if flavour in ["e", "nue", "electron"]:
        psi[0] = 1.0
    elif flavour in ["mu", "numu", "muon"]:
        psi[1] = 1.0
    elif flavour in ["tau", "nutau"]:
        psi[2] = 1.0
    else:
        raise ValueError(f"Unknown flavour: {flavour}")

    return psi


def state_probabilities(psi: torch.Tensor) -> torch.Tensor:
    return torch.abs(psi) ** 2


def state_norm(psi: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(psi) ** 2)


def toy_density_coefficients(
    x_mid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a = 1.0 + 0.4 * torch.exp(-4.0 * x_mid)
    b = 0.15 * torch.ones_like(x_mid)
    c = 0.03 * torch.ones_like(x_mid)

    return a, b, c


# ============================================================
# Main test
# ============================================================

def test_state_evolution_through_toy_medium(savefig=False):

    banner("STATE EVOLUTION THROUGH TOY MATTER MEDIUM")

    # --------------------------------------------------------
    # Oscillation parameters
    # --------------------------------------------------------

    pmns = PMNS(
        theta12=torch.tensor(0.59, dtype=REAL_DTYPE, device=DEVICE),
        theta13=torch.tensor(0.15, dtype=REAL_DTYPE, device=DEVICE),
        theta23=torch.tensor(0.78, dtype=REAL_DTYPE, device=DEVICE),
        delta=torch.tensor(1.20, dtype=REAL_DTYPE, device=DEVICE),
        device=DEVICE,
        real_dtype=REAL_DTYPE,
    )

    DeltamSq21 = torch.tensor(7.42e-5, dtype=REAL_DTYPE, device=DEVICE)
    DeltamSq3l = torch.tensor(2.517e-3, dtype=REAL_DTYPE, device=DEVICE)
    E_MeV = torch.tensor(1000.0, dtype=REAL_DTYPE, device=DEVICE)

    # --------------------------------------------------------
    # Path discretization
    # --------------------------------------------------------

    x_start = torch.tensor(0.0, dtype=REAL_DTYPE, device=DEVICE)
    x_end = torch.tensor(1.0, dtype=REAL_DTYPE, device=DEVICE)

    n_segments = 200

    x_grid = torch.linspace(
        x_start,
        x_end,
        n_segments + 1,
        dtype=REAL_DTYPE,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    psi = flavour_state(
        "mu",
        device=DEVICE,
        dtype=COMPLEX_DTYPE,
    )

    print("Initial state psi:")
    print(psi)

    print("Initial probabilities:")
    print(state_probabilities(psi))

    print(f"Initial norm = {state_norm(psi).item():.12e}")

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    x_values = []
    prob_values = []
    norm_values = []
    density_values = []

    x_values.append(x_grid[0].detach().cpu())
    prob_values.append(state_probabilities(psi).detach().cpu())
    norm_values.append(state_norm(psi).detach().cpu())

    a0, b0, c0 = toy_density_coefficients(x_grid[0])
    density_values.append((a0 + b0 * x_grid[0]**2 + c0 * x_grid[0]**4).detach().cpu())

    # --------------------------------------------------------
    # Segment-by-segment evolution
    # --------------------------------------------------------

    for i in range(n_segments):

        x1 = x_grid[i]
        x2 = x_grid[i + 1]
        x_mid = 0.5 * (x1 + x2)

        a, b, c = toy_density_coefficients(x_mid)

        U_seg = perturbative_segment_evolutor(
            DeltamSq21=DeltamSq21,
            DeltamSq3l=DeltamSq3l,
            pmns=pmns,
            E_MeV=E_MeV,
            x1=x1,
            x2=x2,
            a=a,
            b=b,
            c=c,
            antinu=False,
            debug=False,
        )

        psi = U_seg @ psi

        probs = state_probabilities(psi)
        norm = state_norm(psi)

        density = a + b * x2**2 + c * x2**4

        x_values.append(x2.detach().cpu())
        prob_values.append(probs.detach().cpu())
        norm_values.append(norm.detach().cpu())
        density_values.append(density.detach().cpu())

        if i % 50 == 0:
            print(f"Segment {i:04d}")
            print(f"  x1 = {x1.item():.6e}, x2 = {x2.item():.6e}")
            print(f"  density = {density.item():.6e}")
            print(f"  probabilities = {probs.detach().cpu().numpy()}")
            print(f"  norm = {norm.item():.12e}")

    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    x_values = torch.stack(x_values)
    prob_values = torch.stack(prob_values)
    norm_values = torch.stack(norm_values)
    density_values = torch.stack(density_values)

    max_norm_error = torch.max(torch.abs(norm_values - 1.0)).item()

    print("\nFinal state psi:")
    print(psi)

    print("Final probabilities:")
    print(prob_values[-1])

    print(f"Final norm = {norm_values[-1].item():.12e}")
    print(f"max|norm-1| = {max_norm_error:.6e}")

    assert torch.isfinite(prob_values).all()
    assert torch.isfinite(norm_values).all()
    assert max_norm_error < 1.0e-2

    # --------------------------------------------------------
    # Plot 1: flavour probabilities
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(x_values.numpy(), prob_values[:, 0].numpy(), label=r"$P_e$")
    plt.plot(x_values.numpy(), prob_values[:, 1].numpy(), label=r"$P_\mu$")
    plt.plot(x_values.numpy(), prob_values[:, 2].numpy(), label=r"$P_\tau$")

    plt.xlabel(r"Path coordinate $x$")
    plt.ylabel(r"Flavour probability")
    plt.title("Flavour-state evolution through toy matter medium")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path_prob = os.path.join(OUTPUT_DIR, "state_evolution_probabilities.png")
    if savefig:
        plt.savefig(path_prob, dpi=200)
    plt.show()

    # --------------------------------------------------------
    # Plot 2: norm conservation
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(x_values.numpy(), (norm_values.numpy() - 1))

    plt.xlabel(r"Path coordinate $x$")
    plt.ylabel(r"$||\psi(x)||^2 - 1$")
    plt.title("Norm conservation during state evolution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path_norm = os.path.join(OUTPUT_DIR, "state_evolution_norm.png")
    if savefig:
        plt.savefig(path_norm, dpi=200)
    plt.show()

    # --------------------------------------------------------
    # Plot 3: density profile
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(x_values.numpy(), density_values.numpy())

    plt.xlabel(r"Path coordinate $x$")
    plt.ylabel(r"Toy electron density $n_e(x)$")
    plt.title("Toy matter density profile")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path_density = os.path.join(OUTPUT_DIR, "toy_density_profile.png")
    if savefig:
        plt.savefig(path_density, dpi=200)
    plt.show()

    if savefig:
        print("\nSaved plots:")
        print(f"  {path_prob}")
        print(f"  {path_norm}")
        print(f"  {path_density}")


# ============================================================
# Main
# ============================================================

# ============================================================
# Runner
# ============================================================

def run_test10_core_state_evolution_tests(verbose_traceback=False):
    tests = [
        test_state_evolution_through_toy_medium,
    ]
    return run_test_suite(
        tests,
        suite_name="core TEST10 CORE STATE EVOLUTION tests",
        verbose_traceback=verbose_traceback,
    )


if __name__ == "__main__":
    run_test10_core_state_evolution_tests(verbose_traceback=True)

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
Full atmospheric neutrino propagation test - Torch version.

This script:

    1. Loads Phi_alpha(E, h) for a fixed theta.
    2. Builds the input flavour-flux vector.
    3. Propagates atmosphere + earth using tpeanuts.
    4. Integrates over production height.
    5. Compares initial and detector fluxes.
"""



import torch
import matplotlib.pyplot as plt
from pathlib import Path
import os

THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parents[3]
DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))

OUTPUT_DATA_ROOT = Path(OUTPUT_ROOT / "data")
OUTPUT_DATA_MCEQ = Path(OUTPUT_DATA_ROOT / "atmosphere" / "mceq" / "test")

from tpeanuts.util.test_utils import assert_true, run_test_suite

from tpeanuts.io.io_earth import load_earth_density_from_csv
from tpeanuts.io.io_atmosphere import load_phi_E_h_flavours_for_theta

from tpeanuts.atmosphere.flux import (
    build_probability_matrix,
    propagate_flux_vector,
    propagate_flux_E_h,
    integrate_flux_over_height,
    integrate_detector_flux_over_height,
)
from tpeanuts.core.pmns import PMNS

def test_atmosphere_flux_diagnostics():
    # ============================================================
    # Config
    # ============================================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    print(f"Using device: {device}")

    # ============================================================
    # Oscillation parameters
    # ============================================================

    theta12 = torch.deg2rad(torch.tensor(33.44, device=device, dtype=dtype))
    theta13 = torch.deg2rad(torch.tensor(8.57, device=device, dtype=dtype))
    theta23 = torch.deg2rad(torch.tensor(49.2, device=device, dtype=dtype))
    delta = torch.deg2rad(torch.tensor(195.0, device=device, dtype=dtype))

    DeltamSq21 = torch.tensor(7.42e-5, device=device, dtype=dtype)
    DeltamSq3l = torch.tensor(2.517e-3, device=device, dtype=dtype)

    E_MeV = torch.tensor(1000.0, device=device, dtype=dtype)
    h_km = torch.tensor(20.0, device=device, dtype=dtype)
    theta_deg = torch.tensor(45.0, device=device, dtype=dtype)
    detector_depth_m = torch.tensor(1400.0, device=device, dtype=dtype)


    data_DIR = OUTPUT_DATA_MCEQ
    EARTH_DENSITY_FILE = PACKAGE_DIR / "data" / "density" / "earth_density.csv"

    THETA_DEG = 18.195
    DETECTOR_DEPTH_M = 1000.0

    ANTINU = False

    atmosphere_MATTER = True
    atmosphere_N_STEPS = 40

    REUNITARIZE_earth = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    cdtype = torch.complex128

    print(f"Using device: {device}")

    pmns = PMNS(theta12, theta13, theta23,delta, device=device, real_dtype=dtype)

    density = load_earth_density_from_csv(
        str(EARTH_DENSITY_FILE),
        device=device,
        dtype=dtype,
    )

    print("earth density loaded.")
    print("rj shape:", density.rj.shape)


    # ============================================================
    # 1. Probability matrix
    # ============================================================

    print("\n[1] Probability matrix")

    P, S_total = build_probability_matrix(
        pmns=pmns,
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=E_MeV,
        h_km=h_km,
        theta_deg=theta_deg,
        detector_depth_m=detector_depth_m,
        density=density,
        antinu=False,
        atmosphere_matter=False,
        atmosphere_n_steps=60,
        device=device,
        dtype=dtype,
    )

    print("P.shape       =", P.shape)
    print("S_total.shape =", S_total.shape)

    assert_true(P.shape[-2:] == (3, 3)
    )
    assert_true(S_total.shape[-2:] == (3, 3)
    )

    col_sums = torch.sum(P, dim=-2)
    print("Column sums:", col_sums.detach().cpu())

    prob_error = torch.max(torch.abs(col_sums - 1.0))
    print(f"Probability column error = {prob_error.item():.3e}")

    assert_true(prob_error < 1e-3
    )


    # ============================================================
    # 2. flux vector propagation
    # ============================================================

    print("\n[2] flux vector propagation")

    flux_in = torch.tensor(
        [1.0, 2.0, 0.0],
        device=device,
        dtype=dtype,
    )

    flux_out, P2 = propagate_flux_vector(
        flux_flavour=flux_in,
        pmns=pmns,
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=E_MeV,
        h_km=h_km,
        theta_deg=theta_deg,
        detector_depth_m=detector_depth_m,
        density=density,
        antinu=False,
        atmosphere_matter=False,
        atmosphere_n_steps=300,
        device=device,
        dtype=dtype,
    )

    flux_direct = torch.matmul(P, flux_in[..., None]).squeeze(-1)

    print("flux_in     =", flux_in.detach().cpu())
    print("flux_out    =", flux_out.detach().cpu())
    print("flux_direct =", flux_direct.detach().cpu())

    flux_error = torch.linalg.norm(flux_out - flux_direct)
    print(f"flux consistency error = {flux_error.item():.3e}")

    assert_true(flux_error < 1e-10
    )


    # ============================================================
    # 3. Artificial Phi(E,h) grid
    # ============================================================

    print("\n[3] Grid flux propagation")

    E_grid_GeV = torch.logspace(
        -1,
        1,
        4,
        device=device,
        dtype=dtype,
    )

    h_grid_km = torch.linspace(
        0.0,
        50.0,
        5,
        device=device,
        dtype=dtype,
    )

    EE, HH = torch.meshgrid(E_grid_GeV, h_grid_km, indexing="ij")

    phi_E_h_flavours = {
        "nue": torch.exp(-HH / 15.0) * E_grid_GeV[:, None] ** (-2.0),
        "numu": 2.0 * torch.exp(-HH / 18.0) * E_grid_GeV[:, None] ** (-2.2),
        "nutau": torch.zeros_like(EE),
    }

    phi_detector, P_cache = propagate_flux_E_h(
        E_grid_GeV=E_grid_GeV,
        h_grid_km=h_grid_km,
        phi_E_h_flavours=phi_E_h_flavours,
        theta_deg=theta_deg,
        pmns=pmns,
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        detector_depth_m=detector_depth_m,
        density=density,
        antinu=False,
        atmosphere_matter=False,
        atmosphere_n_steps=40,
        store_probabilities=True,
        verbose=False,
        device=device,
        dtype=dtype,
    )

    for key in ["nue", "numu", "nutau"]:
        print(key, phi_detector[key].shape)
        assert phi_detector[key].shape == (E_grid_GeV.numel(), h_grid_km.numel())
        assert torch.all(torch.isfinite(phi_detector[key]))

    assert_true(P_cache is not None
    )
    assert_true(len(P_cache) == E_grid_GeV.numel() * h_grid_km.numel()
    )


    # ============================================================
    # 4. Height integration
    # ============================================================

    print("\n[4] Height integration")

    phi_prod_E = {
        key: integrate_flux_over_height(
            h_grid_km=h_grid_km,
            phi_E_h=value,
            device=device,
            dtype=dtype,
        )
        for key, value in phi_E_h_flavours.items()
    }

    phi_det_E = integrate_detector_flux_over_height(
        h_grid_km=h_grid_km,
        phi_detector=phi_detector,
        device=device,
        dtype=dtype,
    )

    for key in ["nue", "numu", "nutau"]:
        print(f"{key} integrated shape:", phi_det_E[key].shape)
        assert phi_det_E[key].shape == (E_grid_GeV.numel(),)
        assert torch.all(torch.isfinite(phi_det_E[key]))


    # ============================================================
    # 5. flux conservation check
    # ============================================================

    print("\n[5] Total flux conservation check")

    prod_total = phi_prod_E["nue"] + phi_prod_E["numu"] + phi_prod_E["nutau"]
    det_total = phi_det_E["nue"] + phi_det_E["numu"] + phi_det_E["nutau"]

    relative_error = torch.max(
        torch.abs(det_total - prod_total) / torch.clamp(prod_total, min=1e-30)
    )

    print(f"Max relative total-flux error = {relative_error.item():.3e}")

    # If the earth evolutor is not perfectly unitary, use a realistic tolerance.
    assert_true(relative_error < 1e-3
    )


    # ============================================================
    # 6. Plot
    # ============================================================

    print("\n[6] Plot integrated fluxes")

    E_np = E_grid_GeV.detach().cpu().numpy()

    plt.figure(figsize=(8, 5))

    for key, label in [
        ("nue", r"$\nu_e$"),
        ("numu", r"$\nu_\mu$"),
        ("nutau", r"$\nu_\tau$"),
    ]:
        plt.loglog(
            E_np,
            phi_prod_E[key].detach().cpu().numpy(),
            "--",
            label=label + " production",
        )

        plt.loglog(
            E_np,
            phi_det_E[key].detach().cpu().numpy(),
            label=label + " detector",
        )

    plt.xlabel(r"$E$ [GeV]")
    plt.ylabel(r"Height-integrated flux [arb. units]")
    plt.title(r"flux propagation test")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()


    print("\nAll flux propagation tests passed successfully.")

    # ============================================================
    # ============================================================
    # mceq Propagation flux Diagnostics
    # ============================================================
    # ============================================================

    # ============================================================
    # Load fluxes
    # ============================================================

    E_grid, h_grid_km, phi_E_h_flavours, metadata = load_phi_E_h_flavours_for_theta(
        data_dir=str(data_DIR),
        theta_deg=THETA_DEG,
        theta_tolerance_deg = 1e-2,
        required_flavours=("nue", "numu", "nutau"),
        verbose=False,
        device=device,
        dtype=dtype,
    )

    E_grid = E_grid[:4]
    h_grid_km = h_grid_km[:5]
    phi_E_h_flavours = {
        flavour: values[: E_grid.numel(), : h_grid_km.numel()]
        for flavour, values in phi_E_h_flavours.items()
    }

    print("\nLoaded flux grids:")
    print("E_grid shape    =", E_grid.shape)
    print("h_grid_km shape =", h_grid_km.shape)

    for flavour in ["nue", "numu", "nutau"]:
        print(
            flavour,
            phi_E_h_flavours[flavour].shape,
            phi_E_h_flavours[flavour].device,
        )

    # ============================================================
    # Propagate fluxes
    # ============================================================

    phi_detector, P_cache = propagate_flux_E_h(
        E_grid_GeV=E_grid,
        h_grid_km=h_grid_km,
        phi_E_h_flavours=phi_E_h_flavours,
        theta_deg=torch.tensor(THETA_DEG, device=device, dtype=dtype),
        pmns=pmns,
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        detector_depth_m=torch.tensor(DETECTOR_DEPTH_M, device=device, dtype=dtype),
        density=density,
        antinu=ANTINU,
        atmosphere_matter=atmosphere_MATTER,
        atmosphere_n_steps=atmosphere_N_STEPS,
        store_probabilities=False,
        reunitarize_earth=REUNITARIZE_earth,
        verbose=False,
        device=device,
        dtype=dtype,
    )


    # ============================================================
    # Integrate over height
    # ============================================================

    phi_initial_E = {
        flavour: integrate_flux_over_height(
            h_grid_km=h_grid_km,
            phi_E_h=phi_E_h,
            device=device,
            dtype=dtype,
        )
        for flavour, phi_E_h in phi_E_h_flavours.items()
    }

    phi_detector_E = integrate_detector_flux_over_height(
        h_grid_km=h_grid_km,
        phi_detector=phi_detector,
        device=device,
        dtype=dtype,
    )


    # ============================================================
    # Numerical diagnostics
    # ============================================================

    print("\nIntegrated flux diagnostics:")

    for flavour in ["nue", "numu", "nutau"]:
        init_sum = torch.trapezoid(phi_initial_E[flavour], E_grid)
        det_sum = torch.trapezoid(phi_detector_E[flavour], E_grid)

        print(
            f"{flavour:5s} | "
            f"init integral = {init_sum.item():.6e} | "
            f"det integral = {det_sum.item():.6e}"
        )

    total_initial_E = (
        phi_initial_E["nue"]
        + phi_initial_E["numu"]
        + phi_initial_E["nutau"]
    )

    total_detector_E = (
        phi_detector_E["nue"]
        + phi_detector_E["numu"]
        + phi_detector_E["nutau"]
    )

    total_relative_error = torch.max(
        torch.abs(total_detector_E - total_initial_E)
        / torch.clamp(total_initial_E, min=1e-30)
    )

    print(
        "\nMax relative total-flux difference "
        f"detector vs initial = {total_relative_error.item():.3e}"
    )


    # ============================================================
    # Plot initial vs propagated flux
    # ============================================================

    E_np = E_grid.detach().cpu().numpy()

    plt.figure(figsize=(9, 6))

    for flavour in ["nue", "numu", "nutau"]:

        plt.loglog(
            E_np,
            phi_initial_E[flavour].detach().cpu().numpy(),
            linestyle="--",
            linewidth=2,
            label=rf"{flavour} initial",
        )

        plt.loglog(
            E_np,
            phi_detector_E[flavour].detach().cpu().numpy(),
            linewidth=2,
            label=rf"{flavour} detector",
        )

    plt.xlabel(r"Energy $E$ [GeV]")
    plt.ylabel(r"Height-integrated flux $\Phi_\alpha(E)$")
    plt.title(
        rf"Atmospheric neutrino propagation, "
        rf"$\theta={THETA_DEG:.1f}^\circ$"
    )
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


    # ============================================================
    # Plot detector / initial ratio
    # ============================================================

    plt.figure(figsize=(9, 5))

    for flavour in ["nue", "numu", "nutau"]:

        ratio = torch.where(
            phi_initial_E[flavour] > 0.0,
            phi_detector_E[flavour] / phi_initial_E[flavour],
            torch.zeros_like(phi_detector_E[flavour]),
        )

        plt.semilogx(
            E_np,
            ratio.detach().cpu().numpy(),
            linewidth=2,
            label=flavour,
        )

    plt.xlabel(r"Energy $E$ [GeV]")
    plt.ylabel(
        r"$\Phi_\alpha^{\rm det}(E) / "
        r"\Phi_\alpha^{\rm init}(E)$"
    )
    plt.title(
        rf"Propagation effect, "
        rf"$\theta={THETA_DEG:.1f}^\circ$"
    )
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


    # ============================================================
    # Plot total flux conservation
    # ============================================================

    plt.figure(figsize=(9, 5))

    plt.loglog(
        E_np,
        total_initial_E.detach().cpu().numpy(),
        "--",
        linewidth=2,
        label="Total initial",
    )

    plt.loglog(
        E_np,
        total_detector_E.detach().cpu().numpy(),
        linewidth=2,
        label="Total detector",
    )

    plt.xlabel(r"Energy $E$ [GeV]")
    plt.ylabel(r"Total height-integrated flux")
    plt.title(r"Total flux conservation check")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_atmosphere_tests(verbose_traceback=False):
    return run_test_suite([test_atmosphere_flux_diagnostics], suite_name="atmosphere FLUX tests", verbose_traceback=verbose_traceback)


if __name__ == "__main__":
    run_atmosphere_tests(verbose_traceback=True)


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

import torch
import matplotlib.pyplot as plt
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parents[3]


from tpeanuts.util.test_utils import assert_true, run_test_suite

from tpeanuts.atmosphere.geometry import (
    theta_to_eta,
    atmospheric_path_length,
    underground_path_length,
    total_path_length,
    surface_intersection_angle,
    theta_h_to_eta_and_baseline,
    eta_to_theta, theta_deg_to_rad
)

from tpeanuts.util.torch_util import _default_device

# ============================================================
# Device / dtype
# ============================================================

device = _default_device()
dtype = torch.float64

def test_atmosphere_geometry_diagnostics():
    # ============================================================
    # Test 1: L_atm, L_und, L_total vs h_grid
    # ============================================================

    h_grid = torch.linspace(0.0, 50.0, 100, device=device, dtype=dtype)
    theta_grid_deg = torch.linspace(
        0.0,
        180,
        1000,
        device=device,
        dtype=dtype,
    )
    theta_deg = torch.tensor(45.0, device=device, dtype=dtype)

    h_km = torch.tensor(10.0, device=device, dtype=dtype)
    depth_km = torch.tensor(2.0, device=device, dtype=dtype)

    L_atm = atmospheric_path_length(
        h_km=h_grid,
        theta_deg=theta_deg,
        depth_km=depth_km,
        device=device,
        dtype=dtype,
    )

    L_und = underground_path_length(
        theta_deg=theta_deg,
        depth_km=depth_km,
        device=device,
        dtype=dtype,
    )

    L_total = total_path_length(
        h_km=h_grid,
        theta_deg=theta_deg,
        depth_km=depth_km,
        device=device,
        dtype=dtype,
    )

    eta = theta_to_eta(theta_deg, device=device, dtype=dtype)
    print(f"h [Km]       =  {h_grid[len(h_grid)-1].item():.2f}")
    print(f"Theta [rad]  =  {theta_deg_to_rad(theta_deg).item():.2f}")
    print(f"eta [rad]    =  {eta.item():.2f}")
    print(f"L_atm [km]   =  {L_atm[len(h_grid)-1].item():.2f}")
    print(f"L_und [km]   =  {L_und.item():.2f}")
    print(f"L_total [km] =  {L_total[len(h_grid)-1].item():.2f}")
    geom_error = torch.abs((L_atm[len(h_grid)-1] + L_und) - L_total[len(h_grid)-1])
    print(f"ACCURACY:  |L_atm + L_und - L_total| = {geom_error.item():.3e}")



    # Para plotear en matplotlib pasamos a CPU
    h_np = h_grid.detach().cpu().numpy()
    L_atm_np = L_atm.detach().cpu().numpy()
    L_und_np = torch.broadcast_to(L_und, h_grid.shape).detach().cpu().numpy()
    L_total_np = L_total.detach().cpu().numpy()


    plt.figure(figsize=(8, 5))

    plt.plot(L_atm_np, h_np, label=r"$L_{\rm atm}$")
    plt.plot(L_und_np, h_np, "--", label=r"$L_{\rm und}$")
    plt.plot(L_total_np, h_np, label=r"$L_{\rm total}$")

    plt.xlabel(r"Path length [km]")
    plt.ylabel(r"Production height $h$ [km]")

    plt.title(
        rf"Trajectory lengths for "
        rf"$\theta={theta_deg.item():.1f}^\circ$, "
        rf"depth={depth_km.item():.1f} km"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()



    # ============================================================
    # Test 2:  L_atm, L_und, L_total vs eta_grid
    # ============================================================

    # ------------------------------------------------------------
    # Compute lengths
    # ------------------------------------------------------------

    L_atm = atmospheric_path_length(
        h_km=h_km,
        theta_deg=theta_grid_deg,
        depth_km=depth_km,
        device=device,
        dtype=dtype,
    )

    L_und = underground_path_length(
        theta_deg=theta_grid_deg,
        depth_km=depth_km,
        device=device,
        dtype=dtype,
    )

    L_total = total_path_length(
        h_km=h_km,
        theta_deg=theta_grid_deg,
        depth_km=depth_km,
        device=device,
        dtype=dtype,
    )

    # ------------------------------------------------------------
    # To numpy
    # ------------------------------------------------------------

    theta_np = theta_grid_deg.detach().cpu().numpy()

    L_atm_np = L_atm.detach().cpu().numpy()
    L_und_np = L_und.detach().cpu().numpy()
    L_total_np = L_total.detach().cpu().numpy()

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(theta_np, L_atm_np, label=r"$L_{\rm atm}$", linewidth=2)
    #plt.plot(theta_np, L_und_np, label=r"$L_{\rm und}$", linewidth=2)
    #plt.plot(theta_np, L_total_np, label=r"$L_{\rm total}$", linewidth=2)

    plt.xlabel(r"Zenith angle $\theta$ [deg]")
    plt.ylabel(r"Path length [km]")

    plt.title(
        rf"Atmospheric and underground path lengths "
        rf"for fixed production height $h={h_km.item():.1f}$ km"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()



    # ============================================================
    # Test 3: theta_h_to_eta_and_baseline usando grid de etas
    # ============================================================

    eta_grid = torch.linspace(
        #torch.pi / 2.0, # + torch.deg2rad(torch.tensor(6.0, device=device, dtype=dtype)),
        #torch.pi - torch.deg2rad(torch.tensor(1.0, device=device, dtype=dtype)),
        torch.pi, 0.0, 1000,
        device=device,
        dtype=dtype,
    )

    # Convertimos eta -> theta porque theta_h_to_eta_and_baseline recibe theta_deg
    theta_grid_deg = eta_to_theta(
        eta_grid,
        device=device,
        dtype=dtype,
    )

    # Altura fija
    h0_km = torch.tensor(20.0, device=device, dtype=dtype)

    eta_out, L_total_eta = theta_h_to_eta_and_baseline(
        h_km=h0_km,
        theta_deg=theta_grid_deg,
        depth_km=depth_km,
        device=device,
        dtype=dtype,
    )

    theta_np = theta_grid_deg.detach().cpu().numpy()
    eta_np = eta_out.detach().cpu().numpy()
    L_total_eta_np = L_total_eta.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(theta_np, eta_np)
    axes[0].set_xlabel(r"$\theta$ [deg]")
    axes[0].set_ylabel(r"$\eta$ [rad]")
    axes[0].set_title(r"Conversion $\theta \rightarrow \eta$")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(theta_np, L_total_eta_np)
    axes[1].set_xlabel(r"$\theta$ [deg]")
    axes[1].set_ylabel(r"$L_{\rm total}$ [km]")
    axes[1].set_title(rf"$L_{{\rm total}}(\theta)$ for $h={h0_km.item():.1f}$ km")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ============================================================
    # Test 4: surface intersection angle vs detector angle
    # ============================================================

    # ------------------------------------------------------------
    # Compute surface intersection angles
    # ------------------------------------------------------------

    alpha_surface_deg, s_surface_km = surface_intersection_angle(
        zeta_detector_deg=theta_grid_deg,
        depth_km=depth_km,
        device=device,
        dtype=dtype,
        return_distances=True,
    )

    # ------------------------------------------------------------
    # Move to CPU for plotting
    # ------------------------------------------------------------

    alpha_surface_np = alpha_surface_deg.detach().cpu().numpy()
    s_surface_np = s_surface_km.detach().cpu().numpy()

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ------------------------------------------------------------
    # Left: angle transformation
    # ------------------------------------------------------------

    axes[0].plot(
        theta_np,
        alpha_surface_np,
        linewidth=2,
    )

    axes[0].set_xlabel(r"Detector angle $\alpha_d$ [deg]")
    axes[0].set_ylabel(r"Surface intersection angle $\alpha_s$ [deg]")

    axes[0].set_title(
        r"Surface local angle vs detector local angle"
    )

    axes[0].grid(True, alpha=0.3)

    # ------------------------------------------------------------
    # Right: detector-to-surface distance
    # ------------------------------------------------------------

    axes[1].plot(
        theta_np,
        s_surface_np,
        linewidth=2,
    )

    axes[1].set_xlabel(r"Detector angle $\alpha_d$ [deg]")
    axes[1].set_ylabel(r"Distance to surface crossing [km]")

    axes[1].set_title(
        r"Detector-to-surface distance"
    )

    axes[1].grid(True, alpha=0.3)

    # ------------------------------------------------------------
    # Final layout
    # ------------------------------------------------------------

    plt.tight_layout()
    plt.show()


def run_atmosphere_tests(verbose_traceback=False):
    return run_test_suite([test_atmosphere_geometry_diagnostics], suite_name="atmosphere GEOMETRY tests", verbose_traceback=verbose_traceback)


if __name__ == "__main__":
    run_atmosphere_tests(verbose_traceback=True)

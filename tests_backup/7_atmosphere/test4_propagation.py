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
Test suite for tpeanuts atmospheric propagation.

Checks:
1. Geometry consistency.
2. Unitarity of S_atm.
3. Norm conservation.
4. vacuum numerical propagation vs exact matrix exponential.
5. Convergence with n_steps.
6. Probability Check
7. P_alpha(E,h,theta) vs production height.
8. P_alpha(E,h,theta) vs detector zenith angle.
9. P_alpha(E,h,theta) vs energy for several zenith angles.
"""



import torch
import matplotlib.pyplot as plt
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parents[3]


from tpeanuts.util.test_utils import assert_true, run_test_suite

from tpeanuts.core.pmns import PMNS

from tpeanuts.atmosphere.propagation import (
    atmospheric_evolution_operator,
    propagate_atmosphere,
    atmospheric_hamiltonian,
)

from tpeanuts.atmosphere.geometry import (
    atmospheric_path_length,
    underground_path_length,
    total_path_length,
)

def test_atmosphere_propagation_diagnostics():
    # ============================================================
    # Device
    # ============================================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    cdtype = torch.complex128

    print(f"Using device: {device}")



    # ============================================================
    # Parameters
    # ============================================================

    DeltamSq21 = torch.tensor(7.42e-5, device=device, dtype=dtype)
    DeltamSq3l = torch.tensor(2.517e-3, device=device, dtype=dtype)

    E_MeV = torch.tensor(1000.0, device=device, dtype=dtype)

    h_km = torch.tensor(20.0, device=device, dtype=dtype)
    theta_deg = torch.tensor(45.0, device=device, dtype=dtype)
    depth_km = torch.tensor(1.4, device=device, dtype=dtype)

    nustate_0 = torch.tensor(
        [1.0 + 0j, 0.0 + 0j, 0.0 + 0j],
        device=device,
        dtype=cdtype,
    )


    flavour_labels = [
        r"$P_{\nu_\mu\to\nu_e}$",
        r"$P_{\nu_\mu\to\nu_\mu}$",
        r"$P_{\nu_\mu\to\nu_\tau}$",
    ]

    # ============================================================
    # PMNS
    # ============================================================

    theta12 = torch.deg2rad(torch.tensor(33.44, device=device, dtype=dtype))
    theta13 = torch.deg2rad(torch.tensor(8.57, device=device, dtype=dtype))
    theta23 = torch.deg2rad(torch.tensor(49.2, device=device, dtype=dtype))
    delta = torch.deg2rad(torch.tensor(195.0, device=device, dtype=dtype))
    pmns = PMNS(theta12, theta13, theta23,delta, device=device, real_dtype=dtype)


    nu_surface, S_atm = propagate_atmosphere(
        nustate=nustate_0,
        pmns=pmns,
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=E_MeV,
        h_km=h_km,
        theta_deg=theta_deg,
        antinu=False,
        matter=True,
    )
    print('\n  PROPAGATION TEST\n','='*50)
    print('\t-Height over earth Surface (KM):', h_km.item())
    print('\n\t-Theta (deg)', theta_deg.item())
    print('\n\t-Initial State:\n\n\t', list(nustate_0.detach().cpu().numpy()))
    print('\n\t-earth Surface State:\n\n\t', list(nu_surface.detach().cpu().numpy()))
    print('\n\t-Probs Surface State:\n\n\t', list((torch.abs(nu_surface)**2).detach().cpu().numpy()))
    print()


    # ============================================================
    # 2. Atmospheric vacuum evolution
    # ============================================================

    print("\n[2] vacuum evolution")

    S_atm, x_grid = atmospheric_evolution_operator(
        pmns=pmns,
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=E_MeV,
        h_km=h_km,
        theta_deg=theta_deg,
        depth_km=depth_km,
        antinu=False,
        n_steps=600,
        matter=False,
        device=device,
        dtype=dtype,
    )

    I = torch.eye(3, device=device, dtype=cdtype)
    S_atm = S_atm.squeeze()
    unitarity_error = torch.linalg.norm(S_atm.conj().T @ S_atm - I)

    print(f"Unitarity error ||S†S-I|| = {unitarity_error.item():.3e}")

    assert_true(unitarity_error < 1e-10
    )


    # ============================================================
    # 3. Norm conservation
    # ============================================================

    print("\n[3] Norm conservation")

    nu_surface, S_atm_2 = propagate_atmosphere(
        nustate=nustate_0,
        pmns=pmns,
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=E_MeV,
        h_km=h_km,
        theta_deg=theta_deg,
        depth_km=depth_km,
        antinu=False,
        n_steps=600,
        matter=False,
        device=device,
        dtype=dtype,
    )

    norm_initial = torch.sum(torch.abs(nustate_0) ** 2)
    norm_final = torch.sum(torch.abs(nu_surface) ** 2)
    norm_error = torch.abs(norm_final - norm_initial)

    print(f"Initial norm = {norm_initial.item():.12f}")
    print(f"Final norm   = {norm_final.item():.12f}")
    print(f"Norm error   = {norm_error.item():.3e}")

    assert_true(norm_error < 1e-10
    )


    # ============================================================
    # 4. vacuum exact comparison
    # ============================================================

    print("\n[4] vacuum exact comparison")

    H_vac = atmospheric_hamiltonian(
        pmns=pmns,
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=E_MeV,
        ne_molcm3=torch.zeros((), device=device, dtype=dtype),
        antinu=False,
        device=device,
        dtype=dtype,
    )

    # Use exactly the length integrated by the numerical grid.
    dx_total = torch.sum(x_grid[1:] - x_grid[:-1])

    S_exact = torch.linalg.matrix_exp(
        -1j * H_vac * dx_total.to(dtype=cdtype)
    ).squeeze()

    print("S_atm.shape   =", S_atm.shape)
    print("H_vac.shape   =", H_vac.shape)
    print("S_exact.shape =", S_exact.shape)
    print("dx_total      =", dx_total.item())
    print("x_grid[-1]    =", x_grid[-1].item())

    exact_error = torch.linalg.norm(S_atm - S_exact)

    print(f"vacuum exact error ||S_num-S_exact|| = {exact_error.item():.3e}")

    assert_true(exact_error < 1e-10
    )


    # ============================================================
    # 5. Convergence test
    # ============================================================

    print("\n[5] Convergence with n_steps")

    steps_list = [20, 50, 100, 200, 400, 800]
    errors = []

    for n_steps in steps_list:
        S_n, _ = atmospheric_evolution_operator(
            pmns=pmns,
            DeltamSq21=DeltamSq21,
            DeltamSq3l=DeltamSq3l,
            E_MeV=E_MeV,
            h_km=h_km,
            theta_deg=theta_deg,
            depth_km=depth_km,
            antinu=False,
            n_steps=n_steps,
            matter=False,
            device=device,
            dtype=dtype,
        )

        err = torch.linalg.norm(S_n - S_exact).detach().cpu().item()
        errors.append(err)

        print(f"n_steps = {n_steps:4d} | error = {err:.3e}")


    plt.figure(figsize=(7, 5))
    plt.loglog(steps_list, errors, marker="o")
    plt.xlabel(r"$n_{\rm steps}$")
    plt.ylabel(r"$||S_{\rm num}-S_{\rm exact}||$")
    plt.title("vacuum propagation convergence")
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.show()


    # ============================================================
    # 6. Probability check
    # ============================================================

    print("\n[6] Flavour probability check")

    probs = torch.abs(nu_surface) ** 2
    prob_sum = torch.sum(probs)

    print(f"P_e   = {probs[0].item():.12f}")
    print(f"P_mu  = {probs[1].item():.12f}")
    print(f"P_tau = {probs[2].item():.12f}")
    print(f"Sum   = {prob_sum.item():.12f}")

    assert_true(torch.abs(prob_sum - 1.0) < 1e-10
    )



    # ============================================================
    # Helper
    # ============================================================

    @torch.no_grad()
    def probabilities_from_state(nu):
        return torch.abs(nu) ** 2


    @torch.no_grad()
    def propagate_probabilities(
        E_MeV,
        h_km,
        theta_deg,
        matter=False,
        n_steps=300,
    ):
        nu_surface, S_atm = propagate_atmosphere(
            nustate=nustate_0,
            pmns=pmns,
            DeltamSq21=DeltamSq21,
            DeltamSq3l=DeltamSq3l,
            E_MeV=E_MeV,
            h_km=h_km,
            theta_deg=theta_deg,
            depth_km=depth_km,
            antinu=False,
            n_steps=n_steps,
            matter=matter,
            device=device,
            dtype=dtype,
        )

        probs = probabilities_from_state(nu_surface)

        return probs, S_atm


    # ============================================================
    # 7. Basic geometry + unitarity test
    # ============================================================

    print("\n[7] Basic numerical tests")

    E_test = torch.tensor(1000.0, device=device, dtype=dtype)
    h_test = torch.tensor(20.0, device=device, dtype=dtype)
    theta_test = torch.tensor(45.0, device=device, dtype=dtype)

    L_atm = atmospheric_path_length(
        h_km=h_test,
        theta_deg=theta_test,
        depth_km=depth_km,
        device=device,
        dtype=dtype,
    )

    L_und = underground_path_length(
        theta_deg=theta_test,
        depth_km=depth_km,
        device=device,
        dtype=dtype,
    )

    L_total = total_path_length(
        h_km=h_test,
        theta_deg=theta_test,
        depth_km=depth_km,
        device=device,
        dtype=dtype,
    )

    geom_error = torch.abs(L_atm + L_und - L_total)

    print(f"L_atm   = {L_atm.item():.8f} km")
    print(f"L_und   = {L_und.item():.8f} km")
    print(f"L_total = {L_total.item():.8f} km")
    print(f"Geometry error = {geom_error.item():.3e}")

    assert_true(geom_error < 1e-9
    )


    S_atm, x_grid = atmospheric_evolution_operator(
        pmns=pmns,
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=E_test,
        h_km=h_test,
        theta_deg=theta_test,
        depth_km=depth_km,
        antinu=False,
        n_steps=400,
        matter=False,
        device=device,
        dtype=dtype,
    )

    I = torch.eye(3, device=device, dtype=cdtype)

    SdagS = S_atm.conj().transpose(-2, -1) @ S_atm

    unitarity_error = torch.linalg.norm(
        SdagS - I.expand_as(SdagS)
    )

    print(f"atmosphere unitarity error = {unitarity_error.item():.3e}")

    assert_true(unitarity_error < 1e-10
    )


    nu_surface, _ = propagate_atmosphere(
        nustate=nustate_0,
        pmns=pmns,
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=E_test,
        h_km=h_test,
        theta_deg=theta_test,
        depth_km=depth_km,
        antinu=False,
        n_steps=400,
        matter=False,
        device=device,
        dtype=dtype,
    )

    prob_sum = torch.sum(torch.abs(nu_surface) ** 2)

    print(f"Probability sum = {prob_sum.item():.12f}")

    assert_true(torch.abs(prob_sum - 1.0) < 1e-10
    )


    # ============================================================
    # 8. Oscillations vs height and angle
    # ============================================================

    print("\n[8] Oscillations vs height and angle")

    E_fixed_MeV = torch.tensor(1000.0, device=device, dtype=dtype)
    theta_fixed_deg = torch.tensor(45.0, device=device, dtype=dtype)
    h_fixed_km = torch.tensor(20.0, device=device, dtype=dtype)

    h_grid = torch.linspace(
        1.0,
        80.0,
        12,
        device=device,
        dtype=dtype,
    )

    theta_grid = torch.linspace(
        0.0,
        89.0,
        12,
        device=device,
        dtype=dtype,
    )

    P_vs_h = torch.zeros((h_grid.numel(), 3), device=device, dtype=dtype)

    for i, h_val in enumerate(h_grid):
        probs, _ = propagate_probabilities(
            E_MeV=E_fixed_MeV,
            h_km=h_val,
            theta_deg=theta_fixed_deg,
            matter=False,
            n_steps=250,
        )
        P_vs_h[i] = probs


    P_vs_theta = torch.zeros((theta_grid.numel(), 3), device=device, dtype=dtype)

    for i, th_val in enumerate(theta_grid):
        probs, _ = propagate_probabilities(
            E_MeV=E_fixed_MeV,
            h_km=h_fixed_km,
            theta_deg=th_val,
            matter=False,
            n_steps=250,
        )
        P_vs_theta[i] = probs


    h_np = h_grid.detach().cpu().numpy()
    theta_np = theta_grid.detach().cpu().numpy()
    P_vs_h_np = P_vs_h.detach().cpu().numpy()
    P_vs_theta_np = P_vs_theta.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for a in range(3):
        axes[0].plot(
            h_np,
            P_vs_h_np[:, a],
            linewidth=2,
            label=flavour_labels[a],
        )

    axes[0].set_xlabel(r"Production height $h$ [km]")
    axes[0].set_ylabel(r"Probability")
    axes[0].set_title(
        rf"Atmospheric oscillations vs height, "
        rf"$E={E_fixed_MeV.item()/1000:.1f}$ GeV, "
        rf"$\theta={theta_fixed_deg.item():.1f}^\circ$"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for a in range(3):
        axes[1].plot(
            theta_np,
            P_vs_theta_np[:, a],
            linewidth=2,
            label=flavour_labels[a],
        )

    axes[1].set_xlabel(r"Detector zenith angle $\theta$ [deg]")
    axes[1].set_ylabel(r"Probability")
    axes[1].set_title(
        rf"Atmospheric oscillations vs angle, "
        rf"$E={E_fixed_MeV.item()/1000:.1f}$ GeV, "
        rf"$h={h_fixed_km.item():.1f}$ km"
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


    # ============================================================
    # 9. Oscillations vs energy for several angles
    # ============================================================

    print("\n[9] Oscillations vs energy")

    E_grid_GeV = torch.logspace(
        -1.0,
        2.0,
        12,
        device=device,
        dtype=dtype,
    )

    theta_values_deg = [
        0.0,
        30.0,
        60.0,
        85.0,
    ]

    h_energy_km = torch.tensor(20.0, device=device, dtype=dtype)

    # Guardamos principalmente supervivencia nu_mu -> nu_mu
    P_mumu_vs_E = {}

    for theta_val in theta_values_deg:

        theta_t = torch.tensor(theta_val, device=device, dtype=dtype)
        P_arr = torch.zeros(E_grid_GeV.numel(), device=device, dtype=dtype)

        print(f"  theta = {theta_val:.1f} deg")

        for i, E_GeV in enumerate(E_grid_GeV):

            probs, _ = propagate_probabilities(
                E_MeV=E_GeV * 1.0e3,
                h_km=h_energy_km,
                theta_deg=theta_t,
                matter=False,
                n_steps=250,
            )
        
            # initial state is nu_mu, so index 1 is P_mu_mu
            P_arr[i] = probs[1]

        P_mumu_vs_E[theta_val] = P_arr


    E_np = E_grid_GeV.detach().cpu().numpy()

    plt.figure(figsize=(9, 5))

    for theta_val, P_arr in P_mumu_vs_E.items():

        plt.semilogx(
            E_np,
            P_arr.detach().cpu().numpy(),
            linewidth=2,
            label=rf"$\theta={theta_val:.0f}^\circ$",
        )

    plt.xlabel(r"Energy $E$ [GeV]")
    plt.ylabel(r"$P_{\nu_\mu\to\nu_\mu}$")
    plt.title(
        rf"Atmospheric $\nu_\mu$ survival probability vs energy, "
        rf"$h={h_energy_km.item():.1f}$ km"
    )
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()


    # ============================================================
    # 10. Optional: matter vs vacuum comparison
    # ============================================================

    print("\n[10] Matter vs vacuum comparison")

    theta_matter_deg = torch.tensor(85.0, device=device, dtype=dtype)
    h_matter_km = torch.tensor(80.0, device=device, dtype=dtype)

    P_vac = torch.zeros(E_grid_GeV.numel(), device=device, dtype=dtype)
    P_mat = torch.zeros(E_grid_GeV.numel(), device=device, dtype=dtype)

    for i, E_GeV in enumerate(E_grid_GeV):

        probs_vac, _ = propagate_probabilities(
            E_MeV=E_GeV * 1.0e3,
            h_km=h_matter_km,
            theta_deg=theta_matter_deg,
            matter=False,
            n_steps=300,
        )

        probs_mat, _ = propagate_probabilities(
            E_MeV=E_GeV * 1.0e3,
            h_km=h_matter_km,
            theta_deg=theta_matter_deg,
            matter=True,
            n_steps=300,
        )
        P_vac[i] = probs_vac[1]
        P_mat[i] = probs_mat[1]


    plt.figure(figsize=(9, 5))

    plt.semilogx(
        E_np,
        P_vac.detach().cpu().numpy(),
        linewidth=2,
        label="vacuum",
    )

    plt.semilogx(
        E_np,
        P_mat.detach().cpu().numpy(),
        linewidth=2,
        label="Atmospheric matter",
    )

    plt.xlabel(r"Energy $E$ [GeV]")
    plt.ylabel(r"$P_{\nu_\mu\to\nu_\mu}$")
    plt.title(
        rf"Matter effect in atmosphere, "
        rf"$h={h_matter_km.item():.1f}$ km, "
        rf"$\theta={theta_matter_deg.item():.1f}^\circ$"
    )
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()


    print("\nAtmospheric propagation test completed successfully.")


def run_atmosphere_tests(verbose_traceback=False):
    return run_test_suite([test_atmosphere_propagation_diagnostics], suite_name="atmosphere PROPAGATION tests", verbose_traceback=verbose_traceback)


if __name__ == "__main__":
    run_atmosphere_tests(verbose_traceback=True)

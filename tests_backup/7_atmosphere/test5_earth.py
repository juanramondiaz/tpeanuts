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
tests for peanuts.atmosphere.earth propagation utilities.

Checks:
1. Shape consistency.
2. Unitarity of earth evolution operator.
3. Norm conservation for coherent states.
4. Probability matrix normalization.
5. Consistency between S @ state and propagate_surface_to_detector.
6. Batching in energy and angle.
"""



import torch
import matplotlib.pyplot as plt
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parents[3]


from tpeanuts.util.test_utils import assert_true, run_test_suite

from tpeanuts.core.pmns import PMNS
from tpeanuts.io.io_earth import load_earth_density_from_csv
from tpeanuts.earth.probabilities import pearth
from tpeanuts.atmosphere.geometry import theta_to_eta

from tpeanuts.atmosphere.earth import (
    earth_evolution_operator,
    propagate_surface_to_detector,
)

def test_atmosphere_earth_diagnostics():
    # ============================================================
    # Config
    # ============================================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    cdtype = torch.complex128

    density_file = PACKAGE_DIR / "data" / "density" / "earth_density.csv"

    print(f"Using device: {device}")


    # ============================================================
    # Load density
    # ============================================================

    density = load_earth_density_from_csv(
        str(density_file),
        device=device,
        dtype=dtype,
    )

    print("earth density loaded.")
    print("rj shape:", density.rj.shape)


    # ============================================================
    # PMNS object
    # ============================================================

    theta12 = torch.deg2rad(torch.tensor(33.44, device=device, dtype=dtype))
    theta13 = torch.deg2rad(torch.tensor(8.57, device=device, dtype=dtype))
    theta23 = torch.deg2rad(torch.tensor(49.2, device=device, dtype=dtype))
    delta = torch.deg2rad(torch.tensor(195.0, device=device, dtype=dtype))
    pmns = PMNS(theta12, theta13, theta23,delta, device=device, real_dtype=dtype)


    # ============================================================
    # Oscillation parameters
    # ============================================================

    DeltamSq21 = torch.tensor(7.42e-5, device=device, dtype=dtype)
    DeltamSq3l = torch.tensor(2.517e-3, device=device, dtype=dtype)

    E_MeV = torch.tensor(1000.0, device=device, dtype=dtype)
    theta_deg = torch.tensor(120.0, device=device, dtype=dtype)

    detector_depth_m = torch.tensor(1400.0, device=device, dtype=dtype)

    nu_e = torch.tensor(
        [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        device=device,
        dtype=cdtype,
    )


    # ============================================================
    # 1. Evolution operator
    # ============================================================

    print("\n[1] earth evolution operator")

    S = earth_evolution_operator(
        pmns=pmns,
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=E_MeV,
        theta_deg=theta_deg,
        detector_depth_m=detector_depth_m,
        density=density,
        antinu=False,
        device=device,
        dtype=dtype,
    )

    print("S.shape =", S.shape)

    assert_true(S.shape[-2:] == (3, 3)
    )


    # ============================================================
    # 2. Unitarity
    # ============================================================
    UNITARITY_TOL = 1e-4
    PROB_TOL = 1e-4
    NORM_TOL = 1e-4

    print("\n[2] Unitarity check")

    I = torch.eye(3, device=device, dtype=cdtype)

    SdagS = S.conj().transpose(-2, -1) @ S
    unitarity_error = torch.linalg.norm(SdagS - I.expand_as(SdagS))

    print(f"||S†S - I|| = {unitarity_error.item():.3e}")

    assert_true(unitarity_error < UNITARITY_TOL
    )

    # ============================================================
    # Diagnostic unitarity checks
    # ============================================================

    print("\n[2b] Detailed unitarity diagnostics")

    I = torch.eye(3, device=device, dtype=cdtype)

    SdagS = S.conj().transpose(-2, -1) @ S
    SSdag = S @ S.conj().transpose(-2, -1)

    err_SdagS_fro = torch.linalg.norm(SdagS - I.expand_as(SdagS))
    err_SSdag_fro = torch.linalg.norm(SSdag - I.expand_as(SSdag))

    err_SdagS_max = torch.max(torch.abs(SdagS - I.expand_as(SdagS)))
    err_SSdag_max = torch.max(torch.abs(SSdag - I.expand_as(SSdag)))

    print(f"Frobenius ||S†S-I|| = {err_SdagS_fro.item():.3e}")
    print(f"Frobenius ||SS†-I|| = {err_SSdag_fro.item():.3e}")
    print(f"Max abs   |S†S-I|   = {err_SdagS_max.item():.3e}")
    print(f"Max abs   |SS†-I|   = {err_SSdag_max.item():.3e}")

    print("S†S =")
    print(SdagS.detach().cpu())

    print("SS† =")
    print(SSdag.detach().cpu())

    print("\n[2c] PMNS unitarity")

    U_pmns = pmns.pmns_matrix().to(device=device, dtype=cdtype)

    UdagU = U_pmns.conj().transpose(-2, -1) @ U_pmns

    pmns_unitarity_error = torch.linalg.norm(
        UdagU - I.expand_as(UdagU)
    )

    print("pmns_matrix shape =", U_pmns.shape)
    print(f"PMNS ||U†U-I|| = {pmns_unitarity_error.item():.3e}")

    # ============================================================
    # 3. Norm conservation
    # ============================================================

    print("\n[3] Norm conservation")

    nu_det = propagate_surface_to_detector(
        nustate_surface=nu_e,
        pmns=pmns,
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=E_MeV,
        theta_deg=theta_deg,
        detector_depth_m=detector_depth_m,
        density=density,
        antinu=False,
        device=device,
        dtype=dtype,
    )

    norm_initial = torch.sum(torch.abs(nu_e) ** 2)
    norm_final = torch.sum(torch.abs(nu_det) ** 2, dim=-1)

    norm_error = torch.max(torch.abs(norm_final - norm_initial))

    print("nu_det.shape =", nu_det.shape)
    print(f"Initial norm = {norm_initial.item():.12f}")
    print(f"Final norm   = {norm_final.detach().cpu()}")
    print(f"Norm error   = {norm_error.item():.3e}")

    assert_true(norm_error < NORM_TOL
    )


    # ============================================================
    # 4. Direct S @ state consistency
    # ============================================================

    print("\n[4] Direct matrix multiplication consistency")

    nu_direct = torch.matmul(
        S,
        nu_e.expand(*S.shape[:-1])[..., None],
    ).squeeze(-1)

    direct_error = torch.linalg.norm(nu_det - nu_direct)

    print(f"||nu_det - S nu|| = {direct_error.item():.3e}")

    assert_true(direct_error < 1e-10
    )


    # ============================================================
    # 5. Probability matrix normalization
    # ============================================================

    print("\n[5] Probability matrix normalization")

    P = torch.abs(S) ** 2

    print("P.shape =", P.shape)

    col_sums = torch.sum(P, dim=-2)
    row_sums = torch.sum(P, dim=-1)

    print("Column sums:", col_sums.detach().cpu())
    print("Row sums:   ", row_sums.detach().cpu())

    prob_norm_error = torch.max(torch.abs(col_sums - 1.0))

    print(f"Max column normalization error = {prob_norm_error.item():.3e}")

    assert_true(prob_norm_error < PROB_TOL
    )


    # ============================================================
    # 6. pearth with theta converted to eta
    # ============================================================

    print("\n[6] Pearth incoherent mass-basis probability")

    mass_weights = torch.tensor(
        [1.0, 0.0, 0.0],
        device=device,
        dtype=dtype,
    )

    p_det = pearth(
        nustate=mass_weights,
        density=density,
        pmns=pmns,
        dm21_eV2=DeltamSq21,
        dm3l_eV2=DeltamSq3l,
        E_MeV=E_MeV,
        eta=theta_to_eta(theta_deg, device=device, dtype=dtype),
        depth_m=detector_depth_m,
        method="analytical",
        antinu=False,
        massbasis=True,
    )

    print("p_det =", p_det.detach().cpu())

    p_sum = torch.sum(p_det, dim=-1)

    print("sum p =", p_sum.detach().cpu())

    assert_true(torch.max(torch.abs(p_sum - 1.0)) < UNITARITY_TOL
    )


    # ============================================================
    # 7. Batching test: energies and angles
    # ============================================================

    print("\n[7] Batching test")

    E_grid = torch.logspace(
        2.0,
        4.0,
        24,
        device=device,
        dtype=dtype,
    )

    theta_grid = torch.linspace(
        0.0,
        180.0,
        37,
        device=device,
        dtype=dtype,
    )

    EE, TT = torch.meshgrid(E_grid, theta_grid, indexing="ij")

    S_batch = earth_evolution_operator(
        pmns=pmns,
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=EE,
        theta_deg=TT,
        detector_depth_m=detector_depth_m,
        density=density,
        antinu=False,
        device=device,
        dtype=dtype,
        reunitarize=True,
    )

    print("S_batch.shape =", S_batch.shape)

    assert_true(S_batch.shape[-2:] == (3, 3)
    )
    assert_true(S_batch.shape[:2] == EE.shape
    )

    SdagS_batch = S_batch.conj().transpose(-2, -1) @ S_batch
    unitarity_batch_error = torch.max(
        torch.linalg.norm(
            SdagS_batch - I.expand_as(SdagS_batch),
            dim=(-2, -1),
        )
    )


    # ============================================================
    # 7b. Locate worst unitarity error in batch
    # ============================================================

    unitarity_map = torch.linalg.norm(
        SdagS_batch - I.expand_as(SdagS_batch),
        dim=(-2, -1),
    )

    max_val = torch.max(unitarity_map)
    max_idx = torch.argmax(unitarity_map)

    i_E, i_theta = torch.unravel_index(
        max_idx,
        unitarity_map.shape,
    )

    print("\n[7b] Worst batch unitarity point")
    print(f"Max error = {max_val.item():.6e}")
    print(f"i_E       = {i_E.item()}")
    print(f"i_theta   = {i_theta.item()}")
    print(f"E_MeV     = {E_grid[i_E].item():.6e}")
    print(f"theta_deg = {theta_grid[i_theta].item():.6f}")

    print("\nSelected unitarity errors:")
    for th in [0, 30, 60, 90, 120, 150, 180]:
        idx = torch.argmin(torch.abs(theta_grid - th))
        err_vs_E = unitarity_map[:, idx]
        print(
            f"theta={theta_grid[idx].item():6.1f} deg | "
            f"min={err_vs_E.min().item():.3e} | "
            f"max={err_vs_E.max().item():.3e}"
        )

    print(f"\nBatch max unitarity error = {unitarity_batch_error.item():.3e}")
    assert_true(unitarity_batch_error < 2e-2
    )


    P_batch = torch.abs(S_batch)**2

    prob_sum = torch.sum(P_batch, dim=-2)

    prob_error = torch.max(torch.abs(prob_sum - 1.0))

    print("Max probability normalization error =", prob_error.item())

    plt.figure(figsize=(8, 5))

    plt.pcolormesh(
        theta_grid.detach().cpu().numpy(),
        E_grid.detach().cpu().numpy(),
        unitarity_map.detach().cpu().numpy(),
        shading="auto",
    )

    plt.yscale("log")
    plt.colorbar(label=r"$||S^\dagger S-I||$")
    plt.xlabel(r"$\theta$ [deg]")
    plt.ylabel(r"$E$ [MeV]")
    plt.title("earth evolutor unitarity error")
    plt.tight_layout()
    plt.show()

    # ============================================================
    # 8. Plot probability vs theta
    # ============================================================

    print("\n[8] Plot P_earth vs theta")

    theta_plot = torch.linspace(
        0.0,
        180.0,
        181,
        device=device,
        dtype=dtype,
    )

    S_theta = earth_evolution_operator(
        pmns=pmns,
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=E_MeV,
        theta_deg=theta_plot,
        detector_depth_m=detector_depth_m,
        density=density,
        antinu=False,
        device=device,
        dtype=dtype,
    )
    P_theta = torch.abs(S_theta) ** 2

    # P[beta, alpha]
    P_ee = P_theta[..., 0, 0]
    P_mue = P_theta[..., 1, 0]
    P_taue = P_theta[..., 2, 0]

    theta_np = theta_plot.detach().cpu().numpy()
    P_ee_np = P_ee.detach().cpu().numpy()
    P_mue_np = P_mue.detach().cpu().numpy()
    P_taue_np = P_taue.detach().cpu().numpy()

    plt.figure(figsize=(8, 5))
    plt.plot(theta_np, P_ee_np, label=r"$P_{e e}$")
    plt.plot(theta_np, P_mue_np, label=r"$P_{\mu e}$")
    plt.plot(theta_np, P_taue_np, label=r"$P_{\tau e}$")

    plt.xlabel(r"Detector zenith angle $\theta$ [deg]")
    plt.ylabel("Probability")
    plt.title(r"earth-only transition probabilities, $E=1$ GeV")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


    print("\nAll earth propagation tests passed successfully.")


def run_atmosphere_tests(verbose_traceback=False):
    return run_test_suite([test_atmosphere_earth_diagnostics], suite_name="atmosphere EARTH tests", verbose_traceback=verbose_traceback)


if __name__ == "__main__":
    run_atmosphere_tests(verbose_traceback=True)

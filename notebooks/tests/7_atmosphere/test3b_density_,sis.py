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

import matplotlib.pyplot as plt
from tpeanuts.atmosphere.density_pymsis import(
    atmospheric_density_profile,
    PyMSISatmosphereConfig
    )
import torch

from tpeanuts.util.test_utils import assert_true, run_test_suite

def test_pymsis_density_profile():
    h_km = torch.linspace(0.0, 120.0, 500)

    config = PyMSISatmosphereConfig(
        date="2026-05-10T12:00",
        lon_deg=-0.38,
        lat_deg=39.47,
        f107=150.0,
        f107a=150.0,
        ap=7.0,
        version=2.1,
        Ye=0.499,
        dtype=torch.float64,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    profile = atmospheric_density_profile(
        alt_km=h_km,
        source="pymsis",
        pymsis_config=config,
    )

    print("\nAtmospheric density profile from pymsis")
    print("======================================")
    print(f"source       : {profile['source']}")
    print(f"device       : {profile['alt_km'].device}")
    print(f"dtype        : {profile['alt_km'].dtype}")
    print(f"MSIS version : {profile['metadata']['version']}")
    print(f"date         : {profile['metadata']['date']}")
    print(f"lon, lat     : {profile['metadata']['lon_deg']}, {profile['metadata']['lat_deg']}")
    print(f"rho(0 km)    : {profile['rho_kg_m3'][0].item():.6e} kg/m^3")
    print(f"ne(0 km)     : {profile['ne_cm3'][0].item():.6e} cm^-3")
    print(f"X(0 km)      : {profile['X_g_cm2'][0].item():.6f} g/cm^2")

    h = profile["alt_km"].detach().cpu().numpy()
    rho = profile["rho_kg_m3"].detach().cpu().numpy()
    ne = profile["ne_cm3"].detach().cpu().numpy()
    X = profile["X_g_cm2"].detach().cpu().numpy()

    plt.figure(figsize=(7, 5))
    plt.semilogy(h, rho)
    plt.xlabel("Altitude h [km]")
    plt.ylabel(r"Mass density $\rho(h)$ [kg m$^{-3}$]")
    plt.title("Atmospheric mass density from pymsis")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.semilogy(h, ne)
    plt.xlabel("Altitude h [km]")
    plt.ylabel(r"Electron density $n_e(h)$ [cm$^{-3}$]")
    plt.title(r"Electron density from $n_e=Y_e\rho/m_p$")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(h, X)
    plt.xlabel("Altitude h [km]")
    plt.ylabel(r"Vertical depth $X(h)$ [g cm$^{-2}$]")
    plt.title(r"Atmospheric overburden $X(h)$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    assert_true(torch.all(torch.isfinite(profile["rho_kg_m3"])))
    assert_true(torch.all(torch.isfinite(profile["ne_cm3"])))
    assert_true(torch.all(torch.isfinite(profile["X_g_cm2"])))


def run_atmosphere_tests(verbose_traceback=False):
    return run_test_suite([test_pymsis_density_profile], suite_name="atmosphere PYMSIS DENSITY tests", verbose_traceback=verbose_traceback)


if __name__ == "__main__":
    run_atmosphere_tests(verbose_traceback=True)

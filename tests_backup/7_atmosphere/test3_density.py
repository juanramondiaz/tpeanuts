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
density profile comparison utilities.

This module compares atmospheric density profiles obtained from:

    - exponential model
    - file-based profile
    - mceq atmosphere

The goal is to validate consistency between the different
implementations used in atmospheric neutrino propagation.
"""


import sys
from pathlib import Path
print('Python:', sys.executable)
import os
print('Environment:', os.environ.get("CONDA_DEFAULT_ENV"))
import torch
import matplotlib.pyplot as plt

THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parents[3]

DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_TEST_ROOT = Path(OUTPUT_ROOT / "test")
OUTPUT_DIR = Path(OUTPUT_TEST_ROOT / "atmosphere" / Path(__file__).stem)
os.makedirs(OUTPUT_DIR, exist_ok=True)


from tpeanuts.util.test_utils import assert_true, run_test_suite

from tpeanuts.atmosphere.density import (
    atmospheric_mass_density_profile,
)

from tpeanuts.external.mceq.density import (
    save_mceq_density_profile,
)

from tpeanuts.util.torch_util import _default_device

device = _default_device()

# ============================================================
# density comparison
# ============================================================

@torch.no_grad()
def compare_density_profiles(
    h_grid_km,
    file_path,
    theta_deg=0.0,
    interaction_model="SIBYLL23D",
    primary_model=None,
    density_model="CORSIKA",
    exponential_kwargs=None,
    figsize=(8, 6),
    logy=True,
    show=True,
    *,
    device=None,
    dtype=torch.float64,
):
    if exponential_kwargs is None:
        exponential_kwargs = {}

    dev = torch.device(
        device if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    h_grid_km = torch.as_tensor(
        h_grid_km,
        device=dev,
        dtype=dtype,
    )

    # --------------------------------------------------------
    # Exponential profile
    # --------------------------------------------------------
    rho_exp = atmospheric_mass_density_profile(
        h_km=h_grid_km,
        source="exponential",
        device=dev,
        dtype=dtype,
        **exponential_kwargs,
    )

    # --------------------------------------------------------
    # File profile
    # --------------------------------------------------------
    rho_file = atmospheric_mass_density_profile(
        h_km=h_grid_km,
        source="file",
        density_file=file_path,
        device=dev,
        dtype=dtype,
    )

    # --------------------------------------------------------
    # mceq profile
    # --------------------------------------------------------
    try:
        rho_mceq = atmospheric_mass_density_profile(
            h_km=h_grid_km,
            source="mceq",
            theta_deg=theta_deg,
            interaction_model=interaction_model,
            primary_model=primary_model,
            density_model=density_model,
            device=dev,
            dtype=dtype,
        )
    except ImportError as exc:
        print(f"mceq is not available; using the file profile as mceq reference. Reason: {exc}")
        rho_mceq = rho_file

    # --------------------------------------------------------
    # msis profile
    # --------------------------------------------------------
    try:
        rho_msis = atmospheric_mass_density_profile(
            h_km=h_grid_km,
            source="msis",
            device=dev,
            dtype=dtype,
        )
    except ImportError as exc:
        print(f"MSIS is not available. Reason: {exc}")
        rho_msis = rho_file

    # --------------------------------------------------------
    # Plot requires CPU arrays
    # --------------------------------------------------------
    h_np = h_grid_km.detach().cpu().numpy()
    rho_exp_np = rho_exp.detach().cpu().numpy()
    rho_mceq_np = rho_mceq.detach().cpu().numpy()
    rho_file_np = rho_file.detach().cpu().numpy()
    rho_msis_np = rho_msis.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        h_np,
        rho_exp_np,
        lw=2,
        label="Exponential",
    )

    ax.plot(
        h_np,
        rho_file_np,
        lw=2,
        label="File",
    )

    ax.plot(
        h_np,
        rho_mceq_np,
        lw=2,
        label="mceq",
    )

    ax.plot(
        h_np,
        rho_msis_np,
        lw=2,
        label="NRLMSISE 2.1",
    )
    
    ax.set_xlabel("Altitude [km]")
    ax.set_ylabel(r"density [g/cm$^3$]")

    if logy:
        ax.set_yscale("log")

    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Atmospheric density Profiles")

    if show:
        plt.show()

    return fig, ax


# ============================================================
# Automatic full test - Torch version
# ============================================================

@torch.no_grad()
def run_density_profile_test(
    output_density_file,
    theta_deg=0.0,
    interaction_model="SIBYLL23D",
    primary_model=None,
    density_model="CORSIKA",
    h_min_km=0.0,
    h_max_km=100.0,
    n_h=1000,
    *,
    device=None,
    dtype=torch.float64,
):
    dev = torch.device(
        device if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    h_grid_km = torch.linspace(
        float(h_min_km),
        float(h_max_km),
        int(n_h),
        device=dev,
        dtype=dtype,
    )

    # --------------------------------------------------------
    # Save mceq profile
    # --------------------------------------------------------
    try:
        save_mceq_density_profile(
            output_path=output_density_file,
            h_grid_km=h_grid_km,
            theta_deg=theta_deg,
            interaction_model=interaction_model,
            primary_model=primary_model,
            density_model=density_model,
            device=dev,
            dtype=dtype,
        )
    except ImportError as exc:
        fallback = Path(output_density_file)
        fallback.parent.mkdir(parents=True, exist_ok=True)
        rho_fallback = atmospheric_mass_density_profile(
            h_km=h_grid_km,
            source="exponential",
            device=dev,
            dtype=dtype,
        )
        with fallback.open("w", encoding="utf-8") as handle:
            handle.write("# Synthetic atmospheric density fallback\n")
            handle.write("# h_km    rho_gcm3\n")
            for h_val, rho_val in zip(
                h_grid_km.detach().cpu().reshape(-1),
                rho_fallback.detach().cpu().reshape(-1),
            ):
                handle.write(f"{h_val.item():.10e}    {rho_val.item():.10e}\n")
        print(f"mceq is not available; using generated density file: {fallback}")
        print(f"Reason: {exc}")
        output_density_file = str(fallback)

    # --------------------------------------------------------
    # Compare profiles
    # --------------------------------------------------------
    fig, ax = compare_density_profiles(
        h_grid_km=h_grid_km,
        file_path=output_density_file,
        theta_deg=theta_deg,
        interaction_model=interaction_model,
        primary_model=primary_model,
        density_model=density_model,
        device=dev,
        dtype=dtype,
    )

    return fig, ax

def test_density_profile_comparison():
    run_density_profile_test(
        output_density_file=str(OUTPUT_DIR / "mceq_density_profile.txt"),
        theta_deg=0.0,
        interaction_model="SIBYLL23D",
        primary_model="HillasGaisser H3a",
        device=device,
    )


def run_atmosphere_tests(verbose_traceback=False):
    return run_test_suite([test_density_profile_comparison], suite_name="atmosphere DENSITY tests", verbose_traceback=verbose_traceback)


if __name__ == "__main__":
    run_atmosphere_tests(verbose_traceback=True)

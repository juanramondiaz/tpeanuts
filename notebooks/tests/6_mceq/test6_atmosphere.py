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
Spyder-compatible tests and visual diagnostics for atmosphere.py.

No pytest required.
"""



import os
from pathlib import Path
import shutil

import torch
import matplotlib.pyplot as plt

from tpeanuts.external.mceq.config import MCEqModelConfig
from tpeanuts.external.mceq.core import init_mceq

from tpeanuts.external.mceq.density import (
    get_density_model,
    get_mass_density_gcm3_from_mceq,
    get_mass_overburden_gcm2_from_mceq,
    atmospheric_mass_density_profile_from_mceq,
    atmospheric_mass_overburden_profile_from_mceq,
    save_mceq_density_profile,
)

from tpeanuts.util.test_utils import (
    assert_true,
    assert_close,
    assert_raises,
    run_test_suite,
)


DEVICE = "cpu"
DTYPE = torch.float64

NOTEBOOK_STEM = Path(__file__).stem
DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_TEST_ROOT = Path(OUTPUT_ROOT / "test")
OUTPUT_DIR = OUTPUT_TEST_ROOT / "mceq" / NOTEBOOK_STEM

MODEL_CONFIG = MCEqModelConfig(
    interaction_model="SIBYLL23D",
    primary_model="HillasGaisser H3a",
    density_model="CORSIKA",
    info=False,
)


def make_h_grid(n_h=200, h_min=0.0, h_max=100.0):
    return torch.linspace(
        h_min,
        h_max,
        n_h,
        device=DEVICE,
        dtype=DTYPE,
    )


# ============================================================
# Unit-like mceq atmosphere tests
# ============================================================

def test_get_density_model():
    mceq = init_mceq(
        theta_deg=0.0,
        config=MODEL_CONFIG,
    )

    density_model = get_density_model(mceq)

    print("density_model:", density_model)

    assert_true(density_model is not None)


def test_get_density_model_rejects_bad_object():
    class BadObject:
        pass

    assert_raises(
        AttributeError,
        get_density_model,
        BadObject(),
    )


def test_single_mass_density_positive():
    mceq = init_mceq(
        theta_deg=0.0,
        config=MODEL_CONFIG,
    )

    rho0 = get_mass_density_gcm3_from_mceq(
        mceq=mceq,
        h_km=0.0,
    )

    rho20 = get_mass_density_gcm3_from_mceq(
        mceq=mceq,
        h_km=20.0,
    )

    print("rho(0 km)  =", rho0)
    print("rho(20 km) =", rho20)

    assert_true(rho0 > 0.0)
    assert_true(rho20 >= 0.0)
    assert_true(rho20 < rho0)


def test_single_mass_overburden_positive_if_available():
    mceq = init_mceq(
        theta_deg=0.0,
        config=MODEL_CONFIG,
    )

    X0 = get_mass_overburden_gcm2_from_mceq(
        mceq=mceq,
        h_km=0.0,
    )

    X20 = get_mass_overburden_gcm2_from_mceq(
        mceq=mceq,
        h_km=20.0,
    )

    print("X(0 km)  =", X0)
    print("X(20 km) =", X20)

    assert_true(X0 > 0.0)
    assert_true(X20 >= 0.0)
    assert_true(X20 < X0)


def test_density_profile_shape_and_positive():
    h = make_h_grid()

    mceq = init_mceq(
        theta_deg=0.0,
        config=MODEL_CONFIG,
    )

    rho = atmospheric_mass_density_profile_from_mceq(
        h_km=h,
        mceq=mceq,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("h shape  :", h.shape)
    print("rho shape:", rho.shape)
    print("rho min  :", float(rho.min().item()))
    print("rho max  :", float(rho.max().item()))

    assert_true(rho.shape == h.shape)
    assert_true(torch.all(rho >= 0.0).item())


def test_density_profile_decreases_approximately():
    h = make_h_grid()

    mceq = init_mceq(
        theta_deg=0.0,
        config=MODEL_CONFIG,
    )

    rho = atmospheric_mass_density_profile_from_mceq(
        h_km=h,
        mceq=mceq,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("rho(0 km)   =", float(rho[0].item()))
    print("rho(100 km) =", float(rho[-1].item()))

    assert_true(float(rho[-1].item()) < float(rho[0].item()))


def test_overburden_profile_shape_and_positive():
    h = make_h_grid()

    mceq = init_mceq(
        theta_deg=0.0,
        config=MODEL_CONFIG,
    )

    X = atmospheric_mass_overburden_profile_from_mceq(
        h_km=h,
        mceq=mceq,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("h shape:", h.shape)
    print("X shape:", X.shape)
    print("X min  :", float(X.min().item()))
    print("X max  :", float(X.max().item()))

    assert_true(X.shape == h.shape)
    assert_true(torch.all(X >= 0.0).item())


def test_overburden_profile_decreases_with_height():
    h = make_h_grid()

    mceq = init_mceq(
        theta_deg=0.0,
        config=MODEL_CONFIG,
    )

    X = atmospheric_mass_overburden_profile_from_mceq(
        h_km=h,
        mceq=mceq,
        device=DEVICE,
        dtype=DTYPE,
    )

    dX = torch.diff(X)

    print("max dX:", float(dX.max().item()))
    print("min dX:", float(dX.min().item()))

    assert_true(torch.all(dX <= 1.0e-10).item())


def test_profile_functions_can_initialize_mceq_internally():
    h = make_h_grid(n_h=50)

    rho = atmospheric_mass_density_profile_from_mceq(
        h_km=h,
        theta_deg=0.0,
        config=MODEL_CONFIG,
        device=DEVICE,
        dtype=DTYPE,
    )

    X = atmospheric_mass_overburden_profile_from_mceq(
        h_km=h,
        theta_deg=0.0,
        config=MODEL_CONFIG,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("rho shape:", rho.shape)
    print("X shape  :", X.shape)

    assert_true(rho.shape == h.shape)
    assert_true(X.shape == h.shape)


def test_density_scalar_input_returns_scalar_shape():
    mceq = init_mceq(
        theta_deg=0.0,
        config=MODEL_CONFIG,
    )

    rho = atmospheric_mass_density_profile_from_mceq(
        h_km=10.0,
        mceq=mceq,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("rho:", rho)
    print("rho shape:", rho.shape)

    assert_true(rho.shape == torch.Size([]))
    assert_true(float(rho.item()) >= 0.0)


def test_save_mceq_density_profile_creates_file():
    tmpdir = OUTPUT_DIR
    output_path = os.path.join(tmpdir, "density_profile.txt")

    try:
        h = make_h_grid(n_h=20)

        saved_path = save_mceq_density_profile(
            output_path=output_path,
            theta_deg=0.0,
            h_grid_km=h,
            config=MODEL_CONFIG,
            overwrite=True,
            device=DEVICE,
            dtype=DTYPE,
        )

        print("saved path:", saved_path)

        assert_true(os.path.exists(saved_path))

        with open(saved_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        print("number of lines:", len(lines))
        print("first data line:", lines[3].strip())

        assert_true(len(lines) > 3)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_save_mceq_density_profile_overwrite_false_raises():
    tmpdir = OUTPUT_DIR
    output_path = os.path.join(tmpdir, "density_profile.txt")

    try:
        tmpdir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("existing file")

        assert_raises(
            FileExistsError,
            save_mceq_density_profile,
            output_path,
            theta_deg=0.0,
            config=MODEL_CONFIG,
            overwrite=False,
            device=DEVICE,
            dtype=DTYPE,
        )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# Visual diagnostics
# ============================================================

def plot_density_profile():
    h = make_h_grid(n_h=500)

    mceq = init_mceq(
        theta_deg=0.0,
        config=MODEL_CONFIG,
    )

    rho = atmospheric_mass_density_profile_from_mceq(
        h_km=h,
        mceq=mceq,
        device=DEVICE,
        dtype=DTYPE,
    )

    plt.figure(figsize=(8, 6))

    plt.semilogx(
        rho.cpu().numpy(),
        h.cpu().numpy(),
        lw=2,
        label=r"$\rho(h)$ from mceq",
    )

    plt.xlabel(r"Mass density $\rho(h)$ [g/cm$^3$]")
    plt.ylabel(r"Altitude $h$ [km]")
    plt.title("Atmospheric density profile from mceq")

    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_overburden_profile():
    h = make_h_grid(n_h=500)

    mceq = init_mceq(
        theta_deg=0.0,
        config=MODEL_CONFIG,
    )

    X = atmospheric_mass_overburden_profile_from_mceq(
        h_km=h,
        mceq=mceq,
        device=DEVICE,
        dtype=DTYPE,
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        X.cpu().numpy(),
        h.cpu().numpy(),
        lw=2,
        label=r"$X(h)$ from mceq",
    )

    plt.xlabel(r"Vertical overburden $X(h)$ [g/cm$^2$]")
    plt.ylabel(r"Altitude $h$ [km]")
    plt.title("Atmospheric overburden profile from mceq")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_density_and_overburden_together():
    h = make_h_grid(n_h=500)

    mceq = init_mceq(
        theta_deg=0.0,
        config=MODEL_CONFIG,
    )

    rho = atmospheric_mass_density_profile_from_mceq(
        h_km=h,
        mceq=mceq,
        device=DEVICE,
        dtype=DTYPE,
    )

    X = atmospheric_mass_overburden_profile_from_mceq(
        h_km=h,
        mceq=mceq,
        device=DEVICE,
        dtype=DTYPE,
    )

    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.semilogx(
        rho.cpu().numpy(),
        h.cpu().numpy(),
        lw=2,
        label=r"$\rho(h)$",
    )

    ax1.set_xlabel(r"density $\rho(h)$ [g/cm$^3$]")
    ax1.set_ylabel(r"Altitude $h$ [km]")

    ax2 = ax1.twiny()

    ax2.plot(
        X.cpu().numpy(),
        h.cpu().numpy(),
        lw=2,
        ls="--",
        label=r"$X(h)$",
    )

    ax2.set_xlabel(r"Overburden $X(h)$ [g/cm$^2$]")

    ax1.grid(True, which="both")

    plt.title("mceq atmosphere: density and overburden")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="best",
    )

    plt.tight_layout()
    plt.show()


def run_atmosphere_visual_tests():
    print("\n" + "=" * 80)
    print("atmosphere VISUAL tests")
    print("=" * 80)

    plot_density_profile()
    plot_overburden_profile()
    plot_density_and_overburden_together()

    print("\nFinished atmosphere visual diagnostics.")


# ============================================================
# Runner
# ============================================================

def run_atmosphere_tests(
    verbose_traceback=False,
    make_plots=True,
):
    tests = [
        test_get_density_model,
        test_get_density_model_rejects_bad_object,
        test_single_mass_density_positive,
        test_single_mass_overburden_positive_if_available,
        test_density_profile_shape_and_positive,
        test_density_profile_decreases_approximately,
        test_overburden_profile_shape_and_positive,
        test_overburden_profile_decreases_with_height,
        test_profile_functions_can_initialize_mceq_internally,
        test_density_scalar_input_returns_scalar_shape,
        test_save_mceq_density_profile_creates_file,
        test_save_mceq_density_profile_overwrite_false_raises,
    ]

    ok = run_test_suite(
        tests,
        suite_name="atmosphere mceq tests",
        verbose_traceback=verbose_traceback,
    )

    if make_plots:
        run_atmosphere_visual_tests()

    return ok


if __name__ == "__main__":
    run_atmosphere_tests(
        verbose_traceback=True,
        make_plots=True,
    )

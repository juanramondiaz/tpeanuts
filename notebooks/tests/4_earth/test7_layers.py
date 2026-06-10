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
Spyder-friendly tests and visual diagnostics for tpeanuts.earth.layers.

This script checks:

    1. density layer output normalization.
    2. Quartic coefficient extraction.
    3. Batched vector gathering with index clamping.
    4. Outermost crossed shell selection.
    5. Flipped shell segment construction.
    6. Visual diagnostics for crossed shells and flipped segments.

Run directly in Spyder or from a terminal:

    python tpeanuts/tests/earth/test9_layers.py
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



from tpeanuts.earth.layers import (  # noqa: E402
    build_flipped_shell_segments,
    extract_quartic_coefficients,
    gather_batch_vector,
    normalize_density_layer_output,
    outermost_crossed_shell,
)
from tpeanuts.util.test_utils import (  # noqa: E402
    assert_true,
    run_test_suite,
)


# ============================================================
# Configuration
# ============================================================

DTYPE = torch.float64
DEVICE = torch.device("cpu")

TESTS_DIR = THIS_FILE.parents[1]

DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_TEST_ROOT = Path(OUTPUT_ROOT / "test")
OUTPUT_DIR = Path(OUTPUT_TEST_ROOT / "earth" / Path(__file__).stem)
os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.set_printoptions(
    precision=10,
    sci_mode=True,
    linewidth=160,
)


# ============================================================
# Local helpers
# ============================================================

def assert_tensor_close(actual, expected, message, atol=1.0e-12, rtol=1.0e-12):
    actual_t = torch.as_tensor(actual)
    expected_t = torch.as_tensor(expected, dtype=actual_t.dtype, device=actual_t.device)

    print(f"Checking: {message}")
    print("  actual shape  :", tuple(actual_t.shape))
    print("  expected shape:", tuple(expected_t.shape))
    print("  max abs diff  :", torch.max(torch.abs(actual_t - expected_t)).item())

    assert_true(
        torch.allclose(actual_t, expected_t, atol=atol, rtol=rtol),
        message,
    )


def build_synthetic_layers():
    xj_all = torch.tensor(
        [
            [0.20, 0.40, 0.60, 0.80, 1.00],
            [0.20, 0.40, 0.60, 0.80, 1.00],
            [0.20, 0.40, 0.60, 0.80, 1.00],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )

    a = torch.tensor(
        [
            [10.0, 11.0, 12.0, 13.0, 14.0],
            [20.0, 21.0, 22.0, 23.0, 24.0],
            [30.0, 31.0, 32.0, 33.0, 34.0],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )

    b = a + 100.0
    c = a + 200.0

    crossed = torch.tensor(
        [
            [True, True, True, False, False],
            [True, False, False, False, False],
            [False, False, False, False, False],
        ],
        dtype=torch.bool,
        device=DEVICE,
    )

    return xj_all, a, b, c, crossed


# ============================================================
# tests
# ============================================================

def test_normalize_keeps_standard_batched_output():
    coeffs_all = torch.arange(2 * 4 * 3, dtype=DTYPE).reshape(2, 4, 3)
    xj_all = torch.arange(2 * 4, dtype=DTYPE).reshape(2, 4)
    crossed = xj_all > 2.0

    coeffs_n, xj_n, crossed_n = normalize_density_layer_output(
        coeffs_all,
        xj_all,
        crossed,
    )

    print("\nStandard batched output:")
    print("coeffs shape :", tuple(coeffs_n.shape))
    print("xj shape     :", tuple(xj_n.shape))
    print("crossed shape:", tuple(crossed_n.shape))

    assert_true(coeffs_n.shape == (2, 4, 3), "Standard coefficient shape must be preserved")
    assert_true(xj_n.shape == (2, 4), "Standard shell-boundary shape must be preserved")
    assert_true(crossed_n.shape == (2, 4), "Standard crossing-mask shape must be preserved")
    assert_tensor_close(coeffs_n, coeffs_all, "Standard coefficients are unchanged")


def test_normalize_squeezes_single_leading_axis():
    coeffs_all = torch.arange(1 * 2 * 4 * 3, dtype=DTYPE).reshape(1, 2, 4, 3)
    xj_all = torch.arange(1 * 2 * 4, dtype=DTYPE).reshape(1, 2, 4)
    crossed = xj_all > 2.0

    coeffs_n, xj_n, crossed_n = normalize_density_layer_output(
        coeffs_all,
        xj_all,
        crossed,
    )

    print("\nLeading singleton normalization:")
    print("input coeffs shape :", tuple(coeffs_all.shape))
    print("output coeffs shape:", tuple(coeffs_n.shape))
    print("output xj shape    :", tuple(xj_n.shape))

    assert_true(coeffs_n.shape == (2, 4, 3), "Leading singleton axis must be removed from coefficients")
    assert_true(xj_n.shape == (2, 4), "Leading singleton axis must be removed from xj")
    assert_true(crossed_n.shape == (2, 4), "Leading singleton axis must be removed from crossed")
    assert_tensor_close(coeffs_n, coeffs_all.squeeze(0), "Squeezed coefficients match expected values")


def test_extract_quartic_coefficients():
    coeffs_all = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 99.0], [4.0, 5.0, 6.0, 99.0]],
            [[7.0, 8.0, 9.0, 99.0], [10.0, 11.0, 12.0, 99.0]],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )

    a, b, c = extract_quartic_coefficients(coeffs_all)

    print("\nExtracted quartic coefficients:")
    print("a:", a)
    print("b:", b)
    print("c:", c)

    assert_tensor_close(a, coeffs_all[..., 0], "Coefficient a is the first column")
    assert_tensor_close(b, coeffs_all[..., 1], "Coefficient b is the second column")
    assert_tensor_close(c, coeffs_all[..., 2], "Coefficient c is the third column")


def test_gather_batch_vector_clamps_indices():
    mat = torch.tensor(
        [
            [10.0, 11.0, 12.0, 13.0],
            [20.0, 21.0, 22.0, 23.0],
            [30.0, 31.0, 32.0, 33.0],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )

    idx = torch.tensor([-5.0, 2.0, 99.0], dtype=DTYPE, device=DEVICE)
    values = gather_batch_vector(mat, idx)
    expected = torch.tensor([10.0, 22.0, 33.0], dtype=DTYPE, device=DEVICE)

    print("\nBatched gather with clamped indices:")
    print("matrix:")
    print(mat)
    print("indices:", idx)
    print("values :", values)

    assert_tensor_close(values, expected, "Gathered values use clamped per-row indices")


def test_outermost_crossed_shell_selection():
    xj_all, a, b, c, crossed = build_synthetic_layers()

    data = outermost_crossed_shell(
        a,
        b,
        c,
        xj_all,
        crossed,
        dtype=DTYPE,
        device=DEVICE,
    )

    print("\nOutermost crossed shell data:")
    for key, value in data.items():
        print(f"{key:16s}:", value)

    assert_tensor_close(data["last_pos"], torch.tensor([2.0, 0.0, -1.0e30], dtype=DTYPE), "Last crossed positions")
    assert_tensor_close(data["second_last_pos"], torch.tensor([1.0, -1.0e30, -1.0e30], dtype=DTYPE), "Second-last crossed positions")
    assert_tensor_close(data["a_o"], torch.tensor([12.0, 20.0, 30.0], dtype=DTYPE), "Selected outermost a coefficients")
    assert_tensor_close(data["b_o"], torch.tensor([112.0, 120.0, 130.0], dtype=DTYPE), "Selected outermost b coefficients")
    assert_tensor_close(data["c_o"], torch.tensor([212.0, 220.0, 230.0], dtype=DTYPE), "Selected outermost c coefficients")
    assert_tensor_close(data["x_start"], torch.tensor([0.40, 0.0, 0.0], dtype=DTYPE), "Segment start coordinates")
    assert_true(torch.equal(data["has_any"], torch.tensor([True, True, False])), "has_any mask is correct")
    assert_true(torch.equal(data["has_two"], torch.tensor([True, False, False])), "has_two mask is correct")


def test_build_flipped_shell_segments():
    xj_all, a, b, c, crossed = build_synthetic_layers()

    segments = build_flipped_shell_segments(
        xj_all,
        a,
        b,
        c,
        crossed,
    )

    print("\nFlipped shell segments:")
    for key, value in segments.items():
        print(f"{key:8s}:")
        print(value)

    expected_x_hi = torch.flip(xj_all, dims=(-1,))
    expected_x_lo = torch.zeros_like(expected_x_hi)
    expected_x_lo[..., :-1] = expected_x_hi[..., 1:]
    expected_x_lo[..., -1] = 0.0

    assert_tensor_close(segments["x_hi"], expected_x_hi, "Upper segment boundaries are flipped")
    assert_tensor_close(segments["x_lo"], expected_x_lo, "Lower segment boundaries are shifted flipped boundaries")
    assert_tensor_close(segments["a2"], torch.flip(a, dims=(-1,)), "a coefficients are flipped")
    assert_tensor_close(segments["b2"], torch.flip(b, dims=(-1,)), "b coefficients are flipped")
    assert_tensor_close(segments["c2"], torch.flip(c, dims=(-1,)), "c coefficients are flipped")
    assert_true(torch.equal(segments["crossed2"], torch.flip(crossed, dims=(-1,))), "crossed mask is flipped")


# ============================================================
# Visualization
# ============================================================

def plot_crossed_shell_map(savefig=False):
    xj_all, a, _, _, crossed = build_synthetic_layers()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    image = ax.imshow(
        crossed.detach().cpu().numpy().astype(float),
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )

    for row in range(xj_all.shape[0]):
        for col in range(xj_all.shape[1]):
            label = f"x={xj_all[row, col].item():.2f}\na={a[row, col].item():.0f}"
            color = "white" if crossed[row, col] else "black"
            ax.text(col, row, label, ha="center", va="center", color=color, fontsize=8)

    ax.set_xlabel("Shell index")
    ax.set_ylabel("Batch trajectory")
    ax.set_title("Synthetic crossed-shell mask and layer coefficients")
    ax.set_xticks(range(xj_all.shape[1]))
    ax.set_yticks(range(xj_all.shape[0]))
    fig.colorbar(image, ax=ax, label="Crossed shell")
    fig.tight_layout()

    path = OUTPUT_DIR / "earth_layers_crossed_shell_map.png"
    if savefig:
        fig.savefig(path, dpi=200)
    plt.show()

    if savefig:
        print(f"\nSaved plot: {path}")


def plot_flipped_segments(savefig=False):
    xj_all, a, b, c, crossed = build_synthetic_layers()
    segments = build_flipped_shell_segments(xj_all, a, b, c, crossed)

    row = 0
    x_hi = segments["x_hi"][row].detach().cpu()
    x_lo = segments["x_lo"][row].detach().cpu()
    crossed2 = segments["crossed2"][row].detach().cpu()

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for idx, (lo, hi, is_crossed) in enumerate(zip(x_lo, x_hi, crossed2)):
        color = "tab:blue" if bool(is_crossed) else "lightgray"
        ax.barh(
            y=idx,
            width=(hi - lo).item(),
            left=lo.item(),
            height=0.6,
            color=color,
            edgecolor="black",
            alpha=0.85,
        )
        ax.text(
            (lo + hi).item() / 2.0,
            idx,
            f"[{lo.item():.2f}, {hi.item():.2f}]",
            ha="center",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Trajectory coordinate x")
    ax.set_ylabel("Flipped segment index")
    ax.set_title("Entry-to-center flipped shell segments")
    ax.set_xlim(0.0, 1.05)
    ax.set_yticks(range(x_hi.numel()))
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    path = OUTPUT_DIR / "earth_layers_flipped_segments.png"
    if savefig:
        fig.savefig(path, dpi=200)
    plt.show()

    if savefig:
        print(f"Saved plot: {path}")


def test_visualization_outputs(savefig=False):
    plot_crossed_shell_map(savefig=savefig)
    plot_flipped_segments(savefig=savefig)

    expected_files = [
        OUTPUT_DIR / "earth_layers_crossed_shell_map.png",
        OUTPUT_DIR / "earth_layers_flipped_segments.png",
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
        test_normalize_keeps_standard_batched_output,
        test_normalize_squeezes_single_leading_axis,
        test_extract_quartic_coefficients,
        test_gather_batch_vector_clamps_indices,
        test_outermost_crossed_shell_selection,
        test_build_flipped_shell_segments,
        test_visualization_outputs,
    ]

    run_test_suite(
        tests,
        suite_name="earth LAYERS tests",
        verbose_traceback=True,
    )

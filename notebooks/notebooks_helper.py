#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared helpers for tpeanuts notebooks."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

FLAVOUR_NAMES = ("nue", "numu", "nutau")
FLAVOUR_LABELS = [r"$\nu_e$", r"$\nu_\mu$", r"$\nu_\tau$"]
FLAVOUR_COLORS = ["C0", "C1", "C2"]
FLAVOUR_INDEX = {name: i for i, name in enumerate(FLAVOUR_NAMES)}

REL_FLOOR = 1.0e-12
ABSOLUTE_THRESHOLD = 1.0e-5
TOL_PPM = 1.0e-6
TOL_PPB = 1.0e-9


def to_numpy(x):
    """Convert a tensor-like object to a CPU NumPy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def abs_rel_delta(candidate, reference, *, floor: float = REL_FLOOR):
    """Return absolute and relative differences against a reference array."""
    cand = to_numpy(candidate).astype(float)
    ref = to_numpy(reference).astype(float)
    abs_delta = np.abs(cand - ref)
    rel_delta = abs_delta / np.maximum(np.abs(ref), floor)
    return abs_delta, rel_delta


def add_tolerance_lines(ax):
    """Add the common ppb/ppm relative-error guide lines."""
    ax.axhline(TOL_PPM, color="dimgray", lw=1.0, ls="--", label="1 ppm")
    ax.axhline(TOL_PPB, color="lightgray", lw=1.0, ls=":", label="1 ppb")


def save_and_show(filename: str, fig, *, output_dir: Path, show_plots: bool = True):
    """Save a figure under ``output_dir`` and optionally display it inline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)
    print(path)


def summarize_validation(df: pd.DataFrame, group_cols):
    """Aggregate standard absolute/relative validation residual columns."""
    grouped = df.groupby(list(group_cols), as_index=False)
    return grouped.agg(
        rows=("abs_delta", "size"),
        max_abs_delta=("abs_delta", "max"),
        median_abs_delta=("abs_delta", "median"),
        mean_abs_delta=("abs_delta", "mean"),
        max_rel_delta=("rel_delta", "max"),
        median_rel_delta=("rel_delta", "median"),
        mean_rel_delta=("rel_delta", "mean"),
        min_reference=("reference", "min"),
        median_reference=("reference", "median"),
    ).sort_values("max_rel_delta", ascending=False)


def status_from_rel(max_rel: float) -> str:
    """Map a maximum relative difference to the shared validation label."""
    if max_rel < TOL_PPB:
        return "PASS < ppb"
    if max_rel < TOL_PPM:
        return "PASS < ppm"
    if max_rel < 1.0e-3:
        return "CHECK < 1e-3"
    return "CHECK"


def try_import_nusquids():
    """Try to import nuSQuIDS under any known Python-binding name.

    Returns the module if found, or ``None`` when no binding is importable.
    Tries the names ``nuSQuIDS``, ``nuSQUIDSpy``, and ``nusquids`` in order.
    """
    for name in ("nuSQuIDS", "nuSQUIDSpy", "nusquids"):
        try:
            return importlib.import_module(name)
        except Exception:
            pass
    return None


def nusquids_is_available() -> bool:
    """Return ``True`` if any nuSQuIDS Python binding is importable."""
    return try_import_nusquids() is not None


def compare_probability_grids(
    tp_df: pd.DataFrame,
    nsq_df: pd.DataFrame,
    key_cols: Sequence[str],
    prob_cols: Sequence[str] = ("P_nue", "P_numu", "P_nutau"),
    *,
    floor: float = REL_FLOOR,
) -> pd.DataFrame:
    """Join two probability grids on *key_cols* and compute per-flavour errors.

    Args:
        tp_df: TPeanuts probability table.
        nsq_df: NuSQuIDS reference probability table.
        key_cols: Column names used as join keys (e.g. energy, baseline).
        prob_cols: Probability column names present in both DataFrames.
        floor: Absolute floor for the relative-error denominator.

    Returns:
        Merged DataFrame with added columns ``abs_err_*``, ``rel_err_*``,
        ``max_abs_err``, and ``max_rel_err``.
    """
    merged = tp_df.merge(nsq_df, on=list(key_cols), suffixes=("_tp", "_nsq"))
    if merged.empty:
        return merged
    abs_cols, rel_cols = [], []
    for col in prob_cols:
        tp_c, nsq_c = f"{col}_tp", f"{col}_nsq"
        if tp_c in merged.columns and nsq_c in merged.columns:
            a_col, r_col = f"abs_err_{col}", f"rel_err_{col}"
            merged[a_col] = (merged[tp_c] - merged[nsq_c]).abs()
            merged[r_col] = merged[a_col] / merged[nsq_c].abs().clip(lower=floor)
            abs_cols.append(a_col)
            rel_cols.append(r_col)
    if abs_cols:
        merged["max_abs_err"] = merged[abs_cols].max(axis=1)
    if rel_cols:
        merged["max_rel_err"] = merged[rel_cols].max(axis=1)
    return merged


def nusquids_precision_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """Compact precision metrics for a nuSQuIDS comparison DataFrame.

    Args:
        comparison: Output of :func:`compare_probability_grids`.

    Returns:
        Single-column DataFrame indexed by metric name.
    """
    if comparison.empty:
        return pd.DataFrame({"value": [float("nan")]}, index=["no_data"])
    max_rel = float(comparison["max_rel_err"].max())
    rows = [
        ("rows_compared",  float(len(comparison))),
        ("max_abs_err",    float(comparison["max_abs_err"].max())),
        ("median_abs_err", float(comparison["max_abs_err"].median())),
        ("max_rel_err",    max_rel),
        ("median_rel_err", float(comparison["max_rel_err"].median())),
        ("status",         status_from_rel(max_rel)),
    ]
    for col in ("P_nue", "P_numu", "P_nutau"):
        a_col = f"abs_err_{col}"
        if a_col in comparison.columns:
            rows.append((f"max_{a_col}", float(comparison[a_col].max())))
    df = pd.DataFrame(rows, columns=["metric", "value"]).set_index("metric")
    return df


def plot_comparison_curves(
    E_grid,
    tp_probs,
    nsq_probs,
    *,
    title: str,
    xlabel: str = r"$E$ [GeV]",
    filename: str,
    output_dir: Path,
    show_plots: bool = True,
):
    """Two-panel comparison plot: probability curves + absolute error.

    Args:
        E_grid: x-axis values (energy or other scan variable).
        tp_probs: TPeanuts probability array shaped ``(N, 3)``.
        nsq_probs: NuSQuIDS reference array shaped ``(N, 3)``.
        title: Figure title (top panel).
        xlabel: x-axis label for both panels.
        filename: Output filename passed to :func:`save_and_show`.
        output_dir: Output directory.
        show_plots: If ``False``, close the figure after saving.
    """
    E_np = to_numpy(E_grid)
    tp = to_numpy(tp_probs)
    nsq = to_numpy(nsq_probs)
    abs_err, _ = abs_rel_delta(tp, nsq)

    fig, (ax_p, ax_e) = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=True)
    for i, (label, color) in enumerate(zip(FLAVOUR_LABELS, FLAVOUR_COLORS)):
        ax_p.plot(E_np, nsq[:, i], color=color, lw=1.8, label=f"NuSQuIDS {label}")
        ax_p.plot(E_np, tp[:, i], color=color, lw=1.0, ls="--", alpha=0.8,
                  label=f"TPeanuts {label}")
        ax_e.semilogy(E_np, np.maximum(abs_err[:, i], 1e-300), color=color, label=label)

    ax_p.set_title(title)
    ax_p.set_ylabel("Probability")
    ax_p.legend(fontsize=7, ncol=2)
    ax_e.set_ylabel("|TPeanuts - NuSQuIDS|")
    ax_e.set_xlabel(xlabel)
    add_tolerance_lines(ax_e)
    ax_e.legend(fontsize=7)
    ax_p.set_xscale("log")
    ax_e.set_xscale("log")
    fig.tight_layout()
    save_and_show(filename, fig, output_dir=output_dir, show_plots=show_plots)


def plot_error_heatmap(
    x_vals,
    y_vals,
    z_matrix,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
    output_dir: Path,
    show_plots: bool = True,
):
    """Log-scale colour heatmap of absolute errors on a 2-D grid.

    Args:
        x_vals: x-axis values (e.g. energies), shape ``(Nx,)``.
        y_vals: y-axis values (e.g. baselines or zenith angles), shape ``(Ny,)``.
        z_matrix: Error values shaped ``(Ny, Nx)``.
        xlabel, ylabel, title: Axis and figure labels.
        filename: Output filename.
        output_dir: Output directory.
        show_plots: If ``False``, close the figure after saving.
    """
    import matplotlib.colors as mcolors

    x_np = to_numpy(x_vals)
    y_np = to_numpy(y_vals)
    z_np = np.maximum(to_numpy(z_matrix), 1e-18)

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    mesh = ax.pcolormesh(
        x_np, y_np, z_np,
        norm=mcolors.LogNorm(vmin=z_np.min(), vmax=z_np.max()),
        cmap="plasma",
        shading="auto",
    )
    fig.colorbar(mesh, ax=ax, label="|abs err|")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale("log")
    fig.tight_layout()
    save_and_show(filename, fig, output_dir=output_dir, show_plots=show_plots)


def plot_tripanel(
    x,
    candidate,
    reference,
    *,
    title: str,
    xlabel: str,
    filename: str,
    output_dir: Path,
    xscale: str = "linear",
    labels: Sequence[str] | None = None,
    show_plots: bool = True,
    quantity: str = "probability",
    flux_units: str = r"cm$^{-2}$ s$^{-1}$",
):
    """Plot candidate/reference values plus absolute and relative residuals.

    Args:
        quantity: Either ``"probability"`` or ``"flux"``, selecting the
            top-left panel's y-axis label. ``"probability"`` labels it
            "Probability"; ``"flux"`` labels it "Flux [<flux_units>]" so the
            physical units are always visible next to the curves.
        flux_units: Unit string shown in the y-axis label when
            ``quantity="flux"`` (e.g. ``r"cm$^{-2}$ s$^{-1}$"`` for a real
            physical flux, or ``"a.u."`` for an arbitrary-unit synthetic
            test spectrum). Ignored when ``quantity="probability"``.

    Raises:
        ValueError: If ``quantity`` is not ``"probability"`` or ``"flux"``.
    """
    labels = FLAVOUR_LABELS if labels is None else labels
    if quantity == "flux":
        y0_label = f"Flux [{flux_units}]"
    elif quantity == "probability":
        y0_label = "Probability"
    else:
        raise ValueError(f"quantity must be 'probability' or 'flux', got {quantity!r}")
    cand = to_numpy(candidate)
    ref = to_numpy(reference)
    abs_delta, rel_delta = abs_rel_delta(cand, ref)
    x_np = to_numpy(x)
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8), sharex=True)
    for i, (label, color) in enumerate(zip(labels, FLAVOUR_COLORS)):
        axes[0].plot(x_np, cand[:, i], color=color, label=f"TPeanuts {label}")
        axes[0].plot(
            x_np,
            ref[:, i],
            color=color,
            ls="--",
            alpha=0.65,
            label=f"legacy {label}",
        )
        axes[1].semilogy(
            x_np, np.maximum(abs_delta[:, i], 1.0e-300), color=color, label=label
        )
        axes[2].semilogy(
            x_np, np.maximum(rel_delta[:, i], 1.0e-300), color=color, label=label
        )
    axes[0].set_title(title)
    axes[0].set_ylabel(y0_label)
    axes[1].set_title("abs_delta")
    axes[1].set_ylabel("|TPeanuts - legacy|")
    axes[2].set_title("rel_delta")
    axes[2].set_ylabel("abs_delta / max(|legacy|, floor)")
    add_tolerance_lines(axes[2])
    for ax in axes:
        ax.set_xlabel(xlabel)
        if xscale == "log":
            ax.set_xscale("log")
        ax.legend(fontsize=7)
    fig.tight_layout()
    save_and_show(filename, fig, output_dir=output_dir, show_plots=show_plots)


def validation_metrics(
    case: str,
    reference: torch.Tensor,
    tested: torch.Tensor,
    *,
    reference_label: str = "reference",
    tested_label: str = "tested",
    absolute_threshold: float = ABSOLUTE_THRESHOLD,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Return reusable validation metrics and pointwise discrepancies.

    Relative errors are evaluated only where ``abs(reference)`` is at least
    ``absolute_threshold``. Masked entries are zero in the returned relative
    tensor so reductions used by plots remain finite; they are excluded from
    the reported maximum and mean relative errors.
    """
    if reference.shape != tested.shape:
        raise ValueError("reference and tested must have identical shapes.")
    absolute = torch.abs(tested - reference)
    mask = torch.abs(reference) >= absolute_threshold
    relative = torch.zeros_like(absolute)
    relative[mask] = absolute[mask] / torch.abs(reference[mask])
    selected = relative[mask]
    norm_reference = torch.amax(torch.abs(reference.sum(dim=-1) - 1.0)).item()
    norm_tested = torch.amax(torch.abs(tested.sum(dim=-1) - 1.0)).item()
    row = {
        "case": case,
        "max_abs": absolute.max().item(),
        "mean_abs": absolute.mean().item(),
        "max_rel": selected.max().item() if selected.numel() else float("nan"),
        "mean_rel": selected.mean().item() if selected.numel() else float("nan"),
        f"normalization_{reference_label}": norm_reference,
        f"normalization_{tested_label}": norm_tested,
        "relative_points": int(mask.sum().item()),
        "absolute_threshold": absolute_threshold,
    }
    return row, absolute, relative


def validation_heatmap(
    absolute,
    relative,
    x,
    y,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    output_dir: Path,
    reference_label: str = "reference",
    absolute_threshold: float = ABSOLUTE_THRESHOLD,
    show_plots: bool = True,
):
    """Plot the repeated absolute/thresholded-relative validation heatmaps."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharex=True, sharey=True)
    labels = (
        r"maximum $|\Delta P|$",
        f"maximum relative error vs {reference_label} for |P_reference| >= {absolute_threshold:.0e}",
    )
    for ax, values, subtitle, label in zip(
        axes, (absolute, relative), ("Absolute error", "Relative error"), labels
    ):
        im = ax.pcolormesh(to_numpy(x), to_numpy(y), to_numpy(values).T, shading="auto", cmap="magma")
        ax.set(xlabel=xlabel, ylabel=ylabel, title=subtitle)
        fig.colorbar(im, ax=ax, label=label)
    fig.suptitle(title)
    fig.tight_layout()
    save_and_show(filename, fig, output_dir=output_dir, show_plots=show_plots)


def validation_comparison_grid(
    reference,
    tested,
    energy,
    angle,
    fixed_angle,
    fixed_energy,
    *,
    title: str,
    angle_label: str,
    filename: str,
    output_dir: Path,
    reference_label: str = "reference",
    tested_label: str = "tested",
    absolute_threshold: float = ABSOLUTE_THRESHOLD,
    log_energy: bool = False,
    labels: Sequence[str] | None = None,
    colours: Sequence[str] | None = None,
    show_plots: bool = True,
):
    """Plot probability, absolute-error and relative-error energy/angle scans.

    ``labels`` and ``colours`` default to the three active flavours, but may
    be supplied explicitly for model extensions such as the 3+1 sterile case.
    """
    labels = FLAVOUR_LABELS if labels is None else labels
    colours = FLAVOUR_COLORS if colours is None else colours
    n_flavours = reference.shape[-1]
    if len(labels) != n_flavours or len(colours) != n_flavours:
        raise ValueError(
            "labels and colours must match the final flavour dimension "
            f"({n_flavours}); got {len(labels)} and {len(colours)}."
        )
    i_angle = int(torch.argmin(torch.abs(angle - fixed_angle)))
    i_energy = int(torch.argmin(torch.abs(energy - fixed_energy)))
    _, absolute, relative = validation_metrics(
        title, reference, tested,
        reference_label=reference_label,
        tested_label=tested_label,
        absolute_threshold=absolute_threshold,
    )
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 11), sharex="col")
    for flavour, (label, colour) in enumerate(zip(labels, colours)):
        axes[0, 0].plot(to_numpy(energy), to_numpy(reference[:, i_angle, flavour]), color=colour, label=f"{label} {reference_label}")
        axes[0, 0].plot(to_numpy(energy), to_numpy(tested[:, i_angle, flavour]), "--", color=colour, label=f"{label} {tested_label}")
        axes[1, 0].plot(to_numpy(energy), to_numpy(absolute[:, i_angle, flavour]), color=colour, label=label)
        axes[2, 0].plot(to_numpy(energy), to_numpy(relative[:, i_angle, flavour]), color=colour, label=label)
        axes[0, 1].plot(to_numpy(angle), to_numpy(reference[i_energy, :, flavour]), color=colour, label=f"{label} {reference_label}")
        axes[0, 1].plot(to_numpy(angle), to_numpy(tested[i_energy, :, flavour]), "--", color=colour, label=f"{label} {tested_label}")
        axes[1, 1].plot(to_numpy(angle), to_numpy(absolute[i_energy, :, flavour]), color=colour, label=label)
        axes[2, 1].plot(to_numpy(angle), to_numpy(relative[i_energy, :, flavour]), color=colour, label=label)
    axes[0, 0].set_title(f"Energy scan at {angle_label}={float(angle[i_angle]):.3g}")
    axes[0, 1].set_title(f"Angular scan at E={float(energy[i_energy]):.3g} MeV")
    for ax in axes[0]:
        ax.set_ylabel("Probability"); ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=7, ncol=2)
    for ax in axes[1]: ax.set_ylabel("Absolute error"); ax.set_yscale("log")
    for ax in axes[2]: ax.set_ylabel("Thresholded relative error"); ax.set_yscale("log")
    axes[2, 0].set_xlabel("E [MeV]"); axes[2, 1].set_xlabel(angle_label)
    if log_energy:
        for ax in axes[:, 0]: ax.set_xscale("log")
    for ax in axes.flat: ax.grid(alpha=0.25)
    fig.suptitle(title); fig.tight_layout()
    save_and_show(filename, fig, output_dir=output_dir, show_plots=show_plots)


def validation_summary_plot(
    rows,
    *,
    title: str,
    filename: str,
    output_dir: Path,
    show_plots: bool = True,
):
    """Plot max/mean absolute and relative errors in a reusable 2x2 grid."""
    labels = [row["case"] for row in rows]
    panels = (
        ("max_abs", "Maximum absolute error", "C0"),
        ("max_rel", "Maximum relative error", "C1"),
        ("mean_abs", "Mean absolute error", "C2"),
        ("mean_rel", "Mean relative error", "C3"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6))
    for ax, (key, subtitle, colour) in zip(axes.flat, panels):
        ax.bar(range(len(labels)), [row[key] for row in rows], color=colour)
        ax.set_title(subtitle); ax.set_yscale("log")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle(title); fig.tight_layout()
    save_and_show(filename, fig, output_dir=output_dir, show_plots=show_plots)

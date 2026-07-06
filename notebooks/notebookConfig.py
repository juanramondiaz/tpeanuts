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
Shared notebook configuration: output paths and presentation style.

Every TPeanuts notebook currently repeats the same boilerplate by hand: it
locates the repository root, decides where to write its CSV/figure outputs
under a per-OS output share, and sets matplotlib/numpy/torch/pandas display
options. That boilerplate has drifted slightly between notebooks (different
default figure sizes, different ``torch.set_printoptions`` precisions, the
``v:\\output`` root sometimes Windows-only with no Linux fallback). This
module centralizes it in one ``NotebookConfig`` dataclass so every notebook
shares the same paths and the same look.

``DEFAULT_OUTPUT_ROOT`` is resolved once at import time from
``platform.system()``: ``v:\\output`` on Windows, ``/mnt/v/output`` on Linux
(and other non-Windows platforms), matching the network share mounted at
``/mnt/v`` under WSL/Linux. It can still be overridden per-run with the
``TPEANUTS_OUTPUT_ROOT`` environment variable, exactly as the individual
notebooks did before.

Typical notebook usage::

    from tpeanuts.notebooks.notebookConfig import load_notebook_config

    config = load_notebook_config()
    OUTPUT_DIR = config.output_dir("results", "1_msw_resonance_diagram")
    DEVICE, DTYPE = config.device, config.dtype

Module functions:
    default_output_root(...)
        Resolve the per-OS default output root (Windows vs. Linux/other).
    load_notebook_config(...)
        Build a NotebookConfig and apply its shared presentation style in one
        call; the standard notebook entry point.

Module classes:
    NotebookConfig
        Frozen dataclass bundling repository/output paths, the default torch
        runtime (device, dtype), and shared matplotlib/numpy/torch/pandas
        display settings used across every notebook.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from tpeanuts.util.notebooks import find_repo_root


def default_output_root() -> Path:
    """Resolve the per-OS default output root.

    Returns:
        ``Path(r"v:\\output")`` on Windows, ``Path("/mnt/v/output")`` on
        Linux and any other non-Windows platform. Both point at the same
        network share, mounted at a drive letter on Windows and under
        ``/mnt/v`` on Linux/WSL.
    """
    if platform.system() == "Windows":
        return Path(r"v:\output")
    return Path("/mnt/v/output")


DEFAULT_OUTPUT_ROOT: Path = default_output_root()


@dataclass(frozen=True)
class NotebookConfig:
    """
    Shared notebook configuration: repository/output paths plus the default
    torch runtime and matplotlib/numpy/torch/pandas presentation style used
    consistently across every TPeanuts notebook.

    Parameters
    ----------
    package_dir:
        Repository root, i.e. the directory containing ``pyproject.toml``
        and the ``tpeanuts`` package. Located once via
        ``tpeanuts.util.notebooks.find_repo_root``.

    output_root:
        Root directory for every notebook's generated data and figures.
        Defaults to ``DEFAULT_OUTPUT_ROOT``, overridable per-run with the
        ``TPEANUTS_OUTPUT_ROOT`` environment variable.

    device:
        Default torch device for notebooks that do not need a different
        (e.g. GPU) runtime context.

    dtype:
        Default real torch floating dtype shared by notebook computations.

    figsize:
        Default matplotlib ``figure.figsize``, in inches.

    dpi:
        Default matplotlib ``figure.dpi`` used for inline display. Saved
        figures use the higher resolution already centralized in
        ``tpeanuts.util.notebooks.save_figure``/``save_and_show`` (180 dpi).

    axes_grid:
        Default matplotlib ``axes.grid``.

    grid_alpha:
        Default matplotlib ``grid.alpha``.

    torch_precision:
        Decimal precision passed to ``torch.set_printoptions`` and
        ``numpy.set_printoptions``.

    torch_sci_mode:
        Whether ``torch.set_printoptions`` prints tensors in scientific
        notation.

    print_linewidth:
        Shared ``linewidth`` for ``torch.set_printoptions`` and
        ``numpy.set_printoptions``.

    pandas_precision:
        Decimal precision for ``pandas.set_option("display.precision", ...)``.

    show_plots:
        Whether figures are displayed inline by default (vs. only saved and
        closed), forwarded as the default for
        ``tpeanuts.util.notebooks.save_and_show``/``show_figure`` calls.
    """

    package_dir: Path = field(
        default_factory=lambda: find_repo_root(Path.cwd(), folder="analysis")
    )
    output_root: Path = field(
        default_factory=lambda: Path(
            os.environ.get("TPEANUTS_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT))
        )
    )

    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float64

    figsize: tuple[float, float] = (10.0, 4.8)
    dpi: int = 120
    axes_grid: bool = True
    grid_alpha: float = 0.3

    torch_precision: int = 6
    torch_sci_mode: bool = True
    print_linewidth: int = 160

    pandas_precision: int = 6

    show_plots: bool = True

    @property
    def data_dir(self) -> Path:
        """Repository ``data/`` directory holding input tables (density profiles, the B16 solar model, ...)."""
        return self.package_dir / "data"

    @property
    def legacy_data_dir(self) -> Path:
        """Bundled legacy ``peanuts`` data directory, e.g. ``data/peanuts/earth_density.csv``."""
        return self.data_dir / "peanuts"

    @property
    def external_data_dir(self) -> Path:
        """Repository ``data/external`` directory holding downloaded external reference tables."""
        return self.data_dir / "external"

    @property
    def earth_density_file(self) -> Path:
        """Bundled Earth density table used by Earth-profile notebooks."""
        return self.data_dir / "density" / "earth_density.csv"

    @property
    def prem500_file(self) -> Path:
        """Downloaded PREM500 Earth model CSV table."""
        return self.external_data_dir / "PREM500.csv"

    @property
    def output_data_root(self) -> Path:
        """``output_root/data``: shared root for generated (non-figure) data tables."""
        return self.output_root / "data"

    @property
    def output_analysis_root(self) -> Path:
        """``output_root/analysis``: outputs of the ``notebooks/analysis`` notebooks."""
        return self.output_root / "analysis"

    @property
    def output_benchmark_root(self) -> Path:
        """``output_root/benchmark``: outputs of the ``notebooks/benchmark`` notebooks."""
        return self.output_root / "benchmark"

    @property
    def output_results_root(self) -> Path:
        """``output_root/results``: outputs of the ``notebooks/results`` notebooks."""
        return self.output_root / "results"

    @property
    def output_validation_root(self) -> Path:
        """``output_root/validation``: outputs of the ``notebooks/validation`` notebooks."""
        return self.output_root / "validation"

    @property
    def output_runs_root(self) -> Path:
        """``output_root/runs``: outputs of the ``notebooks/runs`` notebooks."""
        return self.output_root / "runs"

    @property
    def output_test_root(self) -> Path:
        """``output_root/test``: outputs of the ``notebooks/tests`` notebooks."""
        return self.output_root / "test"

    def output_dir(self, *parts: str) -> Path:
        """Build, create, and return a notebook-specific output directory.

        Args:
            *parts: Path components appended to ``output_root``, e.g.
                ``config.output_dir("results", "1_msw_resonance_diagram")``.

        Returns:
            ``output_root.joinpath(*parts)``, created (including parents) if
            it does not already exist.
        """
        path = self.output_root.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def apply_style(self) -> None:
        """Apply the shared matplotlib/torch/numpy/pandas presentation settings.

        Called once per notebook (see ``load_notebook_config``) so every
        notebook renders figures, tensors, arrays, and tables the same way.
        """
        plt.rcParams.update(
            {
                "figure.figsize": self.figsize,
                "figure.dpi": self.dpi,
                "axes.grid": self.axes_grid,
                "grid.alpha": self.grid_alpha,
            }
        )
        torch.set_printoptions(
            precision=self.torch_precision,
            sci_mode=self.torch_sci_mode,
            linewidth=self.print_linewidth,
        )
        np.set_printoptions(
            precision=self.torch_precision,
            suppress=False,
            linewidth=self.print_linewidth,
        )
        pd.set_option("display.precision", self.pandas_precision)


def load_notebook_config(**overrides: Any) -> NotebookConfig:
    """Build a ``NotebookConfig`` and apply its presentation style.

    This is the standard notebook entry point: a single call that resolves
    the repository root, the output root, and the shared display style.

    Args:
        **overrides: Field overrides forwarded to ``NotebookConfig``, e.g.
            ``load_notebook_config(show_plots=False)``.

    Returns:
        The constructed ``NotebookConfig``, with its style already applied.
    """
    config = NotebookConfig(**overrides)
    config.apply_style()
    return config

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
Shared helpers for TPeanuts notebooks.

The helpers in this module keep notebook boilerplate small while preserving the
original Python modules as the source of truth for tests and runs.

Module functions:
    find_repo_root(...): Locate the TPeanuts repository root from a notebook
        working directory.
    add_repo_to_sys_path(...): Add the repository root and its parent to
        sys.path for notebook execution.
    safe_label(...): Return a filesystem-friendly label for generated
        figures.
    print_error_table(...): Print a compact text table in a notebook output
        cell.
    print_comparison(...): Print absolute/relative errors between two
        array-like values.
    save_figure(...): Save a matplotlib figure to a notebook output
        directory.
    show_figure(...): Display or close a matplotlib figure per notebook
        configuration.
    save_and_show(...): Save a figure and then apply the display policy.
    build_notebook_test_runner(...): Factory creating a NotebookTestRunner.

Module classes:
    NotebookTestRunner: Runs imported test functions from a notebook with
        shared figure-saving and output-directory handling.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable


def find_repo_root(start: Path | str | None = None, folder: str = "tests") -> Path:
    """
    Find the TPeanuts repository root from a notebook working directory.

    Args:
        start: Directory or file path used as the starting point. None uses the
            current working directory.
        folder: Notebook subfolder that must exist below notebooks, such as
            "tests", "runs", "analysis", or "benchmark".

    Returns:
        Repository root path containing pyproject.toml and notebooks/folder.
    """
    current = Path.cwd() if start is None else Path(start)
    current = current.resolve()

    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "notebooks" / folder).is_dir():
            return candidate

    raise RuntimeError(
        f"Could not locate the TPeanuts repository root from {current} "
        f"using notebooks/{folder}."
    )


def add_repo_to_sys_path(repo_root: Path | str | None = None, folder: str = "tests") -> Path:
    """Add the repository and its parent to sys.path for notebook execution.

    Args:
        repo_root: Explicit repository root, or a starting path passed to
            find_repo_root. None searches from the current working
            directory.
        folder: Notebook subfolder used to validate the repository root, as
            in find_repo_root.

    Returns:
        Resolved repository root path that was added to sys.path.
    """
    root = find_repo_root(repo_root, folder=folder)
    for path in [root.parent, root]:
        path_s = str(path)
        if path_s not in sys.path:
            sys.path.insert(0, path_s)
    return root


def safe_label(label: Any) -> str:
    """Return a filesystem-friendly label for generated notebook figures.

    Args:
        label: Value to convert. Converted to a string first.

    Returns:
        String with only alphanumeric characters, ".", "_", and "-"
        retained (others replaced by "_"), surrounding underscores
        stripped, and "figure" returned if the result would be empty.
    """
    text = str(label)
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    return safe.strip("_") or "figure"


def print_error_table(title: str, rows: list[str]) -> None:
    """
    Print a compact text table in a notebook output cell.

    Args:
        title: Heading printed before the rows.
        rows: Preformatted row strings.
    """
    print(f"\n{title}")
    print("-" * len(title))
    for row in rows:
        print(row)


def print_comparison(label: str, value: Any, reference: Any) -> Any:
    """
    Print absolute and relative errors for two array-like values.

    Args:
        label: Description printed before the comparison.
        value: Candidate values.
        reference: Reference values.

    Returns:
        NumPy array with the element-wise relative error.
    """
    import numpy as np

    from tpeanuts.util.math import relative_error
    from tpeanuts.util.type import to_numpy

    value_np = to_numpy(value, dtype=float)
    reference_np = to_numpy(reference, dtype=float)
    abs_err = np.abs(value_np - reference_np)
    rel_err = relative_error(value_np, reference_np)

    print(f"\n{label}")
    print("  tpeanuts:", value_np)
    print("  legacy  :", reference_np)
    print(f"  max abs error: {np.max(abs_err):.6e}")
    print(f"  max rel error: {np.max(rel_err):.6e}")
    return rel_err


_figure_counters: dict[Path, int] = {}


def save_figure(
    fig: Any | None = None,
    name: str | Path | None = None,
    *,
    output_dir: Path | str,
    dpi: int = 180,
    bbox_inches: str = "tight",
) -> Path:
    """
    Save a matplotlib figure to a notebook output directory.

    Args:
        fig: Figure to save. None saves the current matplotlib figure.
        name: Output filename relative to output_dir. None creates a numbered
            figure filename for the given output directory.
        output_dir: Directory where the figure file is written.
        dpi: Resolution passed to matplotlib Figure.savefig.
        bbox_inches: Bounding-box mode passed to matplotlib Figure.savefig.

    Returns:
        Full path to the saved figure.
    """
    import matplotlib.pyplot as plt

    target_dir = Path(output_dir)
    if fig is None:
        fig = plt.gcf()

    if name is None:
        count = _figure_counters.get(target_dir, 0) + 1
        _figure_counters[target_dir] = count
        name = f"figure_{count:03d}.png"

    path = target_dir / str(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Saved figure: {path}")
    return path


def show_figure(fig: Any | None = None, *, show_plots: bool = True) -> None:
    """
    Display or close a matplotlib figure according to notebook configuration.

    Args:
        fig: Figure to close when show_plots is False. None uses the current
            matplotlib figure.
        show_plots: Whether figures should be rendered inline in the notebook.
    """
    import matplotlib.pyplot as plt

    if fig is None:
        fig = plt.gcf()

    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def save_and_show(
    name: str | Path | None = None,
    fig: Any | None = None,
    *,
    output_dir: Path | str,
    show_plots: bool = True,
    dpi: int = 180,
    bbox_inches: str = "tight",
) -> Path:
    """
    Save a matplotlib figure and then apply the notebook display policy.

    Args:
        name: Output filename relative to output_dir. None creates a numbered
            figure filename for the given output directory.
        fig: Figure to save and optionally show. None uses the current figure.
        output_dir: Directory where the figure file is written.
        show_plots: Whether the figure should also be displayed inline.
        dpi: Resolution passed to matplotlib Figure.savefig.
        bbox_inches: Bounding-box mode passed to matplotlib Figure.savefig.

    Returns:
        Full path to the saved figure.
    """
    path = save_figure(
        fig=fig,
        name=name,
        output_dir=output_dir,
        dpi=dpi,
        bbox_inches=bbox_inches,
    )
    show_figure(fig=fig, show_plots=show_plots)
    return path


class NotebookTestRunner:
    """Run imported test functions from a notebook with shared output handling.

    Wraps an imported test module so its tests can be executed cell-by-cell
    inside a notebook while figures are automatically saved to a common
    output directory and matplotlib's interactive ``show`` is redirected
    consistently with the notebook's display policy.
    """

    def __init__(
        self,
        test_module: Any,
        output_dir: Path | str,
        *,
        show_plots: bool = True,
        auto_save_figures: bool = True,
        extra_module_attrs: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            test_module: Imported Python test module whose functions will be
                executed via run_test/run_call.
            output_dir: Directory where generated figures are saved.
            show_plots: Whether figures should also be displayed inline.
            auto_save_figures: Whether open matplotlib figures are saved
                automatically after each test runs.
            extra_module_attrs: Optional extra attributes to set on
                test_module before running tests (e.g. shared fixtures).
        """
        self.test_module = test_module
        self.output_dir = Path(output_dir)
        self.show_plots = bool(show_plots)
        self.auto_save_figures = bool(auto_save_figures)
        self.extra_module_attrs = dict(extra_module_attrs or {})
        self.current_label = "figure"
        self.show_counters: dict[str, int] = {}

        try:
            if not self.show_plots:
                import matplotlib

                matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.close(fig)
        except Exception:  # pragma: no cover - notebooks may not need plotting.
            plt = None

        self.plt = plt
        self.original_plt_show = None if plt is None else plt.show

    def _save_open_figures(self, label: Any | None = None) -> list[Path]:
        if not self.auto_save_figures or self.plt is None:
            return []

        safe = safe_label(self.current_label if label is None else label)
        saved: list[Path] = []

        for number in self.plt.get_fignums():
            fig = self.plt.figure(number)
            if fig is None or not fig.get_axes():
                continue

            count = self.show_counters.get(safe, 0) + 1
            self.show_counters[safe] = count
            path = self.output_dir / f"{safe}_figure_{count:03d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=180, bbox_inches="tight")
            saved.append(path)
            print(f"Saved plot: {path}")

        return saved

    def _controlled_show(self, *args: Any, **kwargs: Any) -> Any:
        self._save_open_figures(self.current_label)

        if self.plt is None:
            return None

        if self.show_plots and self.original_plt_show is not None:
            return self.original_plt_show(*args, **kwargs)

        self.plt.close("all")
        return None

    def prepare_module(self) -> None:
        """Configure test_module's output directory, plot flag, and matplotlib hook.

        Creates output_dir if missing, propagates output_dir/show_plots onto
        any matching attributes found on test_module (OUTPUT_DIR,
        OUTPUT_CONFIG.output_dir, SHOW_PLOTS), and installs the controlled
        plt.show wrapper so figures are captured instead of blocking.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(self.test_module, "OUTPUT_DIR"):
            self.test_module.OUTPUT_DIR = self.output_dir
        output_config = getattr(self.test_module, "OUTPUT_CONFIG", None)
        if output_config is not None and hasattr(output_config, "output_dir"):
            output_config.output_dir = str(self.output_dir)
        if hasattr(self.test_module, "SHOW_PLOTS"):
            self.test_module.SHOW_PLOTS = self.show_plots
        if hasattr(self.test_module, "os"):
            self.test_module.os.makedirs(self.output_dir, exist_ok=True)
        if self.plt is not None:
            self.plt.show = self._controlled_show
        if hasattr(self.test_module, "plt"):
            self.test_module.plt.show = self._controlled_show

        for name, value in self.extra_module_attrs.items():
            setattr(self.test_module, name, value)

    def restore_module(self) -> None:
        """Restore matplotlib show after a notebook-managed call finishes."""
        if self.plt is None or self.original_plt_show is None:
            return

        self.plt.show = self.original_plt_show
        if hasattr(self.test_module, "plt"):
            self.test_module.plt.show = self.original_plt_show

    def run_test(self, test_func: Callable[..., Any]) -> Any:
        """Run a single test function with output/figure handling.

        Args:
            test_func: Zero-argument callable to execute, typically a test
                function imported from test_module.

        Returns:
            Value returned by test_func.

        Raises:
            Exception: Re-raises any exception from test_func after printing
                a failure banner and a traceback.
        """
        self.prepare_module()
        self.current_label = test_func.__name__
        print(f"Running {test_func.__name__} ...")

        try:
            result = test_func()
        except Exception:
            print(f"FAILED: {test_func.__name__}")
            traceback.print_exc()
            self.restore_module()
            raise

        self._save_open_figures(self.current_label)
        if not self.show_plots and self.plt is not None:
            self.plt.close("all")

        self.restore_module()
        print('-'*90)
        print(f"PASSED: {test_func.__name__}")
        return result

    def run_call(
        self,
        label: str,
        call: Callable[[], Any],
        *,
        requires_real_data: bool = False,
        run_real_data: bool = True,
    ) -> Any:
        """Run an arbitrary callable with the same output/figure handling as run_test.

        Args:
            label: Description used in progress messages and as the figure
                filename prefix.
            call: Zero-argument callable to execute.
            requires_real_data: Whether call needs real (non-synthetic) data
                files to run.
            run_real_data: Whether real-data calls are permitted to run in
                this notebook session. When False and requires_real_data is
                True, the call is skipped.

        Returns:
            Value returned by call, or None if the call was skipped.

        Raises:
            Exception: Re-raises any exception from call after printing a
                failure banner and a traceback.
        """
        if requires_real_data and not run_real_data:
            print(f"SKIPPED: {label} requires real data. Enable the real-data flag to run it.")
            return None

        self.prepare_module()
        self.current_label = label
        print(f"Running {label} ...")

        try:
            result = call()
        except Exception:
            print(f"FAILED: {label}")
            traceback.print_exc()
            self.restore_module()
            raise

        self._save_open_figures(self.current_label)
        if not self.show_plots and self.plt is not None:
            self.plt.close("all")

        self.restore_module()
        print(f"PASSED: {label}")
        return result


def build_notebook_test_runner(
    test_module: Any,
    output_dir: Path | str,
    *,
    show_plots: bool = True,
    auto_save_figures: bool = True,
    extra_module_attrs: dict[str, Any] | None = None,
) -> NotebookTestRunner:
    """Factory used by test notebooks for a compact import line.

    Args:
        test_module: Imported Python test module to run.
        output_dir: Directory where generated figures are saved.
        show_plots: Whether figures should also be displayed inline.
        auto_save_figures: Whether open matplotlib figures are saved
            automatically after each test runs.
        extra_module_attrs: Optional extra attributes to set on test_module
            before running tests.

    Returns:
        Configured NotebookTestRunner instance.
    """
    return NotebookTestRunner(
        test_module,
        output_dir,
        show_plots=show_plots,
        auto_save_figures=auto_save_figures,
        extra_module_attrs=extra_module_attrs,
    )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
#
# TPeanuts — Fast Simulation of Neutrino Oscillations in Matter.
#
# Developed as part of the Master's Thesis for the MSc in Advanced Physics,
# specialization in Theoretical Physics, at the Universitat de València.
#
# Author
# ------
# Juan Ramón Díaz Santos
# diazjuan@alumni.uv.es
#
# Supervisors
# -----------
# Roberto Ruiz de Austri Bazán
# rruiz@ific.uv.es
#
# Michele Lucente
# michele.lucente@unibo.it
#
# Universitat de València, Spain
# September 2026
# =============================================================================

"""Shared configuration of the tutorial notebooks: where they write, how their figures look, which runtime they use.

Every notebook starts the same way: it locates the repository, decides where its figures and tables are written, and
sets the matplotlib, NumPy, PyTorch and pandas display options. Keeping that in one frozen dataclass makes every
notebook share the same paths and the same look, and makes a change of style a one-line edit.

The output root is resolved in this order: the ``TPEANUTS_OUTPUT_ROOT`` environment variable; then the per-user
network share ``v:\\output`` (Windows) or ``/mnt/v/output`` (Linux, WSL) if it exists; finally
``<repository>/notebooks/_output`` (git-ignored), so that the notebooks run on any checkout. The default runtime is
the CPU in float64, which is deterministic and the reference precision; set ``TPEANUTS_NOTEBOOK_DEVICE=cuda`` to use
a GPU. Set ``TPEANUTS_NOTEBOOK_QUICK=1`` to run every notebook on reduced grids (used by the tests).

Each notebook also declares in its metadata (block ``tpeanuts``) what it needs: ``requires`` lists optional components
(see ``REQUIREMENTS``), ``gpu`` says whether it needs (``required``), can use (``optional``) or ignores (``none``) a
CUDA device, and ``tier`` (``fast`` or ``slow``) says whether the default test run executes it.
``requirement_available`` tells whether a requirement is met on the current machine, so that tests skip what
cannot run instead of failing. Besides the optional packages and the GPU, ``icecube_data`` stands for the downloaded
IceCube release and the generated Honda flux table.

Typical use, in the first code cells of a notebook::

    from tpeanuts.notebooks.notebookConfig import load_notebook_config

    config = load_notebook_config()
    OUTPUT_DIR = config.output_dir("1_physics", "1_foundation")

Module contents:
    find_repo_root(...)
        Walk up from a directory to the repository root (the directory with the tpeanuts ``pyproject.toml``).
    default_output_root(...)
        The output root described above.
    REQUIREMENTS, GPU_MODES, TIERS
        The vocabulary of the ``tpeanuts`` notebook metadata block.
    requirement_available(...)
        Whether one named requirement (optional package or CUDA device) is available here.
    NotebookConfig
        Frozen dataclass: repository and output paths, runtime (device, dtype), quick mode, seed, start time and display
        style.
    load_notebook_config(...)
        Build a ``NotebookConfig`` and apply its style: the standard entry point of a notebook.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def find_repo_root(start: Path) -> Optional[Path]:
    """Return the tpeanuts repository root above ``start``, or ``None`` if ``start`` is not inside a checkout.

    Args:
        start: Directory (or file) to start from; its parents are searched, nearest first.

    Returns:
        The first directory that holds a ``pyproject.toml`` declaring the project ``tpeanuts`` and a ``notebooks``
        directory, i.e. a source checkout. ``None`` for an installed package or an unrelated directory.
    """
    for directory in [start.resolve(), *start.resolve().parents]:
        pyproject = directory / "pyproject.toml"
        if pyproject.is_file() and (directory / "notebooks").is_dir():
            if 'name = "tpeanuts"' in pyproject.read_text(encoding="utf-8"):
                return directory
    return None


def default_output_root(repository: Optional[Path]) -> Path:
    """Resolve the root directory for generated figures and tables.

    Args:
        repository: Repository root, or ``None`` outside a checkout.

    Returns:
        ``$TPEANUTS_OUTPUT_ROOT`` if set; else the network share ``v:\\output`` / ``/mnt/v/output`` if that directory
        exists; else ``<repository>/notebooks/_output`` (or ``./tpeanuts_output`` outside a checkout).
    """
    override = os.environ.get("TPEANUTS_OUTPUT_ROOT")
    if override:
        return Path(override)
    share = Path(r"v:\output") if platform.system() == "Windows" else Path("/mnt/v/output")
    if share.is_dir():
        return share
    if repository is not None:
        return repository / "notebooks" / "_output"
    return Path.cwd() / "tpeanuts_output"


REQUIREMENTS: dict[str, str] = {
    "nusquids": "nuSQuIDS",
    "peanuts": "peanuts",
    "mceq": "MCEq",
    "pymsis": "pymsis",
    "icecube_data": "IceCube data release and Honda flux table",
    "gpu": "cuda",
}
GPU_MODES = ("none", "optional", "required")
TIERS = ("fast", "slow")


def requirement_available(name: str) -> bool:
    """Whether a notebook requirement (a key of ``REQUIREMENTS``) is met on this machine.

    Args:
        name: ``"gpu"`` (a CUDA device is visible to PyTorch) or the name of an optional package.

    Returns:
        ``True`` if the CUDA device is available or the package can be imported; ``False`` otherwise.

    Raises:
        KeyError: ``name`` is not a key of ``REQUIREMENTS``.
    """
    module = REQUIREMENTS[name]
    if module == "cuda":
        return torch.cuda.is_available()
    if name == "peanuts":
        return _original_peanuts_available()
    if name == "icecube_data":
        return _icecube_data_available()
    return importlib.util.find_spec(module) is not None


def _icecube_data_available() -> bool:
    """Whether the IceCube data release (``tpeanuts-data fetch icecube``) and the Honda flux table
    (``tpeanuts-data build honda-flux``) are on disk, which the IceCube forward model needs."""
    import tpeanuts.config.default as default
    from tpeanuts.data.fetch import dataset_present

    return dataset_present("icecube") and (default.ATMOSPHERE_DATA_DIR / "honda" / "flux" / "honda_flux.csv").is_file()


def _original_peanuts_available() -> bool:
    """Whether the original PEANUTS can be imported: its directory (``PEANUTS_PATH`` or a ``PEANUTS`` sibling of the
    repository) holds the package, and the Numba it needs is installed."""
    candidates = [os.environ.get("PEANUTS_PATH"), str(Path(__file__).resolve().parents[3].parent / "PEANUTS")]
    found = any(c and (Path(c) / "peanuts" / "__init__.py").is_file() for c in candidates)
    return found and importlib.util.find_spec("numba") is not None and importlib.util.find_spec("pandas") is not None


def _default_device() -> torch.device:
    return torch.device(os.environ.get("TPEANUTS_NOTEBOOK_DEVICE", "cpu"))


@dataclass(frozen=True)
class NotebookConfig:
    """Paths, runtime and display style shared by every tutorial notebook.

    Attributes:
        package_dir: Repository root (the directory with ``pyproject.toml``), or ``None`` outside a source checkout.
        output_root: Root of every generated figure and table; see :func:`default_output_root`.
        device: Torch device of the notebook's ``RuntimeContext`` (CPU unless ``TPEANUTS_NOTEBOOK_DEVICE`` says
            otherwise).
        dtype: Real floating-point dtype of the notebook's ``RuntimeContext``; float64 is the reference precision.
        quick: ``True`` when ``TPEANUTS_NOTEBOOK_QUICK=1``: notebooks then use coarser grids and fewer repetitions.
        figsize: Default matplotlib ``figure.figsize``, in inches.
        dpi: Default ``figure.dpi`` for inline display (saved figures use 180 dpi, see ``save_and_show``).
        axes_grid: Default ``axes.grid``.
        grid_alpha: Default ``grid.alpha``.
        torch_precision: Decimal precision of ``torch.set_printoptions`` and ``numpy.set_printoptions``.
        torch_sci_mode: Whether tensors print in scientific notation.
        print_linewidth: Line width of printed tensors and arrays.
        pandas_precision: Decimal precision of ``pandas`` tables.
        show_plots: Whether ``save_and_show`` displays figures inline (``False`` saves and closes them).
        seed: Seed that ``load_notebook_config`` gives to the torch and NumPy global generators; calculations that
            need randomness also pass an explicit seed of their own.
        started_at: Wall-clock time (``time.time()``) at which the configuration was built, i.e. the start of the
            notebook; the environment report uses it for the elapsed time.
    """

    package_dir: Optional[Path] = field(default_factory=lambda: find_repo_root(Path.cwd()))
    output_root: Path = field(
        default_factory=lambda: default_output_root(find_repo_root(Path.cwd()))
    )
    device: torch.device = field(default_factory=_default_device)
    dtype: torch.dtype = torch.float64
    quick: bool = field(default_factory=lambda: os.environ.get("TPEANUTS_NOTEBOOK_QUICK") == "1")

    figsize: tuple[float, float] = (10.0, 4.8)
    dpi: int = 120
    axes_grid: bool = True
    grid_alpha: float = 0.3

    torch_precision: int = 6
    torch_sci_mode: bool = True
    print_linewidth: int = 160

    pandas_precision: int = 6

    show_plots: bool = True

    seed: int = 0
    started_at: float = field(default_factory=time.time)

    def output_dir(self, *parts: str) -> Path:
        """Build, create and return a notebook's output directory.

        Args:
            *parts: Path components appended to ``output_root``: the notebook's location relative to the
                ``notebooks`` directory, without the file name (``("1_physics", "1_foundation")`` for
                ``notebooks/1_physics/1_foundation/01_getting_started.ipynb``).

        Returns:
            ``output_root.joinpath(*parts)``, created with its parents if it did not exist.
        """
        path = self.output_root.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def apply_style(self) -> None:
        """Apply the shared matplotlib, torch, NumPy and pandas presentation settings."""
        plt.rcParams.update(
            {
                "figure.figsize": self.figsize,
                "figure.dpi": self.dpi,
                "axes.grid": self.axes_grid,
                "grid.alpha": self.grid_alpha,
            }
        )
        torch.set_printoptions(
            precision=self.torch_precision, sci_mode=self.torch_sci_mode, linewidth=self.print_linewidth
        )
        np.set_printoptions(precision=self.torch_precision, suppress=False, linewidth=self.print_linewidth)
        pd.set_option("display.precision", self.pandas_precision)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)


def load_notebook_config(**overrides: Any) -> NotebookConfig:
    """Build a ``NotebookConfig`` and apply its presentation style.

    Args:
        **overrides: Field overrides forwarded to ``NotebookConfig``, e.g. ``load_notebook_config(show_plots=False)``.

    Returns:
        The configuration, with its style already applied.
    """
    config = NotebookConfig(**overrides)
    config.apply_style()
    return config

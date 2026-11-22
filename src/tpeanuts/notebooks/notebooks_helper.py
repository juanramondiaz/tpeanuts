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

"""Helpers shared by the tutorial notebooks: array conversion, error measures, checks, timing, reference models.

Only presentation, bookkeeping and the small calculations that several notebooks share live here; every physical
calculation of a notebook goes through the tpeanuts API. Figures are always written with :func:`save_and_show`, so
that all of them share one resolution and one location.

Module contents:
    FLAVOUR_NAMES, FLAVOUR_LABELS, FLAVOUR_COLORS, FLAVOUR_INDEX
        Names, LaTeX labels, colours and indices of the three active flavours.
    REL_FLOOR, ABSOLUTE_THRESHOLD, TOL_PPM, TOL_PPB
        Floor of the relative error, and the shared tolerance levels.
    to_numpy(...)
        Convert a tensor-like object to a CPU NumPy array.
    abs_rel_delta(...)
        Absolute and relative differences against a reference.
    add_tolerance_lines(...)
        Draw the ppm and ppb guide lines on an axis.
    status_from_rel(...)
        Map a maximum relative difference to a validation label.
    save_and_show(...)
        Save a figure under an output directory and display it.
    environment_report(...)
        Table of the software, hardware and run settings of a notebook, shown in its last cell.
    CheckLog
        Accumulate the checks of a notebook and report them as a table.
    best_time(...)
        Best wall time of a call after a warm-up, synchronizing CUDA.
    standard_model_from_preset(...)
        The three-flavour Standard Model of an oscillation preset on a runtime context.
    sterile_model_from_preset(...)
        The 3+1 model of a sterile oscillation preset on a runtime context.
    vacuum_evolutor(...)
        Vacuum evolution operator of a model over a baseline, for a batch of energies.
    vacuum_transition_numerical(...), vacuum_transition_closed_form(...)
        Vacuum transition probabilities from the model generator and from the explicit PMNS expression.
    matter_evolutor(...), matter_transition_numerical(...)
        Evolution operator and transition probabilities in a constant electron density.
    to_mass_basis(...)
        Express a flavour-basis density matrix in the vacuum mass basis.
    smear_log_energy(...)
        Average a function of the energy with a Gaussian window in the logarithm of the energy.
    energy_window_samples(...)
        Random energies drawn from a log-normal window around each energy of a grid.
    plot_density_matrix(...)
        Draw the moduli of the entries of a density matrix as an annotated map.
    purity(...), von_neumann_entropy(...)
        Purity and entropy of a density matrix.
"""

from __future__ import annotations

import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import tpeanuts as tp
from tpeanuts.medium.vacuum.vacuum import VacuumMedium
from tpeanuts.physics.common.terms import PropagationEnvironment
from tpeanuts.physics.evolution.numerical.numerical import segment_evolutor_numerical
from tpeanuts.physics.model import PropagationModel
from tpeanuts.util import constant
from tpeanuts.util.context import RuntimeContext

FLAVOUR_NAMES = ("nue", "numu", "nutau")
FLAVOUR_LABELS = [r"$\nu_e$", r"$\nu_\mu$", r"$\nu_\tau$"]
FLAVOUR_COLORS = ["C0", "C1", "C2"]
FLAVOUR_INDEX = {name: index for index, name in enumerate(FLAVOUR_NAMES)}

REL_FLOOR = 1.0e-12
ABSOLUTE_THRESHOLD = 1.0e-5
TOL_PPM = 1.0e-6
TOL_PPB = 1.0e-9


def to_numpy(x: Any) -> np.ndarray:
    """Convert a tensor-like object to a CPU NumPy array.

    Args:
        x: A ``torch.Tensor`` (any device, detached if needed) or anything ``numpy.asarray`` accepts.

    Returns:
        A NumPy array with the same values.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def abs_rel_delta(candidate: Any, reference: Any, *, floor: float = REL_FLOOR) -> Tuple[np.ndarray, np.ndarray]:
    """Absolute and relative differences of ``candidate`` against ``reference``.

    Args:
        candidate: The values under test.
        reference: The reference values, same shape.
        floor: Lower bound of ``|reference|`` in the relative error, which avoids dividing by zero.

    Returns:
        ``(|candidate - reference|, |candidate - reference| / max(|reference|, floor))``.
    """
    cand = to_numpy(candidate).astype(float)
    ref = to_numpy(reference).astype(float)
    absolute = np.abs(cand - ref)
    return absolute, absolute / np.maximum(np.abs(ref), floor)


def add_tolerance_lines(ax: Any) -> None:
    """Draw the 1 ppm and 1 ppb relative-error guide lines on ``ax``."""
    ax.axhline(TOL_PPM, color="dimgray", lw=1.0, ls="--", label="1 ppm")
    ax.axhline(TOL_PPB, color="lightgray", lw=1.0, ls=":", label="1 ppb")


def status_from_rel(max_rel: float) -> str:
    """Map a maximum relative difference to the shared validation label.

    Args:
        max_rel: Largest relative difference found.

    Returns:
        ``"PASS < ppb"``, ``"PASS < ppm"``, ``"CHECK < 1e-3"`` or ``"CHECK"``.
    """
    if max_rel < TOL_PPB:
        return "PASS < ppb"
    if max_rel < TOL_PPM:
        return "PASS < ppm"
    if max_rel < 1.0e-3:
        return "CHECK < 1e-3"
    return "CHECK"


def save_and_show(filename: str, fig: Any, *, output_dir: Path, show_plots: bool = True) -> Path:
    """Save a figure under ``output_dir`` and display it inline (or close it).

    Args:
        filename: File name with extension (``"01_survival.png"``).
        fig: The matplotlib figure.
        output_dir: Directory to write into; created if needed.
        show_plots: ``True`` shows the figure after saving it, ``False`` closes it.

    Returns:
        The path of the saved file (also printed).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)
    print(path)
    return path


class CheckLog:
    """Accumulates the checks of a notebook and reports them as one table.

    Each check has a name, the value that was measured, the acceptance criterion in words and whether it passed. The
    notebooks record their checks as they go and print the table in their summary.
    """

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []
        self.details: Dict[str, Dict[str, Any]] = {}

    def record(self, check: str, value: Any, criterion: str, passed: Any) -> bool:
        """Store one check.

        Args:
            check: Short name of what was checked.
            value: The measured value, as shown in the table.
            criterion: The acceptance criterion, in words.
            passed: Whether the criterion was met (anything truthy).

        Returns:
            ``bool(passed)``.
        """
        self.rows.append(
            {"Check": check, "Value": value, "Criterion": criterion, "Status": "PASS" if passed else "CHECK"}
        )
        return bool(passed)

    def table(self) -> pd.DataFrame:
        """The checks as a table indexed by name."""
        return pd.DataFrame(self.rows).set_index("Check")

    @property
    def all_passed(self) -> bool:
        """``True`` if every recorded check passed."""
        return all(row["Status"] == "PASS" for row in self.rows)

    def assert_all(self) -> None:
        """Raise ``AssertionError`` listing the failed checks, if any."""
        failed = [row["Check"] for row in self.rows if row["Status"] != "PASS"]
        assert not failed, f"checks failed: {failed}"

    def payload(self, notebook: str) -> Dict[str, Any]:
        """The checks of a notebook as plain JSON-serialisable data.

        Args:
            notebook: Name of the notebook that recorded them.

        Returns:
            ``{"notebook", "checks", "cases"}``: the rows of the table with their values as text, and, for the checks
            recorded through a validation case, the measured number, its tolerance and the cause of the tolerance.
        """

        def plain(value: Any) -> Any:
            return float(value) if isinstance(value, (int, float, np.floating)) else str(value)

        return {
            "notebook": notebook,
            "checks": [{key: str(value) for key, value in row.items()} for row in self.rows],
            "cases": {name: {k: plain(v) for k, v in detail.items()} for name, detail in self.details.items()},
        }

    def publish(self, notebook: str) -> None:
        """Store the checks in the output of the cell as a JSON display, so that they are part of the saved notebook.

        A summary notebook can read them from the stored outputs of this one without running it.

        Args:
            notebook: Name of the notebook that recorded them.
        """
        from IPython.display import display

        display({"application/json": self.payload(notebook)}, raw=True)


def environment_report(config: Any) -> pd.DataFrame:
    """Software, hardware and run settings of a notebook, as a two-column table.

    Args:
        config: The ``NotebookConfig`` of the notebook (its device, dtype, quick flag, seed and start time are
            reported).

    Returns:
        A ``pandas.DataFrame`` with columns ``Item`` and ``Value``: ``tpeanuts`` version and commit, Python, PyTorch,
        NumPy, pandas and matplotlib versions, platform, device (with the GPU name if any), dtype, quick mode, seed and
        the elapsed time since the configuration was built.
    """
    commit = "not a source checkout"
    if config.package_dir is not None:
        try:
            done = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=config.package_dir, capture_output=True, text=True, timeout=10,
            )
            commit = done.stdout.strip() or "unknown"
        except (OSError, subprocess.SubprocessError):
            commit = "unknown"
    device = str(config.device)
    if config.device.type == "cuda":
        device += f" ({torch.cuda.get_device_name(config.device)})"
    rows = [
        ("tpeanuts", f"{tp.__version__} (commit {commit})"),
        ("Python", sys.version.split()[0]),
        ("PyTorch", torch.__version__),
        ("NumPy / pandas / matplotlib", f"{np.__version__} / {pd.__version__} / {matplotlib.__version__}"),
        ("Platform", platform.platform()),
        ("Device / dtype", f"{device} / {config.dtype}"),
        ("Quick mode / seed", f"{config.quick} / {config.seed}"),
        ("Elapsed [s]", f"{time.time() - config.started_at:.1f}"),
    ]
    return pd.DataFrame(rows, columns=["Item", "Value"])


def best_time(function: Callable[[], Any], repeats: int) -> float:
    """Best wall time of ``function()`` over ``repeats`` calls, after one warm-up call.

    The warm-up absorbs one-off costs (table loading, kernel compilation). When CUDA is available the device is
    synchronized before and after every timed call, so asynchronous kernels are fully counted.

    Args:
        function: Callable without arguments.
        repeats: Number of timed repetitions.

    Returns:
        The smallest of the ``repeats`` wall times, in seconds.
    """
    function()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return min(times)


def standard_model_from_preset(
    runtime: RuntimeContext, preset: Mapping[str, Any], *, antinu: bool = False
) -> PropagationModel:
    """Build the three-flavour Standard Model of an oscillation preset on a runtime context.

    Args:
        runtime: Device and dtype of every tensor of the model.
        preset: A preset of ``tpeanuts.config.presets.OSCILLATION_PRESETS``: the angles ``theta12_deg``,
            ``theta13_deg``, ``theta23_deg``, the phase ``delta13_deg`` (degrees) and the splittings ``DeltamSq21``,
            ``DeltamSq3l`` (eV^2).
        antinu: ``True`` for the antineutrino convention.

    Returns:
        A Dirac-neutrino ``PropagationModel`` without extensions.
    """
    space = tp.FlavourSpace()
    standard = tp.StandardMixingParameters(
        theta12=math.radians(preset["theta12_deg"]), theta13=math.radians(preset["theta13_deg"]),
        theta23=math.radians(preset["theta23_deg"]), delta13=math.radians(preset["delta13_deg"]), context=runtime,
    )
    mixing = tp.MixingMatrix(space=space, standard=standard, nature=tp.Dirac(), context=runtime)
    masses = tp.MassSpectrum(
        space=space, DeltamSq21=preset["DeltamSq21"], DeltamSq3l=preset["DeltamSq3l"], context=runtime
    )
    return tp.PropagationModel(mixing=mixing, mass_spectrum=masses, antinu=antinu)


def sterile_model_from_preset(
    runtime: RuntimeContext, preset: Mapping[str, Any], *, antinu: bool = False, neutral_current: bool = True
) -> PropagationModel:
    """Build the 3+1 model of a sterile oscillation preset on a runtime context.

    Args:
        runtime: Device and dtype of every tensor of the model.
        preset: A sterile preset of ``tpeanuts.config.presets.OSCILLATION_PRESETS``: the standard keys of
            ``standard_model_from_preset`` plus ``theta14_deg``, ``theta24_deg``, ``theta34_deg``, ``delta14_deg``,
            ``delta24_deg`` and ``DeltamSq41`` (a missing ``theta34_deg`` or phase is taken as zero).
        antinu: ``True`` for the antineutrino convention.
        neutral_current: Whether the sterile neutral-current matter term is included. It needs the neutron density
            of the medium and is irrelevant in vacuum, so a vacuum calculation sets it to ``False``.

    Returns:
        A Dirac-neutrino ``PropagationModel`` on ``FlavourSpace(n_sterile=1)``.
    """
    from tpeanuts.physics import SterileMixingParameters
    from tpeanuts.physics.extensions import SterileNCTerm

    space = tp.FlavourSpace(n_sterile=1)
    standard = tp.StandardMixingParameters(
        theta12=math.radians(preset["theta12_deg"]), theta13=math.radians(preset["theta13_deg"]),
        theta23=math.radians(preset["theta23_deg"]), delta13=math.radians(preset["delta13_deg"]), context=runtime,
    )

    def angle(key: str) -> List[float]:
        return [math.radians(float(preset.get(key, 0.0)))]

    sterile = SterileMixingParameters(
        theta_e=angle("theta14_deg"), theta_mu=angle("theta24_deg"), theta_tau=angle("theta34_deg"),
        delta_e=angle("delta14_deg"), delta_mu=angle("delta24_deg"), delta_tau=angle("delta34_deg"), context=runtime,
    )
    mixing = tp.MixingMatrix(space=space, standard=standard, nature=tp.Dirac(), context=runtime, sterile=sterile)
    masses = tp.MassSpectrum(
        space=space, DeltamSq21=preset["DeltamSq21"], DeltamSq3l=preset["DeltamSq3l"], context=runtime,
        DeltamSq_sterile=[preset["DeltamSq41"]],
    )
    extensions = (SterileNCTerm(),) if neutral_current else ()
    return tp.PropagationModel(mixing=mixing, mass_spectrum=masses, extensions=extensions, antinu=antinu)


def vacuum_evolutor(model: PropagationModel, energies_MeV: torch.Tensor, baseline_m: float) -> torch.Tensor:
    """Vacuum flavour evolution operator ``S`` over a baseline, from the model generator and a matrix exponential.

    Args:
        model: Three-flavour propagation model.
        energies_MeV: Neutrino energies, any batch shape.
        baseline_m: Baseline in metres (also the length scale of the dimensionless generator).

    Returns:
        Complex tensor shaped ``(*energies_MeV.shape, 3, 3)``; ``S[..., final, initial]`` is the amplitude.
    """
    return VacuumMedium(length_m=baseline_m, context=model.context).evolutor(model, energies_MeV)


def vacuum_transition_numerical(model: PropagationModel, energies_MeV: torch.Tensor, baseline_m: float) -> torch.Tensor:
    """Vacuum transition probabilities ``P[..., final, initial]``, the squared moduli of ``vacuum_evolutor``.

    Args:
        model: Three-flavour propagation model.
        energies_MeV: Neutrino energies, any batch shape.
        baseline_m: Baseline in metres.

    Returns:
        ``|S|^2``, shaped ``(*energies_MeV.shape, 3, 3)``.
    """
    return vacuum_evolutor(model, energies_MeV, baseline_m).abs() ** 2


def matter_evolutor(
    model: PropagationModel,
    energies_MeV: torch.Tensor,
    baseline_m: float,
    n_e_mol_cm3: float,
    n_n_mol_cm3: Optional[float] = None,
) -> torch.Tensor:
    """Flavour evolution operator ``S`` over a baseline of constant electron density.

    Args:
        model: Three-flavour propagation model.
        energies_MeV: Neutrino energies, any batch shape.
        baseline_m: Baseline in metres (also the length scale of the dimensionless generator).
        n_e_mol_cm3: Electron number density in mol/cm^3 (zero reproduces ``vacuum_evolutor``).
        n_n_mol_cm3: Neutron number density in mol/cm^3, needed only by the extensions that couple to neutrons (the
            sterile neutral-current term, NSI with neutron couplings); ``None`` leaves it undefined.

    Returns:
        Complex tensor shaped ``(*energies_MeV.shape, 3, 3)``; ``S[..., final, initial]`` is the amplitude.
    """
    environment = PropagationEnvironment(E_MeV=energies_MeV, n_e_mol_cm3=n_e_mol_cm3, n_n_mol_cm3=n_n_mol_cm3)
    generator = model.generator_flavour(environment, evolution_scale_m=baseline_m)
    return segment_evolutor_numerical(generator, 1.0)


def matter_transition_numerical(
    model: PropagationModel,
    energies_MeV: torch.Tensor,
    baseline_m: float,
    n_e_mol_cm3: float,
    n_n_mol_cm3: Optional[float] = None,
) -> torch.Tensor:
    """Transition probabilities ``P[..., final, initial]`` in a constant density, ``|matter_evolutor|^2``."""
    return matter_evolutor(model, energies_MeV, baseline_m, n_e_mol_cm3, n_n_mol_cm3).abs() ** 2


def vacuum_transition_closed_form(
    model: PropagationModel, energies_MeV: torch.Tensor, baseline_m: float
) -> torch.Tensor:
    """Vacuum transition probabilities from the explicit expression in terms of the PMNS matrix.

    ``P_ab = delta_ab - 4 sum_{i>j} Re(U*_ai U_bi U_aj U*_bj) sin^2(D_ij) + 2 sum_{i>j} Im(...) sin(2 D_ij)`` with
    ``D_ij = Delta m^2_ij L / (4 E)``; it uses only the mixing matrix and the two mass splittings of the model, so it
    is independent of the evolution machinery.

    Args:
        model: Three-flavour propagation model.
        energies_MeV: Neutrino energies, any batch shape.
        baseline_m: Baseline in metres.

    Returns:
        ``P[..., final, initial]`` shaped ``(*energies_MeV.shape, 3, 3)``.
    """
    U = model.mixing.pmns_matrix(antinu=model.antinu)
    dm21, dm31 = float(model.mass_spectrum.DeltamSq21), float(model.mass_spectrum.DeltamSq3l)
    splittings = {(1, 0): dm21, (2, 0): dm31, (2, 1): dm31 - dm21}
    scale = 1.0e-12 * baseline_m / (4.0 * constant.HBARC_MeV_m * energies_MeV)
    P = torch.zeros(*energies_MeV.shape, 3, 3, device=energies_MeV.device, dtype=energies_MeV.dtype)
    for a in range(3):
        for b in range(3):
            kronecker = 1.0 if a == b else 0.0
            term = torch.full(energies_MeV.shape, kronecker, device=energies_MeV.device, dtype=energies_MeV.dtype)
            for (i, j), dm2_ij in splittings.items():
                product = U[a, i].conj() * U[b, i] * U[a, j] * U[b, j].conj()
                phase = dm2_ij * scale
                term = term - 4.0 * product.real * torch.sin(phase) ** 2 + 2.0 * product.imag * torch.sin(2.0 * phase)
            P[..., b, a] = term
    return P


def purity(rho: Any) -> torch.Tensor:
    """Purity ``Tr(rho^2)`` of a density matrix: 1 for a pure state, ``1/d`` for the maximally mixed state.

    Args:
        rho: A ``DensityMatrix`` or a complex tensor shaped ``(..., d, d)``.

    Returns:
        Real tensor shaped like the batch of ``rho``.
    """
    matrix = getattr(rho, "tensor", rho)
    return torch.einsum("...ij,...ji->...", matrix, matrix).real


def von_neumann_entropy(rho: Any) -> torch.Tensor:
    """Von Neumann entropy ``-Tr(rho ln rho)`` in nats, from the eigenvalues of ``rho``.

    Args:
        rho: A ``DensityMatrix`` or a Hermitian tensor shaped ``(..., d, d)`` with unit trace.

    Returns:
        Real tensor shaped like the batch of ``rho``: 0 for a pure state, ``ln d`` for the maximally mixed state.
    """
    matrix = getattr(rho, "tensor", rho)
    eigenvalues = torch.linalg.eigvalsh(matrix).clamp_min(0.0)
    safe = torch.where(eigenvalues > 0.0, eigenvalues, torch.ones_like(eigenvalues))
    return -(eigenvalues * torch.log(safe)).sum(dim=-1)


def to_mass_basis(model: PropagationModel, rho: Any) -> torch.Tensor:
    """Express a flavour-basis density matrix in the vacuum mass basis, ``rho^m = U^dagger rho^f U``.

    Args:
        model: Three-flavour propagation model whose PMNS matrix defines the mass basis.
        rho: A ``DensityMatrix`` or a complex tensor shaped ``(..., 3, 3)`` in the flavour basis.

    Returns:
        The matrix in the mass basis, same shape as ``rho``; its diagonal holds the mass-state populations.
    """
    U = model.mixing.pmns_matrix(antinu=model.antinu)
    matrix = getattr(rho, "tensor", rho)
    return U.conj().transpose(-2, -1) @ matrix @ U


def smear_log_energy(energies_MeV: torch.Tensor, values: torch.Tensor, width: float) -> torch.Tensor:
    """Average ``values`` over a Gaussian window of standard deviation ``width`` in ``ln E``.

    A finite energy resolution of a detector, or a spread of production energies, averages the rapid oscillations:
    each output point is the mean of ``values`` weighted by ``exp[-(ln E' - ln E)^2 / (2 width^2)]`` over the grid
    (normalized on the grid, so the edges are averaged over the available points only).

    Args:
        energies_MeV: Strictly positive energy grid, shape ``(n,)``.
        values: Function sampled on the grid, shape ``(n, ...)``.
        width: Relative energy resolution (standard deviation of ``ln E``).

    Returns:
        The smeared function, same shape as ``values``.
    """
    log_energy = torch.log(energies_MeV)
    weights = torch.exp(-0.5 * ((log_energy[:, None] - log_energy[None, :]) / width) ** 2)
    weights = weights / weights.sum(dim=1, keepdim=True)
    return torch.tensordot(weights, values, dims=([1], [0]))


def plot_density_matrix(
    ax: Any, rho: Any, labels: Sequence[str], title: str, *, vmax: float = 1.0
) -> None:
    """Draw ``|rho_ij|`` as an annotated map, rows and columns labelled by the basis states.

    Args:
        ax: Matplotlib axis to draw on.
        rho: A ``DensityMatrix`` or a ``(d, d)`` complex tensor (one matrix, no batch).
        labels: Names of the ``d`` basis states, used on both axes.
        title: Title of the panel.
        vmax: Upper end of the colour scale.
    """
    moduli = np.abs(to_numpy(getattr(rho, "tensor", rho)))
    image = ax.imshow(moduli, cmap="Blues", vmin=0.0, vmax=vmax)
    size = moduli.shape[0]
    ax.set_xticks(range(size), labels)
    ax.set_yticks(range(size), labels)
    ax.grid(False)
    for row in range(size):
        for column in range(size):
            value = moduli[row, column]
            colour = "white" if value > 0.6 * vmax else "black"
            ax.text(column, row, f"{value:.3f}", ha="center", va="center", color=colour)
    ax.set_title(title)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def energy_window_samples(
    energies_MeV: torch.Tensor, width: float, n_samples: int, *, seed: int = 0
) -> torch.Tensor:
    """Draw ``n_samples`` energies from a log-normal window of relative width ``width`` around each energy.

    Averaging a function that oscillates faster than any practical grid (large phases) over a finite energy
    resolution is done by sampling: the mean of the function over these energies estimates its Gaussian average in
    ``ln E`` with a statistical error of order ``std / sqrt(n_samples)``. The draw is deterministic for a given seed.

    Args:
        energies_MeV: Central energies, any shape ``(...)``.
        width: Standard deviation of ``ln E`` of the window.
        n_samples: Number of energies per central energy.
        seed: Seed of the random generator.

    Returns:
        Tensor of energies shaped ``(*energies_MeV.shape, n_samples)``.
    """
    generator = torch.Generator(device=energies_MeV.device).manual_seed(seed)
    z = torch.randn(
        (*energies_MeV.shape, n_samples), generator=generator, device=energies_MeV.device, dtype=energies_MeV.dtype
    )
    return energies_MeV[..., None] * torch.exp(width * z)

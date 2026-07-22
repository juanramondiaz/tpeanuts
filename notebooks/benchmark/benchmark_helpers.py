#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared helper functions for benchmark notebooks.

The benchmark notebooks intentionally keep their configuration cells local:
grid sizes, output directories, loaded profiles, and backend-specific
configuration differ between the legacy-peanuts and nuSQuIDS notebooks. This
module stores the helper logic in one place and is configured from a notebook
namespace with ``configure_benchmark_helpers(globals(), backend=...)``.
"""

from __future__ import annotations

import gc
import math
import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.core.common.pmns import PMNSParams
from tpeanuts.core.SM.sm_mass_spectrum import MassSpectrum_SM
from tpeanuts.core.SM.sm_pmns import PMNS_SM
from tpeanuts.medium.earth.profile import EarthParameters, EarthProfile
from tpeanuts.medium.earth.probability import earth_probability_state
from tpeanuts.medium.solar.profile import SolarParameters, SolarProfile
from tpeanuts.notebooks.notebooks_helper import FLAVOUR_NAMES, save_and_show, to_numpy
from tpeanuts.util.context import RuntimeContext


BENCHMARK_BACKEND = "legacy"


def synchronize_device(device: torch.device | str) -> None:
    """Synchronize pending work when *device* is a CUDA device."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def release_benchmark_memory(
    device: torch.device | str,
    *,
    empty_cuda_cache: bool = True,
) -> None:
    """Collect Python garbage and optionally release unused CUDA cache blocks."""
    gc.collect()
    device = torch.device(device)
    if empty_cuda_cache and device.type == "cuda":
        torch.cuda.empty_cache()


def process_ram_gb(process: psutil.Process | None = None) -> float:
    """Return resident memory used by the current process in GiB."""
    process = psutil.Process(os.getpid()) if process is None else process
    return process.memory_info().rss / 1024**3


def check_ram_budget(
    max_ram_gb: float,
    *,
    process: psutil.Process | None = None,
) -> None:
    """Raise ``MemoryError`` when process RSS exceeds *max_ram_gb*."""
    used_gb = process_ram_gb(process)
    if used_gb > max_ram_gb:
        raise MemoryError(
            f"Process RAM {used_gb:.2f} GiB exceeds "
            f"max_ram_gb={max_ram_gb:.2f} GiB"
        )


def iter_grid_chunks(
    energy: torch.Tensor,
    angles: torch.Tensor,
    *,
    energy_chunk_size: int,
    angle_chunk_size: int,
    max_ram_gb: float | None = None,
    process: psutil.Process | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield Cartesian energy-angle chunks while enforcing an RSS budget."""
    if energy_chunk_size <= 0 or angle_chunk_size <= 0:
        raise ValueError("Chunk sizes must be positive integers.")
    for energy_start in range(0, energy.numel(), energy_chunk_size):
        for angle_start in range(0, angles.numel(), angle_chunk_size):
            if max_ram_gb is not None:
                check_ram_budget(max_ram_gb, process=process)
            yield (
                energy[energy_start : energy_start + energy_chunk_size],
                angles[angle_start : angle_start + angle_chunk_size],
            )


def reduce_tensor_chunks(
    chunks: Iterable[torch.Tensor],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    max_ram_gb: float | None = None,
    process: psutil.Process | None = None,
) -> torch.Tensor:
    """Materialize tensor chunks and return a scalar checksum without retaining them."""
    checksum = torch.zeros((), device=device, dtype=dtype)
    for value in chunks:
        checksum = checksum + value.real.sum()
        del value
        if max_ram_gb is not None:
            check_ram_budget(max_ram_gb, process=process)
    return checksum


def _sample_resources(
    stop: threading.Event,
    peaks: dict[str, float],
    *,
    process: psutil.Process,
    device: torch.device,
    interval_s: float,
    cpu_normalization_cores: int,
) -> None:
    process.cpu_percent(None)
    while not stop.wait(interval_s):
        peaks["ram_gb"] = max(peaks["ram_gb"], process_ram_gb(process))
        normalized_cpu = process.cpu_percent(None) / cpu_normalization_cores
        peaks["cpu_percent"] = max(
            peaks["cpu_percent"], min(100.0, normalized_cpu)
        )
        if device.type == "cuda":
            try:
                peaks["gpu_percent"] = max(
                    peaks["gpu_percent"], float(torch.cuda.utilization(device))
                )
            except (ModuleNotFoundError, RuntimeError):
                pass


def measure_resources(
    func: Callable[[], torch.Tensor],
    *,
    device: torch.device | str,
    max_ram_gb: float,
    sample_interval_s: float = 0.02,
    empty_cuda_cache: bool = True,
    cpu_normalization_cores: int | None = None,
    process: psutil.Process | None = None,
) -> tuple[float, dict[str, float]]:
    """Time a tensor callable and return elapsed seconds and peak resources.

    CPU use is normalized by ``cpu_normalization_cores`` and therefore lies
    between 0 and 100 percent. By default the denominator is the machine's
    logical CPU count; pass a benchmark thread budget to measure utilization
    relative to that explicitly configured capacity.
    """
    device = torch.device(device)
    process = psutil.Process(os.getpid()) if process is None else process
    cpu_normalization_cores = (
        psutil.cpu_count(logical=True)
        if cpu_normalization_cores is None
        else int(cpu_normalization_cores)
    )
    if cpu_normalization_cores <= 0:
        raise ValueError("cpu_normalization_cores must be a positive integer.")
    release_benchmark_memory(device, empty_cuda_cache=empty_cuda_cache)
    check_ram_budget(max_ram_gb, process=process)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    peaks = {
        "ram_gb": process_ram_gb(process),
        "cpu_percent": 0.0,
        "gpu_percent": 0.0,
    }
    stop = threading.Event()
    sampler = threading.Thread(
        target=_sample_resources,
        kwargs={
            "stop": stop,
            "peaks": peaks,
            "process": process,
            "device": device,
            "interval_s": sample_interval_s,
            "cpu_normalization_cores": cpu_normalization_cores,
        },
        daemon=True,
    )
    sampler.start()
    synchronize_device(device)
    start = time.perf_counter()
    try:
        result = func()
        synchronize_device(device)
        elapsed = time.perf_counter() - start
    finally:
        stop.set()
        sampler.join()
    peaks["ram_gb"] = max(peaks["ram_gb"], process_ram_gb(process))
    peaks["vram_allocated_gb"] = (
        torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    )
    peaks["vram_reserved_gb"] = (
        torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else 0.0
    )
    if not torch.isfinite(result).all():
        raise ValueError("Benchmark callable returned a non-finite tensor.")
    return elapsed, peaks


def timed_resources(
    func: Callable[[], torch.Tensor],
    *,
    device: torch.device | str,
    repeats: int,
    warmups: int,
    max_ram_gb: float,
    sample_interval_s: float = 0.02,
    empty_cuda_cache: bool = True,
    cpu_normalization_cores: int | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Run warmups and repeated resource-monitored timings of a tensor callable."""
    if repeats <= 0 or warmups < 0:
        raise ValueError("repeats must be positive and warmups cannot be negative.")
    for _ in range(warmups):
        func()
        synchronize_device(device)
    measurements = [
        measure_resources(
            func,
            device=device,
            max_ram_gb=max_ram_gb,
            sample_interval_s=sample_interval_s,
            empty_cuda_cache=empty_cuda_cache,
            cpu_normalization_cores=cpu_normalization_cores,
        )
        for _ in range(repeats)
    ]
    samples = np.asarray([measurement[0] for measurement in measurements])
    resources = {
        key: max(measurement[1][key] for measurement in measurements)
        for key in measurements[0][1]
    }
    return samples, resources


def configure_benchmark_helpers(namespace: dict[str, Any], *, backend: str | None = None) -> None:
    """Expose notebook configuration values to this helper module.

    Args:
        namespace: Usually ``globals()`` from the benchmark notebook.
        backend: Optional reference backend selector: ``"legacy"`` or
            ``"nusquids"``. If omitted, it is inferred from the namespace.
    """
    globals().update(namespace)
    selected = backend
    if selected is None:
        selected = "nusquids" if "NSQ_CFG" in namespace or "NSQ_AVAILABLE" in namespace else "legacy"
    globals()["BENCHMARK_BACKEND"] = str(selected).lower().strip()


def _backend_settings() -> dict[str, str]:
    if BENCHMARK_BACKEND == "nusquids":
        return {
            "label": "nuSQuIDS",
            "short": "nsq",
            "best_col": "nusquids_best_s",
            "mean_col": "nusquids_mean_s",
            "std_col": "nusquids_std_s",
            "speedup_col": "speedup_nusquids_over_tpeanuts",
            "speedup_label": "Speedup (nuSQuIDS / tpeanuts)",
            "nadir_label": "Nadir/cos-zenith-grid size",
        }
    return {
        "label": "legacy",
        "short": "leg",
        "best_col": "legacy_best_s",
        "mean_col": "legacy_mean_s",
        "std_col": "legacy_std_s",
        "speedup_col": "speedup_legacy_over_tpeanuts",
        "speedup_label": "Speedup (legacy / tpeanuts)",
        "nadir_label": "Nadir-grid size",
    }


def energy_grid(n, *, solar=True):
    """Build the standard benchmark energy grid."""
    if solar:
        return torch.linspace(0.5, 15.0, int(n), dtype=DTYPE, device=DEVICE)
    return torch.logspace(
        torch.log10(torch.tensor(100.0, dtype=DTYPE)),
        torch.log10(torch.tensor(2.0e4, dtype=DTYPE)),
        int(n),
        dtype=DTYPE,
        device=DEVICE,
    )


def nadir_grid(n):
    """Build the standard benchmark nadir-angle grid."""
    return torch.linspace(0.05, math.pi - 0.05, int(n), dtype=DTYPE, device=DEVICE)


def cos_zenith_from_nadir(eta):
    """Convert tpeanuts nadir angle to nuSQuIDS cos(zenith)."""
    return -torch.cos(eta)


def synthetic_flux(E_MeV):
    """Synthetic three-flavour atmospheric-like flux used for timing."""
    x = E_MeV / 1000.0
    return torch.stack(
        [
            1.00e2 * x.pow(-2.15) * torch.exp(-E_MeV / 5.0e4),
            2.50e2 * x.pow(-2.05) * torch.exp(-E_MeV / 6.0e4),
            1.50e1 * x.pow(-2.00) * torch.exp(-E_MeV / 7.0e4),
        ],
        dim=-1,
    )


def solar_source_spectrum(E_MeV):
    """Simple normalized spectrum shape for solar-flux timing."""
    E = torch.as_tensor(E_MeV, dtype=DTYPE, device=DEVICE)
    shape = torch.clamp(E * (16.0 - E), min=0.0) ** 2
    return shape / torch.max(shape).clamp_min(1.0e-30)


def timed_call(func, repeats=None, warmups=None):
    """Run warmups, then timed repeats, and return best/mean/std seconds."""
    repeats = TIMING_REPEATS if repeats is None else repeats
    warmups = TIMING_WARMUPS if warmups is None else warmups
    for _ in range(warmups):
        func()
    times = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    return min(times), float(np.mean(times)), float(np.std(times))


def benchmark_pair(section, ne, neta, tpeanuts_func, reference_func):
    """Benchmark tpeanuts against the configured reference backend."""
    cfg = _backend_settings()
    t_best, t_mean, t_std = timed_call(tpeanuts_func)
    r_best, r_mean, r_std = timed_call(reference_func)
    speedup = r_best / max(t_best, 1.0e-15)
    tag = f"NE={ne:4d} | Neta={str(neta):>4s}"
    print(
        f"{section:34s} | {tag} | t={t_best:.4e}s+/-{t_std:.1e} | "
        f"{cfg['short']}={r_best:.4e}s | x{speedup:.2f}"
    )
    return {
        "section": section,
        "n_energy": int(ne),
        "n_nadir": None if neta is None else int(neta),
        "tpeanuts_best_s": t_best,
        cfg["best_col"]: r_best,
        "tpeanuts_mean_s": t_mean,
        cfg["mean_col"]: r_mean,
        "tpeanuts_std_s": t_std,
        cfg["std_col"]: r_std,
        cfg["speedup_col"]: speedup,
    }


def save_results(df, name):
    """Save one benchmark DataFrame under the configured output directory."""
    path = OUTPUT_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def _fit_slope(x, y):
    """Log-log linear fit; returns slope."""
    mask = (np.array(x) > 0) & (np.array(y) > 0)
    if mask.sum() < 2:
        return float("nan")
    lx, ly = np.log10(np.array(x)[mask]), np.log10(np.array(y)[mask])
    return float(np.polyfit(lx, ly, 1)[0])


def plot_energy_scaling(df, title, filename):
    """Log-log timing plot with speedup panel."""
    cfg = _backend_settings()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.errorbar(
        df["n_energy"],
        df["tpeanuts_best_s"],
        yerr=df["tpeanuts_std_s"],
        marker="o",
        capsize=3,
        label="tpeanuts",
    )
    ax.errorbar(
        df["n_energy"],
        df[cfg["best_col"]],
        yerr=df[cfg["std_col"]],
        marker="x",
        capsize=3,
        label=cfg["label"],
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    slope_t = _fit_slope(df["n_energy"], df["tpeanuts_best_s"])
    slope_r = _fit_slope(df["n_energy"], df[cfg["best_col"]])
    ax.set_title(f"{title}\nslope tpeanuts={slope_t:.2f}, {cfg['label']}={slope_r:.2f}")
    ax.set_xlabel("Energy-grid size")
    ax.set_ylabel("Best time [s]")
    ax.legend()

    ax2 = axes[1]
    ax2.semilogx(df["n_energy"], df[cfg["speedup_col"]], marker="o", color="C2")
    ax2.axhline(1.0, color="black", lw=1.0, alpha=0.5)
    ax2.set_xlabel("Energy-grid size")
    ax2.set_ylabel(cfg["speedup_label"])
    ax2.set_title("Speedup")
    fig.tight_layout()
    save_and_show(filename, fig, output_dir=OUTPUT_DIR, show_plots=SHOW_PLOTS)


def plot_heatmap(df, title, filename):
    """Speedup heat-map with LogNorm colour scale."""
    cfg = _backend_settings()
    pivot = df.pivot(index="n_nadir", columns="n_energy", values=cfg["speedup_col"])
    vals = pivot.values
    vmin, vmax = max(vals.min(), 1.0e-2), vals.max()
    norm = mcolors.LogNorm(vmin=vmin, vmax=max(vmax, vmin * 2))
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(vals, origin="lower", aspect="auto", norm=norm, cmap="RdYlGn")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(v) for v in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index])
    for r in range(vals.shape[0]):
        for c in range(vals.shape[1]):
            ax.text(c, r, f"{vals[r, c]:.1f}x", ha="center", va="center", fontsize=7)
    ax.set_xlabel("Energy-grid size")
    ax.set_ylabel(cfg["nadir_label"])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Speedup (log scale)")
    fig.tight_layout()
    save_and_show(filename, fig, output_dir=OUTPUT_DIR, show_plots=SHOW_PLOTS)


def plot_speedup_cross_sections(df, title, filename):
    """Speedup slices versus energy-grid and nadir-grid size."""
    cfg = _backend_settings()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for neta in sorted(df["n_nadir"].dropna().unique()):
        sub = df[df["n_nadir"] == neta].sort_values("n_energy")
        axes[0].semilogx(sub["n_energy"], sub[cfg["speedup_col"]], marker="o", label=f"Neta={int(neta)}")
    for ne in sorted(df["n_energy"].unique()):
        sub = df[df["n_energy"] == ne].sort_values("n_nadir")
        axes[1].semilogx(sub["n_nadir"], sub[cfg["speedup_col"]], marker="o", label=f"NE={int(ne)}")
    for ax in axes:
        ax.axhline(1.0, color="black", lw=1.0, alpha=0.5)
    axes[0].set_xlabel("Energy-grid size")
    axes[0].set_ylabel("Speedup")
    axes[0].set_title("Speedup vs energy grid")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel(cfg["nadir_label"])
    axes[1].set_ylabel("Speedup")
    axes[1].set_title("Speedup vs nadir grid")
    axes[1].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    save_and_show(filename, fig, output_dir=OUTPUT_DIR, show_plots=SHOW_PLOTS)


def plot_grid_resource_scaling(
    frame: pd.DataFrame,
    *,
    title: str,
    filename: str,
    output_dir,
    show_plots: bool,
    methods: tuple[str, ...] = ("perturbative", "numerical"),
) -> None:
    """Plot peak resources against energy and angular grid sizes.

    The upper row shows the maximum over all angular grids at each energy-grid
    size. The lower row shows the maximum over all energy grids at each
    angular-grid size. Columns contain process RAM, allocated VRAM, normalized
    CPU utilization and GPU utilization, respectively.
    """
    metrics = (
        ("ram_gb", "Peak RAM [GiB]"),
        ("vram_allocated_gb", "Peak allocated VRAM [GiB]"),
        ("cpu_percent", "Peak CPU [%]"),
        ("gpu_percent", "Peak GPU [%]"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=False)
    for row, (grid_column, xlabel) in enumerate(
        (("n_energy", "Energy-grid size"), ("n_angle", "Angular-grid size"))
    ):
        for column, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, column]
            for method in methods:
                values = (
                    frame.groupby(grid_column, as_index=False)[f"{method}_{metric}"]
                    .max()
                    .sort_values(grid_column)
                )
                ax.plot(
                    values[grid_column],
                    values[f"{method}_{metric}"],
                    "o-",
                    label=method.capitalize(),
                )
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted(frame[grid_column].unique()))
            ax.set_xticklabels([str(value) for value in sorted(frame[grid_column].unique())])
            ax.set(xlabel=xlabel, ylabel=ylabel, title=ylabel)
            if metric.endswith("percent"):
                ax.set_ylim(0.0, 105.0)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    fig.suptitle(f"{title} — peak resource scaling")
    fig.tight_layout()
    save_and_show(filename, fig, output_dir=output_dir, show_plots=show_plots)


_CONTEXT_CACHE: dict = {}
_OSCILLATION_CACHE: dict = {}
_EARTH_PROFILE_CACHE: dict = {}
_SOLAR_PROFILE_CACHE: dict = {}


def _context_for(device):
    if device == DEVICE:
        return context
    if device not in _CONTEXT_CACHE:
        _CONTEXT_CACHE[device] = RuntimeContext.resolve(device, DTYPE)
    return _CONTEXT_CACHE[device]


def _oscillation_for(device):
    # Cached per device: rebuilding this (and the profiles below) inside a
    # timed benchmark call would measure object construction / disk I/O
    # instead of the actual tensor computation, which previously dominated
    # the GPU-vs-CPU timing comparison at every grid size.
    if device == DEVICE:
        return oscillation
    if device not in _OSCILLATION_CACHE:
        _ctx = _context_for(device)
        _pmns = PMNS_SM(PMNSParams(
            theta12=THETA12, theta13=THETA13, theta23=THETA23, delta=DELTA_CP, context=_ctx,
        ))
        _mass_spectrum = MassSpectrum_SM(
            DeltamSq21=torch.as_tensor(DM21_EV2, device=_ctx.device, dtype=_ctx.dtype),
            DeltamSq3l=torch.as_tensor(DM3L_EV2, device=_ctx.device, dtype=_ctx.dtype),
        )
        _OSCILLATION_CACHE[device] = OscillationParameters(
            pmns=_pmns, mass_spectrum=_mass_spectrum, antinu=False,
        )
    return _OSCILLATION_CACHE[device]


def _earth_profile_for(device):
    if device == DEVICE:
        return earth_profile_t
    if device not in _EARTH_PROFILE_CACHE:
        _EARTH_PROFILE_CACHE[device] = EarthProfile(
            params=EarthParameters(
                profile_perturbative_kwargs={"density_file": EARTH_DENSITY_TPEANUTS, "tabulated_density": False},
            ),
            context=_context_for(device),
        )
    return _EARTH_PROFILE_CACHE[device]


def _solar_profile_for(device):
    if device == DEVICE:
        return solar_profile
    if device not in _SOLAR_PROFILE_CACHE:
        _SOLAR_PROFILE_CACHE[device] = SolarProfile.default(
            context=_context_for(device),
        )
    return _SOLAR_PROFILE_CACHE[device]


def torch_pearth_probability(state, E_MeV, eta, depth_m, *, massbasis=True, device=None):
    """tpeanuts Earth probability helper shared by benchmark notebooks."""
    dev = device if device is not None else DEVICE
    if BENCHMARK_BACKEND == "nusquids":
        return earth_probability_state_analytical(state, earth_profile_t, oscillation, E_MeV, eta, depth_m, massbasis=massbasis)

    ctx = _context_for(dev)
    E_t = torch.as_tensor(E_MeV, device=dev, dtype=DTYPE)
    eta_t = torch.as_tensor(eta, device=dev, dtype=DTYPE)
    state_t = torch.as_tensor(state, device=dev)
    osc_dev = _oscillation_for(dev)
    profile_dev = _earth_profile_for(dev)
    if EARTH_METHOD == "analytical":
        return earth_probability_state(state_t, profile_dev, osc_dev, E_t, eta_t, depth_m, method="analytical", massbasis=massbasis, context=ctx)
    if EARTH_METHOD == "numerical":
        E_b, eta_b = torch.broadcast_tensors(E_t, eta_t)
        return torch.stack(
            [
                earth_probability_state(
                    state_t,
                    profile_dev,
                    osc_dev,
                    ev,
                    etav,
                    depth_m,
                    method="numerical",
                    massbasis=massbasis,
                    nsteps=EARTH_NUMERICAL_STEPS,
                    ode_method=EARTH_ODE_METHOD,
                    context=ctx,
                )
                for ev, etav in zip(E_b.reshape(-1), eta_b.reshape(-1))
            ]
        ).reshape(*E_b.shape, 3)
    raise ValueError("EARTH_METHOD must be analytical or numerical.")


def legacy_vacuum_matrix(E, L_km):
    E_np = to_numpy(E).reshape(-1)
    matrix = np.empty((len(E_np), 3, 3))
    for i, energy in enumerate(E_np):
        for alpha in range(3):
            matrix[i, :, alpha] = legacy_pvacuum(
                np.eye(3)[alpha],
                legacy_pmns,
                DM21_EV2,
                DM3L_EV2,
                float(energy),
                L_km,
                massbasis=False,
            )
    return matrix


def nusquids_vacuum_matrix(E_MeV, L_km):
    E_np = to_numpy(E_MeV).reshape(-1)
    matrix = np.empty((len(E_np), 3, 3))
    for i, energy in enumerate(E_np):
        for alpha, flavour in enumerate(FLAVOUR_NAMES):
            matrix[i, :, alpha] = nusquids_probability_vacuum(
                E_GeV=float(energy) * 1.0e-3,
                baseline_km=L_km,
                initial_flavour=flavour,
                config=NSQ_CFG,
            )
    return matrix


def tpeanuts_solar_point(E_MeV):
    ne_r0 = solar_profile.electron_density(torch.tensor(SOLAR_R0, dtype=DTYPE, device=DEVICE))
    return Tei(oscillation, E_MeV, ne_r0.unsqueeze(0).expand(E_MeV.numel()))


def nusquids_solar_point(E_MeV):
    E_np = to_numpy(E_MeV).reshape(-1)
    return np.stack(
        [
            nusquids_probability_solar_point(
                E_MeV=float(e),
                r0=SOLAR_R0,
                initial_flavour="nue",
                config=NSQ_CFG_SOLAR,
                return_mass_weights=True,
            )
            for e in E_np
        ]
    )


def tpeanuts_solar_mass(E, device=None):
    dev = device if device is not None else DEVICE
    if BENCHMARK_BACKEND == "nusquids":
        return solar_probability_mass(oscillation, E, solar_profile, SOLAR_SOURCE)
    return solar_probability_mass(
        _oscillation_for(dev),
        torch.as_tensor(E, device=dev, dtype=DTYPE),
        _solar_profile_for(dev),
        SOLAR_SOURCE,
        legacy_precision=SOLAR_LEGACY_PRECISION,
    )


def legacy_solar_mass(E):
    return np.stack(
        [
            legacy_solar_module.solar_flux_mass(
                legacy_pmns_solar.theta12,
                legacy_pmns_solar.theta13,
                DM21_EV2,
                DM3L_EV2,
                float(e),
                legacy_model.radius(),
                legacy_model.density(),
                legacy_model.fraction(SOLAR_SOURCE),
            )
            for e in to_numpy(E)
        ]
    )


def tpeanuts_integrated(E_1d, eta_1d, depth_m, device=None):
    dev = device if device is not None else DEVICE
    E = torch.as_tensor(E_1d, device=dev, dtype=DTYPE)
    eta = torch.as_tensor(eta_1d, device=dev, dtype=DTYPE)
    mass = tpeanuts_solar_mass(E, device=dev)
    P_eta = torch_pearth_probability(mass, E[:, None], eta[None, :], depth_m, massbasis=True, device=dev)
    exposure = torch.ones_like(eta)
    exposure = exposure / torch.trapz(exposure, x=eta).clamp_min(torch.finfo(DTYPE).tiny)
    return torch.trapz(P_eta * exposure[None, :, None], x=eta, dim=1)


def legacy_integrated(E_np, eta_np, depth_m):
    exposure = np.ones_like(eta_np, dtype=float)
    exposure /= max(numpy_trapezoid(exposure, x=eta_np), np.finfo(float).tiny)
    rows = []
    for energy in E_np:
        mass = legacy_solar_mass(np.array([energy]))[0]
        P_eta = np.stack(
            [
                legacy_pearth(
                    mass,
                    earth_density_l,
                    legacy_pmns,
                    DM21_EV2,
                    DM3L_EV2,
                    float(energy),
                    float(angle),
                    depth_m,
                    mode=LEGACY_EARTH_MODE,
                    massbasis=True,
                )
                for angle in eta_np
            ]
        )
        rows.append(numpy_trapezoid(P_eta * exposure[:, None], x=eta_np, axis=0))
    return np.stack(rows)


def nusquids_integrated(mass_np, E_1d, eta_1d, depth_m):
    E_np, eta_np = to_numpy(E_1d), to_numpy(eta_1d)
    exposure = np.ones_like(eta_np, dtype=float)
    exposure /= max(numpy_trapezoid(exposure, x=eta_np), np.finfo(float).tiny)
    rows = []
    for i, energy in enumerate(E_np):
        P_eta = np.stack(
            [
                nusquids_probability_earth_massbasis(
                    E_GeV=float(energy) * 1.0e-3,
                    cos_zenith=float(cos_zenith_from_nadir(torch.tensor(eta))),
                    mass_weights=mass_np[i],
                    config=NSQ_CFG,
                )
                for eta in eta_np
            ]
        )
        rows.append(numpy_trapezoid(P_eta * exposure[:, None], x=eta_np, axis=0))
    return np.stack(rows)


def tpeanuts_solar_detector_probabilities(E_1d, eta_1d, device=None):
    dev = device if device is not None else DEVICE
    E = torch.as_tensor(E_1d, device=dev, dtype=DTYPE)
    eta = torch.as_tensor(eta_1d, device=dev, dtype=DTYPE)
    mass = tpeanuts_solar_mass(E, device=dev)
    return torch_pearth_probability(mass, E[:, None], eta[None, :], SOLAR_DETECTOR_DEPTH_M, massbasis=True, device=dev)


def legacy_solar_detector_probabilities(E_np, eta_np):
    rows = []
    for energy in E_np:
        mass = legacy_solar_mass(np.array([energy]))[0]
        rows.append(
            np.stack(
                [
                    legacy_pearth(
                        mass,
                        earth_density_l,
                        legacy_pmns,
                        DM21_EV2,
                        DM3L_EV2,
                        float(energy),
                        float(angle),
                        SOLAR_DETECTOR_DEPTH_M,
                        mode=LEGACY_EARTH_MODE,
                        massbasis=True,
                    )
                    for angle in eta_np
                ]
            )
        )
    return np.stack(rows)


def nusquids_solar_detector_probabilities(mass_np, E_1d, eta_1d):
    E_np, eta_np = to_numpy(E_1d), to_numpy(eta_1d)
    return np.stack(
        [
            np.stack(
                [
                    nusquids_probability_earth_massbasis(
                        E_GeV=float(energy) * 1.0e-3,
                        cos_zenith=float(cos_zenith_from_nadir(torch.tensor(eta))),
                        mass_weights=mass_np[i],
                        config=NSQ_CFG,
                    )
                    for eta in eta_np
                ]
            )
            for i, energy in enumerate(E_np)
        ]
    )


def _tpeanuts_earth_flux(E_1d, eta_1d, flux, *, device=None):
    """Flavour-basis Earth flux via a single shared evolutor.

    The Earth evolution operator does not depend on the initial state, so
    building it once and reading off the full transition-probability matrix
    is equivalent to (but three times cheaper than) calling
    ``torch_pearth_probability`` once per basis vector of ``torch.eye(3)``,
    since each of those calls would otherwise rebuild the same evolutor.
    """
    dev = device if device is not None else DEVICE
    osc_dev = _oscillation_for(dev)
    profile_dev = _earth_profile_for(dev)
    E_t = torch.as_tensor(E_1d, device=dev, dtype=DTYPE)
    eta_t = torch.as_tensor(eta_1d, device=dev, dtype=DTYPE)

    return earth_probability_state(
        nustate=flux,
        profile_earth=profile_dev,
        oscillation=osc_dev,
        E_MeV=E_t[:, None],
        eta=eta_t[None, :],
        depth_m=EARTH_DEPTH_M,
        method="analytical",
        massbasis=True,
    )


def _legacy_earth_flux(E_np, eta_np, flux_np):
    cols = [
        np.stack(
            [
                legacy_pearth(
                    np.eye(3)[i],
                    earth_density_l,
                    legacy_pmns,
                    DM21_EV2,
                    DM3L_EV2,
                    float(e),
                    float(a),
                    EARTH_DEPTH_M,
                    mode=LEGACY_EARTH_MODE,
                    massbasis=True,
                )
                for e in E_np
                for a in eta_np
            ]
        ).reshape(len(E_np), len(eta_np), 3)
        for i in range(3)
    ]
    return np.einsum("enba,ea->enb", np.stack(cols, axis=-1), flux_np)


def _nusquids_earth_flux(E_1d, eta_1d, flux_np):
    E_np, eta_np = to_numpy(E_1d), to_numpy(eta_1d)
    transition = np.empty((len(E_np), len(eta_np), 3, 3))
    for i, energy in enumerate(E_np):
        for j, eta in enumerate(eta_np):
            transition[i, j] = nusquids_transition_matrix_earth(
                E_GeV=float(energy) * 1.0e-3,
                cos_zenith=float(cos_zenith_from_nadir(torch.tensor(eta))),
                config=NSQ_CFG,
            )
    return np.einsum("enba,ea->enb", transition, flux_np)


def _tpeanuts_pearth_analytical(E_1d, eta_1d):
    return earth_probability_state(
        MASS_WEIGHTS,
        earth_profile_t,
        oscillation,
        E_1d[:, None],
        eta_1d[None, :],
        EARTH_DEPTH_M,
        method="analytical",
        massbasis=True,
        context=context,
    )


def _tpeanuts_pearth_numerical(E_1d, eta_1d):
    E_b, eta_b = torch.broadcast_tensors(E_1d[:, None], eta_1d[None, :])
    return torch.stack(
        [
            earth_probability_state(
                MASS_WEIGHTS,
                earth_profile_t,
                oscillation,
                ev,
                etav,
                EARTH_DEPTH_M,
                method="numerical",
                massbasis=True,
                nsteps=EARTH_NUMERICAL_STEPS,
                ode_method=EARTH_ODE_METHOD,
                context=context,
            )
            for ev, etav in zip(E_b.reshape(-1), eta_b.reshape(-1))
        ]
    ).reshape(*E_b.shape, 3)


def nusquids_earth_matrix(state_np, E_1d, eta_1d, *, use_transition_matrix=False):
    """nuSQuIDS Earth probabilities for a fixed incoherent mass state."""
    E_np, eta_np = to_numpy(E_1d), to_numpy(eta_1d)
    out = np.empty((len(E_np), len(eta_np), 3))
    for i, energy in enumerate(E_np):
        for j, eta in enumerate(eta_np):
            cosz = float(cos_zenith_from_nadir(torch.tensor(eta)))
            if use_transition_matrix:
                transition = nusquids_transition_matrix_earth(E_GeV=float(energy) * 1.0e-3, cos_zenith=cosz, config=NSQ_CFG)
                out[i, j] = transition @ state_np
            else:
                out[i, j] = nusquids_probability_earth_massbasis(
                    E_GeV=float(energy) * 1.0e-3,
                    cos_zenith=cosz,
                    mass_weights=state_np,
                    config=NSQ_CFG,
                )
    return out


def tpeanuts_atm_probability(E_MeV, cos_zenith, *, initial_flavour="numu"):
    theta_deg = math.degrees(math.acos(float(np.clip(cos_zenith, -1.0, 1.0))))
    theta_t = torch.tensor(theta_deg, dtype=DTYPE, device=DEVICE)
    P_mat = atmosphere_probability_transition(
        oscillation,
        E_MeV,
        ATM_H_PROD_KM,
        theta_t,
        ATM_DEPTH_KM,
        atmosphere=atm_params,
        context=context,
    )
    return P_mat[:, FLAVOUR_INDEX[initial_flavour]]


def nusquids_atm_matrix(E_1d, cosz_1d, *, initial_flavour="numu"):
    E_np, cosz_np = to_numpy(E_1d), np.asarray(cosz_1d)
    return np.stack(
        [
            np.stack(
                [
                    nusquids_probability_atmosphere(
                        E_GeV=float(energy) * 1.0e-3,
                        cos_zenith=float(cz),
                        initial_flavour=initial_flavour,
                        config=NSQ_CFG,
                    )
                    for cz in cosz_np
                ]
            )
            for energy in E_np
        ]
    )


def _tpeanuts_solar_detector_device(E_1d, eta_1d, device):
    dev_ctx = RuntimeContext.resolve(device, DTYPE)
    osc_dev = PropagationConfig.oscillation_parameters_from_preset("_SM_NUFIT52_NO", antinu=False, context=dev_ctx)
    prof_dev = SolarProfile.default(context=dev_ctx)
    earth_dev = EarthProfile(
        params=EarthParameters(
            profile_perturbative_kwargs={
                "density_file": str(config.earth_density_file),
                "tabulated_density": False,
            }
        ),
        context=dev_ctx,
    )
    mass = solar_probability_mass(osc_dev, E_1d, prof_dev, SOLAR_SOURCE)
    return earth_probability_state_analytical(mass, earth_dev, osc_dev, E_1d[:, None], eta_1d[None, :], SOLAR_DETECTOR_DEPTH_M, massbasis=True)


def get_largest_timing_row(df: pd.DataFrame, largest_ne: int, largest_neta: int):
    """Return the largest-grid timing row from a benchmark DataFrame."""
    sub = df[df["n_energy"] == largest_ne]
    if "n_nadir" in sub.columns and sub["n_nadir"].notna().any():
        sub = sub[sub["n_nadir"] == largest_neta]
    return sub.iloc[-1] if len(sub) else None


__all__ = [
    "configure_benchmark_helpers",
    "synchronize_device",
    "release_benchmark_memory",
    "process_ram_gb",
    "check_ram_budget",
    "iter_grid_chunks",
    "reduce_tensor_chunks",
    "measure_resources",
    "timed_resources",
    "energy_grid",
    "nadir_grid",
    "cos_zenith_from_nadir",
    "synthetic_flux",
    "solar_source_spectrum",
    "timed_call",
    "benchmark_pair",
    "save_results",
    "plot_energy_scaling",
    "plot_heatmap",
    "plot_speedup_cross_sections",
    "plot_grid_resource_scaling",
    "torch_pearth_probability",
    "legacy_vacuum_matrix",
    "nusquids_vacuum_matrix",
    "tpeanuts_solar_point",
    "nusquids_solar_point",
    "tpeanuts_solar_mass",
    "legacy_solar_mass",
    "tpeanuts_integrated",
    "legacy_integrated",
    "nusquids_integrated",
    "tpeanuts_solar_detector_probabilities",
    "legacy_solar_detector_probabilities",
    "nusquids_solar_detector_probabilities",
    "_tpeanuts_earth_flux",
    "_legacy_earth_flux",
    "_nusquids_earth_flux",
    "_tpeanuts_pearth_analytical",
    "_tpeanuts_pearth_numerical",
    "nusquids_earth_matrix",
    "tpeanuts_atm_probability",
    "nusquids_atm_matrix",
    "_tpeanuts_solar_detector_device",
    "get_largest_timing_row",
]

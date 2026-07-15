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
import time
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.common.probability import probability_transition
from tpeanuts.medium.earth.evolutor import earth_evolutor
from tpeanuts.medium.earth.profile import EarthParameters, EarthProfile
from tpeanuts.medium.earth.probability import pearth
from tpeanuts.medium.solar.profile import SolarParameters, SolarProfile
from tpeanuts.notebooks.notebooks_helper import FLAVOUR_NAMES, save_and_show, to_numpy
from tpeanuts.util.context import RuntimeContext


BENCHMARK_BACKEND = "legacy"


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
        _OSCILLATION_CACHE[device] = OscillationParameters.build(
            theta12=THETA12,
            theta13=THETA13,
            theta23=THETA23,
            delta=DELTA_CP,
            DeltamSq21=DM21_EV2,
            DeltamSq3l=DM3L_EV2,
            antinu=False,
            context=_context_for(device),
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
            params=SolarParameters(model_path=SOLAR_MODEL_CSV_FILE, fluxes_path=SOLAR_FLUX_CSV_FILE),
            context=_context_for(device),
        )
    return _SOLAR_PROFILE_CACHE[device]


def torch_pearth_probability(state, E_MeV, eta, depth_m, *, massbasis=True, device=None):
    """tpeanuts Earth probability helper shared by benchmark notebooks."""
    dev = device if device is not None else DEVICE
    if BENCHMARK_BACKEND == "nusquids":
        return pearth_analytical(state, earth_profile_t, oscillation, E_MeV, eta, depth_m, massbasis=massbasis)

    ctx = _context_for(dev)
    E_t = torch.as_tensor(E_MeV, device=dev, dtype=DTYPE)
    eta_t = torch.as_tensor(eta, device=dev, dtype=DTYPE)
    state_t = torch.as_tensor(state, device=dev)
    osc_dev = _oscillation_for(dev)
    profile_dev = _earth_profile_for(dev)
    if EARTH_METHOD == "analytical":
        return pearth(state_t, profile_dev, osc_dev, E_t, eta_t, depth_m, method="analytical", massbasis=massbasis, context=ctx)
    if EARTH_METHOD == "numerical":
        E_b, eta_b = torch.broadcast_tensors(E_t, eta_t)
        return torch.stack(
            [
                pearth(
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

    U_earth = earth_evolutor(
        profile_earth=profile_dev,
        oscillation=osc_dev,
        E=E_t,
        eta=eta_t,
        depth_m=EARTH_DEPTH_M,
    )
    U = osc_dev.pmns.pmns_matrix(antinu=osc_dev.antinu).to(device=U_earth.device, dtype=U_earth.dtype)
    probabilities = probability_transition(U_earth @ U)
    return torch.einsum("enba,ea->enb", probabilities, flux)


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
    return pearth(
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
            pearth(
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
    P_mat = atmosphere_probability(
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
    osc_dev = OscillationParameters.from_preset("_SM_NUFIT52_NO", antinu=False, context=dev_ctx)
    prof_dev = SolarProfile.default(context=dev_ctx)
    earth_dev = EarthProfile(
        params=EarthParameters(
            profile_perturbative_kwargs={
                "density_file": str(config.data_dir / "density" / "earth_density.csv"),
                "tabulated_density": False,
            }
        ),
        context=dev_ctx,
    )
    mass = solar_probability_mass(osc_dev, E_1d, prof_dev, SOLAR_SOURCE)
    return pearth_analytical(mass, earth_dev, osc_dev, E_1d[:, None], eta_1d[None, :], SOLAR_DETECTOR_DEPTH_M, massbasis=True)


def get_largest_timing_row(df: pd.DataFrame, largest_ne: int, largest_neta: int):
    """Return the largest-grid timing row from a benchmark DataFrame."""
    sub = df[df["n_energy"] == largest_ne]
    if "n_nadir" in sub.columns and sub["n_nadir"].notna().any():
        sub = sub[sub["n_nadir"] == largest_neta]
    return sub.iloc[-1] if len(sub) else None


__all__ = [
    "configure_benchmark_helpers",
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

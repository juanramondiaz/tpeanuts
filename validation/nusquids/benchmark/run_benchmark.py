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

"""Phase D: single-point timing benchmarks, tpeanuts vs nuSQuIDS, plus tpeanuts CPU vs GPU.

The benchmark counterpart to Phase C's six ``compare/*.py`` scripts:
where those check *agreement*, this module checks *speed*, one fixed,
representative point per scenario (vacuum, earth, atmosphere, solar,
solar_earth, bsm_sterile), using ``compare.grid``'s already-verified
``ProbabilityFn`` pairs directly -- no new physics, only timing.

Every ``benchmark_*`` function always times the tpeanuts side (never
needs nuSQuIDS); the nuSQuIDS side is added only when
``validation.nusquids.adapter.is_available()`` is True, so this module
runs to completion (with a tpeanuts-only result) in this project's own
environment, exactly like every ``compare/*.py`` script's own
``skipif(not is_available())`` gating for the real cross-solver
number.

The GPU benchmark is tpeanuts-only by construction (nuSQuIDS is a
CPU-only C++ solver, per this whole package's own docstring) --
``benchmark_gpu_vs_cpu`` compares tpeanuts's own CPU and CUDA
``RuntimeContext``s on the same vacuum-oscillation call, skipped
entirely when no CUDA device is available.

Module contents:
    benchmark_vacuum(...), benchmark_earth(...),
    benchmark_atmosphere(...), benchmark_solar(...),
    benchmark_solar_earth(...), benchmark_bsm_sterile_atmosphere(...),
    benchmark_bsm_sterile_solar(...)
        One fixed representative point each, returning
        ``{"tpeanuts": TimingResult, ["nusquids": TimingResult]}``.
    benchmark_gpu_vs_cpu(...)
        tpeanuts-only CPU vs CUDA timing on the same vacuum call, or
        None when no CUDA device is available.
    run_all(...)
        Run every benchmark above and return the combined results,
        plus a ready-to-print summary table.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Dict, Optional

import torch

from benchmarks.common.timing import TimingResult, format_timing_table, time_call
from tpeanuts.util.context import RuntimeContext
from validation.nusquids.adapter import (
    NUSQUIDS_EARTHATM_HEIGHT_M,
    is_available,
    probability_atmosphere,
    probability_earth,
    probability_solar_point,
    probability_vacuum,
)
from validation.nusquids.compare.atmosphere import tpeanuts_probability_atmosphere
from validation.nusquids.compare.bsm_sterile import (
    tpeanuts_probability_atmosphere_sterile,
    tpeanuts_probability_solar_point_sterile,
)
from validation.nusquids.compare.earth import tpeanuts_probability_earth
from validation.nusquids.compare.solar import tpeanuts_probability_solar_point
from validation.nusquids.compare.solar_earth import tpeanuts_probability_solar_earth
from validation.nusquids.compare.vacuum import tpeanuts_probability_vacuum
from validation.nusquids.config import NuSQuIDSConfig

_N_REPEATS = 10
_N_WARMUP = 2


def _benchmark_pair(
    name: str, tpeanuts_call: Callable[[], object], nusquids_call: Optional[Callable[[], object]],
    *, n_repeats: int = _N_REPEATS, n_warmup: int = _N_WARMUP,
) -> Dict[str, TimingResult]:
    """Time the tpeanuts side always, and the nuSQuIDS side only when available.

    Args:
        name: Scenario name, used to build each result's own ``name``
            (``time_call`` would otherwise name every entry after its
            wrapping lambda, "<lambda>", which is not useful in a
            printed table).
        tpeanuts_call: Zero-argument callable running one tpeanuts
            evaluation.
        nusquids_call: Zero-argument callable running one nuSQuIDS
            evaluation, or None to skip it unconditionally (used by
            callers that already know nuSQuIDS cannot represent this
            case -- none currently do, kept for symmetry with
            ``compare/*.py``'s own signatures).
        n_repeats: Timed repetitions, forwarded to ``time_call``.
        n_warmup: Warm-up repetitions, forwarded to ``time_call``.

    Returns:
        ``{"tpeanuts": TimingResult}``, plus ``"nusquids": TimingResult``
        when ``nusquids_call`` is given and
        ``validation.nusquids.adapter.is_available()`` is True.
    """
    results = {
        "tpeanuts": dataclasses.replace(
            time_call(tpeanuts_call, n_repeats=n_repeats, n_warmup=n_warmup), name=f"{name}_tpeanuts",
        ),
    }
    if nusquids_call is not None and is_available():
        results["nusquids"] = dataclasses.replace(
            time_call(nusquids_call, n_repeats=n_repeats, n_warmup=n_warmup), name=f"{name}_nusquids",
        )
    return results


def benchmark_vacuum(config: Optional[NuSQuIDSConfig] = None) -> Dict[str, TimingResult]:
    """Time one vacuum-oscillation call, tpeanuts vs nuSQuIDS."""
    cfg = NuSQuIDSConfig() if config is None else config
    return _benchmark_pair(
        "vacuum",
        lambda: tpeanuts_probability_vacuum(E_GeV=1.0, baseline_km=1000.0, initial_flavour="numu", config=cfg),
        lambda: probability_vacuum(E_GeV=1.0, baseline_km=1000.0, initial_flavour="numu", config=cfg),
    )


def benchmark_earth(config: Optional[NuSQuIDSConfig] = None) -> Dict[str, TimingResult]:
    """Time one Earth-crossing oscillation call, tpeanuts vs nuSQuIDS."""
    cfg = NuSQuIDSConfig() if config is None else config
    return _benchmark_pair(
        "earth",
        lambda: tpeanuts_probability_earth(E_GeV=2.0, cos_zenith=-0.6, initial_flavour="numu", config=cfg, nsteps=200),
        lambda: probability_earth(E_GeV=2.0, cos_zenith=-0.6, initial_flavour="numu", config=cfg),
    )


def benchmark_atmosphere(config: Optional[NuSQuIDSConfig] = None) -> Dict[str, TimingResult]:
    """Time one composed atmosphere+Earth oscillation call, tpeanuts vs nuSQuIDS."""
    cfg = NuSQuIDSConfig() if config is None else config
    return _benchmark_pair(
        "atmosphere",
        lambda: tpeanuts_probability_atmosphere(
            E_GeV=2.0, cos_zenith=-0.6, h_m=NUSQUIDS_EARTHATM_HEIGHT_M, initial_flavour="numu", config=cfg, nsteps=200,
        ),
        lambda: probability_atmosphere(E_GeV=2.0, cos_zenith=-0.6, initial_flavour="numu", config=cfg),
    )


def benchmark_solar(config: Optional[NuSQuIDSConfig] = None) -> Dict[str, TimingResult]:
    """Time one solar-production-point oscillation call, tpeanuts vs nuSQuIDS."""
    cfg = NuSQuIDSConfig() if config is None else config
    return _benchmark_pair(
        "solar",
        lambda: tpeanuts_probability_solar_point(E_MeV=8.0, r0=0.1, config=cfg),
        lambda: probability_solar_point(E_MeV=8.0, r0=0.1, initial_flavour="nue", impact=0.0, config=cfg),
    )


def benchmark_solar_earth(config: Optional[NuSQuIDSConfig] = None) -> Dict[str, TimingResult]:
    """Time one full Sun->(Earth regeneration)->detector call, tpeanuts vs nuSQuIDS.

    The nuSQuIDS side is timed as the composed call
    ``probability_solar_point(..., return_mass_weights=True)`` then
    ``probability_earth_massbasis(...)`` -- both calls together, since
    that composition is what ``compare/solar_earth.py`` itself
    verifies against, not a single adapter function.
    """
    cfg = NuSQuIDSConfig() if config is None else config

    def _nusquids_call():
        from validation.nusquids.adapter import (
            probability_earth_massbasis,  # local import: keeps this one-off composition visible at its call site
        )

        weights = probability_solar_point(
            E_MeV=8.0, r0=0.1, initial_flavour="nue", impact=0.0, return_mass_weights=True, config=cfg,
        )
        return probability_earth_massbasis(E_GeV=8.0e-3, cos_zenith=-0.6, mass_weights=weights, config=cfg)

    return _benchmark_pair(
        "solar_earth",
        lambda: tpeanuts_probability_solar_earth(E_MeV=8.0, r0=0.1, cos_zenith=-0.6, config=cfg, nsteps=200),
        _nusquids_call,
    )


def benchmark_bsm_sterile_atmosphere(config: Optional[NuSQuIDSConfig] = None) -> Dict[str, TimingResult]:
    """Time one sterile-extended composed atmosphere+Earth call, tpeanuts vs nuSQuIDS."""
    cfg = NuSQuIDSConfig(numneu=4) if config is None else config
    return _benchmark_pair(
        "bsm_sterile_atmosphere",
        lambda: tpeanuts_probability_atmosphere_sterile(
            E_GeV=3.0, cos_zenith=-0.6, h_m=NUSQUIDS_EARTHATM_HEIGHT_M, initial_flavour="numu", config=cfg, nsteps=200,
        ),
        lambda: probability_atmosphere(E_GeV=3.0, cos_zenith=-0.6, initial_flavour="numu", config=cfg),
    )


def benchmark_bsm_sterile_solar(config: Optional[NuSQuIDSConfig] = None) -> Dict[str, TimingResult]:
    """Time one sterile-extended solar-production-point call, tpeanuts vs nuSQuIDS."""
    cfg = NuSQuIDSConfig(numneu=4) if config is None else config
    return _benchmark_pair(
        "bsm_sterile_solar",
        lambda: tpeanuts_probability_solar_point_sterile(E_MeV=8.0, r0=0.1, config=cfg),
        lambda: probability_solar_point(E_MeV=8.0, r0=0.1, initial_flavour="nue", impact=0.0, config=cfg),
    )


def benchmark_gpu_vs_cpu(config: Optional[NuSQuIDSConfig] = None) -> Optional[Dict[str, TimingResult]]:
    """Time one vacuum-oscillation call on tpeanuts's own CPU vs CUDA RuntimeContext.

    tpeanuts-only by construction: nuSQuIDS is a CPU-only C++ solver
    (see this module's own docstring), so there is no CUDA nuSQuIDS
    side to compare against.

    Args:
        config: Shared oscillation configuration. Defaults to
            ``NuSQuIDSConfig()`` when None.

    Returns:
        ``{"cpu": TimingResult, "cuda": TimingResult}``, or None when
        no CUDA device is available.
    """
    if not torch.cuda.is_available():
        return None

    cfg = NuSQuIDSConfig() if config is None else config
    cpu_context = RuntimeContext.resolve("cpu", torch.float64)
    cuda_context = RuntimeContext.resolve("cuda", torch.float64)

    cpu_result = time_call(
        lambda: tpeanuts_probability_vacuum(
            E_GeV=1.0, baseline_km=1000.0, initial_flavour="numu", config=cfg, context=cpu_context,
        ),
        n_repeats=_N_REPEATS, n_warmup=_N_WARMUP,
    )
    cuda_result = time_call(
        lambda: tpeanuts_probability_vacuum(
            E_GeV=1.0, baseline_km=1000.0, initial_flavour="numu", config=cfg, context=cuda_context,
        ),
        n_repeats=_N_REPEATS, n_warmup=_N_WARMUP,
    )
    return {
        "cpu": dataclasses.replace(cpu_result, name="vacuum_gpu_benchmark_cpu"),
        "cuda": dataclasses.replace(cuda_result, name="vacuum_gpu_benchmark_cuda"),
    }


def run_all(config: Optional[NuSQuIDSConfig] = None) -> Dict[str, Dict[str, TimingResult]]:
    """Run every benchmark above and return the combined results.

    Args:
        config: Shared oscillation configuration for every scenario
            except the two ``bsm_sterile_*`` ones (which need
            ``numneu=4`` regardless -- passing a plain ``config`` there
            is overridden internally). Defaults to ``NuSQuIDSConfig()``
            when None.

    Returns:
        Dict keyed by scenario name (``"vacuum"``, ``"earth"``,
        ``"atmosphere"``, ``"solar"``, ``"solar_earth"``,
        ``"bsm_sterile_atmosphere"``, ``"bsm_sterile_solar"``, and
        ``"gpu_vs_cpu"`` when CUDA is available), each mapping to its
        own ``{"tpeanuts": TimingResult, ...}`` dict.
    """
    results: Dict[str, Dict[str, TimingResult]] = {
        "vacuum": benchmark_vacuum(config),
        "earth": benchmark_earth(config),
        "atmosphere": benchmark_atmosphere(config),
        "solar": benchmark_solar(config),
        "solar_earth": benchmark_solar_earth(config),
        "bsm_sterile_atmosphere": benchmark_bsm_sterile_atmosphere(),
        "bsm_sterile_solar": benchmark_bsm_sterile_solar(),
    }
    gpu_result = benchmark_gpu_vs_cpu(config)
    if gpu_result is not None:
        results["gpu_vs_cpu"] = gpu_result
    return results


def format_all(results: Dict[str, Dict[str, TimingResult]]) -> str:
    """Format every scenario's own TimingResults into one combined table.

    Args:
        results: As returned by ``run_all``.

    Returns:
        A single string, one ``format_timing_table`` block per
        scenario, in insertion order.
    """
    blocks = []
    for scenario, timings in results.items():
        blocks.append(f"-- {scenario} --")
        blocks.append(format_timing_table(list(timings.values())))
    return "\n".join(blocks)


if __name__ == "__main__":
    print(format_all(run_all()))

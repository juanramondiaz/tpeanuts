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
#      August 2026
# =============================================================================

"""
Loaders for the real SNO salt-phase (Phase II) data, see ``data/detector/sno_ii/``.

Every loader validates its table's real structure on load (dimension,
symmetry, unit diagonal, positive-semi-definiteness, channel order) rather
than trusting the CSV blindly -- these tables were manually transcribed
from a rendered PDF (see ``data/detector/sno_ii/metadata/source.json``),
so a dimension/symmetry/PSD failure here is the first line of defense
against a transcription error, not a defensive-programming reflex.

Module contents:
    IntegratedFlux
        (day, night) NC/ES/CC-integral flux with separate stat/syst
        uncertainty, one entry of ``load_integrated_fluxes_day_night``'s
        returned dict.
    CCSpectralSystematics
        The per-bin percent systematic sensitivities (Table XXXIV), keyed
        by systematic name.
    load_cc_spectrum_day_night(...)
        ``observation/cc_spectrum_day_night.csv`` -> (day, night)
        Observations, the 17-bin CC equivalent-flux spectrum.
    load_integrated_fluxes_day_night(...)
        ``observation/integrated_fluxes_day_night.csv`` -> ``{"NC": ...,
        "ES": ..., "CC_integral": ...}``.
    load_statistical_correlation(...)
        ``covariance/statistical_correlation_{day,night}.csv`` -> validated
        19x19 correlation matrix.
    load_zenith_exposure(...)
        ``exposure/zenith_livetime.csv`` -> (eta, exposure) on this
        project's nadir-angle convention, mirroring
        ``tpeanuts.detector.sno.io.load_coszenith_exposure``.
    load_cc_spectral_systematics(...)
        ``systematics/cc_spectral_systematics.csv`` -> ``CCSpectralSystematics``.
    load_observed_vector_and_covariance(...)
        Assembles the real 38-observable data vector and its 38x38
        block-diagonal (day/night) statistical covariance V_stat, directly
        comparable to ``tpeanuts.detector.sno_ii.inference_model
        .SNOPhaseIIObservableModel.predict``'s own output.
    build_systematic_covariance(...)
        The systematic covariance V_syst (Eq. 21), from Table XXXIV's
        per-bin percent sensitivities.
    load_observed_vector_and_total_covariance(...)
        ``(value, V_stat + V_syst)`` (Eq. 20), convenience wrapper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

import tpeanuts.config.default as default
from tpeanuts.detector.common.observation import Observation
from tpeanuts.detector.sno_ii.parameters import (
    CHANNEL_ORDER,
    N_CC_BINS,
    N_CHANNELS,
    N_OBSERVABLES_PER_PERIOD,
)
from tpeanuts.medium.earth.exposure_math import make_eta_grid
from tpeanuts.util.io import package_dir
from tpeanuts.util.math import interp1d_linear

_OBS_DIR = "sno_ii/observation"
_COV_DIR = "sno_ii/covariance"
_EXP_DIR = "sno_ii/exposure"
_SYS_DIR = "sno_ii/systematics"


def load_cc_spectrum_day_night(
    path: Optional[str | Path] = None,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[Observation, Observation]:
    """Load the real 17-bin CC equivalent-flux spectrum, day and night (Table XXX).

    Args:
        path: Optional override path to the CSV. None loads the bundled
            table at ``data/detector/sno_ii/observation/cc_spectrum_day_night.csv``.
        device: Target torch device.
        dtype: Target real dtype.

    Returns:
        ``(day, night)`` Observations, ``bin_width_MeV`` set, ``value`` in
        units of 1e6 cm^-2 s^-1 (equivalent 8B flux per bin, not a raw
        count -- see ``tpeanuts.detector.sno_ii``'s package docstring),
        ``sigma_minus == sigma_plus`` = the statistical uncertainty only
        (systematic uncertainty is handled separately, see
        ``load_cc_spectral_systematics``).

    Raises:
        ValueError: If the table does not have exactly
            ``tpeanuts.detector.sno_ii.parameters.N_CC_BINS`` rows.
    """
    if path is None:
        path = package_dir() / default.detector_data_dir / _OBS_DIR / "cc_spectrum_day_night.csv"
    table = pd.read_csv(path)
    if len(table) != N_CC_BINS:
        raise ValueError(f"cc_spectrum_day_night.csv has {len(table)} rows, expected {N_CC_BINS}.")

    labels = tuple(f"CC{int(b)}" for b in table["bin"])
    energy_low = torch.tensor(table["energy_low_MeV"].to_numpy(), device=device, dtype=dtype)
    energy_high = torch.tensor(table["energy_high_MeV"].to_numpy(), device=device, dtype=dtype)
    x_MeV = 0.5 * (energy_low + energy_high)
    width_MeV = energy_high - energy_low

    day_flux = torch.tensor(table["day_flux"].to_numpy(), device=device, dtype=dtype)
    day_sigma = torch.tensor(table["day_sigma_stat"].to_numpy(), device=device, dtype=dtype)
    night_flux = torch.tensor(table["night_flux"].to_numpy(), device=device, dtype=dtype)
    night_sigma = torch.tensor(table["night_sigma_stat"].to_numpy(), device=device, dtype=dtype)

    day = Observation.from_symmetric(labels, x_MeV, day_flux, day_sigma, bin_width_MeV=width_MeV)
    night = Observation.from_symmetric(labels, x_MeV, night_flux, night_sigma, bin_width_MeV=width_MeV)
    return day, night


@dataclass(frozen=True)
class IntegratedFlux:
    """(day, night) integrated flux with separate statistical/systematic uncertainty.

    Units: 1e6 cm^-2 s^-1 (see ``tpeanuts.detector.sno_ii``'s package docstring).
    """

    day: torch.Tensor
    day_sigma_stat: torch.Tensor
    day_sigma_syst: torch.Tensor
    night: torch.Tensor
    night_sigma_stat: torch.Tensor
    night_sigma_syst: torch.Tensor


def load_integrated_fluxes_day_night(
    path: Optional[str | Path] = None,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, IntegratedFlux]:
    """Load the real integrated NC/ES fluxes, day and night (Table XXIV).

    Args:
        path: Optional override path to the CSV. None loads the bundled
            table at ``data/detector/sno_ii/observation/integrated_fluxes_day_night.csv``.
        device: Target torch device.
        dtype: Target real dtype.

    Returns:
        ``{"NC": IntegratedFlux, "ES": IntegratedFlux, "CC_integral": IntegratedFlux}``.
        ``"CC_integral"`` is included only as a cross-check against
        ``load_cc_spectrum_day_night``'s own bin sum (see
        ``data/detector/sno_ii/metadata/source.json``'s
        ``cross_checks_passed``) -- it is not an independent 38th/39th
        observable and must not be added to a fit alongside the 17-bin
        spectrum.
    """
    if path is None:
        path = package_dir() / default.detector_data_dir / _OBS_DIR / "integrated_fluxes_day_night.csv"
    table = pd.read_csv(path).set_index("channel")

    result: dict[str, IntegratedFlux] = {}
    for channel in table.index:
        row = table.loc[channel]
        result[channel] = IntegratedFlux(
            day=torch.tensor(float(row["day_flux"]), device=device, dtype=dtype),
            day_sigma_stat=torch.tensor(float(row["day_sigma_stat"]), device=device, dtype=dtype),
            day_sigma_syst=torch.tensor(float(row["day_sigma_syst"]), device=device, dtype=dtype),
            night=torch.tensor(float(row["night_flux"]), device=device, dtype=dtype),
            night_sigma_stat=torch.tensor(float(row["night_sigma_stat"]), device=device, dtype=dtype),
            night_sigma_syst=torch.tensor(float(row["night_sigma_syst"]), device=device, dtype=dtype),
        )
    return result


def load_statistical_correlation(
    period: str,
    path: Optional[str | Path] = None,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    atol: float = 1.0e-6,
) -> torch.Tensor:
    """Load and validate the real 19x19 statistical correlation matrix (Tables XXXII/XXXIII).

    Args:
        period: ``"day"`` or ``"night"``, selecting
            ``covariance/statistical_correlation_{period}.csv``.
        path: Optional override path to the CSV. None loads the bundled
            table for ``period``.
        device: Target torch device.
        dtype: Target real dtype.
        atol: Absolute tolerance for the symmetry and unit-diagonal checks.

    Returns:
        Real tensor shaped ``(19, 19)``, rows/columns ordered as
        ``tpeanuts.detector.sno_ii.parameters.CHANNEL_ORDER``.

    Raises:
        ValueError: If ``period`` is invalid, the table's shape or column
            order does not match ``CHANNEL_ORDER``, or the matrix fails
            symmetry, unit-diagonal, or positive-semi-definiteness checks.
    """
    if period not in ("day", "night"):
        raise ValueError(f"period must be 'day' or 'night', got {period!r}.")
    if path is None:
        path = package_dir() / default.detector_data_dir / _COV_DIR / f"statistical_correlation_{period}.csv"
    table = pd.read_csv(path, index_col=0)

    if tuple(table.columns) != CHANNEL_ORDER or tuple(table.index) != CHANNEL_ORDER:
        raise ValueError(
            f"statistical_correlation_{period}.csv's row/column order does not match "
            f"CHANNEL_ORDER={CHANNEL_ORDER}; got columns={tuple(table.columns)}."
        )
    m = torch.tensor(table.to_numpy(), device=device, dtype=dtype)
    if m.shape != (N_CHANNELS, N_CHANNELS):
        raise ValueError(f"statistical_correlation_{period}.csv has shape {tuple(m.shape)}, expected ({N_CHANNELS}, {N_CHANNELS}).")
    if not torch.allclose(m, m.T, atol=atol):
        raise ValueError(f"statistical_correlation_{period}.csv is not symmetric (within atol={atol}).")
    if not torch.allclose(torch.diag(m), torch.ones(N_CHANNELS, dtype=dtype, device=device), atol=atol):
        raise ValueError(f"statistical_correlation_{period}.csv's diagonal is not all 1 (within atol={atol}).")
    eigvals = torch.linalg.eigvalsh(m)
    if eigvals.min() < -atol:
        raise ValueError(
            f"statistical_correlation_{period}.csv is not positive-semi-definite "
            f"(min eigenvalue {float(eigvals.min())} < -{atol})."
        )
    return m


def load_zenith_exposure(
    path: Optional[str | Path] = None,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the real SNO salt-phase cos(zenith) livetime distribution as (eta, exposure).

    The real table (Table XXXI) has 60 uniformly-spaced bins spanning
    cos(zenith) in [-1, 1]; this applies the exact same nadir-angle
    conversion (uniform cos(zenith) grid -> eta via ``make_eta_grid`` +
    ``interp1d_linear`` + the sin(eta) Jacobian) as
    ``tpeanuts.detector.sno.io.load_coszenith_exposure`` uses for Phase-I's
    own (differently-tabulated but likewise 480-point-uniform) exposure.

    Args:
        path: Optional override path to the CSV. None loads the bundled
            table at ``data/detector/sno_ii/exposure/zenith_livetime.csv``.
        device: Target torch device.
        dtype: Target real dtype.

    Returns:
        ``(eta, exposure)``: nadir-angle grid in radians, shape ``(60,)``,
        strictly increasing 0 to pi, and the corresponding (non-negative,
        un-normalized) exposure weight, in seconds per radian of eta.
        Mirrors ``tpeanuts.detector.sno.io.load_coszenith_exposure``'s own
        caveat: the day/night *ratio* of ``exposure`` is exact, but its
        absolute scale carries an overall constant Jacobian factor -- do
        not ``trapezoid``-integrate this to recover the real 391.4-day
        total livetime (it will not match); renormalize each day/night
        half to sum to 1 before using it as an averaging weight, as
        ``tpeanuts.detector.sno.inference_model.SNODayNightModel
        .from_real_exposure`` does for Phase I.
    """
    if path is None:
        path = package_dir() / default.detector_data_dir / _EXP_DIR / "zenith_livetime.csv"
    table = pd.read_csv(path)

    raw = torch.tensor(table["livetime_s"].to_numpy(), device=device, dtype=dtype)
    ns = raw.numel()

    eta = make_eta_grid(ns, daynight=None, device=device, dtype=dtype)
    cz = torch.linspace(-1.0, 1.0, ns, device=device, dtype=dtype)
    dcz = cz[1] - cz[0]
    deta = math.pi / (ns - 1)

    exposure = interp1d_linear(
        x=-torch.cos(eta), xp=cz, fp=raw, left=0.0, right=0.0, device=device, dtype=dtype,
    )
    exposure = exposure * torch.sin(eta) * deta / dcz
    return eta, exposure.clamp_min(0.0)


# Table XXXIV's own printed column order (NC, ES, CC1..CC17) -- distinct
# from CHANNEL_ORDER (NC, CC1..CC17, ES), which follows Tables
# XXX/XXXII/XXXIII instead. Kept separate deliberately: reordering the
# transcribed columns to match CHANNEL_ORDER would make future spot-checks
# against the primary source (page 45) harder, not easier.
SYSTEMATICS_CHANNEL_ORDER: tuple[str, ...] = (
    "NC", "ES",
    "CC1", "CC2", "CC3", "CC4", "CC5", "CC6", "CC7", "CC8", "CC9",
    "CC10", "CC11", "CC12", "CC13", "CC14", "CC15", "CC16", "CC17",
)


@dataclass(frozen=True)
class CCSpectralSystematics:
    """Per-bin percent systematic sensitivities (Table XXXIV), the Eq. 21 ``dY_i/dS_k`` inputs.

    Parameters
    ----------
    names:
        One name per systematic source (16 entries), e.g. ``"energy_scale_const"``.
    channels:
        Channel order of ``plus``/``minus``'s second axis --
        ``SYSTEMATICS_CHANNEL_ORDER`` (NC, ES, CC1..CC17), *not*
        ``tpeanuts.detector.sno_ii.parameters.CHANNEL_ORDER`` (NC,
        CC1..CC17, ES) -- see that constant's own docstring for why.
    plus, minus:
        Signed percent sensitivity per (systematic, channel), each shape
        ``(16, 19)``.
    """

    names: tuple[str, ...]
    channels: tuple[str, ...]
    plus: torch.Tensor
    minus: torch.Tensor


def load_cc_spectral_systematics(
    path: Optional[str | Path] = None,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> CCSpectralSystematics:
    """Load the real per-bin percent systematic sensitivities (Table XXXIV).

    See ``data/detector/sno_ii/metadata/source.json``'s ``needs_verification``
    entry: the smallest-magnitude rows (vertex/internal-gamma/selection-
    efficiency/backgrounds) are lower-confidence transcriptions and should
    be spot-checked against the primary source before use in a real fit.

    Args:
        path: Optional override path to the CSV. None loads the bundled
            table at ``data/detector/sno_ii/systematics/cc_spectral_systematics.csv``.
        device: Target torch device.
        dtype: Target real dtype.

    Returns:
        ``CCSpectralSystematics``.

    Raises:
        ValueError: If the table's channel columns do not match
            ``CHANNEL_ORDER`` (both a ``_plus`` and a ``_minus`` column per
            channel).
    """
    if path is None:
        path = package_dir() / default.detector_data_dir / _SYS_DIR / "cc_spectral_systematics.csv"
    table = pd.read_csv(path, comment="#")

    expected_columns = ["systematic"] + [
        f"{c}_{s}" for c in SYSTEMATICS_CHANNEL_ORDER for s in ("plus", "minus")
    ]
    if list(table.columns) != expected_columns:
        raise ValueError(
            f"cc_spectral_systematics.csv's columns do not match the expected "
            f"SYSTEMATICS_CHANNEL_ORDER-derived layout; got {list(table.columns)}."
        )

    names = tuple(table["systematic"])
    plus = torch.tensor(
        table[[f"{c}_plus" for c in SYSTEMATICS_CHANNEL_ORDER]].to_numpy(), device=device, dtype=dtype,
    )
    minus = torch.tensor(
        table[[f"{c}_minus" for c in SYSTEMATICS_CHANNEL_ORDER]].to_numpy(), device=device, dtype=dtype,
    )
    return CCSpectralSystematics(names=names, channels=SYSTEMATICS_CHANNEL_ORDER, plus=plus, minus=minus)


def load_observed_vector_and_covariance(
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real observed 38-vector and its 38x38 statistical covariance V_stat.

    Assembles ``load_cc_spectrum_day_night`` (17 CC bins) and
    ``load_integrated_fluxes_day_night`` (NC, ES -- ``"CC_integral"`` is
    excluded, see that function's own docstring) into one
    ``CHANNEL_ORDER``-ordered vector per period, and combines each period's
    statistical sigma with ``load_statistical_correlation`` into
    ``V_stat = diag(sigma) @ corr @ diag(sigma)``. Day and night are
    statistically independent (Eq. 19-20 of the primary source), so the
    full 38x38 matrix is block-diagonal:

        V_stat = [[V_stat_day, 0], [0, V_stat_night]].

    This is the statistical-only covariance -- pass it through
    ``tpeanuts.inference.likelihood.cholesky_from_covariance`` directly for
    a stat-only fit (this package's Phase 7), or add the systematic
    covariance from ``load_cc_spectral_systematics`` before factorizing for
    the full V_stat+V_syst treatment (a later phase, see
    ``tpeanuts.detector.sno_ii``'s package docstring).

    Args:
        device: Target torch device.
        dtype: Target real dtype.

    Returns:
        ``(value, V_stat)``: ``value`` shape ``(38,)``, ``V_stat`` shape
        ``(38, 38)``, both ordered day-then-night,
        ``tpeanuts.detector.sno_ii.parameters.CHANNEL_ORDER`` within each
        period -- directly comparable to
        ``tpeanuts.detector.sno_ii.inference_model.SNOPhaseIIObservableModel
        .predict``'s own output.
    """
    day_spectrum, night_spectrum = load_cc_spectrum_day_night(device=device, dtype=dtype)
    fluxes = load_integrated_fluxes_day_night(device=device, dtype=dtype)
    corr_day = load_statistical_correlation("day", device=device, dtype=dtype)
    corr_night = load_statistical_correlation("night", device=device, dtype=dtype)

    value_day = torch.cat([fluxes["NC"].day.reshape(1), day_spectrum.value, fluxes["ES"].day.reshape(1)])
    sigma_day = torch.cat([
        fluxes["NC"].day_sigma_stat.reshape(1), day_spectrum.sigma_minus, fluxes["ES"].day_sigma_stat.reshape(1),
    ])

    value_night = torch.cat(
        [fluxes["NC"].night.reshape(1), night_spectrum.value, fluxes["ES"].night.reshape(1)],
    )
    sigma_night = torch.cat([
        fluxes["NC"].night_sigma_stat.reshape(1),
        night_spectrum.sigma_minus,
        fluxes["ES"].night_sigma_stat.reshape(1),
    ])

    V_day = corr_day * sigma_day[:, None] * sigma_day[None, :]
    V_night = corr_night * sigma_night[:, None] * sigma_night[None, :]

    n = V_day.shape[0]
    V_stat = torch.zeros(2 * n, 2 * n, device=device, dtype=dtype)
    V_stat[:n, :n] = V_day
    V_stat[n:, n:] = V_night

    value = torch.cat([value_day, value_night])
    return value, V_stat


def build_systematic_covariance(value: torch.Tensor) -> torch.Tensor:
    """Systematic covariance V_syst from Table XXXIV's per-bin percent sensitivities.

    Eq. 21 of the primary source: ``sigma_ij^2(syst) = sum_k (dY_i/dS_k)
    (dY_j/dS_k) (Delta S_k)^2``, with ``Delta S_k = 1`` (the tabulated
    percentages already are per-1-sigma sensitivities) and ``dY_i/dS_k =
    (percent_ik / 100) * value_i`` (Table XXXIV gives *fractional*
    sensitivity, not absolute -- converted here using each observable's own
    real central value, day and night separately). Vectorized as
    ``V_syst = dY^T @ dY`` where ``dY`` has shape ``(16 systematics, 38
    observables)``, which is a Gram matrix and therefore automatically
    symmetric and positive-semi-definite (rank <= 16).

    Each systematic's plus/minus percentages are symmetrized (averaged in
    absolute value) before use -- Eq. 21's simple quadratic form has no
    asymmetric-uncertainty counterpart; using the paper's own directional
    detail would require a more elaborate treatment (see
    ``tpeanuts.detector.sno_ii``'s package docstring and this project's
    plan for a nuisance-pull alternative, deferred).

    The *same* row of Table XXXIV is applied to both the day and night
    value of a channel (each with its own central value) -- these are
    genuine detector systematics (energy scale, resolution, ...), 100%
    correlated across periods since they come from the same instrument,
    unlike the day/night-independent statistical covariance
    (``load_observed_vector_and_covariance``). This is what gives V_syst
    its non-block-diagonal day/night cross terms, matching this package's
    own design note ("systematics can correlate day and night").

    **Known incompleteness**: Table XXXIV excludes NC's own two dedicated
    systematics (internal neutron background, neutron capture efficiency
    -- see the primary source's own Appendix A note, and
    ``data/detector/sno_ii/metadata/source.json``'s ``not_yet_transcribed``
    entry for Table XIX/XX). V_syst therefore understates NC's true
    systematic uncertainty; NC's total error here is statistics-only until
    those tables are added.

    Args:
        value: The real 38-observable vector (day then night,
            ``tpeanuts.detector.sno_ii.parameters.CHANNEL_ORDER`` order),
            e.g. from ``load_observed_vector_and_covariance``'s own
            return value -- used only to convert Table XXXIV's percentages
            to absolute units.

    Returns:
        Real tensor shaped ``(38, 38)``, symmetric and positive-semi-definite.
    """
    systematics = load_cc_spectral_systematics(device=value.device, dtype=value.dtype)
    fractional = 0.5 * (systematics.plus.abs() + systematics.minus.abs()) / 100.0  # (16, 19)

    channel_index_in_systematics = [systematics.channels.index(c) for c in CHANNEL_ORDER]
    fractional_reordered = fractional[:, channel_index_in_systematics]  # (16, 19), now CHANNEL_ORDER-ordered

    value_day = value[:N_OBSERVABLES_PER_PERIOD]
    value_night = value[N_OBSERVABLES_PER_PERIOD:]
    dY_day = fractional_reordered * value_day[None, :]
    dY_night = fractional_reordered * value_night[None, :]
    dY = torch.cat([dY_day, dY_night], dim=-1)  # (16, 38)

    return dY.T @ dY


def load_observed_vector_and_total_covariance(
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real observed 38-vector and its total covariance V_stat + V_syst (Eq. 20).

    Convenience wrapper combining ``load_observed_vector_and_covariance``
    and ``build_systematic_covariance`` -- see both for details and
    caveats (in particular, V_syst's known incompleteness around NC's own
    dedicated systematics).

    Args:
        device: Target torch device.
        dtype: Target real dtype.

    Returns:
        ``(value, V_total)``, same shapes as
        ``load_observed_vector_and_covariance``.
    """
    value, V_stat = load_observed_vector_and_covariance(device=device, dtype=dtype)
    V_syst = build_systematic_covariance(value)
    return value, V_stat + V_syst

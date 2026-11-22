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

"""Loaders for the real SNO salt-phase (Phase II) data, see ``data/experiments/sno_ii/``.

Ported from v1_0's ``experiments/sno_ii/io.py``, replacing pandas with
``csv.DictReader`` (this project's own convention -- no pandas
anywhere in v2). Every loader validates its table's real structure on
load (dimension, symmetry, unit diagonal, positive-semi-definiteness,
channel order) rather than trusting the CSV blindly -- these tables
were manually transcribed from a rendered PDF (see
``data/experiments/sno_ii/metadata/source.json``), so a dimension/
symmetry/PSD failure here is the first line of defense against a
transcription error, not a defensive-programming reflex.

**Known transcription caveats, carried forward verbatim from
``data/experiments/sno_ii/metadata/source.json`` (not silently resolved
or hidden)**:
    - ``covariance/statistical_correlation_day.csv``: the printed table
      itself has CC10-CC11 = 0.0000 (row CC10) but CC11-CC10 = 0.0001
      (row CC11) -- a genuine 4-decimal rounding inconsistency in the
      original publication (Table XXXII), verified against the
      rendered PDF twice. Symmetrized to 0.0001 in both cells in the
      transcribed CSV so the matrix is exactly symmetric for Cholesky
      use.
    - ``systematics/cc_spectral_systematics.csv``: the
      ``vertex_x``/``vertex_y``/``vertex_z``/``internal_gamma``/
      ``selection_efficiency``/``backgrounds`` rows (the smallest-print,
      sub-1%-magnitude rows) are lower-confidence transcriptions and
      should be spot-checked against page 45 of the primary source
      before use in a real fit. The ``vertex_resolution`` and
      ``angular_resolution`` rows' CC9-CC17 columns are SET TO 0.0
      (placeholder, not a real measurement) -- a value is missing
      partway through both rows in the rendered PDF, confirmed via a
      column-count check.
    - Table XIX/XX (NC-only systematics: internal neutron background,
      neutron-capture efficiency uncertainty) are explicitly required
      for a complete oscillation analysis per the primary source's own
      Appendix A note, but were never transcribed (not in v1_0 either)
      -- ``build_systematic_covariance`` therefore understates NC's
      true systematic uncertainty; NC's total error here is
      statistics-only until those tables are added.

Module contents:
    IntegratedFlux
        (day, night) NC/ES/CC-integral flux with separate stat/syst
        uncertainty, one entry of
        ``load_integrated_fluxes_day_night``'s returned dict.
    CCSpectralSystematics
        The per-bin percent systematic sensitivities (Table XXXIV),
        keyed by systematic name.
    load_cc_spectrum_day_night(...)
        ``observation/cc_spectrum_day_night.csv`` -> (day, night)
        Observations, the 17-bin CC equivalent-flux spectrum.
    load_integrated_fluxes_day_night(...)
        ``observation/integrated_fluxes_day_night.csv`` -> ``{"NC":
        ..., "ES": ..., "CC_integral": ...}``.
    load_statistical_correlation(...)
        ``covariance/statistical_correlation_{day,night}.csv`` ->
        validated 19x19 correlation matrix.
    load_zenith_exposure(...)
        ``exposure/zenith_livetime.csv`` -> (eta_rad, exposure) on this
        project's nadir-angle convention, reusing
        ``tpeanuts.experiments.sno.io.eta_exposure_from_uniform_coszenith``.
    load_cc_spectral_systematics(...)
        ``systematics/cc_spectral_systematics.csv`` ->
        ``CCSpectralSystematics``.
    load_observed_vector_and_covariance(...)
        Assembles the real 38-observable data vector and its 38x38
        block-diagonal (day/night) statistical covariance V_stat,
        directly comparable to
        ``tpeanuts.experiments.sno_ii.inference_model
        .SNOPhaseIIModel.predict``'s own output.
    build_systematic_covariance(...)
        The systematic covariance V_syst (Eq. 21), from Table XXXIV's
        per-bin percent sensitivities.
    load_observed_vector_and_total_covariance(...)
        ``(value, V_stat + V_syst)`` (Eq. 20), convenience wrapper.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch

import tpeanuts.config.default as default
from tpeanuts.detector.common.observation import Observation
from tpeanuts.experiments.sno.io import eta_exposure_from_uniform_coszenith
from tpeanuts.experiments.sno_ii.parameters import CHANNEL_ORDER, N_CC_BINS, N_CHANNELS, N_OBSERVABLES_PER_PERIOD
from tpeanuts.util.context import RuntimeContext

Tensor = torch.Tensor

_OBS_DIR = "observation"
_COV_DIR = "covariance"
_EXP_DIR = "exposure"
_SYS_DIR = "systematics"


def load_cc_spectrum_day_night(
    path: Optional[Union[str, Path]] = None, *, context: RuntimeContext,
) -> Tuple[Observation, Observation]:
    """Load the real 17-bin CC equivalent-flux spectrum, day and night (Table XXX).

    Args:
        path: Optional override path to the CSV. None loads the
            bundled table at ``data/experiments/sno_ii/observation/
            cc_spectrum_day_night.csv``.
        context: RuntimeContext the returned tensors are built on.

    Returns:
        ``(day, night)`` Observations, ``bin_width_MeV`` set, ``value``
        in units of 1e6 cm^-2 s^-1 (equivalent 8B flux per bin, not a
        raw count -- see ``tpeanuts.experiments.sno_ii``'s package
        docstring), ``sigma_minus == sigma_plus`` = the statistical
        uncertainty only (systematic uncertainty is handled
        separately, see ``load_cc_spectral_systematics``).

    Raises:
        ValueError: If the table does not have exactly
            ``tpeanuts.experiments.sno_ii.parameters.N_CC_BINS`` rows.
    """
    if path is None:
        path = default.SNO_II_DATA_DIR / _OBS_DIR / "cc_spectrum_day_night.csv"

    labels, energy_low, energy_high = [], [], []
    day_flux, day_sigma, night_flux, night_sigma = [], [], [], []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            labels.append(f"CC{int(float(row['bin']))}")
            energy_low.append(float(row["energy_low_MeV"]))
            energy_high.append(float(row["energy_high_MeV"]))
            day_flux.append(float(row["day_flux"]))
            day_sigma.append(float(row["day_sigma_stat"]))
            night_flux.append(float(row["night_flux"]))
            night_sigma.append(float(row["night_sigma_stat"]))

    if len(labels) != N_CC_BINS:
        raise ValueError(f"cc_spectrum_day_night.csv has {len(labels)} rows, expected {N_CC_BINS}.")

    energy_low_t = torch.tensor(energy_low, device=context.device, dtype=context.dtype)
    energy_high_t = torch.tensor(energy_high, device=context.device, dtype=context.dtype)
    x_MeV = 0.5 * (energy_low_t + energy_high_t)
    width_MeV = energy_high_t - energy_low_t

    day = Observation.from_symmetric(
        tuple(labels), x_MeV, day_flux, day_sigma, context=context, bin_width_MeV=width_MeV,
    )
    night = Observation.from_symmetric(
        tuple(labels), x_MeV, night_flux, night_sigma, context=context, bin_width_MeV=width_MeV,
    )
    return day, night


@dataclass(frozen=True)
class IntegratedFlux:
    """(day, night) integrated flux with separate statistical/systematic uncertainty.

    Units: 1e6 cm^-2 s^-1 (see ``tpeanuts.experiments.sno_ii``'s package
    docstring).
    """

    day: Tensor
    day_sigma_stat: Tensor
    day_sigma_syst: Tensor
    night: Tensor
    night_sigma_stat: Tensor
    night_sigma_syst: Tensor


def load_integrated_fluxes_day_night(
    path: Optional[Union[str, Path]] = None, *, context: RuntimeContext,
) -> Dict[str, IntegratedFlux]:
    """Load the real integrated NC/ES fluxes, day and night (Table XXIV).

    Args:
        path: Optional override path to the CSV. None loads the
            bundled table at ``data/experiments/sno_ii/observation/
            integrated_fluxes_day_night.csv``.
        context: RuntimeContext the returned tensors are built on.

    Returns:
        ``{"NC": IntegratedFlux, "ES": IntegratedFlux, "CC_integral":
        IntegratedFlux}``. ``"CC_integral"`` is included only as a
        cross-check against ``load_cc_spectrum_day_night``'s own bin
        sum (see ``data/experiments/sno_ii/metadata/source.json``'s
        ``cross_checks_passed``) -- it is not an independent 38th/39th
        observable and must not be added to a fit alongside the 17-bin
        spectrum.
    """
    if path is None:
        path = default.SNO_II_DATA_DIR / _OBS_DIR / "integrated_fluxes_day_night.csv"

    result: Dict[str, IntegratedFlux] = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["channel"]] = IntegratedFlux(
                day=torch.tensor(float(row["day_flux"]), device=context.device, dtype=context.dtype),
                day_sigma_stat=torch.tensor(float(row["day_sigma_stat"]), device=context.device, dtype=context.dtype),
                day_sigma_syst=torch.tensor(float(row["day_sigma_syst"]), device=context.device, dtype=context.dtype),
                night=torch.tensor(float(row["night_flux"]), device=context.device, dtype=context.dtype),
                night_sigma_stat=torch.tensor(
                    float(row["night_sigma_stat"]), device=context.device, dtype=context.dtype,
                ),
                night_sigma_syst=torch.tensor(
                    float(row["night_sigma_syst"]), device=context.device, dtype=context.dtype,
                ),
            )
    return result


def load_statistical_correlation(
    period: str, path: Optional[Union[str, Path]] = None, *, context: RuntimeContext, atol: float = 1.0e-6,
) -> Tensor:
    """Load and validate the real 19x19 statistical correlation matrix (Tables XXXII/XXXIII).

    Args:
        period: ``"day"`` or ``"night"``, selecting
            ``covariance/statistical_correlation_{period}.csv``.
        path: Optional override path to the CSV. None loads the
            bundled table for ``period``.
        context: RuntimeContext the returned tensor is built on.
        atol: Absolute tolerance for the symmetry and unit-diagonal
            checks.

    Returns:
        Real tensor shaped ``(19, 19)``, rows/columns ordered as
        ``tpeanuts.experiments.sno_ii.parameters.CHANNEL_ORDER``.

    Raises:
        ValueError: If ``period`` is invalid, the table's shape or
            column order does not match ``CHANNEL_ORDER``, or the
            matrix fails symmetry, unit-diagonal, or
            positive-semi-definiteness checks.
    """
    if period not in ("day", "night"):
        raise ValueError(f"period must be 'day' or 'night', got {period!r}.")
    if path is None:
        path = default.SNO_II_DATA_DIR / _COV_DIR / f"statistical_correlation_{period}.csv"

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple((reader.fieldnames or ())[1:])
        rows = list(reader)
    row_labels = tuple(row["channel"] for row in rows)
    if columns != CHANNEL_ORDER or row_labels != CHANNEL_ORDER:
        raise ValueError(
            f"statistical_correlation_{period}.csv's row/column order does not match "
            f"CHANNEL_ORDER={CHANNEL_ORDER}; got columns={columns}."
        )

    values = [[float(row[c]) for c in CHANNEL_ORDER] for row in rows]
    m = torch.tensor(values, device=context.device, dtype=context.dtype)
    if m.shape != (N_CHANNELS, N_CHANNELS):
        raise ValueError(
            f"statistical_correlation_{period}.csv has shape {tuple(m.shape)}, expected "
            f"({N_CHANNELS}, {N_CHANNELS})."
        )
    if not torch.allclose(m, m.T, atol=atol):
        raise ValueError(f"statistical_correlation_{period}.csv is not symmetric (within atol={atol}).")
    if not torch.allclose(torch.diag(m), torch.ones(N_CHANNELS, dtype=context.dtype, device=context.device), atol=atol):
        raise ValueError(f"statistical_correlation_{period}.csv's diagonal is not all 1 (within atol={atol}).")
    eigvals = torch.linalg.eigvalsh(m)
    if eigvals.min() < -atol:
        raise ValueError(
            f"statistical_correlation_{period}.csv is not positive-semi-definite "
            f"(min eigenvalue {float(eigvals.min())} < -{atol})."
        )
    return m


def load_zenith_exposure(path: Optional[Union[str, Path]] = None, *, context: RuntimeContext) -> Tuple[Tensor, Tensor]:
    """Load the real SNO salt-phase cos(zenith) livetime distribution as (eta_rad, exposure).

    The real table (Table XXXI) has 60 uniformly-spaced bins spanning
    cos(zenith) in ``[-1, 1]`` (confirmed directly: the CSV's own
    ``cos_zenith_low``/``cos_zenith_high``/``period`` columns are
    informative only -- ``period`` labels each row day/night by sign,
    but is not needed for the conversion, since the rows are already
    monotonically ordered over a uniform grid); this applies the exact
    same nadir-angle conversion as
    ``tpeanuts.experiments.sno.io.eta_exposure_from_uniform_coszenith``
    uses for Phase-I's own (differently-tabulated but likewise
    480-point-uniform) exposure.

    Args:
        path: Optional override path to the CSV. None loads the
            bundled table at ``data/experiments/sno_ii/exposure/
            zenith_livetime.csv``.
        context: RuntimeContext the returned tensors are built on.

    Returns:
        ``(eta_rad, exposure)``: nadir-angle grid in radians, shape
        ``(60,)``, strictly increasing 0 to pi, and the corresponding
        (non-negative, un-normalized) exposure weight, in seconds per
        radian of eta_rad. Mirrors
        ``tpeanuts.experiments.sno.io.load_nadir_exposure_raw``'s own
        caveat: the day/night *ratio* of ``exposure`` is exact, but its
        absolute scale carries an overall constant Jacobian factor --
        do not ``trapezoid``-integrate this to recover the real
        391.4-day total livetime (it will not match); renormalize each
        day/night half before using it as an averaging weight, as
        ``tpeanuts.experiments.sno_ii.inference_model
        .SNOPhaseIIModel.from_real_exposure`` does.
    """
    if path is None:
        path = default.SNO_II_DATA_DIR / _EXP_DIR / "zenith_livetime.csv"

    raw = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            raw.append(float(row["livetime_s"]))

    raw_t = torch.tensor(raw, device=context.device, dtype=context.dtype)
    return eta_exposure_from_uniform_coszenith(raw_t, context=context)


# Table XXXIV's own printed column order (NC, ES, CC1..CC17) -- distinct
# from CHANNEL_ORDER (NC, CC1..CC17, ES), which follows Tables
# XXX/XXXII/XXXIII instead. Kept separate deliberately: reordering the
# transcribed columns to match CHANNEL_ORDER would make future spot-checks
# against the primary source (page 45) harder, not easier.
SYSTEMATICS_CHANNEL_ORDER: Tuple[str, ...] = (
    "NC", "ES",
    "CC1", "CC2", "CC3", "CC4", "CC5", "CC6", "CC7", "CC8", "CC9",
    "CC10", "CC11", "CC12", "CC13", "CC14", "CC15", "CC16", "CC17",
)


@dataclass(frozen=True)
class CCSpectralSystematics:
    """Per-bin percent systematic sensitivities (Table XXXIV), the Eq. 21 ``dY_i/dS_k`` inputs.

    Attributes:
        names: One name per systematic source (16 entries), e.g.
            ``"energy_scale_const"``.
        channels: Channel order of ``plus``/``minus``'s second axis --
            ``SYSTEMATICS_CHANNEL_ORDER`` (NC, ES, CC1..CC17), *not*
            ``tpeanuts.experiments.sno_ii.parameters.CHANNEL_ORDER`` (NC,
            CC1..CC17, ES) -- see that constant's own docstring for
            why.
        plus, minus: Signed percent sensitivity per (systematic,
            channel), each shape ``(16, 19)``.
    """

    names: Tuple[str, ...]
    channels: Tuple[str, ...]
    plus: Tensor
    minus: Tensor


def load_cc_spectral_systematics(
    path: Optional[Union[str, Path]] = None, *, context: RuntimeContext,
) -> CCSpectralSystematics:
    """Load the real per-bin percent systematic sensitivities (Table XXXIV).

    See this module's own docstring's "known transcription caveats"
    section: the smallest-magnitude rows are lower-confidence
    transcriptions and should be spot-checked against the primary
    source before use in a real fit.

    Args:
        path: Optional override path to the CSV. None loads the
            bundled table at ``data/experiments/sno_ii/systematics/
            cc_spectral_systematics.csv``.
        context: RuntimeContext the returned tensors are built on.

    Returns:
        A ``CCSpectralSystematics``.

    Raises:
        ValueError: If the table's channel columns do not match the
            expected ``SYSTEMATICS_CHANNEL_ORDER``-derived layout.
    """
    if path is None:
        path = default.SNO_II_DATA_DIR / _SYS_DIR / "cc_spectral_systematics.csv"

    expected_columns = ["systematic"] + [
        f"{c}_{s}" for c in SYSTEMATICS_CHANNEL_ORDER for s in ("plus", "minus")
    ]
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = None
        rows = []
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if header is None:
                header = row
                continue
            rows.append(row)

    if header != expected_columns:
        raise ValueError(
            "cc_spectral_systematics.csv's columns do not match the expected "
            f"SYSTEMATICS_CHANNEL_ORDER-derived layout; got {header}."
        )

    names = tuple(row[0] for row in rows)
    plus_idx = [header.index(f"{c}_plus") for c in SYSTEMATICS_CHANNEL_ORDER]
    minus_idx = [header.index(f"{c}_minus") for c in SYSTEMATICS_CHANNEL_ORDER]
    plus = torch.tensor([[float(row[i]) for i in plus_idx] for row in rows], device=context.device, dtype=context.dtype)
    minus = torch.tensor(
        [[float(row[i]) for i in minus_idx] for row in rows], device=context.device, dtype=context.dtype
    )
    return CCSpectralSystematics(names=names, channels=SYSTEMATICS_CHANNEL_ORDER, plus=plus, minus=minus)


def load_observed_vector_and_covariance(*, context: RuntimeContext) -> Tuple[Tensor, Tensor]:
    """Real observed 38-vector and its 38x38 statistical covariance V_stat.

    Assembles ``load_cc_spectrum_day_night`` (17 CC bins) and
    ``load_integrated_fluxes_day_night`` (NC, ES -- ``"CC_integral"``
    is excluded, see that function's own docstring) into one
    ``CHANNEL_ORDER``-ordered vector per period, and combines each
    period's statistical sigma with ``load_statistical_correlation``
    into ``V_stat = diag(sigma) @ corr @ diag(sigma)``. Day and night
    are statistically independent (Eq. 19-20 of the primary source),
    so the full 38x38 matrix is block-diagonal:

        V_stat = [[V_stat_day, 0], [0, V_stat_night]].

    This is the statistical-only covariance -- pass it through
    ``tpeanuts.inference.likelihood.cholesky_from_covariance`` directly
    for a stat-only fit, or add the systematic covariance from
    ``load_cc_spectral_systematics`` before factorizing for the full
    V_stat+V_syst treatment (``load_observed_vector_and_total_covariance``).

    Args:
        context: RuntimeContext the returned tensors are built on.

    Returns:
        ``(value, V_stat)``: ``value`` shape ``(38,)``, ``V_stat``
        shape ``(38, 38)``, both ordered day-then-night,
        ``tpeanuts.experiments.sno_ii.parameters.CHANNEL_ORDER`` within
        each period -- directly comparable to
        ``tpeanuts.experiments.sno_ii.inference_model
        .SNOPhaseIIModel.predict``'s own output.
    """
    day_spectrum, night_spectrum = load_cc_spectrum_day_night(context=context)
    fluxes = load_integrated_fluxes_day_night(context=context)
    corr_day = load_statistical_correlation("day", context=context)
    corr_night = load_statistical_correlation("night", context=context)

    value_day = torch.cat(
        [fluxes["NC"].day.reshape(1), torch.as_tensor(day_spectrum.value), fluxes["ES"].day.reshape(1)]
    )
    sigma_day = torch.cat([
        torch.as_tensor(fluxes["NC"].day_sigma_stat).reshape(1),
        torch.as_tensor(day_spectrum.sigma_minus),
        torch.as_tensor(fluxes["ES"].day_sigma_stat).reshape(1),
    ])

    value_night = torch.cat(
        [fluxes["NC"].night.reshape(1), torch.as_tensor(night_spectrum.value), fluxes["ES"].night.reshape(1)],
    )
    sigma_night = torch.cat([
        torch.as_tensor(fluxes["NC"].night_sigma_stat).reshape(1),
        torch.as_tensor(night_spectrum.sigma_minus),
        torch.as_tensor(fluxes["ES"].night_sigma_stat).reshape(1),
    ])

    V_day = corr_day * sigma_day[:, None] * sigma_day[None, :]
    V_night = corr_night * sigma_night[:, None] * sigma_night[None, :]

    n = V_day.shape[0]
    V_stat = torch.zeros(2 * n, 2 * n, device=context.device, dtype=context.dtype)
    V_stat[:n, :n] = V_day
    V_stat[n:, n:] = V_night

    value = torch.cat([value_day, value_night])
    return value, V_stat


def build_systematic_covariance(value: Tensor, *, context: RuntimeContext) -> Tensor:
    """Systematic covariance V_syst from Table XXXIV's per-bin percent sensitivities.

    Eq. 21 of the primary source: ``sigma_ij^2(syst) = sum_k
    (dY_i/dS_k)(dY_j/dS_k)(Delta S_k)^2``, with ``Delta S_k = 1`` (the
    tabulated percentages already are per-1-sigma sensitivities) and
    ``dY_i/dS_k = (percent_ik / 100) * value_i`` (Table XXXIV gives
    *fractional* sensitivity, not absolute -- converted here using each
    observable's own real central value, day and night separately).
    Vectorized as ``V_syst = dY^T @ dY`` where ``dY`` has shape ``(16
    systematics, 38 observables)``, which is a Gram matrix and
    therefore automatically symmetric and positive-semi-definite (rank
    <= 16).

    Each systematic's plus/minus percentages are symmetrized (averaged
    in absolute value) before use -- Eq. 21's simple quadratic form has
    no asymmetric-uncertainty counterpart.

    The *same* row of Table XXXIV is applied to both the day and night
    value of a channel (each with its own central value) -- these are
    genuine detector systematics (energy scale, resolution, ...), 100%
    correlated across periods since they come from the same
    instrument, unlike the day/night-independent statistical
    covariance (``load_observed_vector_and_covariance``). This is what
    gives V_syst its non-block-diagonal day/night cross terms.

    **Known incompleteness**: Table XXXIV excludes NC's own two
    dedicated systematics (internal neutron background, neutron
    capture efficiency -- see this module's own "known transcription
    caveats" section). V_syst therefore understates NC's true
    systematic uncertainty; NC's total error here is statistics-only
    until those tables are added.

    Args:
        value: The real 38-observable vector (day then night,
            ``tpeanuts.experiments.sno_ii.parameters.CHANNEL_ORDER``
            order), e.g. from ``load_observed_vector_and_covariance``'s
            own return value -- used only to convert Table XXXIV's
            percentages to absolute units.
        context: RuntimeContext ``load_cc_spectral_systematics``'s
            table is loaded on.

    Returns:
        Real tensor shaped ``(38, 38)``, symmetric and
        positive-semi-definite.
    """
    systematics = load_cc_spectral_systematics(context=context)
    fractional = 0.5 * (systematics.plus.abs() + systematics.minus.abs()) / 100.0  # (16, 19)

    channel_index_in_systematics = [systematics.channels.index(c) for c in CHANNEL_ORDER]
    fractional_reordered = fractional[:, channel_index_in_systematics]  # (16, 19), now CHANNEL_ORDER-ordered

    value_day = value[:N_OBSERVABLES_PER_PERIOD]
    value_night = value[N_OBSERVABLES_PER_PERIOD:]
    dY_day = fractional_reordered * value_day[None, :]
    dY_night = fractional_reordered * value_night[None, :]
    dY = torch.cat([dY_day, dY_night], dim=-1)  # (16, 38)

    return dY.T @ dY


def load_observed_vector_and_total_covariance(*, context: RuntimeContext) -> Tuple[Tensor, Tensor]:
    """Real observed 38-vector and its total covariance V_stat + V_syst (Eq. 20).

    Convenience wrapper combining ``load_observed_vector_and_covariance``
    and ``build_systematic_covariance`` -- see both for details and
    caveats (in particular, V_syst's known incompleteness around NC's
    own dedicated systematics).

    Args:
        context: RuntimeContext the returned tensors are built on.

    Returns:
        ``(value, V_total)``, same shapes as
        ``load_observed_vector_and_covariance``.
    """
    value, V_stat = load_observed_vector_and_covariance(context=context)
    V_syst = build_systematic_covariance(value, context=context)
    return value, V_stat + V_syst

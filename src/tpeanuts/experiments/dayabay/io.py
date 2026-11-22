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

"""Loaders for the Daya Bay data release, cached CSVs under ``data/experiments/dayabay/``.

The tables that describe the reactor itself (Huber-Mueller spectra, non-equilibrium correction, fission fractions,
thermal power) are not here: they belong to the source and are loaded by ``tpeanuts.source.reactor.io``.

Ported from v1_0's ``experiments/dayabay/io.py``, translated to this
project's own established data-loading conventions (``csv.DictReader``,
no ``pandas``; ``context: RuntimeContext`` instead of separate
``device``/``dtype`` kwargs -- exactly like
``tpeanuts.source.solar.io``/``tpeanuts.source.atmosphere.io`` already
do). Every function here is a thin, direct read -- no caching, no
derived computation (e.g. the per-reactor relative-weight reduction
over ``load_neutrino_rate_weekly``'s own rows lives in
``tpeanuts.experiments.dayabay.parameters``, not here).

``load_neutrino_rate_weekly`` returns only the ``(n_det,
neutrino_rate_per_s)`` pairs its one consumer
(``parameters._reactor_relative_weights``) actually reduces over, not
every column of the source CSV (``period``/``day``/``start_utc``/
``end_utc``/``n_days``/``n_det_mask`` are unused by anything in this
port).

Module contents:
    load_baselines(...)
        (detector, reactor) -> baseline in km, published geometry.
    load_n_protons(...)
        Target proton count per detector.
    load_eres_parameters(...)
        3-parameter energy-resolution formula coefficients.
    load_detector_efficiency(...)
        IBD-selection efficiency (a bare scalar).
    load_ibd_spectrum(...)
        Observed IBD prompt-energy spectrum for one detector.
    load_background_rates(...)
        Background rates/uncertainties table (all detectors/categories).
    load_background_shape(...)
        Normalized background spectrum shape for one (category, detector).
    load_exposure(...)
        8AD-period effective livetime per detector.
    load_final_erec_bin_edges(...)
        Analysis-binning edges.
    load_survival_probability_truth(...)
        Daya Bay's own published best-fit parameters, for comparison only.
    load_ibd_constants(...)
        IBD cross-section coupling constants (f, g, f2, PhaseSpaceFactor).
    load_iav_matrix(...)
        IAV energy-redistribution matrix, 240x240, 0-12 MeV.
    load_lsnl_curve(...)
        LSNL nominal energy-scale nonlinearity curve.
    load_lsnl_curve_pulls(...)
        LSNL systematic-variation ("pull") curves, 4 of them.
    load_snf_correction(...)
        Per-reactor spent-nuclear-fuel flux correction (R1-R6).
    load_neutrino_rate_weekly(...)
        Per-reactor (n_det, neutrino_rate_per_s) history, 8AD-period
        weeks included.
    load_global_normalization(...)
        Daya Bay's own nominal global-normalization value (1.0).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch

import tpeanuts.config.default as default
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import as_tensor

Tensor = torch.Tensor


def _read_csv_rows(path: Union[str, Path]) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read a CSV file into its own field names and rows, in file order.

    Raises:
        ValueError: If the file has no header row.
    """
    with open(path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header row: {path!r}.")
        return list(reader.fieldnames), list(reader)


def load_baselines(*, context: RuntimeContext) -> Dict[str, Dict[str, Tensor]]:
    """Baseline (km) for every (detector, reactor) pair.

    Returns:
        Nested dict ``{detector: {reactor: baseline_km_tensor}}``.
    """
    _, rows = _read_csv_rows(default.DAYABAY_DATA_DIR / "baselines.csv")
    result: Dict[str, Dict[str, Tensor]] = {}
    for row in rows:
        result.setdefault(row["detector"], {})[row["reactor"]] = as_tensor(
            float(row["baseline_km"]), device=context.device, dtype=context.dtype,
        )
    return result


def load_n_protons(*, context: RuntimeContext) -> Dict[str, Tensor]:
    """Target proton count per detector."""
    _, rows = _read_csv_rows(default.DAYABAY_DATA_DIR / "n_protons.csv")
    return {
        row["detector"]: as_tensor(float(row["n_protons"]), device=context.device, dtype=context.dtype)
        for row in rows
    }


def load_eres_parameters() -> Tuple[float, float, float]:
    """Energy-resolution formula coefficients (a_nonuniform, b_stat, c_noise)."""
    _, rows = _read_csv_rows(default.DAYABAY_DATA_DIR / "eres_parameters.csv")
    values = {row["parameter"]: float(row["value"]) for row in rows}
    return values["a_nonuniform"], values["b_stat"], values["c_noise"]


def load_detector_efficiency() -> float:
    """IBD-selection efficiency (Gd-capture + delayed-coincidence + analysis cuts).

    Applied separately from the effective livetime; no normalization
    agreement with the observed spectrum is used to define or validate it.
    """
    return float((default.DAYABAY_DATA_DIR / "detector_efficiency.txt").read_text())


def load_ibd_spectrum(detector: str, *, context: RuntimeContext) -> Tuple[Tensor, Tensor, Tensor]:
    """Observed IBD prompt-energy spectrum for one detector, 0.05 MeV bins, 0-12 MeV.

    Args:
        detector: Detector name, e.g. ``"AD11"``.
        context: RuntimeContext the returned tensors are built on.

    Returns:
        ``(E_min_MeV, E_max_MeV, N)``, each shape ``(240,)``; ``N`` is the
        observed integer candidate count per bin.
    """
    _, rows = _read_csv_rows(default.DAYABAY_DATA_DIR / "ibd_spectra" / f"ibd_spectrum_{detector}.csv")
    return (
        as_tensor([float(row["E_min_MeV"]) for row in rows], device=context.device, dtype=context.dtype),
        as_tensor([float(row["E_max_MeV"]) for row in rows], device=context.device, dtype=context.dtype),
        as_tensor([float(row["N"]) for row in rows], device=context.device, dtype=context.dtype),
    )


def load_background_rates() -> Dict[str, Dict[str, float]]:
    """Background rates/uncertainties table, all detectors/categories.

    Returns:
        Nested dict ``{label: {detector: value}}``, ``label`` one of
        ``"<category>_rate"``/``"<category>_uncertainty"`` for each entry
        of ``tpeanuts.experiments.dayabay.parameters.BG_CATEGORIES``.
    """
    fieldnames, rows = _read_csv_rows(default.DAYABAY_DATA_DIR / "background_rates_8AD.csv")
    detectors = [name for name in fieldnames if name != "Label"]
    return {row["Label"]: {detector: float(row[detector]) for detector in detectors} for row in rows}


def load_background_shape(category: str, detector: str, *, context: RuntimeContext) -> Tensor:
    """Normalized background spectrum shape (integrates to 1), 0.05 MeV bins, 0-12 MeV.

    Args:
        category: One of ``"accidentals"``, ``"alpha_neutron"``, ``"amc"``,
            ``"fast_neutrons"``, ``"lithium_helium"``.
        detector: Detector name, e.g. ``"AD11"``.
        context: RuntimeContext the returned tensor is built on.

    Returns:
        Tensor shaped ``(240,)``.
    """
    _, rows = _read_csv_rows(
        default.DAYABAY_DATA_DIR / "background_shapes" / f"spectrum_shape_{category}_{detector}.csv"
    )
    return as_tensor([float(row["N"]) for row in rows], device=context.device, dtype=context.dtype)


def load_exposure(*, context: RuntimeContext) -> Dict[str, Tensor]:
    """8AD-period effective livetime (seconds, already efficiency-corrected) per detector."""
    _, rows = _read_csv_rows(default.DAYABAY_DATA_DIR / "exposure_8AD.csv")
    return {
        row["detector"]: as_tensor(float(row["eff_livetime_seconds"]), device=context.device, dtype=context.dtype)
        for row in rows
    }


def load_final_erec_bin_edges(*, context: RuntimeContext) -> Tensor:
    """Non-uniform analysis-binning edges, 0.7-12.0 MeV, shape (27,)."""
    _, rows = _read_csv_rows(default.DAYABAY_DATA_DIR / "final_erec_bin_edges.csv")
    return as_tensor([float(row["E_rec_MeV"]) for row in rows], device=context.device, dtype=context.dtype)


def load_survival_probability_truth() -> Dict[str, float]:
    """Daya Bay's own published best-fit parameters, for comparison only (never a model input)."""
    _, rows = _read_csv_rows(default.DAYABAY_DATA_DIR / "survival_probability_truth.csv")
    return {row["parameter"]: float(row["value"]) for row in rows}


def load_ibd_constants() -> Dict[str, float]:
    """IBD cross-section coupling constants: f, g, f2, PhaseSpaceFactor."""
    _, rows = _read_csv_rows(default.DAYABAY_DATA_DIR / "ibd_constants.csv")
    return {row["parameter"]: float(row["value"]) for row in rows}


def load_iav_matrix(*, context: RuntimeContext) -> Tensor:
    """IAV energy-redistribution matrix R(reco|true), 240x240, 0.05 MeV bins, 0-12 MeV.

    Columns sum to 1 (probability mass, not a density -- see
    ``tpeanuts.experiments.dayabay.response`` for the density conversion).
    Header-less CSV (a raw matrix, not a labeled table), so this uses
    ``csv.reader`` directly rather than ``_read_csv_rows``' own
    ``DictReader``.

    Returns:
        Tensor shaped ``(240, 240)``.
    """
    with open(default.DAYABAY_DATA_DIR / "iav_matrix.csv", newline="", encoding="utf-8") as csv_file:
        rows = [[float(value) for value in row] for row in csv.reader(csv_file)]
    return torch.as_tensor(rows, device=context.device, dtype=context.dtype)


def load_lsnl_curve(*, context: RuntimeContext) -> Tuple[Tensor, Tensor]:
    """LSNL nominal energy-scale nonlinearity curve, f(E) = E_reco / E_true_deposited.

    Returns:
        ``(E_MeV, f)``, each 1-D tensor, sorted by energy.
    """
    _, rows = _read_csv_rows(default.DAYABAY_DATA_DIR / "lsnl_curve_nominal.csv")
    rows = sorted(rows, key=lambda row: float(row["E_MeV"]))
    return (
        as_tensor([float(row["E_MeV"]) for row in rows], device=context.device, dtype=context.dtype),
        as_tensor([float(row["f"]) for row in rows], device=context.device, dtype=context.dtype),
    )


def load_lsnl_curve_pulls(*, context: RuntimeContext) -> List[Tuple[Tensor, Tensor]]:
    """LSNL systematic-variation ("pull") curves, f_k(E), k=0..3.

    Combined with the nominal curve via the official linear model
    ``f(E) = f0(E) + sum_k a_k*(f_k(E)-f0(E))``, each ``a_k`` a free
    nuisance with a unit Gaussian prior -- see
    ``tpeanuts.experiments.dayabay.response``.

    Returns:
        A 4-element list of ``(E_MeV, f_k)`` tensor pairs, in pull-index
        order, each sorted by energy.
    """
    _, rows = _read_csv_rows(default.DAYABAY_DATA_DIR / "lsnl_curve_pulls.csv")
    grouped: Dict[int, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(int(row["pull_index"]), []).append(row)

    result: List[Tuple[Tensor, Tensor]] = []
    for pull_index in sorted(grouped):
        pull_rows = sorted(grouped[pull_index], key=lambda row: float(row["E_MeV"]))
        result.append((
            as_tensor([float(row["E_MeV"]) for row in pull_rows], device=context.device, dtype=context.dtype),
            as_tensor([float(row["f"]) for row in pull_rows], device=context.device, dtype=context.dtype),
        ))
    return result


def load_snf_correction(*, context: RuntimeContext) -> Dict[str, Tuple[Tensor, Tensor]]:
    """Per-reactor spent-nuclear-fuel flux correction C(E), all 6 cores.

    Returns:
        ``{reactor: (E_MeV, C)}``, each a 1-D tensor, sorted by energy.
    """
    _, rows = _read_csv_rows(default.DAYABAY_DATA_DIR / "snf_correction.csv")
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["reactor"], []).append(row)

    result: Dict[str, Tuple[Tensor, Tensor]] = {}
    for reactor, reactor_rows in grouped.items():
        reactor_rows = sorted(reactor_rows, key=lambda row: float(row["E_MeV"]))
        result[reactor] = (
            as_tensor([float(row["E_MeV"]) for row in reactor_rows], device=context.device, dtype=context.dtype),
            as_tensor([float(row["C"]) for row in reactor_rows], device=context.device, dtype=context.dtype),
        )
    return result


def load_neutrino_rate_weekly() -> Dict[str, List[Tuple[int, float]]]:
    """Per-reactor weekly ``(n_det, neutrino_rate_per_s)`` history, full 2011-2020 data-taking period.

    Only the two columns
    ``tpeanuts.experiments.dayabay.parameters._reactor_relative_weights``
    actually reduces over are kept (see module docstring).

    Returns:
        ``{reactor: [(n_det, neutrino_rate_per_s), ...]}``, one entry per
        weekly row, in file order.
    """
    _, rows = _read_csv_rows(default.DAYABAY_DATA_DIR / "neutrino_rate_weekly.csv")
    result: Dict[str, List[Tuple[int, float]]] = {}
    for row in rows:
        result.setdefault(row["reactor"], []).append((int(row["n_det"]), float(row["neutrino_rate_per_s"])))
    return result


def load_global_normalization() -> float:
    """Daya Bay's own nominal global-normalization value (1.0)."""
    return float((default.DAYABAY_DATA_DIR / "global_normalization.txt").read_text())

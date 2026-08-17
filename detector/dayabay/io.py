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
Loaders for the real Daya Bay data release, cached CSVs under ``data/detector/dayabay/``.

Every function here reads a file written by
``notebooks/external/dayabay/DayaBay1_generator.ipynb`` -- see that
notebook and this package's own module docstring for provenance and scope.

Module contents:
    load_baselines(...)
        (detector, reactor) -> baseline in km, real published geometry.
    load_reactor_parameters(...)
        Real fission fractions, energy-per-fission, and thermal power.
    load_huber_mueller_spectra(...)
        Real per-isotope antineutrino spectrum, (E_MeV, N_per_fission_per_MeV)
        pairs.
    load_n_protons(...)
        Real target proton count per detector.
    load_eres_parameters(...)
        Real 3-parameter energy-resolution formula coefficients.
    load_ibd_spectrum(...)
        Real observed IBD prompt-energy spectrum for one detector.
    load_background_rates(...)
        Real background rates/uncertainties table (all detectors/categories).
    load_background_shape(...)
        Real normalized background spectrum shape for one (category, detector).
    load_exposure(...)
        Real 8AD-period effective livetime per detector.
    load_final_erec_bin_edges(...)
        Real analysis-binning edges.
    load_survival_probability_truth(...)
        Daya Bay's own published best-fit parameters, for comparison only.
    load_ibd_constants(...)
        Real IBD cross-section coupling constants (f, g, f2, phase-space
        factor).
    load_iav_matrix(...)
        Real IAV energy-redistribution matrix, 240x240, 0-12 MeV.
    load_lsnl_curve(...)
        Real LSNL nominal energy-scale nonlinearity curve.
    load_lsnl_curve_pulls(...)
        Real LSNL systematic-variation ("pull") curves, 4 of them.
    load_nonequilibrium_correction(...)
        Real per-isotope non-equilibrium flux correction (U235/Pu239/Pu241).
    load_snf_correction(...)
        Real per-reactor spent-nuclear-fuel flux correction (R1-R6).
    load_neutrino_rate_weekly(...)
        Real weekly-resolved per-reactor antineutrino rate history.
    load_global_normalization(...)
        Daya Bay's own real nominal global-normalization value (1.0).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import torch

from tpeanuts.util.io import package_dir

_DAYABAY_DIR = package_dir() / "data" / "detector" / "dayabay"


def load_baselines(
    *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64,
) -> dict[str, dict[str, torch.Tensor]]:
    """Real baseline (km) for every (detector, reactor) pair.

    Returns:
        Nested dict ``{detector: {reactor: baseline_km_tensor}}``.
    """
    table = pd.read_csv(_DAYABAY_DIR / "baselines.csv")
    result: dict[str, dict[str, torch.Tensor]] = {}
    for _, row in table.iterrows():
        result.setdefault(row["detector"], {})[row["reactor"]] = torch.tensor(
            float(row["baseline_km"]), device=device, dtype=dtype,
        )
    return result


def load_reactor_parameters() -> tuple[dict[str, float], dict[str, float], float]:
    """Real fission fractions, energy-per-fission, and nominal thermal power.

    Returns:
        ``(fission_fractions, energy_per_fission_MeV, thermal_power_gw)``:
        the first two are ``{isotope: value}`` dicts (isotope in
        ``detector.dayabay.parameters.ISOTOPES``); the third is a scalar
        float, GW, shared by every one of the 6 real reactor cores in this
        data tier (see package module docstring).
    """
    table = pd.read_csv(_DAYABAY_DIR / "reactor_parameters.csv")
    fission_fractions = dict(zip(table["isotope"], table["fission_fraction"]))
    energy_per_fission = dict(zip(table["isotope"], table["energy_per_fission_MeV"]))
    thermal_power_gw = float((_DAYABAY_DIR / "reactor_thermal_power_gw.txt").read_text())
    return fission_fractions, energy_per_fission, thermal_power_gw


def load_huber_mueller_spectra(
    *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Real per-isotope Huber-Mueller antineutrino spectrum.

    Returns:
        ``{isotope: (E_MeV, N_per_fission_per_MeV)}``, each a 1-D tensor,
        sorted by energy.
    """
    table = pd.read_csv(_DAYABAY_DIR / "huber_mueller_spectra.csv")
    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for isotope, group in table.groupby("isotope"):
        group = group.sort_values("E_MeV")
        E = torch.tensor(group["E_MeV"].to_numpy(), device=device, dtype=dtype)
        N = torch.tensor(group["N_per_fission_per_MeV"].to_numpy(), device=device, dtype=dtype)
        result[isotope] = (E, N)
    return result


def load_n_protons(
    *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Real target proton count per detector."""
    table = pd.read_csv(_DAYABAY_DIR / "n_protons.csv")
    return {
        row["detector"]: torch.tensor(float(row["n_protons"]), device=device, dtype=dtype)
        for _, row in table.iterrows()
    }


def load_eres_parameters() -> tuple[float, float, float]:
    """Real energy-resolution formula coefficients (a_nonuniform, b_stat, c_noise)."""
    table = pd.read_csv(_DAYABAY_DIR / "eres_parameters.csv")
    values = dict(zip(table["parameter"], table["value"]))
    return float(values["a_nonuniform"]), float(values["b_stat"]), float(values["c_noise"])


def load_detector_efficiency() -> float:
    """Real IBD-selection efficiency (Gd-capture + delayed-coincidence + analysis cuts).

    Loaded directly from ``parameters/detector_efficiency.yaml`` in the
    official data release. It is applied separately from the effective
    livetime; no normalization agreement with the observed spectrum is
    used to define or validate it.
    """
    return float((_DAYABAY_DIR / "detector_efficiency.txt").read_text())


def load_ibd_spectrum(
    detector: str,
    *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Real observed IBD prompt-energy spectrum for one detector, 0.05 MeV bins, 0-12 MeV.

    Args:
        detector: Detector name, e.g. "AD11".

    Returns:
        ``(E_min_MeV, E_max_MeV, N)``, each shape ``(240,)``; ``N`` is the
        real observed integer candidate count per bin.
    """
    table = pd.read_csv(_DAYABAY_DIR / "ibd_spectra" / f"ibd_spectrum_{detector}.csv")
    return (
        torch.tensor(table["E_min_MeV"].to_numpy(), device=device, dtype=dtype),
        torch.tensor(table["E_max_MeV"].to_numpy(), device=device, dtype=dtype),
        torch.tensor(table["N"].to_numpy(), device=device, dtype=dtype),
    )


def load_background_rates() -> pd.DataFrame:
    """Real background rates/uncertainties table, indexed by Label, one column per detector."""
    return pd.read_csv(_DAYABAY_DIR / "background_rates_8AD.csv").set_index("Label")


def load_background_shape(
    category: str,
    detector: str,
    *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Real normalized background spectrum shape (integrates to 1), 0.05 MeV bins, 0-12 MeV.

    Args:
        category: One of "accidentals", "alpha_neutron", "amc",
            "fast_neutrons", "lithium_helium".
        detector: Detector name, e.g. "AD11".

    Returns:
        Real tensor shaped ``(240,)``.
    """
    table = pd.read_csv(_DAYABAY_DIR / "background_shapes" / f"spectrum_shape_{category}_{detector}.csv")
    return torch.tensor(table["N"].to_numpy(), device=device, dtype=dtype)


def load_exposure(
    *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Real 8AD-period effective livetime (seconds, already efficiency-corrected) per detector."""
    table = pd.read_csv(_DAYABAY_DIR / "exposure_8AD.csv")
    return {
        row["detector"]: torch.tensor(float(row["eff_livetime_seconds"]), device=device, dtype=dtype)
        for _, row in table.iterrows()
    }


def load_final_erec_bin_edges(
    *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Real (non-uniform) analysis-binning edges, 0.7-12.0 MeV, shape (27,)."""
    table = pd.read_csv(_DAYABAY_DIR / "final_erec_bin_edges.csv")
    return torch.tensor(table["E_rec_MeV"].to_numpy(), device=device, dtype=dtype)


def load_survival_probability_truth() -> dict[str, float]:
    """Daya Bay's own published best-fit parameters, for comparison only (never a model input)."""
    table = pd.read_csv(_DAYABAY_DIR / "survival_probability_truth.csv")
    return dict(zip(table["parameter"], table["value"]))


def load_ibd_constants() -> dict[str, float]:
    """Real IBD cross-section coupling constants: f, g, f2, PhaseSpaceFactor."""
    table = pd.read_csv(_DAYABAY_DIR / "ibd_constants.csv")
    return dict(zip(table["parameter"], table["value"]))


def load_iav_matrix(
    *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Real IAV energy-redistribution matrix R(reco|true), 240x240, 0.05 MeV bins, 0-12 MeV.

    Columns sum to 1 (probability mass, not a density -- see
    ``detector.dayabay.response`` for the density conversion).

    Returns:
        Real tensor shaped ``(240, 240)``.
    """
    table = pd.read_csv(_DAYABAY_DIR / "iav_matrix.csv", header=None)
    return torch.tensor(table.to_numpy(), device=device, dtype=dtype)


def load_lsnl_curve(
    *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real LSNL nominal energy-scale nonlinearity curve, f(E) = E_reco / E_true_deposited.

    Returns:
        ``(E_MeV, f)``, each 1-D tensor, sorted by energy.
    """
    table = pd.read_csv(_DAYABAY_DIR / "lsnl_curve_nominal.csv").sort_values("E_MeV")
    return (
        torch.tensor(table["E_MeV"].to_numpy(), device=device, dtype=dtype),
        torch.tensor(table["f"].to_numpy(), device=device, dtype=dtype),
    )


def load_lsnl_curve_pulls(
    *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Real LSNL systematic-variation ("pull") curves, f_k(E), k=0..3.

    Combined with the nominal curve via the official linear model
    ``f(E) = f0(E) + sum_k a_k*(f_k(E)-f0(E))``
    (``parameters/detector_lsnl.yaml``), each ``a_k`` a real free nuisance
    with a unit Gaussian prior -- see ``detector.dayabay.response``.

    Returns:
        A 4-element list of ``(E_MeV, f_k)`` tensor pairs, in pull-index
        order, each sorted by energy.
    """
    table = pd.read_csv(_DAYABAY_DIR / "lsnl_curve_pulls.csv")
    result: list[tuple[torch.Tensor, torch.Tensor]] = []
    for k in sorted(table["pull_index"].unique()):
        group = table[table["pull_index"] == k].sort_values("E_MeV")
        result.append((
            torch.tensor(group["E_MeV"].to_numpy(), device=device, dtype=dtype),
            torch.tensor(group["f"].to_numpy(), device=device, dtype=dtype),
        ))
    return result


def load_nonequilibrium_correction(
    *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Real per-isotope non-equilibrium flux correction C(E) (U235, Pu239, Pu241 only).

    Returns:
        ``{isotope: (E_MeV, C)}``, each a 1-D tensor, sorted by energy.
    """
    table = pd.read_csv(_DAYABAY_DIR / "nonequilibrium_correction.csv")
    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for isotope, group in table.groupby("isotope"):
        group = group.sort_values("E_MeV")
        result[isotope] = (
            torch.tensor(group["E_MeV"].to_numpy(), device=device, dtype=dtype),
            torch.tensor(group["C"].to_numpy(), device=device, dtype=dtype),
        )
    return result


def load_snf_correction(
    *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Real per-reactor spent-nuclear-fuel flux correction C(E), all 6 real cores.

    Returns:
        ``{reactor: (E_MeV, C)}``, each a 1-D tensor, sorted by energy.
    """
    table = pd.read_csv(_DAYABAY_DIR / "snf_correction.csv")
    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for reactor, group in table.groupby("reactor"):
        group = group.sort_values("E_MeV")
        result[reactor] = (
            torch.tensor(group["E_MeV"].to_numpy(), device=device, dtype=dtype),
            torch.tensor(group["C"].to_numpy(), device=device, dtype=dtype),
        )
    return result


def load_neutrino_rate_weekly() -> pd.DataFrame:
    """Real weekly-resolved per-reactor antineutrino rate history, full 2011-2020 data-taking period.

    Returns:
        A single ``pandas.DataFrame`` with columns ``reactor, period, day,
        start_utc, end_utc, n_days, n_det, n_det_mask, neutrino_rate_per_s``,
        all 6 real reactor cores concatenated.
    """
    return pd.read_csv(_DAYABAY_DIR / "neutrino_rate_weekly.csv")


def load_global_normalization() -> float:
    """Daya Bay's own real nominal global-normalization value (``detector_normalization.yaml``, 1.0)."""
    return float((_DAYABAY_DIR / "global_normalization.txt").read_text())

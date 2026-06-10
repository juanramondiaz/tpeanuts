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
Small solar-analysis helpers shared by notebooks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_spectrum_table(source: str, spectra_dir: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a solar source spectrum CSV as energy and density arrays.

    Args:
        source: Solar source name used in files named spectrum_<source>.csv.
        spectra_dir: Directory containing the spectrum CSV files.

    Returns:
        Tuple with energy values and spectrum densities as float arrays.

    Raises:
        FileNotFoundError: If the expected source CSV is missing.
    """
    path = Path(spectra_dir) / f"spectrum_{source}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing spectrum file for {source}: {path}")

    table = pd.read_csv(path)
    return table.iloc[:, 0].to_numpy(dtype=float), table.iloc[:, 1].to_numpy(dtype=float)


def normalized_spectrum(
    source: str,
    energy_mev: np.ndarray,
    spectra_dir: Path | str,
) -> np.ndarray:
    """
    Interpolate and normalize a solar source spectrum on an energy grid.

    Args:
        source: Solar source name.
        energy_mev: Target energy grid in MeV.
        spectra_dir: Directory containing spectrum_<source>.csv files.

    Returns:
        Interpolated spectrum normalized to unit trapezoidal integral.
    """
    spec_E, spec_y = load_spectrum_table(source, spectra_dir)
    values = np.interp(energy_mev, spec_E, spec_y, left=0.0, right=0.0)
    norm = np.trapz(values, x=energy_mev)
    return values / norm if norm > 0 else values


def relative_flux_density(
    source: str,
    energy_mev: np.ndarray,
    flux_value: float,
    spectra_dir: Path | str,
) -> np.ndarray:
    """
    Scale a normalized solar source spectrum by a total flux value.

    Args:
        source: Solar source name.
        energy_mev: Target energy grid in MeV.
        flux_value: Integrated source flux used as the normalization factor.
        spectra_dir: Directory containing spectrum_<source>.csv files.

    Returns:
        Differential flux density on energy_mev.
    """
    return float(flux_value) * normalized_spectrum(source, energy_mev, spectra_dir)

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

"""Perturbative density-profile model selection.

This module centralizes the mapping between public model names and concrete
perturbative profile classes. Medium modules can request a model by name
without importing model-specific classes directly.

Module functions:
    perturbative_profile_selection(...): Instantiate and return the layered
        profile for a perturbative model name and its construction kwargs.
"""

from __future__ import annotations

from tpeanuts.core.perturbative.models.even_power.profile_layered import (
    EvenPowerProfileLayered,
)
from tpeanuts.core.perturbative.models.prem.profile_layered import (
    PremTabulatedProfile,
)


def perturbative_profile_selection(
    profile_perturbative_name: str,
    profile_perturbative_kwargs: dict,
) -> object:
    """Instantiate and return the layered perturbative profile selected by name.

    Args:
        profile_perturbative_name: Public model name. Supported values are:

            * ``"even_power"`` / ``"even-power"`` / ``"EvenPowerProfileLayered"``
              — polynomial even-power model.
            * ``"prem500"`` / ``"prem_tabulated"`` / ``"prem"`` /
              ``"PremTabulatedProfile"`` — PREM piecewise-linear-in-r² model.

        profile_perturbative_kwargs: Keyword arguments forwarded verbatim to
            the selected model's constructor.

    Returns:
        Instantiated layered perturbative profile for the requested model.

    Raises:
        ValueError: If the requested model name is unknown.
    """
    normalized = profile_perturbative_name.strip().lower().replace("-", "_")

    if normalized in {
        "even_power",
        "even_power_profile",
        "even_power_profile_layered",
        "evenpowerprofilelayered",
    }:
        return EvenPowerProfileLayered(**profile_perturbative_kwargs)

    if normalized in {
        "prem500",
        "prem_tabulated",
        "prem",
        "premtabulatedprofile",
        "prem_tabulated_profile",
    }:
        return PremTabulatedProfile(**profile_perturbative_kwargs)

    raise ValueError(
        f"Unknown perturbative profile model: {profile_perturbative_name!r}."
    )

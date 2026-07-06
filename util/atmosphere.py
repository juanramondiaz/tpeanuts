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
Small Atmosphere-analysis helpers shared by notebooks.

The functions here are intentionally lightweight: they do not load Atmosphere
datasets or perform physics calculations, but provide common labels and grid
selection rules used by Atmosphere notebooks.
"""

from __future__ import annotations

from typing import Any


def angle_grid_for(group: dict[str, Any]) -> tuple[str, Any]:
    """
    Return the angular-grid name and values stored in an Atmosphere group.

    Args:
        group: Loaded Atmosphere group returned by the Atmosphere I/O layer.

    Returns:
        Tuple with the angle name, either "alpha" or "theta", and the
        corresponding grid values.

    Raises:
        KeyError: If neither alpha_grid_deg nor theta_grid_deg is available.
    """
    if "alpha_grid_deg" in group:
        return "alpha", group["alpha_grid_deg"]

    if "theta_grid_deg" in group:
        return "theta", group["theta_grid_deg"]

    raise KeyError("Atmosphere group must contain alpha_grid_deg or theta_grid_deg.")


def particle_label(name: str) -> str:
    """
    Return a LaTeX label for a standard neutrino particle name.

    Args:
        name: Particle key such as "nue", "numu", or "antinumu".

    Returns:
        LaTeX label for known neutrino keys; otherwise the original name.
    """
    labels = {
        "nue": r"$\nu_e$",
        "numu": r"$\nu_\mu$",
        "nutau": r"$\nu_\tau$",
        "antinue": r"$\bar\nu_e$",
        "antinumu": r"$\bar\nu_\mu$",
        "antinutau": r"$\bar\nu_\tau$",
    }
    return labels.get(str(name), str(name))

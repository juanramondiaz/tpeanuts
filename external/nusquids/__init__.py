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
Optional nuSQuIDS reference backend.

nuSQuIDS is an independent, external C++ (with Python bindings) neutrino
oscillation solver. This package is a thin tpeanuts-native adapter around
the optional nuSQuIDS Python bindings: tpeanuts does not reimplement
nuSQuIDS's physics here, it only calls into the installed nuSQuIDS module to
build a three-flavour solver, configure its mixing parameters, and evaluate
flavour-transition probabilities in vacuum, through nuSQuIDS's built-in
"Earth" body, or through its "EarthAtm" atmosphere+Earth body. The sole
purpose of this backend is cross-validation: running the same oscillation
scenario through nuSQuIDS and through tpeanuts's own torch propagation code
to check that the two independent implementations agree.

Module contents:
    NuSQuIDSConfig
        Dataclass with the oscillation parameters (mixing angles, CP phase,
        mass splittings) and numerical solver tolerances passed to
        nuSQuIDS.
    NuSQuIDSError
        Raised when the nuSQuIDS bindings are unavailable or a configured
        operation is not supported by the installed bindings.
    is_available(...)
        Check whether the nuSQuIDS Python bindings can be imported.
    require_nusquids(...)
        Import and return the nuSQuIDS Python module, raising
        NuSQuIDSError with installation guidance if unavailable.
    init_solver(...), configure_solver(...)
        Create and configure a single-energy three-flavour nuSQuIDS solver.
    units(...), initial_state(...), eval_probabilities(...),
    evolve_with_body(...)
        Shared public helpers for notebook and validation code.
    probability_vacuum(...), probability_earth(...),
    probability_atmosphere(...), probability_grid_vacuum(...)
        Evaluate final-flavour probabilities after propagation through
        vacuum, the nuSQuIDS Earth body, or the nuSQuIDS EarthAtm body.
    atmosphere_density_nusquids(...)
        Evaluate the (tpeanuts-native reimplementation of the) nuSQuIDS
        EarthAtm exponential atmosphere mass-density formula, for use as a
        density backend inside tpeanuts's own propagation code.
"""

from tpeanuts.external.nusquids.core import (
    NuSQuIDSConfig,
    NuSQuIDSError,
    configure_solver,
    eval_probabilities,
    eval_flavour_averaged,
    eval_mass_weights,
    evolve_with_body,
    init_solver,
    initial_state,
    is_available,
    make_solar_track,
    neutrino_type,
    normalise_flavour_label,
    probability_atmosphere,
    probability_earth,
    probability_earth_massbasis,
    probability_grid_vacuum,
    probability_solar_point,
    probability_vacuum,
    require_nusquids,
    set_cp_phase,
    sun_asnu_radius,
    sun_asnu_track_fraction,
    transition_matrix_earth_mass_to_flavour,
    units,
)
from tpeanuts.external.nusquids.density import (
    atmosphere_density_nusquids,
)

__all__ = [
    "NuSQuIDSConfig",
    "NuSQuIDSError",
    "atmosphere_density_nusquids",
    "configure_solver",
    "eval_probabilities",
    "eval_flavour_averaged",
    "eval_mass_weights",
    "evolve_with_body",
    "init_solver",
    "initial_state",
    "is_available",
    "make_solar_track",
    "neutrino_type",
    "normalise_flavour_label",
    "probability_atmosphere",
    "probability_earth",
    "probability_earth_massbasis",
    "probability_grid_vacuum",
    "probability_solar_point",
    "probability_vacuum",
    "require_nusquids",
    "set_cp_phase",
    "sun_asnu_radius",
    "sun_asnu_track_fraction",
    "transition_matrix_earth_mass_to_flavour",
    "units",
]

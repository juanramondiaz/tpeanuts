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

"""Oscillation and numerical configuration passed to nuSQuIDS.

Ported from v1_0's ``validation/nusquids/core.py::NuSQuIDSConfig``, split
into its own module. Imports nothing from ``tpeanuts`` -- unlike
v1_0's own version, whose ``nusquids_rho0_gcm3``/
``nusquids_scale_height_km``/``nusquids_Ye`` fields defaulted to
``tpeanuts.config.default`` values, this module's own defaults are its
own literal constants (nuSQuIDS's real ``EarthAtm`` atmosphere-density
formula, ``rho(h) = 0.0012 * exp(-h / 7.594 km)``, a well-known,
independently-checkable value, not borrowed from tpeanuts's own
config system).

Module contents:
    EARTHATM_RHO0_GCM3, EARTHATM_SCALE_HEIGHT_KM, EARTHATM_YE
        nuSQuIDS's own hard-coded ``EarthAtm`` exponential-atmosphere
        constants (see ``NuSQuIDSConfig``'s own docstring for how a
        tpeanuts-side comparison reproduces this, without needing any
        nuSQuIDS-side reimplementation -- see
        ``validation.nusquids.compare.atmosphere``, added in a later
        phase).
    NuSQuIDSConfig
        Dataclass with the oscillation parameters (mixing angles, CP
        phase, mass splittings) and numerical solver tolerances passed
        to nuSQuIDS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# nuSQuIDS's own hard-coded EarthAtm exponential atmosphere-density formula:
# rho(h) = EARTHATM_RHO0_GCM3 * exp(-h_km / EARTHATM_SCALE_HEIGHT_KM), for h
# above the Earth's surface. A real, well-known constant of the nuSQuIDS
# solver itself -- not a tpeanuts default. The electron fraction of that atmosphere is 0.494, read from
# EarthAtm.ye() along a track (notebooks/3_validation/2_external/1_nusquids/03_atmosphere); it is not 0.5.
EARTHATM_RHO0_GCM3: float = 0.0012
EARTHATM_SCALE_HEIGHT_KM: float = 7.594
EARTHATM_YE: float = 0.494


@dataclass(frozen=True)
class NuSQuIDSConfig:
    """Numerical and oscillation configuration for nuSQuIDS.

    Angles are in radians, mass splittings are in eV^2, energies passed
    to the public probability helpers are in GeV, and distances are in
    km. These values are forwarded to nuSQuIDS's own setter methods
    (e.g. ``Set_MixingAngle``, ``Set_SquareMassDifference``) in
    ``adapter.configure_solver`` -- they never touch tpeanuts's own
    oscillation-parameter configuration, which a comparison script
    (``validation.nusquids.compare.*``) sets up completely independently,
    from the very same physical values, on its own side.

    Attributes:
        theta12: Solar mixing angle theta_12 in radians.
        theta13: Reactor mixing angle theta_13 in radians.
        theta23: Atmospheric mixing angle theta_23 in radians.
        delta_cp: Dirac CP-violating phase in radians.
        DeltamSq21: Solar mass-squared splitting Delta m^2_21 in eV^2.
        DeltamSq3l: Atmospheric mass-squared splitting Delta m^2_3l in
            eV^2 (l=1 for normal ordering, l=2 for inverted, per
            nuSQuIDS convention).
        numneu: Number of neutrino flavours passed to
            ``nsq.nuSQUIDS``. 3 for the plain Standard Model (default);
            4 for the 3+1 sterile extension, in which case
            ``theta14``/``theta24``/``theta34``/``delta14``/
            ``delta24``/``DeltamSq41`` below are also applied.
        theta14, theta24, theta34: Active-sterile mixing angles in
            radians (ignored unless ``numneu >= 4``).
        delta14, delta24: Active-sterile CP phases in radians (ignored
            unless ``numneu >= 4``); there is no ``delta34`` (R_34 is a
            real rotation for Dirac 3+1 neutrinos).
        DeltamSq41: Sterile mass-squared splitting Delta m^2_41 in eV^2
            (ignored unless ``numneu >= 4``).
        rel_error: Relative error tolerance for nuSQuIDS's internal ODE
            integrator (``Set_rel_error``), if supported by the
            installed bindings.
        abs_error: Absolute error tolerance for nuSQuIDS's internal ODE
            integrator (``Set_abs_error``), if supported.
        h_max_km: Optional maximum internal integrator step size in km
            (converted to nuSQuIDS's natural units and passed to
            ``Set_h_max``). None disables this setting.
        set_default_mixing_first: If True, call nuSQuIDS's
            ``Set_MixingParametersToDefault`` before applying the
            angles/splittings above, when the installed bindings
            expose it.
        set_cp_phase: If True, attempt to set ``delta_cp`` on the
            solver using whichever CP-phase setter the installed
            nuSQuIDS bindings expose.
        strict_cp: If True, raise ``NuSQuIDSError`` when no compatible
            CP-phase setter is found; if False, only warn and continue
            with the bindings' default CP phase.
        earthatm_rho0_gcm3: Zero-altitude reference mass density in
            g/cm^3 for nuSQuIDS's own ``EarthAtm`` exponential
            atmosphere formula (see ``EARTHATM_RHO0_GCM3``).
        earthatm_scale_height_km: Exponential scale height in km for
            the same atmosphere mass-density formula.
        earthatm_Ye: Electron fraction (dimensionless, electrons per
            nucleon) used to convert that mass density into electron
            density.
    """

    theta12: float = 0.59
    theta13: float = 0.15
    theta23: float = 0.78
    delta_cp: float = 1.20
    DeltamSq21: float = 7.42e-5
    DeltamSq3l: float = 2.517e-3
    numneu: int = 3
    theta14: float = 0.0
    theta24: float = 0.0
    theta34: float = 0.0
    delta14: float = 0.0
    delta24: float = 0.0
    DeltamSq41: float = 1.0
    rel_error: float = 1.0e-11
    abs_error: float = 1.0e-13
    h_max_km: Optional[float] = 100.0
    set_default_mixing_first: bool = False
    set_cp_phase: bool = True
    strict_cp: bool = False
    earthatm_rho0_gcm3: float = EARTHATM_RHO0_GCM3
    earthatm_scale_height_km: float = EARTHATM_SCALE_HEIGHT_KM
    earthatm_Ye: float = EARTHATM_YE

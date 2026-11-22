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

"""Shared tpeanuts-side oscillation model construction, from one NuSQuIDSConfig.

Extracted out of ``compare/vacuum.py`` once ``compare/earth.py`` needed
the exact same construction a second time -- both build a plain
three-flavour Dirac ``MixingMatrix``/``MassSpectrum`` from the very
same ``validation.nusquids.config.NuSQuIDSConfig`` instance's
oscillation fields, so that a comparison's two sides can never differ
because they used different physical inputs.
``sterile_mixing_and_mass_spectrum`` is the 3+1 sterile-extended
sibling, added for ``compare/bsm_sterile.py``; ``evolutor_from_profile``
is the shared "build a PropagationEnvironment from a MediumProfile and
dispatch to evolve_path" step, extracted once
``compare/bsm_sterile.py``'s atmosphere leg needed the exact same
composition ``compare/atmosphere.py`` already had.

Module contents:
    mixing_and_mass_spectrum(...)
        Build a plain three-flavour Dirac MixingMatrix/MassSpectrum
        from a NuSQuIDSConfig's own oscillation fields.
    nusquids_sterile_mixing_matrix(...)
        The 4x4 mixing matrix of nuSQuIDS for the angles and phases of a
        NuSQuIDSConfig, ``R34 R24 R14 R23 R13 R12``.
    tpeanuts_sterile_parameters(...)
        The tpeanuts mixing parameters that give the same physical 3+1
        mixing as a NuSQuIDSConfig, up to rephasing.
    sterile_mixing_and_mass_spectrum(...)
        Build a 3+1 sterile-extended Dirac MixingMatrix/MassSpectrum
        from the same NuSQuIDSConfig's sterile fields.
    atmosphere_density_kwargs(...)
        nuSQuIDS's own hard-coded EarthAtm exponential-atmosphere
        constants, as AtmosphereMedium density_kwargs.
    evolutor_from_profile(...)
        Build the evolution operator for one already-built
        MediumProfile leg, at a fixed energy.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple

import numpy as np
import torch

from tpeanuts.physics.common.conventions import sterile_parameters_from_rotation_order
from tpeanuts.physics.common.mass_spectrum import MassSpectrum
from tpeanuts.physics.common.mixing import MixingMatrix, StandardMixingParameters
from tpeanuts.physics.common.nature import Dirac
from tpeanuts.physics.common.space import FlavourSpace
from tpeanuts.physics.common.terms import PropagationEnvironment
from tpeanuts.physics.evolution import PropagationMethod
from tpeanuts.physics.evolution.dispatch import evolve_path
from tpeanuts.physics.model import PropagationModel
from tpeanuts.util.constant import GCM3_TO_NUCLEON_MOLCM3, R_E
from tpeanuts.util.context import RuntimeContext
from validation.common import reference
from validation.common.tpeanuts_side import mixing_from_parameters
from validation.nusquids.config import NuSQuIDSConfig


def mixing_and_mass_spectrum(
    config: NuSQuIDSConfig, context: RuntimeContext,
) -> Tuple[MixingMatrix, MassSpectrum]:
    """Build a plain three-flavour Dirac MixingMatrix/MassSpectrum from ``config``'s own fields.

    Args:
        config: Oscillation parameters (``theta12``/``theta13``/
            ``theta23``/``delta_cp``/``DeltamSq21``/``DeltamSq3l``).
        context: RuntimeContext for the returned objects.

    Returns:
        ``(mixing, mass_spectrum)``, a plain ``FlavourSpace()`` (no
        sterile extension), ``nature=Dirac()``.
    """
    space = FlavourSpace()
    mixing = MixingMatrix(
        space=space,
        standard=StandardMixingParameters(
            theta12=config.theta12, theta13=config.theta13, theta23=config.theta23,
            delta13=config.delta_cp, context=context,
        ),
        nature=Dirac(),
        context=context,
    )
    mass_spectrum = MassSpectrum(
        space=space, DeltamSq21=config.DeltamSq21, DeltamSq3l=config.DeltamSq3l, context=context,
    )
    return mixing, mass_spectrum


def nusquids_sterile_mixing_matrix(config: NuSQuIDSConfig) -> np.ndarray:
    """The 4x4 mixing matrix of nuSQuIDS, ``R34 R24(delta24) R14(delta14) R23 R13(delta_cp) R12``.

    This is the order that reproduces the vacuum probabilities of the installed nuSQuIDS to rounding
    (notebooks/3_validation/2_external/1_nusquids/06_sterile_3p1, Section 3.1), with the mass splittings
    ``(0, Delta m^2_21, Delta m^2_3l, Delta m^2_41)``.

    Args:
        config: Supplies the three standard angles, ``delta_cp`` and the sterile angles and phases.

    Returns:
        Complex array shaped ``(4, 4)``.
    """
    return reference.mixing_matrix(replace(config, numneu=4))


def tpeanuts_sterile_parameters(config: NuSQuIDSConfig, context: RuntimeContext) -> np.ndarray:
    """tpeanuts mixing parameters that describe the same 3+1 mixing as the nuSQuIDS angles of ``config``.

    The conversion is ``tpeanuts.physics.common.conventions.sterile_parameters_from_rotation_order`` with the order of
    nuSQuIDS, ``R34 R24 R14 R23 R13 R12``: tpeanuts uses ``R23 Delta [R24 R34] R13 Delta^dagger R12 R14``, so the same
    numerical angles describe different physical mixings.

    Args:
        config: Oscillation parameters in the convention of nuSQuIDS.
        context: RuntimeContext of the intermediate parameter objects.

    Returns:
        Array of nine values ``(theta12, theta13, theta23, delta13, theta14, theta24, theta34, delta14, delta24)``
        in the convention of tpeanuts (``delta34 = 0``).
    """
    standard, sterile = sterile_parameters_from_rotation_order(
        config.theta12, config.theta13, config.theta23, config.delta_cp,
        config.theta14, config.theta24, config.theta34, config.delta14, config.delta24, context=context,
    )
    return np.array([
        float(standard.theta12), float(standard.theta13), float(standard.theta23), float(standard.delta13),
        float(sterile.theta_e[0]), float(sterile.theta_mu[0]), float(sterile.theta_tau[0]),
        float(sterile.delta_e[0]), float(sterile.delta_mu[0]),
    ])


def sterile_mixing_and_mass_spectrum(
    config: NuSQuIDSConfig, context: RuntimeContext,
) -> Tuple[MixingMatrix, MassSpectrum]:
    """Build a 3+1 sterile-extended Dirac MixingMatrix/MassSpectrum describing the physical mixing of ``config``.

    The angles and phases of ``config`` are those of nuSQuIDS, ``U = R34 R24 R14 R23 R13 R12``; they are converted
    with ``tpeanuts_sterile_parameters`` to the parametrisation of tpeanuts, which orders the rotations
    differently, so that both codes describe the same mixing. Passing the same numbers to both, as an earlier
    version of this function did, differs by up to 8e-2 in a probability.
    ``config.delta34`` does not exist: R_34 is a real rotation for Dirac 3+1 neutrinos in nuSQuIDS, which is
    ``delta_tau = 0`` in the conversion.

    Args:
        config: Oscillation parameters, including the sterile fields
            (``theta14``/``theta24``/``theta34``/``delta14``/
            ``delta24``/``DeltamSq41``).
        context: RuntimeContext for the returned objects.

    Returns:
        ``(mixing, mass_spectrum)``, ``FlavourSpace(n_sterile=1)``,
        ``nature=Dirac()``.
    """
    mixing = mixing_from_parameters(tpeanuts_sterile_parameters(config, context), context)
    mass_spectrum = MassSpectrum(
        space=mixing.space, DeltamSq21=config.DeltamSq21, DeltamSq3l=config.DeltamSq3l,
        DeltamSq_sterile=[config.DeltamSq41], context=context,
    )
    return mixing, mass_spectrum


def atmosphere_density_kwargs(config: NuSQuIDSConfig, *, match_nusquids_conversion: bool = True) -> dict:
    """nuSQuIDS's own hard-coded EarthAtm exponential-atmosphere constants, as AtmosphereMedium density_kwargs.

    Args:
        config: Supplies ``earthatm_rho0_gcm3``/``earthatm_scale_height_km``/
            ``earthatm_Ye``.
        match_nusquids_conversion: tpeanuts turns a mass density into a
            nucleon number density by dividing by the proton mass
            (``GCM3_TO_NUCLEON_MOLCM3``, 1/1.00728 mol per gram), while
            nuSQuIDS uses ``N_e = N_A rho Ye`` with one gram per mole. If
            True (default) the electron fraction passed to tpeanuts is
            divided by that factor, so both codes get the same electron
            density, which is what a propagation comparison needs; if
            False the fraction is passed as it is and the 0.72 % difference
            of the two conventions remains
            (notebooks/3_validation/2_external/1_nusquids/03_atmosphere).

    Returns:
        Dict with ``rho0_gcm3``/``scale_height_m``/``Ye`` keys, ready to
        pass as ``AtmosphereMedium(..., density_kwargs=...)``. This
        project's own general-purpose atmosphere defaults
        (``tpeanuts.config.default.ATMOSPHERE_RHO0_GCM3`` etc.) are a
        *different*, unrelated reference atmosphere and must not be
        used in a nuSQuIDS comparison.
    """
    electron_fraction = config.earthatm_Ye / GCM3_TO_NUCLEON_MOLCM3 if match_nusquids_conversion else config.earthatm_Ye
    return {
        "rho0_gcm3": config.earthatm_rho0_gcm3,
        "scale_height_m": config.earthatm_scale_height_km * 1.0e3,
        "Ye": electron_fraction,
    }


def evolutor_from_profile(
    medium_profile, model: PropagationModel, E_GeV: float, context: RuntimeContext, *, reverse: bool = False,
):
    """Build the evolution operator for one already-built MediumProfile leg, at a fixed energy.

    Args:
        medium_profile: A built ``tpeanuts.interfaces.medium.MediumProfile``
            (e.g. from ``VacuumMedium``/``EarthMedium``/
            ``AtmosphereMedium``).
        model: Built PropagationModel.
        E_GeV: Neutrino energy in GeV.
        context: RuntimeContext.
        reverse: True when segment 0 of the profile is the last one a neutrino crosses, as in the
            atmosphere leg (tiled from the detector up to the production point).

    Returns:
        The evolution operator, via ``PropagationMethod.ANALYTIC``
        (exact per constant-density segment).
    """
    E_MeV = torch.tensor(float(E_GeV) * 1.0e3, dtype=context.dtype, device=context.device)
    environment = PropagationEnvironment(
        E_MeV=E_MeV, n_e_mol_cm3=medium_profile.n_e_mol_cm3, n_n_mol_cm3=medium_profile.n_n_mol_cm3,
    )
    dx_segments = medium_profile.length_m / R_E
    return evolve_path(
        model, environment, dx_segments, method=PropagationMethod.ANALYTIC,
        perturbative_profile=medium_profile.perturbative_profile, evolution_scale_m=R_E, reverse=reverse,
    )

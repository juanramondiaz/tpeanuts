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

"""The tpeanuts evaluation of the quantities of ``validation.common.reference``: operators, weights, models, media.

Each function here computes with tpeanuts what the function of the same purpose in ``validation.common.reference``
computes in NumPy, with the same arguments and the same conventions (``P[initial, final]`` for probabilities,
``M[final, mass]`` for ``mass_to_flavour``, densities in mol/cm^3, energies in GeV or MeV as the name says), so that a
comparison only has to change the module. It returns NumPy arrays, so what follows is independent of torch. These are
the helpers that the notebooks of ``notebooks/3_validation/2_external`` repeated, and they serve any external package.

Module contents:
    mixing_and_mass_spectrum(...)
        Three-flavour ``MixingMatrix`` and ``MassSpectrum`` from a configuration object.
    mixing_from_parameters(...)
        3+1 ``MixingMatrix`` of tpeanuts from its nine parameters.
    propagation_model(...)
        ``PropagationModel`` with or without the sterile neutral-current term.
    segment_operator(...)
        Evolution operator of segments of constant density with ``evolve_path``.
    vacuum_operator(...)
        Evolution operator over a vacuum baseline.
    earth_operator(...), atmosphere_operator(...)
        Operators of the Earth chord and of the atmosphere leg of tpeanuts, with its own media.
    earth_shells(...), earth_operator_shells(...)
        The Earth of tpeanuts with the polynomial shells given by the caller, as a context and as one operator.
    probabilities(...), mass_to_flavour(...)
        Probabilities of an operator and from each vacuum mass eigenstate.
    solar_medium(...), solar_mass_weights(...)
        A ``SolarMedium`` from a density table and the adiabatic mass weights at a production radius.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator, Optional, Sequence, Tuple
from unittest import mock

import numpy as np
import torch

from tpeanuts.medium.atmosphere.atmosphere import AtmosphereMedium
from tpeanuts.medium.earth.earth import EarthMedium
from tpeanuts.medium.earth.even_power import EvenPowerShellTable
from tpeanuts.medium.solar.solar import SolarMedium
from tpeanuts.physics.common.mass_spectrum import MassSpectrum
from tpeanuts.physics.common.mixing import MixingMatrix, StandardMixingParameters, SterileMixingParameters
from tpeanuts.physics.common.nature import Dirac
from tpeanuts.physics.common.space import FlavourSpace
from tpeanuts.physics.common.terms import PropagationEnvironment
from tpeanuts.physics.evolution import PropagationMethod
from tpeanuts.physics.evolution.adiabatic import adiabatic
from tpeanuts.physics.evolution.dispatch import evolve_path
from tpeanuts.physics.extensions.sterile import SterileNCTerm
from tpeanuts.physics.model import PropagationModel
from tpeanuts.util.constant import R_E
from tpeanuts.util.context import RuntimeContext


def _real(context: RuntimeContext) -> dict:
    return dict(device=context.device, dtype=context.dtype)


def mixing_and_mass_spectrum(config: object, context: RuntimeContext) -> Tuple[MixingMatrix, MassSpectrum]:
    """Plain three-flavour Dirac ``MixingMatrix`` and ``MassSpectrum`` from the fields of a configuration.

    Args:
        config: Any object with ``theta12``, ``theta13``, ``theta23``, ``delta_cp``, ``DeltamSq21`` and
            ``DeltamSq3l`` (the configurations of the nuSQuIDS and PEANUTS series).
        context: RuntimeContext of the returned objects.

    Returns:
        ``(mixing, mass_spectrum)`` on ``FlavourSpace()``, without sterile states.
    """
    space = FlavourSpace()
    mixing = MixingMatrix(
        space=space,
        standard=StandardMixingParameters(
            theta12=config.theta12,
            theta13=config.theta13,
            theta23=config.theta23,
            delta13=config.delta_cp,
            context=context,
        ),
        nature=Dirac(),
        context=context,
    )
    spectrum = MassSpectrum(space=space, DeltamSq21=config.DeltamSq21, DeltamSq3l=config.DeltamSq3l, context=context)
    return mixing, spectrum


def mixing_from_parameters(parameters: Sequence[float], context: RuntimeContext) -> MixingMatrix:
    """3+1 ``MixingMatrix`` of tpeanuts from its nine parameters.

    The parameters are ``(theta12, theta13, theta23, delta13, theta14, theta24, theta34, delta14, delta24)``.

    The fourth phase of the muon-tau-sterile rotation, ``delta_tau``, is zero, as in nuSQuIDS.

    Args:
        parameters: The nine values, in the parametrisation of tpeanuts.
        context: RuntimeContext of the returned object.

    Returns:
        Dirac ``MixingMatrix`` on ``FlavourSpace(n_sterile=1)``.
    """
    t12, t13, t23, d13, t14, t24, t34, d14, d24 = (float(x) for x in parameters)
    return MixingMatrix(
        space=FlavourSpace(n_sterile=1),
        standard=StandardMixingParameters(theta12=t12, theta13=t13, theta23=t23, delta13=d13, context=context),
        nature=Dirac(),
        context=context,
        sterile=SterileMixingParameters(
            theta_e=[t14],
            theta_mu=[t24],
            theta_tau=[t34],
            delta_e=[d14],
            delta_mu=[d24],
            delta_tau=[0.0],
            context=context,
        ),
    )


def propagation_model(
    mixing: MixingMatrix,
    spectrum: MassSpectrum,
    *,
    antinu: bool = False,
    neutral_current: bool = False,
) -> PropagationModel:
    """``PropagationModel`` of a mixing and a spectrum, with the sterile neutral-current term if asked.

    Args:
        mixing: Mixing matrix.
        spectrum: Mass spectrum.
        antinu: If True, the antineutrino.
        neutral_current: If True, add ``SterileNCTerm`` (only meaningful with a sterile neutrino).

    Returns:
        The model.
    """
    return PropagationModel(
        mixing=mixing, mass_spectrum=spectrum, antinu=antinu, extensions=(SterileNCTerm(),) if neutral_current else ()
    )


def segment_operator(
    model: PropagationModel,
    energy_gev: float,
    lengths_m: np.ndarray,
    electron_density: np.ndarray,
    neutron_density: Optional[np.ndarray] = None,
    *,
    context: RuntimeContext,
    method: PropagationMethod = PropagationMethod.NUMERICAL,
) -> np.ndarray:
    """Evolution operator of consecutive segments of constant density, the first crossed first, with ``evolve_path``.

    Args:
        model: Propagation model.
        energy_gev: Neutrino energy in GeV.
        lengths_m: Segment lengths in m.
        electron_density: Electron density of each segment in mol/cm^3.
        neutron_density: Neutron density of each segment, zero if None.
        context: RuntimeContext.
        method: Propagation method of ``evolve_path``; the numerical one is the reference for a given profile.

    Returns:
        Complex array shaped ``(n, n)``, ``S[final, initial]``.
    """
    real = _real(context)
    n_n = np.zeros(len(electron_density)) if neutron_density is None else neutron_density
    environment = PropagationEnvironment(
        E_MeV=torch.tensor(float(energy_gev) * 1.0e3, **real),
        n_e_mol_cm3=torch.as_tensor(electron_density, **real),
        n_n_mol_cm3=torch.as_tensor(n_n, **real),
    )
    operator = evolve_path(
        model, environment, torch.as_tensor(lengths_m / R_E, **real), method=method, evolution_scale_m=R_E
    )
    return operator.detach().resolve_conj().cpu().numpy()


def vacuum_operator(
    model: PropagationModel, energy_mev: float, baseline_m: float, *, context: RuntimeContext
) -> np.ndarray:
    """Evolution operator over a vacuum baseline, ``S[final, initial]``.

    Args:
        model: Propagation model.
        energy_mev: Neutrino energy in MeV.
        baseline_m: Baseline in m.
        context: RuntimeContext.

    Returns:
        Complex array shaped ``(n, n)``.
    """
    real = _real(context)
    environment = PropagationEnvironment(
        E_MeV=torch.tensor(float(energy_mev), **real),
        n_e_mol_cm3=torch.zeros(1, **real),
        n_n_mol_cm3=torch.zeros(1, **real),
    )
    operator = evolve_path(
        model,
        environment,
        torch.tensor([baseline_m / R_E], **real),
        method=PropagationMethod.ANALYTIC,
        evolution_scale_m=R_E,
    )
    return operator.detach().resolve_conj().cpu().numpy()


def earth_operator(
    model: PropagationModel,
    energy_gev: float,
    cos_zenith: float,
    *,
    nsteps: int,
    context: RuntimeContext,
    method: PropagationMethod = PropagationMethod.ANALYTIC,
) -> np.ndarray:
    """Operator of the chord of the Earth of tpeanuts (``EarthMedium``, PREM table) for a surface detector.

    Args:
        model: Propagation model.
        energy_gev: Neutrino energy in GeV.
        cos_zenith: cosine of the zenith angle, negative.
        nsteps: Number of equal segments of the chord.
        context: RuntimeContext.
        method: Propagation method of the medium.

    Returns:
        Complex array shaped ``(n, n)``.
    """
    medium = EarthMedium(eta_rad=math.acos(-cos_zenith), context=context, method=method, nsteps=nsteps)
    energy = torch.tensor(float(energy_gev) * 1.0e3, **_real(context))
    return medium.evolutor(model, energy).detach().resolve_conj().cpu().numpy()


@contextmanager
def earth_shells(
    shells: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], context: RuntimeContext
) -> Iterator[None]:
    """Inside the block, every ``EarthMedium`` of tpeanuts reads the polynomial shells of the caller.

    ``EarthMedium`` reads its shells with ``load_even_power_shells``; while the block lasts that reader returns the
    shells given, with a zero neutron density, so the perturbative evolutor runs on exactly the profile of another
    code, also through the public pipeline (``EarthLeg``, ``solar_flux``). Nothing is changed in the package, and the
    reader is restored on leaving the block.

    Args:
        shells: ``(radii, alpha, beta, gamma)``, ``n_e = alpha + beta r^2 + gamma r^4`` up to each outer radius.
        context: RuntimeContext of the tensors of the table.

    Yields:
        None.
    """
    real = _real(context)
    radii, alpha, beta, gamma = (torch.as_tensor(np.asarray(item, dtype=float), **real) for item in shells)
    zeros = torch.zeros_like(alpha)
    table = EvenPowerShellTable(
        rj=radii, alpha=alpha, beta=beta, gamma=gamma, alpha_n=zeros, beta_n=zeros.clone(), gamma_n=zeros.clone()
    )
    with mock.patch("tpeanuts.medium.earth.earth.load_even_power_shells", lambda _context: table):
        yield


def earth_operator_shells(
    model: PropagationModel,
    energy_gev: float,
    cos_zenith: float,
    shells: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    context: RuntimeContext,
    nsteps: int = 200,
    method: PropagationMethod = PropagationMethod.ANALYTIC_PERTURBATIVE,
) -> np.ndarray:
    """Operator of the Earth chord of tpeanuts with the polynomial shells of the caller instead of its own fit.

    Args:
        model: Propagation model.
        energy_gev: Neutrino energy in GeV.
        cos_zenith: cosine of the zenith angle, negative.
        shells: ``(radii, alpha, beta, gamma)``, ``n_e = alpha + beta r^2 + gamma r^4`` up to each outer radius.
        context: RuntimeContext.
        nsteps: Number of segments, only used by the methods that sample the chord.
        method: Propagation method of the medium.

    Returns:
        Complex array shaped ``(n, n)``.
    """
    with earth_shells(shells, context):
        return earth_operator(model, energy_gev, cos_zenith, nsteps=nsteps, context=context, method=method)


def atmosphere_operator(
    model: PropagationModel,
    energy_gev: float,
    cos_zenith: float,
    height_m: float,
    *,
    nsteps: int,
    context: RuntimeContext,
    density_kwargs: Optional[dict] = None,
    method: PropagationMethod = PropagationMethod.ANALYTIC,
) -> np.ndarray:
    """Operator of the atmosphere leg of tpeanuts, from the production altitude to the surface.

    Args:
        model: Propagation model.
        energy_gev: Neutrino energy in GeV.
        cos_zenith: cosine of the zenith angle.
        height_m: Production altitude in m.
        nsteps: Number of segments.
        context: RuntimeContext.
        density_kwargs: Parameters of the exponential atmosphere (``rho0_gcm3``, ``scale_height_m``, ``Ye``).
        method: Propagation method of the medium.

    Returns:
        Complex array shaped ``(n, n)``.
    """
    medium = AtmosphereMedium(
        eta_rad=math.acos(-cos_zenith),
        h_m=float(height_m),
        context=context,
        method=method,
        nsteps=nsteps,
        density_source="exponential",
        density_kwargs=density_kwargs,
    )
    energy = torch.tensor(float(energy_gev) * 1.0e3, **_real(context))
    return medium.evolutor(model, energy).detach().resolve_conj().cpu().numpy()


def probabilities(operator: np.ndarray) -> np.ndarray:
    """Probabilities of an operator, ``P[initial, final]``."""
    return (np.abs(operator) ** 2).T


def mass_to_flavour(model: PropagationModel, operator: np.ndarray) -> np.ndarray:
    """Probability that each vacuum mass eigenstate ends in each flavour, ``M[final, mass]`` ``= |S U|^2``.

    Args:
        model: Propagation model; its mixing, conjugated for the antineutrino, gives the vacuum mass eigenstates.
        operator: Evolution operator ``S[final, initial]``.

    Returns:
        Real array shaped ``(n, n)``.
    """
    U = model.mixing.pmns_matrix(antinu=model.antinu).detach().resolve_conj().cpu().numpy()
    return np.abs(operator @ U) ** 2


def solar_medium(
    radius: np.ndarray,
    electron_density: np.ndarray,
    neutron_density: Optional[np.ndarray] = None,
    *,
    context: RuntimeContext,
) -> SolarMedium:
    """``SolarMedium`` from a table of the density of the Sun.

    Args:
        radius: Radii in units of the solar radius, increasing from 0 to 1.
        electron_density: Electron density in mol/cm^3.
        neutron_density: Neutron density in mol/cm^3, or None.
        context: RuntimeContext.

    Returns:
        The medium.
    """
    real = _real(context)
    return SolarMedium(
        radius=torch.as_tensor(radius, **real),
        density=torch.as_tensor(electron_density, **real),
        density_n=None if neutron_density is None else torch.as_tensor(neutron_density, **real),
    )


def solar_mass_weights(
    medium: SolarMedium,
    model: PropagationModel,
    energy_mev: float,
    r0: float,
    *,
    context: RuntimeContext,
    method: str = "adiabatic_approximated",
) -> np.ndarray:
    """Weights of the vacuum mass eigenstates at the solar surface of a ``nu_e`` produced at ``r0``.

    With a sterile neutrino the exact adiabatic weights of the full Hamiltonian are used, since the approximated
    closed form is for three flavours; otherwise ``SolarMedium.mass_weights`` with ``method``.

    Args:
        medium: The Sun.
        model: Propagation model.
        energy_mev: Neutrino energy in MeV.
        r0: Production radius in units of the solar radius.
        context: RuntimeContext.
        method: Method of ``SolarMedium.mass_weights`` for three flavours.

    Returns:
        Weights shaped ``(n,)`` in the order of the vacuum masses.
    """
    real = _real(context)
    if model.mixing.space.n_sterile > 0:
        environment = PropagationEnvironment(
            E_MeV=torch.tensor(float(energy_mev), **real),
            n_e_mol_cm3=medium.electron_density(torch.tensor(float(r0), **real)),
            n_n_mol_cm3=medium.neutron_density(torch.tensor(float(r0), **real)),
        )
        return adiabatic.mass_weights_adiabatic(model, environment, variant="exact").detach().cpu().numpy().reshape(-1)
    weights = medium.mass_weights(
        model, torch.tensor(float(energy_mev), **real), torch.tensor([float(r0)], **real), method=method
    )
    return weights.detach().cpu().numpy().reshape(-1)[:3]

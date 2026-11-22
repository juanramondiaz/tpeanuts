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

"""Daya Bay reactor antineutrino flux: Huber-Mueller spectrum, normalization and corrections.

The per-isotope spectrum interpolation and fission-fraction-weighted
combination, the non-equilibrium correction and the fission rate are the
reactor source's own (``tpeanuts.source.reactor.reactor.ReactorSource``,
built here from the loaded parameters by ``reactor_source``); this module adds
what is specific to Daya Bay's own 6 cores -- the per-core relative weights
and the spent-fuel correction -- on top of the physics::

    S_i^corrected(E) = S_i(E) * (1 + C_i^neq(E))     i in {U235, Pu239, Pu241}
    Phi_face,r(E) = (P_th / <e_fission>) * w_r * (1 + C_r^snf(E))
                    * sum_i f_i S_i^corrected(E)                    [nu/s/MeV]
    Phi_detector,r(E, L) = Phi_face,r(E) / (4 pi L^2)                [nu/s/MeV/cm^2]

with ``<e_fission> = sum_i f_i * e_fission,i`` the fission-fraction-weighted
mean energy per fission, ``C_i^neq`` the per-isotope non-equilibrium
correction (long-lived fission-daughter beta decays not yet in secular
equilibrium; not published for U238), ``C_r^snf`` the per-reactor spent
-nuclear-fuel correction, and ``w_r`` the per-reactor relative power/
output weight for the 8AD period
(``tpeanuts.experiments.dayabay.parameters.DayaBayParameters.reactor_relative_weight``).

The per-core factor ``w_r * (1 + C_r^snf(E))`` is the source's own
(``ReactorSource.core_scaling``, also reachable as
``flux(relative_weight=..., snf_correction=...)``); this module computes the
emission once and rescales it per core with each core's weight and spent-fuel
table from the loaded parameters. ``reactor_flux_at_face`` and ``flux_at_detector`` accept **one or
several** reactor names and stack the per-core fluxes on a leading reactor axis.

Module contents:
    reactor_source(...)
        The Daya Bay reactor as a ``ReactorSource``, from the loaded parameters.
    reactor_spectrum_shape(...)
        Fission-fraction-weighted sum over the 4 isotopes, with the
        non-equilibrium correction folded into U235/Pu239/Pu241.
    mean_energy_per_fission_MeV(...)
        Fission-fraction-weighted mean energy released per fission.
    reactor_flux_at_face(...)
        Absolute flux at one or several reactor cores, before geometric
        dilution.
    reactor_fluxes(...)
        The same, as a ``SourceFlux``: one antineutrino ``Flux`` per core, the
        start of the ``source -> VacuumMedium.propagate -> detector.observe`` chain.
    flux_at_detector(...)
        Phi_detector,r(E, L), see above.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence, Union

import torch

from tpeanuts.experiments.dayabay.parameters import REACTORS, DayaBayParameters
from tpeanuts.interfaces.flux import SourceFlux
from tpeanuts.source.reactor.reactor import COMPONENT, ReactorSource

Tensor = torch.Tensor


def reactor_source(params: DayaBayParameters, E_grid_MeV: Tensor) -> ReactorSource:
    """The Daya Bay reactor as a ``ReactorSource``, built from the loaded ``params``.

    The tables are the ones ``ReactorSource.default`` packages (Huber-Mueller curves, time-averaged fission
    fractions, energy per fission, thermal power, non-equilibrium correction) as ``params`` holds them, so the
    spectrum, the non-equilibrium correction and the fission rate are the source's own; this module adds only what is
    specific to the six cores.

    Args:
        params: Loaded Daya Bay parameters.
        E_grid_MeV: True antineutrino energy grid, shape ``(n_E,)``.

    Returns:
        ReactorSource on ``params.context`` with ``E_grid_MeV`` as its grid.
    """
    return ReactorSource(
        energy_MeV=E_grid_MeV, context=params.context, curves=params.huber_mueller_curves,
        fission_fractions=params.fission_fractions, energy_per_fission_MeV=params.energy_per_fission_MeV,
        thermal_power_gw=params.thermal_power_gw, nonequilibrium=params.nonequilibrium_correction,
        reactor_provider="dayabay",
    )


def reactor_spectrum_shape(params: DayaBayParameters, E_grid_MeV: Tensor) -> Tensor:
    """Fission-fraction-weighted sum over the 4 isotopes, antineutrinos/fission/MeV.

    Includes the non-equilibrium correction (see
    ``ReactorSource.corrected_curves``); shared by all 6 reactor cores
    (the correction is per-isotope, not per-reactor).

    Args:
        params: Loaded Daya Bay parameters.
        E_grid_MeV: True antineutrino energy grid, shape ``(n_E,)``.

    Returns:
        Tensor shaped ``(n_E,)``.
    """
    return reactor_source(params, E_grid_MeV).spectrum_shape()


def mean_energy_per_fission_MeV(params: DayaBayParameters) -> float:
    """Fission-fraction-weighted mean energy released per fission, MeV."""
    return reactor_source(params, params.e_nu_grid_MeV).mean_energy_per_fission_MeV


def reactor_flux_at_face(
    params: DayaBayParameters, E_grid_MeV: Tensor, reactors: Union[str, Sequence[str]],
) -> Tensor:
    """Absolute antineutrino flux at one or several reactor cores, before geometric dilution.

    ``Phi_face,r(E) = (P_th / <e_fission>) * w_r * (1 + C_r^snf(E)) *
    reactor_spectrum_shape(E)``, nu/s/MeV -- see module docstring for
    ``w_r``/``C_r^snf``; the differential flux of ``reactor_fluxes`` stacked.

    Args:
        params: Loaded Daya Bay parameters.
        E_grid_MeV: True antineutrino energy grid, shape ``(n_E,)``.
        reactors: A single reactor name (e.g. ``"R1"``), or a sequence of
            them.

    Returns:
        Tensor shaped ``(n_E,)`` for a single reactor name, or
        ``(n_reactor, n_E)`` for a sequence, nu/s/MeV.
    """
    single = isinstance(reactors, str)
    reactor_list = [reactors] if isinstance(reactors, str) else list(reactors)
    emitted = reactor_fluxes(params, E_grid_MeV, reactor_list)
    result = torch.stack([emitted[reactor].differential_flux for reactor in reactor_list])  # (n_reactor, n_E)
    return result[0] if single else result


def flux_at_detector(
    params: DayaBayParameters, E_grid_MeV: Tensor, baseline_km: Tensor, reactors: Union[str, Sequence[str]],
) -> Tensor:
    """``Phi_detector,r(E, L) = Phi_face,r(E) / (4 pi L^2)``, nu/s/MeV/cm^2.

    Args:
        params: Loaded Daya Bay parameters.
        E_grid_MeV: True antineutrino energy grid, shape ``(n_E,)``.
        baseline_km: Reactor-to-detector distance(s), km -- a scalar for
            a single reactor name, or shape ``(n_reactor,)`` matching
            ``reactors``.
        reactors: A single reactor name, or a sequence of them (see
            ``reactor_flux_at_face``).

    Returns:
        Tensor shaped ``(n_E,)`` for a single reactor, or ``(n_reactor,
        n_E)`` for a sequence.
    """
    L_cm = baseline_km * 1.0e5
    flux_face = reactor_flux_at_face(params, E_grid_MeV, reactors)
    if isinstance(reactors, str):
        return flux_face / (4.0 * torch.pi * L_cm ** 2)
    return flux_face / (4.0 * torch.pi * L_cm[:, None] ** 2)


def reactor_fluxes(
    params: DayaBayParameters, E_grid_MeV: Tensor, reactors: Sequence[str] = REACTORS,
) -> SourceFlux:
    """The antineutrino flux emitted by each reactor core, as a ``SourceFlux`` with one component per core.

    Each component is a pure electron-antineutrino ``Flux`` whose ``differential_flux`` is
    ``reactor_flux_at_face`` for that core, in ``antinu s^-1 MeV^-1``: the spectrum of
    ``tpeanuts.source.reactor.reactor.ReactorSource.flux`` (corrected Huber-Mueller curves, time-averaged fission
    fractions, fission rate from the thermal power) times the core's relative weight and spent-fuel correction.
    Propagating it with ``VacuumMedium(...).propagate(flux, model, dilute=True)`` spreads it over the baseline.

    Args:
        params: Loaded Daya Bay parameters.
        E_grid_MeV: True antineutrino energy grid, shape ``(n_E,)``.
        reactors: Names of the cores, in the order of the returned components (default: all six).

    Returns:
        SourceFlux keyed by reactor name.
    """
    source = reactor_source(params, E_grid_MeV)
    emitted = source.flux()[COMPONENT]
    return SourceFlux({
        reactor: dataclasses.replace(
            emitted,
            differential_flux=source.core_scaling(
                E_grid_MeV, params.reactor_relative_weight[reactor], params.snf_correction[reactor],
            ) * emitted.differential_flux,
        )
        for reactor in reactors
    })

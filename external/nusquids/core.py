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
Small, pure-nuSQuIDS reference helpers.

The functions in this module do not call tpeanuts propagation code. They only
adapt the optional nuSQuIDS Python bindings to a compact API for initializing
a three-flavour solver and evaluating flavour probabilities in vacuum, through
Earth-like bodies, and through the nuSQuIDS EarthAtm atmosphere model.

Every public function here is a wrapper that, when called, executes the
external nuSQuIDS C++ solver via its Python bindings (not a tpeanuts/torch
computation): mixing angles and the CP phase are set through nuSQuIDS's own
``Set_MixingAngle``/CP-phase setters, the neutrino state is evolved by
nuSQuIDS's ``EvolveState``, and final flavour probabilities are read back
with nuSQuIDS's ``EvalFlavor``. This backend exists purely as an independent
cross-check of tpeanuts's own (torch-based) oscillation-in-matter code.

Module functions:
    NuSQuIDSConfig
        Dataclass with the oscillation parameters (mixing angles, CP phase,
        mass splittings) and numerical tolerances passed to nuSQuIDS.
    NuSQuIDSError
        Raised when nuSQuIDS is unavailable or cannot run a requested setup.
    require_nusquids(...)
        Import and return the installed nuSQuIDS Python module.
    is_available(...)
        Check whether the nuSQuIDS Python bindings can be imported.
    units(...)
        Return the nuSQuIDS unit-conversion object.
    normalise_flavour_label(...), initial_state(...), neutrino_type(...)
        Public helpers for flavour labels, initial states, and the
        neutrino/antineutrino enum.
    set_cp_phase(...), eval_probabilities(...), evolve_with_body(...)
        Shared solver utilities used by the probability wrappers.
    configure_solver(...)
        Apply oscillation and numerical settings to an existing nuSQuIDS
        solver object.
    init_solver(...)
        Create and configure a single-energy three-flavour nuSQuIDS solver.
    probability_vacuum(...)
        Return final-flavour probabilities after vacuum propagation over a
        fixed baseline.
    probability_earth(...)
        Return final-flavour probabilities through the nuSQuIDS Earth body
        for a given cos(zenith).
    transition_matrix_earth_mass_to_flavour(...)
        Return the nuSQuIDS Earth probability matrix from an incoherent
        vacuum-mass input basis to the final flavour basis.
    probability_earth_massbasis(...)
        Propagate an incoherent vacuum-mass mixture through the nuSQuIDS
        Earth body and return the final flavour probabilities.
    probability_atmosphere(...)
        Return final-flavour probabilities through nuSQuIDS's EarthAtm
        body (atmosphere production height plus Earth) for a given
        cos(zenith).
    probability_solar_point(...)
        Return decohered solar probabilities from a fixed SunASnu production
        radius.
    sun_asnu_track_fraction(...)
        Convert a physical solar radius r/R_sun into the SunASnu track
        coordinate fraction used by the Python body interface.
    probability_grid_vacuum(...)
        Return a table of vacuum probabilities over an energy/baseline grid.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from typing import Iterable, Optional, Sequence, Union
import warnings

import numpy as np
import torch

from tpeanuts.core.common.neutrino import flavour_index
from tpeanuts.core.common.pmns import PMNSParams
from tpeanuts.core.SM.pmns import PMNS_SM
from tpeanuts.util.context import RuntimeContext
import tpeanuts.util.default as default


class NuSQuIDSError(RuntimeError):
    """Raised when nuSQuIDS is unavailable or cannot run a requested setup."""


@dataclass(frozen=True)
class NuSQuIDSConfig:
    """
    Numerical and oscillation configuration for nuSQuIDS.

    Angles are in radians, mass splittings are in eV^2, energies passed to the
    public probability helpers are in GeV, and distances are in km. These
    values are forwarded to nuSQuIDS's own setter methods (e.g.
    ``Set_MixingAngle``, ``Set_SquareMassDifference``) in configure_solver;
    they do not affect tpeanuts's own torch oscillation code, which keeps
    its own independent oscillation-parameter configuration.

    Attributes:
        theta12: Solar mixing angle theta_12 in radians.
        theta13: Reactor mixing angle theta_13 in radians.
        theta23: Atmospheric mixing angle theta_23 in radians.
        delta_cp: Dirac CP-violating phase in radians.
        DeltamSq21: Solar mass-squared splitting Delta m^2_21 in eV^2.
        DeltamSq3l: Atmospheric mass-squared splitting Delta m^2_3l in eV^2
            (l=1 for normal ordering, l=2 for inverted, per nuSQuIDS
            convention).
        rel_error: Relative error tolerance for nuSQuIDS's internal ODE
            integrator (Set_rel_error), if supported by the installed
            bindings.
        abs_error: Absolute error tolerance for nuSQuIDS's internal ODE
            integrator (Set_abs_error), if supported.
        h_max_km: Optional maximum internal integrator step size in km
            (converted to nuSQuIDS's natural units and passed to
            Set_h_max). None disables this setting.
        set_default_mixing_first: If True, call nuSQuIDS's
            Set_MixingParametersToDefault before applying the angles/
            splittings above, when the installed bindings expose it.
        set_cp_phase: If True, attempt to set delta_cp on the solver using
            whichever CP-phase setter the installed nuSQuIDS bindings
            expose.
        strict_cp: If True, raise NuSQuIDSError when no compatible CP-phase
            setter is found; if False, only warn and continue with the
            bindings' default CP phase.
        nusquids_rho0_gcm3: Zero-altitude reference mass density in g/cm^3
            for the tpeanuts reimplementation of nuSQuIDS's EarthAtm
            exponential atmosphere formula (see
            external.nusquids.density.atmosphere_density_nusquids).
        nusquids_scale_height_km: Exponential scale height in km for the
            same atmosphere mass-density formula.
        nusquids_Ye: Electron fraction (dimensionless, electrons per
            nucleon) used to convert that mass density into electron
            density.
    """

    theta12: float = 0.59
    theta13: float = 0.15
    theta23: float = 0.78
    delta_cp: float = 1.20
    DeltamSq21: float = 7.42e-5
    DeltamSq3l: float = 2.517e-3
    rel_error: float = 1.0e-11
    abs_error: float = 1.0e-13
    h_max_km: Optional[float] = 100.0
    set_default_mixing_first: bool = False
    set_cp_phase: bool = True
    strict_cp: bool = False
    nusquids_rho0_gcm3: float = default.atmosphere_nusquids_rho0_gcm3
    nusquids_scale_height_km: float = default.atmosphere_nusquids_scale_height_km
    nusquids_Ye: float = default.atmosphere_nusquids_Ye


def require_nusquids():
    """
    Import and return the nuSQuIDS Python module.

    Several builds expose different import names, so the resolver tries the
    common spellings used by wheel and source installations.

    Returns:
        The imported nuSQuIDS Python module object.

    Raises:
        NuSQuIDSError: If none of the known module names ("nuSQuIDS",
            "nuSQUIDSpy", "nusquids") can be imported in this environment.
            The message includes installation guidance and the per-name
            import errors encountered.
    """
    errors = []
    for module_name in ("nuSQuIDS", "nuSQUIDSpy", "nusquids"):
        try:
            return importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")

    message = (
        "nuSQuIDS Python bindings are not installed or not importable.\n"
        "Install them in the active environment, for example:\n\n"
        "    pip install nusquids\n\n"
        "If using a source build, add the bindings directory to PYTHONPATH.\n"
        "Import attempts:\n  - "
        + "\n  - ".join(errors)
    )
    raise NuSQuIDSError(message)


def is_available() -> bool:
    """
    Check whether the nuSQuIDS Python bindings can be imported.

    Returns:
        True if require_nusquids() succeeds, False if it raises
        NuSQuIDSError.
    """
    try:
        require_nusquids()
    except NuSQuIDSError:
        return False
    return True


def units(nsq):
    """
    Return a unit-conversion object exposing GeV and km in nuSQuIDS's natural units.

    nuSQuIDS internally works in natural units; its ``Const()`` helper
    exposes the multiplicative factors needed to convert physical GeV and
    km into those units. If the installed bindings lack ``Const()``, a
    fallback object with hard-coded standard nuSQuIDS unit constants is
    used instead (with a warning), since some minimal/older builds omit it.

    Args:
        nsq: The imported nuSQuIDS module (from require_nusquids()).

    Returns:
        An object with float attributes ``GeV`` and ``km`` giving the
        natural-unit value of one GeV and one km respectively.
    """
    if hasattr(nsq, "Const"):
        return nsq.Const()

    class FallbackUnits:
        GeV = 1.0e9
        km = 5.067730716e18

    warnings.warn(
        "nuSQuIDS module has no Const() helper; using fallback conversion constants.",
        RuntimeWarning,
        stacklevel=2,
    )
    return FallbackUnits()


def normalise_flavour_label(flavour: Union[str, int]) -> Union[str, int]:
    """
    Normalize a flavour label to the bare flavour key used by flavour_index.

    Strips antineutrino/"total_" decorations (e.g. "antinumu", "numubar",
    "total_nue") down to the bare flavour name (e.g. "numu", "nue"), since
    the neutrino/antineutrino distinction is handled separately via the
    nuSQuIDS NeutrinoType, not via the flavour index itself.

    Args:
        flavour: Flavour label as a string (e.g. "numu", "antinue") or an
            already-integer flavour index, passed through unchanged.

    Returns:
        The normalized lowercase flavour string, or the original int.
    """
    if isinstance(flavour, int):
        return flavour

    key = str(flavour).lower()
    key = key.replace("total_", "")
    key = key.replace("anti", "")
    key = key.replace("bar", "")
    key = key.strip("_")
    return key


def initial_state(flavour: Union[str, int], n_flavours: int = 3) -> np.ndarray:
    """
    Build a pure flavour-eigenstate occupation vector for nuSQuIDS.

    Args:
        flavour: Initial neutrino flavour, as a string label (e.g. "numu")
            or integer flavour index (0=e, 1=mu, 2=tau in the standard
            tpeanuts/nuSQuIDS ordering).
        n_flavours: Number of flavours in the solver (3 for the standard
            three-flavour scenario used throughout this module).

    Returns:
        1D float array of length n_flavours, with a 1.0 at the initial
        flavour's index and 0.0 elsewhere, ready to pass to nuSQuIDS's
        Set_initial_state in the flavor basis.
    """
    state = np.zeros(n_flavours, dtype=float)
    state[flavour_index(normalise_flavour_label(flavour))] = 1.0
    return state


def neutrino_type(nsq, *, antinu: bool):
    """
    Return the nuSQuIDS NeutrinoType enum value for neutrino or antineutrino.

    Args:
        nsq: The imported nuSQuIDS module.
        antinu: If True, select antineutrino; if False, select neutrino.

    Returns:
        The corresponding ``nsq.NeutrinoType`` enum member.
    """
    return nsq.NeutrinoType.antineutrino if antinu else nsq.NeutrinoType.neutrino


def set_cp_phase(nuSQ, config: NuSQuIDSConfig) -> None:
    """
    Set the Dirac CP-violating phase on a nuSQuIDS solver object.

    Different nuSQuIDS builds expose different method names/signatures for
    setting the CP phase (e.g. indexed Set_CPPhase(i, j, delta) for the
    (1,3)/(2,4) mixing-matrix element pairs, or a flat Set_CPPhase(delta)/
    Set_DeltaCP(delta)/Set_CP_Phase(delta)). This tries each known
    candidate in turn and stops at the first one that succeeds.

    Args:
        nuSQ: An initialized nuSQuIDS solver object.
        config: NuSQuIDSConfig providing delta_cp (radians), whether to
            attempt setting it at all (set_cp_phase), and whether failure
            to find a compatible setter should raise (strict_cp) or only
            warn.

    Returns:
        None. Mutates nuSQ in place.

    Raises:
        NuSQuIDSError: If config.strict_cp is True and no compatible
            CP-phase setter is found on nuSQ.
    """
    if not config.set_cp_phase:
        return

    candidates = (
        ("Set_CPPhase", (0, 2, config.delta_cp)),
        ("Set_CPPhase", (1, 3, config.delta_cp)),
        ("Set_CPPhase", (config.delta_cp,)),
        ("Set_DeltaCP", (config.delta_cp,)),
        ("Set_CP_Phase", (config.delta_cp,)),
    )
    for method_name, args in candidates:
        method = getattr(nuSQ, method_name, None)
        if method is None:
            continue
        try:
            method(*args)
            return
        except TypeError:
            continue

    message = "Could not set delta_cp on this nuSQuIDS Python object."
    if config.strict_cp:
        raise NuSQuIDSError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def configure_solver(nuSQ, config: Optional[NuSQuIDSConfig] = None):
    """
    Apply oscillation and numerical settings to an existing nuSQuIDS solver.

    Sets the three mixing angles (theta12, theta13, theta23), the two
    mass-squared splittings, the CP phase, and (when supported by the
    installed bindings) the integrator's relative/absolute error
    tolerances and maximum step size, all via nuSQuIDS's own setter
    methods.

    Args:
        nuSQ: An initialized nuSQuIDS solver object (e.g. from
            ``nsq.nuSQUIDS(3, neutrino_type)``).
        config: Oscillation/numerical configuration. Defaults to
            ``NuSQuIDSConfig()`` if None.

    Returns:
        The same nuSQ object, configured in place, for convenient chaining.
    """
    if config is None:
        config = NuSQuIDSConfig()

    if config.set_default_mixing_first and hasattr(nuSQ, "Set_MixingParametersToDefault"):
        nuSQ.Set_MixingParametersToDefault()

    nuSQ.Set_MixingAngle(0, 1, float(config.theta12))
    nuSQ.Set_MixingAngle(0, 2, float(config.theta13))
    nuSQ.Set_MixingAngle(1, 2, float(config.theta23))
    nuSQ.Set_SquareMassDifference(1, float(config.DeltamSq21))
    nuSQ.Set_SquareMassDifference(2, float(config.DeltamSq3l))
    set_cp_phase(nuSQ, config)

    if hasattr(nuSQ, "Set_rel_error"):
        nuSQ.Set_rel_error(float(config.rel_error))
    if hasattr(nuSQ, "Set_abs_error"):
        nuSQ.Set_abs_error(float(config.abs_error))
    if config.h_max_km is not None and hasattr(nuSQ, "Set_h_max"):
        unit_constants = units(require_nusquids())
        nuSQ.Set_h_max(float(config.h_max_km) * unit_constants.km)

    return nuSQ


def init_solver(
    *,
    antinu: bool = False,
    config: Optional[NuSQuIDSConfig] = None,
):
    """
    Create and configure a single-energy three-flavour nuSQuIDS solver.

    Args:
        antinu: If True, create an antineutrino solver; if False, a
            neutrino solver.
        config: Oscillation/numerical configuration applied via
            configure_solver. Defaults to ``NuSQuIDSConfig()`` if None.

    Returns:
        A configured ``nsq.nuSQUIDS(3, ...)`` solver object, ready to have
        its body, track, energy, and initial state set before calling
        EvolveState.
    """
    nsq = require_nusquids()
    nuSQ = nsq.nuSQUIDS(3, neutrino_type(nsq, antinu=antinu))
    configure_solver(nuSQ, config=config)
    return nuSQ


def eval_probabilities(
    nuSQ,
    *,
    output_energy_GeV: Optional[float] = None,
    antinu_index: int = 0,
) -> np.ndarray:
    """
    Read final flavour probabilities off an already-evolved nuSQuIDS solver.

    Args:
        nuSQ: A nuSQuIDS solver object after EvolveState() has been called.
        output_energy_GeV: Optional output energy for bindings whose
            EvalFlavor overload expects an explicit energy argument.
        antinu_index: Optional neutrino/antineutrino index for bindings whose
            EvalFlavor overload expects a third argument.

    Returns:
        1D float array of length 3 with P(nu_e), P(nu_mu), P(nu_tau) (or
        the antineutrino equivalents) from nuSQuIDS's EvalFlavor.
    """
    if output_energy_GeV is None:
        return np.asarray([float(nuSQ.EvalFlavor(i)) for i in range(3)], dtype=float)

    nsq = require_nusquids()
    unit_constants = units(nsq)
    values = []
    for flavour in range(3):
        try:
            values.append(
                float(
                    nuSQ.EvalFlavor(
                        flavour,
                        float(output_energy_GeV) * unit_constants.GeV,
                        antinu_index,
                    )
                )
            )
        except TypeError:
            values.append(float(nuSQ.EvalFlavor(flavour, float(output_energy_GeV) * unit_constants.GeV)))
    return np.asarray(values, dtype=float)


def _pmns_from_config(config: Optional[NuSQuIDSConfig] = None) -> PMNS_SM:
    """Build a common PMNS object from a nuSQuIDS oscillation config."""
    cfg = NuSQuIDSConfig() if config is None else config
    context = RuntimeContext.resolve(None, torch.float64)
    params = PMNSParams(
        theta12=float(cfg.theta12),
        theta13=float(cfg.theta13),
        theta23=float(cfg.theta23),
        delta=float(cfg.delta_cp),
        context=context,
    )
    return PMNS_SM(params)


def sun_asnu_radius(nsq=None) -> float:
    """
    Return the nuSQuIDS SunASnu radial final coordinate for a radial track.

    Args:
        nsq: Optional imported nuSQuIDS module. If None, require_nusquids()
            is used.

    Returns:
        Solar radius in nuSQuIDS natural-coordinate units.
    """
    nsq = require_nusquids() if nsq is None else nsq
    if not hasattr(nsq, "SunASnu"):
        raise NuSQuIDSError("The installed nuSQuIDS bindings do not expose SunASnu.")
    return nsq.SunASnu.Track(0.0).GetFinalX()


def sun_asnu_track_fraction(r_solar: float) -> float:
    """
    Map a physical solar radius r/R_sun to the SunASnu track coordinate.

    For the radial SunASnu trajectory exposed by the Python bindings, the
    physical solar interior from centre to surface is sampled over the second
    half of the track coordinate: centre -> 0.5 and surface -> 1.0. Passing
    r/R_sun directly as the track fraction samples the wrong part of the
    trajectory and can make the apparent central density tend to zero.

    Args:
        r_solar: Physical solar radius in units of R_sun, with 0 at the solar
            centre and 1 at the solar surface.

    Returns:
        Normalized SunASnu track coordinate fraction.
    """
    return 0.5 + 0.5 * float(r_solar)


def make_solar_track(r0: float, *, impact: float = 0.0, nsq=None):
    """
    Build a radial SunASnu track for a solar production radius.

    Args:
        r0: Physical production radius in solar-radius units, r/R_sun.
        impact: Impact parameter in solar-radius units.
        nsq: Optional imported nuSQuIDS module.

    Returns:
        A nuSQuIDS SunASnu.Track object.
    """
    nsq = require_nusquids() if nsq is None else nsq
    radius = sun_asnu_radius(nsq)
    xini = sun_asnu_track_fraction(r0) * radius
    return nsq.SunASnu.Track(xini, float(impact) * radius)


def eval_mass_weights(nuSQ) -> Optional[np.ndarray]:
    """
    Read normalized mass-basis weights from an evolved nuSQuIDS solver.

    Args:
        nuSQ: A nuSQuIDS solver after EvolveState().

    Returns:
        Normalized length-3 mass-weight array, or None when the bindings do
        not expose EvalMass.
    """
    eval_mass = getattr(nuSQ, "EvalMass", None)
    if eval_mass is None:
        return None
    weights = np.asarray([float(eval_mass(i)) for i in range(3)], dtype=float)
    norm = weights.sum()
    if np.isfinite(norm) and norm > 0.0:
        weights = weights / norm
    return weights


def eval_flavour_averaged(nuSQ) -> Optional[np.ndarray]:
    """
    Read averaged final-flavour probabilities from bindings supporting it.

    This is a fallback for nuSQuIDS builds without EvalMass. It tries the
    averaged EvalFlavor overload used by some bindings.

    Args:
        nuSQ: A nuSQuIDS solver after EvolveState().

    Returns:
        Length-3 probability array, or None if the overload is unavailable.
    """
    values = []
    for flv in range(3):
        averaged = [False, False, False]
        try:
            values.append(float(nuSQ.EvalFlavor(flv, 0.0, averaged)))
        except TypeError:
            return None
    return np.asarray(values, dtype=float)


def evolve_with_body(
    *,
    body,
    track,
    E_GeV: float,
    initial_flavour: Union[str, int],
    antinu: bool,
    config: Optional[NuSQuIDSConfig],
) -> np.ndarray:
    """
    Configure a nuSQuIDS solver for one body/track/energy and evolve it.

    Shared implementation behind probability_vacuum, probability_earth, and
    probability_atmosphere: builds a fresh single-energy three-flavour
    solver, attaches the given nuSQuIDS body (the matter profile, e.g.
    Vacuum/Earth/EarthAtm) and track (the trajectory through that body,
    typically built from a baseline or cos(zenith)), sets the neutrino
    energy and a pure initial flavour eigenstate, evolves the state, and
    reads back the final flavour probabilities.

    Args:
        body: A nuSQuIDS body object (e.g. ``nsq.Vacuum()``,
            ``nsq.Earth()``, ``nsq.EarthAtm()``) describing the matter
            density profile to propagate through.
        track: A nuSQuIDS track object compatible with body, describing
            the specific trajectory (baseline or cos(zenith)) through it.
        E_GeV: Neutrino energy in GeV.
        initial_flavour: Initial flavour label or index (see
            initial_state).
        antinu: If True, propagate the antineutrino state.
        config: Oscillation/numerical configuration for the solver.
            Defaults to NuSQuIDSConfig() when None (via init_solver).

    Returns:
        1D float array of length 3 with the final flavour probabilities
        (see eval_probabilities).
    """
    nsq = require_nusquids()
    unit_constants = units(nsq)
    nuSQ = init_solver(antinu=antinu, config=config)

    nuSQ.Set_Body(body)
    nuSQ.Set_Track(track)
    nuSQ.Set_E(float(E_GeV) * unit_constants.GeV)
    nuSQ.Set_initial_state(initial_state(initial_flavour), nsq.Basis.flavor)
    nuSQ.EvolveState()

    return eval_probabilities(nuSQ)


def _build_track_from_cosine(body, cos_zenith: float):
    """Build an Earth/EarthAtm track from cos(zenith) across binding variants."""
    cosz = float(cos_zenith)
    phi = math.acos(max(-1.0, min(1.0, cosz)))

    make_with_cosine = getattr(body, "MakeTrackWithCosine", None)
    if callable(make_with_cosine):
        return make_with_cosine(cosz)

    make_track = getattr(body, "MakeTrack", None)
    if callable(make_track):
        # Older bindings may expose only MakeTrack(phi), with phi the zenith
        # angle in radians, while some wrappers may accept cos(phi) directly.
        for arg in (phi, cosz):
            try:
                return make_track(arg)
            except TypeError:
                continue

    track_cls = getattr(type(body), "Track", None) or getattr(body, "Track", None)
    if track_cls is not None:
        for method_name in ("makeWithCosine", "MakeWithCosine", "MakeTrackWithCosine"):
            method = getattr(track_cls, method_name, None)
            if callable(method):
                try:
                    return method(cosz)
                except TypeError:
                    continue

    raise NuSQuIDSError(
        "The nuSQuIDS Earth body cannot build a track from cos(zenith): "
        "tried MakeTrackWithCosine and MakeTrack(phi) binding variants."
    )


def _earth_baseline_km_from_cosine(earth, cos_zenith: float) -> float:
    """Convert detector cos(zenith) to an Earth-only chord baseline in km."""
    cosz = float(cos_zenith)
    radius_km = 6371.0
    get_radius = getattr(earth, "GetRadius", None)
    if callable(get_radius):
        try:
            radius_km = float(get_radius())
        except TypeError:
            radius_km = 6371.0
    return max(0.0, -2.0 * radius_km * cosz)


def _build_earth_track_from_cosine(nsq, earth, cos_zenith: float):
    """Build an Earth.Track baseline from cos(zenith), without an atmosphere leg."""
    track_cls = getattr(type(earth), "Track", None) or getattr(earth, "Track", None)
    if track_cls is None:
        raise NuSQuIDSError("The nuSQuIDS Earth binding does not expose Earth.Track.")

    unit_constants = units(nsq)
    baseline_km = _earth_baseline_km_from_cosine(earth, cos_zenith)
    return track_cls(baseline_km * unit_constants.km)


def _earth_body_and_track(nsq, cos_zenith: float):
    """Build a nuSQuIDS Earth-only body and a compatible detector track."""
    earth_factory = getattr(nsq, "Earth", None)
    if earth_factory is not None:
        earth = earth_factory()
        try:
            return earth, _build_earth_track_from_cosine(nsq, earth, cos_zenith)
        except NuSQuIDSError:
            pass
        except TypeError:
            pass

    earth_atm = nsq.EarthAtm()
    track = _build_track_from_cosine(earth_atm, cos_zenith)
    return earth_atm, track


def probability_vacuum(
    *,
    E_GeV: float,
    baseline_km: float,
    initial_flavour: Union[str, int] = "numu",
    antinu: bool = False,
    config: Optional[NuSQuIDSConfig] = None,
) -> np.ndarray:
    """
    Return final-flavour probabilities after vacuum propagation.

    Propagates a pure initial flavour eigenstate over a fixed straight-line
    baseline with no matter effects (nuSQuIDS's ``Vacuum`` body), i.e. pure
    vacuum oscillation. Useful as the simplest cross-check between
    nuSQuIDS and tpeanuts's own vacuum oscillation code.

    Args:
        E_GeV: Neutrino energy in GeV.
        baseline_km: Vacuum propagation distance in km.
        initial_flavour: Initial flavour label (e.g. "numu", "nue") or
            integer flavour index.
        antinu: If True, propagate the antineutrino state.
        config: Oscillation/numerical configuration. Defaults to
            NuSQuIDSConfig() when None.

    Returns:
        1D float array of length 3 with the final P(nu_e), P(nu_mu),
        P(nu_tau) (or antineutrino equivalents), summing to 1.
    """
    nsq = require_nusquids()
    unit_constants = units(nsq)
    body = nsq.Vacuum()
    track = nsq.Vacuum.Track(float(baseline_km) * unit_constants.km)

    return evolve_with_body(
        body=body,
        track=track,
        E_GeV=E_GeV,
        initial_flavour=initial_flavour,
        antinu=antinu,
        config=config,
    )


def probability_earth(
    *,
    E_GeV: float,
    cos_zenith: float,
    initial_flavour: Union[str, int] = "numu",
    antinu: bool = False,
    config: Optional[NuSQuIDSConfig] = None,
) -> np.ndarray:
    """
    Return final-flavour probabilities through the nuSQuIDS Earth body.

    The track is built from ``cos_zenith``. When the installed bindings expose
    an ``Earth`` body, the zenith direction is converted to an Earth-only
    chord baseline ``L = max(0, -2 R_earth cos_zenith)`` and passed to
    ``Earth.Track``. Older or reduced bindings that cannot construct
    ``Earth.Track`` fall back to ``EarthAtm``. The preferred ``Earth`` path
    assumes the neutrino is already at the Earth surface, so it does not add
    an atmosphere production-height leg.

    Args:
        E_GeV: Neutrino energy in GeV.
        cos_zenith: cos(zenith angle) at the detector. cos_zenith=1 is a
            neutrino arriving straight down (minimal Earth crossing);
            cos_zenith=-1 is straight up through the Earth's diameter.
        initial_flavour: Initial flavour label or integer flavour index.
        antinu: If True, propagate the antineutrino state.
        config: Oscillation/numerical configuration. Defaults to
            NuSQuIDSConfig() when None.

    Returns:
        1D float array of length 3 with the final flavour probabilities,
        summing to 1.

    Raises:
        NuSQuIDSError: If neither an "Earth" nor an "EarthAtm" body can
            build a compatible track from the provided cos(zenith).
    """
    nsq = require_nusquids()
    earth, track = _earth_body_and_track(nsq, cos_zenith)

    return evolve_with_body(
        body=earth,
        track=track,
        E_GeV=E_GeV,
        initial_flavour=initial_flavour,
        antinu=antinu,
        config=config,
    )


def transition_matrix_earth_mass_to_flavour(
    *,
    E_GeV: float,
    cos_zenith: float,
    antinu: bool = False,
    config: Optional[NuSQuIDSConfig] = None,
) -> np.ndarray:
    """
    Return the nuSQuIDS Earth probability matrix from mass to flavour.

    This helper is the Earth-side analogue of the decohered solar projection
    used in probability_solar_point: each pure vacuum mass eigenstate
    ``|nu_i>`` is propagated independently through the nuSQuIDS Earth body
    and the final flavour probabilities are assembled into a matrix

    ``T_{alpha i} = P(nu_i -> nu_alpha ; E, cos_zenith)``.

    This is the correct object for solar-neutrino Earth regeneration after
    Sun-Earth decoherence, where the state arriving at the Earth is an
    incoherent mass-basis mixture.

    Args:
        E_GeV: Neutrino energy in GeV.
        cos_zenith: cos(zenith angle) at the detector. ``cos_zenith=1`` is
            straight down (minimal Earth crossing) and ``cos_zenith=-1`` is
            straight up through the Earth's diameter.
        antinu: If True, propagate antineutrinos.
        config: Oscillation/numerical configuration. Defaults to
            ``NuSQuIDSConfig()`` when None.

    Returns:
        Array shaped ``(3, 3)`` with flavour rows and initial-mass columns.

    Raises:
        NuSQuIDSError: If the installed bindings do not expose
            ``nsq.Basis.mass``.
    """
    nsq = require_nusquids()
    unit_constants = units(nsq)
    basis_mass = getattr(getattr(nsq, "Basis", None), "mass", None)
    if basis_mass is None:
        raise NuSQuIDSError(
            "This nuSQuIDS binding does not expose Basis.mass, so Earth "
            "propagation from incoherent solar mass weights is unavailable."
        )

    matrix = np.zeros((3, 3), dtype=float)
    for mass_idx in range(3):
        nuSQ = init_solver(antinu=antinu, config=config)
        earth, track = _earth_body_and_track(nsq, cos_zenith)
        nuSQ.Set_Body(earth)
        nuSQ.Set_Track(track)
        nuSQ.Set_E(float(E_GeV) * unit_constants.GeV)
        state = np.zeros(3, dtype=float)
        state[mass_idx] = 1.0
        nuSQ.Set_initial_state(state, basis_mass)
        nuSQ.EvolveState()
        matrix[:, mass_idx] = eval_probabilities(nuSQ)

    return matrix


def probability_earth_massbasis(
    *,
    E_GeV: float,
    cos_zenith: float,
    mass_weights: Sequence[float],
    antinu: bool = False,
    config: Optional[NuSQuIDSConfig] = None,
) -> np.ndarray:
    """
    Propagate an incoherent vacuum-mass mixture through the nuSQuIDS Earth.

    Args:
        E_GeV: Neutrino energy in GeV.
        cos_zenith: cos(zenith angle) at the detector.
        mass_weights: Length-3 incoherent mass-eigenstate weights
            ``w_i >= 0`` at Earth entry, typically produced by solar MSW
            propagation after Sun-Earth decoherence.
        antinu: If True, propagate antineutrinos.
        config: Oscillation/numerical configuration. Defaults to
            ``NuSQuIDSConfig()`` when None.

    Returns:
        Length-3 final flavour-probability vector.
    """
    weights = np.asarray(mass_weights, dtype=float).reshape(-1)
    if weights.shape != (3,):
        raise ValueError("mass_weights must be a length-3 vector.")

    norm = weights.sum()
    if np.isfinite(norm) and norm > 0.0:
        weights = weights / norm

    matrix = transition_matrix_earth_mass_to_flavour(
        E_GeV=E_GeV,
        cos_zenith=cos_zenith,
        antinu=antinu,
        config=config,
    )
    return matrix @ weights


def probability_atmosphere(
    *,
    E_GeV: float,
    cos_zenith: float,
    initial_flavour: Union[str, int] = "numu",
    antinu: bool = False,
    config: Optional[NuSQuIDSConfig] = None,
) -> np.ndarray:
    """
    Return final-flavour probabilities through nuSQuIDS ``EarthAtm``.

    ``EarthAtm`` is nuSQuIDS's combined atmosphere-plus-Earth body: it
    accounts for both the neutrino production height in the atmosphere and
    the subsequent matter-affected propagation through the Earth's
    interior along the line of sight set by cos_zenith. This is the
    nuSQuIDS reference counterpart to tpeanuts's own
    medium.atmosphere + medium.earth propagation pipeline, used to
    cross-validate the combined atmosphere-to-detector oscillation
    probability computed by tpeanuts's torch code.

    Args:
        E_GeV: Neutrino energy in GeV.
        cos_zenith: cos(zenith angle) at the detector. cos_zenith=1 is
            vertically downward (minimal/no Earth crossing); cos_zenith=-1
            is vertically upward through the full Earth diameter.
        initial_flavour: Initial flavour label or integer flavour index.
        antinu: If True, propagate the antineutrino state.
        config: Oscillation/numerical configuration. Defaults to
            NuSQuIDSConfig() when None.

    Returns:
        1D float array of length 3 with the final flavour probabilities,
        summing to 1.
    """
    nsq = require_nusquids()
    earth_atm = nsq.EarthAtm()
    track = earth_atm.MakeTrackWithCosine(float(cos_zenith))

    return evolve_with_body(
        body=earth_atm,
        track=track,
        E_GeV=E_GeV,
        initial_flavour=initial_flavour,
        antinu=antinu,
        config=config,
    )


def probability_solar_point(
    *,
    E_MeV: float,
    r0: float,
    initial_flavour: Union[str, int] = "nue",
    impact: float = 0.0,
    antinu: bool = False,
    return_mass_weights: bool = False,
    config: Optional[NuSQuIDSConfig] = None,
) -> np.ndarray:
    """
    Return decohered SunASnu probabilities from a fixed production radius.

    The state is evolved through nuSQuIDS SunASnu from a production radius
    r0/R_sun to the solar surface. The physical production radius is mapped
    to the SunASnu track coordinate with sun_asnu_track_fraction before the
    track is constructed. When EvalMass is available, the final mass-basis
    weights are projected to flavour with vacuum |U_{alpha i}|^2, matching
    the decohered solar-neutrino observable.

    Args:
        E_MeV: Neutrino energy in MeV.
        r0: Production radius in solar-radius units.
        initial_flavour: Initial flavour label or integer index.
        impact: Impact parameter in solar-radius units.
        antinu: If True, propagate antineutrinos.
        return_mass_weights: If True, return the final mass weights instead
            of flavour probabilities. If EvalMass is unavailable, returns NaNs.
        config: Oscillation/numerical configuration. Defaults to
            NuSQuIDSConfig() when None.

    Returns:
        Length-3 array of final flavour probabilities, or mass weights when
        return_mass_weights=True.
    """
    nsq = require_nusquids()
    unit_constants = units(nsq)
    cfg = NuSQuIDSConfig() if config is None else config
    nuSQ = init_solver(antinu=antinu, config=cfg)

    sun = nsq.SunASnu()
    nuSQ.Set_Body(sun)
    nuSQ.Set_Track(make_solar_track(r0, impact=impact, nsq=nsq))
    nuSQ.Set_E(float(E_MeV) * 1.0e-3 * unit_constants.GeV)
    nuSQ.Set_initial_state(initial_state(initial_flavour), nsq.Basis.flavor)
    nuSQ.EvolveState()

    mass_weights = eval_mass_weights(nuSQ)
    if mass_weights is not None:
        if return_mass_weights:
            return mass_weights
        projector = _pmns_from_config(cfg).vacuum_flavour_projector().detach().cpu().numpy()
        return projector @ mass_weights

    averaged_flavour = eval_flavour_averaged(nuSQ)
    if averaged_flavour is not None:
        if return_mass_weights:
            return np.full(3, float("nan"))
        return averaged_flavour

    raise NuSQuIDSError(
        "This nuSQuIDS binding exposes neither EvalMass nor averaged EvalFlavor; "
        "coherent EvalFlavor is not a valid solar-decohered reference."
    )


def probability_grid_vacuum(
    *,
    E_GeV: Sequence[float],
    baseline_km: Sequence[float],
    initial_flavours: Iterable[Union[str, int]] = ("nue", "numu", "nutau"),
    antinu: bool = False,
    config: Optional[NuSQuIDSConfig] = None,
):
    """
    Evaluate vacuum probabilities over a rectangular parameter grid.

    Args:
        E_GeV: Energies in GeV.
        baseline_km: Vacuum baselines in km.
        initial_flavours: Initial flavour labels or integer flavour indices.
        antinu: If True, propagate antineutrinos.
        config: Oscillation/numerical configuration. Defaults to
            NuSQuIDSConfig() when None.

    Returns:
        pandas.DataFrame with one row per grid point and columns
        ``P_nue``, ``P_numu``, and ``P_nutau``.
    """
    import pandas as pd

    rows = []
    for beta in initial_flavours:
        for energy in E_GeV:
            for baseline in baseline_km:
                probs = probability_vacuum(
                    E_GeV=float(energy),
                    baseline_km=float(baseline),
                    initial_flavour=beta,
                    antinu=antinu,
                    config=config,
                )
                rows.append(
                    {
                        "backend": "nuSQuIDS",
                        "mode": "vacuum",
                        "E_GeV": float(energy),
                        "baseline_km": float(baseline),
                        "initial_flavour": str(beta),
                        "antinu": bool(antinu),
                        "P_nue": probs[0],
                        "P_numu": probs[1],
                        "P_nutau": probs[2],
                    }
                )
    return pd.DataFrame.from_records(rows)

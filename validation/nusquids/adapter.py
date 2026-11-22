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

"""Small, pure-nuSQuIDS reference helpers.

Ported from v1_0's ``validation/nusquids/core.py``. The functions here
do not call tpeanuts propagation code, and -- unlike v1_0's own
version -- import nothing from ``tpeanuts`` at all, not even for the
two small pieces of physics it used to borrow (flavour-label lookup,
the vacuum mass -> flavour projector): both are reimplemented locally
in plain numpy below (``_FLAVOUR_INDEX``, ``_vacuum_flavour_projector``).
This is not just decoupling for its own sake -- letting an
"independent" nuSQuIDS cross-check quietly reuse tpeanuts's own PMNS
code would let a real bug in that PMNS class propagate silently to
*both* sides of the comparison, defeating the point of independent
validation.

Every public function here is a wrapper that, when called, executes
the external nuSQuIDS C++ solver via its Python bindings: mixing
angles and the CP phase are set through nuSQuIDS's own
``Set_MixingAngle``/CP-phase setters, the neutrino state is evolved by
nuSQuIDS's ``EvolveState``, and final flavour probabilities are read
back with nuSQuIDS's ``EvalFlavor``. This backend exists purely as an
independent cross-check of tpeanuts's own (torch-based)
oscillation-in-matter code -- see ``validation.nusquids.compare`` for
the scripts that actually run both sides and compare them.

Module contents:
    NuSQuIDSError
        Raised when nuSQuIDS is unavailable or cannot run a requested
        setup.
    require_nusquids(...)
        Import and return the installed nuSQuIDS Python module.
    is_available(...)
        Check whether the nuSQuIDS Python bindings can be imported.
    NUSQUIDS_EARTHATM_HEIGHT_M
        nuSQuIDS's fixed ``EarthAtm`` atmosphere height, the only
        production height for which a tpeanuts comparison is valid.
    data_file_args(...)
        Positional constructor argument(s) pointing an Earth/Sun body at
        a data file under ``NUSQUIDS_DATA_DIR`` (empty if unset).
    units(...)
        Return the nuSQuIDS unit-conversion object.
    normalise_flavour_label(...), initial_state(...), neutrino_type(...)
        Public helpers for flavour labels, initial states, and the
        neutrino/antineutrino enum.
    solver_vacuum_flavour_projector(...)
        Decohered vacuum ``|U_{alpha i}|^2`` read out of nuSQuIDS itself,
        for any number of flavours (including 3+1).
    set_cp_phase(...), eval_probabilities(...), evolve_with_body(...)
        Shared solver utilities used by the probability wrappers.
    configure_solver(...)
        Apply oscillation and numerical settings to an existing
        nuSQuIDS solver object.
    init_solver(...)
        Create and configure a single-energy three-flavour nuSQuIDS
        solver.
    probability_vacuum(...)
        Return final-flavour probabilities after vacuum propagation
        over a fixed baseline.
    probability_earth(...)
        Return final-flavour probabilities through the nuSQuIDS Earth
        body for a given cos(zenith).
    transition_matrix_earth_mass_to_flavour(...)
        Return the nuSQuIDS Earth probability matrix from an
        incoherent vacuum-mass input basis to the final flavour basis.
    probability_earth_massbasis(...)
        Propagate an incoherent vacuum-mass mixture through the
        nuSQuIDS Earth body and return the final flavour
        probabilities.
    probability_atmosphere(...)
        Return final-flavour probabilities through nuSQuIDS's
        ``EarthAtm`` body (atmosphere production height plus Earth)
        for a given cos(zenith).
    probability_solar_point(...)
        Return decohered solar probabilities from a fixed SunASnu
        production radius.
    sun_asnu_track_fraction(...)
        Convert a physical solar radius r/R_sun into the SunASnu track
        coordinate fraction used by the Python body interface.
    probability_grid_vacuum(...)
        Return a table of vacuum probabilities over an energy/baseline
        grid.
"""

from __future__ import annotations

import importlib
import math
import os
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np

from validation.nusquids.config import NuSQuIDSConfig

# Flavour label -> computational basis index, this module's own copy of
# tpeanuts.core.common.neutrino.flavour_index's active-flavour ordering
# (0=e, 1=mu, 2=tau, 3=sterile) -- kept independent, see module docstring.
_FLAVOUR_INDEX = {"nue": 0, "numu": 1, "nutau": 2, "sterile": 3}


class NuSQuIDSError(RuntimeError):
    """Raised when nuSQuIDS is unavailable or cannot run a requested setup."""


def require_nusquids():
    """Import and return the nuSQuIDS Python module.

    Several builds expose different import names, so the resolver
    tries the common spellings used by wheel and source installations.

    Returns:
        The imported nuSQuIDS Python module object.

    Raises:
        NuSQuIDSError: If none of the known module names ("nuSQuIDS",
            "nuSQUIDSpy", "nusquids") can be imported in this
            environment. The message includes installation guidance
            and the per-name import errors encountered.
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
        "Import attempts:\n  - " + "\n  - ".join(errors)
    )
    raise NuSQuIDSError(message)


def is_available() -> bool:
    """Check whether the nuSQuIDS Python bindings can be imported.

    Returns:
        True if ``require_nusquids()`` succeeds, False if it raises
        ``NuSQuIDSError``.
    """
    try:
        require_nusquids()
    except NuSQuIDSError:
        return False
    return True


NUSQUIDS_DATA_DIR_ENV = "NUSQUIDS_DATA_DIR"

# nuSQuIDS's EarthAtm body has a fixed 22 km atmosphere on top of the Earth, and
# ``probability_atmosphere`` takes no production-height argument. A tpeanuts
# comparison is only like-for-like when its own ``h_m`` equals this value
# (measured: the numu error vs nuSQuIDS is minimal at h_m = 22 km and grows on both
# sides, e.g. 4.5e-2 at 10 km, 2.6e-2 at 15 km, 3.8e-3 at 22 km, 2.5e-2 at 30 km).
NUSQUIDS_EARTHATM_HEIGHT_M = 22000.0


def data_file_args(filename: str) -> tuple:
    """Return constructor arguments pointing a body at one nuSQuIDS data file.

    Some wheel builds (e.g. the PyPI ``nusquids`` wheel) bake an absolute
    resource path (``/project/share/nuSQuIDS/``) into the bodies' default
    constructor arguments but do not ship the ``astro/*.dat`` files, so
    ``Earth()``/``EarthAtm()``/``SunASnu()`` fail to open their model
    file. When the environment variable ``NUSQUIDS_DATA_DIR`` names a
    directory holding those files (either that directory itself or its
    parent, containing ``astro/``), this returns ``(path_to_file,)`` for
    ``filename`` so the caller can pass it as the body's model argument.

    Args:
        filename: Data file name, e.g. ``"EARTH_MODEL_PREM.dat"`` or
            ``"bs05_agsop.dat"``.

    Returns:
        ``(path,)`` if ``NUSQUIDS_DATA_DIR`` is set, otherwise ``()`` so the
        bindings' own default is used unchanged.

    Raises:
        NuSQuIDSError: If ``NUSQUIDS_DATA_DIR`` is set but ``filename`` is
            found in neither candidate location.
    """
    root = os.environ.get(NUSQUIDS_DATA_DIR_ENV)
    if not root:
        return ()
    for candidate in (Path(root) / filename, Path(root) / "astro" / filename):
        if candidate.is_file():
            return (str(candidate),)
    raise NuSQuIDSError(
        f"{NUSQUIDS_DATA_DIR_ENV}={root!r} does not contain {filename!r} "
        "(looked in the directory itself and in its 'astro/' subdirectory)."
    )


def units(nsq):
    """Return a unit-conversion object exposing GeV and km in nuSQuIDS's natural units.

    nuSQuIDS internally works in natural units; its ``Const()`` helper
    exposes the multiplicative factors needed to convert physical GeV
    and km into those units. If the installed bindings lack
    ``Const()``, a fallback object with hard-coded standard nuSQuIDS
    unit constants is used instead (with a warning), since some
    minimal/older builds omit it.

    Args:
        nsq: The imported nuSQuIDS module (from ``require_nusquids()``).

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
    """Normalize a flavour label to the bare flavour key used by ``_FLAVOUR_INDEX``.

    Strips antineutrino/"total_" decorations (e.g. "antinumu",
    "numubar", "total_nue") down to the bare flavour name (e.g. "numu",
    "nue"), since the neutrino/antineutrino distinction is handled
    separately via the nuSQuIDS ``NeutrinoType``, not via the flavour
    index itself.

    Args:
        flavour: Flavour label as a string (e.g. "numu", "antinue") or
            an already-integer flavour index, passed through
            unchanged.

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


def _flavour_index(flavour: str) -> int:
    """Resolve a normalized flavour label to its computational basis index.

    Raises:
        ValueError: If ``flavour`` is not one of ``_FLAVOUR_INDEX``'s
            own keys.
    """
    if flavour not in _FLAVOUR_INDEX:
        raise ValueError(f"Unknown flavour label {flavour!r}; expected one of {sorted(_FLAVOUR_INDEX)}.")
    return _FLAVOUR_INDEX[flavour]


def initial_state(flavour: Union[str, int], n_flavours: int = 3) -> np.ndarray:
    """Build a pure flavour-eigenstate occupation vector for nuSQuIDS.

    Args:
        flavour: Initial neutrino flavour, as a string label (e.g.
            "numu") or integer flavour index (0=e, 1=mu, 2=tau,
            3=sterile in the standard ordering). Integer indices are
            used directly (bypassing ``_flavour_index``'s
            active-flavour-only validation), so the sterile index (3,
            for ``n_flavours=4``) is reachable; string labels still
            only resolve the three active flavours.
        n_flavours: Number of flavours in the solver: 3 for the
            standard three-flavour scenario, 4 for the 3+1 sterile
            extension.

    Returns:
        1D float array of length ``n_flavours``, with a 1.0 at the
        initial flavour's index and 0.0 elsewhere, ready to pass to
        nuSQuIDS's ``Set_initial_state`` in the flavor basis.

    Raises:
        ValueError: If the resolved index is out of range for
            ``n_flavours``.
    """
    idx = flavour if isinstance(flavour, int) else _flavour_index(normalise_flavour_label(flavour))
    if not (0 <= idx < n_flavours):
        raise ValueError(f"Flavour index {idx} is out of range for n_flavours={n_flavours}.")
    state = np.zeros(n_flavours, dtype=float)
    state[idx] = 1.0
    return state


def neutrino_type(nsq, *, antinu: bool):
    """Return the nuSQuIDS ``NeutrinoType`` enum value for neutrino or antineutrino.

    Args:
        nsq: The imported nuSQuIDS module.
        antinu: If True, select antineutrino; if False, select
            neutrino.

    Returns:
        The corresponding ``nsq.NeutrinoType`` enum member.
    """
    return nsq.NeutrinoType.antineutrino if antinu else nsq.NeutrinoType.neutrino


def set_cp_phase(nuSQ, config: NuSQuIDSConfig) -> None:
    """Set the Dirac CP-violating phase on a nuSQuIDS solver object.

    Different nuSQuIDS builds expose different method names/signatures
    for setting the CP phase (e.g. indexed ``Set_CPPhase(i, j, delta)``
    for the (1,3)/(2,4) mixing-matrix element pairs, or a flat
    ``Set_CPPhase(delta)``/``Set_DeltaCP(delta)``/
    ``Set_CP_Phase(delta)``). This tries each known candidate in turn
    and stops at the first one that succeeds.

    Args:
        nuSQ: An initialized nuSQuIDS solver object.
        config: ``NuSQuIDSConfig`` providing ``delta_cp`` (radians),
            whether to attempt setting it at all (``set_cp_phase``),
            and whether failure to find a compatible setter should
            raise (``strict_cp``) or only warn.

    Raises:
        NuSQuIDSError: If ``config.strict_cp`` is True and no
            compatible CP-phase setter is found on ``nuSQ``.
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
    """Apply oscillation and numerical settings to an existing nuSQuIDS solver.

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
        The same ``nuSQ`` object, configured in place, for convenient
        chaining.
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

    if config.numneu >= 4:
        # 3+1 sterile extension: active-sterile mixing angles (indices 0-3,
        # 1-3, 2-3), their CP phases (R_34 stays real), and the sterile
        # mass-squared splitting.
        nuSQ.Set_MixingAngle(0, 3, float(config.theta14))
        nuSQ.Set_MixingAngle(1, 3, float(config.theta24))
        nuSQ.Set_MixingAngle(2, 3, float(config.theta34))
        nuSQ.Set_CPPhase(0, 3, float(config.delta14))
        nuSQ.Set_CPPhase(1, 3, float(config.delta24))
        nuSQ.Set_SquareMassDifference(3, float(config.DeltamSq41))

    if hasattr(nuSQ, "Set_rel_error"):
        nuSQ.Set_rel_error(float(config.rel_error))
    if hasattr(nuSQ, "Set_abs_error"):
        nuSQ.Set_abs_error(float(config.abs_error))
    if config.h_max_km is not None and hasattr(nuSQ, "Set_h_max"):
        unit_constants = units(require_nusquids())
        nuSQ.Set_h_max(float(config.h_max_km) * unit_constants.km)

    return nuSQ


def init_solver(*, antinu: bool = False, config: Optional[NuSQuIDSConfig] = None):
    """Create and configure a single-energy three-flavour nuSQuIDS solver.

    Args:
        antinu: If True, create an antineutrino solver; if False, a
            neutrino solver.
        config: Oscillation/numerical configuration applied via
            ``configure_solver``. Defaults to ``NuSQuIDSConfig()`` if
            None.

    Returns:
        A configured ``nsq.nuSQUIDS(config.numneu, ...)`` solver object
        (3 flavours by default, 4 for the 3+1 sterile extension),
        ready to have its body, track, energy, and initial state set
        before calling ``EvolveState``.
    """
    nsq = require_nusquids()
    cfg = NuSQuIDSConfig() if config is None else config
    nuSQ = nsq.nuSQUIDS(cfg.numneu, neutrino_type(nsq, antinu=antinu))
    configure_solver(nuSQ, config=cfg)
    return nuSQ


def eval_probabilities(
    nuSQ, *, numneu: int = 3, output_energy_GeV: Optional[float] = None, antinu_index: int = 0,
) -> np.ndarray:
    """Read final flavour probabilities off an already-evolved nuSQuIDS solver.

    Args:
        nuSQ: A nuSQuIDS solver object after ``EvolveState()`` has been
            called.
        numneu: Number of flavours the solver was built with (3 for
            the standard case, 4 for the 3+1 sterile extension); sizes
            the returned array and the ``EvalFlavor`` index range.
        output_energy_GeV: Optional output energy for bindings whose
            ``EvalFlavor`` overload expects an explicit energy
            argument.
        antinu_index: Optional neutrino/antineutrino index for
            bindings whose ``EvalFlavor`` overload expects a third
            argument.

    Returns:
        1D float array of length ``numneu`` with P(nu_e), P(nu_mu),
        P(nu_tau) (and P(nu_sterile) when ``numneu=4``), or the
        antineutrino equivalents, from nuSQuIDS's ``EvalFlavor``.
    """
    if output_energy_GeV is None:
        return np.asarray([float(nuSQ.EvalFlavor(i)) for i in range(numneu)], dtype=float)

    nsq = require_nusquids()
    unit_constants = units(nsq)
    values = []
    for flavour in range(numneu):
        try:
            values.append(
                float(nuSQ.EvalFlavor(flavour, float(output_energy_GeV) * unit_constants.GeV, antinu_index))
            )
        except TypeError:
            values.append(float(nuSQ.EvalFlavor(flavour, float(output_energy_GeV) * unit_constants.GeV)))
    return np.asarray(values, dtype=float)


def _vacuum_flavour_projector(config: NuSQuIDSConfig, *, antinu: bool = False) -> np.ndarray:
    """Build the real, decohered vacuum flavour projector ``|U_PMNS,alpha i|^2``.

    A plain-numpy reimplementation of
    ``tpeanuts.core.common.pmns.PMNS.vacuum_flavour_projector`` --
    deliberately not imported from tpeanuts, see this module's own
    docstring for why. Uses the same PMNS convention (peanuts/PDG):
    ``U = R23 @ Delta @ R13 @ Delta.conj() @ R12``, ``Delta = diag(1,
    1, exp(i*delta))``, antineutrinos via ``U.conj()``.

    Args:
        config: Supplies ``theta12``/``theta13``/``theta23``/
            ``delta_cp``.
        antinu: If True, use the antineutrino (complex-conjugate) PMNS
            matrix. Mathematically inert here: ``|U.conj()|^2 ==
            |U|^2`` always, so this flag never actually changes the
            returned array -- kept only so the call site reads the
            same as ``adapter.py``'s other ``antinu``-aware functions,
            and because this is the physically correct statement (CP
            violation does not affect the decohered, vacuum-averaged
            mass -> flavour projection).

    Returns:
        Real array shaped ``(3, 3)``, flavour rows and mass-eigenstate
        columns.
    """
    c12, s12 = math.cos(config.theta12), math.sin(config.theta12)
    c13, s13 = math.cos(config.theta13), math.sin(config.theta13)
    c23, s23 = math.cos(config.theta23), math.sin(config.theta23)

    R12 = np.array([[c12, s12, 0.0], [-s12, c12, 0.0], [0.0, 0.0, 1.0]], dtype=complex)
    R13 = np.array([[c13, 0.0, s13], [0.0, 1.0, 0.0], [-s13, 0.0, c13]], dtype=complex)
    R23 = np.array([[1.0, 0.0, 0.0], [0.0, c23, s23], [0.0, -s23, c23]], dtype=complex)
    delta_matrix = np.diag([1.0, 1.0, np.exp(1j * config.delta_cp)])

    U = R23 @ delta_matrix @ R13 @ delta_matrix.conj() @ R12
    if antinu:
        U = U.conj()
    return np.abs(U) ** 2


def solver_vacuum_flavour_projector(nsq, config: NuSQuIDSConfig, *, antinu: bool = False) -> np.ndarray:
    """Read the decohered vacuum projector ``|U_{alpha i}|^2`` out of nuSQuIDS itself.

    Prepares each mass eigenstate in a vacuum solver and reads the
    flavour occupation without evolving, so the projector follows
    nuSQuIDS's own mixing convention for any ``config.numneu`` --
    including the 3+1 sterile extension, which
    ``_vacuum_flavour_projector`` (3x3, PMNS only) does not cover. For
    ``numneu=3`` it agrees with ``_vacuum_flavour_projector`` to machine
    precision.

    Args:
        nsq: The imported nuSQuIDS module (from ``require_nusquids()``).
        config: Oscillation configuration (mixing parameters and
            ``numneu``).
        antinu: If True, build the antineutrino solver.

    Returns:
        Real array shaped ``(numneu, numneu)``, flavour rows and
        mass-eigenstate columns; rows and columns each sum to 1.
    """
    unit_constants = units(nsq)
    nuSQ = init_solver(antinu=antinu, config=config)
    nuSQ.Set_Body(nsq.Vacuum())
    nuSQ.Set_Track(nsq.Vacuum.Track(1.0 * unit_constants.km))
    nuSQ.Set_E(1.0 * unit_constants.GeV)
    n = config.numneu
    projector = np.zeros((n, n), dtype=float)
    for mass_index in range(n):
        state = np.zeros(n, dtype=float)
        state[mass_index] = 1.0
        nuSQ.Set_initial_state(state, nsq.Basis.mass)
        for flavour_index in range(n):
            projector[flavour_index, mass_index] = float(nuSQ.EvalFlavor(flavour_index))
    return projector


def sun_asnu_radius(nsq=None) -> float:
    """Return the nuSQuIDS SunASnu radial final coordinate for a radial track.

    Args:
        nsq: Optional imported nuSQuIDS module. If None,
            ``require_nusquids()`` is used.

    Returns:
        Solar radius in nuSQuIDS natural-coordinate units.

    Raises:
        NuSQuIDSError: If the installed bindings do not expose
            ``SunASnu``.
    """
    nsq = require_nusquids() if nsq is None else nsq
    if not hasattr(nsq, "SunASnu"):
        raise NuSQuIDSError("The installed nuSQuIDS bindings do not expose SunASnu.")
    return nsq.SunASnu.Track(0.0).GetFinalX()


def sun_asnu_track_fraction(r_solar: float) -> float:
    """Map a physical solar radius r/R_sun to the SunASnu track coordinate.

    For the radial SunASnu trajectory exposed by the Python bindings,
    the physical solar interior from centre to surface is sampled over
    the second half of the track coordinate: centre -> 0.5 and surface
    -> 1.0. Passing r/R_sun directly as the track fraction samples the
    wrong part of the trajectory and can make the apparent central
    density tend to zero.

    Args:
        r_solar: Physical production radius in units of R_sun, with 0
            at the solar centre and 1 at the solar surface.

    Returns:
        Normalized SunASnu track coordinate fraction.
    """
    return 0.5 + 0.5 * float(r_solar)


def make_solar_track(r0: float, *, impact: float = 0.0, nsq=None):
    """Build a radial SunASnu track for a solar production radius.

    Args:
        r0: Physical production radius in solar-radius units, r/R_sun.
        impact: Impact parameter in solar-radius units.
        nsq: Optional imported nuSQuIDS module.

    Returns:
        A nuSQuIDS ``SunASnu.Track`` object.
    """
    nsq = require_nusquids() if nsq is None else nsq
    radius = sun_asnu_radius(nsq)
    xini = sun_asnu_track_fraction(r0) * radius
    return nsq.SunASnu.Track(xini, float(impact) * radius)


def eval_mass_weights(nuSQ, n_flavours: int = 3) -> Optional[np.ndarray]:
    """Read normalized mass-basis weights from an evolved nuSQuIDS solver.

    Args:
        nuSQ: A nuSQuIDS solver after ``EvolveState()``.
        n_flavours: Number of mass eigenstates in the solver (3, or 4
            for the 3+1 sterile extension).

    Returns:
        Normalized length-``n_flavours`` mass-weight array, or None when
        the bindings do not expose ``EvalMass``.
    """
    eval_mass = getattr(nuSQ, "EvalMass", None)
    if eval_mass is None:
        return None
    weights = np.asarray([float(eval_mass(i)) for i in range(n_flavours)], dtype=float)
    norm = weights.sum()
    if np.isfinite(norm) and norm > 0.0:
        weights = weights / norm
    return weights


def eval_flavour_averaged(nuSQ) -> Optional[np.ndarray]:
    """Read averaged final-flavour probabilities from bindings supporting it.

    This is a fallback for nuSQuIDS builds without ``EvalMass``. It
    tries the averaged ``EvalFlavor`` overload used by some bindings.

    Args:
        nuSQ: A nuSQuIDS solver after ``EvolveState()``.

    Returns:
        Length-3 probability array, or None if the overload is
        unavailable.
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
    *, body, track, E_GeV: float, initial_flavour: Union[str, int], antinu: bool,
    config: Optional[NuSQuIDSConfig],
) -> np.ndarray:
    """Configure a nuSQuIDS solver for one body/track/energy and evolve it.

    Shared implementation behind ``probability_vacuum``,
    ``probability_earth``, and ``probability_atmosphere``: builds a
    fresh single-energy three-flavour solver, attaches the given
    nuSQuIDS body (the matter profile, e.g. Vacuum/Earth/EarthAtm) and
    track (the trajectory through that body, typically built from a
    baseline or cos(zenith)), sets the neutrino energy and a pure
    initial flavour eigenstate, evolves the state, and reads back the
    final flavour probabilities.

    Args:
        body: A nuSQuIDS body object (e.g. ``nsq.Vacuum()``,
            ``nsq.Earth()``, ``nsq.EarthAtm()``) describing the matter
            density profile to propagate through.
        track: A nuSQuIDS track object compatible with ``body``,
            describing the specific trajectory (baseline or
            cos(zenith)) through it.
        E_GeV: Neutrino energy in GeV.
        initial_flavour: Initial flavour label or index (see
            ``initial_state``).
        antinu: If True, propagate the antineutrino state.
        config: Oscillation/numerical configuration for the solver.
            Defaults to ``NuSQuIDSConfig()`` when None (via
            ``init_solver``).

    Returns:
        1D float array of length ``config.numneu`` (3, or 4 for the
        3+1 sterile extension) with the final flavour probabilities
        (see ``eval_probabilities``).
    """
    nsq = require_nusquids()
    unit_constants = units(nsq)
    cfg = NuSQuIDSConfig() if config is None else config
    nuSQ = init_solver(antinu=antinu, config=cfg)

    nuSQ.Set_Body(body)
    nuSQ.Set_Track(track)
    nuSQ.Set_E(float(E_GeV) * unit_constants.GeV)
    nuSQ.Set_initial_state(initial_state(initial_flavour, n_flavours=cfg.numneu), nsq.Basis.flavor)
    nuSQ.EvolveState()

    return eval_probabilities(nuSQ, numneu=cfg.numneu)


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
        earth = earth_factory(*data_file_args("EARTH_MODEL_PREM.dat"))
        try:
            return earth, _build_earth_track_from_cosine(nsq, earth, cos_zenith)
        except (NuSQuIDSError, TypeError):
            pass

    earth_atm = nsq.EarthAtm(*data_file_args("EARTH_MODEL_PREM.dat"))
    track = _build_track_from_cosine(earth_atm, cos_zenith)
    return earth_atm, track


def probability_vacuum(
    *, E_GeV: float, baseline_km: float, initial_flavour: Union[str, int] = "numu", antinu: bool = False,
    config: Optional[NuSQuIDSConfig] = None,
) -> np.ndarray:
    """Return final-flavour probabilities after vacuum propagation.

    Propagates a pure initial flavour eigenstate over a fixed
    straight-line baseline with no matter effects (nuSQuIDS's
    ``Vacuum`` body), i.e. pure vacuum oscillation. Useful as the
    simplest cross-check between nuSQuIDS and tpeanuts's own vacuum
    oscillation code.

    Args:
        E_GeV: Neutrino energy in GeV.
        baseline_km: Vacuum propagation distance in km.
        initial_flavour: Initial flavour label (e.g. "numu", "nue") or
            integer flavour index.
        antinu: If True, propagate the antineutrino state.
        config: Oscillation/numerical configuration. Defaults to
            ``NuSQuIDSConfig()`` when None.

    Returns:
        1D float array of length 3 with the final P(nu_e), P(nu_mu),
        P(nu_tau) (or antineutrino equivalents), summing to 1.
    """
    nsq = require_nusquids()
    unit_constants = units(nsq)
    body = nsq.Vacuum()
    track = nsq.Vacuum.Track(float(baseline_km) * unit_constants.km)

    return evolve_with_body(
        body=body, track=track, E_GeV=E_GeV, initial_flavour=initial_flavour, antinu=antinu, config=config,
    )


def probability_earth(
    *, E_GeV: float, cos_zenith: float, initial_flavour: Union[str, int] = "numu", antinu: bool = False,
    config: Optional[NuSQuIDSConfig] = None,
) -> np.ndarray:
    """Return final-flavour probabilities through the nuSQuIDS Earth body.

    The track is built from ``cos_zenith``. When the installed
    bindings expose an ``Earth`` body, the zenith direction is
    converted to an Earth-only chord baseline
    ``L = max(0, -2 R_earth cos_zenith)`` and passed to
    ``Earth.Track``. Older or reduced bindings that cannot construct
    ``Earth.Track`` fall back to ``EarthAtm``. The preferred ``Earth``
    path assumes the neutrino is already at the Earth surface, so it
    does not add an atmosphere production-height leg.

    Args:
        E_GeV: Neutrino energy in GeV.
        cos_zenith: cos(zenith angle) at the detector. ``cos_zenith=1``
            is a neutrino arriving straight down (minimal Earth
            crossing); ``cos_zenith=-1`` is straight up through the
            Earth's diameter.
        initial_flavour: Initial flavour label or integer flavour
            index.
        antinu: If True, propagate the antineutrino state.
        config: Oscillation/numerical configuration. Defaults to
            ``NuSQuIDSConfig()`` when None.

    Returns:
        1D float array of length 3 with the final flavour
        probabilities, summing to 1.

    Raises:
        NuSQuIDSError: If neither an "Earth" nor an "EarthAtm" body can
            build a compatible track from the provided cos(zenith).
    """
    nsq = require_nusquids()
    earth, track = _earth_body_and_track(nsq, cos_zenith)

    return evolve_with_body(
        body=earth, track=track, E_GeV=E_GeV, initial_flavour=initial_flavour, antinu=antinu, config=config,
    )


def transition_matrix_earth_mass_to_flavour(
    *, E_GeV: float, cos_zenith: float, antinu: bool = False, config: Optional[NuSQuIDSConfig] = None,
) -> np.ndarray:
    """Return the nuSQuIDS Earth probability matrix from mass to flavour.

    This helper is the Earth-side analogue of the decohered solar
    projection used in ``probability_solar_point``: each pure vacuum
    mass eigenstate ``|nu_i>`` is propagated independently through the
    nuSQuIDS Earth body and the final flavour probabilities are
    assembled into a matrix
    ``T_{alpha i} = P(nu_i -> nu_alpha ; E, cos_zenith)``.

    This is the correct object for solar-neutrino Earth regeneration
    after Sun-Earth decoherence, where the state arriving at the Earth
    is an incoherent mass-basis mixture.

    Args:
        E_GeV: Neutrino energy in GeV.
        cos_zenith: cos(zenith angle) at the detector. ``cos_zenith=1``
            is straight down (minimal Earth crossing) and
            ``cos_zenith=-1`` is straight up through the Earth's
            diameter.
        antinu: If True, propagate antineutrinos.
        config: Oscillation/numerical configuration. Defaults to
            ``NuSQuIDSConfig()`` when None.

    Returns:
        Array shaped ``(numneu, numneu)`` with flavour rows and
        initial-mass columns (``numneu`` from ``config.numneu``, 3 by
        default or 4 for the 3+1 sterile extension).

    Raises:
        NuSQuIDSError: If the installed bindings do not expose
            ``Basis.mass``.
    """
    nsq = require_nusquids()
    unit_constants = units(nsq)
    cfg = NuSQuIDSConfig() if config is None else config
    basis_mass = getattr(getattr(nsq, "Basis", None), "mass", None)
    if basis_mass is None:
        raise NuSQuIDSError(
            "This nuSQuIDS binding does not expose Basis.mass, so Earth "
            "propagation from incoherent solar mass weights is unavailable."
        )

    matrix = np.zeros((cfg.numneu, cfg.numneu), dtype=float)
    for mass_idx in range(cfg.numneu):
        nuSQ = init_solver(antinu=antinu, config=cfg)
        earth, track = _earth_body_and_track(nsq, cos_zenith)
        nuSQ.Set_Body(earth)
        nuSQ.Set_Track(track)
        nuSQ.Set_E(float(E_GeV) * unit_constants.GeV)
        state = np.zeros(cfg.numneu, dtype=float)
        state[mass_idx] = 1.0
        nuSQ.Set_initial_state(state, basis_mass)
        nuSQ.EvolveState()
        matrix[:, mass_idx] = eval_probabilities(nuSQ, numneu=cfg.numneu)

    return matrix


def probability_earth_massbasis(
    *, E_GeV: float, cos_zenith: float, mass_weights: Sequence[float], antinu: bool = False,
    config: Optional[NuSQuIDSConfig] = None,
) -> np.ndarray:
    """Propagate an incoherent vacuum-mass mixture through the nuSQuIDS Earth.

    Args:
        E_GeV: Neutrino energy in GeV.
        cos_zenith: cos(zenith angle) at the detector.
        mass_weights: Incoherent mass-eigenstate weights ``w_i >= 0``
            at Earth entry (length ``config.numneu``: 3, or 4 with a
            zero-padded sterile entry for the 3+1 extension), typically
            produced by solar MSW propagation after Sun-Earth
            decoherence.
        antinu: If True, propagate antineutrinos.
        config: Oscillation/numerical configuration. Defaults to
            ``NuSQuIDSConfig()`` when None.

    Returns:
        Final flavour-probability vector, length ``config.numneu``.

    Raises:
        ValueError: If ``mass_weights`` is not a length-``numneu``
            vector.
    """
    cfg = NuSQuIDSConfig() if config is None else config
    weights = np.asarray(mass_weights, dtype=float).reshape(-1)
    if weights.shape != (cfg.numneu,):
        raise ValueError(f"mass_weights must be a length-{cfg.numneu} vector.")

    norm = weights.sum()
    if np.isfinite(norm) and norm > 0.0:
        weights = weights / norm

    matrix = transition_matrix_earth_mass_to_flavour(E_GeV=E_GeV, cos_zenith=cos_zenith, antinu=antinu, config=config)
    return matrix @ weights


def probability_atmosphere(
    *, E_GeV: float, cos_zenith: float, initial_flavour: Union[str, int] = "numu", antinu: bool = False,
    config: Optional[NuSQuIDSConfig] = None,
) -> np.ndarray:
    """Return final-flavour probabilities through nuSQuIDS ``EarthAtm``.

    ``EarthAtm`` is nuSQuIDS's combined atmosphere-plus-Earth body: it
    accounts for both the neutrino production height in the atmosphere
    and the subsequent matter-affected propagation through the Earth's
    interior along the line of sight set by ``cos_zenith``. This is
    the nuSQuIDS reference counterpart to tpeanuts's own
    ``medium.atmosphere`` + ``medium.earth`` propagation pipeline.

    Args:
        E_GeV: Neutrino energy in GeV.
        cos_zenith: cos(zenith angle) at the detector. ``cos_zenith=1``
            is vertically downward (minimal/no Earth crossing);
            ``cos_zenith=-1`` is vertically upward through the full
            Earth diameter.
        initial_flavour: Initial flavour label or integer flavour
            index (0=e, 1=mu, 2=tau, 3=sterile for
            ``config.numneu=4``).
        antinu: If True, propagate the antineutrino state.
        config: Oscillation/numerical configuration. Defaults to
            ``NuSQuIDSConfig()`` when None. Set ``numneu=4`` (plus the
            sterile angle/splitting fields) for the 3+1 extension.

    Returns:
        1D float array of length ``config.numneu`` with the final
        flavour probabilities, summing to 1.
    """
    nsq = require_nusquids()
    earth_atm = nsq.EarthAtm(*data_file_args("EARTH_MODEL_PREM.dat"))
    track = earth_atm.MakeTrackWithCosine(float(cos_zenith))

    return evolve_with_body(
        body=earth_atm, track=track, E_GeV=E_GeV, initial_flavour=initial_flavour, antinu=antinu, config=config,
    )


def probability_solar_point(
    *, E_MeV: float, r0: float, initial_flavour: Union[str, int] = "nue", impact: float = 0.0,
    antinu: bool = False, return_mass_weights: bool = False, config: Optional[NuSQuIDSConfig] = None,
) -> np.ndarray:
    """Return decohered SunASnu probabilities from a fixed production radius.

    The state is evolved through nuSQuIDS SunASnu from a production
    radius r0/R_sun to the solar surface. The physical production
    radius is mapped to the SunASnu track coordinate with
    ``sun_asnu_track_fraction`` before the track is constructed. When
    ``EvalMass`` is available, the final mass-basis weights are
    projected to flavour with the vacuum ``|U_{alpha i}|^2`` projector
    (``_vacuum_flavour_projector``, this module's own pure-numpy PMNS,
    not tpeanuts's), matching the decohered solar-neutrino observable.

    Args:
        E_MeV: Neutrino energy in MeV.
        r0: Production radius in solar-radius units.
        initial_flavour: Initial flavour label or integer index.
        impact: Impact parameter in solar-radius units.
        antinu: If True, propagate antineutrinos.
        return_mass_weights: If True, return the final mass weights
            instead of flavour probabilities. If ``EvalMass`` is
            unavailable, returns NaNs.
        config: Oscillation/numerical configuration. Defaults to
            ``NuSQuIDSConfig()`` when None.

    Returns:
        Length-``config.numneu`` array of final flavour probabilities, or
        mass weights when ``return_mass_weights=True``.

    Raises:
        NuSQuIDSError: If the installed bindings expose neither
            ``EvalMass`` nor averaged ``EvalFlavor``.
    """
    nsq = require_nusquids()
    unit_constants = units(nsq)
    cfg = NuSQuIDSConfig() if config is None else config
    nuSQ = init_solver(antinu=antinu, config=cfg)

    sun = nsq.SunASnu(*data_file_args("bs05_agsop.dat"))
    nuSQ.Set_Body(sun)
    nuSQ.Set_Track(make_solar_track(r0, impact=impact, nsq=nsq))
    nuSQ.Set_E(float(E_MeV) * 1.0e-3 * unit_constants.GeV)
    nuSQ.Set_initial_state(initial_state(initial_flavour, n_flavours=cfg.numneu), nsq.Basis.flavor)
    nuSQ.EvolveState()

    mass_weights = eval_mass_weights(nuSQ, n_flavours=cfg.numneu)
    if mass_weights is not None:
        if return_mass_weights:
            return mass_weights
        if cfg.numneu == 3:
            projector = _vacuum_flavour_projector(cfg, antinu=antinu)
        else:
            projector = solver_vacuum_flavour_projector(nsq, cfg, antinu=antinu)
        return projector @ mass_weights

    averaged_flavour = eval_flavour_averaged(nuSQ)
    if averaged_flavour is not None:
        if return_mass_weights:
            return np.full(cfg.numneu, float("nan"))
        return averaged_flavour

    raise NuSQuIDSError(
        "This nuSQuIDS binding exposes neither EvalMass nor averaged EvalFlavor; "
        "coherent EvalFlavor is not a valid solar-decohered reference."
    )


def probability_grid_vacuum(
    *, E_GeV: Sequence[float], baseline_km: Sequence[float],
    initial_flavours: Iterable[Union[str, int]] = ("nue", "numu", "nutau"), antinu: bool = False,
    config: Optional[NuSQuIDSConfig] = None,
) -> List[dict]:
    """Evaluate vacuum probabilities over a rectangular parameter grid.

    Args:
        E_GeV: Energies in GeV.
        baseline_km: Vacuum baselines in km.
        initial_flavours: Initial flavour labels or integer flavour
            indices.
        antinu: If True, propagate antineutrinos.
        config: Oscillation/numerical configuration. Defaults to
            ``NuSQuIDSConfig()`` when None.

    Returns:
        A list of plain dicts, one per grid point, each with keys
        ``"backend"``, ``"mode"``, ``"E_GeV"``, ``"baseline_km"``,
        ``"initial_flavour"``, ``"antinu"``, ``"P_nue"``, ``"P_numu"``,
        ``"P_nutau"`` -- deliberately not a ``pandas.DataFrame`` (this
        project avoids a pandas dependency); pass this to
        ``validation.nusquids.compare.grid.export_csv``-style tooling,
        or wrap in ``pandas.DataFrame.from_records`` yourself if
        wanted.
    """
    rows = []
    for beta in initial_flavours:
        for energy in E_GeV:
            for baseline in baseline_km:
                probs = probability_vacuum(
                    E_GeV=float(energy), baseline_km=float(baseline), initial_flavour=beta,
                    antinu=antinu, config=config,
                )
                rows.append({
                    "backend": "nuSQuIDS",
                    "mode": "vacuum",
                    "E_GeV": float(energy),
                    "baseline_km": float(baseline),
                    "initial_flavour": str(beta),
                    "antinu": bool(antinu),
                    "P_nue": probs[0],
                    "P_numu": probs[1],
                    "P_nutau": probs[2],
                })
    return rows

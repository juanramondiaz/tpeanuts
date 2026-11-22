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

"""Full solar-to-detector chain comparison: Sun -> (Earth regeneration) -> detector, tpeanuts vs nuSQuIDS.

Fifth ``validation.nusquids.compare`` script (Phase C) -- the combined
chain v1_0's own ``notebooks/validation/nusquids/nusquids5_*.ipynb``
covered ("SolarNeutrino combined chain"). Composed entirely from the
two already-built, already-verified pieces
(``compare/solar.py``, ``compare/earth.py``): no new physics is
written here, only their composition.

Solar (Sun-to-surface) mass weights leaving the Sun as an incoherent
mixture (``compare/solar.py``'s ``return_mass_weights=True`` path) are
handed to the Earth-regeneration machinery
(``compare/earth.py``'s ``*_probability_earth_massbasis``, the
"incoherent vacuum-mass mixture through the Earth" case that module's
own docstring already describes as exactly the solar-after-Sun-Earth
-decoherence scenario). ``E_MeV`` is converted to
``compare/earth.py``'s own ``E_GeV`` convention (``E_GeV = E_MeV /
1000``) at the seam.

Day/night, handled without an artificial domain restriction (same
completeness choice as ``compare/atmosphere.py``): for
``cos_zenith < 0`` (a genuine night-time, Earth-crossing trajectory)
the full Sun -> Earth-regenerated -> detector chain applies. For
``cos_zenith >= 0`` (daytime, no Earth crossing for a surface
detector) there is no regeneration at all -- the answer is exactly
``compare/solar.py``'s own flavour-probability output (the Sun-to
-surface mass weights rotated by the same vacuum PMNS row that a
trivial, identity Earth leg would otherwise apply), so this module
calls that directly rather than needlessly composing with an identity
matrix.

Same scope restriction as ``compare/solar.py``: standard nu_e
production, purely radial SunASnu track (``impact=0``) -- see that
module's own docstring for why.

Verification without nuSQuIDS installed: no closed-form oracle for the
full composed chain, but two exact internal consistency checks are
available without nuSQuIDS -- (1) the day-time branch must equal
``compare/solar.py``'s own output exactly (same code path, checked
directly here as a regression guard); (2) the vacuum limit
(``r0=1.0``, negligible density) composed with a genuine Earth
crossing must match ``compare/earth.py``'s own
``*_probability_earth_massbasis`` called with the plain vacuum
``|U_{e i}|^2`` weights directly -- confirmed numerically. Direct
nuSQuIDS agreement is deferred to whenever the bindings are installed.

Module contents:
    tpeanuts_probability_solar_earth(...)
        Full Sun -> (Earth regeneration) -> detector flavour
        probabilities, at one (E, r0, cos_zenith) point.
    compare_solar_earth(...)
        Run ``validation.nusquids.compare.grid.compare_grid`` between
        ``tpeanuts_probability_solar_earth`` and the equivalent
        composition of ``adapter.probability_solar_point``/
        ``adapter.probability_earth_massbasis``, over an
        energy/radius/cos(zenith) grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from tpeanuts.util.context import RuntimeContext
from validation.nusquids.adapter import probability_earth_massbasis, probability_solar_point
from validation.nusquids.compare.earth import tpeanuts_probability_earth_massbasis
from validation.nusquids.compare.grid import ComparisonResult, GridPoint, compare_grid, export_csv
from validation.nusquids.compare.solar import tpeanuts_probability_solar_point
from validation.nusquids.config import NuSQuIDSConfig

_LABELS = ("P_nue", "P_numu", "P_nutau")


def tpeanuts_probability_solar_earth(
    *, E_MeV: float, r0: float, cos_zenith: float, antinu: bool = False,
    config: Optional[NuSQuIDSConfig] = None, context: Optional[RuntimeContext] = None, nsteps: int = 200,
) -> np.ndarray:
    """Full Sun -> (Earth regeneration) -> detector flavour probabilities, tpeanuts side.

    Args:
        E_MeV: Neutrino energy in MeV.
        r0: Solar production radius, r/R_sun, in ``[0, 1]``.
        cos_zenith: Detector cos(zenith angle), in ``[-1, 1]``.
            ``cos_zenith < 0`` composes with an Earth-regeneration leg
            (night); ``cos_zenith >= 0`` has no Earth leg at all (day)
            -- see this module's own docstring.
        antinu: If True, use the antineutrino convention throughout.
        config: Shared oscillation configuration. Defaults to
            ``NuSQuIDSConfig()`` when None.
        context: RuntimeContext. Defaults to CPU/float64 when None.
        nsteps: Earth-leg chord discretization (ignored for
            ``cos_zenith >= 0``, see ``compare/earth.py``).

    Returns:
        1D float64 numpy array of length 3, summing to 1.
    """
    mass_weights = tpeanuts_probability_solar_point(
        E_MeV=E_MeV, r0=r0, antinu=antinu, return_mass_weights=True, config=config, context=context,
    )
    if cos_zenith >= 0.0:
        return tpeanuts_probability_solar_point(E_MeV=E_MeV, r0=r0, antinu=antinu, config=config, context=context)

    E_GeV = E_MeV * 1.0e-3
    return tpeanuts_probability_earth_massbasis(
        E_GeV=E_GeV, cos_zenith=cos_zenith, mass_weights=mass_weights, antinu=antinu,
        config=config, context=context, nsteps=nsteps,
    )


def _nusquids_probability_solar_earth(
    *, E_MeV: float, r0: float, cos_zenith: float, antinu: bool, config: NuSQuIDSConfig,
) -> np.ndarray:
    """Full Sun -> (Earth regeneration) -> detector flavour probabilities, nuSQuIDS side.

    Composed the same way v1_0's own combined-chain notebook did:
    ``adapter.probability_solar_point(..., return_mass_weights=True)``
    handed to ``adapter.probability_earth_massbasis``. For
    ``cos_zenith >= 0`` (no Earth crossing), ``probability_solar_point``'s
    own flavour-probability output is the complete answer -- mirrors
    ``tpeanuts_probability_solar_earth``'s own day-time branch exactly.
    """
    mass_weights = probability_solar_point(
        E_MeV=E_MeV, r0=r0, initial_flavour="nue", impact=0.0, antinu=antinu,
        return_mass_weights=True, config=config,
    )
    if cos_zenith >= 0.0:
        return probability_solar_point(
            E_MeV=E_MeV, r0=r0, initial_flavour="nue", impact=0.0, antinu=antinu, config=config,
        )

    E_GeV = E_MeV * 1.0e-3
    return probability_earth_massbasis(
        E_GeV=E_GeV, cos_zenith=cos_zenith, mass_weights=mass_weights, antinu=antinu, config=config,
    )


def compare_solar_earth(
    energies_MeV: Sequence[float], radii: Sequence[float], cos_zeniths: Sequence[float],
    *, antinu: bool = False, config: Optional[NuSQuIDSConfig] = None, context: Optional[RuntimeContext] = None,
    nsteps: int = 200,
) -> ComparisonResult:
    """Compare tpeanuts and nuSQuIDS full solar-to-detector probabilities over an energy/radius/cos(zenith) grid.

    Args:
        energies_MeV: Neutrino energies in MeV.
        radii: Production radii in solar-radius units, each in
            ``[0, 1]``.
        cos_zeniths: Detector cos(zenith) values, each in ``[-1, 1]``.
        antinu: If True, compare the antineutrino channel.
        config: Shared oscillation configuration. Defaults to
            ``NuSQuIDSConfig()`` when None.
        context: tpeanuts RuntimeContext. Defaults to CPU/float64 when
            None.
        nsteps: See ``tpeanuts_probability_solar_earth``.

    Returns:
        A single ``ComparisonResult`` (nuSQuIDS as ``reference``,
        tpeanuts as ``candidate``) -- same one-flavour scope as
        ``compare_solar``.

    Raises:
        validation.nusquids.adapter.NuSQuIDSError: If nuSQuIDS bindings
            are not installed (this function calls the real solver).
    """
    cfg = NuSQuIDSConfig() if config is None else config
    grid_points: Sequence[GridPoint] = [
        {"E_MeV": e, "r0": r, "cos_zenith": c} for e in energies_MeV for r in radii for c in cos_zeniths
    ]

    return compare_grid(
        grid_points,
        lambda E_MeV, r0, cos_zenith: _nusquids_probability_solar_earth(
            E_MeV=E_MeV, r0=r0, cos_zenith=cos_zenith, antinu=antinu, config=cfg,
        ),
        lambda E_MeV, r0, cos_zenith: tpeanuts_probability_solar_earth(
            E_MeV=E_MeV, r0=r0, cos_zenith=cos_zenith, antinu=antinu, config=cfg, context=context, nsteps=nsteps,
        ),
        labels=_LABELS,
    )


def export_all_csv(result: ComparisonResult, output_dir: Union[str, Path]) -> None:
    """Export this comparison's ``ComparisonResult`` to a CSV file.

    Args:
        result: As returned by ``compare_solar_earth``.
        output_dir: Directory to write ``solar_earth_nue.csv`` into
            (created if missing).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    export_csv(result, output_path / "solar_earth_nue.csv")

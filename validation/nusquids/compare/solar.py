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

"""Solar-production (MSW/adiabatic) comparison: tpeanuts vs nuSQuIDS.

Fourth ``validation.nusquids.compare`` script (Phase C). Compares
``tpeanuts_probability_solar_point`` against
``validation.nusquids.adapter.probability_solar_point`` -- both give the
decohered flavour probabilities for a neutrino produced at a fixed
solar radius ``r0`` and propagated adiabatically to the surface.

Scope restriction, deliberate: nuSQuIDS's own ``probability_solar_point``
supports an arbitrary ``initial_flavour`` and a non-zero ``impact``
parameter (the SunASnu track's impact parameter, a genuinely 3-D
geometric effect). Neither has a tpeanuts equivalent: this project's
own solar pipeline (``tpeanuts.physics.evolution.adiabatic.adiabatic.
mass_weights_adiabatic_approximated``) hardcodes the standard nu_e
-production convention (real solar 8B/pp neutrinos are always produced
as nu_e) and is inherently 1-D radial (``tpeanuts.medium.solar.solar.
SolarMedium`` indexes density by radius alone). This module
therefore only compares the physically standard case,
``initial_flavour="nue"``, ``impact=0.0`` -- the case every other
tpeanuts solar-neutrino detector model (Borexino, SNO, SNO-II) already
uses.

The adiabatic approximation itself is not new physics for this
project: it already backs three detector models. What this module
checks is a genuinely new comparison target (nuSQuIDS's own
independent numerical solver), not a first-time trust of the
approximation.

Verification without nuSQuIDS installed: the vacuum limit is an exact,
closed-form check available with no external solver at all --
``mass_weights_adiabatic_approximated`` at zero density must reduce
exactly to ``|U_{e i}|^2`` (the plain vacuum PMNS row), since
``theta_12^M -> theta_12``/``theta_13^M -> theta_13`` as ``n_e -> 0``.
Confirmed directly: the shipped solar density table's own outermost
radius already has electron density ~2e-11 mol/cm^3 (numerically
zero), so ``r0=1.0`` already exercises this limit through the full
wrapper, not just the bare formula.

Module contents:
    tpeanuts_probability_solar_point(...)
        Adiabatic mass-eigenstate weights at a fixed production radius,
        rotated into the flavour basis (or returned as mass weights).
    compare_solar(...)
        Run ``validation.nusquids.compare.grid.compare_grid`` between
        ``tpeanuts_probability_solar_point`` and
        ``adapter.probability_solar_point``, over an energy/production
        -radius grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch

from tpeanuts.medium.solar.solar import SolarMedium
from tpeanuts.physics.common.state import DensityMatrix
from tpeanuts.physics.common.terms import PropagationEnvironment
from tpeanuts.physics.evolution.adiabatic import adiabatic
from tpeanuts.physics.model import PropagationModel
from tpeanuts.util.context import RuntimeContext
from validation.nusquids.adapter import probability_solar_point
from validation.nusquids.compare.grid import ComparisonResult, GridPoint, compare_grid, export_csv
from validation.nusquids.compare.oscillation_model import mixing_and_mass_spectrum
from validation.nusquids.config import NuSQuIDSConfig

_LABELS = ("P_nue", "P_numu", "P_nutau")


def tpeanuts_probability_solar_point(
    *, E_MeV: float, r0: float, antinu: bool = False, return_mass_weights: bool = False,
    config: Optional[NuSQuIDSConfig] = None, context: Optional[RuntimeContext] = None,
) -> np.ndarray:
    """Adiabatic MSW mass weights at production radius ``r0``, rotated into the flavour basis.

    Always assumes a standard nu_e production and a purely radial
    (``impact=0``) trajectory -- see this module's own docstring for
    why. Same call signature (minus ``initial_flavour``/``impact``) as
    ``validation.nusquids.adapter.probability_solar_point``, restricted
    to the case both sides can actually represent.

    Args:
        E_MeV: Neutrino energy in MeV.
        r0: Production radius in solar-radius units, r/R_sun, in
            ``[0, 1]``.
        antinu: If True, use the antineutrino matter-potential sign.
        return_mass_weights: If True, return the mass-basis weights
            instead of flavour probabilities.
        config: Shared oscillation configuration. Defaults to
            ``NuSQuIDSConfig()`` when None.
        context: RuntimeContext. Defaults to CPU/float64 when None.

    Returns:
        Length-3 array: flavour probabilities (summing to 1), or mass
        weights when ``return_mass_weights=True``.

    Raises:
        ValueError: If ``r0`` is outside ``SolarMedium``'s own
            tabulated radial domain (raised by ``electron_density``).
    """
    cfg = NuSQuIDSConfig() if config is None else config
    ctx = RuntimeContext.resolve("cpu", torch.float64) if context is None else context

    mixing, mass_spectrum = mixing_and_mass_spectrum(cfg, ctx)
    model = PropagationModel(mixing=mixing, mass_spectrum=mass_spectrum, antinu=antinu)

    medium = SolarMedium.default(context=ctx)
    n_e = medium.electron_density(torch.tensor(float(r0), device=ctx.device, dtype=ctx.dtype))
    E_t = torch.tensor(float(E_MeV), device=ctx.device, dtype=ctx.dtype)
    environment = PropagationEnvironment(E_MeV=E_t, n_e_mol_cm3=n_e)

    weights = adiabatic.mass_weights_adiabatic(model, environment, variant="approximated")
    if return_mass_weights:
        return weights.detach().cpu().numpy().astype(np.float64).reshape(3)

    rho_mass = DensityMatrix.from_weights(weights, mixing.space, ctx)
    rho_flavour = rho_mass.conjugate_by(mixing.pmns_matrix(antinu=antinu))
    return rho_flavour.probabilities().detach().cpu().numpy().astype(np.float64).reshape(3)


def compare_solar(
    energies_MeV: Sequence[float], radii: Sequence[float],
    *, antinu: bool = False, config: Optional[NuSQuIDSConfig] = None, context: Optional[RuntimeContext] = None,
) -> ComparisonResult:
    """Compare tpeanuts and nuSQuIDS solar-point probabilities over an energy/radius grid.

    Args:
        energies_MeV: Neutrino energies in MeV.
        radii: Production radii in solar-radius units, each in
            ``[0, 1]``.
        antinu: If True, compare the antineutrino channel.
        config: Shared oscillation configuration. Defaults to
            ``NuSQuIDSConfig()`` when None.
        context: tpeanuts RuntimeContext. Defaults to CPU/float64 when
            None.

    Returns:
        A single ``ComparisonResult`` (nuSQuIDS as ``reference``,
        tpeanuts as ``candidate``) -- unlike ``compare_vacuum``/
        ``compare_earth``, there is only one initial flavour to compare
        (see this module's own docstring), so no per-flavour dict is
        needed.

    Raises:
        validation.nusquids.adapter.NuSQuIDSError: If nuSQuIDS bindings
            are not installed (this function calls the real solver).
    """
    cfg = NuSQuIDSConfig() if config is None else config
    grid_points: Sequence[GridPoint] = [{"E_MeV": e, "r0": r} for e in energies_MeV for r in radii]

    return compare_grid(
        grid_points,
        lambda E_MeV, r0: probability_solar_point(
            E_MeV=E_MeV, r0=r0, initial_flavour="nue", impact=0.0, antinu=antinu, config=cfg,
        ),
        lambda E_MeV, r0: tpeanuts_probability_solar_point(
            E_MeV=E_MeV, r0=r0, antinu=antinu, config=cfg, context=context,
        ),
        labels=_LABELS,
    )


def export_all_csv(result: ComparisonResult, output_dir: Union[str, Path]) -> None:
    """Export this comparison's ``ComparisonResult`` to a CSV file.

    Args:
        result: As returned by ``compare_solar``.
        output_dir: Directory to write ``solar_nue.csv`` into (created
            if missing).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    export_csv(result, output_path / "solar_nue.csv")

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

"""3+1 sterile-extension comparison (atmosphere and solar): tpeanuts vs nuSQuIDS.

Sixth and final ``validation.nusquids.compare`` script (Phase C),
mirroring v1_0's own paired
``notebooks/validation/nusquids/nusquids{6,7}_*.ipynb`` (6 = BSM
atmosphere sterile, 7 = BSM solar sterile) in a single module, since
both share the same sterile-extended oscillation model.

Reuses every non-sterile-specific piece already built: geometry
reconciliation (``eta_rad = arccos(-cos_zenith)``), atmosphere+Earth leg
composition, and the atmosphere density constants all come from
``compare/atmosphere.py``'s own established pattern (via
``compare/oscillation_model.py``'s shared helpers); only the
oscillation model itself changes.

Model reconciliation, atmosphere leg: ``NuSQuIDSConfig``'s
``numneu=4``/``theta14``/``theta24``/``theta34``/``delta14``/
``delta24``/``DeltamSq41`` map onto
``compare.oscillation_model.sterile_mixing_and_mass_spectrum``
(``FlavourSpace(n_sterile=1)``, ``SterileMixingParameters``, with
``delta_tau=0`` reconciling nuSQuIDS's own real-R_34 convention with
tpeanuts's more general one -- see that function's own docstring). The
sterile neutral-current matter potential
(``tpeanuts.physics.extensions.sterile.SterileNCTerm``) is added as an
active extension, matching nuSQuIDS's own ``configure_solver`` sterile
handling.

Model reconciliation, solar leg: ``tpeanuts.physics.evolution.adiabatic.
adiabatic.mass_weights_adiabatic_approximated`` (used by
``compare/solar.py``) is a plain-SM-only closed-form reduction that
explicitly rejects a sterile-extended space -- so this module uses
``variant="exact"`` instead (pointwise diagonalisation of the full
flavour-basis generator, active extensions included), the same variant
this project's own detector models fall back to whenever an extension
is active.

Scope: same restriction as ``compare/solar.py`` (standard nu_e
production, radial track) and the same day/night handling as
``compare/atmosphere.py``/``compare/solar_earth.py`` -- reused, not
reinvented, here.

Verification without nuSQuIDS installed: the sterile mixing angles set
to zero must reproduce the plain 3-flavour result exactly (both
functions' own closure to the already-verified non-sterile
comparisons in ``compare/atmosphere.py``/``compare/solar.py``); a
non-zero sterile mixing angle must measurably change the result
(confirms the sterile extension is genuinely wired in, not silently
inert). Direct nuSQuIDS agreement is deferred to whenever the bindings
are installed.

Module contents:
    tpeanuts_probability_atmosphere_sterile(...)
        Pure flavour eigenstate through the composed sterile-extended
        atmosphere(+Earth) path.
    tpeanuts_probability_solar_point_sterile(...)
        Sterile-extended adiabatic mass weights at a fixed solar
        production radius, rotated into the flavour basis.
    compare_bsm_sterile_atmosphere(...)
        ``compare_grid`` between the tpeanuts atmosphere-sterile
        function and ``adapter.probability_atmosphere`` (``numneu=4``).
    compare_bsm_sterile_solar(...)
        ``compare_grid`` between the tpeanuts solar-sterile function
        and ``adapter.probability_solar_point`` (``numneu=4``).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch

from tpeanuts.medium.atmosphere.atmosphere import AtmosphereMedium
from tpeanuts.medium.earth.earth import EarthMedium
from tpeanuts.medium.solar.solar import SolarMedium
from tpeanuts.physics.common.state import DensityMatrix
from tpeanuts.physics.common.terms import PropagationEnvironment
from tpeanuts.physics.evolution import PropagationMethod
from tpeanuts.physics.evolution.adiabatic import adiabatic
from tpeanuts.physics.extensions.sterile import SterileNCTerm
from tpeanuts.physics.model import PropagationModel
from tpeanuts.util.context import RuntimeContext
from validation.nusquids.adapter import initial_state, probability_atmosphere, probability_solar_point
from validation.nusquids.compare.grid import ComparisonResult, GridPoint, compare_grid, export_csv
from validation.nusquids.compare.oscillation_model import (
    atmosphere_density_kwargs,
    evolutor_from_profile,
    sterile_mixing_and_mass_spectrum,
)
from validation.nusquids.config import NuSQuIDSConfig

_LABELS = ("P_nue", "P_numu", "P_nutau", "P_sterile")


def tpeanuts_probability_atmosphere_sterile(
    *, E_GeV: float, cos_zenith: float, h_m: float, initial_flavour: str, antinu: bool = False,
    config: Optional[NuSQuIDSConfig] = None, context: Optional[RuntimeContext] = None, nsteps: int = 200,
) -> np.ndarray:
    """Propagate a pure initial flavour eigenstate through the composed sterile-extended atmosphere(+Earth) path.

    Same call signature as
    ``validation.nusquids.adapter.probability_atmosphere`` (with a
    ``config.numneu=4``), so both are valid
    ``validation.nusquids.compare.grid.ProbabilityFn`` implementations
    for the same grid.

    Args:
        E_GeV: Neutrino energy in GeV.
        cos_zenith: Detector cos(zenith angle), in ``[-1, 1]``.
        h_m: Production altitude above the surface, in metres.
        initial_flavour: Initial flavour label ("nue", "numu", "nutau",
            or "sterile").
        antinu: If True, propagate the antineutrino state.
        config: Shared oscillation configuration, including the
            sterile fields. Defaults to ``NuSQuIDSConfig(numneu=4)``
            when None.
        context: RuntimeContext. Defaults to CPU/float64 when None.
        nsteps: Chord discretization for both legs.

    Returns:
        1D float64 numpy array of length 4, summing to 1.
    """
    cfg = NuSQuIDSConfig(numneu=4) if config is None else config
    ctx = RuntimeContext.resolve("cpu", torch.float64) if context is None else context

    eta_rad = math.acos(-float(cos_zenith))
    mixing, mass_spectrum = sterile_mixing_and_mass_spectrum(cfg, ctx)
    model = PropagationModel(
        mixing=mixing, mass_spectrum=mass_spectrum, antinu=antinu, extensions=(SterileNCTerm(),),
    )

    atmosphere_profile = AtmosphereMedium(
        eta_rad=eta_rad, h_m=float(h_m), context=ctx, method=PropagationMethod.ANALYTIC, nsteps=nsteps,
        density_source="exponential", density_kwargs=atmosphere_density_kwargs(cfg),
    ).profile()
    evolutor = evolutor_from_profile(atmosphere_profile, model, E_GeV, ctx, reverse=True)

    if cos_zenith < 0.0:
        earth_profile = EarthMedium(
            eta_rad=eta_rad, context=ctx, method=PropagationMethod.ANALYTIC, nsteps=nsteps,
        ).profile()
        S_earth = evolutor_from_profile(earth_profile, model, E_GeV, ctx)
        evolutor = S_earth @ evolutor

    flavour_idx = int(np.argmax(initial_state(initial_flavour, n_flavours=4)))
    initial = DensityMatrix.basis_state(flavour_idx, mixing.space, ctx)
    evolved = initial.conjugate_by(evolutor)

    return evolved.probabilities().detach().cpu().numpy().astype(np.float64).reshape(4)


def tpeanuts_probability_solar_point_sterile(
    *, E_MeV: float, r0: float, antinu: bool = False, return_mass_weights: bool = False,
    config: Optional[NuSQuIDSConfig] = None, context: Optional[RuntimeContext] = None,
) -> np.ndarray:
    """Sterile-extended adiabatic mass weights at production radius ``r0``, rotated into the flavour basis.

    Uses ``variant="exact"`` (pointwise diagonalisation), since
    ``variant="approximated"`` explicitly rejects a sterile-extended
    space. Same scope restriction as ``compare/solar.py`` (standard
    nu_e production, radial track).

    Args:
        E_MeV: Neutrino energy in MeV.
        r0: Production radius in solar-radius units, r/R_sun, in
            ``[0, 1]``.
        antinu: If True, use the antineutrino matter-potential sign.
        return_mass_weights: If True, return the mass-basis weights
            (length 4) instead of flavour probabilities.
        config: Shared oscillation configuration, including the
            sterile fields. Defaults to ``NuSQuIDSConfig(numneu=4)``
            when None.
        context: RuntimeContext. Defaults to CPU/float64 when None.

    Returns:
        Length-4 array: flavour probabilities (summing to 1), or mass
        weights when ``return_mass_weights=True``.
    """
    cfg = NuSQuIDSConfig(numneu=4) if config is None else config
    ctx = RuntimeContext.resolve("cpu", torch.float64) if context is None else context

    mixing, mass_spectrum = sterile_mixing_and_mass_spectrum(cfg, ctx)
    model = PropagationModel(
        mixing=mixing, mass_spectrum=mass_spectrum, antinu=antinu, extensions=(SterileNCTerm(),),
    )

    medium = SolarMedium.default(context=ctx)
    n_e = medium.electron_density(torch.tensor(float(r0), device=ctx.device, dtype=ctx.dtype))
    n_n = medium.neutron_density(torch.tensor(float(r0), device=ctx.device, dtype=ctx.dtype))
    E_t = torch.tensor(float(E_MeV), device=ctx.device, dtype=ctx.dtype)
    environment = PropagationEnvironment(E_MeV=E_t, n_e_mol_cm3=n_e, n_n_mol_cm3=n_n)

    weights = adiabatic.mass_weights_adiabatic(model, environment, variant="exact")
    if return_mass_weights:
        return weights.detach().cpu().numpy().astype(np.float64).reshape(4)

    rho_mass = DensityMatrix.from_weights(weights, mixing.space, ctx)
    rho_flavour = rho_mass.conjugate_by(mixing.pmns_matrix(antinu=antinu))
    return rho_flavour.probabilities().detach().cpu().numpy().astype(np.float64).reshape(4)


def compare_bsm_sterile_atmosphere(
    energies_GeV: Sequence[float], cos_zeniths: Sequence[float], h_m: float,
    *, initial_flavours: Sequence[str] = ("nue", "numu", "nutau", "sterile"), antinu: bool = False,
    config: Optional[NuSQuIDSConfig] = None, context: Optional[RuntimeContext] = None, nsteps: int = 200,
) -> Dict[str, ComparisonResult]:
    """Compare tpeanuts and nuSQuIDS sterile-extended atmosphere(+Earth) probabilities over an energy/cos(zenith) grid.

    Args:
        energies_GeV: Neutrino energies in GeV.
        cos_zeniths: Detector cos(zenith) values, each in ``[-1, 1]``.
        h_m: Fixed production altitude above the surface, in metres.
            nuSQuIDS's ``EarthAtm`` has a fixed atmosphere height and
            ignores this, so the comparison is only like-for-like at
            ``validation.nusquids.adapter.NUSQUIDS_EARTHATM_HEIGHT_M``.
        initial_flavours: Initial flavour labels, including "sterile".
        antinu: If True, compare the antineutrino channel.
        config: Shared oscillation configuration. Defaults to
            ``NuSQuIDSConfig(numneu=4)`` when None.
        context: tpeanuts RuntimeContext. Defaults to CPU/float64 when
            None.
        nsteps: See ``tpeanuts_probability_atmosphere_sterile``.

    Returns:
        Dict mapping each initial flavour label to its own
        ``ComparisonResult`` (4 labels: nue/numu/nutau/sterile).

    Raises:
        validation.nusquids.adapter.NuSQuIDSError: If nuSQuIDS bindings
            are not installed.
    """
    cfg = NuSQuIDSConfig(numneu=4) if config is None else config
    grid_points: Sequence[GridPoint] = [
        {"E_GeV": e, "cos_zenith": c} for e in energies_GeV for c in cos_zeniths
    ]

    results: Dict[str, ComparisonResult] = {}
    for flavour in initial_flavours:
        results[flavour] = compare_grid(
            grid_points,
            lambda E_GeV, cos_zenith, _flavour=flavour: probability_atmosphere(
                E_GeV=E_GeV, cos_zenith=cos_zenith, initial_flavour=_flavour, antinu=antinu, config=cfg,
            ),
            lambda E_GeV, cos_zenith, _flavour=flavour: tpeanuts_probability_atmosphere_sterile(
                E_GeV=E_GeV, cos_zenith=cos_zenith, h_m=h_m, initial_flavour=_flavour, antinu=antinu,
                config=cfg, context=context, nsteps=nsteps,
            ),
            labels=_LABELS,
        )
    return results


def compare_bsm_sterile_solar(
    energies_MeV: Sequence[float], radii: Sequence[float],
    *, antinu: bool = False, config: Optional[NuSQuIDSConfig] = None, context: Optional[RuntimeContext] = None,
) -> ComparisonResult:
    """Compare tpeanuts and nuSQuIDS sterile-extended solar-point probabilities over an energy/radius grid.

    Args:
        energies_MeV: Neutrino energies in MeV.
        radii: Production radii in solar-radius units, each in
            ``[0, 1]``.
        antinu: If True, compare the antineutrino channel.
        config: Shared oscillation configuration. Defaults to
            ``NuSQuIDSConfig(numneu=4)`` when None.
        context: tpeanuts RuntimeContext. Defaults to CPU/float64 when
            None.

    Returns:
        A single ``ComparisonResult`` (same one-flavour scope as
        ``compare_solar``).

    Raises:
        validation.nusquids.adapter.NuSQuIDSError: If nuSQuIDS bindings
            are not installed.
    """
    cfg = NuSQuIDSConfig(numneu=4) if config is None else config
    grid_points: Sequence[GridPoint] = [{"E_MeV": e, "r0": r} for e in energies_MeV for r in radii]

    return compare_grid(
        grid_points,
        lambda E_MeV, r0: probability_solar_point(
            E_MeV=E_MeV, r0=r0, initial_flavour="nue", impact=0.0, antinu=antinu, config=cfg,
        ),
        lambda E_MeV, r0: tpeanuts_probability_solar_point_sterile(
            E_MeV=E_MeV, r0=r0, antinu=antinu, config=cfg, context=context,
        ),
        labels=_LABELS,
    )


def export_all_csv(
    results: Union[Dict[str, ComparisonResult], ComparisonResult], output_dir: Union[str, Path], *, prefix: str,
) -> None:
    """Export one or more ComparisonResults to CSV files under a shared prefix.

    Args:
        results: Either a dict (atmosphere case, one entry per flavour)
            or a single ``ComparisonResult`` (solar case), as returned
            by ``compare_bsm_sterile_atmosphere``/
            ``compare_bsm_sterile_solar``.
        output_dir: Directory to write into (created if missing).
        prefix: File name prefix, e.g. ``"bsm_sterile_atmosphere"`` or
            ``"bsm_sterile_solar"``.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if isinstance(results, ComparisonResult):
        export_csv(results, output_path / f"{prefix}.csv")
        return
    for label, result in results.items():
        export_csv(result, output_path / f"{prefix}_{label}.csv")

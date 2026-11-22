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

"""``propagate``-shaped nuSQuIDS reference: a ``Flux`` in, a ``Flux`` out, same medium objects as tpeanuts.

``nusquids_propagate(medium, flux, model)`` mirrors ``medium.propagate(flux, model)`` so the same chained-API objects
(a ``VacuumMedium``, a surface-detector ``EarthMedium``) can be run through both engines and compared leg by leg.
It is a validation tool, not a tpeanuts backend: nuSQuIDS stays an independent oracle (see
``validation.nusquids``'s package docstring), so this module lives here, the one place allowed to import both sides.

What is shared and what is not: the medium geometry (baseline, nadir angle) and the flux (energy grid, flavour
populations, location/history) come from the tpeanuts objects; the oscillation parameters come from ``config``, a
``NuSQuIDSConfig``, exactly as in every other ``compare`` module, and only ``model.antinu`` is read from the model.
nuSQuIDS returns probabilities, so the result is a flavour-incoherent state (``DensityMatrix.from_weights``); the
input state must therefore be flavour-diagonal (a pure flavour or an incoherent mixture of flavours), and the energy
grid one-dimensional.

Module contents:
    nusquids_propagate(...)
        Propagate a flavour-diagonal ``Flux`` through a ``VacuumMedium`` or a surface ``EarthMedium`` with nuSQuIDS.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from tpeanuts.interfaces.flux import Flux
from tpeanuts.medium.earth.earth import EarthMedium
from tpeanuts.medium.vacuum.vacuum import VacuumMedium
from tpeanuts.physics.common.state import DensityMatrix
from tpeanuts.physics.model import PropagationModel
from validation.nusquids.adapter import probability_earth, probability_vacuum
from validation.nusquids.config import NuSQuIDSConfig

_FLAVOURS = ("nue", "numu", "nutau")
_DIAGONAL_TOLERANCE = 1.0e-9


def _flavour_weights(flux: Flux) -> np.ndarray:
    """Flavour populations of a diagonal 3-flavour state, shaped ``(n_E, 3)``."""
    tensor = flux.state.tensor.detach().cpu()
    if tensor.shape[-1] != 3:
        raise ValueError("nusquids_propagate supports the plain three-flavour space only")
    off_diagonal = tensor - torch.diag_embed(torch.diagonal(tensor, dim1=-2, dim2=-1))
    if float(off_diagonal.abs().max()) > _DIAGONAL_TOLERANCE:
        raise ValueError("nusquids_propagate needs a flavour-diagonal state (a pure flavour or an incoherent mixture)")
    weights = flux.state.probabilities().detach().cpu().numpy().astype(np.float64)
    return np.broadcast_to(weights, (flux.energy_MeV.shape[0], 3))


def nusquids_propagate(
    medium: object, flux: Flux, model: PropagationModel, *, config: Optional[NuSQuIDSConfig] = None,
    location: Optional[str] = None,
) -> Flux:
    """Propagate ``flux`` through ``medium`` with nuSQuIDS and return it as a ``Flux``.

    Args:
        medium: A ``VacuumMedium`` (baseline ``length_m``) or an ``EarthMedium`` with a scalar nadir angle ``eta_rad``
            and ``depth_m == 0`` (nuSQuIDS side: ``cos_zenith = -cos(eta_rad)``, see ``compare.earth``).
        flux: A flavour-diagonal Flux on a one-dimensional energy grid.
        model: Supplies ``antinu`` only; the oscillation parameters are ``config``'s.
        config: Shared oscillation configuration; ``NuSQuIDSConfig()`` if None.
        location: ``location`` of the result; as ``medium.propagate`` does if None (``"detector"`` for the Earth,
            the flux's own for a vacuum leg after a source medium, ``"detector"`` straight from a source).

    Returns:
        A new Flux with the evolved flavour-incoherent state and the medium's step appended to its history.

    Raises:
        ValueError: If the state is not flavour-diagonal, the energy grid is not one-dimensional, the Earth medium is
            batched or has a detector depth, or the medium is of another kind.
        NuSQuIDSError: If nuSQuIDS is not installed (from the adapter).
    """
    cfg = NuSQuIDSConfig() if config is None else config
    if flux.energy_MeV.ndim != 1:
        raise ValueError("nusquids_propagate needs a one-dimensional energy grid")
    if model.antinu != flux.antinu:
        raise ValueError("model and flux disagree on the neutrino/antineutrino convention")

    if isinstance(medium, VacuumMedium):
        step, default_location = "vacuum", ("detector" if flux.location == "source" else flux.location)

        def probabilities(E_GeV: float, flavour: str) -> np.ndarray:
            return probability_vacuum(
                E_GeV=E_GeV, baseline_km=float(medium.length_m) / 1.0e3, initial_flavour=flavour,
                antinu=model.antinu, config=cfg,
            )
    elif isinstance(medium, EarthMedium):
        eta_rad = torch.as_tensor(medium.eta_rad)
        if eta_rad.ndim != 0 or float(torch.as_tensor(medium.depth_m)) != 0.0:
            raise ValueError("nusquids_propagate supports a scalar-eta_rad, surface-detector (depth_m = 0) EarthMedium")
        cos_zenith = -math.cos(float(eta_rad))
        step, default_location = "earth", "detector"

        def probabilities(E_GeV: float, flavour: str) -> np.ndarray:
            return probability_earth(
                E_GeV=E_GeV, cos_zenith=cos_zenith, initial_flavour=flavour, antinu=model.antinu, config=cfg,
            )
    else:
        raise ValueError(f"nusquids_propagate supports VacuumMedium and EarthMedium, got {type(medium).__name__}")

    weights = _flavour_weights(flux)
    energies_GeV = flux.energy_MeV.detach().cpu().numpy().astype(np.float64) * 1.0e-3
    final = np.zeros_like(weights)
    for i, E_GeV in enumerate(energies_GeV):
        for k, flavour in enumerate(_FLAVOURS):
            if weights[i, k] > 0.0:
                final[i] += weights[i, k] * probabilities(float(E_GeV), flavour)

    state = DensityMatrix.from_weights(
        torch.as_tensor(final, device=flux.context.device, dtype=flux.context.dtype), flux.space, flux.context,
    )
    return flux.with_state(state, location=location or default_location, step=step)

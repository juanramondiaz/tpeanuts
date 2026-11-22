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

"""Applying a medium's evolution operator to a ``Flux``: what every medium's ``propagate`` has in common.

Each medium builds its own evolution operator (``EarthMedium.evolutor`` and its siblings -- how the energy axis meets
the medium's own batch axes is specific to each one), and then the steps are the same: check that the flux and the
model agree, apply the operator to the flux's state, and record where the flux now is. They are written once here.

Module contents:
    check_compatible(...)
        Raise if a flux and a model do not describe the same particle type or live on different devices/dtypes.
    evolve_flux(...)
        Apply an evolution operator to a flux's state and return the flux in its new state.
"""

from __future__ import annotations

from typing import Optional

import torch

from tpeanuts.interfaces.flux import Flux
from tpeanuts.physics.model import PropagationModel
from tpeanuts.util.context import RuntimeContext


def check_compatible(flux: Flux, model: PropagationModel, context: RuntimeContext) -> None:
    """Raise if ``flux`` and ``model`` cannot be propagated together through a medium built on ``context``.

    Args:
        flux: The flux to propagate.
        model: The propagation model.
        context: RuntimeContext of the medium doing the propagation.

    Raises:
        ValueError: If the flux is on another device/dtype than the medium, or if the flux and the model disagree on
            whether they describe neutrinos or antineutrinos -- which would otherwise run silently with the wrong sign
            of the matter potential.
    """
    if flux.context.device != context.device or flux.context.dtype != context.dtype:
        raise ValueError(
            f"the flux is on {flux.context.device}/{flux.context.dtype} but the medium is on "
            f"{context.device}/{context.dtype}."
        )
    flux_antinu = torch.as_tensor(flux.antinu)
    model_antinu = torch.as_tensor(model.antinu)
    try:
        torch.broadcast_shapes(flux_antinu.shape, model_antinu.shape)
    except RuntimeError as exc:
        raise ValueError("flux.antinu and model.antinu have shapes that cannot be broadcast.") from exc
    if bool(torch.any(flux_antinu.cpu() != model_antinu.cpu())):
        raise ValueError(
            f"flux.antinu ({flux.antinu!r}) and model.antinu ({model.antinu!r}) disagree: build the model with the "
            "same antinu convention as the flux it propagates."
        )


def evolve_flux(
    flux: Flux,
    model: PropagationModel,
    operator: torch.Tensor,
    *,
    location: str,
    step: str,
    keep_location: Optional[bool] = None,
) -> Flux:
    """Apply an evolution operator to ``flux``'s state and return the flux in its new state.

    A unitary operator conjugates the state; if ``model`` has active dissipators the operator is the Liouville-space
    one and acts on the vectorised density matrix instead.

    Args:
        flux: The flux to propagate.
        model: The propagation model the operator was built from.
        operator: The evolution operator, shaped ``(..., dim, dim)`` (``(..., dim*dim, dim*dim)`` with dissipators).
        location: The ``location`` the returned flux is labelled with.
        step: The label appended to the returned flux's ``history``.
        keep_location: If True, the flux keeps its own ``location`` instead of ``location``.

    Returns:
        The same flux (energy grid, intensity, production distribution shared) in the evolved state.
    """
    if model.dissipators:
        state = flux.state.apply_superoperator(operator)
    else:
        state = flux.state.conjugate_by(operator)
    return flux.with_state(state, location=flux.location if keep_location else location, step=step)

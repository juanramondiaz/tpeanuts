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

"""The trivial medium: a single zero-density segment spanning the baseline.

The simplest possible ``tpeanuts.interfaces.medium.MediumProfile`` producer
-- exactly one segment, ``n_e_mol_cm3 == 0`` (no matter effect at all), and
neither ``n_n_mol_cm3`` nor ``B_field_T`` set (neither the sterile
neutral-current term, NSI's neutron coupling, nor the magnetic-moment
extension have anything to sample here). Validation (a strictly positive
baseline) is not duplicated here: ``MediumProfile.__post_init__`` already
enforces it, so a non-positive ``length_m`` fails there instead.

Module contents:
    VacuumMedium
        Frozen dataclass: a baseline distance and a RuntimeContext, with a
        ``profile()`` method building the corresponding ``MediumProfile`` and
        an ``evolutor()`` method building the evolution operator over it, and a
        ``propagate()`` method applying it to a ``Flux``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

import torch

from tpeanuts.interfaces.flux import Flux
from tpeanuts.interfaces.medium import MediumProfile
from tpeanuts.medium.common.propagation import check_compatible, evolve_flux
from tpeanuts.physics.common.terms import PropagationEnvironment
from tpeanuts.physics.evolution.numerical.numerical import segment_evolutor_numerical
from tpeanuts.physics.model import PropagationModel
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import TensorLike, as_tensor


@dataclass(frozen=True)
class VacuumMedium:
    """A source-to-detector path with no matter in between.

    Attributes:
        length_m: Baseline distance from source to detector, in metres;
            scalar or batched over scenarios.
        context: RuntimeContext the returned profile's tensors are built on.

    Examples:
        >>> import math
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> from tpeanuts.medium import VacuumMedium
        >>> profile = VacuumMedium(length_m=torch.tensor(1.0e5, dtype=torch.float64), context=context).profile()
        >>> profile.n_segments, float(profile.n_e_mol_cm3.sum())   # one segment, no electrons
        (1, 0.0)
    """

    length_m: torch.Tensor
    context: RuntimeContext

    if TYPE_CHECKING:  # the runtime __init__ is the dataclass one; tensor fields accept any TensorLike

        def __init__(
            self,
            length_m: TensorLike,
            context: RuntimeContext,
        ) -> None: ...

    def profile(self) -> MediumProfile:
        """Build this medium's single-segment, zero-density MediumProfile.

        Returns:
            MediumProfile with ``n_segments == 1``: ``length_m`` shaped
            ``(..., 1)`` (a trailing segment axis added to this instance's
            own, segment-axis-free ``length_m``) and ``n_e_mol_cm3 == 0``.

        Raises:
            ValueError: If ``length_m`` is not strictly positive (raised by
                ``MediumProfile`` itself).
        """
        length = as_tensor(self.length_m, device=self.context.device, dtype=self.context.dtype)
        return MediumProfile(
            length_m=length.unsqueeze(-1),
            n_e_mol_cm3=0.0,
            context=self.context,
        )

    def evolutor(self, model: PropagationModel, energy_MeV: TensorLike) -> torch.Tensor:
        """Build the vacuum evolution operator over the baseline for ``model`` at ``energy_MeV``.

        The generator is built with the baseline as its length scale and exponentiated over a single unit step, the
        recipe every vacuum consumer in this package used inline before it lived here. Energy and baseline broadcast
        like ordinary tensors.

        Args:
            model: Built PropagationModel (mixing, mass spectrum, and any active extensions/dissipators).
            energy_MeV: Neutrino energy in MeV, any shape (a scalar included).

        Returns:
            The evolution operator, shaped ``(*batch_shape, n_flavour, n_flavour)``; ``S[..., final, initial]`` is the
            amplitude.

        Raises:
            ValueError: If ``length_m`` is not strictly positive (raised by ``profile()``).
        """
        medium_profile = self.profile()
        environment = PropagationEnvironment(
            E_MeV=as_tensor(energy_MeV, device=self.context.device, dtype=self.context.dtype),
            n_e_mol_cm3=medium_profile.n_e_mol_cm3,
        )
        generator = model.generator_flavour(environment, evolution_scale_m=self.length_m)
        return segment_evolutor_numerical(generator, 1.0)

    def propagate(
        self,
        flux: Flux,
        model: PropagationModel,
        *,
        location: Optional[str] = None,
        dilute: bool = False,
    ) -> Flux:
        """Propagate ``flux`` over the baseline and return it in its evolved state.

        The operator is ``evolutor(model, flux.energy_MeV)``, applied to ``flux.state`` (a Liouville evolution if
        ``model`` has dissipators).

        Args:
            flux: The flux to propagate, on this medium's device/dtype and with the same neutrino/antineutrino
                convention as ``model``.
            model: Built PropagationModel.
            location: ``location`` of the returned flux. If omitted, a flux leaving the source arrives at the
                ``"detector"`` (a reactor baseline), and a flux already past its source medium keeps its location
                (a vacuum leg in the middle of a chain: ``"surface"`` stays ``"surface"``).
            dilute: Treat the source as a point emitting isotropically and spread its emission rate over the sphere
                that the baseline reaches: ``differential_flux`` is divided by ``4 pi L^2`` (``L`` in cm) and its
                units go from ``... s^-1 ...`` to ``... cm^-2 s^-1 ...``. For a reactor, whose flux is an emission
                rate (``antinu s^-1 MeV^-1``); a flux that is already per unit area (the Sun's, at the Earth) is not
                diluted. ``length_m`` batches like the energies: a column of baselines ``(n_L, 1)`` against an energy
                grid ``(n_E,)`` gives a ``(n_L, n_E)`` flux.

        Returns:
            A new Flux, with ``"vacuum"`` appended to its ``history``.

        Raises:
            ValueError: If ``flux`` and ``model`` or this medium disagree (device, dtype, antinu), as
                ``evolutor``, or (``dilute``) if the flux has no ``differential_flux`` or its units are not an
                emission rate (containing ``s^-1`` and not ``cm^-2``).
        """
        check_compatible(flux, model, self.context)
        if dilute:
            if flux.differential_flux is None:
                raise ValueError("dilute=True needs a flux with a differential_flux.")
            if "s^-1" not in flux.flux_units or "cm^-2" in flux.flux_units:
                raise ValueError(
                    f"dilute=True needs an emission rate (flux_units with 's^-1' and no 'cm^-2'), got "
                    f"{flux.flux_units!r}."
                )
        operator = self.evolutor(model, flux.energy_MeV)
        propagated = evolve_flux(
            flux, model, operator,
            location=location or "detector", step="vacuum",
            keep_location=location is None and flux.location != "source",
        )
        if not dilute:
            return propagated
        length_cm = as_tensor(self.length_m, device=self.context.device, dtype=self.context.dtype) * 100.0
        area_cm2 = 4.0 * math.pi * length_cm ** 2  # batches like the evolutor (a column of baselines)
        return replace(
            propagated,
            differential_flux=propagated.differential_flux / area_cm2,
            flux_units=propagated.flux_units.replace("s^-1", "cm^-2 s^-1", 1),
        )

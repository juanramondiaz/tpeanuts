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
solar profile containers and interpolation utilities.
"""



from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch

from tpeanuts.io.io_solar import load_b16_fluxes, load_b16_solar_model
from tpeanuts.util.math import interp1d_linear, normalize_trapz


Tensor = torch.Tensor


@dataclass
class SolarProfile:
    """
    Torch representation of a solar model.
    """

    radius: Tensor
    density: Tensor
    fractions: dict[str, Tensor]
    fluxes: dict[str, Tensor]

    @property
    def device(self) -> torch.device:
        return self.radius.device

    @property
    def dtype(self) -> torch.dtype:
        return self.radius.dtype

    def electron_density(self, r_query: Tensor) -> Tensor:
        return interp1d_linear(
            x=r_query,
            xp=self.radius,
            fp=self.density,
            left=self.density[0],
            right=self.density[-1],
            device=self.device,
            dtype=self.dtype,
        )

    def production_fraction(self, source: str, r_query: Tensor | None = None) -> Tensor:
        if source not in self.fractions:
            raise KeyError(f"Unknown solar source: {source}")

        fraction = self.fractions[source]

        if r_query is None:
            return fraction

        return interp1d_linear(
            x=r_query,
            xp=self.radius,
            fp=fraction,
            left=fraction[0],
            right=fraction[-1],
            device=self.device,
            dtype=self.dtype,
        )

    def normalized_fraction(self, source: str) -> Tensor:
        return normalize_trapz(
            self.production_fraction(source),
            x=self.radius,
            clamp_min=0.0,
        )

    def flux(self, source: str) -> Tensor:
        if source not in self.fluxes:
            raise KeyError(f"Unknown solar flux source: {source}")

        return self.fluxes[source]


def load_default_solar_profile(
    *,
    device: Union[str, torch.device] | None = None,
    dtype: torch.dtype = torch.float64,
) -> SolarProfile:
    model = load_b16_solar_model(device=device, dtype=dtype)
    fluxes = load_b16_fluxes(device=device, dtype=dtype)

    return SolarProfile(
        radius=model["radius"],
        density=model["density"],
        fractions=model["fractions"],
        fluxes=fluxes,
    )

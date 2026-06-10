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
solar neutrino probabilities in the adiabatic approximation.
"""



from __future__ import annotations

from typing import Union

import torch

from tpeanuts.solar.matter_mixing import th12_M, th13_M
from tpeanuts.core.hamiltonian import _select_antinu_matrix


TensorLike = Union[float, int, torch.Tensor]


def Tei(
    th12: TensorLike,
    th13: TensorLike,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    E: TensorLike,
    ne: TensorLike,
) -> torch.Tensor:
    th13m = th13_M(th12, th13, DeltamSq21, DeltamSq3l, E, ne)
    th12m = th12_M(th12, th13, DeltamSq21, DeltamSq3l, E, ne)

    c13m = torch.cos(th13m)
    s13m = torch.sin(th13m)
    c12m = torch.cos(th12m)
    s12m = torch.sin(th12m)

    weights = torch.stack(
        [
            (c13m * c12m) ** 2,
            (c13m * s12m) ** 2,
            s13m**2,
        ],
        dim=-1,
    )
    return weights

def solar_flux_mass(
    th12: TensorLike,
    th13: TensorLike,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    E: TensorLike,
    radius_samples: torch.Tensor,
    density: torch.Tensor,
    fraction: torch.Tensor,
) -> torch.Tensor:
    E_t = torch.as_tensor(E, device=radius_samples.device, dtype=radius_samples.dtype)
    weights_r = Tei(th12, th13, DeltamSq21, DeltamSq3l, E_t[..., None], density)

    weighted = weights_r * fraction[..., None]
    norm = torch.trapz(fraction, x=radius_samples, dim=-1)
    integral = torch.trapz(weighted, x=radius_samples, dim=-2)

    return integral / torch.clamp(norm, min=torch.finfo(radius_samples.dtype).tiny)[..., None]


def solar_flux_mass_sources(
    th12: TensorLike,
    th13: TensorLike,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    E: TensorLike,
    radius_samples: torch.Tensor,
    density: torch.Tensor,
    fractions: torch.Tensor,
) -> torch.Tensor:
    E_t = torch.as_tensor(E, device=radius_samples.device, dtype=radius_samples.dtype)
    weights_r = Tei(th12, th13, DeltamSq21, DeltamSq3l, E_t[..., None], density)

    source_shape = fractions.shape[:-1]
    energy_ndim = E_t.ndim

    fractions_lifted = fractions.reshape(
        *source_shape,
        *((1,) * energy_ndim),
        fractions.shape[-1],
    )

    weights_lifted = weights_r.reshape(
        *((1,) * len(source_shape)),
        *weights_r.shape,
    )

    weighted = weights_lifted * fractions_lifted[..., None]
    norm = torch.trapz(fractions, x=radius_samples, dim=-1)
    integral = torch.trapz(weighted, x=radius_samples, dim=-2)

    norm_lifted = norm.reshape(
        *source_shape,
        *((1,) * energy_ndim),
    )

    return integral / torch.clamp(norm_lifted, min=torch.finfo(radius_samples.dtype).tiny)[..., None]


def psolar(
    pmns: object,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    E: TensorLike,
    radius_samples: torch.Tensor,
    density: torch.Tensor,
    fraction: torch.Tensor,
    *,
    antinu: Union[bool, torch.Tensor] = False,
) -> torch.Tensor:
    weights = solar_flux_mass(
        pmns.theta12,
        pmns.theta13,
        DeltamSq21,
        DeltamSq3l,
        E,
        radius_samples,
        density,
        fraction,
    )

    U = pmns.pmns_matrix().to(device=radius_samples.device)
    U = _select_antinu_matrix(U, antinu)
    probs_i_to_alpha = torch.abs(torch.conj(U)) ** 2

    return torch.einsum("...ai,...i->...a", probs_i_to_alpha, weights)


def psolar_sources(
    pmns: object,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    E: TensorLike,
    radius_samples: torch.Tensor,
    density: torch.Tensor,
    fractions: torch.Tensor,
    *,
    antinu: Union[bool, torch.Tensor] = False,
) -> torch.Tensor:
    weights = solar_flux_mass_sources(
        pmns.theta12,
        pmns.theta13,
        DeltamSq21,
        DeltamSq3l,
        E,
        radius_samples,
        density,
        fractions,
    )

    U = pmns.pmns_matrix().to(device=radius_samples.device)
    U = _select_antinu_matrix(U, antinu)
    probs_i_to_alpha = torch.abs(torch.conj(U)) ** 2

    return torch.einsum("...ai,...i->...a", probs_i_to_alpha, weights)

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
Generic flux utilities built on top of probability tensors.

This module contains medium-independent helpers that convert flavour
probabilities into flavour-resolved fluxes by applying source normalizations
and optional spectral weights. It does not construct probabilities or know
about solar, Atmosphere, Earth, or vacuum propagation models.

Module functions:
    flux_from_probability(...)
        Multiply final flavour probabilities by a flux normalization and an
        optional spectrum, preserving the final flavour dimension.
"""



from __future__ import annotations

import torch


def _lift_to_probability_shape(value: torch.Tensor, probability: torch.Tensor) -> torch.Tensor:
    """Append singleton axes so value broadcasts against a probability tensor."""
    if value.ndim > probability.ndim:
        raise ValueError("value cannot have more dimensions than probability.")

    return value.reshape(*value.shape, *((1,) * (probability.ndim - value.ndim)))


def flux_from_probability(
    probability: torch.Tensor,
    flux: torch.Tensor | float,
    spectrum: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Build a flavour-resolved flux from final flavour probabilities.

    Args:
        probability: Final flavour probabilities shaped ``(..., 3)``.
        flux: Flux normalization broadcastable with the leading probability
            dimensions. Scalars, one value per source, or full grids are
            accepted.
        spectrum: Optional spectral weight broadcastable with the leading
            probability dimensions.

    Returns:
        Flavour-resolved flux with the same shape as ``probability``.

    Raises:
        ValueError: If ``probability`` does not have final flavour dimension 3
            or if ``flux``/``spectrum`` have too many dimensions.
    """
    if probability.shape[-1] != 3:
        raise ValueError("probability must have final flavour dimension 3.")

    flux_t = torch.as_tensor(flux, device=probability.device, dtype=probability.dtype)
    out = probability * _lift_to_probability_shape(flux_t, probability)

    if spectrum is not None:
        spectrum_t = torch.as_tensor(
            spectrum,
            device=probability.device,
            dtype=probability.dtype,
        )
        out = out * _lift_to_probability_shape(spectrum_t, probability)

    return out

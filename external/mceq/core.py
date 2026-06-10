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
core mceq initialization utilities.

This module contains only the logic required to create and configure
an mceqRun instance.
"""



from __future__ import annotations

import logging
from typing import Optional, Union

import torch

from tpeanuts.util.type import _as_tensor

from tpeanuts.external.mceq.config import (
    MCEqModelConfig,
    PRIMARY_MODELS,
    DENSITY_MODELS,
    DEFAULT_INTERACTION_MODEL,
    DEFAULT_PRIMARY_MODEL,
    DEFAULT_DENSITY_MODEL,
)


try:
    from MCEq.core import MCEqRun as mceqRun
except ImportError:
    mceqRun = None


MCEQ_IMPORT_ERROR = (
    "mceq could not be imported. Install the optional MCEq dependencies in "
    "the same Python environment before using this workflow. For this "
    "project, run: python -m pip install -e .[mceq]"
)


def ensure_mceq_available() -> None:
    if mceqRun is None:
        raise ImportError(MCEQ_IMPORT_ERROR)


TensorLike = Union[float, int, torch.Tensor]


# ============================================================
# Model resolvers
# ============================================================

def resolve_primary_model(
    primary_model: Optional[Union[str, tuple]] = DEFAULT_PRIMARY_MODEL,
):
    if primary_model is None:
        primary_model = DEFAULT_PRIMARY_MODEL

    if isinstance(primary_model, str):
        if primary_model not in PRIMARY_MODELS:
            raise ValueError(
                f"Unknown primary_model='{primary_model}'. "
                f"Available: {list(PRIMARY_MODELS.keys())}"
            )

        return PRIMARY_MODELS[primary_model]

    return primary_model


def resolve_density_model(
    density_model: Optional[str] = DEFAULT_DENSITY_MODEL,
):
    if density_model is None:
        density_model = DEFAULT_DENSITY_MODEL

    if density_model not in DENSITY_MODELS:
        raise ValueError(
            f"Unknown density_model='{density_model}'. "
            f"Available: {list(DENSITY_MODELS.keys())}"
        )

    return DENSITY_MODELS[density_model]


def theta_to_float(theta_deg: TensorLike) -> float:
    theta_t = _as_tensor(theta_deg, device="cpu", dtype=torch.float64)
    theta_float = float(theta_t.detach().cpu().reshape(-1)[0].item())

    if theta_float < 0.0 or theta_float >= 90.0:
        raise ValueError(
            f"theta_deg must satisfy 0 <= theta_deg < 90. "
            f"Received theta_deg={theta_float}."
        )

    return theta_float


def set_mceq_logging(info: bool = False) -> None:
    if info:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.ERROR)


# ============================================================
# Init mceq
# ============================================================

def init_mceq(
    theta_deg: TensorLike,
    config: Optional[MCEqModelConfig] = None,
    interaction_model: Optional[str] = None,
    primary_model: Optional[Union[str, tuple]] = None,
    density_model: Optional[str] = None,
    info: Optional[bool] = None,
):
    ensure_mceq_available()

    if config is None:
        config = MCEqModelConfig()

    interaction_model = (
        config.interaction_model
        if interaction_model is None
        else interaction_model
    )

    primary_model = (
        config.primary_model
        if primary_model is None
        else primary_model
    )

    density_model = (
        config.density_model
        if density_model is None
        else density_model
    )

    info = config.info if info is None else info

    effective_config = MCEqModelConfig(
        interaction_model=interaction_model,
        primary_model=primary_model,
        density_model=density_model,
        info=info,
    )

    effective_config.validate()

    theta_float = theta_to_float(theta_deg)

    primary_model_obj = resolve_primary_model(effective_config.primary_model)
    density_model_obj = resolve_density_model(effective_config.density_model)

    set_mceq_logging(effective_config.info)

    mceq = mceqRun(
        interaction_model=effective_config.interaction_model,
        primary_model=primary_model_obj,
        theta_deg=theta_float,
    )

    mceq.set_density_model(density_model_obj)

    return mceq


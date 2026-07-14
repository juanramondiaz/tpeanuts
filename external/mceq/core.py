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
an mceqRun instance (an alias for MCEq.core.MCEqRun, the external MCEq
package's cascade-equation solver object). It is the single place in
tpeanuts where an MCEqRun is constructed: init_mceq() resolves the
tpeanuts-native model names stored in MCEqModelConfig into the actual
MCEq/crflux objects, instantiates
MCEqRun for a given zenith angle, and attaches an Atmosphere density
model to it. Everything else in this module
(resolve_primary_model/resolve_density_model/set_mceq_logging/
ensure_mceq_available) is thin tpeanuts-native
configuration plumbing around that one external call.

Module functions:
    ensure_mceq_available:
        Raise ImportError with an actionable message if the optional
        MCEq dependency was not installed.
    resolve_primary_model:
        Resolve a tpeanuts primary-model name (or raw tuple) into the
        crflux model object/tag pair expected by MCEqRun.
    resolve_density_model:
        Resolve a tpeanuts density-model name into the
        (model_name, (location, season)) tuple expected by
        MCEqRun.set_density_model.
    set_mceq_logging:
        Set the verbosity of MCEq's internal Python logger.
    init_mceq:
        Build and return a fully configured MCEqRun instance for a
        given zenith angle and physics-model selection. This is the
        only function in this module that directly constructs an
        external MCEq object.
"""



from __future__ import annotations

import logging
from typing import Optional, Union

import torch

from tpeanuts.util.torch_util import scalar_float

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
    """
    Raise an actionable ImportError if the optional MCEq package is
    unavailable in the current Python environment.

    This is tpeanuts-native plumbing: it does not call MCEq itself, it
    only checks whether the earlier `from MCEq.core import MCEqRun`
    import at module load time succeeded.

    Raises:
        ImportError: If MCEq.core.MCEqRun could not be imported,
            instructing the user to install the optional mceq extras.
    """
    if mceqRun is None:
        raise ImportError(MCEQ_IMPORT_ERROR)


TensorLike = Union[float, int, torch.Tensor]


# ============================================================
# Model resolvers
# ============================================================

def resolve_primary_model(
    primary_model: Optional[Union[str, tuple]] = DEFAULT_PRIMARY_MODEL,
):
    """
    Resolve a tpeanuts primary cosmic-ray model name into the crflux
    object/tag pair expected by the MCEqRun constructor.

    The primary model is the cosmic-ray flux injected at the top of the
    atmosphere (the boundary condition for MCEq's cascade-equation
    solve), expressed as all-particle/per-nucleus flux versus energy.
    Physical model choices (e.g. Hillas-Gaisser, Gaisser-Stanev-Tilav)
    differ in their assumed cosmic-ray source/propagation physics and
    therefore in the resulting atmospheric secondary flux.

    Args:
        primary_model: Either a key of PRIMARY_MODELS (a human-readable
            tpeanuts name such as "HillasGaisser H3a"), a raw
            (model_class, model_tag) tuple already in the form MCEq
            expects, or None to fall back to DEFAULT_PRIMARY_MODEL.

    Returns:
        A (model_class, model_tag) tuple from the crflux.models module,
        ready to be passed as primary_model to MCEqRun.

    Raises:
        ValueError: If primary_model is a string not found in
            PRIMARY_MODELS.
    """
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
    """
    Resolve a tpeanuts Atmosphere density-model name into the
    (model_name, (location, season)) tuple expected by
    MCEqRun.set_density_model.

    The density model determines the Atmosphere mass-density profile
    rho(h) (and hence the altitude <-> slant-depth mapping X(h, alpha))
    used by MCEq's cascade-equation solve; different choices represent
    different site/season Atmosphere conditions (e.g. US Standard
    Atmosphere, NRLMSISE-00 at the South Pole).

    Args:
        density_model: A key of DENSITY_MODELS (e.g. "CORSIKA", "NASA",
            "ICECUBE", "ISOTHERMAL"), or None to fall back to
            DEFAULT_DENSITY_MODEL.

    Returns:
        A (model_name, (location, season)) tuple as required by
        MCEqRun.set_density_model.

    Raises:
        ValueError: If density_model is not a key of DENSITY_MODELS.
    """
    if density_model is None:
        density_model = DEFAULT_DENSITY_MODEL

    if density_model not in DENSITY_MODELS:
        raise ValueError(
            f"Unknown density_model='{density_model}'. "
            f"Available: {list(DENSITY_MODELS.keys())}"
        )

    return DENSITY_MODELS[density_model]


def set_mceq_logging(info: bool = False) -> None:
    """
    Set the verbosity of MCEq's internal Python logger.

    Args:
        info: If True, set the root logger level to INFO (MCEq prints
            verbose diagnostic messages during initialization and
            solving). If False, restrict it to ERROR level to suppress
            MCEq's routine console output.
    """
    if info:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.ERROR)


# ============================================================
# Init mceq
# ============================================================

def init_mceq(
    alpha_deg: TensorLike,
    config: Optional[MCEqModelConfig] = None,
    info: Optional[bool] = None,
):
    """
    Build and return a fully configured MCEqRun instance.

    This is the single entry point in tpeanuts that directly constructs
    an external MCEq object: it instantiates MCEq.core.MCEqRun for the
    requested surface zenith angle, hadronic interaction model and primary
    cosmic-ray flux model, then attaches the requested Atmosphere
    density model via MCEqRun.set_density_model. The returned object
    encapsulates the cascade-equation transport problem (production and
    propagation of secondary hadrons, muons and neutrinos through the
    atmosphere) and can subsequently be solved (mceq.solve(...)) and
    queried (mceq.get_solution(...)) by the solver module.

    Args:
        alpha_deg: Surface/MCEq zenith angle in degrees
            (0 <= alpha_deg < 90) of the shower/neutrino trajectory;
            alpha_deg=0 is vertical.
        config: Optional MCEqModelConfig providing interaction_model,
            primary_model, density_model and info.
        info: Optional override for MCEq logger verbosity; defaults to
            config.info.

    Returns:
        An initialized MCEq.core.MCEqRun instance (returned under the
        local alias mceqRun) for the given surface zenith angle, with its
        density model already set, ready to be solved.

    Raises:
        ImportError: If the optional MCEq package is not installed (via
            ensure_mceq_available).
        ValueError: If the resolved interaction_model, primary_model or
            density_model is not recognised, or if alpha_deg is outside
            [0, 90) degrees.
    """
    if config is None:
        config = MCEqModelConfig()

    info = config.info if info is None else info

    effective_config = MCEqModelConfig(
        interaction_model=config.interaction_model,
        primary_model=config.primary_model,
        density_model=config.density_model,
        info=info,
    )

    effective_config.validate()

    alpha_float = scalar_float(alpha_deg)

    if alpha_float < 0.0 or alpha_float >= 90.0:
        raise ValueError(
            f"alpha_deg must satisfy 0 <= alpha_deg < 90. "
            f"Received alpha_deg={alpha_float}."
        )

    ensure_mceq_available()

    primary_model_obj = resolve_primary_model(effective_config.primary_model)
    density_model_obj = resolve_density_model(effective_config.density_model)

    set_mceq_logging(effective_config.info)

    mceq = mceqRun(
        interaction_model=effective_config.interaction_model,
        primary_model=primary_model_obj,
        theta_deg=alpha_float,
    )

    mceq.set_density_model(density_model_obj)

    return mceq

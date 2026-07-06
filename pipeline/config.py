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
Single configuration object shared by every tpeanuts pipeline.

PropagationConfig composes the runtime, oscillation, exposure, and
per-medium profile settings that used to travel as 20-38 separate keyword
arguments into each pipeline entry point. There is one PropagationConfig
class for the whole project, not one per pipeline: fields that a given
pipeline does not need (e.g. ``exposure``/``solar`` for the atmosphere
pipeline) are simply left at their defaults and unused.

Pipeline functions destructure this object into the smaller pieces that
``medium.*``/``core.*`` functions actually need (``RuntimeContext``,
``OscillationParameters``, a built profile object, ...) rather than passing
the whole config down — internal layers stay decoupled from the
pipeline-level object.

Module contents:
    PropagationConfig
        Composed configuration for solar, Earth, and atmosphere pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.medium.atmosphere.profile import AtmosphereParameters
from tpeanuts.medium.earth.exposure_table import ExposureParameters
from tpeanuts.medium.earth.profile import EarthParameters
from tpeanuts.medium.solar.profile import SolarParameters
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import TensorLike

ProductionMode = Literal["point", "coherent", "incoherent"]


@dataclass
class PropagationConfig:
    """
    Single configuration object shared by every tpeanuts pipeline.

    Parameters
    ----------
    runtime:
        Torch device/dtype used throughout the computation.

    oscillation:
        Built pmns object plus mass splittings and antinu selection.

    exposure:
        Nadir-exposure table settings. Used by the solar pipelines that
        integrate over a detector exposure window; unused by the
        atmosphere/vacuum pipelines.

    earth:
        Earth electron-density profile construction settings. Used by every
        pipeline that includes an Earth-crossing stage.

    solar:
        Solar model construction settings. Used by the solar pipelines only.

    atmosphere:
        Atmosphere density profile construction settings. Used by the
        atmosphere pipeline only.

    production_mode:
        Solar production treatment: "point", "coherent", or "incoherent"
        (solar pipelines only).

    rho0:
        Production radius for ``production_mode="point"``, as a fraction of
        the solar radius (solar pipelines only).

    source:
        Solar neutrino source key (e.g. "8B", "pp") selecting the
        production-radius distribution for ``production_mode`` in
        {"coherent", "incoherent"} (solar pipelines only).

    earth_distance_km:
        Explicit Sun-Earth distance in km. None selects the tabulated or
        constant fallback (solar pipelines only).

    sun_earth_distance_path:
        Optional path to the tabulated Sun-Earth distance file (solar
        pipelines only).

    use_sun_earth_distance_table:
        If True and ``earth_distance_km`` is None, use the tabulated
        distance mean instead of the constant fallback (solar pipelines
        only).

    detector_depth_m:
        Detector depth below the Earth's surface, in metres. Shifts the
        Earth-crossing path length for every pipeline that includes an
        Earth-regeneration stage.

    reunitarize_earth:
        If True, project the Earth evolution operator back onto the nearest
        unitary matrix to absorb small numerical drift.
    """

    runtime: RuntimeContext
    oscillation: OscillationParameters
    exposure: ExposureParameters = field(default_factory=ExposureParameters)
    earth: EarthParameters = field(default_factory=EarthParameters)
    solar: SolarParameters = field(default_factory=SolarParameters)
    atmosphere: AtmosphereParameters = field(default_factory=AtmosphereParameters)
    production_mode: ProductionMode = "point"
    rho0: Optional[TensorLike] = None
    source: Optional[str] = None
    earth_distance_km: Optional[TensorLike] = None
    sun_earth_distance_path: Optional[str] = None
    use_sun_earth_distance_table: bool = True
    detector_depth_m: float = 0.0
    reunitarize_earth: bool = False

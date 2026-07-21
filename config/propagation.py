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
        ``oscillation_parameters_from_preset`` is a static factory building
        the ``oscillation`` field (an ``OscillationParameters``) from a named
        preset; it is the composition layer allowed to depend on concrete
        PMNS, sterile, NSI, and preset implementations, keeping
        ``OscillationParameters`` itself a passive domain container.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

import torch

from tpeanuts.config.presets import OSCILLATION_PRESETS, get_preset
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.common.pmns import PMNSParams
from tpeanuts.medium.atmosphere.profile import AtmosphereParameters
from tpeanuts.medium.earth.exposure_table import ExposureParameters
from tpeanuts.medium.earth.profile import EarthParameters
from tpeanuts.medium.solar.profile import SolarParameters
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import TensorLike, as_tensor

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
    detector_depth_m: float = 0.0
    reunitarize_earth: bool = False

    @staticmethod
    def oscillation_parameters_from_preset(
        preset_name: str = "_SM_NUFIT52_NO",
        *,
        antinu: Union[bool, torch.Tensor] = False,
        NSI_extension: Optional[str] = None,
        context: Optional[RuntimeContext] = None,
    ) -> OscillationParameters:
        """Build a concrete SM/BSM ``OscillationParameters`` from a named preset.

        Looks up ``preset_name`` in ``tpeanuts.config.presets.
        OSCILLATION_PRESETS``. A preset is a 3+1 sterile preset iff it
        carries a ``theta14_deg`` key, in which case a ``PMNS_sterile`` and a
        ``MassSpectrum_BSM`` (carrying ``DeltamSq41``) are built; otherwise a
        plain 3-flavour ``PMNS_SM`` and a ``MassSpectrum_SM`` are built.
        Concrete SM/BSM/NSI/preset implementations are imported lazily to
        avoid an import-time dependency from this module onto them.

        Args:
            preset_name: Preset name. Defaults to ``"_SM_NUFIT52_NO"``.
            antinu: Bool or boolean tensor selecting antineutrino
                propagation.
            NSI_extension: Optional name of an NSI preset (see
                ``tpeanuts.config.presets.NSI_PRESETS``, consumed via
                ``tpeanuts.core.BSM.bsm_nsi.NSIConfig.from_preset``).
                Oscillation presets carry no NSI information themselves, so
                when given, this builds the ``NSIConfig`` on ``context``'s
                device/dtype (populating its ``epsilon`` tensor), and stores
                it as ``oscillation.nsi``. None (default) means the plain
                Standard Model matter potential.
            context: Runtime context for stored tensors. If omitted, the
                default device and ``torch.float64`` are used.

        Returns:
            OscillationParameters built from the preset registry, with
            ``preset_name``, ``ordering``, ``label``, and ``description``
            set from the preset.

        Raises:
            ValueError: If ``preset_name`` or ``NSI_extension`` is unknown.
        """
        if context is None:
            context = RuntimeContext.resolve(None, torch.float64)

        nsi_obj = None
        if NSI_extension is not None:
            from tpeanuts.core.BSM.bsm_nsi import NSIConfig

            nsi_obj = NSIConfig.from_preset(
                NSI_extension, device=context.device, real_dtype=context.dtype,
            )

        preset = get_preset(OSCILLATION_PRESETS, preset_name, kind="oscillation preset")

        sm_params = PMNSParams(
            theta12=math.radians(float(preset["theta12_deg"])),
            theta13=math.radians(float(preset["theta13_deg"])),
            theta23=math.radians(float(preset["theta23_deg"])),
            delta=math.radians(float(preset["delta13_deg"])),
            context=context,
        )

        dm21 = as_tensor(float(preset["DeltamSq21"]), device=context.device, dtype=context.dtype)
        dm3l = as_tensor(float(preset["DeltamSq3l"]), device=context.device, dtype=context.dtype)

        if "theta14_deg" in preset:
            from tpeanuts.core.BSM.bsm_sterile import PMNSSterileParams, PMNS_sterile
            from tpeanuts.core.BSM.bsm_mass_spectrum import MassSpectrum_BSM

            sterile_params = PMNSSterileParams(
                theta14=math.radians(float(preset["theta14_deg"])),
                theta24=math.radians(float(preset["theta24_deg"])),
                theta34=math.radians(float(preset["theta34_deg"])),
                delta14=math.radians(float(preset["delta14_deg"])),
                delta24=math.radians(float(preset["delta24_deg"])),
                delta34=0.0,   # always zero: R34 is real for Dirac 3+1 neutrinos
                context=context,
            )

            pmns_obj = PMNS_sterile(sm_params, sterile_params)
            dm41 = as_tensor(float(preset["DeltamSq41"]), device=context.device, dtype=context.dtype)
            mass_spectrum = MassSpectrum_BSM(DeltamSq21=dm21, DeltamSq3l=dm3l, DeltamSq41=dm41)
        else:
            from tpeanuts.core.SM.sm_pmns import PMNS_SM
            from tpeanuts.core.SM.sm_mass_spectrum import MassSpectrum_SM

            pmns_obj = PMNS_SM(sm_params)
            mass_spectrum = MassSpectrum_SM(DeltamSq21=dm21, DeltamSq3l=dm3l)

        return OscillationParameters(
            pmns=pmns_obj,
            mass_spectrum=mass_spectrum,
            antinu=antinu,
            nsi=nsi_obj,
            preset_name=preset_name,
            ordering=str(preset.get("ordering", "")),
            label=str(preset.get("label", "")),
            description=str(preset.get("description", "")),
        )

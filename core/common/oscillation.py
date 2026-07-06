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
Bundled oscillation inputs shared by core, medium, coherent, and pipeline code.

OscillationParameters replaces the (pmns, DeltamSq21, DeltamSq3l, antinu) group
that previously travelled as four separate arguments through almost every
propagation function. It always stores an already-built PMNS-compatible
object, so the raw mixing angles (theta12, theta13, theta23, delta) no longer
need to be threaded past the point where the mixing matrix is constructed.

Mass-squared splittings and named-preset metadata live here rather than on
``PMNSParams``: the PMNS mixing matrix itself does not depend on
DeltamSq21/DeltamSq3l, so keeping them on the full oscillation state avoids
storing the same values twice.

Named presets (both 3-flavour SM and 3+1 sterile) live in
``tpeanuts.core.common.presets.OSCILLATION_PRESETS`` — this module only
consumes that registry via ``from_preset``; it does not define preset data
itself.

Module contents:
    OscillationParameters
        Frozen container for the built pmns object, both mass-squared
        splittings, and the antineutrino selector. ``from_preset(...)``
        builds one directly from a named oscillation preset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

import torch

from tpeanuts.core.common.pmns import PMNSParams
from tpeanuts.core.common.presets import OSCILLATION_PRESETS, get_preset
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import TensorLike, as_tensor


@dataclass(frozen=True)
class OscillationParameters:
    """
    Complete set of standard 3-flavour oscillation inputs.

    Parameters
    ----------
    pmns:
        Built PMNS-compatible mixing object (``PMNS`` or a subclass such as
        ``PMNS_sterile``). Supplies the mixing angles theta12/theta13/theta23
        and the CP phase delta already packed into rotation matrices, so
        downstream physics functions never need the raw angles again.

    DeltamSq21:
        Solar mass-squared splitting Delta m^2_21 in eV^2. Sets the
        oscillation frequency of the "slow", solar-driven term.

    DeltamSq3l:
        Atmospheric mass-squared splitting Delta m^2_3l in eV^2
        (Delta m^2_31 for normal ordering, Delta m^2_32 for inverted
        ordering, by sign convention). Sets the oscillation frequency of the
        "fast", atmospheric-driven term.

    DeltamSq41:
        Sterile mass-squared splitting Delta m^2_41 in eV^2, for 3+1
        ``PMNS_sterile`` objects. Like ``DeltamSq21``/``DeltamSq3l``, it
        does not enter any mixing-matrix rotation (only the active-sterile
        mixing angles do), so it lives here rather than on
        ``PMNSSterileParams``. None for plain 3-flavour ``PMNS_SM`` objects.

    antinu:
        Bool or boolean tensor selecting antineutrino propagation. Where
        True, the mixing matrix and matter potential use the
        charge-conjugate convention (U -> U*, V -> -V).

    preset_name:
        Name of the named oscillation preset used to build this object via
        ``from_preset``, or "custom" when built from explicit values.

    ordering:
        Mass ordering label carried by the preset, for example "NO" or
        "IO". Empty when built from explicit values.

    label:
        Short identifier string carried by the preset (mirrors
        ``NSIConfig``'s ``label`` field). Empty when built from explicit
        values.

    description:
        Human-readable description and literature reference carried by the
        preset. Empty when built from explicit values.
    """

    pmns: object
    DeltamSq21: torch.Tensor
    DeltamSq3l: torch.Tensor
    DeltamSq41: Optional[torch.Tensor] = None
    antinu: Union[bool, torch.Tensor] = False
    preset_name: str = "custom"
    ordering: str = ""
    label: str = ""
    description: str = ""

    @classmethod
    def build(
        cls,
        *,
        pmns: Optional[object] = None,
        theta12: Optional[TensorLike] = None,
        theta13: Optional[TensorLike] = None,
        theta23: Optional[TensorLike] = None,
        delta: Optional[TensorLike] = None,
        DeltamSq21: TensorLike,
        DeltamSq3l: TensorLike,
        DeltamSq41: Optional[TensorLike] = None,
        antinu: Union[bool, torch.Tensor] = False,
        context: RuntimeContext,
    ) -> "OscillationParameters":
        """Build from an existing pmns object or from raw mixing angles.

        Args:
            pmns: Existing PMNS-compatible object. If provided, the mixing
                angles below are ignored.
            theta12: Solar mixing angle in radians, used when ``pmns`` is
                omitted.
            theta13: Reactor mixing angle in radians, used when ``pmns``
                is omitted.
            theta23: Atmospheric mixing angle in radians, used when ``pmns``
                is omitted.
            delta: CP-violating phase in radians, used when ``pmns`` is
                omitted.
            DeltamSq21: Solar mass-squared splitting in eV^2.
            DeltamSq3l: Atmospheric mass-squared splitting in eV^2.
            DeltamSq41: Sterile mass-squared splitting in eV^2. Required for
                BSM Hamiltonian construction when ``pmns`` is a 3+1
                ``PMNS_sterile`` object; ignored for 3-flavour ``PMNS_SM``.
            antinu: Bool or boolean tensor selecting antineutrino
                propagation.
            context: Runtime device/dtype used to build a new pmns object and
                to cast the mass splittings.

        Returns:
            OscillationParameters with a built pmns object and tensor mass
            splittings on ``context.device``/``context.dtype``.
        """
        if pmns is not None:
            pmns_obj = pmns
        else:
            missing = [
                name
                for name, value in {
                    "theta12": theta12,
                    "theta13": theta13,
                    "theta23": theta23,
                    "delta": delta,
                }.items()
                if value is None
            ]
            if missing:
                raise ValueError(
                    "Provide pmns or all PMNS angles: " + ", ".join(missing)
                )
            params = PMNSParams(
                theta12=as_tensor(theta12, device=context.device, dtype=context.dtype),
                theta13=as_tensor(theta13, device=context.device, dtype=context.dtype),
                theta23=as_tensor(theta23, device=context.device, dtype=context.dtype),
                delta=as_tensor(delta, device=context.device, dtype=context.dtype),
                context=context,
            )

            from tpeanuts.core.SM.pmns import PMNS_SM

            pmns_obj = PMNS_SM(params)

        return cls(
            pmns=pmns_obj,
            DeltamSq21=as_tensor(DeltamSq21, device=context.device, dtype=context.dtype),
            DeltamSq3l=as_tensor(DeltamSq3l, device=context.device, dtype=context.dtype),
            DeltamSq41=(
                None if DeltamSq41 is None
                else as_tensor(DeltamSq41, device=context.device, dtype=context.dtype)
            ),
            antinu=antinu,
        )

    @classmethod
    def from_preset(
        cls,
        preset_name: str = "_SM_NUFIT52_NO",
        *,
        antinu: Union[bool, torch.Tensor] = False,
        context: Optional[RuntimeContext] = None,
    ) -> "OscillationParameters":
        """Build oscillation parameters from a named preset.

        Looks up ``preset_name`` in ``tpeanuts.core.common.presets.
        OSCILLATION_PRESETS``. A preset is a 3+1 sterile preset iff it
        carries a ``theta14_deg`` key, in which case a ``PMNS_sterile`` is
        built from a ``PMNSParams`` (SM sector) and a ``PMNSSterileParams``
        (sterile extension), imported lazily to avoid a ``common -> BSM``
        import-time dependency; otherwise a plain 3-flavour ``PMNS_SM`` is
        built from a ``PMNSParams``.

        Args:
            preset_name: Preset name. Defaults to ``"_SM_NUFIT52_NO"``.
            antinu: Bool or boolean tensor selecting antineutrino
                propagation.
            context: Runtime context for stored tensors. If omitted, the
                default device and ``torch.float64`` are used.

        Returns:
            OscillationParameters built from the preset registry, with
            ``preset_name``, ``ordering``, ``label``, and ``description``
            set from the preset.

        Raises:
            ValueError: If ``preset_name`` is unknown.
        """
        if context is None:
            context = RuntimeContext.resolve(None, torch.float64)

        preset = get_preset(OSCILLATION_PRESETS, preset_name, kind="oscillation preset")

        sm_params = PMNSParams(
            theta12=math.radians(float(preset["theta12_deg"])),
            theta13=math.radians(float(preset["theta13_deg"])),
            theta23=math.radians(float(preset["theta23_deg"])),
            delta=math.radians(float(preset["delta13_deg"])),
            context=context,
        )

        DeltamSq41 = None
        if "theta14_deg" in preset:
            from tpeanuts.core.BSM.PMNS_sterile import PMNSSterileParams, PMNS_sterile

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
            DeltamSq41 = as_tensor(float(preset["DeltamSq41"]), device=context.device, dtype=context.dtype)
        else:
            from tpeanuts.core.SM.pmns import PMNS_SM

            pmns_obj = PMNS_SM(sm_params)

        return cls(
            pmns=pmns_obj,
            DeltamSq21=as_tensor(float(preset["DeltamSq21"]), device=context.device, dtype=context.dtype),
            DeltamSq3l=as_tensor(float(preset["DeltamSq3l"]), device=context.device, dtype=context.dtype),
            DeltamSq41=DeltamSq41,
            antinu=antinu,
            preset_name=preset_name,
            ordering=str(preset.get("ordering", "")),
            label=str(preset.get("label", "")),
            description=str(preset.get("description", "")),
        )

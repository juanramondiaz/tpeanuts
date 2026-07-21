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

The mass-squared splittings live in a ``MassSpectrum`` object
(``tpeanuts.core.common.mass_spectrum.MassSpectrum``) rather than as individual
fields here, and rather than on ``PMNSParams``: the PMNS mixing matrix itself
does not depend on DeltamSq21/DeltamSq3l, so keeping them off it avoids
storing the same values twice. ``mass_spectrum`` is a
``tpeanuts.core.SM.sm_mass_spectrum.MassSpectrum_SM`` for a plain 3-flavour
``pmns``, or a ``tpeanuts.core.BSM.bsm_mass_spectrum.MassSpectrum_BSM`` (carrying
``DeltamSq41``) for a 3+1 sterile ``pmns``. Both are assembled by the
factories in ``tpeanuts.config.oscillation`` according to
``pmns.n_flavours``.

Named presets (both 3-flavour SM and 3+1 sterile) live in
``tpeanuts.config.presets.OSCILLATION_PRESETS``. This module neither imports
nor consumes that registry.

Module contents:
    OscillationParameters
        Frozen container for the built PMNS object, the mass spectrum, the
        antineutrino selector, and optional NSI state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch

if TYPE_CHECKING:
    from tpeanuts.core.common.mass_spectrum import MassSpectrum


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

    mass_spectrum:
        Built ``MassSpectrum`` object (``MassSpectrum_SM`` for a plain
        3-flavour ``pmns``, ``MassSpectrum_BSM`` for a 3+1 sterile ``pmns``)
        carrying ``DeltamSq21``/``DeltamSq3l`` (and ``DeltamSq41`` for the
        sterile case) plus the model-specific ``difference_vector()``
        recipe consumed by ``tpeanuts.core.common.hamiltonian.
        kinetic_eigenvalue_vector``.

    antinu:
        Bool or boolean tensor selecting antineutrino propagation. Where
        True, the mixing matrix and matter potential use the
        charge-conjugate convention (U -> U*, V -> -V).

    nsi:
        Optional ``NSIConfig`` instance (``tpeanuts.core.BSM.bsm_nsi``)
        carrying the Non-Standard Interaction coupling matrix. ``None`` means
        the plain Standard Model matter potential (no NSI). When set, the
        epsilon coupling matrix is read as ``oscillation.nsi.epsilon`` —
        this is the single place NSI configuration lives; ``core.BSM``,
        ``core.numerical``, and ``core.perturbative`` all read it from here
        rather than taking a separate ``epsilon`` argument. Build it via
        ``PropagationConfig.oscillation_parameters_from_preset``'s
        ``NSI_extension`` argument, or pass an explicit ``NSIConfig`` (with
        ``epsilon`` already populated by ``NSIConfig.from_preset`` or
        ``epsilon_tensor()``) to the constructor directly.

    BSM_extension_sterile:
        Read-only property (not a stored field): True iff ``pmns`` is a 3+1
        ``PMNS_sterile`` object (``pmns.n_flavours == 4``). Auto-derived so
        it can never disagree with the mixing object actually being used.

    BSM_extension_NSI:
        Read-only property (not a stored field): True iff ``nsi`` is not
        None. Auto-derived so it can never disagree with ``nsi``.

    BSM_extension:
        Read-only property (not a stored field): True iff either
        ``BSM_extension_NSI`` or ``BSM_extension_sterile`` is True, i.e. any
        BSM extension is active. False means the plain Standard Model is
        used. This is the flag ``core.BSM``/``core.numerical``/
        ``core.perturbative`` check to route between the SM-only fast path
        and the general BSM path.

    preset_name:
        Name of the named oscillation preset used to build this object via
        ``PropagationConfig.oscillation_parameters_from_preset``, or "custom"
        when assembled from explicit values.

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
    mass_spectrum: "MassSpectrum"
    antinu: Union[bool, torch.Tensor] = False
    nsi: Optional[object] = None
    preset_name: str = "custom"
    ordering: str = ""
    label: str = ""
    description: str = ""

    @property
    def BSM_extension_sterile(self) -> bool:
        """True iff ``pmns`` is a 3+1 sterile object (``n_flavours == 4``).

        Auto-derived from the mixing object itself rather than stored
        separately, so it is always in sync with ``pmns`` and cannot be set
        inconsistently.
        """
        return int(self.pmns.n_flavours) == 4

    @property
    def BSM_extension_NSI(self) -> bool:
        """True iff ``nsi`` is set. Auto-derived so it cannot disagree with it."""
        return self.nsi is not None

    @property
    def BSM_extension(self) -> bool:
        """True iff any BSM extension (NSI and/or sterile) is active."""
        return self.BSM_extension_NSI or self.BSM_extension_sterile

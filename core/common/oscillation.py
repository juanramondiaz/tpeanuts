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
    resolve_include_matter_nc(...)
        Shared tri-state (True/False/None) policy for the 3+1 sterile
        neutral-current matter term, used identically by
        ``medium.solar``/``medium.earth``/``medium.atmosphere``.
"""

from __future__ import annotations

import warnings
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


def resolve_include_matter_nc(
    include_matter_nc: Optional[bool],
    oscillation: "OscillationParameters",
    *,
    has_neutron_data: bool,
    context_name: str,
) -> bool:
    """Resolve the tri-state ``include_matter_nc`` policy to a concrete bool.

    The 3+1 sterile extension's neutral-current term is *not* an optional
    refinement the way it is sometimes treated: once the common V_NC phase
    is subtracted, a genuine relative potential survives on the sterile
    diagonal entry, with magnitude comparable to V_CC itself (see
    ``core.common.hamiltonian``'s module docstring). Silently defaulting to
    the CC-only Hamiltonian whenever a caller selects a sterile oscillation
    but says nothing about ``include_matter_nc`` is therefore a silent
    physical incompleteness, not a harmless simplification -- unlike the
    3-flavour case, where V_NC truly is an unobservable common phase and
    omitting it is exact, not approximate.

    This function is the single place that policy is decided, shared
    identically by ``medium.solar``, ``medium.earth``, and
    ``medium.atmosphere``:

        - ``include_matter_nc`` given explicitly (``True`` or ``False``):
          returned unchanged. This always wins, and is how any existing
          CC-only benchmark, cross-validation test, or comparison to
          earlier project phases keeps working exactly as before.
        - ``include_matter_nc=None`` (the caller did not say): resolves to
          ``True`` when ``oscillation`` is the 3+1 sterile extension *and*
          ``has_neutron_data`` is True (the caller has already checked that
          the profile/model in hand can actually supply n_n); ``False``
          otherwise. When sterile is active but ``has_neutron_data`` is
          False, this still resolves to ``False`` (reproducing the CC-only
          Hamiltonian, not raising), but emits an explicit ``RuntimeWarning``
          instead of resolving silently, since that fallback is exactly the
          kind of physical incompleteness this function exists to avoid
          hiding.
        - For the plain 3-flavour case, ``include_matter_nc=None`` always
          resolves to ``False`` regardless of ``has_neutron_data``: V_NC is
          an unobservable common phase there, so there is nothing
          incomplete about omitting it, and no warning is warranted.

    Args:
        include_matter_nc: The caller-supplied tri-state value.
        oscillation: Built oscillation parameters (only
            ``BSM_extension_sterile`` is read).
        has_neutron_data: Whether the medium-specific profile/model in hand
            can actually supply a neutron density for the sterile diagonal
            term (e.g. ``profile.density_n is not None`` for solar,
            ``EarthProfile.has_neutron_density`` for Earth). Irrelevant
            (never evaluated for its truthiness in a way that matters) when
            ``include_matter_nc`` is given explicitly or when ``oscillation``
            is not the sterile extension.
        context_name: Short, human-readable identifier for the calling
            function (e.g. ``"solar_probability_mass"``), used only in the
            fallback warning message.

    Returns:
        The concrete boolean to use for this call.
    """
    if include_matter_nc is not None:
        return bool(include_matter_nc)

    if not oscillation.BSM_extension_sterile:
        return False

    if not has_neutron_data:
        warnings.warn(
            f"{context_name}: the 3+1 sterile extension is active but no "
            "neutron-density data is available for this profile/model, so "
            "the neutral-current matter term cannot be included. Falling "
            "back to the CC-only sterile Hamiltonian (include_matter_nc="
            "False) -- pass include_matter_nc=False explicitly to silence "
            "this warning, or build the profile with neutron-density "
            "support to get the physically complete 3+1 Hamiltonian.",
            RuntimeWarning,
            stacklevel=3,
        )
        return False

    return True

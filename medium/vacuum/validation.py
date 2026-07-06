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
Validation helpers comparing tpeanuts.medium.vacuum against legacy peanuts.

The legacy vacuum implementation is scalar and NumPy-based. The helpers in
this module therefore evaluate one energy, one baseline, and one initial state
at a time, then compare the result with the torch-native vacuum functions.

Module functions:
    ensure_legacy_importable(...)
        Return the bundled legacy peanuts package path.
    legacy_modules(...)
        Import the legacy PMNS and vacuum modules.
    legacy_pmns_from_torch(...)
        Build a legacy peanuts.pmns.PMNS object from a torch PMNS object.
    compare_pvacuum_with_legacy(...)
        Compare final vacuum probabilities against peanuts.vacuum.Pvacuum.
    compare_vacuum_evolved_state_with_legacy(...)
        Compare coherent evolved states against
        peanuts.vacuum.vacuum_evolved_state.
"""



from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.medium.vacuum.evolutor import vacuum_evolved_state
from tpeanuts.medium.vacuum.probability import pvacuum
from tpeanuts.util.constant import R_E
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import TensorLike


def ensure_legacy_importable() -> Path:
    """Return the bundled legacy peanuts package path.

    Returns:
        Path pointing to the local ``peanuts`` package used for legacy
        validation. Importing is delegated to Python's normal import
        machinery; this helper only centralizes the expected location.
    """
    package_dir = Path(__file__).resolve().parents[2]
    return package_dir / "peanuts"


def legacy_modules():
    """Import the legacy PMNS and vacuum modules.

    Returns:
        Pair ``(legacy_pmns, legacy_vacuum)`` corresponding to
        ``peanuts.pmns`` and ``peanuts.vacuum``.
    """
    ensure_legacy_importable()
    legacy_pmns = importlib.import_module("peanuts.pmns")
    legacy_vacuum = importlib.import_module("peanuts.vacuum")
    return legacy_pmns, legacy_vacuum


def legacy_pmns_from_torch(pmns_torch: object):
    """Build a legacy PMNS object from a torch-native PMNS object.

    Args:
        pmns_torch: PMNS-like object exposing a ``params`` attribute with
            tensor fields ``theta12``, ``theta13``, ``theta23``, and
            ``delta``.

    Returns:
        Instance of ``peanuts.pmns.PMNS`` with scalar NumPy/Numba angles.
    """
    legacy_pmns_module, _ = legacy_modules()

    return legacy_pmns_module.PMNS(
        float(pmns_torch.params.theta12.detach().cpu()),
        float(pmns_torch.params.theta13.detach().cpu()),
        float(pmns_torch.params.theta23.detach().cpu()),
        float(pmns_torch.params.delta.detach().cpu()),
    )


def _numpy_state(nustate: TensorLike, *, complex_state: bool) -> np.ndarray:
    dtype = np.complex128 if complex_state else float
    if torch.is_tensor(nustate):
        return np.asarray(nustate.detach().cpu().numpy(), dtype=dtype)
    return np.asarray(nustate, dtype=dtype)


def _diff_summary(torch_value: np.ndarray, legacy_value: np.ndarray) -> dict[str, Any]:
    abs_diff = np.abs(torch_value - legacy_value)
    return {
        "torch": torch_value,
        "legacy": legacy_value,
        "abs_diff": abs_diff,
        "max_abs": float(np.max(abs_diff)),
    }


def compare_pvacuum_with_legacy(
    nustate: TensorLike,
    oscillation: OscillationParameters,
    E_MeV: float,
    L_km: float,
    *,
    massbasis: bool = True,
    context: RuntimeContext = RuntimeContext(device=torch.device("cpu"), dtype=torch.float64),
    evolution_scale_m: TensorLike = R_E,
) -> dict[str, Any]:
    """Compare ``pvacuum`` with legacy ``peanuts.vacuum.Pvacuum``.

    Args:
        nustate: Initial state. Interpreted as mass weights when
            ``massbasis=True`` and coherent flavour amplitudes otherwise.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Scalar neutrino energy in MeV.
        L_km: Scalar vacuum baseline in km.
        massbasis: Basis convention passed to both implementations.
        context: Runtime device/dtype for the new implementation.
        evolution_scale_m: Evolution scale used by the torch implementation.
            The default matches the legacy Earth-radius normalization.

    Returns:
        Dictionary with ``torch``, ``legacy``, ``abs_diff`` and ``max_abs``.

    Notes:
        The bundled legacy code has a known ``antinu=True`` and
        ``massbasis=True`` branch that references ``nb`` inside
        ``peanuts.vacuum``. This helper intentionally does not patch legacy;
        if that branch fails, the legacy exception is propagated.
    """
    _, legacy_vacuum = legacy_modules()
    legacy_pmns = legacy_pmns_from_torch(oscillation.pmns)
    device, dtype = context.device, context.dtype

    torch_p = pvacuum(
        nustate,
        oscillation,
        torch.tensor(E_MeV, device=device, dtype=dtype),
        torch.tensor(L_km, device=device, dtype=dtype),
        massbasis=massbasis,
        context=context,
        evolution_scale_m=evolution_scale_m,
    )

    legacy_p = legacy_vacuum.Pvacuum(
        _numpy_state(nustate, complex_state=not massbasis),
        legacy_pmns,
        float(oscillation.DeltamSq21),
        float(oscillation.DeltamSq3l),
        float(E_MeV),
        float(L_km),
        antinu=bool(oscillation.antinu),
        massbasis=bool(massbasis),
    )

    return _diff_summary(
        torch_p.detach().cpu().numpy(),
        np.asarray(legacy_p, dtype=float),
    )


def compare_vacuum_evolved_state_with_legacy(
    nustate: TensorLike,
    oscillation: OscillationParameters,
    E_MeV: float,
    L_km: float,
    *,
    context: RuntimeContext = RuntimeContext(device=torch.device("cpu"), dtype=torch.float64),
    evolution_scale_m: TensorLike = R_E,
) -> dict[str, Any]:
    """Compare coherent evolved states with the legacy vacuum implementation.

    Args:
        nustate: Initial coherent flavour-basis amplitudes.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Scalar neutrino energy in MeV.
        L_km: Scalar vacuum baseline in km.
        context: Runtime device/dtype for the new implementation.
        evolution_scale_m: Evolution scale used by the torch implementation.

    Returns:
        Dictionary with ``torch``, ``legacy``, ``abs_diff`` and ``max_abs``.
    """
    _, legacy_vacuum = legacy_modules()
    legacy_pmns = legacy_pmns_from_torch(oscillation.pmns)
    device, dtype = context.device, context.dtype

    torch_state = vacuum_evolved_state(
        nustate,
        oscillation,
        torch.tensor(E_MeV, device=device, dtype=dtype),
        torch.tensor(L_km, device=device, dtype=dtype),
        context=context,
        evolution_scale_m=evolution_scale_m,
    )

    legacy_state = legacy_vacuum.vacuum_evolved_state(
        _numpy_state(nustate, complex_state=True),
        legacy_pmns,
        float(oscillation.DeltamSq21),
        float(oscillation.DeltamSq3l),
        float(E_MeV),
        float(L_km),
        antinu=bool(oscillation.antinu),
    )

    return _diff_summary(
        torch_state.detach().cpu().numpy(),
        np.asarray(legacy_state, dtype=np.complex128),
    )

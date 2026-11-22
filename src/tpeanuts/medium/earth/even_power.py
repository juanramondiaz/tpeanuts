#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
#
# TPeanuts — Fast Simulation of Neutrino Oscillations in Matter.
#
# Developed as part of the Master's Thesis for the MSc in Advanced Physics,
# specialization in Theoretical Physics, at the Universitat de València.
#
# Author
# ------
# Juan Ramón Díaz Santos
# diazjuan@alumni.uv.es
#
# Supervisors
# -----------
# Roberto Ruiz de Austri Bazán
# rruiz@ific.uv.es
#
# Michele Lucente
# michele.lucente@unibo.it
#
# Universitat de València, Spain
# September 2026
# =============================================================================

"""The PREM even-power shell fit, and its shift from radius to chord coordinate.

Reads ``tpeanuts.data.earth``'s ``fit/even_power_electron.csv``/
``fit/even_power_neutron.csv`` (6 shells each, ``rj,alpha,beta,gamma``,
``n(r) = alpha + beta*r**2 + gamma*r**4`` for ``r <= rj``, both files
sharing the same ``rj`` boundaries -- see that package's own ``README.md``
for why 6, not v1_0's own 5: the outermost shell was re-fit into two,
mantle+crust and ocean separately, after confirming directly that
lumping them into one shell missed the crust/ocean density jump by a
factor of ~3) -- the compact closed-form density fit
``tpeanuts.medium.earth.earth.EarthMedium``'s
``PropagationMethod.ANALYTIC_PERTURBATIVE`` path uses to build a
``tpeanuts.physics.evolution.perturbative.profile.EvenPowerProfile`` with few, PREM
-shell-aligned segments, instead of ``NUMERICAL``/``ANALYTIC``'s many
near-constant-density ones.

``shift_shell_coefficients_to_chord`` is the one piece of real math here:
substituting a chord's own ``r**2 = x**2 + sin(eta_rad)**2`` into a shell's
``r``-polynomial gives an ``x``-polynomial directly (the same
substitution v1_0's own ``shifted`` performs, without its layer
-reindexing machinery -- that lives in
``tpeanuts.medium.earth.geometry.crossed_shell_segments`` instead, kept
separate since it is pure geometry, not density algebra).

Module contents:
    EvenPowerShellTable
        Frozen dataclass: the per-shell ``rj``/electron/neutron
        coefficient tensors.
    load_even_power_shells(...)
        Read both fit CSVs into an ``EvenPowerShellTable``.
    shift_shell_coefficients_to_chord(...)
        ``r``-polynomial coefficients -> ``x``-polynomial coefficients,
        for one nadir angle.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import torch

from tpeanuts.data import package_data_dir
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import TensorLike, as_tensor

_EVEN_POWER_ELECTRON_CSV_PATH = package_data_dir() / "earth" / "fit" / "even_power_electron.csv"
_EVEN_POWER_NEUTRON_CSV_PATH = package_data_dir() / "earth" / "fit" / "even_power_neutron.csv"


@dataclass(frozen=True)
class EvenPowerShellTable:
    """The per-shell PREM even-power fit (6 shells -- see this module's own docstring), loaded as sorted 1-D tensors.

    Attributes:
        rj: Outer radius fraction of each shell, shape ``(n_shells,)``, sorted
            ascending (the last entry is exactly 1.0, the surface).
        alpha, beta, gamma: Electron-density coefficients per shell,
            shape ``(n_shells,)`` each -- ``n_e(r) = alpha + beta*r**2 +
            gamma*r**4`` for ``r`` in that shell.
        alpha_n, beta_n, gamma_n: Neutron-density counterparts, same
            shape/layout, same ``rj`` boundaries.
    """

    rj: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    gamma: torch.Tensor
    alpha_n: torch.Tensor
    beta_n: torch.Tensor
    gamma_n: torch.Tensor


def _load_fit_csv(path: Path, context: RuntimeContext) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rj, alpha, beta, gamma = [], [], [], []
    with open(path, newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            rj.append(float(row["rj"]))
            alpha.append(float(row["alpha"]))
            beta.append(float(row["beta"]))
            gamma.append(float(row["gamma"]))
    return (
        as_tensor(rj, device=context.device, dtype=context.dtype),
        as_tensor(alpha, device=context.device, dtype=context.dtype),
        as_tensor(beta, device=context.device, dtype=context.dtype),
        as_tensor(gamma, device=context.device, dtype=context.dtype),
    )


@lru_cache(maxsize=None)
def load_even_power_shells(context: RuntimeContext) -> EvenPowerShellTable:
    """Read both even-power fit CSVs into an ``EvenPowerShellTable`` on the given context.

    Cached per ``context`` -- same reasoning as
    ``tpeanuts.medium.earth.density.load_prem_table``'s own cache: fixed,
    read-only data, never mutated in place anywhere downstream.

    Args:
        context: RuntimeContext the returned tensors are built on.

    Returns:
        EvenPowerShellTable with both files' shells in their own
        (radius-ascending) order.

    Raises:
        ValueError: If the electron and neutron files do not share the
            same ``rj`` shell boundaries (they are expected to, both
            being fits to the same PREM layering).
    """
    rj, alpha, beta, gamma = _load_fit_csv(_EVEN_POWER_ELECTRON_CSV_PATH, context)
    rj_n, alpha_n, beta_n, gamma_n = _load_fit_csv(_EVEN_POWER_NEUTRON_CSV_PATH, context)
    if not torch.allclose(rj, rj_n, atol=1e-9):
        raise ValueError(
            f"even_power_electron.csv and even_power_neutron.csv have different shell boundaries "
            f"(rj={rj.tolist()!r} vs rj_n={rj_n.tolist()!r}) -- expected identical PREM layering."
        )
    return EvenPowerShellTable(
        rj=rj, alpha=alpha, beta=beta, gamma=gamma, alpha_n=alpha_n, beta_n=beta_n, gamma_n=gamma_n
    )


def shift_shell_coefficients_to_chord(
    alpha: TensorLike, beta: TensorLike, gamma: TensorLike, sin_eta_sq: TensorLike
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Re-express a shell's ``r``-polynomial as an ``x``-polynomial along a chord at one nadir angle.

    Substitutes ``r**2 = x**2 + sin(eta_rad)**2`` into ``n(r) = alpha +
    beta*r**2 + gamma*r**4``:

        a = alpha + beta*s + gamma*s**2
        b = beta + 2*gamma*s
        c = gamma

    with ``s = sin_eta_sq``. Exact for any ``x`` while the chord stays
    within the one shell these coefficients belong to (the segment
    boundaries ``tpeanuts.medium.earth.geometry.crossed_shell_segments``
    builds guarantee this).

    Args:
        alpha, beta, gamma: Shell coefficients (see
            ``EvenPowerShellTable``), any shape.
        sin_eta_sq: ``sin(eta_rad)**2`` for the nadir angle this chord is at,
            broadcastable against ``alpha``/``beta``/``gamma``.

    Returns:
        ``(a, b, c)``, the ``x``-polynomial coefficients
        ``tpeanuts.physics.evolution.perturbative.profile.EvenPowerProfile`` expects,
        broadcast shape of the inputs.
    """
    alpha_t = as_tensor(alpha)
    beta_t = as_tensor(beta, device=alpha_t.device, dtype=alpha_t.dtype)
    gamma_t = as_tensor(gamma, device=alpha_t.device, dtype=alpha_t.dtype)
    s = as_tensor(sin_eta_sq, device=alpha_t.device, dtype=alpha_t.dtype)

    a = alpha_t + beta_t * s + gamma_t * s * s
    b = beta_t + 2.0 * gamma_t * s
    c = gamma_t
    return a, b, c

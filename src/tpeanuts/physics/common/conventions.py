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

"""Conversion of 3+1 mixing parameters between rotation-order conventions.

A 3+1 mixing matrix is a product of six Givens rotations, and the order of
the product is a convention. ``MixingMatrix`` factorizes it as
``R23 Delta [R24 R34] R13 Delta^dagger R12 R14`` so that the factor that
commutes with the charged-current potential can be separated, whereas the
published 3+1 fits and nuSQuIDS use ``R34 R24 R14 R23 R13 R12``. The same
numbers in the two orders describe different mixings (by up to ``8e-2`` in a
probability for the angles of a typical fit), so a user who takes the
angles of a publication has to convert them.

``sterile_parameters_from_rotation_order`` does the conversion: it builds the
matrix in the order of the publication and finds, by least squares on the
complex matrix, the parameters of ``MixingMatrix`` that reproduce it. A free
rephasing of the flavours and of the mass eigenstates is allowed in the fit,
because it changes no probability: the phases of the flavours commute with the
matter potentials and those of the mass eigenstates drop out of every
probability.

Module contents:
    NUSQUIDS_ROTATION_ORDER
        The order of the rotations of nuSQuIDS and of most 3+1 fits.
    sterile_parameters_from_rotation_order
        The ``MixingMatrix`` parameters that describe a 3+1 mixing given in
        another rotation order.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import least_squares

from tpeanuts.physics.common.mixing import MixingMatrix, StandardMixingParameters, SterileMixingParameters
from tpeanuts.physics.common.nature import Dirac
from tpeanuts.physics.common.space import FlavourSpace
from tpeanuts.util.context import RuntimeContext

NUSQUIDS_ROTATION_ORDER: Tuple[str, ...] = ("34", "24", "14", "23", "13", "12")

_PLANES = {"12": (0, 1), "13": (0, 2), "23": (1, 2), "14": (0, 3), "24": (1, 3), "34": (2, 3)}


def _rotation(i: int, j: int, theta: float, phase: float = 0.0) -> np.ndarray:
    """Complex ``4 x 4`` rotation in the ``(i, j)`` plane, with ``exp(-i phase)`` on element ``(i, j)``."""
    matrix = np.eye(4, dtype=complex)
    matrix[i, i] = matrix[j, j] = math.cos(theta)
    matrix[i, j] = math.sin(theta) * np.exp(-1j * phase)
    matrix[j, i] = -math.sin(theta) * np.exp(1j * phase)
    return matrix


def _target_matrix(angles: dict, phases: dict, order: Sequence[str]) -> np.ndarray:
    """Product of the rotations of the given order, ``U = R_a R_b ...`` from left to right."""
    matrix = np.eye(4, dtype=complex)
    for name in order:
        i, j = _PLANES[name]
        matrix = matrix @ _rotation(i, j, angles[name], phases.get(name, 0.0))
    return matrix


def _mixing_from_values(values: Sequence[float], context: RuntimeContext) -> MixingMatrix:
    """``MixingMatrix`` of ``(theta12, theta13, theta23, delta13, theta14, theta24, theta34, delta14, delta24)``."""
    t12, t13, t23, d13, t14, t24, t34, d14, d24 = (float(x) for x in values)
    return MixingMatrix(
        space=FlavourSpace(n_sterile=1),
        standard=StandardMixingParameters(theta12=t12, theta13=t13, theta23=t23, delta13=d13, context=context),
        nature=Dirac(),
        context=context,
        sterile=SterileMixingParameters(
            theta_e=[t14], theta_mu=[t24], theta_tau=[t34], delta_e=[d14], delta_mu=[d24], delta_tau=[0.0],
            context=context,
        ),
    )


def sterile_parameters_from_rotation_order(
    theta12: float,
    theta13: float,
    theta23: float,
    delta13: float,
    theta14: float,
    theta24: float,
    theta34: float,
    delta14: float,
    delta24: float,
    *,
    context: RuntimeContext,
    rotation_order: Sequence[str] = NUSQUIDS_ROTATION_ORDER,
    tolerance: float = 1.0e-24,
    n_trials: int = 8,
) -> Tuple[StandardMixingParameters, SterileMixingParameters]:
    """Parameters of ``MixingMatrix`` that describe a 3+1 mixing given as a product of rotations in another order.

    The matrix ``U = R_a R_b ...`` of the given order (the phase ``delta13`` on ``R13``, ``delta14`` on ``R14`` and
    ``delta24`` on ``R24``, ``R34`` real as for Dirac 3+1 neutrinos) is built, and the parameters of the factorization
    of ``MixingMatrix`` are found by least squares on the complex matrix with a free rephasing of the flavours and of
    the mass eigenstates. The fit starts from the given numbers and is repeated from random starts until the squared
    residual is below ``tolerance``. If the three sterile angles vanish the mixing is the same in every order and the
    numbers are returned as they are.

    Args:
        theta12, theta13, theta23, delta13: Standard angles and CP phase, in radians, in the given order.
        theta14, theta24, theta34: Active-sterile angles, in radians.
        delta14, delta24: CP phases of ``R14`` and ``R24``, in radians.
        context: RuntimeContext of the returned parameters.
        rotation_order: Order of the product, the six names ``"12"``, ``"13"``, ``"23"``, ``"14"``, ``"24"`` and
            ``"34"`` once each. Defaults to ``NUSQUIDS_ROTATION_ORDER``.
        tolerance: Largest accepted squared residual of the fit.
        n_trials: Number of starts of the fit.

    Returns:
        ``(standard, sterile)``, to be passed as the ``standard`` and ``sterile`` arguments of a ``MixingMatrix`` on
        ``FlavourSpace(n_sterile=1)``.

    Raises:
        ValueError: If ``rotation_order`` is not a permutation of the six rotations.
        RuntimeError: If no start reaches the tolerance.

    Examples:
        >>> import math
        >>> import torch
        >>> import tpeanuts as tp
        >>> from tpeanuts.physics.common.conventions import sterile_parameters_from_rotation_order
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> standard, sterile = sterile_parameters_from_rotation_order(
        ...     0.58, 0.15, 0.85, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, context=context)
        >>> tuple(sterile.theta_e.shape)
        (1,)
    """
    if sorted(rotation_order) != sorted(_PLANES):
        raise ValueError(
            f"rotation_order must contain each of {sorted(_PLANES)} once, got {tuple(rotation_order)!r}."
        )
    initial = np.array(
        [theta12, theta13, theta23, delta13, theta14, theta24, theta34, delta14, delta24], dtype=float,
    )
    if theta14 == theta24 == theta34 == 0.0:
        values = initial
    else:
        angles = {"12": theta12, "13": theta13, "23": theta23, "14": theta14, "24": theta24, "34": theta34}
        phases = {"13": delta13, "14": delta14, "24": delta24}
        target = _target_matrix(angles, phases, rotation_order)
        fit_context = RuntimeContext.resolve("cpu", torch.float64)

        def residual(z: np.ndarray) -> np.ndarray:
            matrix = _mixing_from_values(z[:9].tolist(), fit_context).pmns_matrix().detach().cpu().numpy()
            flavour = np.exp(1j * np.concatenate([[0.0], z[9:12]]))
            mass = np.exp(1j * z[12:16])
            difference = matrix - flavour[:, None] * target * mass[None, :]
            return np.concatenate([difference.real.ravel(), difference.imag.ravel()])

        start = np.concatenate([initial, np.zeros(7)])
        best = None
        for trial in range(n_trials):
            guess = start if trial == 0 else start + 0.3 * np.random.default_rng(trial).standard_normal(16)
            result = least_squares(residual, guess, xtol=1.0e-15, ftol=1.0e-15, gtol=1.0e-15)
            if best is None or result.cost < best.cost:
                best = result
            if best.cost < 1.0e-2 * tolerance:
                break
        assert best is not None
        if best.cost > tolerance:
            raise RuntimeError(f"No parametrisation of the mixing was found (residual {best.cost:.1e}).")
        values = best.x[:9]

    t12, t13, t23, d13, t14, t24, t34, d14, d24 = (float(x) for x in values)
    standard = StandardMixingParameters(theta12=t12, theta13=t13, theta23=t23, delta13=d13, context=context)
    sterile = SterileMixingParameters(
        theta_e=[t14], theta_mu=[t24], theta_tau=[t34], delta_e=[d14], delta_mu=[d24], delta_tau=[0.0],
        context=context,
    )
    return standard, sterile

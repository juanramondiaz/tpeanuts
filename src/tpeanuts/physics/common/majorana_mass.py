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

"""The effective Majorana mass: a zero-baseline observable, not a propagation one.

Ported from v1_0's ``core/BSM/bsm_majorana.py::effective_majorana_mass``,
split into its own module here because v2 already relocated everything
else that file held (the Majorana CP-phase factor itself) into
``tpeanuts.physics.common.nature.Majorana`` -- see that class's own
docstring for the rephasing-invariance argument this module's own
docstring restates below. Nothing in ``tpeanuts.physics.evolution`` or
``tpeanuts.medium`` imports this module, and this module imports
neither: like v1_0's original, it is deliberately isolated from every
propagation code path.

Physics background: whether neutrinos are Majorana particles at all is
itself a BSM question the Standard Model does not answer. If they are,
the effective Majorana mass ``m_bb = |sum_i U_{e i}^2 m_i|`` governs the
rate of neutrinoless double-beta decay. Unlike an oscillation
amplitude -- where every Majorana phase contributed by ``U_bi`` cancels
exactly against its complex conjugate from ``U*_ai``, for every mass
eigenstate ``i``, independent of the propagation method (see
``tpeanuts.physics.common.nature.Majorana``'s own docstring) -- this
quantity is quadratic, not linear, in the electron row of the mixing
matrix, so that cancellation does not happen: the Majorana phases
survive here, and only here.

Deliberate generalization beyond v1_0: v1_0's own implementation
summed only the first three (active) columns of the electron row,
since it explicitly did not support combining its Majorana extension
with its sterile extension (``core/BSM/test/test4_bsm_majorana.py::
test_majorana_with_sterile_preset_raises``). v2's ``Majorana.
phase_matrix`` already generalizes to ``space.n_flavour - 1`` phases
for any ``n_sterile`` (see that class's own docstring), and
``MassSpectrum.absolute_mass_squared_vector`` already returns a full
``space.n_flavour``-length vector including any sterile masses -- so
this port sums over every column of ``pmns_matrix``'s electron row,
not just the first three. For ``space.n_sterile == 0`` this is
identical to v1_0's own formula; for a 3+N sterile Majorana scenario
it correctly includes the sterile states' own contribution to the
electron flavour, rather than silently dropping it.

Module contents:
    effective_majorana_mass(...)
        m_bb = |sum_i U_{e i}^2 m_i|, from a full PMNS-convention
        mixing matrix and the absolute mass-squared vector.
"""

from __future__ import annotations

import torch


def effective_majorana_mass(
    pmns_matrix: torch.Tensor,
    mass_squared_vector_eV2: torch.Tensor,
) -> torch.Tensor:
    """Compute the effective Majorana mass m_bb.

    Formula: ``m_bb = |sum_i U_{e i}^2 * m_i|``, using the electron row
    (index 0) of ``pmns_matrix`` and the masses ``m_i =
    sqrt(mass_squared_vector_eV2[..., i])``. This is the (e, e) entry
    of the Majorana mass matrix in the flavour basis.

    A zero-baseline, algebraic quantity: no dependence on energy,
    baseline, a matter profile, or a propagation method.

    Args:
        pmns_matrix: Mixing matrix shaped ``(..., n_flavour,
            n_flavour)``, as returned by
            ``tpeanuts.physics.common.mixing.MixingMatrix.pmns_matrix()``.
            For a Dirac ``Nature`` this still evaluates (the phase
            convention is then arbitrary and the result physically
            meaningless, since 0-nu-beta-beta requires Majorana
            neutrinos), but is only physically meaningful when built
            with ``nature=Majorana(...)``. Every column is read (not
            just the first three): see this module's own docstring for
            why that is a deliberate generalization beyond v1_0.
        mass_squared_vector_eV2: Absolute mass-squared vector shaped
            ``(..., n_flavour)``, in eV^2 -- typically
            ``MassSpectrum.absolute_mass_squared_vector(m_lightest_eV)``.
            Must be real and non-negative (negative entries are
            clamped to zero rather than raising, since a tiny negative
            value from upstream floating-point noise should not fail
            an otherwise-valid call).

    Returns:
        Real tensor with the batch shape broadcast from the two
        inputs, the effective Majorana mass in eV.

    Raises:
        RuntimeError: If ``pmns_matrix``'s and
            ``mass_squared_vector_eV2``'s final dimensions disagree
            (surfaced by the underlying tensor broadcast, not checked
            explicitly here).

    Examples:
        >>> import math
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> params = tp.StandardMixingParameters(
        ...     theta12=math.radians(33.4), theta13=math.radians(8.6), theta23=math.radians(42.0),
        ...     delta13=math.radians(230.0), context=context)
        >>> phases = torch.tensor([0.4, 1.1], dtype=torch.float64)
        >>> mixing = tp.MixingMatrix(
        ...     space=tp.FlavourSpace(), standard=params, nature=tp.Majorana(phases=phases, context=context),
        ...     context=context)
        >>> masses = tp.MassSpectrum(space=tp.FlavourSpace(), DeltamSq21=7.4e-5, DeltamSq3l=2.5e-3, context=context)
        >>> from tpeanuts.physics import effective_majorana_mass
        >>> m_bb = effective_majorana_mass(mixing.pmns_matrix(), masses.absolute_mass_squared_vector(0.01))
        >>> round(float(m_bb) * 1e3, 2)   # meV
        11.35
    """
    real_dtype = mass_squared_vector_eV2.dtype
    cdtype = pmns_matrix.dtype

    U_e = pmns_matrix[..., 0, :]
    m = torch.sqrt(mass_squared_vector_eV2.clamp_min(0.0)).to(dtype=real_dtype)

    return (U_e**2 * m.to(dtype=cdtype)).sum(dim=-1).abs()

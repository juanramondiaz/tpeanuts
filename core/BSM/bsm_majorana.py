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
Majorana-neutrino extension of the 3-flavour PMNS matrix.

Physics background
-------------------
Whether neutrinos are Dirac or Majorana particles is itself a BSM question
(the Standard Model has no right-handed neutrino field and no
lepton-number-violating mass term at all), so this extension lives here,
alongside NSI (``bsm_nsi.py``) and the 3+1 sterile scheme
(``bsm_sterile.py``), rather than inside ``core.SM`` or being bolted onto
``PMNSParams``/``PMNS_SM``. It follows exactly the same pattern as
``PMNS_sterile``: a companion parameter dataclass
(``MajoranaPhases``, mirroring ``PMNSSterileParams``) plus a sibling
``PMNS`` subclass (``PMNS_Majorana``, mirroring ``PMNS_sterile``) that
reuses the shared rotation/phase generators (``R12``, ``R13``, ``R23``,
``Delta``, ``_phase_diag`` -- all inherited unchanged from
``tpeanuts.core.common.pmns.PMNS``) without requiring any change to
``core.common`` or ``core.SM``.

If neutrinos are Majorana particles, the PMNS matrix carries two additional
physical phases beyond the Dirac phase delta, conventionally attached to
mass eigenstates 2 and 3:

    U = U_Dirac @ P,   P = diag(1, exp(i alpha21/2), exp(i alpha31/2))

By a standard theorem, these phases have **no effect on any oscillation
probability**: in the amplitude A(nu_a -> nu_b) = sum_i U_bi U*_ai
exp(-i k_i x), the factor P_ii contributed by U_bi cancels exactly against
P_ii* from U*_ai, for every i, independent of k_i, of the matter profile,
and of the propagation method. This is a rephasing-invariance statement,
not an approximation, and it holds for every pipeline in this project
(solar, atmospheric, Earth-regeneration, analytic or numerical), since all
of them build their amplitudes from the same U returned by
``pmns_matrix()``/``reduced()`` (see
``documentation/guide/chapters/03_core_bsm.tex``).

Consequently, ``PMNS_Majorana`` is a pure drop-in replacement for
``PMNS_SM`` wherever a mixing object is consumed: the matter Hamiltonian
(``core.common.hamiltonian``), every evolutor (``core.numerical``,
``core.perturbative``), and every probability/flux function are completely
unaffected by which of the two is used with the same angles -- BSM
extension in name (it lives here, not in ``core.SM``), but a no-op in
every propagation code path in this project. It matters only for
observables that read U directly rather than through an oscillation
probability, such as the effective Majorana mass
``effective_majorana_mass`` below, which is quadratic (not linear) in the
electron row and therefore does not enjoy the same cancellation. It
governs the rate of neutrinoless double-beta decay and has no relation to
oscillation baselines, matter profiles, or propagation methods.

Module contents:
    MajoranaPhases
        Immutable container for the two Majorana CP phases alpha21, alpha31
        (companion to a ``PMNSParams`` SM-sector object, exactly like
        ``PMNSSterileParams``).
    PMNS_Majorana
        3-flavour ``PMNS`` subclass, sibling of ``PMNS_SM``: identical
        ``outer_block``, and ``pmns_matrix``/``reduced`` right-multiplied by
        the Majorana phase factor P.
    effective_majorana_mass(...)
        m_bb = | sum_i U_ei^2 m_i |, from a full PMNS matrix and the
        absolute active-sector mass-squared vector
        (``core.common.mass_spectrum.MassSpectrum.absolute_mass_squared_vector``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch

from tpeanuts.core.common.pmns import PMNS, PMNSParams
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import TensorLike, as_tensor


# ---------------------------------------------------------------------------
# Parameter container -- Majorana-phase extension only
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MajoranaPhases:
    """Immutable container for the two physical Majorana CP-violating phases.

    Attributes
    ----------
    alpha21, alpha31:
        Majorana phases in radians, carried by mass eigenstates 2 and 3
        respectively (mass eigenstate 1 is the phase-convention reference,
        as usual). Physically real, independent of the Dirac phase delta.
    context:
        Runtime device/dtype used to store tensor parameters.

    Notes
    -----
    Companion to a ``PMNSParams`` (SM-sector) instance, exactly like
    ``PMNSSterileParams`` in ``bsm_sterile.py`` -- it does not modify
    ``PMNSParams`` itself. See the module docstring for why these phases
    are provably inert for every oscillation probability.
    """

    alpha21: TensorLike
    alpha31: TensorLike
    context: RuntimeContext

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "alpha21",
            as_tensor(self.alpha21, device=self.context.device, dtype=self.context.dtype),
        )
        object.__setattr__(
            self,
            "alpha31",
            as_tensor(self.alpha31, device=self.context.device, dtype=self.context.dtype),
        )

    @property
    def device(self) -> torch.device:
        """Return the torch device that stores the Majorana phase tensors."""
        return self.context.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the real torch dtype used by the Majorana phase tensors."""
        return self.context.dtype


# ---------------------------------------------------------------------------
# PMNS_Majorana
# ---------------------------------------------------------------------------

class PMNS_Majorana(PMNS):
    """3-flavour PMNS matrix for Majorana neutrinos.

    Sibling of ``tpeanuts.core.SM.sm_pmns.PMNS_SM`` (both are direct
    ``PMNS`` subclasses, mirroring how ``PMNS_sterile`` sits alongside
    ``PMNS_SM`` rather than inheriting from it): identical
    ``outer_block`` (``R23 @ Delta``, completely unaffected by the
    Dirac/Majorana choice -- the extra phase lives entirely on the
    mass-eigenstate side of the factorisation ``U = O @ U_red``), and
    ``pmns_matrix``/``reduced`` right-multiplied by the Majorana phase
    factor ``P = diag(1, exp(i alpha21/2), exp(i alpha31/2))``:

        U     = R23 @ Delta @ R13 @ Delta.conj() @ R12 @ P
        U_red = R13 @ R12 @ P

    Parameters
    ----------
    sm_params:
        ``PMNSParams`` bundling the 3-flavour SM mixing angles
        (theta12/theta13/theta23), the Dirac CP phase delta, and a
        RuntimeContext -- the same object type accepted by ``PMNS_SM``.
    majorana_params:
        ``MajoranaPhases`` bundling alpha21/alpha31 and a RuntimeContext.

    See the module docstring for why this is a no-op for every oscillation
    probability computed in this project, and matters only for the
    effective Majorana mass (``effective_majorana_mass`` below).
    """

    n_flavours: int = 3
    n_active: int = 3
    n_sterile: int = 0

    def __init__(
        self,
        sm_params: PMNSParams,
        majorana_params: MajoranaPhases,
    ) -> None:
        """Build a Majorana PMNS object from SM angles and Majorana phases.

        Mirrors ``PMNS_sterile(sm_params, sterile_params)``: both params
        objects are expected to already be fully built.

        Args:
            sm_params: SM-sector mixing parameters (see class docstring).
            majorana_params: Majorana-phase extension parameters (see class
                docstring).
        """
        # majorana_params must exist before super().__init__() runs, since
        # PMNS.__init__ calls self.pmns_matrix()/self.reduced() (which read
        # self.majorana_params) before returning here -- same reasoning as
        # PMNS_sterile.__init__'s object.__setattr__ use.
        object.__setattr__(self, "majorana_params", majorana_params)
        super().__init__(sm_params)

    def _majorana_phase_matrix(self) -> torch.Tensor:
        """Build ``diag(1, exp(i alpha21/2), exp(i alpha31/2))``.

        Reuses the shared ``PMNS._phase_diag`` generator (the same one
        ``Delta()`` is built from) at indices 1 and 2, so no new generic
        machinery is needed in ``core.common``.

        Returns:
            Complex diagonal tensor shaped (..., 3, 3).
        """
        p21 = self._phase_diag(1, self.majorana_params.alpha21 / 2)
        p31 = self._phase_diag(2, self.majorana_params.alpha31 / 2)
        return p21 @ p31

    def outer_block(
        self,
        antinu: Union[bool, torch.Tensor] = False,
    ) -> torch.Tensor:
        """Build the outer block ``R23 @ Delta`` (identical to ``PMNS_SM``)."""
        O = self.R23() @ self.Delta()
        return self.select_antinu(O, antinu)

    def reduced(
        self,
        antinu: Union[bool, torch.Tensor] = False,
    ) -> torch.Tensor:
        """Build the reduced mixing matrix ``U_red = R13 @ R12 @ P``.

        Args:
            antinu: Boolean scalar or tensor mask. True selects U_red*.

        Returns:
            Complex reduced mixing matrix shaped (..., 3, 3).
        """
        Ured = self.R13() @ self.R12() @ self._majorana_phase_matrix()
        return self.select_antinu(Ured, antinu)

    def pmns_matrix(
        self,
        antinu: Union[bool, torch.Tensor] = False,
    ) -> torch.Tensor:
        """Build the full mixing matrix ``U = R23 Delta R13 Delta^dagger R12 P``.

        Args:
            antinu: Boolean scalar or tensor mask. True selects U*.

        Returns:
            Complex full PMNS matrix shaped (..., 3, 3).
        """
        delt = self.Delta()
        U = (
            self.R23() @ delt @ self.R13() @ delt.conj()
            @ self.R12() @ self._majorana_phase_matrix()
        )
        return self.select_antinu(U, antinu)


# ---------------------------------------------------------------------------
# Effective Majorana mass (0-neutrino double-beta decay observable)
# ---------------------------------------------------------------------------

def effective_majorana_mass(
    pmns_matrix: torch.Tensor,
    mass_squared_vector_eV2: torch.Tensor,
) -> torch.Tensor:
    """Compute the effective Majorana mass m_bb.

    Formula: ``m_bb = | sum_i U_{e i}^2 * m_i |``, using the electron row
    (index 0) of the active 3x3 block of ``pmns_matrix`` and the active
    masses ``m_i = sqrt(mass_squared_vector_eV2[..., i])``. This is the
    (e, e) entry of the Majorana mass matrix in the flavour basis: it is
    quadratic (not linear) in the electron row, so -- unlike an oscillation
    amplitude -- the Majorana phases do not cancel here.

    A zero-baseline, algebraic quantity: no dependence on L, E, a matter
    profile, or a propagation method. Deliberately isolated from every
    other module in this project -- nothing in ``core.numerical``,
    ``core.perturbative``, or ``medium.*`` imports this function, and this
    module imports none of them.

    Args:
        pmns_matrix: Full PMNS matrix shaped (..., n_flavours, n_flavours)
            with ``n_flavours >= 3``, as returned by
            ``PMNS_Majorana.pmns_matrix()`` (or ``PMNS_SM.pmns_matrix()``,
            for the Dirac case, where this reduces to the trivial
            phase-convention-dependent combination with no BSM content).
            Only the electron row and the first three (active) columns are
            read.
        mass_squared_vector_eV2: Active mass-squared vector shaped
            (..., 3), in eV^2 -- typically
            ``MassSpectrum.absolute_mass_squared_vector(m_lightest_eV)``.
            Must be real and non-negative.

    Returns:
        Real tensor with the batch shape broadcast from the two inputs,
        the effective Majorana mass in eV.
    """
    real_dtype = mass_squared_vector_eV2.dtype
    cdtype = pmns_matrix.dtype

    U_e_active = pmns_matrix[..., 0, :3]
    m_active = torch.sqrt(mass_squared_vector_eV2.clamp_min(0.0)).to(dtype=real_dtype)

    return (U_e_active**2 * m_active.to(dtype=cdtype)).sum(dim=-1).abs()

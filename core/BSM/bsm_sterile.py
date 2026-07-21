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
4-flavor PMNS matrix for the 3+1 sterile neutrino scenario.

Physics background
------------------
In the 3+1 scheme, a light sterile neutrino ``ν_s`` is added to the three
active flavors (e, μ, τ).  The sterile state does not couple to the W or Z
bosons, so the matter potential remains

    H_mat = diag(V_CC, 0, 0, 0)

with V_CC = ±√2 G_F n_e · L_scale.

The 4×4 PMNS matrix is parametrized as

    U_4 = O_4 · R13_4 · Δ†_4 · R12_4 · R14

where the ``outer`` block

    O_4 = R23_4 · Δ_4 · R24 · R34

commutes with H_mat (it only acts in the {μ, τ, s} subspace).  This mirrors
exactly the 3-flavor peanuts factorization:

    U_SM = (R23 · Δ) · (R13 · Δ† · R12)
             O_SM          U_red_SM

with the extension:

    O_4  = R23_4 · Δ_4 · R24 · R34
    U_red_4 = R13_4 · R12_4 · R14

The identity:

     ``Δ† (R12 R14 diag(k) R14† R12†) Δ = R12 R14 diag(k) R14† R12†``

holds because neither R12 (acting on {e,μ}) nor R14 (acting on {e,s}) mixes
with index 2 (τ), which is the only non-trivial index of Δ.  

The reduction to

    U_red_4 = R13_4 · R12_4 · R14

is therefore valid, and the perturbative/numerical evolutors can be reused
for 4-flavor propagation simply by passing a ``PMNS_sterile`` instance.

Reduction to SM limit
---------------------
When θ14 = θ24 = θ34 = 0 the 4-flavor matrices block-diagonalize:

    U_red_4 → embed(U_red_SM, 4x4)
    O_4     → embed(O_SM, 4x4)
    U_4     → embed(U_SM, 4x4)

so the 3-flavor SM results are recovered exactly.

Module contents
---------------
PMNSSterileParams
    Immutable dataclass storing the sterile-sector-only extension
    parameters:

    theta14/24/34 and their CP phases.

    The SM sector (theta12/13/23, delta) lives in a companion ``PMNSParams``
    instance. The sterile mass splitting ``DeltamSq41`` is NOT stored here:
    like ``DeltamSq21``/``DeltamSq3l``, it does not enter any mixing-matrix
    rotation, so it lives on ``OscillationParameters`` instead (see
    ``tpeanuts.core.common.oscillation``).

PMNS_sterile
    Subclass of the abstract

        ``tpeanuts.core.common.pmns.PMNS``

    implementing the 4×4 mixing matrix and the peanuts-compatible basis
    transformations.

    ``R12``/``R13``/``R23``/``Delta``,
    ``reduced_basis``, ``refresh``, ``select_antinu``

    and the other flavour-count-agnostic methods are inherited unchanged from
    the base (sized automatically by ``n_flavours = 4``);
    only the genuinely 3+1-specific pieces are defined here.

    Key public interface (same contract as ``PMNS``):
        ``reduced(antinu)``               → U_red_4 shaped (..., 4, 4)
        ``pmns_matrix(antinu)``           → U_4 shaped (..., 4, 4)
        ``flavour_basis(...)``            → O_4 · op · O_4†

    The 4×4 matter Hamiltonian ``diag(V, 0, 0, 0)`` is assembled by
    ``hamiltonian_reduced``/``hamiltonian_flavour`` in
    ``tpeanuts.core.common.hamiltonian`` (the no-NSI branch; this module
    only builds mixing matrices and basis transformations, like its SM
    counterpart).

    Additional 4×4 rotation builders (not in parent):
        ``R14()``, ``R24()``, ``R34()``

    Construction mirrors ``PMNS_SM(params: PMNSParams)``: build a
    ``PMNSParams`` (SM sector) and a ``PMNSSterileParams`` (sterile
    extension), then pass both to the constructor::

        from tpeanuts.core.common.pmns import PMNSParams
        from tpeanuts.core.BSM.bsm_sterile import PMNSSterileParams, PMNS_sterile

        sm_params = PMNSParams(theta12, theta13, theta23, delta, context=context)
        sterile_params = PMNSSterileParams(
            theta14, theta24, theta34, delta14, delta24, delta34,
            context=context,
        )
        pmns4 = PMNS_sterile(sm_params, sterile_params)

    The sterile mass splitting ``DeltamSq41`` is supplied separately, as a
    field on ``OscillationParameters`` (alongside ``DeltamSq21``/
    ``DeltamSq3l``), not as part of either params object.

    Named presets (e.g. ``"sterile_3p1_bestfit_giunti2017"``) are built via
    ``tpeanuts.core.common.oscillation.oscillation_parameters_from_preset``,
    which constructs both params objects internally and returns the
    ``PMNS_sterile`` instance as ``oscillation.pmns``, with ``DeltamSq41``
    set on the returned ``OscillationParameters``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch

from tpeanuts.core.common.pmns import PMNS, PMNSParams
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import TensorLike, as_tensor


# ---------------------------------------------------------------------------
# Parameter container — sterile-sector extension only
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PMNSSterileParams:
    """Immutable container for the 3+1 sterile-sector extension parameters.

    Attributes
    ----------
    theta14, theta24, theta34:
        Active-sterile mixing angles (radians).
    delta14, delta24, delta34:
        CP phases associated with the sterile-sector rotations (radians).
        ``delta14`` is the phase carried by R14, etc.
    context:
        Runtime device/dtype used to store tensor parameters.

    Notes
    -----
    Does NOT carry the sterile mass splitting ``DeltamSq41`` — like
    ``DeltamSq21``/``DeltamSq3l``, it does not enter any mixing-matrix
    rotation, so it lives on ``OscillationParameters`` instead.
    """

    theta14: TensorLike
    theta24: TensorLike
    theta34: TensorLike
    delta14: TensorLike
    delta24: TensorLike
    delta34: TensorLike
    context: RuntimeContext

    def __post_init__(self) -> None:
        """Convert all angle and phase fields to tensors on the configured device and dtype."""
        for name in (
            "theta14", "theta24", "theta34",
            "delta14", "delta24", "delta34",
        ):
            object.__setattr__(
                self,
                name,
                as_tensor(getattr(self, name), device=self.context.device, dtype=self.context.dtype),
            )

    @property
    def device(self) -> torch.device:
        """Torch device of the stored parameter tensors."""
        return self.context.device

    @property
    def dtype(self) -> torch.dtype:
        """Real torch dtype of the stored parameter tensors."""
        return self.context.dtype


# ---------------------------------------------------------------------------
# PMNS_sterile
# ---------------------------------------------------------------------------

class PMNS_sterile(PMNS):
    """4-flavor PMNS matrix for the 3+1 sterile neutrino scenario.

    Inherits the 3-flavor Standard Model sector from ``PMNS`` and extends it
    with one sterile neutrino. ``n_flavours = 4`` automatically sizes the
    inherited ``R12``/``R13``/``R23``/``Delta`` builders; ``pmns_matrix``,
    ``reduced``, and ``flavour_basis`` are overridden here with the
    3+1 product structure so that ``PMNS_sterile`` can be passed directly to
    the existing Hamiltonian and evolutor functions.

    Parameters
    ----------
    sm_params:
        ``PMNSParams`` bundling the 3-flavour SM mixing angles
        (theta12/theta13/theta23), the Dirac CP phase delta, and a
        RuntimeContext. Stored as ``self.params``, satisfying the base
        ``PMNS`` contract for ``R12``/``R13``/``R23``/``Delta`` — same
        object type accepted by ``PMNS_SM``.
    sterile_params:
        ``PMNSSterileParams`` bundling theta14/theta24/theta34, their CP
        phases, and a RuntimeContext. Stored as ``self.sterile_params``,
        read by ``R14``/``R24``/``R34``.

    Notes
    -----
    The 4-flavor parametrization is

        U_4 = R23_4 · Δ_4 · R24 · R34 · R13_4 · Δ†_4 · R12_4 · R14

    which reduces to the SM result when θ14 = θ24 = θ34 = 0.

    The ``reduced()`` method returns

        U_red_4 = R13_4 · R12_4 · R14

    consistent with the peanuts factorization H_full = O_4 H_red O_4†.
    """

    #: Total number of neutrino flavors (active + sterile).
    n_flavours: int = 4
    #: Number of active (SM) flavors.
    n_active: int = 3
    #: Number of sterile flavors.
    n_sterile: int = 1

    def __init__(
        self,
        sm_params: PMNSParams,
        sterile_params: PMNSSterileParams,
    ) -> None:
        """Build a 3+1 PMNS object from a SM params object and its sterile extension.

        Mirrors ``PMNS_SM(params: PMNSParams)``: both params objects are
        expected to already be fully built (angles converted to tensors on
        the target device/dtype); this constructor does no conversion of
        its own.

        Args:
            sm_params: SM-sector mixing parameters (see class docstring).
            sterile_params: Sterile-extension parameters (see class
                docstring).
        """
        # self.sterile_params must exist before super().__init__() runs,
        # since PMNS.__init__ calls self.pmns_matrix()/self.reduced() (which
        # read self.sterile_params via R14/R24/R34) before returning here.
        # object.__setattr__ bypasses nn.Module.__setattr__'s bookkeeping
        # (not yet initialized at this point), writing straight to
        # self.__dict__ -- safe for a plain (non-tensor) attribute.
        object.__setattr__(self, "sterile_params", sterile_params)
        super().__init__(sm_params)

    # ------------------------------------------------------------------
    # 4×4 rotation matrices — sterile sector
    # ------------------------------------------------------------------
    #
    # R12/R13/R23/Delta (the SM sector embedded in 4×4) are inherited
    # unchanged from PMNS: they are built by the same self._rot/
    # self._phase_diag formulas as the 3-flavour case, sized automatically
    # by n_flavours = 4. Only the active-sterile rotations below, which have
    # no SM-sector counterpart, are specific to this class.

    def R14(self) -> torch.Tensor:
        """Build the 4×4 rotation in the (e, s) = (0, 3) subspace.

        Carries the CP phase δ14.

        Returns:
            Complex tensor shaped (..., 4, 4).
        """
        return self._rot(0, 3, self.sterile_params.theta14, phase=self.sterile_params.delta14)

    def R24(self) -> torch.Tensor:
        """Build the 4×4 rotation in the (μ, s) = (1, 3) subspace.

        Carries the CP phase δ24.

        Returns:
            Complex tensor shaped (..., 4, 4).
        """
        return self._rot(1, 3, self.sterile_params.theta24, phase=self.sterile_params.delta24)

    def R34(self) -> torch.Tensor:
        """Build the 4×4 rotation in the (τ, s) = (2, 3) subspace.

        Carries the CP phase δ34.

        Returns:
            Complex tensor shaped (..., 4, 4).
        """
        return self._rot(2, 3, self.sterile_params.theta34, phase=self.sterile_params.delta34)


    # ------------------------------------------------------------------
    # Mixing matrices (override PMNS)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def outer_block(
        self,
        antinu: Union[bool, torch.Tensor] = False,
    ) -> torch.Tensor:
        """Build the 3+1 outer block ``R23 @ Delta @ R24 @ R34``."""
        O4 = self.R23() @ self.Delta() @ self.R24() @ self.R34()
        return self.select_antinu(O4, antinu)
    
    @torch.no_grad()
    def reduced(
        self,
        antinu: Union[bool, torch.Tensor] = False,
    ) -> torch.Tensor:
        """Return the 4×4 reduced mixing matrix U_red_4 = R13_4 · R12_4 · R14.

        This is the direct 3+1 generalization of the 3-flavor peanuts reduced
        matrix U_red = R13 R12.  R14 is included because it mixes the electron
        flavor (index 0) with the sterile state (index 3) and therefore does
        NOT commute with H_mat = diag(V, 0, 0, 0).

        The corresponding ``outer`` block O_4 = R23_4 · Δ_4 · R24 · R34
        commutes with H_mat and is applied by ``flavour_basis``.

        Args:
            antinu: Bool or boolean tensor. ``True`` returns U_red_4*.

        Returns:
            Complex tensor shaped (..., 4, 4).
        """
        Ured4 = self.R13() @ self.R12() @ self.R14()
        return self.select_antinu(Ured4, antinu)

    @torch.no_grad()
    def pmns_matrix(
        self,
        antinu: Union[bool, torch.Tensor] = False,
    ) -> torch.Tensor:
        """Build the full 4×4 PMNS matrix.

        Convention:

            U_4 = R23_4 · Δ_4 · R24 · R34 · R13_4 · Δ†_4 · R12_4 · R14

        which generalizes U_SM = R23 · Δ · R13 · Δ† · R12 and reduces to
        it when θ14 = θ24 = θ34 = 0.

        Args:
            antinu: Bool or boolean tensor. ``True`` returns U_4*.

        Returns:
            Complex tensor shaped (..., 4, 4).
        """
        delt = self.Delta()
        U4 = (
            self.R23()
            @ delt
            @ self.R24()
            @ self.R34()
            @ self.R13()
            @ delt.conj()
            @ self.R12()
            @ self.R14()
        )
        return self.select_antinu(U4, antinu)

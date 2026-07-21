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

"""Perturbative segment for the PREM piecewise-linear-in-r² density profile.

For a single traversed shell the electron density in trajectory coordinates
x is a first-degree polynomial in x²:

    n_e(x) = a + b · x²,

where a = A + B·sin²(η) and b = B come from the parent layered profile after
the coordinate shift r² = x² + sin²(η).

All quantities required by the perturbative evolutor (average density, segment
length, MSW matter potential, and the oscillatory residual integral) are
derived analytically from this two-coefficient representation.  The residual
integral

    I_ab = exp(-i·λ_a·x₂ + i·λ_b·x₁) · b · [I₂(f) - x̄²·I₀(f)],

where f = λ_b − λ_a and I_k(f) = ∫_{x₁}^{x₂} xᵏ exp(i·f·x) dx, is computed
in closed form or via a Taylor expansion near f = 0 to avoid numerical
instability at degenerate eigenvalues.

Module classes:
    PremProfileSegment
        Density-profile segment for the PREM tabulated model.
"""

from __future__ import annotations

from math import factorial
from typing import Optional, Union

import torch

from tpeanuts.core.common.potential import matter_potential_cc, matter_potential_nc
from tpeanuts.util.constant import R_E
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.torch_util import infer_device_dtype
from tpeanuts.util.type import TensorLike, as_tensor


class PremProfileSegment:
    """PREM linear-in-r² density profile for one trajectory segment.

    Args:
        x1: Segment start in the profile coordinate (normalized by
            ``profile_scale_m``).
        x2: Segment end in the profile coordinate.
        a: Constant term of n_e(x) in profile coordinates  [mol cm⁻³].
        b: Coefficient of the x² term  [mol cm⁻³ (profile_unit)⁻²].
        a_n: Optional constant term of n_n(x), enabling the 3+1 sterile
            extension's neutral-current matter term.
        b_n: Optional coefficient of the x² term of n_n(x).
        antinu: Antineutrino selector for the MSW potential sign.
        profile_scale_m: Positive scale in metres defining the profile
            coordinate.
        evolution_scale_m: Positive scale in metres defining the Hamiltonian
            coordinate.  ``a`` and ``b`` are rescaled internally so that the
            stored coordinates and coefficients are in evolution units.
        device: Optional torch device.
        dtype: Optional real dtype.

    Attributes:
        x1, x2: Segment endpoints in evolution coordinates.
        a, b: Density polynomial coefficients in evolution coordinates.
        length: x2 - x1 in evolution coordinates.
        zero_mask: Boolean mask for zero-length segments.
        average: Segment-average electron density in mol cm⁻³.
        potential: MSW matter potential built from ``average``.
    """

    def __init__(
        self,
        x1: TensorLike,
        x2: TensorLike,
        a: TensorLike,
        b: TensorLike,
        *,
        a_n: TensorLike | None = None,
        b_n: TensorLike | None = None,
        antinu: Union[bool, torch.Tensor] = False,
        profile_scale_m: TensorLike = R_E,
        evolution_scale_m: TensorLike = R_E,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        legacy_precision: bool = False,
    ) -> None:
        infer_args = (x1, x2, a, b, profile_scale_m, evolution_scale_m)
        device, dtype = infer_device_dtype(*infer_args, device=device, dtype=dtype)

        x1 = as_tensor(x1, device=device, dtype=dtype)
        x2 = as_tensor(x2, device=device, dtype=dtype)
        a = as_tensor(a, device=device, dtype=dtype)
        b = as_tensor(b, device=device, dtype=dtype)
        profile_scale = as_tensor(profile_scale_m, device=device, dtype=dtype)
        evolution_scale = as_tensor(evolution_scale_m, device=device, dtype=dtype)
        if torch.any(profile_scale <= 0) or torch.any(evolution_scale <= 0):
            raise ValueError(
                "profile_scale_m and evolution_scale_m must be strictly positive."
            )

        # ratio = (evolution_scale / profile_scale)
        # x_evo  = x_profile / ratio
        # a_evo  = a  (density is independent of the coordinate scale)
        # b_evo  = b * ratio²  (polynomial coefficient for x²)
        ratio = evolution_scale / profile_scale
        base_shape = torch.broadcast_shapes(
            x1.shape, x2.shape, a.shape, b.shape, ratio.shape
        )
        ratio = torch.broadcast_to(ratio, base_shape)

        self.x1 = torch.broadcast_to(x1 / ratio, base_shape)
        self.x2 = torch.broadcast_to(x2 / ratio, base_shape)
        self.a = torch.broadcast_to(a, base_shape)
        self.b = torch.broadcast_to(b * ratio ** 2, base_shape)

        self.length = self.x2 - self.x1
        self.zero_mask = self.length == 0
        safe_length = torch.where(
            self.zero_mask, torch.ones_like(self.length), self.length
        )

        # Segment-average of n_e(x) = a + b·x²:
        #   avg = (1 / L) · ∫_{x1}^{x2} (a + b·x²) dx
        #       = a + b · (x1² + x1·x2 + x2²) / 3
        self.x_sq_avg = (self.x1 ** 2 + self.x1 * self.x2 + self.x2 ** 2) / 3.0
        integral = self.a * self.length + self.b * (self.x2 ** 3 - self.x1 ** 3) / 3.0
        self.average = torch.where(
            self.zero_mask,
            torch.zeros_like(integral),
            integral / safe_length,
        )

        self.potential = matter_potential_cc(
            self.average,
            antinu=antinu,
            evolution_scale_m=evolution_scale,
            context=RuntimeContext(device=device, dtype=dtype),
            legacy_precision=legacy_precision,
        )
        self.potential = torch.where(
            self.zero_mask, torch.zeros_like(self.potential), self.potential
        )

        self.a_n = None
        self.b_n = None
        self.average_n = None
        self.potential_n = None
        if a_n is not None:
            a_n = as_tensor(a_n, device=device, dtype=dtype)
            b_n = as_tensor(b_n, device=device, dtype=dtype)
            self.a_n = torch.broadcast_to(a_n, base_shape)
            self.b_n = torch.broadcast_to(b_n * ratio ** 2, base_shape)

            integral_n = self.a_n * self.length + self.b_n * (self.x2 ** 3 - self.x1 ** 3) / 3.0
            self.average_n = torch.where(
                self.zero_mask,
                torch.zeros_like(integral_n),
                integral_n / safe_length,
            )
            self.potential_n = matter_potential_nc(
                self.average_n,
                antinu=antinu,
                evolution_scale_m=evolution_scale,
                context=RuntimeContext(device=device, dtype=dtype),
            )
            self.potential_n = torch.where(
                self.zero_mask, torch.zeros_like(self.potential_n), self.potential_n
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Static helpers: oscillatory monomial integrals
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _oscillatory_monomial(
        power: int,
        x1: torch.Tensor,
        x2: torch.Tensor,
        freq: torch.Tensor,
    ) -> torch.Tensor:
        """Exact closed-form ∫_{x₁}^{x₂} xᵖᵒʷᵉʳ exp(i·freq·x) dx.

        Uses repeated integration by parts.  Numerically unstable when
        ``freq`` is small relative to the integration endpoints; in that
        regime use ``_taylor_monomial`` instead.

        Args:
            power: Non-negative integer monomial power (0 or 2 for PREM).
            x1: Lower bound (broadcast-compatible with ``freq``).
            x2: Upper bound.
            freq: Oscillation frequency (complex-valued, typically la - lb).

        Returns:
            Complex definite integral with the broadcast input shape.
        """
        total = torch.zeros_like(freq)
        i_freq = 1j * freq
        exp2 = torch.exp(i_freq * x2)
        exp1 = torch.exp(i_freq * x1)
        for order in range(power + 1):
            factor = (
                (-1) ** order
                * factorial(power)
                / factorial(power - order)
                / i_freq ** (order + 1)
            )
            total = total + factor * (
                x2 ** (power - order) * exp2 - x1 ** (power - order) * exp1
            )
        return total

    @staticmethod
    def _taylor_monomial(
        power: int,
        x1: torch.Tensor,
        x2: torch.Tensor,
        freq: torch.Tensor,
        *,
        order: int = 2,
    ) -> torch.Tensor:
        """Taylor expansion of the oscillatory monomial integral for small freq.

        Expands exp(i·freq·x) to ``order``-th order in freq and integrates
        the resulting polynomial term by term.  Avoids the 1/freq instability
        of the closed form near degenerate eigenvalues.

        Args:
            power: Non-negative integer monomial power.
            x1: Lower bound.
            x2: Upper bound.
            freq: Small oscillation frequency (typically la - lb ≈ 0).
            order: Truncation order of the Taylor series in freq.

        Returns:
            Complex approximate definite integral.
        """
        total = torch.zeros_like(freq)
        for series_order in range(order + 1):
            total = total + (
                (1j * freq) ** series_order
                / factorial(series_order)
                * (
                    x2 ** (power + series_order + 1)
                    - x1 ** (power + series_order + 1)
                )
                / (power + series_order + 1)
            )
        return total

    # ──────────────────────────────────────────────────────────────────────────
    # Perturbative evolutor interface
    # ──────────────────────────────────────────────────────────────────────────

    def _lift(self, x: torch.Tensor, target_ndim: int) -> torch.Tensor:
        """Unsqueeze x to broadcast against (..., 3, 3) matrix batches."""
        while x.ndim < target_ndim - 2:
            x = x.unsqueeze(-1)
        return x.unsqueeze(-1).unsqueeze(-1)

    @torch.no_grad()
    def residual_integral(
        self,
        la: torch.Tensor,
        lb: torch.Tensor,
        *,
        small_ratio: float = 1.0e-2,
        dl_zero_eps: float = 0.0,
    ) -> torch.Tensor:
        """Oscillatory residual integral for the linear-in-x² profile.

        The residual density is

            δn(x) = n_e(x) - average = b · (x² − x̄²),

        where x̄² = (x₁² + x₁·x₂ + x₂²) / 3.  The integral

            I_ab = ∫_{x₁}^{x₂} δn(x) exp(i·(λ_b − λ_a)·x) dx
                 = b · [I_mono(2, f) − x̄² · I_mono(0, f)]

        is multiplied by the vacuum phase

            exp(−i·λ_a·x₂ + i·λ_b·x₁)

        to match the convention expected by ``evolutor_first_order``.

        Args:
            la: Bra Hamiltonian eigenvalues, shape ``(..., 3, 1)``.
            lb: Ket Hamiltonian eigenvalues, shape ``(..., 1, 3)``.
            small_ratio: |la−lb|/|la+lb| threshold below which the Taylor
                expansion replaces the closed-form integral.
            dl_zero_eps: Absolute tolerance for treating la−lb as exactly zero
                (degenerate).  Zero (default) uses exact equality only.

        Returns:
            Complex tensor with shape ``(..., 3, 3)`` containing I_ab for
            each (a, b) eigenvalue pair.
        """
        if not torch.is_complex(la):
            cdtype = torch.complex128 if la.dtype == torch.float64 else torch.complex64
            la = la.to(dtype=cdtype)
        if not torch.is_complex(lb):
            lb = lb.to(dtype=la.dtype)

        target_ndim = la.ndim
        x1 = self._lift(self.x1, target_ndim).to(device=la.device, dtype=la.dtype)
        x2 = self._lift(self.x2, target_ndim).to(device=la.device, dtype=la.dtype)
        x_sq_avg = self._lift(self.x_sq_avg, target_ndim).to(
            device=la.device, dtype=la.dtype
        )
        b = self._lift(self.b, target_ndim).to(device=la.device, dtype=la.dtype)

        dl = la - lb  # shape (..., 3, 3)

        if dl_zero_eps > 0.0:
            is_zero = torch.abs(dl) < dl_zero_eps
        else:
            is_zero = dl == 0

        denom = la + lb
        ratio = torch.where(
            torch.abs(denom) > 0,
            torch.abs(dl / denom),
            torch.full_like(torch.abs(dl), float("inf")),
        )
        is_small = ratio < small_ratio

        # Use safe_dl (= 1 where degenerate) so we can evaluate the closed
        # form without dividing by zero; those entries are masked out below.
        safe_dl = torch.where(is_small | is_zero, torch.ones_like(dl), dl)

        i2_full = self._oscillatory_monomial(2, x1, x2, safe_dl)
        i0_full = self._oscillatory_monomial(0, x1, x2, safe_dl)
        i2_taylor = self._taylor_monomial(2, x1, x2, dl)
        i0_taylor = self._taylor_monomial(0, x1, x2, dl)

        i2 = torch.where(is_small, i2_taylor, i2_full)
        i0 = torch.where(is_small, i0_taylor, i0_full)

        # Vacuum phase factor (see module docstring).
        phase = torch.exp(-1j * la * x2 + 1j * lb * x1)
        out = phase * b * (i2 - x_sq_avg * i0)
        return torch.where(is_zero, torch.zeros_like(out), out)

    @torch.no_grad()
    def residual_integral_neutron(
        self,
        la: torch.Tensor,
        lb: torch.Tensor,
        *,
        small_ratio: float = 1.0e-2,
        dl_zero_eps: float = 0.0,
    ) -> torch.Tensor:
        """Neutron-density counterpart of ``residual_integral`` (uses ``b_n``).

        Enables the 3+1 sterile extension's neutral-current matter term (see
        ``core.perturbative.evolutor.evolutor_first_order``). Requires
        ``a_n``/``b_n`` to have been supplied at construction time. Note
        ``x_sq_avg`` is purely geometric (shared with ``residual_integral``);
        only ``b_n`` enters the residual, since the constant term ``a_n``
        cancels against ``average_n`` exactly, as for the electron-density
        formula (see the module docstring).

        Raises:
            ValueError: If this segment has no neutron-density coefficients.
        """
        if self.b_n is None:
            raise ValueError(
                "This segment has no neutron-density coefficients. Build it "
                "with a_n/b_n to enable the 3+1 sterile extension's "
                "neutral-current matter term."
            )
        if not torch.is_complex(la):
            cdtype = torch.complex128 if la.dtype == torch.float64 else torch.complex64
            la = la.to(dtype=cdtype)
        if not torch.is_complex(lb):
            lb = lb.to(dtype=la.dtype)

        target_ndim = la.ndim
        x1 = self._lift(self.x1, target_ndim).to(device=la.device, dtype=la.dtype)
        x2 = self._lift(self.x2, target_ndim).to(device=la.device, dtype=la.dtype)
        x_sq_avg = self._lift(self.x_sq_avg, target_ndim).to(
            device=la.device, dtype=la.dtype
        )
        b_n = self._lift(self.b_n, target_ndim).to(device=la.device, dtype=la.dtype)

        dl = la - lb

        if dl_zero_eps > 0.0:
            is_zero = torch.abs(dl) < dl_zero_eps
        else:
            is_zero = dl == 0

        denom = la + lb
        ratio = torch.where(
            torch.abs(denom) > 0,
            torch.abs(dl / denom),
            torch.full_like(torch.abs(dl), float("inf")),
        )
        is_small = ratio < small_ratio

        safe_dl = torch.where(is_small | is_zero, torch.ones_like(dl), dl)

        i2_full = self._oscillatory_monomial(2, x1, x2, safe_dl)
        i0_full = self._oscillatory_monomial(0, x1, x2, safe_dl)
        i2_taylor = self._taylor_monomial(2, x1, x2, dl)
        i0_taylor = self._taylor_monomial(0, x1, x2, dl)

        i2 = torch.where(is_small, i2_taylor, i2_full)
        i0 = torch.where(is_small, i0_taylor, i0_full)

        phase = torch.exp(-1j * la * x2 + 1j * lb * x1)
        out = phase * b_n * (i2 - x_sq_avg * i0)
        return torch.where(is_zero, torch.zeros_like(out), out)

    def has_perturbation(self) -> torch.Tensor:
        """Return True for segments where b ≠ 0 (non-constant residual).

        Segments with b = 0 have zero residual everywhere, so their
        first-order correction U1 vanishes and can be skipped.

        Returns:
            Boolean tensor shaped ``(...,)`` matching the segment batch.
        """
        return torch.abs(self.b) > 0

    def has_perturbation_neutron(self) -> torch.Tensor:
        """Return True for segments where b_n ≠ 0 (non-constant NC residual).

        Mirrors ``has_perturbation`` for the neutron-density coefficient.

        Raises:
            ValueError: If this segment has no neutron-density coefficients.
        """
        if self.b_n is None:
            raise ValueError(
                "This segment has no neutron-density coefficients. Build it "
                "with a_n/b_n to enable the 3+1 sterile extension's "
                "neutral-current matter term."
            )
        return torch.abs(self.b_n) > 0

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

"""Even-power density-profile model for perturbative segment evolution.

The perturbative evolutor needs the segment-average electron density, the
segment length, and the analytic oscillatory integral of the residual density.
This module implements those quantities for profiles of the form

    n_e(x) = a + b*x**2 + c*x**4 + delta1*x**6 + delta2*x**8 + ...

The class is independent of any medium geometry. Earth, atmosphere, or another
medium may build an ``EvenPowerProfileSegment`` from their own trajectory and density
data before handing it to the perturbative core evolutor.

Module classes:
    EvenPowerProfileSegment
        Segment profile with even-power coefficients and analytic residual
        integrals for the first-order perturbative correction.
"""

from __future__ import annotations

from math import factorial
from typing import Optional, Union

import torch

from tpeanuts.core.common.potential import matter_potential
from tpeanuts.util.constant import R_E
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.torch_util import infer_device_dtype
from tpeanuts.util.type import TensorLike, as_tensor


class EvenPowerProfileSegment:
    """Even-power density model for one perturbative segment.

    Args:
        x1: Initial coordinate normalized by ``profile_scale_m``.
        x2: Final coordinate normalized by ``profile_scale_m``.
        a: Constant density coefficient in profile coordinates.
        b: Quadratic density coefficient in profile coordinates.
        c: Quartic density coefficient in profile coordinates.
        deltas: Optional higher even-power coefficients. The last dimension
            enumerates ``x**6, x**8, ...``.
        antinu: Antineutrino selector used to build the matter potential.
        profile_scale_m: Scale defining the input profile coordinate.
        evolution_scale_m: Scale defining the Hamiltonian/evolution coordinate.
        device: Optional torch device.
        dtype: Optional real dtype.

    Attributes:
        coefficients: Even-power coefficients in evolution coordinates, with
            last dimension ordered as ``x**0, x**2, x**4, ...``.
        average: Segment-average electron density.
        length: Segment length in evolution coordinates.
        zero_mask: Mask selecting zero-length segments.
        potential: Matter potential built from ``average``.
    """

    @classmethod
    def from_coefficients(
        cls,
        x1: TensorLike,
        x2: TensorLike,
        coefficients: TensorLike,
        **kwargs,
    ) -> "EvenPowerProfileSegment":
        """Build a segment from a full even-power coefficient tensor.

        Args:
            x1: Initial coordinate normalized by ``profile_scale_m``.
            x2: Final coordinate normalized by ``profile_scale_m``.
            coefficients: Tensor with final dimension ordered as
                ``a, b, c, delta1, ...``.
            **kwargs: Additional constructor options.

        Returns:
            Initialized segment profile.
        """
        coefficients = torch.as_tensor(coefficients)
        return cls(
            x1=x1,
            x2=x2,
            a=coefficients[..., 0],
            b=coefficients[..., 1],
            c=coefficients[..., 2],
            deltas=coefficients[..., 3:],
            **kwargs,
        )

    @classmethod
    def constant(
        cls,
        x1: TensorLike,
        x2: TensorLike,
        density: TensorLike,
        **kwargs,
    ) -> "EvenPowerProfileSegment":
        """Build a constant-density segment profile.

        Args:
            x1: Initial coordinate normalized by ``profile_scale_m``.
            x2: Final coordinate normalized by ``profile_scale_m``.
            density: Constant electron density.
            **kwargs: Additional constructor options.

        Returns:
            Initialized constant segment profile.
        """
        density_tensor = torch.as_tensor(density)
        zeros = torch.zeros_like(density_tensor)
        return cls(
            x1=x1,
            x2=x2,
            a=density_tensor,
            b=zeros,
            c=zeros,
            **kwargs,
        )

    def __init__(
        self,
        x1: TensorLike,
        x2: TensorLike,
        a: TensorLike,
        b: TensorLike,
        c: TensorLike,
        *,
        deltas: TensorLike | None = None,
        antinu: Union[bool, torch.Tensor] = False,
        profile_scale_m: TensorLike = R_E,
        evolution_scale_m: TensorLike = R_E,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        legacy_precision: bool = False,
    ) -> None:
        infer_values = (x1, x2, a, b, c, profile_scale_m, evolution_scale_m)
        if deltas is not None:
            infer_values = (*infer_values, deltas)
        device, dtype = infer_device_dtype(*infer_values, device=device, dtype=dtype)

        x1 = as_tensor(x1, device=device, dtype=dtype)
        x2 = as_tensor(x2, device=device, dtype=dtype)
        a = as_tensor(a, device=device, dtype=dtype)
        b = as_tensor(b, device=device, dtype=dtype)
        c = as_tensor(c, device=device, dtype=dtype)
        if deltas is not None:
            deltas = as_tensor(deltas, device=device, dtype=dtype)

        profile_scale = as_tensor(profile_scale_m, device=device, dtype=dtype)
        evolution_scale = as_tensor(evolution_scale_m, device=device, dtype=dtype)
        if torch.any(profile_scale <= 0) or torch.any(evolution_scale <= 0):
            raise ValueError("Profile and evolution scales must be positive.")

        ratio = evolution_scale / profile_scale
        base_shape = torch.broadcast_shapes(
            x1.shape,
            x2.shape,
            a.shape,
            b.shape,
            c.shape,
            ratio.shape,
            () if deltas is None else deltas.shape[:-1],
        )
        ratio = torch.broadcast_to(ratio, base_shape)
        a = torch.broadcast_to(a, base_shape)
        b = torch.broadcast_to(b, base_shape) * ratio**2
        c = torch.broadcast_to(c, base_shape) * ratio**4
        if deltas is None:
            deltas = torch.empty((*base_shape, 0), device=device, dtype=dtype)
        else:
            deltas = torch.broadcast_to(deltas, (*base_shape, deltas.shape[-1]))
            powers = 2 * torch.arange(
                3,
                3 + deltas.shape[-1],
                device=device,
                dtype=dtype,
            )
            deltas = deltas * ratio[..., None] ** powers
        base_shape = torch.broadcast_shapes(
            x1.shape,
            x2.shape,
            a.shape,
            b.shape,
            c.shape,
            deltas.shape[:-1],
        )

        self.x1 = torch.broadcast_to(x1 / ratio, base_shape)
        self.x2 = torch.broadcast_to(x2 / ratio, base_shape)
        self.a = torch.broadcast_to(a, base_shape)
        self.b = torch.broadcast_to(b, base_shape)
        self.c = torch.broadcast_to(c, base_shape)
        self.deltas = torch.broadcast_to(deltas, (*base_shape, deltas.shape[-1]))
        self.coefficients = torch.cat(
            [
                self.a[..., None],
                self.b[..., None],
                self.c[..., None],
                self.deltas,
            ],
            dim=-1,
        )

        self.length = self.x2 - self.x1
        self.zero_mask = self.length == 0
        safe_length = torch.where(
            self.zero_mask,
            torch.ones_like(self.length),
            self.length,
        )

        powers = 2 * torch.arange(
            self.coefficients.shape[-1],
            device=device,
            dtype=dtype,
        )
        integral = torch.sum(
            self.coefficients
            * (
                self.x2[..., None] ** (powers + 1)
                - self.x1[..., None] ** (powers + 1)
            )
            / (powers + 1),
            dim=-1,
        )
        self.average = torch.where(
            self.zero_mask,
            torch.zeros_like(integral),
            integral / safe_length,
        )

        self.potential = matter_potential(
            self.average,
            antinu=antinu,
            evolution_scale_m=evolution_scale,
            context=RuntimeContext(device=device, dtype=dtype),
            legacy_precision=legacy_precision,
        )
        self.potential = torch.where(
            self.zero_mask,
            torch.zeros_like(self.potential),
            self.potential,
        )

    def lift_to_matrix_batch(
        self,
        x: torch.Tensor,
        target_ndim: int,
    ) -> torch.Tensor:
        """Reshape a scalar-batch tensor to broadcast with matrix batches.

        Appends singleton dimensions so a per-segment scalar (such as ``x1``
        or ``x2``, shaped ``(...,)``) can broadcast against a tensor whose
        last two dimensions index spectral eigenvalue pairs (shaped
        ``(..., 3, 3)``), as used by ``residual_integral``.

        Args:
            x: Tensor with one value per segment/batch element.
            target_ndim: Number of dimensions of the matrix-batched tensor
                ``x`` must broadcast against (including the trailing 3x3
                pair dimensions).

        Returns:
            ``x`` reshaped to ``(..., 1, 1)`` with enough leading singleton
            dimensions inserted to reach ``target_ndim``.
        """
        while x.ndim < target_ndim - 2:
            x = x.unsqueeze(-1)

        return x.unsqueeze(-1).unsqueeze(-1)

    @staticmethod
    def oscillatory_monomial_integral(
        power: int,
        x1: torch.Tensor,
        x2: torch.Tensor,
        frequency: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``int_x1^x2 x**power exp(i*frequency*x) dx``.

        Args:
            power: Non-negative integer monomial power.
            x1: Lower integration endpoint.
            x2: Upper integration endpoint.
            frequency: Non-zero oscillation frequency.

        Returns:
            Complex definite integral broadcast over the inputs.
        """
        total = torch.zeros_like(frequency)
        i_frequency = 1j * frequency
        exp_x2 = torch.exp(i_frequency * x2)
        exp_x1 = torch.exp(i_frequency * x1)

        for order in range(power + 1):
            factor = (
                (-1) ** order
                * factorial(power)
                / factorial(power - order)
                / i_frequency ** (order + 1)
            )
            total = total + factor * (
                x2 ** (power - order) * exp_x2
                - x1 ** (power - order) * exp_x1
            )

        return total

    @staticmethod
    def taylor_monomial_integral(
        power: int,
        x1: torch.Tensor,
        x2: torch.Tensor,
        frequency: torch.Tensor,
        *,
        order: int = 2,
    ) -> torch.Tensor:
        """Taylor expansion of the oscillatory monomial integral.

        Expands ``exp(i*frequency*x)`` in its Taylor series in ``frequency``
        up to ``order`` and integrates the resulting polynomial term by term,
        approximating ``int_x1^x2 x**power exp(i*frequency*x) dx`` for small
        ``frequency`` (i.e. near-degenerate spectral eigenvalues), where the
        closed-form expression in ``oscillatory_monomial_integral`` becomes
        numerically unstable due to division by ``frequency**(order+1)``.

        Args:
            power: Non-negative integer monomial power.
            x1: Lower integration endpoint.
            x2: Upper integration endpoint.
            frequency: Small oscillation frequency (here, an eigenvalue
                difference ``la - lb``) being expanded around zero.
            order: Truncation order of the Taylor series in ``frequency``.

        Returns:
            Complex approximate definite integral broadcast over the inputs.
        """
        total = torch.zeros_like(frequency)
        for series_order in range(order + 1):
            total = total + (
                (1j * frequency) ** series_order
                / factorial(series_order)
                * (
                    x2 ** (power + series_order + 1)
                    - x1 ** (power + series_order + 1)
                )
                / (power + series_order + 1)
            )

        return total

    @torch.no_grad()
    def residual_integral(
        self,
        la: torch.Tensor,
        lb: torch.Tensor,
        *,
        small_ratio: float = 1.0e-2,
        dl_zero_eps: float = 0.0,
    ) -> torch.Tensor:
        """Evaluate the spectral oscillatory integral for this profile.

        The residual profile is the even-power polynomial after subtracting
        its segment average from the constant coefficient, i.e.
        ``delta_n(x) = n_e(x) - average``. This computes, for each pair of
        Hamiltonian eigenvalues ``(la, lb)``, the integral

            I_ab = int_{x1}^{x2} delta_n(x) exp(-i*la*x + i*lb*x) dx,

        which is the quantity required by ``evolutor_first_order`` to build
        the first-order perturbative correction U1 from the residual density.
        For ordinary spectral gaps (``la - lb`` not small relative to
        ``la + lb``) the integral is evaluated in closed form via
        ``oscillatory_monomial_integral``. Near degeneracy (small relative
        gap) the method switches to a Taylor expansion in ``la - lb`` via
        ``taylor_monomial_integral`` to avoid the numerical instability of
        the closed form as the frequency approaches zero. Exactly degenerate
        entries (``la == lb``) are set to zero, matching the convention used
        for the original even-power model in the quartic limit (no
        first-order phase accumulates between a mode and itself).

        Args:
            la: "Bra" Hamiltonian eigenvalue(s), shaped to broadcast with the
                3x3 eigenvalue-pair grid (e.g. ``(..., 3, 1)``).
            lb: "Ket" Hamiltonian eigenvalue(s), shaped to broadcast with the
                3x3 eigenvalue-pair grid (e.g. ``(..., 1, 3)``).
            small_ratio: Threshold on ``|la - lb| / |la + lb|`` below which
                the Taylor expansion is used instead of the closed form.
            dl_zero_eps: Optional absolute tolerance for treating ``la - lb``
                as exactly zero (degenerate). When zero (default), only exact
                equality triggers the zero-output branch.

        Returns:
            Complex residual oscillatory integral I_ab, broadcast over the
            input eigenvalue shapes with an added trailing coefficient/segment
            batch structure matching ``self.coefficients``.
        """
        if not torch.is_complex(la):
            cdtype = torch.complex128 if la.dtype == torch.float64 else torch.complex64
            la = la.to(dtype=cdtype)
        if not torch.is_complex(lb):
            lb = lb.to(dtype=la.dtype)

        target_ndim = la.ndim
        x1 = self.lift_to_matrix_batch(self.x1, target_ndim).to(
            device=la.device,
            dtype=la.dtype,
        )
        x2 = self.lift_to_matrix_batch(self.x2, target_ndim).to(
            device=la.device,
            dtype=la.dtype,
        )
        coefficients = self.coefficients.clone()
        coefficients[..., 0] = coefficients[..., 0] - self.average

        while coefficients.ndim < target_ndim - 1:
            coefficients = coefficients.unsqueeze(-2)
        coefficients = coefficients.unsqueeze(-1).unsqueeze(-1).to(
            device=la.device,
            dtype=la.dtype,
        )

        dl = la - lb
        if dl_zero_eps > 0.0:
            is_zero = torch.abs(dl) < dl_zero_eps
        else:
            is_zero = dl == 0

        denominator = la + lb
        ratio = torch.where(
            torch.abs(denominator) > 0,
            torch.abs(dl / denominator),
            torch.full_like(torch.abs(dl), float("inf")),
        )
        is_small = ratio < small_ratio

        dl_poly = dl.unsqueeze(-3)
        safe_dl = torch.where(
            is_small.unsqueeze(-3) | is_zero.unsqueeze(-3),
            torch.ones_like(dl_poly),
            dl_poly,
        )
        phase = torch.exp(-1j * la * x2 + 1j * lb * x1)

        integral_full = torch.zeros_like(coefficients[..., 0, :, :])
        integral_taylor = torch.zeros_like(integral_full)
        for index in range(coefficients.shape[-3]):
            power = 2 * index
            coeff = coefficients[..., index, :, :]
            integral_full = integral_full + coeff * self.oscillatory_monomial_integral(
                power,
                x1.unsqueeze(-3),
                x2.unsqueeze(-3),
                safe_dl,
            ).squeeze(-3)
            integral_taylor = integral_taylor + coeff * self.taylor_monomial_integral(
                power,
                x1.unsqueeze(-3),
                x2.unsqueeze(-3),
                dl_poly,
            ).squeeze(-3)

        out = phase * torch.where(is_small, integral_taylor, integral_full)
        return torch.where(is_zero, torch.zeros_like(out), out)

    def has_perturbation(self) -> torch.Tensor:
        """Return a mask selecting segments with non-constant residual terms.

        A segment whose density is constant (``b = c = deltas = 0``) has zero
        residual ``delta_n(x) = n_e(x) - average`` everywhere, so its
        first-order correction U1 vanishes identically and can be skipped.
        This checks whether any non-constant coefficient (quadratic, quartic,
        or higher even powers) is non-zero.

        Returns:
            Boolean tensor shaped ``(...,)`` (one entry per segment), True
            where at least one non-constant coefficient is non-zero.
        """
        return torch.any(torch.abs(self.coefficients[..., 1:]) > 0, dim=-1)

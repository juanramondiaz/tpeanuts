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

from tpeanuts.core.common.potential import matter_potential_cc, matter_potential_nc
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
        *,
        coefficients_n: TensorLike | None = None,
        **kwargs,
    ) -> "EvenPowerProfileSegment":
        """Build a segment from a full even-power coefficient tensor.

        Args:
            x1: Initial coordinate normalized by ``profile_scale_m``.
            x2: Final coordinate normalized by ``profile_scale_m``.
            coefficients: Tensor with final dimension ordered as
                ``a, b, c, delta1, ...``.
            coefficients_n: Optional neutron-density coefficient tensor,
                same layout as ``coefficients``, enabling the 3+1 sterile
                extension's neutral-current matter term.
            **kwargs: Additional constructor options.

        Returns:
            Initialized segment profile.
        """
        coefficients = torch.as_tensor(coefficients)
        neutron_kwargs = {}
        if coefficients_n is not None:
            coefficients_n = torch.as_tensor(coefficients_n)
            neutron_kwargs = dict(
                a_n=coefficients_n[..., 0],
                b_n=coefficients_n[..., 1],
                c_n=coefficients_n[..., 2],
                deltas_n=coefficients_n[..., 3:],
            )
        return cls(
            x1=x1,
            x2=x2,
            a=coefficients[..., 0],
            b=coefficients[..., 1],
            c=coefficients[..., 2],
            deltas=coefficients[..., 3:],
            **neutron_kwargs,
            **kwargs,
        )

    @classmethod
    def constant(
        cls,
        x1: TensorLike,
        x2: TensorLike,
        density: TensorLike,
        *,
        density_n: TensorLike | None = None,
        **kwargs,
    ) -> "EvenPowerProfileSegment":
        """Build a constant-density segment profile.

        Args:
            x1: Initial coordinate normalized by ``profile_scale_m``.
            x2: Final coordinate normalized by ``profile_scale_m``.
            density: Constant electron density.
            density_n: Optional constant neutron density, enabling the 3+1
                sterile extension's neutral-current matter term.
            **kwargs: Additional constructor options.

        Returns:
            Initialized constant segment profile.
        """
        density_tensor = torch.as_tensor(density)
        zeros = torch.zeros_like(density_tensor)
        neutron_kwargs = {}
        if density_n is not None:
            density_n_tensor = torch.as_tensor(density_n)
            neutron_kwargs = dict(
                a_n=density_n_tensor,
                b_n=torch.zeros_like(density_n_tensor),
                c_n=torch.zeros_like(density_n_tensor),
            )
        return cls(
            x1=x1,
            x2=x2,
            a=density_tensor,
            b=zeros,
            c=zeros,
            _known_constant=True,
            **neutron_kwargs,
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
        a_n: TensorLike | None = None,
        b_n: TensorLike | None = None,
        c_n: TensorLike | None = None,
        deltas_n: TensorLike | None = None,
        antinu: Union[bool, torch.Tensor] = False,
        profile_scale_m: TensorLike = R_E,
        evolution_scale_m: TensorLike = R_E,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        legacy_precision: bool = False,
        _known_constant: bool = False,
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
        ratio_sq = ratio * ratio
        b = torch.broadcast_to(b, base_shape) * ratio_sq
        c = torch.broadcast_to(c, base_shape) * (ratio_sq * ratio_sq)
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

        self.potential = matter_potential_cc(
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

        self.coefficients_n = None
        self.average_n = None
        self.potential_n = None
        if a_n is not None:
            a_n = torch.broadcast_to(as_tensor(a_n, device=device, dtype=dtype), base_shape)
            b_n = torch.broadcast_to(as_tensor(b_n, device=device, dtype=dtype), base_shape) * ratio_sq
            c_n = torch.broadcast_to(as_tensor(c_n, device=device, dtype=dtype), base_shape) * (ratio_sq * ratio_sq)
            if deltas_n is None:
                deltas_n = torch.zeros((*base_shape, self.deltas.shape[-1]), device=device, dtype=dtype)
            else:
                deltas_n = torch.broadcast_to(
                    as_tensor(deltas_n, device=device, dtype=dtype), (*base_shape, deltas_n.shape[-1])
                )
                powers_n = 2 * torch.arange(3, 3 + deltas_n.shape[-1], device=device, dtype=dtype)
                deltas_n = deltas_n * ratio[..., None] ** powers_n

            self.coefficients_n = torch.cat(
                [a_n[..., None], b_n[..., None], c_n[..., None], deltas_n], dim=-1,
            )

            powers_avg_n = 2 * torch.arange(self.coefficients_n.shape[-1], device=device, dtype=dtype)
            integral_n = torch.sum(
                self.coefficients_n
                * (
                    self.x2[..., None] ** (powers_avg_n + 1)
                    - self.x1[..., None] ** (powers_avg_n + 1)
                )
                / (powers_avg_n + 1),
                dim=-1,
            )
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
                self.zero_mask,
                torch.zeros_like(self.potential_n),
                self.potential_n,
            )

        # Structural metadata used by the evolutor without reading a CUDA
        # tensor back to the host. Only the constant constructor can assert
        # this without inspecting coefficient values.
        self.any_perturbation = False if _known_constant else None

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

    @staticmethod
    def oscillatory_even_monomial_integrals(
        n_terms: int,
        x1: torch.Tensor,
        x2: torch.Tensor,
        frequency: torch.Tensor,
    ) -> torch.Tensor:
        """Compute all even monomial integrals with one recurrence.

        Returns integrals for powers ``0, 2, ..., 2*(n_terms-1)`` stacked on
        the third-to-last dimension. Exponentials and reciprocal frequency
        are evaluated once for the complete coefficient batch.
        """
        if n_terms <= 0:
            return frequency.new_empty((*frequency.shape[:-2], 0, *frequency.shape[-2:]))

        i_frequency = 1j * frequency
        inv_i_frequency = i_frequency.reciprocal()
        exp_x1 = torch.exp(i_frequency * x1)
        exp_x2 = torch.exp(i_frequency * x2)

        integral = (exp_x2 - exp_x1) * inv_i_frequency
        even_integrals = [integral]
        x1_power = torch.ones_like(x1)
        x2_power = torch.ones_like(x2)

        for power in range(1, 2 * n_terms - 1):
            x1_power = x1_power * x1
            x2_power = x2_power * x2
            boundary = (x2_power * exp_x2 - x1_power * exp_x1) * inv_i_frequency
            integral = boundary - power * inv_i_frequency * integral
            if power % 2 == 0:
                even_integrals.append(integral)

        return torch.stack(even_integrals, dim=-3)

    @staticmethod
    def taylor_even_monomial_integrals(
        n_terms: int,
        x1: torch.Tensor,
        x2: torch.Tensor,
        frequency: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the order-two Taylor integrals for all even monomials.

        Endpoint powers are generated once by recurrence and reused by every
        polynomial coefficient.
        """
        if n_terms <= 0:
            return frequency.new_empty((*frequency.shape[:-2], 0, *frequency.shape[-2:]))

        max_power = 2 * (n_terms - 1) + 3
        x1_powers = [torch.ones_like(x1)]
        x2_powers = [torch.ones_like(x2)]
        for _ in range(max_power):
            x1_powers.append(x1_powers[-1] * x1)
            x2_powers.append(x2_powers[-1] * x2)

        i_frequency = 1j * frequency
        frequency_second = 0.5 * i_frequency * i_frequency
        integrals = []
        for index in range(n_terms):
            power = 2 * index
            term0 = (x2_powers[power + 1] - x1_powers[power + 1]) / (power + 1)
            term1 = i_frequency * (
                x2_powers[power + 2] - x1_powers[power + 2]
            ) / (power + 2)
            term2 = frequency_second * (
                x2_powers[power + 3] - x1_powers[power + 3]
            ) / (power + 3)
            integrals.append(term0 + term1 + term2)

        return torch.stack(integrals, dim=-3)

    @torch.no_grad()
    def residual_integral(
        self,
        la: torch.Tensor,
        lb: torch.Tensor,
        *,
        coefficients: torch.Tensor | None = None,
        average: torch.Tensor | None = None,
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
            coefficients: Optional override for the polynomial coefficients,
                same layout as ``self.coefficients``. Defaults to
                ``self.coefficients`` (electron density); pass
                ``self.coefficients_n`` (with ``average=self.average_n``) to
                compute the neutron-density residual integral instead,
                enabling the 3+1 sterile extension's neutral-current matter
                term.
            average: Optional override for the segment average paired with
                ``coefficients``. Defaults to ``self.average``.
            small_ratio: Threshold on ``|la - lb| / |la + lb|`` below which
                the Taylor expansion is used instead of the closed form.
            dl_zero_eps: Optional absolute tolerance for treating ``la - lb``
                as exactly zero (degenerate). When zero (default), only exact
                equality triggers the zero-output branch.

        Returns:
            Complex residual oscillatory integral I_ab, broadcast over the
            input eigenvalue shapes with an added trailing coefficient/segment
            batch structure matching ``coefficients``.
        """
        if not torch.is_complex(la):
            cdtype = torch.complex128 if la.dtype == torch.float64 else torch.complex64
            la = la.to(dtype=cdtype)
        if not torch.is_complex(lb):
            lb = lb.to(dtype=la.dtype)

        if coefficients is None:
            coefficients = self.coefficients
        if average is None:
            average = self.average

        target_ndim = la.ndim
        # The segment constructor already guarantees a common device. Convert
        # each lifted tensor once to the spectral complex dtype: leaving it
        # real would trigger an implicit real-to-complex copy in every term of
        # the polynomial loop below.
        if self.x1.device != la.device or coefficients.device != la.device:
            raise ValueError("Profile geometry and eigenvalues must share a device.")
        x1 = self.lift_to_matrix_batch(self.x1, target_ndim).to(dtype=la.dtype)
        x2 = self.lift_to_matrix_batch(self.x2, target_ndim).to(dtype=la.dtype)

        while coefficients.ndim < target_ndim - 1:
            coefficients = coefficients.unsqueeze(-2)
        coefficients = coefficients.unsqueeze(-1).unsqueeze(-1).to(dtype=la.dtype)

        while average.ndim < target_ndim - 2:
            average = average.unsqueeze(-1)
        average = average.unsqueeze(-1).unsqueeze(-1).to(dtype=la.dtype)

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

        # The Taylor branch is only ever read out where `is_small` is True
        # (see the final torch.where below), so when no eigenvalue pair in
        # this batch is near-degenerate its value is fully discarded. Skip
        # computing it in that (common) case to avoid the extra per-term
        # Taylor-series loop below.
        # Exactly degenerate pairs are forced to zero below, so they must not
        # make the diagonal of every spectral matrix trigger the Taylor path.
        # On CUDA this host decision costs ~0.08 ms for representative batches,
        # versus ~0.55 ms or more for the Taylor batch it can avoid.
        needs_taylor = bool(torch.any(is_small & ~is_zero))

        n_terms = coefficients.shape[-3]
        monomial_full = self.oscillatory_even_monomial_integrals(
            n_terms,
            x1,
            x2,
            safe_dl.squeeze(-3),
        )
        integral_full = torch.sum(coefficients * monomial_full, dim=-3)
        integral_full = integral_full - average * monomial_full[..., 0, :, :]

        integral_taylor = None
        if needs_taylor:
            monomial_taylor = self.taylor_even_monomial_integrals(
                n_terms,
                x1,
                x2,
                dl_poly.squeeze(-3),
            )
            integral_taylor = torch.sum(coefficients * monomial_taylor, dim=-3)
            integral_taylor = integral_taylor - average * monomial_taylor[..., 0, :, :]

        out = phase * (integral_full if not needs_taylor else torch.where(is_small, integral_taylor, integral_full))
        return torch.where(is_zero, torch.zeros_like(out), out)

    @torch.no_grad()
    def residual_integral_neutron(
        self,
        la: torch.Tensor,
        lb: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Neutron-density counterpart of ``residual_integral``.

        Enables the 3+1 sterile extension's neutral-current matter term (see
        ``core.perturbative.evolutor.evolutor_first_order``). Requires
        neutron-density coefficients (see ``__init__``'s
        ``a_n``/``b_n``/``c_n``/``deltas_n`` or ``from_coefficients``'s
        ``coefficients_n``).

        Raises:
            ValueError: If this segment has no neutron-density coefficients.
        """
        if self.coefficients_n is None:
            raise ValueError(
                "This segment has no neutron-density coefficients. Build it "
                "with a_n/b_n/c_n (or coefficients_n via from_coefficients) "
                "to enable the 3+1 sterile extension's neutral-current "
                "matter term."
            )
        return self.residual_integral(
            la, lb, coefficients=self.coefficients_n, average=self.average_n, **kwargs,
        )

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

    def has_perturbation_neutron(self) -> torch.Tensor:
        """Return a mask selecting segments with a non-constant neutron residual.

        Mirrors ``has_perturbation`` for ``coefficients_n``. Requires
        neutron-density coefficients (see ``__init__``'s ``a_n``/``b_n``/
        ``c_n``/``deltas_n`` or ``from_coefficients``'s ``coefficients_n``).

        Returns:
            Boolean tensor shaped ``(...,)``.

        Raises:
            ValueError: If this segment has no neutron-density coefficients.
        """
        if self.coefficients_n is None:
            raise ValueError(
                "This segment has no neutron-density coefficients. Build it "
                "with a_n/b_n/c_n (or coefficients_n via from_coefficients) "
                "to enable the 3+1 sterile extension's neutral-current "
                "matter term."
            )
        return torch.any(torch.abs(self.coefficients_n[..., 1:]) > 0, dim=-1)

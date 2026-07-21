"""General local-polynomial model for perturbative atmosphere segments.

Module classes:
    AtmospherePolynomialSegment
        Supply average density and analytic residual integrals to the generic
        first-order perturbative evolutor.
"""

from __future__ import annotations

from math import factorial
from typing import Union

import torch

from tpeanuts.core.common.potential import matter_potential_cc, matter_potential_nc
from tpeanuts.util.constant import R_E
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.torch_util import infer_device_dtype
from tpeanuts.util.type import TensorLike, as_tensor


class AtmospherePolynomialSegment:
    """Polynomial density model for one or more atmosphere segments.

    The density is represented in the well-conditioned local coordinate
    ``q = (x-centre)/half_width`` as ``n_e(q)=sum_k c_k q**k``.

    Args:
        x1: Initial segment coordinate normalized by ``profile_scale_m``.
        x2: Final segment coordinate normalized by ``profile_scale_m``.
        coefficients: Local polynomial coefficients with final dimension
            ordered as ``q**0, q**1, ...``.
        coefficients_n: Optional neutron-density local polynomial
            coefficients, same layout as ``coefficients``, enabling the 3+1
            sterile extension's neutral-current matter term.
        antinu: Scalar or batched antineutrino selector.
        profile_scale_m: Physical scale defining the supplied coordinates.
        evolution_scale_m: Physical scale used by the Hamiltonian.
        legacy_precision: Use the legacy matter-potential prefactor.

    Attributes:
        coefficients: Polynomial density coefficients in local coordinates.
        average: Exact segment-average density of the fitted polynomial.
        length: Segment length in evolution coordinates.
        zero_mask: Mask of zero-length segments.
        potential: Matter potential evaluated at ``average``.
    """

    def __init__(
        self,
        x1: TensorLike,
        x2: TensorLike,
        coefficients: TensorLike,
        *,
        coefficients_n: TensorLike | None = None,
        antinu: Union[bool, torch.Tensor] = False,
        profile_scale_m: TensorLike = R_E,
        evolution_scale_m: TensorLike = R_E,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        legacy_precision: bool = False,
    ) -> None:
        device, dtype = infer_device_dtype(
            x1, x2, coefficients, profile_scale_m, evolution_scale_m,
            device=device, dtype=dtype,
        )
        x1 = as_tensor(x1, device=device, dtype=dtype)
        x2 = as_tensor(x2, device=device, dtype=dtype)
        coefficients = as_tensor(coefficients, device=device, dtype=dtype)
        profile_scale = as_tensor(profile_scale_m, device=device, dtype=dtype)
        evolution_scale = as_tensor(evolution_scale_m, device=device, dtype=dtype)
        if torch.any(profile_scale <= 0) or torch.any(evolution_scale <= 0):
            raise ValueError("Profile and evolution scales must be positive.")

        ratio = profile_scale / evolution_scale
        base = torch.broadcast_shapes(x1.shape, x2.shape, coefficients.shape[:-1], ratio.shape)
        self.x1 = torch.broadcast_to(x1 * ratio, base)
        self.x2 = torch.broadcast_to(x2 * ratio, base)
        self.coefficients = torch.broadcast_to(coefficients, (*base, coefficients.shape[-1]))
        self.length = self.x2 - self.x1
        self.zero_mask = self.length == 0
        self.centre = 0.5 * (self.x1 + self.x2)
        self.half_width = 0.5 * self.length

        powers = torch.arange(self.coefficients.shape[-1], device=device, dtype=dtype)
        integrals = torch.where(
            (powers.remainder(2) == 0),
            2.0 / (powers + 1.0),
            torch.zeros_like(powers),
        )
        self.average = 0.5 * torch.sum(self.coefficients * integrals, dim=-1)
        self.average = torch.where(self.zero_mask, torch.zeros_like(self.average), self.average)
        self.potential = matter_potential_cc(
            self.average,
            antinu=antinu,
            evolution_scale_m=evolution_scale,
            context=RuntimeContext(device=device, dtype=dtype),
            legacy_precision=legacy_precision,
        )
        self.potential = torch.where(self.zero_mask, torch.zeros_like(self.potential), self.potential)
        self.any_perturbation = False if coefficients.shape[-1] == 1 else None

        self.coefficients_n = None
        self.average_n = None
        self.potential_n = None
        if coefficients_n is not None:
            coefficients_n = as_tensor(coefficients_n, device=device, dtype=dtype)
            self.coefficients_n = torch.broadcast_to(coefficients_n, (*base, coefficients_n.shape[-1]))
            integrals_n = torch.where(
                (torch.arange(self.coefficients_n.shape[-1], device=device, dtype=dtype).remainder(2) == 0),
                2.0 / (torch.arange(self.coefficients_n.shape[-1], device=device, dtype=dtype) + 1.0),
                torch.zeros(self.coefficients_n.shape[-1], device=device, dtype=dtype),
            )
            self.average_n = 0.5 * torch.sum(self.coefficients_n * integrals_n, dim=-1)
            self.average_n = torch.where(self.zero_mask, torch.zeros_like(self.average_n), self.average_n)
            self.potential_n = matter_potential_nc(
                self.average_n,
                antinu=antinu,
                evolution_scale_m=evolution_scale,
                context=RuntimeContext(device=device, dtype=dtype),
            )
            self.potential_n = torch.where(
                self.zero_mask, torch.zeros_like(self.potential_n), self.potential_n
            )

    @staticmethod
    def _monomial_integrals(n_terms: int, frequency: torch.Tensor, *, taylor: bool) -> torch.Tensor:
        """Compute all ``int_-1^1 q**k exp(i*f*q)dq`` in one recurrence."""
        if taylor:
            values = []
            iw = 1j * frequency
            for power in range(n_terms):
                total = torch.zeros_like(frequency)
                for order in range(3):
                    exponent = power + order
                    if exponent % 2 == 0:
                        total = total + iw**order / factorial(order) * 2.0 / (exponent + 1)
                values.append(total)
            return torch.stack(values, dim=-3)

        iw = 1j * frequency
        inv = iw.reciprocal()
        ep, em = torch.exp(iw), torch.exp(-iw)
        current = (ep - em) * inv
        values = [current]
        for power in range(1, n_terms):
            boundary = (ep - ((-1) ** power) * em) * inv
            current = boundary - power * inv * current
            values.append(current)
        return torch.stack(values, dim=-3)

    @torch.no_grad()
    def residual_integral(
        self,
        la: torch.Tensor,
        lb: torch.Tensor,
        *,
        coefficients: torch.Tensor | None = None,
        average: torch.Tensor | None = None,
        small_ratio: float = 1.0e-2,
    ) -> torch.Tensor:
        """Return the oscillatory integral of ``n_e(x)-average``.

        Args:
            la: "Bra" Hamiltonian eigenvalue(s).
            lb: "Ket" Hamiltonian eigenvalue(s).
            coefficients: Optional override for the polynomial coefficients,
                same layout as ``self.coefficients``. Defaults to
                ``self.coefficients`` (electron density); pass
                ``self.coefficients_n`` (with ``average=self.average_n``) to
                compute the neutron-density residual integral instead (see
                ``residual_integral_neutron``).
            average: Optional override for the segment average paired with
                ``coefficients``. Defaults to ``self.average``.
            small_ratio: Threshold on ``|la - lb| / |la + lb|`` below which
                the Taylor expansion is used instead of the closed form.
        """
        if coefficients is None:
            coefficients = self.coefficients
        if average is None:
            average = self.average

        cdtype = torch.complex128 if la.dtype in (torch.float64, torch.complex128) else torch.complex64
        la, lb = la.to(cdtype), lb.to(cdtype)
        dl = la - lb
        is_zero = dl == 0
        denominator = la + lb
        ratio = torch.where(torch.abs(denominator) > 0, torch.abs(dl / denominator), torch.full_like(torch.abs(dl), float("inf")))
        is_small = ratio < small_ratio

        def lift(value: torch.Tensor) -> torch.Tensor:
            while value.ndim < la.ndim - 2:
                value = value.unsqueeze(-1)
            return value.unsqueeze(-1).unsqueeze(-1).to(cdtype)

        centre, half, average = lift(self.centre), lift(self.half_width), lift(average)
        while coefficients.ndim < la.ndim - 1:
            coefficients = coefficients.unsqueeze(-2)
        coefficients = coefficients.unsqueeze(-1).unsqueeze(-1).to(cdtype)
        frequency = dl * half
        safe_frequency = torch.where(is_small | is_zero, torch.ones_like(frequency), frequency)
        full = self._monomial_integrals(coefficients.shape[-3], safe_frequency, taylor=False)
        integral_full = torch.sum(coefficients * full, dim=-3) - average * full[..., 0, :, :]
        needs_taylor = bool(torch.any(is_small & ~is_zero))
        if needs_taylor:
            series = self._monomial_integrals(coefficients.shape[-3], frequency, taylor=True)
            integral_series = torch.sum(coefficients * series, dim=-3) - average * series[..., 0, :, :]
            integral = torch.where(is_small, integral_series, integral_full)
        else:
            integral = integral_full
        phase = torch.exp(-1j * la * lift(self.x2) + 1j * lb * lift(self.x1) + 1j * dl * centre)
        return torch.where(is_zero, torch.zeros_like(integral), phase * half * integral)

    @torch.no_grad()
    def residual_integral_neutron(self, la: torch.Tensor, lb: torch.Tensor, **kwargs) -> torch.Tensor:
        """Neutron-density counterpart of ``residual_integral``.

        Enables the 3+1 sterile extension's neutral-current matter term (see
        ``core.perturbative.evolutor.evolutor_first_order``). Requires
        ``coefficients_n`` to have been supplied at construction time.

        Raises:
            ValueError: If this segment has no neutron-density coefficients.
        """
        if self.coefficients_n is None:
            raise ValueError(
                "This segment has no neutron-density coefficients. Build it "
                "with coefficients_n to enable the 3+1 sterile extension's "
                "neutral-current matter term."
            )
        return self.residual_integral(
            la, lb, coefficients=self.coefficients_n, average=self.average_n, **kwargs,
        )

    def has_perturbation(self) -> torch.Tensor:
        """Return which fitted segments contain non-constant terms."""
        return torch.any(torch.abs(self.coefficients[..., 1:]) > 0, dim=-1)

    def has_perturbation_neutron(self) -> torch.Tensor:
        """Return which fitted segments contain a non-constant NC term.

        Raises:
            ValueError: If this segment has no neutron-density coefficients.
        """
        if self.coefficients_n is None:
            raise ValueError(
                "This segment has no neutron-density coefficients. Build it "
                "with coefficients_n to enable the 3+1 sterile extension's "
                "neutral-current matter term."
            )
        return torch.any(torch.abs(self.coefficients_n[..., 1:]) > 0, dim=-1)

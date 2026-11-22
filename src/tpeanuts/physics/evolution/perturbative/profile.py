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

"""Even-power density-profile model, for the analytic evolver's first-order correction.

Ported from ``v1_0/core/perturbative/models/even_power/profile_segment.py``,
trimmed of ``v1_0``-specific concerns that do not carry over: the
constructor no longer bakes in ``antinu`` or eagerly builds a
``matter_potential_cc``-derived ``potential`` field (this project always
defers ``antinu`` to call time -- see ``tpeanuts.physics.common.potential``'s
own functions -- so the caller computes the potential from ``average`` at
the point ``antinu`` is actually known, exactly like every other term in
this project does); ``legacy_precision`` has no v2 counterpart; and the
``@torch.no_grad()`` decorators v1_0 still carried on two methods here are
dropped entirely, per this project's own foundational rule that autograd is
never hardcoded inside a routine (``tpeanuts.util.context.RuntimeContext``'s
module docstring) -- whether gradients track through a profile's residual
integral is the caller's decision, not this module's.

The model represents one segment's density as

    n_e(x) = a + b*x**2 + c*x**4 + delta1*x**6 + delta2*x**8 + ...

-- deliberately independent of any medium geometry (Earth, atmosphere, or
any future medium builds one of these from its own trajectory/density
data; nothing here assumes a specific geometry), matching
``EvenPowerProfileSegment``'s own stated genericity in v1_0. Only the
evolution-side machinery and this one profile model are ported in this
pass -- the Earth/atmosphere-specific "layered" wrapper that turns a whole
multi-shell trajectory into a batch of these
(``v1_0/core/perturbative/models/even_power/profile_layered.py``) is not:
no real Earth medium exists in this project yet to consume it (see
``tpeanuts.medium``'s own module docstring).

``PolynomialProfile``, ported from
``v1_0/core/perturbative/models/atmosphere/profile_segment.py``, is a
second, structurally different density model needed by
``tpeanuts.medium.atmosphere``: a general (not just even-power)
polynomial in the *local* coordinate ``q = (x-centre)/half_width in
[-1,1]``, rather than ``EvenPowerProfile``'s even powers in the *global*
chord coordinate ``x``. This is not an arbitrary alternative -- Earth's
chord is symmetric about its own point of closest approach, so an
even-power fit in global ``x`` is the natural, exact representation
there; a single atmosphere-only segment has no such symmetric origin, so
odd powers are genuinely needed, and a normalized local coordinate keeps
the fit well-conditioned regardless of a segment's own absolute position
along the path. Both classes satisfy the same duck-typed
``perturbative_profile`` contract below; which one a medium uses is a
property of that medium's own geometry, not of this module.

Module contents:
    EvenPowerProfile
        Segment density model with even-power coefficients in the global
        chord coordinate, the segment-average density (electron and
        optional neutron), and the analytic spectral residual integral
        (``tpeanuts.physics.evolution.perturbative.analytic``'s first-order correction
        consumes ``residual_integral``/``residual_integral_neutron``/
        ``has_perturbation``/``has_perturbation_neutron``/``average``/
        ``length``/``zero_mask`` -- the same duck-typed contract
        ``tpeanuts.interfaces.medium.MediumProfile.perturbative_profile``
        documents).
    PolynomialProfile
        General-power segment density model in a local, per-segment
        coordinate -- the same duck-typed contract as ``EvenPowerProfile``.
"""

from __future__ import annotations

from math import factorial
from typing import Any, Dict, Optional, Tuple

import torch

from tpeanuts.util.constant import R_E
from tpeanuts.util.tensor import TensorLike, as_tensor, infer_device_dtype


class EvenPowerProfile:
    """Even-power density model for one perturbative-correction segment.

    Attributes:
        x1: Segment start coordinate, in evolution-scale units, shaped
            ``(...,)``.
        x2: Segment end coordinate, in evolution-scale units, shaped
            ``(...,)``.
        coefficients: Even-power electron-density coefficients in
            evolution-scale units, shaped ``(..., n_terms)``, ordered
            ``a, b, c, delta1, delta2, ...`` (powers ``x**0, x**2, x**4,
            x**6, ...``).
        length: ``x2 - x1``, shaped ``(...,)``.
        zero_mask: Boolean tensor shaped ``(...,)``, True where
            ``length == 0``.
        average: Segment-average electron density, shaped ``(...,)`` --
            exactly zero (not the polynomial's own value) on a zero-length
            segment.
        coefficients_n: Optional neutron-density counterpart of
            ``coefficients``, same shape/layout. None unless built with
            ``a_n``/``coefficients_n``.
        average_n: Optional neutron-density counterpart of ``average``.
            None unless ``coefficients_n`` is present.
    """

    @classmethod
    def from_coefficients(
        cls,
        x1: TensorLike,
        x2: TensorLike,
        coefficients: TensorLike,
        *,
        coefficients_n: Optional[TensorLike] = None,
        **kwargs,
    ) -> "EvenPowerProfile":
        """Build a segment from a full even-power coefficient tensor.

        Args:
            x1: Initial coordinate, normalized by ``profile_scale_m``.
            x2: Final coordinate, normalized by ``profile_scale_m``.
            coefficients: Tensor with final dimension ordered as
                ``a, b, c, delta1, ...``.
            coefficients_n: Optional neutron-density coefficient tensor,
                same layout as ``coefficients``, enabling the sterile
                neutral-current matter term.
            **kwargs: Forwarded to ``__init__`` (``profile_scale_m``,
                ``evolution_scale_m``, ``device``, ``dtype``).

        Returns:
            Initialized profile.
        """
        coefficients = as_tensor(coefficients)
        neutron_kwargs: Dict[str, Any] = {}
        if coefficients_n is not None:
            coefficients_n = as_tensor(coefficients_n)
            neutron_kwargs = dict(
                a_n=coefficients_n[..., 0],
                b_n=coefficients_n[..., 1],
                c_n=coefficients_n[..., 2],
                deltas_n=coefficients_n[..., 3:],
            )
        return cls(
            x1=x1, x2=x2,
            a=coefficients[..., 0], b=coefficients[..., 1], c=coefficients[..., 2],
            deltas=coefficients[..., 3:],
            **neutron_kwargs, **kwargs,
        )

    @classmethod
    def constant(
        cls,
        x1: TensorLike,
        x2: TensorLike,
        density: TensorLike,
        *,
        density_n: Optional[TensorLike] = None,
        **kwargs,
    ) -> "EvenPowerProfile":
        """Build a constant-density segment profile.

        A constant profile has ``has_perturbation() == False``
        everywhere, so ``tpeanuts.physics.evolution.perturbative.analytic``'s
        first-order correction is exactly zero for it -- the analytic
        evolver then reduces to exactly its own zero-order (constant
        -Hamiltonian) result, the same recovery discipline every optional
        feature in this project follows.

        Args:
            x1: Initial coordinate, normalized by ``profile_scale_m``.
            x2: Final coordinate, normalized by ``profile_scale_m``.
            density: Constant electron density.
            density_n: Optional constant neutron density, enabling the
                sterile neutral-current matter term.
            **kwargs: Forwarded to ``__init__``.

        Returns:
            Initialized constant profile.
        """
        density_tensor = as_tensor(density)
        zeros = torch.zeros_like(density_tensor)
        neutron_kwargs: Dict[str, Any] = {}
        if density_n is not None:
            density_n_tensor = as_tensor(density_n)
            neutron_kwargs = dict(
                a_n=density_n_tensor,
                b_n=torch.zeros_like(density_n_tensor),
                c_n=torch.zeros_like(density_n_tensor),
            )
        return cls(x1=x1, x2=x2, a=density_tensor, b=zeros, c=zeros, **neutron_kwargs, **kwargs)

    def __init__(
        self,
        x1: TensorLike,
        x2: TensorLike,
        a: TensorLike,
        b: TensorLike,
        c: TensorLike,
        *,
        deltas: Optional[TensorLike] = None,
        a_n: Optional[TensorLike] = None,
        b_n: Optional[TensorLike] = None,
        c_n: Optional[TensorLike] = None,
        deltas_n: Optional[TensorLike] = None,
        profile_scale_m: TensorLike = R_E,
        evolution_scale_m: TensorLike = R_E,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Build an even-power density profile for one segment.

        Args:
            x1: Initial coordinate, normalized by ``profile_scale_m``.
            x2: Final coordinate, normalized by ``profile_scale_m``.
            a: Constant electron-density coefficient, in ``profile_scale_m``
                coordinates.
            b: Quadratic coefficient.
            c: Quartic coefficient.
            deltas: Optional higher even-power coefficients; the last
                dimension enumerates ``x**6, x**8, ...``.
            a_n, b_n, c_n, deltas_n: Optional neutron-density counterparts
                of ``a``/``b``/``c``/``deltas``, enabling the sterile
                neutral-current matter term. ``a_n`` alone (with
                ``b_n``/``c_n``/``deltas_n`` omitted) is a shorthand for a
                constant neutron density.
            profile_scale_m: Length scale (metres) the coefficients and
                ``x1``/``x2`` are expressed in.
            evolution_scale_m: Length scale (metres) the returned
                ``x1``/``x2``/``coefficients``/``average`` are rescaled
                into -- must match whatever scale the generator this
                profile accompanies was itself built with (see
                ``tpeanuts.physics.common.potential``'s own
                ``evolution_scale_m`` convention).
            device: Optional explicit device; inferred from the tensor
                inputs when omitted.
            dtype: Optional explicit real dtype; inferred from the tensor
                inputs when omitted.

        Raises:
            ValueError: If ``profile_scale_m``/``evolution_scale_m`` is
                not positive.
        """
        infer_values: Tuple[Any, ...] = (x1, x2, a, b, c, profile_scale_m, evolution_scale_m)
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
            raise ValueError("profile_scale_m and evolution_scale_m must be positive.")

        ratio = evolution_scale / profile_scale
        base_shape = torch.broadcast_shapes(
            x1.shape, x2.shape, a.shape, b.shape, c.shape, ratio.shape,
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
            powers = 2 * torch.arange(3, 3 + deltas.shape[-1], device=device, dtype=dtype)
            deltas = deltas * ratio[..., None] ** powers
        assert isinstance(deltas, torch.Tensor)
        base_shape = torch.broadcast_shapes(
            x1.shape, x2.shape, a.shape, b.shape, c.shape, deltas.shape[:-1],
        )

        self.x1 = torch.broadcast_to(x1 / ratio, base_shape)
        self.x2 = torch.broadcast_to(x2 / ratio, base_shape)
        self.coefficients = torch.cat(
            [
                torch.broadcast_to(a, base_shape)[..., None],
                torch.broadcast_to(b, base_shape)[..., None],
                torch.broadcast_to(c, base_shape)[..., None],
                torch.broadcast_to(deltas, (*base_shape, deltas.shape[-1])),
            ],
            dim=-1,
        )

        self.length = self.x2 - self.x1
        self.zero_mask = self.length == 0
        safe_length = torch.where(self.zero_mask, torch.ones_like(self.length), self.length)

        powers = 2 * torch.arange(self.coefficients.shape[-1], device=device, dtype=dtype)
        integral = torch.sum(
            self.coefficients
            * (self.x2[..., None] ** (powers + 1) - self.x1[..., None] ** (powers + 1))
            / (powers + 1),
            dim=-1,
        )
        self.average = torch.where(self.zero_mask, torch.zeros_like(integral), integral / safe_length)

        self.coefficients_n: Optional[torch.Tensor] = None
        self.average_n: Optional[torch.Tensor] = None
        if a_n is not None:
            a_n = torch.broadcast_to(as_tensor(a_n, device=device, dtype=dtype), base_shape)
            # b_n/c_n default to zero -- a_n alone is a constant-neutron
            # -density shorthand, mirroring how deltas_n=None already
            # defaults to zero below.
            b_n_value = torch.zeros_like(a_n) if b_n is None else as_tensor(b_n, device=device, dtype=dtype)
            c_n_value = torch.zeros_like(a_n) if c_n is None else as_tensor(c_n, device=device, dtype=dtype)
            b_n = torch.broadcast_to(b_n_value, base_shape) * ratio_sq
            c_n = torch.broadcast_to(c_n_value, base_shape) * (ratio_sq * ratio_sq)
            if deltas_n is None:
                deltas_n = torch.zeros((*base_shape, self.coefficients.shape[-1] - 3), device=device, dtype=dtype)
            else:
                deltas_n_input = as_tensor(deltas_n, device=device, dtype=dtype)
                deltas_n = torch.broadcast_to(deltas_n_input, (*base_shape, deltas_n_input.shape[-1]))
                powers_n = 2 * torch.arange(3, 3 + deltas_n.shape[-1], device=device, dtype=dtype)
                deltas_n = deltas_n * ratio[..., None] ** powers_n

            assert isinstance(deltas_n, torch.Tensor) and b_n is not None and c_n is not None
            self.coefficients_n = torch.cat([a_n[..., None], b_n[..., None], c_n[..., None], deltas_n], dim=-1)

            powers_n_avg = 2 * torch.arange(self.coefficients_n.shape[-1], device=device, dtype=dtype)
            integral_n = torch.sum(
                self.coefficients_n
                * (self.x2[..., None] ** (powers_n_avg + 1) - self.x1[..., None] ** (powers_n_avg + 1))
                / (powers_n_avg + 1),
                dim=-1,
            )
            self.average_n = torch.where(self.zero_mask, torch.zeros_like(integral_n), integral_n / safe_length)

    def _lift_to_matrix_batch(self, x: torch.Tensor, target_ndim: int) -> torch.Tensor:
        """Reshape a per-segment scalar tensor to broadcast against a ``(..., N, N)`` eigenvalue-pair grid.

        Args:
            x: Tensor with one value per segment/batch element, shaped
                ``(...,)``.
            target_ndim: Number of dimensions of the matrix-batched tensor
                ``x`` must broadcast against (including the trailing
                ``N, N`` pair dimensions).

        Returns:
            ``x`` reshaped to ``(..., 1, 1)``, with enough leading
            singleton dimensions inserted to reach ``target_ndim``.
        """
        while x.ndim < target_ndim - 2:
            x = x.unsqueeze(-1)
        return x.unsqueeze(-1).unsqueeze(-1)

    @staticmethod
    def _oscillatory_even_monomial_integrals(
        n_terms: int, x1: torch.Tensor, x2: torch.Tensor, frequency: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``int_x1^x2 x**power exp(i*frequency*x) dx`` for every even power 0, 2, ..., 2*(n_terms-1) at once.

        One shared recurrence (integration by parts, descending power),
        rather than ``n_terms`` independent closed-form evaluations:
        exponentials and the reciprocal frequency are computed once for
        the whole batch. Verified (this module's own tests) against the
        single-power closed form term by term.

        Returns:
            Complex tensor shaped ``(..., n_terms, N, N)`` -- the new
            ``n_terms`` axis inserted at position -3, immediately before
            ``frequency``'s own trailing ``(N, N)`` eigenvalue-pair grid
            (matching ``coefficients``'s own ``(..., n_terms, 1, 1)``
            shape in ``residual_integral``, so the two broadcast-multiply
            directly).
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
    def _taylor_even_monomial_integrals(
        n_terms: int, x1: torch.Tensor, x2: torch.Tensor, frequency: torch.Tensor,
    ) -> torch.Tensor:
        """Order-2 Taylor-in-``frequency`` approximation of ``_oscillatory_even_monomial_integrals``.

        Used near a (near-)degenerate eigenvalue pair, where the closed
        form above loses precision from dividing by a near-zero
        ``frequency``. Endpoint powers are generated once by recurrence
        and reused for every polynomial term.

        Returns:
            Complex tensor shaped ``(..., n_terms, N, N)`` -- same axis
            convention as ``_oscillatory_even_monomial_integrals``.
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
            term1 = i_frequency * (x2_powers[power + 2] - x1_powers[power + 2]) / (power + 2)
            term2 = frequency_second * (x2_powers[power + 3] - x1_powers[power + 3]) / (power + 3)
            integrals.append(term0 + term1 + term2)

        return torch.stack(integrals, dim=-3)

    def residual_integral(
        self,
        la: torch.Tensor,
        lb: torch.Tensor,
        *,
        coefficients: Optional[torch.Tensor] = None,
        average: Optional[torch.Tensor] = None,
        small_ratio: float = 1.0e-2,
    ) -> torch.Tensor:
        """Evaluate the spectral oscillatory integral of the residual density for this segment.

        The residual profile is the even-power polynomial after
        subtracting its own segment average, ``delta_n(x) = n_e(x) -
        average``. For each pair of Hamiltonian eigenvalues ``(la, lb)``,
        this computes

            I_ab = int_{x1}^{x2} delta_n(x) exp(-i*la*x + i*lb*x) dx,

        the quantity ``tpeanuts.physics.evolution.perturbative.analytic``'s first-order
        correction needs. For ordinary spectral gaps (``la - lb`` not
        small relative to ``la + lb``) the integral is evaluated in closed
        form; near degeneracy (small relative gap) it switches to a
        Taylor expansion in ``la - lb`` to avoid the closed form's
        numerical instability as the frequency approaches zero. Exactly
        degenerate entries (``la == lb``) are exactly zero (no first-order
        phase accumulates between a mode and itself).

        **Known conditioning characteristic, inherited unchanged from
        v1_0 (not something specific to this port):** the closed-form
        branch's degeneracy test is purely *relative*
        (``|la - lb| / |la + lb|``), not absolute. A pair whose gap
        ``la - lb`` is small in absolute terms but not small *relative* to
        ``la + lb`` (e.g. both eigenvalues themselves tiny, of comparable
        magnitude with opposite sign) still takes the closed-form branch,
        whose recurrence divides by that small absolute frequency
        repeatedly -- confirmed directly to amplify float32 rounding by
        many orders of magnitude in that specific regime (see
        ``tpeanuts.physics.evolution.test.test_analytic``'s own
        ``test_perturbative_stays_close_to_unitary_for_a_small_perturbation``
        docstring for a worked example). This is not triggered by ordinary
        physical eigenvalues (kinetic + matter terms at this project's
        usual ``evolution_scale_m`` are not simultaneously near-zero and
        near-degenerate in this specific relative sense); it is called out
        here as a known characteristic to be aware of, not fixed, since
        doing so would mean deviating from v1_0's own validated math
        rather than porting it.

        Args:
            la: "Bra" Hamiltonian eigenvalue(s), shaped to broadcast with
                the ``N x N`` eigenvalue-pair grid (e.g. ``(..., N, 1)``).
            lb: "Ket" Hamiltonian eigenvalue(s), shaped to broadcast with
                the ``N x N`` eigenvalue-pair grid (e.g. ``(..., 1, N)``).
            coefficients: Optional override for the polynomial
                coefficients, same layout as ``self.coefficients``.
                Defaults to ``self.coefficients`` (electron density); pass
                ``self.coefficients_n`` (with ``average=self.average_n``)
                to compute the neutron-density residual integral instead
                (see ``residual_integral_neutron``).
            average: Optional override for the segment average paired
                with ``coefficients``. Defaults to ``self.average``.
            small_ratio: Threshold on ``|la - lb| / |la + lb|`` below which
                the Taylor expansion is used instead of the closed form.

        Returns:
            Complex residual oscillatory integral I_ab, broadcast over the
            input eigenvalue shapes.

        Raises:
            ValueError: If ``la``'s device does not match this profile's
                own.
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
        if self.x1.device != la.device or coefficients.device != la.device:
            raise ValueError("Profile geometry and eigenvalues must share a device.")

        target_ndim = la.ndim
        x1 = self._lift_to_matrix_batch(self.x1, target_ndim).to(dtype=la.dtype)
        x2 = self._lift_to_matrix_batch(self.x2, target_ndim).to(dtype=la.dtype)

        while coefficients.ndim < target_ndim - 1:
            coefficients = coefficients.unsqueeze(-2)
        coefficients = coefficients.unsqueeze(-1).unsqueeze(-1).to(dtype=la.dtype)

        while average.ndim < target_ndim - 2:
            average = average.unsqueeze(-1)
        average = average.unsqueeze(-1).unsqueeze(-1).to(dtype=la.dtype)

        dl = la - lb
        is_zero = dl == 0

        denominator = la + lb
        ratio = torch.where(
            torch.abs(denominator) > 0, torch.abs(dl / denominator), torch.full_like(torch.abs(dl), float("inf")),
        )
        is_small = ratio < small_ratio

        dl_poly = dl.unsqueeze(-3)
        safe_dl = torch.where(
            is_small.unsqueeze(-3) | is_zero.unsqueeze(-3), torch.ones_like(dl_poly), dl_poly,
        )
        phase = torch.exp(-1j * la * x2 + 1j * lb * x1)

        # The Taylor branch is only ever read out where is_small is True
        # (see the final torch.where below), so skip computing it entirely
        # when no eigenvalue pair in this batch is near-degenerate.
        needs_taylor = bool(torch.any(is_small & ~is_zero))

        n_terms = coefficients.shape[-3]
        monomial_full = self._oscillatory_even_monomial_integrals(n_terms, x1, x2, safe_dl.squeeze(-3))
        integral_full = torch.sum(coefficients * monomial_full, dim=-3)
        integral_full = integral_full - average * monomial_full[..., 0, :, :]

        if needs_taylor:
            monomial_taylor = self._taylor_even_monomial_integrals(n_terms, x1, x2, dl_poly.squeeze(-3))
            integral_taylor = torch.sum(coefficients * monomial_taylor, dim=-3)
            integral_taylor = integral_taylor - average * monomial_taylor[..., 0, :, :]
            result = phase * torch.where(is_small, integral_taylor, integral_full)
        else:
            result = phase * integral_full

        return torch.where(is_zero, torch.zeros_like(result), result)

    def residual_integral_neutron(self, la: torch.Tensor, lb: torch.Tensor, **kwargs) -> torch.Tensor:
        """Neutron-density counterpart of ``residual_integral``.

        Enables the sterile neutral-current matter term
        (``tpeanuts.physics.extensions.sterile.SterileNCTerm``) in the
        first-order correction. Requires neutron-density coefficients (see
        ``__init__``'s ``a_n``/``b_n``/``c_n``/``deltas_n`` or
        ``from_coefficients``'s ``coefficients_n``).

        Raises:
            ValueError: If this profile has no neutron-density coefficients.
        """
        if self.coefficients_n is None:
            raise ValueError(
                "This profile has no neutron-density coefficients. Build it "
                "with a_n/b_n/c_n (or coefficients_n via from_coefficients) "
                "to enable the sterile neutral-current matter term."
            )
        return self.residual_integral(la, lb, coefficients=self.coefficients_n, average=self.average_n, **kwargs)

    def has_perturbation(self) -> torch.Tensor:
        """Return a mask selecting segments with a non-constant electron-density residual.

        A segment whose density is constant (``b = c = deltas = 0``) has
        zero residual everywhere, so the first-order correction vanishes
        identically and can be skipped.

        Returns:
            Boolean tensor shaped ``(...,)``, True where at least one
            non-constant coefficient is non-zero.
        """
        return torch.any(torch.abs(self.coefficients[..., 1:]) > 0, dim=-1)

    def has_perturbation_neutron(self) -> torch.Tensor:
        """Neutron-density counterpart of ``has_perturbation``.

        Raises:
            ValueError: If this profile has no neutron-density coefficients.
        """
        if self.coefficients_n is None:
            raise ValueError(
                "This profile has no neutron-density coefficients. Build it "
                "with a_n/b_n/c_n (or coefficients_n via from_coefficients) "
                "to enable the sterile neutral-current matter term."
            )
        return torch.any(torch.abs(self.coefficients_n[..., 1:]) > 0, dim=-1)


class PolynomialProfile:
    """General-power density model, in local per-segment coordinates, for one perturbative-correction segment.

    Ported from ``v1_0/core/perturbative/models/atmosphere/
    profile_segment.py::AtmospherePolynomialSegment``, trimmed the same
    way ``EvenPowerProfile`` already is: the constructor no longer bakes
    in ``antinu`` or eagerly builds a ``matter_potential_cc``-derived
    ``potential`` field (the caller computes the potential from
    ``average`` at the point ``antinu`` is actually known); ``legacy_precision``
    has no v2 counterpart; and the ``@torch.no_grad()`` decorators v1_0
    still carried are dropped, per this project's foundational autograd
    rule (see ``EvenPowerProfile``'s own docstring for the identical
    reasoning).

    The model represents one segment's density as
    ``n_e(q) = c_0 + c_1*q + c_2*q**2 + ...`` in the local coordinate
    ``q = (x - centre) / half_width in [-1, 1]``, ``centre =
    (x1+x2)/2``, ``half_width = (x2-x1)/2`` -- unlike ``EvenPowerProfile``,
    every power is allowed (a segment centred away from any natural
    symmetry point has no reason to be an even function of its own
    coordinate). See this module's own docstring for why this needs a
    genuinely different representation from ``EvenPowerProfile``, not a
    generalization of it.

    Attributes:
        x1: Segment start coordinate, in evolution-scale units, shaped
            ``(...,)``.
        x2: Segment end coordinate, in evolution-scale units, shaped
            ``(...,)``.
        coefficients: Local polynomial electron-density coefficients,
            shaped ``(..., n_terms)``, ordered ``q**0, q**1, q**2, ...``.
        length: ``x2 - x1``, shaped ``(...,)``.
        zero_mask: Boolean tensor shaped ``(...,)``, True where
            ``length == 0``.
        centre: ``(x1+x2)/2``, shaped ``(...,)``.
        half_width: ``(x2-x1)/2``, shaped ``(...,)``.
        average: Segment-average electron density, shaped ``(...,)`` --
            exactly zero (not the polynomial's own value) on a zero-length
            segment.
        coefficients_n: Optional neutron-density counterpart of
            ``coefficients``, same shape/layout. None unless built with
            ``coefficients_n``.
        average_n: Optional neutron-density counterpart of ``average``.
            None unless ``coefficients_n`` is present.
    """

    def __init__(
        self,
        x1: TensorLike,
        x2: TensorLike,
        coefficients: TensorLike,
        *,
        coefficients_n: Optional[TensorLike] = None,
        profile_scale_m: TensorLike = R_E,
        evolution_scale_m: TensorLike = R_E,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Build a general-power local-coordinate density profile for one (or a batch of) segment(s).

        Args:
            x1: Initial coordinate, normalized by ``profile_scale_m``.
            x2: Final coordinate, normalized by ``profile_scale_m``.
            coefficients: Local polynomial electron-density coefficients,
                in ``profile_scale_m`` coordinates, final dimension
                ordered ``q**0, q**1, q**2, ...``.
            coefficients_n: Optional neutron-density counterpart of
                ``coefficients``, same layout, enabling the sterile
                neutral-current matter term.
            profile_scale_m: Length scale (metres) ``x1``/``x2`` and
                ``coefficients`` are expressed in.
            evolution_scale_m: Length scale (metres) the returned
                ``x1``/``x2``/``average`` are rescaled into -- must match
                whatever scale the generator this profile accompanies was
                itself built with.
            device: Optional explicit device; inferred from the tensor
                inputs when omitted.
            dtype: Optional explicit real dtype; inferred from the tensor
                inputs when omitted.

        Raises:
            ValueError: If ``profile_scale_m``/``evolution_scale_m`` is
                not positive.
        """
        device, dtype = infer_device_dtype(
            x1, x2, coefficients, profile_scale_m, evolution_scale_m, device=device, dtype=dtype
        )
        x1 = as_tensor(x1, device=device, dtype=dtype)
        x2 = as_tensor(x2, device=device, dtype=dtype)
        coefficients = as_tensor(coefficients, device=device, dtype=dtype)
        profile_scale = as_tensor(profile_scale_m, device=device, dtype=dtype)
        evolution_scale = as_tensor(evolution_scale_m, device=device, dtype=dtype)
        if torch.any(profile_scale <= 0) or torch.any(evolution_scale <= 0):
            raise ValueError("profile_scale_m and evolution_scale_m must be positive.")

        # Coefficients are dimensionless (a pure density value per term,
        # unlike EvenPowerProfile's x**power terms), so only x1/x2 need
        # the profile_scale_m -> evolution_scale_m rescaling -- the
        # inverse ratio EvenPowerProfile itself uses, since here x1/x2 are
        # being converted the same direction (profile units -> evolution
        # units) but coefficients carry no compensating power-of-ratio
        # factor to also apply.
        ratio = profile_scale / evolution_scale
        base_shape = torch.broadcast_shapes(x1.shape, x2.shape, coefficients.shape[:-1], ratio.shape)
        self.x1 = torch.broadcast_to(x1 * ratio, base_shape)
        self.x2 = torch.broadcast_to(x2 * ratio, base_shape)
        self.coefficients = torch.broadcast_to(coefficients, (*base_shape, coefficients.shape[-1]))

        self.length = self.x2 - self.x1
        self.zero_mask = self.length == 0
        self.centre = 0.5 * (self.x1 + self.x2)
        self.half_width = 0.5 * self.length

        n_terms = self.coefficients.shape[-1]
        powers = torch.arange(n_terms, device=device, dtype=dtype)
        # int_{-1}^{1} q**k dq = 2/(k+1) for even k, 0 for odd k.
        segment_integrals = torch.where(powers.remainder(2) == 0, 2.0 / (powers + 1.0), torch.zeros_like(powers))
        average = 0.5 * torch.sum(self.coefficients * segment_integrals, dim=-1)
        self.average = torch.where(self.zero_mask, torch.zeros_like(average), average)

        self.coefficients_n: Optional[torch.Tensor] = None
        self.average_n: Optional[torch.Tensor] = None
        if coefficients_n is not None:
            coefficients_n = as_tensor(coefficients_n, device=device, dtype=dtype)
            self.coefficients_n = torch.broadcast_to(coefficients_n, (*base_shape, coefficients_n.shape[-1]))
            n_terms_n = self.coefficients_n.shape[-1]
            powers_n = torch.arange(n_terms_n, device=device, dtype=dtype)
            segment_integrals_n = torch.where(
                powers_n.remainder(2) == 0, 2.0 / (powers_n + 1.0), torch.zeros_like(powers_n)
            )
            average_n = 0.5 * torch.sum(self.coefficients_n * segment_integrals_n, dim=-1)
            self.average_n = torch.where(self.zero_mask, torch.zeros_like(average_n), average_n)

    @staticmethod
    def _monomial_integrals(n_terms: int, frequency: torch.Tensor, *, taylor: bool) -> torch.Tensor:
        """Compute all ``int_-1^1 q**k exp(i*frequency*q) dq`` for ``k=0..n_terms-1`` at once.

        Args:
            n_terms: Number of powers to compute (``k = 0, ..., n_terms-1``
                -- every power, unlike
                ``EvenPowerProfile._oscillatory_even_monomial_integrals``'s
                even-only selection).
            frequency: ``dl * half_width``, shaped to broadcast against
                the ``N x N`` eigenvalue-pair grid.
            taylor: When True, use the order-2 Taylor-in-``frequency``
                approximation instead of the closed form (near a
                (near-)degenerate eigenvalue pair, where the closed form's
                repeated division by a near-zero ``frequency`` loses
                precision).

        Returns:
            Complex tensor shaped ``(..., n_terms, N, N)`` -- the new
            ``n_terms`` axis inserted at position -3, matching
            ``coefficients``'s own ``(..., n_terms, 1, 1)`` shape in
            ``residual_integral``.
        """
        if taylor:
            values = []
            i_frequency = 1j * frequency
            for power in range(n_terms):
                total = torch.zeros_like(frequency)
                for order in range(3):
                    exponent = power + order
                    if exponent % 2 == 0:
                        total = total + i_frequency**order / factorial(order) * 2.0 / (exponent + 1)
                values.append(total)
            return torch.stack(values, dim=-3)

        i_frequency = 1j * frequency
        inv_i_frequency = i_frequency.reciprocal()
        exp_plus = torch.exp(i_frequency)
        exp_minus = torch.exp(-i_frequency)

        integral = (exp_plus - exp_minus) * inv_i_frequency
        values = [integral]
        for power in range(1, n_terms):
            boundary = (exp_plus - ((-1) ** power) * exp_minus) * inv_i_frequency
            integral = boundary - power * inv_i_frequency * integral
            values.append(integral)
        return torch.stack(values, dim=-3)

    def residual_integral(
        self,
        la: torch.Tensor,
        lb: torch.Tensor,
        *,
        coefficients: Optional[torch.Tensor] = None,
        average: Optional[torch.Tensor] = None,
        small_ratio: float = 1.0e-2,
    ) -> torch.Tensor:
        """Evaluate the spectral oscillatory integral of the residual density for this segment.

        The residual profile is the local polynomial after subtracting
        its own segment average, ``delta_n(q) = n_e(q) - average``. For
        each pair of Hamiltonian eigenvalues ``(la, lb)``, this computes

            I_ab = int_{x1}^{x2} delta_n(x) exp(-i*la*x + i*lb*x) dx,

        via the substitution ``x = centre + half_width*q``: the physical
        phase ``exp(-i*la*(x2-x) - i*lb*(x-x1))`` reduces, after that
        substitution, to a constant prefactor ``exp(-i*(la+lb)*half_width)``
        times ``half_width * int_{-1}^{1} delta_n(q) exp(i*(la-lb)*half_width*q) dq``
        -- confirmed algebraically (not just ported unverified) that this
        is exactly ``EvenPowerProfile.residual_integral``'s own physical
        definition, expressed in the local coordinate. Near-degenerate
        eigenvalue pairs use a Taylor expansion instead of the closed
        form, same conditioning characteristic (and the same purely
        *relative* degeneracy test) as ``EvenPowerProfile.residual_integral``
        -- see that method's own docstring, unchanged here.

        Args:
            la: "Bra" Hamiltonian eigenvalue(s), shaped to broadcast with
                the ``N x N`` eigenvalue-pair grid (e.g. ``(..., N, 1)``).
            lb: "Ket" Hamiltonian eigenvalue(s), shaped to broadcast with
                the ``N x N`` eigenvalue-pair grid (e.g. ``(..., 1, N)``).
            coefficients: Optional override for the polynomial
                coefficients, same layout as ``self.coefficients``.
                Defaults to ``self.coefficients`` (electron density); pass
                ``self.coefficients_n`` (with ``average=self.average_n``)
                to compute the neutron-density residual integral instead
                (see ``residual_integral_neutron``).
            average: Optional override for the segment average paired
                with ``coefficients``. Defaults to ``self.average``.
            small_ratio: Threshold on ``|la - lb| / |la + lb|`` below which
                the Taylor expansion is used instead of the closed form.

        Returns:
            Complex residual oscillatory integral I_ab, broadcast over the
            input eigenvalue shapes.
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

        def lift(value: torch.Tensor) -> torch.Tensor:
            while value.ndim < target_ndim - 2:
                value = value.unsqueeze(-1)
            return value.unsqueeze(-1).unsqueeze(-1).to(dtype=la.dtype)

        half_width = lift(self.half_width)
        average = lift(average)

        while coefficients.ndim < target_ndim - 1:
            coefficients = coefficients.unsqueeze(-2)
        coefficients = coefficients.unsqueeze(-1).unsqueeze(-1).to(dtype=la.dtype)

        dl = la - lb
        is_zero = dl == 0
        denominator = la + lb
        ratio = torch.where(
            torch.abs(denominator) > 0, torch.abs(dl / denominator), torch.full_like(torch.abs(dl), float("inf"))
        )
        is_small = ratio < small_ratio

        frequency = dl * half_width
        safe_frequency = torch.where(is_small | is_zero, torch.ones_like(frequency), frequency)

        n_terms = coefficients.shape[-3]
        monomial_full = self._monomial_integrals(n_terms, safe_frequency, taylor=False)
        integral_full = torch.sum(coefficients * monomial_full, dim=-3) - average * monomial_full[..., 0, :, :]

        needs_taylor = bool(torch.any(is_small & ~is_zero))
        if needs_taylor:
            monomial_taylor = self._monomial_integrals(n_terms, frequency, taylor=True)
            integral_taylor = (
                torch.sum(coefficients * monomial_taylor, dim=-3) - average * monomial_taylor[..., 0, :, :]
            )
            integral = torch.where(is_small, integral_taylor, integral_full)
        else:
            integral = integral_full

        phase = torch.exp(-1j * (la + lb) * half_width)
        return torch.where(is_zero, torch.zeros_like(integral), phase * half_width * integral)

    def residual_integral_neutron(self, la: torch.Tensor, lb: torch.Tensor, **kwargs) -> torch.Tensor:
        """Neutron-density counterpart of ``residual_integral``.

        Enables the sterile neutral-current matter term
        (``tpeanuts.physics.extensions.sterile.SterileNCTerm``) in the
        first-order correction. Requires neutron-density coefficients
        (``coefficients_n`` at construction time).

        Raises:
            ValueError: If this profile has no neutron-density coefficients.
        """
        if self.coefficients_n is None:
            raise ValueError(
                "This profile has no neutron-density coefficients. Build it "
                "with coefficients_n to enable the sterile neutral-current "
                "matter term."
            )
        return self.residual_integral(la, lb, coefficients=self.coefficients_n, average=self.average_n, **kwargs)

    def has_perturbation(self) -> torch.Tensor:
        """Return a mask selecting segments with a non-constant electron-density residual.

        Returns:
            Boolean tensor shaped ``(...,)``, True where at least one
            non-constant coefficient is non-zero.
        """
        return torch.any(torch.abs(self.coefficients[..., 1:]) > 0, dim=-1)

    def has_perturbation_neutron(self) -> torch.Tensor:
        """Neutron-density counterpart of ``has_perturbation``.

        Raises:
            ValueError: If this profile has no neutron-density coefficients.
        """
        if self.coefficients_n is None:
            raise ValueError(
                "This profile has no neutron-density coefficients. Build it "
                "with coefficients_n to enable the sterile neutral-current "
                "matter term."
            )
        return torch.any(torch.abs(self.coefficients_n[..., 1:]) > 0, dim=-1)

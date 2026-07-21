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

"""Layered even-power density profiles.

This module contains the model-dependent algebra for profiles whose
coefficients are stored per layer as increasing even powers. It does not know
about Earth angles or detector geometry; callers provide layer coordinates and
crossing masks.

Module classes:
    EvenPowerProfileLayered
        Store per-layer coefficients, shift them under ``r**2 = x**2 + s**2``,
        select crossed layers, and build ``EvenPowerProfileSegment`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Union

import torch

import tpeanuts.config.default as default
from tpeanuts.util.math import binom
from tpeanuts.util.type import TensorLike, as_tensor
from tpeanuts.core.perturbative.models.even_power.profile_segment import (
    EvenPowerProfileSegment,
)
from tpeanuts.core.perturbative.models.even_power.io import (
    load_earth_density_from_csv,
)
from tpeanuts.core.perturbative.models.interface import (
    PerturbativeOuterSegment,
    PerturbativeSegmentBatch,
)
from tpeanuts.util.torch_util import default_device


@dataclass
class EvenPowerProfileLayered:
    """Even-power coefficients attached to ordered density layers.

    Args:
        coefficients: Optional tensor shaped ``(..., n_layers, n_coefficients)``.
            The last dimension stores coefficients of ``x**0, x**2, x**4, ...``.
        density_file: Optional CSV file using the even-power density format.
            When provided without ``coefficients``, the constructor loads
            ``rj`` and coefficients from the file.
        tabulated_density: If True, keep only the constant density term when
            loading from ``density_file``.
        include_neutron: If True and ``coefficients`` is not pre-built, also
            load neutron-density coefficients from ``density_file_n`` (or the
            default ``earth_density_filename_nn`` table), enabling the 3+1
            sterile extension's neutral-current matter term (see
            ``evaluate_neutron`` and
            ``core.common.hamiltonian.hamiltonian_matter_reduced``). Ignored
            when ``coefficients`` is supplied directly; pass
            ``coefficients_n`` explicitly in that case instead.
        density_file_n: Optional CSV file with the neutron-density companion
            table (same ``rj`` shells and column format as ``density_file``).
            Defaults to ``earth_density_dir/earth_density_filename_nn`` when
            ``include_neutron`` is True and neither this nor
            ``coefficients_n`` is supplied.
        coefficients_n: Optional pre-built neutron-density coefficient
            tensor, same shape and layer convention as ``coefficients``.
        device: Torch device used when loading from ``density_file``.
        dtype: Real dtype used when loading from ``density_file``.
    """

    coefficients: torch.Tensor | None = None
    density_file: str | None = None
    tabulated_density: bool = default.earth_tabulated_density
    include_neutron: bool = False
    density_file_n: str | None = None
    coefficients_n: torch.Tensor | None = None
    device: Union[str, torch.device, None] = None
    dtype: torch.dtype = default.dtype
    rj: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.device = default_device(self.device)
        if self.coefficients is None:
            if self.density_file is None:
                self.density_file = os.path.join(
                    default.earth_density_dir,
                    default.earth_density_filename,
                )

            self.rj, self.coefficients = load_earth_density_from_csv(
                self.density_file,
                tabulated_density=self.tabulated_density,
                device=self.device,
                dtype=self.dtype,
            )

            if self.include_neutron and self.coefficients_n is None:
                if self.density_file_n is None:
                    self.density_file_n = os.path.join(
                        default.earth_density_dir,
                        default.earth_density_filename_nn,
                    )
                rj_n, self.coefficients_n = load_earth_density_from_csv(
                    self.density_file_n,
                    tabulated_density=self.tabulated_density,
                    device=self.device,
                    dtype=self.dtype,
                )
                if rj_n.shape != self.rj.shape or not torch.allclose(rj_n, self.rj):
                    raise ValueError(
                        "Neutron-density table shell boundaries (rj) must "
                        "match the electron-density table exactly."
                    )

        self.coefficients = torch.as_tensor(
            self.coefficients,
            device=self.device,
            dtype=self.dtype,
        )
        if not torch.is_floating_point(self.coefficients):
            self.coefficients = self.coefficients.to(dtype=torch.float64)
        if self.rj is not None:
            self.rj = torch.as_tensor(
                self.rj,
                device=self.coefficients.device,
                dtype=self.coefficients.dtype,
            )
        if self.coefficients.ndim < 2:
            raise ValueError("coefficients must have shape (..., n_layers, n_coefficients).")
        if self.coefficients.shape[-1] < 3:
            raise ValueError("coefficients must include at least a, b, c terms.")

        if self.coefficients_n is not None:
            self.coefficients_n = torch.as_tensor(
                self.coefficients_n,
                device=self.coefficients.device,
                dtype=self.coefficients.dtype,
            )
            if self.coefficients_n.shape != self.coefficients.shape:
                raise ValueError(
                    "coefficients_n must have the same shape as coefficients."
                )

        self.device = self.coefficients.device
        self.dtype = self.coefficients.dtype

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "EvenPowerProfileLayered":
        """Return a copy on the requested device and dtype.

        Args:
            device: Optional target torch device. Defaults to the current
                device when omitted.
            dtype: Optional target real dtype. Defaults to the current dtype
                when omitted.

        Returns:
            New ``EvenPowerProfileLayered`` with ``coefficients`` (and ``rj``
            when present) moved to the requested device/dtype.
        """
        output_device = self.device if device is None else device
        output_dtype = self.dtype if dtype is None else dtype
        profile = type(self)(
            coefficients=self.coefficients.to(
                device=output_device,
                dtype=output_dtype,
            ),
            coefficients_n=(
                self.coefficients_n.to(device=output_device, dtype=output_dtype)
                if self.coefficients_n is not None
                else None
            ),
            device=output_device,
            dtype=output_dtype,
        )
        if self.rj is not None:
            profile.rj = self.rj.to(device=output_device, dtype=output_dtype)
        return profile

    def squeeze_leading_batch_if_scalar(self) -> "EvenPowerProfileLayered":
        """Remove the artificial leading batch used for scalar trajectories.

        Returns:
            Profile with the first coefficient dimension squeezed when it has
            shape ``(1, ..., n_layers, n_coefficients)``; otherwise returns the
            current profile unchanged.
        """
        if self.coefficients.ndim == 4 and self.coefficients.shape[0] == 1:
            profile = type(self)(
                coefficients=self.coefficients.squeeze(0),
                coefficients_n=(
                    self.coefficients_n.squeeze(0)
                    if self.coefficients_n is not None
                    else None
                ),
                device=self.device,
                dtype=self.dtype,
            )
            if self.rj is not None:
                profile.rj = self.rj
            return profile

        return self

    def extract_coefficients(
        self,
        coefficients: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split a coefficient tensor into its even-power terms.

        ``coefficients`` stores ``n_e(x) = a + b*x**2 + c*x**4 + delta1*x**6
        + ...`` along its last dimension, ordered
        ``alpha, beta, gamma, delta1, delta2, ...``. This unpacks the first
        three (constant, quadratic, quartic) terms by name and keeps the
        rest as a single ``deltas`` tensor.

        Args:
            coefficients: Even-power coefficient tensor shaped
                ``(..., n_coefficients)`` with ``n_coefficients >= 3``.

        Returns:
            Tuple ``(a, b, c, deltas)`` where ``a``, ``b``, ``c`` are the
            constant, quadratic, and quartic coefficients (shape ``(...,)``)
            and ``deltas`` holds any sixth-power-and-higher coefficients
            (shape ``(..., n_coefficients - 3)``).
        """
        return (
            coefficients[..., 0],
            coefficients[..., 1],
            coefficients[..., 2],
            coefficients[..., 3:],
        )

    def rescale_coefficients(
        self,
        coefficients: torch.Tensor,
        coordinate_ratio: TensorLike,
    ) -> torch.Tensor:
        """Rescale coefficients under a linear change of coordinate.

        Substituting ``x_old = coordinate_ratio * x_new`` into the even-power
        polynomial ``n_e(x_old) = a + b*x_old**2 + c*x_old**4 + ...`` gives,
        in the new coordinate, the same functional form with each power-``2k``
        coefficient multiplied by ``coordinate_ratio**(2k)``. This is used to
        convert profile coefficients between the medium's native coordinate
        (e.g. profile_scale_m) and the dimensionless evolution coordinate
        (e.g. evolution_scale_m).

        Args:
            coefficients: Even-power coefficients shaped
                ``(..., n_coefficients)``, ordered ``a, b, c, delta1, ...``.
            coordinate_ratio: Scalar or tensor ratio ``x_old / x_new``,
                broadcastable with the leading dimensions of ``coefficients``.

        Returns:
            Rescaled coefficients with the same shape as ``coefficients``,
            expressed in the new coordinate ``x_new``.
        """
        a, b, c, deltas = self.extract_coefficients(coefficients)
        ratio = as_tensor(coordinate_ratio, device=coefficients.device, dtype=coefficients.dtype)
        base_shape = torch.broadcast_shapes(
            a.shape,
            b.shape,
            c.shape,
            ratio.shape,
            deltas.shape[:-1],
        )

        ratio = torch.broadcast_to(ratio, base_shape)
        a = torch.broadcast_to(a, base_shape)
        b = torch.broadcast_to(b, base_shape)
        c = torch.broadcast_to(c, base_shape)
        deltas = torch.broadcast_to(deltas, (*base_shape, deltas.shape[-1]))

        if deltas.shape[-1] > 0:
            powers = 2 * torch.arange(
                3,
                3 + deltas.shape[-1],
                device=deltas.device,
                dtype=deltas.dtype,
            )
            deltas = deltas * ratio[..., None] ** powers

        return torch.cat(
            [
                a[..., None],
                (b * ratio**2)[..., None],
                (c * ratio**4)[..., None],
                deltas,
            ],
            dim=-1,
        )

    def shifted(
        self,
        offset: TensorLike,
    ) -> "EvenPowerProfileLayered":
        """Transform coefficients under ``r**2 = x**2 + offset``.

        Args:
            offset: Additive offset in the squared coordinate, broadcastable
                with the leading layer-batch dimensions.

        Returns:
            New layered profile in the shifted coordinate ``x``.
        """
        offset = as_tensor(offset, device=self.device, dtype=self.dtype)
        shifted_coefficients = self._shift_coefficients(self.coefficients, offset)
        shifted_coefficients_n = (
            self._shift_coefficients(self.coefficients_n, offset)
            if self.coefficients_n is not None
            else None
        )

        return type(self)(
            coefficients=shifted_coefficients,
            coefficients_n=shifted_coefficients_n,
            device=self.device,
            dtype=self.dtype,
        )

    def _shift_coefficients(
        self,
        coefficients: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the ``r**2 = x**2 + offset`` binomial shift to ``coefficients``."""
        n_terms = coefficients.shape[-1]
        leading_shape = torch.broadcast_shapes(
            coefficients.shape[:-2],
            offset.shape,
        )
        n_layers = coefficients.shape[-2]
        coefficients = torch.broadcast_to(
            coefficients,
            (*leading_shape, n_layers, n_terms),
        )
        offset = torch.broadcast_to(offset, leading_shape)

        shifted_terms = []
        for target_power in range(n_terms):
            term = torch.zeros(
                (*leading_shape, n_layers),
                device=self.device,
                dtype=self.dtype,
            )
            for source_power in range(target_power, n_terms):
                coefficient = binom(
                    source_power,
                    target_power,
                    self.device,
                    self.dtype,
                )
                term = term + (
                    coefficients[..., source_power]
                    * coefficient
                    * offset[..., None] ** (source_power - target_power)
                )
            shifted_terms.append(term)

        return torch.stack(shifted_terms, dim=-1)

    def evaluate(
        self,
        x: TensorLike,
        *,
        layer_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate the even-power polynomial.

        Args:
            x: Coordinate value.
            layer_index: Optional index selecting one layer per leading batch
                element. When omitted, all layers are evaluated.

        Returns:
            Electron-density value with the selected/broadcast shape.
        """
        x = as_tensor(x, device=self.device, dtype=self.dtype)
        coefficients = self.coefficients
        if layer_index is not None:
            coefficients = self.gather_layers(layer_index)

        x2 = x * x
        powers = torch.arange(
            coefficients.shape[-1],
            device=self.device,
            dtype=self.dtype,
        )
        return torch.sum(coefficients * x2[..., None] ** powers, dim=-1)

    def evaluate_neutron(
        self,
        x: TensorLike,
        *,
        layer_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate the even-power neutron-density polynomial (see ``evaluate``).

        Requires neutron-density coefficients, built via
        ``EvenPowerProfileLayered(..., include_neutron=True)`` or supplied
        directly as ``coefficients_n``.

        Args:
            x: Coordinate value.
            layer_index: Optional index selecting one layer per leading batch
                element. When omitted, all layers are evaluated.

        Returns:
            Neutron-density value with the selected/broadcast shape.

        Raises:
            ValueError: If neutron-density coefficients were not loaded.
        """
        if self.coefficients_n is None:
            raise ValueError(
                "This EvenPowerProfileLayered has no neutron-density "
                "coefficients. Build it with include_neutron=True (or pass "
                "coefficients_n explicitly) to enable the 3+1 sterile "
                "extension's neutral-current matter term."
            )
        x = as_tensor(x, device=self.device, dtype=self.dtype)
        coefficients = self.coefficients_n
        if layer_index is not None:
            coefficients = self._gather_layers(self.coefficients_n, layer_index)

        x2 = x * x
        powers = torch.arange(
            coefficients.shape[-1],
            device=self.device,
            dtype=self.dtype,
        )
        return torch.sum(coefficients * x2[..., None] ** powers, dim=-1)

    def _gather_layers(
        self,
        coefficients: torch.Tensor,
        layer_index: torch.Tensor,
    ) -> torch.Tensor:
        """Gather one layer coefficient vector per batch item from ``coefficients``."""
        layer_index = layer_index.to(device=self.device, dtype=torch.long)
        n_layers = coefficients.shape[-2]
        layer_index = layer_index.clamp(min=0, max=n_layers - 1)
        gather_index = layer_index[..., None, None].expand(
            *layer_index.shape,
            1,
            coefficients.shape[-1],
        )
        return torch.gather(coefficients, dim=-2, index=gather_index).squeeze(-2)

    def gather_layers(
        self,
        layer_index: torch.Tensor,
    ) -> torch.Tensor:
        """Gather one layer coefficient vector per batch item.

        Args:
            layer_index: Integer tensor selecting, for each leading batch
                element, which density layer's coefficients to extract.
                Values are clamped to ``[0, n_layers - 1]``.

        Returns:
            Coefficient vectors shaped ``(..., n_coefficients)``, one per
            batch element, taken from the selected layer.
        """
        return self._gather_layers(self.coefficients, layer_index)

    def outermost_segment(
        self,
        xj_all: torch.Tensor,
        crossed: torch.Tensor,
    ) -> PerturbativeOuterSegment:
        """Return metadata and model data for the outermost crossed layer.

        For each trajectory (leading batch element), finds the layer index
        with the largest position among those marked ``crossed`` (the
        detector-side / outermost shell actually traversed), and also the
        second-largest such index. The starting coordinate of the outermost
        segment is the boundary of that second-outermost crossed layer (or
        zero if fewer than two layers are crossed), matching the convention
        used by ``EvenPowerProfileSegment`` outer-segment construction.

        Args:
            xj_all: Layer boundary coordinates shaped ``(batch_size,
                n_layers)``, in the coordinate system used for crossing tests.
            crossed: Boolean mask shaped ``(batch_size, n_layers)`` selecting
                layers actually crossed by each trajectory.

        Returns:
            ``PerturbativeOuterSegment`` whose ``model_data`` is a
            ``(coefficients, coefficients_n)`` tuple for the outermost
            layer (``coefficients_n`` is None unless this profile was built
            with ``include_neutron=True``), the starting coordinate
            ``x_start``, and boolean/index metadata (``has_any``, ``has_two``,
            ``last_pos``, ``second_last_pos``) describing the crossing.
        """
        batch_size, n_layers = xj_all.shape
        pos = torch.arange(n_layers, device=xj_all.device)
        pos_b = pos.unsqueeze(0).expand(batch_size, n_layers)
        neg_inf = torch.full(
            (batch_size, n_layers),
            -1.0e30,
            device=xj_all.device,
            dtype=xj_all.dtype,
        )
        pos_masked = torch.where(crossed, pos_b.to(dtype=xj_all.dtype), neg_inf)
        last_pos = pos_masked.max(dim=-1).values
        has_any = last_pos > -1.0e20
        pos_masked_without_last = torch.where(
            pos_b.to(dtype=xj_all.dtype) == last_pos.unsqueeze(-1),
            neg_inf,
            pos_masked,
        )
        second_last_pos = pos_masked_without_last.max(dim=-1).values
        has_two = second_last_pos > -1.0e20
        x_start = torch.zeros((batch_size,), device=xj_all.device, dtype=xj_all.dtype)
        x_start = torch.where(
            has_two,
            xj_all.gather(
                dim=-1,
                index=second_last_pos.to(torch.long).clamp(min=0, max=n_layers - 1).unsqueeze(-1),
            ).squeeze(-1),
            x_start,
        )

        last_pos_long = last_pos.to(torch.long)
        coefficients_n = (
            self._gather_layers(self.coefficients_n, last_pos_long)
            if self.coefficients_n is not None
            else None
        )

        return PerturbativeOuterSegment(
            model_data=(self.gather_layers(last_pos_long), coefficients_n),
            x_start=x_start,
            has_any=has_any,
            has_two=has_two,
            last_pos=last_pos,
            second_last_pos=second_last_pos,
        )

    def ordered_segments(
        self,
        xj_all: torch.Tensor,
        crossed: torch.Tensor,
    ) -> PerturbativeSegmentBatch:
        """Build ordered shell segments with opaque model data.

        Reverses the layer ordering (so the innermost shell comes first,
        matching the order in which a trajectory crosses shells from the
        centre outward) and pairs each layer's outer boundary ``x2`` with the
        previous layer's boundary as its inner boundary ``x1`` (the
        innermost segment starts at ``x1 = 0``).

        Args:
            xj_all: Layer boundary coordinates shaped ``(..., n_layers)``.
            crossed: Boolean mask shaped ``(..., n_layers)`` selecting layers
                crossed by each trajectory.

        Returns:
            ``PerturbativeSegmentBatch`` with reversed-order segment
            boundaries ``x1``/``x2``, the reversed ``crossed`` mask, and
            ``model_data`` set to a ``(coefficients, coefficients_n)`` tuple
            of reversed per-layer coefficients (``coefficients_n`` is None
            unless this profile was built with ``include_neutron=True``).
        """
        xs = torch.flip(xj_all, dims=(-1,))
        coefficients = torch.flip(self.coefficients, dims=(-2,))
        coefficients_n = (
            torch.flip(self.coefficients_n, dims=(-2,))
            if self.coefficients_n is not None
            else None
        )
        crossed_flipped = torch.flip(crossed, dims=(-1,))
        x_lo = torch.zeros_like(xs)
        x_lo[..., :-1] = xs[..., 1:]
        x_lo[..., -1] = 0.0

        return PerturbativeSegmentBatch(
            x1=x_lo,
            x2=xs,
            crossed=crossed_flipped,
            model_data=(coefficients, coefficients_n),
        )

    def segment_model(
        self,
        segments: PerturbativeSegmentBatch,
        *,
        coordinate_ratio: TensorLike | None = None,
        **kwargs,
    ) -> EvenPowerProfileSegment:
        """Build a perturbative segment object from opaque segment data.

        Args:
            segments: ``PerturbativeSegmentBatch`` produced by
                ``ordered_segments``/``outermost_segment``, whose
                ``model_data`` holds a ``(coefficients, coefficients_n)``
                tuple for this model (``coefficients_n`` may be None).
            coordinate_ratio: Optional ratio used to rescale the coefficients
                (see ``rescale_coefficients``) into the evolution coordinate
                before constructing the segment, e.g. when the segment
                boundaries are expressed in a different scale than the
                coefficients.
            **kwargs: Additional keyword arguments forwarded to
                ``EvenPowerProfileSegment.from_coefficients`` (e.g. ``antinu``,
                ``evolution_scale_m``).

        Returns:
            ``EvenPowerProfileSegment`` built from ``segments.x1``,
            ``segments.x2``, and the (optionally rescaled) coefficients.
        """
        coefficients, coefficients_n = segments.model_data
        if coordinate_ratio is not None:
            coefficients = self.rescale_coefficients(coefficients, coordinate_ratio)
            if coefficients_n is not None:
                coefficients_n = self.rescale_coefficients(coefficients_n, coordinate_ratio)

        return EvenPowerProfileSegment.from_coefficients(
            x1=segments.x1,
            x2=segments.x2,
            coefficients=coefficients,
            coefficients_n=coefficients_n,
            **kwargs,
        )

    def constant_segment_model(
        self,
        *,
        x1: TensorLike,
        x2: TensorLike,
        density: TensorLike,
        density_n: TensorLike | None = None,
        **kwargs,
    ) -> EvenPowerProfileSegment:
        """Build a constant-density perturbative segment object.

        Args:
            x1: Initial segment coordinate.
            x2: Final segment coordinate.
            density: Constant electron density over the segment, in mol/cm^3
                (or the convention expected by ``matter_potential_cc``).
            density_n: Optional constant neutron density, enabling the 3+1
                sterile extension's neutral-current matter term.
            **kwargs: Additional keyword arguments forwarded to
                ``EvenPowerProfileSegment.constant`` (e.g. ``antinu``,
                ``evolution_scale_m``).

        Returns:
            ``EvenPowerProfileSegment`` with zero quadratic/quartic
            coefficients, i.e. ``n_e(x) = density`` over the segment.
        """
        return EvenPowerProfileSegment.constant(
            x1=x1,
            x2=x2,
            density=density,
            density_n=density_n,
            **kwargs,
        )

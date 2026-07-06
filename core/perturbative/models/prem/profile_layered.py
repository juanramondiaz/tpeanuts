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

"""Layered PREM piecewise-linear-in-r² density profile.

Within each shell the electron density is a first-degree polynomial in r²:

    n_e(r²) = A + B · r²,

fit by linear interpolation between consecutive PREM500 tabulated points.
This representation is exact at both endpoints of every sub-interval, and the
coordinate shift r² = x² + sin²η required for a nadir-angle trajectory maps
the density to an equally simple polynomial in x²:

    n_e(x²) = (A + B · sin²η) + B · x².

This preserves analytical tractability of the oscillatory residual integral
and makes the model a drop-in replacement for ``EvenPowerProfileLayered``
inside ``EarthProfile``.

Module classes:
    PremTabulatedProfile
        Layered PREM profile with piecewise-linear-in-r² electron density.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Union

import torch

import tpeanuts.util.default as default
from tpeanuts.util.type import TensorLike, as_tensor
from tpeanuts.util.torch_util import default_device
from tpeanuts.core.perturbative.models.interface import (
    PerturbativeOuterSegment,
    PerturbativeSegmentBatch,
)
from tpeanuts.core.perturbative.models.prem.io import load_prem500_profile
from tpeanuts.core.perturbative.models.prem.profile_segment import PremProfileSegment


# Default PREM500 CSV path relative to the project working directory.
_DEFAULT_PREM_FILE: str = os.path.join("data", "external", "PREM500.csv")


@dataclass
class PremTabulatedProfile:
    """Layered PREM electron-density profile (piecewise-linear in r²).

    Implements the same interface as ``EvenPowerProfileLayered`` so that
    ``EarthProfile`` and the perturbative Earth evolutor can use it without
    modification.

    Args:
        prem_file: Path to the PREM500 CSV file (nine columns, no header).
            Defaults to ``data/external/PREM500.csv``.  Ignored when both
            ``rj`` and ``coefficients`` are supplied.
        rj: Optional pre-built shell outer boundaries, normalized by R_E,
            strictly increasing in ``(0, 1]``.  Must be paired with
            ``coefficients``.
        coefficients: Optional pre-built coefficient tensor of shape
            ``(..., n_shells, 2)`` with columns ``[A, B]`` for
            n_e(r²) = A + B·r² in each shell.  Must be paired with ``rj``.
        device: Torch device for all tensors.
        dtype: Real dtype for all tensors.
    """

    prem_file: str | None = None
    rj: torch.Tensor | None = None
    coefficients: torch.Tensor | None = None
    device: Union[str, torch.device, None] = None
    dtype: torch.dtype = default.dtype

    def __post_init__(self) -> None:
        self.device = default_device(self.device)

        if self.coefficients is None:
            # Load from PREM500 CSV.
            if self.prem_file is None:
                self.prem_file = _DEFAULT_PREM_FILE
            self.rj, self.coefficients = load_prem500_profile(
                self.prem_file,
                device=self.device,
                dtype=self.dtype,
            )

        self.coefficients = torch.as_tensor(
            self.coefficients, device=self.device, dtype=self.dtype
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
            raise ValueError(
                "coefficients must have at least 2 dimensions (..., n_shells, 2)."
            )
        if self.coefficients.shape[-1] != 2:
            raise ValueError(
                "PREM profile coefficients must have exactly 2 entries per shell "
                "(A constant term and B quadratic term)."
            )

        self.device = self.coefficients.device
        self.dtype = self.coefficients.dtype

    # ──────────────────────────────────────────────────────────────────────────
    # Device / dtype management
    # ──────────────────────────────────────────────────────────────────────────

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "PremTabulatedProfile":
        """Return a copy on the requested device and dtype.

        Args:
            device: Target torch device.  Defaults to the current device.
            dtype: Target real dtype.  Defaults to the current dtype.

        Returns:
            New ``PremTabulatedProfile`` with all tensors moved to the
            requested device/dtype.
        """
        out_device = self.device if device is None else device
        out_dtype = self.dtype if dtype is None else dtype
        return PremTabulatedProfile(
            rj=(
                self.rj.to(device=out_device, dtype=out_dtype)
                if self.rj is not None
                else None
            ),
            coefficients=self.coefficients.to(device=out_device, dtype=out_dtype),
            device=out_device,
            dtype=out_dtype,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Coordinate shift: radial r → trajectory x  (r² = x² + sin²η)
    # ──────────────────────────────────────────────────────────────────────────

    def shifted(self, offset: TensorLike) -> "PremTabulatedProfile":
        """Transform coefficients under r² = x² + offset (trajectory shift).

        Substituting r² = x² + s (s = sin²η) into n_e(r²) = A + B·r² gives
        n_e(x) = (A + B·s) + B·x², so only the constant term A is modified.

        Args:
            offset: sin²η, broadcastable with the leading batch dimensions of
                the coefficient tensor.

        Returns:
            New ``PremTabulatedProfile`` in trajectory-coordinate (x²) space
            with updated constant-term coefficients.
        """
        offset = as_tensor(offset, device=self.device, dtype=self.dtype)
        n_shells = self.coefficients.shape[-2]
        leading_shape = torch.broadcast_shapes(
            self.coefficients.shape[:-2], offset.shape
        )
        coefficients = torch.broadcast_to(
            self.coefficients, (*leading_shape, n_shells, 2)
        )
        offset = torch.broadcast_to(offset, leading_shape)

        A = coefficients[..., 0]  # (..., n_shells)
        B = coefficients[..., 1]  # (..., n_shells)
        A_shifted = A + B * offset[..., None]

        return PremTabulatedProfile(
            rj=self.rj,
            coefficients=torch.stack([A_shifted, B], dim=-1),
            device=self.device,
            dtype=self.dtype,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Pointwise evaluation
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        x: TensorLike,
        *,
        layer_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate n_e(x) = A + B · x².

        Useful for computing pointwise electron densities along a trajectory
        (called by ``EarthProfile.density_x_eta``).

        Args:
            x: Coordinate value in trajectory or radial units, depending on
                whether ``shifted`` has been applied.
            layer_index: Optional integer tensor selecting one shell per batch
                element.  When omitted all shells are evaluated.

        Returns:
            Electron density values with the appropriate broadcast shape.
        """
        x = as_tensor(x, device=self.device, dtype=self.dtype)
        coefficients = self.coefficients
        if layer_index is not None:
            coefficients = self.gather_layers(layer_index)
        return coefficients[..., 0] + coefficients[..., 1] * x * x

    def gather_layers(self, layer_index: torch.Tensor) -> torch.Tensor:
        """Gather per-batch shell coefficient vectors.

        Args:
            layer_index: Integer tensor with one shell index per leading batch
                element.  Values are clamped to ``[0, n_shells − 1]``.

        Returns:
            Coefficient tensor shaped ``(..., 2)``.
        """
        layer_index = layer_index.to(device=self.device, dtype=torch.long)
        n_shells = self.coefficients.shape[-2]
        layer_index = layer_index.clamp(0, n_shells - 1)
        if self.coefficients.ndim == 2:
            # Unbatched coefficients: direct index into the shell dimension.
            return self.coefficients[layer_index]
        idx = layer_index[..., None, None].expand(*layer_index.shape, 1, 2)
        return torch.gather(self.coefficients, dim=-2, index=idx).squeeze(-2)

    # ──────────────────────────────────────────────────────────────────────────
    # Leading-batch handling (scalar trajectory)
    # ──────────────────────────────────────────────────────────────────────────

    def squeeze_leading_batch_if_scalar(self) -> "PremTabulatedProfile":
        """Remove the artificial leading batch used for scalar trajectories.

        When ``eta`` is a scalar, ``shifted`` broadcasts it to shape ``(1,)``,
        producing a coefficient tensor shaped ``(1, n_shells, 2)``.  This
        method removes that singleton so downstream tensor operations do not
        carry an unexpected batch dimension.

        Returns:
            Profile with the leading dimension squeezed when it equals 1;
            otherwise returns self unchanged.
        """
        if self.coefficients.ndim == 3 and self.coefficients.shape[0] == 1:
            return PremTabulatedProfile(
                rj=self.rj,
                coefficients=self.coefficients.squeeze(0),
                device=self.device,
                dtype=self.dtype,
            )
        return self

    # ──────────────────────────────────────────────────────────────────────────
    # Segment geometry  (called after shifted())
    # ──────────────────────────────────────────────────────────────────────────

    def ordered_segments(
        self,
        xj_all: torch.Tensor,
        crossed: torch.Tensor,
    ) -> PerturbativeSegmentBatch:
        """Build ordered shell segments with (A, B) coefficients as model data.

        Reverses the layer ordering (innermost shell first) so that segments
        are composed along the trajectory from the Earth's centre outward.

        Args:
            xj_all: Shell boundary coordinates shaped ``(..., n_shells)``.
            crossed: Boolean mask for physically crossed shells, same shape.

        Returns:
            ``PerturbativeSegmentBatch`` whose ``model_data`` is the
            reversed-order ``(A, B)`` coefficient tensor.
        """
        xs = torch.flip(xj_all, dims=(-1,))
        coefficients = torch.flip(self.coefficients, dims=(-2,))
        crossed_flipped = torch.flip(crossed, dims=(-1,))

        x_lo = torch.zeros_like(xs)
        x_lo[..., :-1] = xs[..., 1:]
        x_lo[..., -1] = 0.0

        return PerturbativeSegmentBatch(
            x1=x_lo,
            x2=xs,
            crossed=crossed_flipped,
            model_data=coefficients,
        )

    def outermost_segment(
        self,
        xj_all: torch.Tensor,
        crossed: torch.Tensor,
    ) -> PerturbativeOuterSegment:
        """Return metadata and (A, B) for the outermost crossed shell.

        Identifies the shell with the largest boundary coordinate among those
        marked as crossed (the detector-side / outermost shell) and the
        second-outermost crossed shell (used to determine the outer segment
        start coordinate).

        Args:
            xj_all: Shell boundary coordinates shaped ``(batch, n_shells)``.
            crossed: Boolean mask for crossed shells, same shape.

        Returns:
            ``PerturbativeOuterSegment`` whose ``model_data`` holds the (A, B)
            coefficients of the outermost crossed layer.
        """
        batch_size, n_shells = xj_all.shape
        crossed = crossed.to(device=xj_all.device)
        pos = torch.arange(n_shells, device=xj_all.device)
        pos_b = pos.unsqueeze(0).expand(batch_size, n_shells)
        neg_inf = torch.full(
            (batch_size, n_shells), -1.0e30,
            device=xj_all.device, dtype=xj_all.dtype,
        )

        pos_masked = torch.where(crossed, pos_b.to(dtype=xj_all.dtype), neg_inf)
        last_pos = pos_masked.max(dim=-1).values
        has_any = last_pos > -1.0e20

        pos_nolast = torch.where(
            pos_b.to(dtype=xj_all.dtype) == last_pos.unsqueeze(-1),
            neg_inf,
            pos_masked,
        )
        second_last_pos = pos_nolast.max(dim=-1).values
        has_two = second_last_pos > -1.0e20

        x_start = torch.zeros((batch_size,), device=xj_all.device, dtype=xj_all.dtype)
        x_start = torch.where(
            has_two,
            xj_all.gather(
                dim=-1,
                index=second_last_pos.to(torch.long).clamp(0, n_shells - 1).unsqueeze(-1),
            ).squeeze(-1),
            x_start,
        )

        return PerturbativeOuterSegment(
            model_data=self.gather_layers(last_pos.to(torch.long)),
            x_start=x_start,
            has_any=has_any,
            has_two=has_two,
            last_pos=last_pos,
            second_last_pos=second_last_pos,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Segment model construction
    # ──────────────────────────────────────────────────────────────────────────

    def segment_model(
        self,
        segments: PerturbativeSegmentBatch,
        *,
        coordinate_ratio: TensorLike | None = None,
        **kwargs,
    ) -> PremProfileSegment:
        """Build a ``PremProfileSegment`` from opaque segment data.

        The ``model_data`` field of ``segments`` holds the (A, B) coefficient
        pair for each shell.  If the trajectory coordinates and profile
        coefficients are expressed in different scales (``profile_scale_m ≠
        R_E``), an optional ``coordinate_ratio`` rescales the B coefficient
        so it is consistent with the x₁/x₂ coordinates in ``segments``.

        Args:
            segments: ``PerturbativeSegmentBatch`` with ``(A, B)`` in
                ``model_data``.
            coordinate_ratio: Optional rescaling factor
                ``profile_scale_m / R_E`` applied to the B coefficient.
                In the default configuration both scales equal R_E so the
                ratio is 1 and the rescaling is a no-op.
            **kwargs: Additional arguments forwarded to ``PremProfileSegment``
                (e.g. ``antinu``, ``profile_scale_m``, ``evolution_scale_m``,
                ``device``, ``dtype``).

        Returns:
            Initialized ``PremProfileSegment``.
        """
        coefficients = segments.model_data
        if coordinate_ratio is not None:
            ratio = as_tensor(
                coordinate_ratio,
                device=coefficients.device,
                dtype=coefficients.dtype,
            )
            A = coefficients[..., 0]
            B = coefficients[..., 1] * ratio ** 2
            coefficients = torch.stack([A, B], dim=-1)

        return PremProfileSegment(
            x1=segments.x1,
            x2=segments.x2,
            a=coefficients[..., 0],
            b=coefficients[..., 1],
            **kwargs,
        )

    def constant_segment_model(
        self,
        *,
        x1: TensorLike,
        x2: TensorLike,
        density: TensorLike,
        **kwargs,
    ) -> PremProfileSegment:
        """Build a constant-density segment (b = 0, no first-order correction).

        Used for Case B (short, shallow trajectories) where the density is
        approximated as spatially uniform over the path.

        Args:
            x1: Segment start coordinate.
            x2: Segment end coordinate.
            density: Constant electron density in mol cm⁻³.
            **kwargs: Forwarded to ``PremProfileSegment``.

        Returns:
            ``PremProfileSegment`` with B = 0.
        """
        density_t = torch.as_tensor(density)
        return PremProfileSegment(
            x1=x1,
            x2=x2,
            a=density_t,
            b=torch.zeros_like(density_t),
            **kwargs,
        )

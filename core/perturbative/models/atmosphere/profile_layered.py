"""Automatic piecewise-polynomial fitting of atmosphere density profiles.

Module classes:
    AtmospherePolynomialProfile
        Fit density samples at fixed local nodes and build a batched
        perturbative segment model.
"""

from __future__ import annotations

import torch

from tpeanuts.core.perturbative.models.atmosphere.profile_segment import (
    AtmospherePolynomialSegment,
)


class AtmospherePolynomialProfile:
    """Piecewise polynomial atmosphere profile in local segment coordinates.

    Args:
        boundaries: Evolution-coordinate boundaries shaped ``(..., S + 1)``.
        density_nodes: Electron density at the interpolation nodes, shaped
            ``(..., S, degree + 1)``. Nodes are ordered from segment start to
            end and are equally spaced in the local coordinate ``q in [-1,1]``.

    Attributes:
        boundaries: Segment boundaries in evolution coordinates.
        coefficients: Local polynomial coefficients ordered as
            ``q**0, q**1, ...``.
        degree: Polynomial degree.
        n_segments: Number of fitted trajectory segments.
    """

    def __init__(self, boundaries: torch.Tensor, density_nodes: torch.Tensor) -> None:
        if boundaries.ndim < 1 or density_nodes.ndim < 2:
            raise ValueError("Atmosphere boundaries and density nodes need segment dimensions.")
        if boundaries.shape[-1] != density_nodes.shape[-2] + 1:
            raise ValueError("Expected one more boundary than fitted segments.")
        n_nodes = density_nodes.shape[-1]
        if n_nodes < 1:
            raise ValueError("At least one interpolation node is required.")

        q = (
            torch.zeros(1, device=density_nodes.device, dtype=density_nodes.dtype)
            if n_nodes == 1
            else torch.linspace(-1.0, 1.0, n_nodes, device=density_nodes.device, dtype=density_nodes.dtype)
        )
        vandermonde = torch.vander(q, N=n_nodes, increasing=True)
        self.boundaries = boundaries
        self.coefficients = density_nodes @ torch.linalg.inv(vandermonde).T
        self.degree = n_nodes - 1
        self.n_segments = density_nodes.shape[-2]

    def segment_model(
        self,
        *,
        coefficients_n: torch.Tensor | None = None,
        **kwargs,
    ) -> AtmospherePolynomialSegment:
        """Return the batched perturbative model for all fitted segments.

        Args:
            coefficients_n: Optional neutron-density local polynomial
                coefficients (see ``AtmospherePolynomialSegment``), typically
                the ``.coefficients`` of a second ``AtmospherePolynomialProfile``
                fitted to a neutron-density sample at the same nodes,
                enabling the 3+1 sterile extension's neutral-current matter
                term.
            **kwargs: Forwarded to ``AtmospherePolynomialSegment`` (e.g.
                ``antinu``, ``evolution_scale_m``).
        """
        return AtmospherePolynomialSegment(
            x1=self.boundaries[..., :-1],
            x2=self.boundaries[..., 1:],
            coefficients=self.coefficients,
            coefficients_n=coefficients_n,
            **kwargs,
        )

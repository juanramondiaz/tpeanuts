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
Atmosphere profile container for numerical propagation.

This module connects atmosphere geometry with density models and stores the
trajectory object consumed by ``core.numerical.evolutor``. It does not build
Hamiltonians or evolution operators; it only samples and stores an
electron-density profile along an atmosphere path.

Module functions:
    AtmosphereProfile(...)
        Build and store a numerical trajectory plus electron-density samples
        for the medium-independent numerical evolutor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

import tpeanuts.config.default as default
from tpeanuts.core.numerical.geometry import Trajectory, segment_sample_points
from tpeanuts.medium.atmosphere.density import atmosphere_density
from tpeanuts.medium.atmosphere.geometry import (
    altitude_along_detector_path,
    atmosphere_path_length,
    underground_path_length,
)
from tpeanuts.util.constant import R_E
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import TensorLike, as_tensor


@dataclass(frozen=True)
class AtmosphereParameters:
    """
    Construction settings for the numerical atmosphere density profile.

    Parameters
    ----------
    atmosphere_density_source:
        Density backend name: "exponential", "file", "mceq", "msis", or
        "pymsis". Selects which physical/empirical model supplies electron
        density along the trajectory.

    atmosphere_density_kwargs:
        Backend-specific keyword arguments forwarded to
        ``medium.atmosphere.density.atmosphere_density`` (e.g.
        ``pymsis_config``, ``mceq_config``, ``density_file``).

    nsteps:
        Number of numerical integration segments along the
        production-to-surface trajectory.

    method:
        Segment sampling rule: "midpoint", "left", or "right".

    matter:
        If False, the profile reports zero electron density everywhere
        (vacuum approximation), regardless of ``atmosphere_density_source``.

    evolution_scale_m:
        Physical scale, in metres, used to non-dimensionalize the
        Hamiltonian evolution coordinate along the trajectory.

    perturbative_segments:
        Number of local polynomial segments used by analytical propagation.

    perturbative_degree:
        Degree of the automatically fitted polynomial in each analytical
        segment.

    include_matter_nc:
        If True, also sample neutron density along the trajectory and expose
        it as ``n_n_molcm3``, enabling the 3+1 sterile extension's
        neutral-current matter term in ``atmosphere_evolutor_numerical`` (see
        ``core.common.hamiltonian.hamiltonian_matter_reduced``). If False,
        reproduces the pre-existing CC-only behaviour exactly and leaves
        ``n_n_molcm3`` as ``None``. If ``None`` (the default), auto-resolved
        by ``atmosphere_evolutor_numerical``/``atmosphere_evolutor_analytical``
        via ``core.common.oscillation.resolve_include_matter_nc``: ``True``
        when ``oscillation.BSM_extension_sterile`` is set, ``False``
        otherwise.
    """

    atmosphere_density_source: str = default.atmosphere_source_density
    atmosphere_density_kwargs: dict | None = None
    nsteps: int = 599
    method: str | None = "midpoint"
    matter: bool = True
    evolution_scale_m: TensorLike = R_E
    perturbative_segments: int = 4
    perturbative_degree: int = 3
    include_matter_nc: Optional[bool] = None


class AtmosphereProfile:
    """Numerical atmosphere profile sampled along a production trajectory.

    The constructor builds the dimensionless trajectory from production
    altitude to the Earth surface and stores one electron-density sample per
    numerical segment. The density backend is selected by
    ``params.atmosphere_density_source`` and evaluated through
    ``medium.atmosphere.density.atmosphere_density``.

    Args:
        h_km: Production altitude in km.
        theta_deg: Atmosphere zenith angle in degrees.
        depth_km: Detector depth below the surface in km.
        params: Atmosphere density profile construction settings.
        context: Runtime device/dtype used for trajectory and density
            tensors.

    Attributes:
        trajectory: ``Trajectory`` consumed by ``core.numerical.evolutor``.
        n_e_molcm3: Electron-density samples in mol/cm^3.
        n_n_molcm3: Neutron-density samples in mol/cm^3, or ``None`` when
            ``params.include_matter_nc`` is False.
        atmosphere_density_source: Density backend name used for the profile.
        altitude_km: Altitude samples associated with ``n_e_molcm3``.
        x: Dimensionless path grid.
        dx_evolution: Dimensionless segment widths.
    """

    @torch.no_grad()
    def __init__(
        self,
        h_km: TensorLike,
        theta_deg: TensorLike,
        depth_km: TensorLike = 0.0,
        *,
        params: Optional[AtmosphereParameters] = None,
        context: RuntimeContext,
    ) -> None:
        params = params or AtmosphereParameters()
        if params.nsteps < 1:
            raise ValueError("nsteps must be at least 1.")

        self.params = params
        self.atmosphere_density_source = str(params.atmosphere_density_source)
        self.atmosphere_density_kwargs = dict(params.atmosphere_density_kwargs or {})
        self.nsteps = int(params.nsteps)
        self.method = params.method
        self.matter = bool(params.matter)
        self.dtype = context.dtype

        dtype = context.dtype
        h_km = as_tensor(h_km, device=context.device, dtype=dtype)
        dev = h_km.device
        theta_deg = as_tensor(theta_deg, device=dev, dtype=dtype)
        depth_km = as_tensor(depth_km, device=dev, dtype=dtype)

        self.device = dev
        self.h_km = h_km
        self.theta_deg = theta_deg
        self.depth_km = depth_km

        L_atm_km = atmosphere_path_length(
            h_km=h_km,
            theta_deg=theta_deg,
            depth_km=depth_km,
            device=dev,
            dtype=dtype,
            check_geometry=False,
        )
        L_und_km = underground_path_length(
            theta_deg=theta_deg,
            depth_km=depth_km,
            device=dev,
            dtype=dtype,
            check_geometry=False,
        )

        evolution_scale_km = as_tensor(
            params.evolution_scale_m,
            device=dev,
            dtype=dtype,
        ) / 1.0e3
        if torch.any(evolution_scale_km <= 0):
            raise ValueError("evolution_scale_m must be positive.")

        u_grid = torch.linspace(
            0.0,
            1.0,
            self.nsteps + 1,
            device=dev,
            dtype=dtype,
        )
        x_grid = (L_atm_km / evolution_scale_km)[..., None] * u_grid
        sample_x = segment_sample_points(x_grid, self.method)
        dx_evolution = x_grid[..., 1:] - x_grid[..., :-1]

        s_atm_km = sample_x * evolution_scale_km
        s_detector_km = L_und_km[..., None] + s_atm_km

        altitude_km = altitude_along_detector_path(
            s_km=s_detector_km,
            theta_deg=theta_deg[..., None],
            depth_km=depth_km[..., None],
            device=dev,
            dtype=dtype,
        )

        self.include_matter_nc = bool(params.include_matter_nc)

        if self.matter:
            n_e = atmosphere_density(
                altitude_km,
                source=self.atmosphere_density_source,
                density_type="electron_density",
                context=RuntimeContext(device=dev, dtype=dtype),
                **self.atmosphere_density_kwargs,
            )
            n_n = (
                atmosphere_density(
                    altitude_km,
                    source=self.atmosphere_density_source,
                    density_type="neutron_density",
                    context=RuntimeContext(device=dev, dtype=dtype),
                    **self.atmosphere_density_kwargs,
                )
                if self.include_matter_nc
                else None
            )
        else:
            n_e = torch.zeros_like(altitude_km)
            n_n = torch.zeros_like(altitude_km) if self.include_matter_nc else None

        self.L_atm_km = L_atm_km
        self.L_und_km = L_und_km
        self.altitude_km = altitude_km
        self.n_e_molcm3 = as_tensor(n_e, device=dev, dtype=dtype)
        self.n_n_molcm3 = (
            None if n_n is None else as_tensor(n_n, device=dev, dtype=dtype)
        )

        self.trajectory = Trajectory(
            x=x_grid,
            dx_evolution=dx_evolution,
            sample_x=sample_x,
            meta={
                "kind": "atmosphere",
                "h_km": h_km,
                "theta_deg": theta_deg,
                "depth_km": depth_km,
                "L_atm_km": L_atm_km,
                "L_und_km": L_und_km,
                "altitude_km": altitude_km,
                "atmosphere_density_source": self.atmosphere_density_source,
            },
        )

    @property
    def x(self) -> torch.Tensor:
        """Return the dimensionless atmosphere path grid."""
        return self.trajectory.x

    @property
    def dx_evolution(self) -> torch.Tensor:
        """Return the dimensionless segment widths."""
        return self.trajectory.dx_evolution

    def __str__(self) -> str:
        """Return a compact summary of the atmosphere profile configuration."""
        h     = float(self.h_km)
        theta = float(self.theta_deg)
        depth = float(self.depth_km)
        L_atm = float(self.L_atm_km)
        L_und = float(self.L_und_km)
        return (
            f"AtmosphereProfile | "
            f"source={self.atmosphere_density_source} | "
            f"h={h:.1f} km | "
            f"θ={theta:.1f}° | "
            f"depth={depth:.2f} km | "
            f"L_atm={L_atm:.1f} km | "
            f"L_und={L_und:.2f} km | "
            f"nsteps={self.nsteps} | "
            f"matter={self.matter} | "
            f"{self.device} / {self.dtype}"
        )

    __repr__ = __str__

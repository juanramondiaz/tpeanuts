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
Earth electron-density profiles and trajectory geometry.

The EarthProfile model stores Earth shell radii and delegates the density
model algebra to a perturbative layered profile. It transforms the model to a
neutrino trajectory, provides crossed-shell geometry, and evaluates pointwise
electron densities along Earth-crossing paths.

EarthProfile methods:

    shells_x(...): Return shell intersections and crossing masks.
    trajectory_profile(...): Build the perturbative profile in trajectory coordinates.
    density_x_eta(...): Evaluate pointwise trajectory electron density.
    density_n_x_eta(...): Evaluate pointwise trajectory neutron density.
    call(...) and __call__(...): Pointwise compatibility interfaces.
    call_neutron(...): Pointwise neutron-density compatibility interface.

density_n_x_eta / call_neutron require a perturbative profile model built
with neutron-density coefficients (EvenPowerProfileLayered and
PremTabulatedProfile both support include_neutron=True, reading a
composition-derived n_n(r) table alongside their default n_e(r) table); they
raise a clear ValueError otherwise. See
core.common.hamiltonian.hamiltonian_matter_reduced for the 3+1 sterile
extension's neutral-current matter term these feed.

"""



from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch

import tpeanuts.util.constant as constant
import tpeanuts.config.default as default
from tpeanuts.core.numerical.geometry import OdeMethod
from tpeanuts.medium.earth.probability import PearthMethod
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import TensorLike, as_tensor
from tpeanuts.core.perturbative.models import perturbative_profile_selection

Tensor = torch.Tensor


@dataclass(frozen=True)
class EarthParameters:
    """
    Earth electron-density profile and matter-regeneration probability
    settings.

    Parameters
    ----------
    profile_perturbative_name:
        Public name of the perturbative layered-density model used to build
        Earth's shell structure (e.g. "even_power").

    profile_perturbative_kwargs:
        Keyword arguments forwarded to that model's constructor, e.g. the
        density CSV path or whether to use the tabulated density. If
        omitted, the selected model applies its own defaults.

    profile_scale_m:
        Physical scale, in metres, of the profile's radial coordinate
        (Earth radius by default).

    evolution_scale_m:
        Physical scale, in metres, used to non-dimensionalize the
        Hamiltonian evolution. Normally equal to ``profile_scale_m``; kept
        separate to allow independent unit testing.

    depth_m:
        Detector depth below the Earth's surface, in metres. Shifts the
        Earth-crossing path length used by both the analytical and
        numerical Earth-regeneration probability pipelines.

    method:
        Earth matter-regeneration probability pipeline: "analytical" uses
        the perturbative evolutor, "numerical" integrates the
        medium-independent segment evolutor along a sampled trajectory.

    reunitarize:
        For ``method="analytical"``, project the Earth evolution operator
        back onto the nearest unitary matrix to absorb small numerical
        drift from the perturbative expansion.

    nsteps:
        Number of numerical trajectory segments used by
        ``method="numerical"``.

    chunk_eta:
        Number of nadir-angle exposure samples evaluated per batch in
        exposure-integrated probabilities. ``None`` or a non-positive value
        evaluates the full eta grid at once.

    ode_method:
        Sampling rule used to pick the representative point of each
        numerical trajectory segment ("midpoint", "left", or "right").
        Only used by ``method="numerical"``.

    massbasis:
        Selects whether the propagated state is interpreted as incoherent
        mass-basis weights (``True``) or coherent flavour-basis amplitudes
        (``False``).

    full_oscillation:
        For ``method="numerical"``, return probabilities along the full
        sampled trajectory plus the x grid instead of only the final point.
    """

    profile_perturbative_name: str = default.earth_profile_perturbative_name
    profile_perturbative_kwargs: dict[str, Any] | None = None
    profile_scale_m: TensorLike = constant.R_E
    evolution_scale_m: TensorLike = constant.R_E
    depth_m: float = default.earth_depth_m
    method: PearthMethod = default.earth_method
    reunitarize: bool = default.earth_reunitarize
    nsteps: int = default.earth_probability_nsteps
    chunk_eta: Optional[int] = default.earth_chunk_eta
    ode_method: OdeMethod | None = default.earth_numerical_method
    massbasis: bool = default.earth_massbasis
    full_oscillation: bool = default.earth_full_oscillation


@dataclass
class EarthProfile:
    """
    Torch representation of the peanuts earth electron-density profile.

    Parameters
    ----------
    params:
        Earth electron-density profile construction settings.

    context:
        Runtime device/dtype used to build and evaluate the profile.
    """

    params: EarthParameters = field(default_factory=EarthParameters)
    context: RuntimeContext = field(
        default_factory=lambda: RuntimeContext.resolve(default.device, default.dtype)
    )

    def __post_init__(self) -> None:
        """Normalize and validate shell geometry and delegated profile model."""
        device = self.context.device
        dtype = self.context.dtype
        profile_kwargs = (
            {}
            if self.params.profile_perturbative_kwargs is None
            else dict(self.params.profile_perturbative_kwargs)
        )
        profile_kwargs.setdefault("device", device)
        profile_kwargs.setdefault("dtype", dtype)
        self.profile_perturbative_kwargs = profile_kwargs
        self.profile_perturbative_name = self.params.profile_perturbative_name
        self.dtype = dtype

        self._profile_perturbative = perturbative_profile_selection(
            self.profile_perturbative_name,
            profile_kwargs,
        )

        if not hasattr(self._profile_perturbative, "rj"):
            raise ValueError(
                "Selected perturbative profile must expose shell radii as rj."
            )
        self.rj = torch.as_tensor(
            self._profile_perturbative.rj,
            device=self._profile_perturbative.device,
            dtype=self._profile_perturbative.dtype,
        )
        if not torch.is_floating_point(self.rj):
            self.rj = self.rj.to(dtype=torch.float64)
        self.device = self.rj.device
        self.dtype = self.rj.dtype
        if self.rj.ndim != 1 or self.rj.numel() == 0:
            raise ValueError("rj must be a non-empty one-dimensional tensor.")
        if torch.any(torch.diff(self.rj) <= 0):
            raise ValueError("rj must be strictly increasing.")
        if torch.any((self.rj <= 0) | (self.rj > 1)):
            raise ValueError("rj values must satisfy 0 < rj <= 1.")

        self.profile_scale_m = as_tensor(
            self.params.profile_scale_m,
            device=self.device,
            dtype=self.dtype,
        )
        self.evolution_scale_m = as_tensor(
            self.params.evolution_scale_m,
            device=self.device,
            dtype=self.dtype,
        )
        if self.profile_scale_m.ndim != 0 or self.profile_scale_m <= 0:
            raise ValueError("profile_scale_m must be a positive scalar.")
        if self.evolution_scale_m.ndim != 0 or self.evolution_scale_m <= 0:
            raise ValueError("evolution_scale_m must be a positive scalar.")

    @torch.no_grad()
    def shells_x(
        self,
        eta: TensorLike,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return shell coordinates, crossing mask, and first crossed index.

        Args:
            eta: Detector nadir angle in radians.

        Returns:
            Tuple (xj, crossed, idx0) for every shell and input angle.
        """
        eta = as_tensor(eta, device=self.device, dtype=self.dtype)

        s = torch.sin(eta)
        s_col = s[..., None]
        r = self.rj.reshape(*((1,) * eta.ndim), -1)
        crossed = r > s_col

        idx0 = torch.searchsorted(self.rj, s)

        x2 = torch.clamp(r * r - s_col * s_col,min=0.0)
        xj_all = torch.sqrt(x2)

        return xj_all, crossed, idx0

    @torch.no_grad()
    def trajectory_profile(
        self,
        eta: TensorLike,
    ) -> tuple[object, Tensor, Tensor]:
        """Build the layered model in trajectory coordinates.

        ``profile_perturbative_name`` selects the model class used to interpret
        the stored perturbative profile data. 
        
        Internally, ``EarthProfile`` keeps that model in the radial 
        coordinate ``r``.
        
        For a neutrino trajectory with nadir angle ``eta``, the chord 
        coordinate ``x`` satisfies:
            
            r**2 = x**2 + sin(eta)**2
        
        The shifted profile returned here is therefore the same layered density 
        model rewritten in ``x`` for that particular trajectory.

        Args:
            eta: Detector nadir angle in radians.

        Returns:
            Tuple ``(profile_trajectory, xj_all, crossed)``. 
            
        Notes:
                The first element contains the selected perturbative profile
            already transformed from radial coordinates to the trajectory
            coordinate.
                ``xj_all`` stores shell-intersection  positions along the trajectory
            and ``crossed`` marks which shells are physically crossed.
        """
        eta = as_tensor(eta, device=self.device, dtype=self.dtype)
        xj_all, crossed, _ = self.shells_x(eta)
        profile_trajectory = self._profile_perturbative.shifted(
            torch.sin(eta) ** 2
        )
        return profile_trajectory, xj_all, crossed

    def constant_segment_model(
        self,
        **kwargs,
    ) -> object:
        """Build a constant-density segment model with the selected profile."""
        return self._profile_perturbative.constant_segment_model(**kwargs)

    @torch.no_grad()
    def density_x_eta(
        self,
        x: TensorLike,
        eta: TensorLike,
    ) -> Tensor:
        """Evaluate pointwise electron density along a trajectory.

        Args:
            x: Dimensionless coordinate along the Earth chord.
            eta: Detector nadir angle in radians.

        Returns:
            Electron density in mol/cm^3 with broadcast input shape.
        """
        dev = self.device
        dt = self.dtype
        x = as_tensor(x, device=dev, dtype=dt)
        eta = as_tensor(eta, device=dev, dtype=dt)

        xabs = torch.abs(x)
        eta_b = eta + torch.zeros_like(xabs)
        cos_eta = torch.cos(eta_b)
        outside = xabs > cos_eta

        trajectory_profile, xj, crossed = self.trajectory_profile(eta_b)

        neg_inf = torch.tensor(float("-inf"), device=dev,dtype=dt )
        xj_eff = torch.where(crossed, xj, neg_inf)

        idx = torch.searchsorted(xj_eff, xabs[..., None], right=False).squeeze(-1)

        idx = torch.clamp(idx, 0, xj_eff.shape[-1] - 1)

        n_e = trajectory_profile.evaluate(xabs, layer_index=idx)
        n_e = torch.where(outside, torch.zeros_like(n_e), n_e)

        return n_e

    @torch.no_grad()
    def density_n_x_eta(
        self,
        x: TensorLike,
        eta: TensorLike,
    ) -> Tensor:
        """Evaluate pointwise neutron density along a trajectory.

        Enables the 3+1 sterile extension's neutral-current matter term (see
        ``core.common.hamiltonian.hamiltonian_matter_reduced``). Requires the
        selected perturbative profile model to be built with neutron-density
        coefficients: both ``EvenPowerProfileLayered`` and
        ``PremTabulatedProfile`` support this via ``include_neutron=True``
        (or an explicit ``coefficients_n``).

        Args:
            x: Dimensionless coordinate along the Earth chord.
            eta: Detector nadir angle in radians.

        Returns:
            Neutron density in mol/cm^3 with broadcast input shape.

        Raises:
            ValueError: If the selected profile model was not built with
                neutron-density coefficients (``include_neutron=True`` /
                ``coefficients_n``), or does not support neutron density at
                all.
        """
        if not hasattr(self._profile_perturbative, "evaluate_neutron"):
            raise ValueError(
                f"The '{self.profile_perturbative_name}' Earth profile model "
                "does not provide neutron-density data required for the 3+1 "
                "sterile extension's neutral-current (NC) matter term. Use "
                "the PREM tabulated profile instead "
                "(profile_perturbative_name='prem500', "
                "profile_perturbative_kwargs={'include_neutron': True})."
            )

        dev = self.device
        dt = self.dtype
        x = as_tensor(x, device=dev, dtype=dt)
        eta = as_tensor(eta, device=dev, dtype=dt)

        xabs = torch.abs(x)
        eta_b = eta + torch.zeros_like(xabs)
        cos_eta = torch.cos(eta_b)
        outside = xabs > cos_eta

        trajectory_profile, xj, crossed = self.trajectory_profile(eta_b)

        neg_inf = torch.tensor(float("-inf"), device=dev, dtype=dt)
        xj_eff = torch.where(crossed, xj, neg_inf)

        idx = torch.searchsorted(xj_eff, xabs[..., None], right=False).squeeze(-1)
        idx = torch.clamp(idx, 0, xj_eff.shape[-1] - 1)

        n_n = trajectory_profile.evaluate_neutron(xabs, layer_index=idx)
        n_n = torch.where(outside, torch.zeros_like(n_n), n_n)

        return n_n

    def call(
        self,
        r: Tensor,
        eta: Tensor,
    ) -> Tensor:
        return self.density_x_eta(r, eta)

    def call_neutron(
        self,
        r: Tensor,
        eta: Tensor,
    ) -> Tensor:
        return self.density_n_x_eta(r, eta)

    def __call__(
        self,
        r: Tensor,
        eta: Tensor,
    ) -> Tensor:
        return self.call(r, eta)

    def __str__(self) -> str:
        """Return a compact summary of the Earth profile configuration."""
        n_shells = self.rj.numel()
        r_inner  = float(self.rj[0])
        r_outer  = float(self.rj[-1])
        scale_km = float(self.profile_scale_m) / 1.0e3
        depth_m  = float(self.params.depth_m)
        return (
            f"EarthProfile | "
            f"model={self.profile_perturbative_name} | "
            f"n_shells={n_shells} | "
            f"r=[{r_inner:.4f}, {r_outer:.4f}] R_E ({scale_km:.0f} km scale) | "
            f"depth={depth_m:.0f} m | "
            f"method={self.params.method} | "
            f"{self.device} / {self.dtype}"
        )

    __repr__ = __str__


def build_earth_profile(
    earth_profile: Optional[EarthProfile],
    *,
    params: Optional[EarthParameters] = None,
    context: RuntimeContext,
) -> EarthProfile:
    """Return an existing Earth profile or build one from configured defaults."""
    if earth_profile is not None:
        return earth_profile
    return EarthProfile(params=params or EarthParameters(), context=context)
    

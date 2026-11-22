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

"""``AtmosphericSource``: the domain object the full generated flux grid becomes, and its bridge to
``interfaces.Flux``.

Mirrors ``tpeanuts.source.solar.solar.SolarSource``'s role: the
flat rows a loader returns (here, ``source.atmosphere.io.
AtmosphericFluxTable``, read from files ``mceq.py``/``honda.py``
generated, one per zenith angle) are reshaped and stacked into a
dense, validated, provider-neutral in-memory object with real
behaviour, not just carried around as flat tables. Holds the *entire*
loaded flux, every energy, height, **and** zenith angle at once
(``eta_rad`` a genuine 1-D grid, not one angle at a time) -- every
computation here, and every downstream one in
``tpeanuts.pipeline.atmospheric_neutrino``, is batchable across all
three axes independently (see ``tpeanuts.pipeline.atmosphere.
atmosphere_evolutor``'s own docstring for how that batching actually
broadcasts through the propagation machinery). Unlike the solar case,
production height and energy are *not* decoupled here into a separate
``fractions``/``fluxes`` split -- a neutrino's mean production height
genuinely depends on its energy (higher-energy secondaries decay
higher up), so ``phi_Eh`` is kept as one joint ``(n_angle, n_E, n_h)``
tensor per particle, and ``production_distribution``/``total_flux``
are *views* over it (normalized-in-h, and h-integrated respectively)
rather than independently-stored factors.

Only files with a production-height axis are accepted (``altitude_m``
must be present, i.e. one of ``mceq.py``/``honda.py``'s own generated
per-angle files) -- the flat, no-height-axis Honda table this project
also ships (``tpeanuts/data/atmosphere/honda/flux/honda_flux.csv``) has
no real use here: ``tpeanuts.medium.atmosphere.geometry.
atmosphere_chord_length_m`` itself requires ``h_m > 0`` strictly (a
neutrino produced at or below the surface has no atmosphere leg to
propagate), so a flat table simply cannot supply what this pipeline
needs.

``AtmosphericSource.particle_flux`` is the piece ``tpeanuts.source.atmosphere.io``'s
own module docstring left for "a future ``tpeanuts.pipeline.atmosphere``
to handle" -- converting a loaded file's own ``cos_zenith`` into this
project's own nadir-angle ``eta_rad`` (``eta_rad = arccos(cos_zenith)``, both
already the same "0 = straight down" convention, see
``tpeanuts.medium.atmosphere.geometry``'s own module docstring, done
once per angle inside ``.default()``) and building the
``interfaces.Flux`` ``tpeanuts.pipeline.atmosphere`` already accepts
today, unchanged -- mirroring ``tpeanuts.source.reactor.reactor.
ReactorSource.flux()``'s own "build a plain Standard Model
FlavourSpace, a pure basis state, done" shape, not solar's
weights-based composition (the atmosphere pipeline already evolves a
coherent ``Flux.state`` through matter via
``evolve_path``/``conjugate_by`` -- there is no MSW-style adiabatic
mass-eigenstate collapse at production the way there is in the Sun, so
there is nothing for this bridge to do beyond picking the right basis
state and energy grid). ``Flux`` itself carries no angle axis --
``particle``'s initial flavour composition does not depend on the
production angle, only the propagation trajectory does, so the angle
grid stays entirely on the ``AtmosphericSource``/medium side.

``AtmosphericSource.flux_weights_integrate`` is the atmosphere
analogue of ``SolarSource.mass_weights_integrate``:
``tpeanuts.pipeline.atmospheric_neutrino`` returns every ``(angle, h,
E)`` result stacked, unintegrated (see that module's own docstring for
why height stays a caller-visible axis rather than being folded in
there) -- this method is the piece that stack is for, folding its own
height axis away via a flux-weighted (``production_distribution``
-weighted, trapezoidal) reduction into one ``(angle, E, ...)`` result,
mirroring ``mass_weights_integrate``'s own "reduce a medium-computed,
per-radius quantity against this source's own production distribution"
role exactly, just over height instead of radius, and with a fixed
``(angle, h, E, ...)`` axis convention (matching
``tpeanuts.pipeline.atmospheric_neutrino``'s own, the only producer of
the shape this consumes) rather than solar's more general
``energy_ndim`` parameterization -- production height genuinely depends
on energy here (see above), so there is no "many sources, one shared
energy batch" shape to generalize over the way solar's own several
independent production sources need.

Module contents:
    AtmosphericSource
        Frozen dataclass: the dense per-particle ``Phi(angle, E, h)``
        grid for every generated zenith angle loaded, with
        ``production_distribution``/``total_flux``/
        ``flux_weights_integrate`` views, a ``.flux()`` presenting it as a ``SourceFlux`` with a direction axis, and a
        ``.default()`` loader, a ``.particle_flux()`` giving the plain ``interfaces.Flux`` of one species, and
        ``device``/``dtype``/``flux_unit`` properties and a compact ``__str__``.
    AtmosphericSourceParameters
        Frozen dataclass of the construction settings for ``AtmosphericSource.default`` (angles, provider,
        directory).
    build_atmospheric_source(...)
        Return the given source, or load the default one from ``AtmosphericSourceParameters``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch

from tpeanuts.config import default
from tpeanuts.interfaces.flux import Flux, ProductionDistribution, SourceFlux
from tpeanuts.physics.common.space import FlavourSpace
from tpeanuts.physics.common.state import DensityMatrix
from tpeanuts.source.atmosphere.io import angle_output_path, load_atmospheric_flux
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.geometry import theta_detector_to_alpha_surface
from tpeanuts.util.tensor import TensorLike, as_tensor

_FLAVOUR_INDEX = {"nue": 0, "antinue": 0, "numu": 1, "antinumu": 1, "nutau": 2, "antinutau": 2}

# Factor taking a provider's native unit to cm^-2 s^-1 sr^-1 MeV^-1 (MCEq writes cm^-2, Honda m^-2; both per GeV).
_TO_PER_CM2_PER_MEV = {"mceq": 1.0e-3, "honda": 1.0e-7}

_NATIVE_UNIT = {"mceq": "cm^-2 s^-1 sr^-1 GeV^-1", "honda": "m^-2 s^-1 sr^-1 GeV^-1"}
_PROVIDER_FLUX_DIRNAME = {
    "mceq": default.ATMOSPHERE_MCEQ_FLUX_DIRNAME,
    "honda": default.ATMOSPHERE_HONDA_FLUX_DIRNAME,
}


def _pivot_flat_table_to_grid(
    table, *, path: Path
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Reshape a flat ``AtmosphericFluxTable`` (one row per particle/energy/altitude) into a dense ``(n_E, n_h)`` grid
    per particle.

    Relies on the row order ``tpeanuts.source.atmosphere.io.
    write_angle_flux_csv`` itself writes in: every particle's rows
    contiguous, nested energy-then-altitude -- exactly the layout every
    file this module reads was written in.

    Raises:
        ValueError: If the table has no ``altitude_m`` axis, if its row
            count is not evenly divisible across particles/energies/
            altitudes, if the extracted energy/altitude grid is not
            strictly increasing, or if particles disagree on the grid.
    """
    if table.altitude_m is None:
        raise ValueError(
            f"{path}: this table has no production-height axis (no 'altitude_km' column) -- "
            "AtmosphericSource needs a file generated by "
            "tpeanuts.source.atmosphere.mceq/honda, not the flat Honda surface table."
        )

    particles = []
    seen = set()
    for p in table.particle:
        if p not in seen:
            particles.append(p)
            seen.add(p)

    n_rows = table.energy_MeV.numel()
    n_particles = len(particles)
    if n_particles == 0 or n_rows % n_particles != 0:
        raise ValueError(f"{path}: row count ({n_rows}) is not evenly divisible across {n_particles} particles.")
    block = n_rows // n_particles

    energy_MeV: Optional[torch.Tensor] = None
    altitude_m: Optional[torch.Tensor] = None
    phi_Eh: Dict[str, torch.Tensor] = {}

    for i, particle in enumerate(particles):
        rows = slice(i * block, (i + 1) * block)
        e_block, h_block, flux_block = table.energy_MeV[rows], table.altitude_m[rows], table.flux[rows]

        n_h = int(torch.sum(e_block == e_block[0]).item())
        if n_h == 0 or block % n_h != 0:
            raise ValueError(f"{path}: could not infer a rectangular (n_energy, n_altitude) grid for {particle!r}.")
        n_E = block // n_h

        e_grid = e_block.reshape(n_E, n_h)[:, 0]
        h_grid = h_block.reshape(n_E, n_h)[0, :]
        if torch.any(torch.diff(e_grid) <= 0) or torch.any(torch.diff(h_grid) <= 0):
            raise ValueError(
                f"{path}: extracted energy/altitude grid for {particle!r} is not strictly increasing -- "
                "rows must be in the nested (particle, energy, altitude) order write_angle_flux_csv writes."
            )

        if energy_MeV is None:
            energy_MeV, altitude_m = e_grid, h_grid
        elif altitude_m is None or not (torch.allclose(energy_MeV, e_grid) and torch.allclose(altitude_m, h_grid)):
            raise ValueError(
                f"{path}: particle {particle!r} uses a different energy/altitude grid than the first particle."
            )

        phi_Eh[particle] = flux_block.reshape(n_E, n_h)

    assert energy_MeV is not None and altitude_m is not None  # the table has at least one particle
    return energy_MeV, altitude_m, phi_Eh


def _available_alphas(output_dir: Path, provider: str) -> list:
    """Surface zenith angles (degrees, increasing) of the ``flux_alpha_<alpha>deg.csv`` files in ``output_dir``."""
    alphas = []
    for path in output_dir.glob("flux_alpha_*deg.csv") if output_dir.is_dir() else ():
        try:
            alphas.append(float(path.name[len("flux_alpha_"):-len("deg.csv")]))
        except ValueError:
            continue
    if not alphas:
        raise FileNotFoundError(
            f"No {provider} flux files (flux_alpha_<alpha>deg.csv) in {output_dir} -- generate them first with "
            f"tpeanuts.source.atmosphere.{provider}.generate_{provider}_flux, or give the angles explicitly."
        )
    return sorted(alphas)


@dataclass(frozen=True)
class AtmosphericSourceParameters:
    """Construction settings of ``AtmosphericSource.default`` (the equivalent of ``SolarSourceParameters``).

    Attributes:
        theta_deg: Detector zenith angle(s) in degrees, scalar or a 1-D grid. Mutually exclusive with ``alpha_deg``.
        alpha_deg: Surface/table zenith angle(s) in degrees, scalar or a 1-D grid. Mutually exclusive with
            ``theta_deg``. With neither, every generated angle found is loaded.
        depth_m: Detector depth below the surface, in metres, used only to convert ``theta_deg`` to ``alpha_deg``.
        provider: ``"mceq"`` or ``"honda"`` -- which generator output directory to look in.
        output_dir: Directory the per-angle CSVs are read from. None uses the provider directory under
            ``default.ATMOSPHERE_DATA_DIR``.
    """

    theta_deg: Optional[TensorLike] = None
    alpha_deg: Optional[TensorLike] = None
    depth_m: TensorLike = 0.0
    provider: str = "mceq"
    output_dir: Optional[Path] = None


@dataclass(frozen=True)
class AtmosphericSource:
    """The dense, validated ``Phi(angle, E, h)`` grid this project's atmospheric-flux generators have produced.

    Holds the *full* atmospheric neutrino flux -- every energy, every
    production height, and every zenith angle loaded -- not one angle
    at a time: ``eta_rad`` is a genuine 1-D grid, and every other field
    carries that same leading angle axis, so every computation here
    (``production_distribution``/``total_flux``, and everything
    downstream in ``tpeanuts.pipeline.atmospheric_neutrino``) is
    batchable across angle exactly as it already is across energy and
    height.

    Attributes:
        energy_MeV: Strictly increasing 1-D energy grid, shape
            ``(n_E,)``.
        altitude_m: Strictly increasing 1-D production-altitude grid
            above the surface, shape ``(n_h,)``, in metres.
        eta_rad: Nadir angle, in radians, ``[0, pi]``, shape ``(n_angle,)``
            -- one entry per generated file loaded into this source
            (``eta_rad=0`` straight down, the same convention
            ``tpeanuts.medium.atmosphere.geometry`` uses throughout;
            see ``.default()`` for how a file's own ``cos_zenith``
            becomes one entry here).
        phi_Eh: Mapping from particle label to its height-differential
            flux, each shaped ``(n_angle, n_E, n_h)``, in whatever unit
            the generator's own metadata documents (``cm^-2 s^-1 sr^-1
            GeV^-1`` for both ``mceq.py``/``honda.py``).
        provider: Which generator produced this source's own files
            (``"mceq"`` or ``"honda"``), carried as provenance only.
        context: RuntimeContext every tensor here is built on.

    Raises:
        ValueError: If ``energy_MeV``/``altitude_m``/``eta_rad`` is not
            one-dimensional (``energy_MeV``/``altitude_m`` strictly
            increasing, ``eta_rad`` within ``[0, pi]``), or if any
            ``phi_Eh`` entry's shape does not match ``(len(eta_rad),
            len(energy_MeV), len(altitude_m))`` or contains a
            negative/non-finite value.

    Examples:
        >>> import torch
        >>> import tpeanuts as tp
        >>> from tpeanuts.source import AtmosphericSource
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> source = AtmosphericSource(   # flux per (angle, energy, height); normally read from files
        ...     energy_MeV=torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64),
        ...     altitude_m=torch.tensor([0.0, 5000.0, 15000.0], dtype=torch.float64),
        ...     eta_rad=torch.tensor([0.6], dtype=torch.float64),
        ...     phi_Eh={"numu": torch.ones(1, 3, 3, dtype=torch.float64)},
        ...     provider="mceq", context=context)
        >>> source.phi_Eh["numu"].shape   # (angle, energy, height)
        torch.Size([1, 3, 3])
    """

    energy_MeV: torch.Tensor
    altitude_m: torch.Tensor
    eta_rad: torch.Tensor
    phi_Eh: Dict[str, torch.Tensor]
    provider: str
    context: RuntimeContext

    def __post_init__(self) -> None:
        energy_MeV = as_tensor(self.energy_MeV, device=self.context.device, dtype=self.context.dtype)
        altitude_m = as_tensor(self.altitude_m, device=self.context.device, dtype=self.context.dtype)
        eta_rad = as_tensor(self.eta_rad, device=self.context.device, dtype=self.context.dtype).reshape(-1)
        if energy_MeV.ndim != 1 or torch.any(torch.diff(energy_MeV) <= 0):
            raise ValueError("energy_MeV must be one-dimensional and strictly increasing.")
        if altitude_m.ndim != 1 or torch.any(torch.diff(altitude_m) <= 0):
            raise ValueError("altitude_m must be one-dimensional and strictly increasing.")
        if torch.any(eta_rad < 0.0) or torch.any(eta_rad > math.pi):
            raise ValueError("eta_rad must satisfy 0 <= eta_rad <= pi.")

        phi_Eh: Dict[str, torch.Tensor] = {}
        expected_shape = (eta_rad.numel(), energy_MeV.numel(), altitude_m.numel())
        for particle, flux_Eh in self.phi_Eh.items():
            flux_t = as_tensor(flux_Eh, device=self.context.device, dtype=self.context.dtype)
            if tuple(flux_t.shape) != expected_shape:
                raise ValueError(
                    f"phi_Eh[{particle!r}] has shape {tuple(flux_t.shape)}, expected {expected_shape}."
                )
            if torch.any(~torch.isfinite(flux_t)) or torch.any(flux_t < 0):
                raise ValueError(f"phi_Eh[{particle!r}] must be finite and non-negative everywhere.")
            phi_Eh[particle] = flux_t

        object.__setattr__(self, "energy_MeV", energy_MeV)
        object.__setattr__(self, "altitude_m", altitude_m)
        object.__setattr__(self, "eta_rad", eta_rad)
        object.__setattr__(self, "phi_Eh", phi_Eh)

    def production_distribution(self, particle: str) -> torch.Tensor:
        """Normalized production-height density ``f(h|angle,E)`` for one particle.

        ``phi_Eh[particle]`` divided by its own trapezoidal integral
        over ``altitude_m``, per ``(angle, energy)`` row -- the
        atmosphere analogue of ``SolarSource.
        production_distribution``, except kept as a joint ``(n_angle,
        n_E, n_h)`` view rather than a separately-stored factor (see
        this module's own docstring for why).

        Args:
            particle: Particle label, a key of ``phi_Eh``.

        Returns:
            Tensor shaped ``(n_angle, n_E, n_h)``, integrating to 1
            over ``h`` for every ``(angle, energy)`` row whose total
            flux is positive (an all-zero row otherwise, not NaN/Inf).

        Raises:
            KeyError: If ``particle`` is not in ``phi_Eh``.
        """
        flux_Eh = self.phi_Eh[particle]
        norm = torch.trapezoid(flux_Eh, x=self.altitude_m, dim=-1)
        return torch.where(
            (norm > 0.0).unsqueeze(-1),
            flux_Eh / norm.clamp_min(torch.finfo(flux_Eh.dtype).tiny).unsqueeze(-1),
            torch.zeros_like(flux_Eh),
        )

    def total_flux(self, particle: str) -> torch.Tensor:
        """Height-integrated flux ``Phi(angle,E) = integral phi_Eh(angle,E,h) dh`` for one particle.

        Unlike ``SolarSource.total_flux`` (a single scalar per
        source), this stays a spectrum, shape ``(n_angle, n_E)``: the
        atmospheric flux is intrinsically energy- (and angle-)
        differential, with no separately-known "total integrated
        number" to normalize against (see this module's own
        docstring).

        Args:
            particle: Particle label, a key of ``phi_Eh``.

        Returns:
            Tensor shaped ``(n_angle, n_E)``.

        Raises:
            KeyError: If ``particle`` is not in ``phi_Eh``.
        """
        return torch.trapezoid(self.phi_Eh[particle], x=self.altitude_m, dim=-1)

    def flux_weights_integrate(self, particle: str, weights: torch.Tensor) -> torch.Tensor:
        """Fold a propagated result's own height axis away, weighted by ``production_distribution(particle)``.

        The atmosphere analogue of
        ``tpeanuts.source.solar.solar.SolarSource.mass_weights_integrate``
        (see this module's own docstring for how the two differ) --
        intended for ``weights`` shaped exactly as
        ``tpeanuts.pipeline.atmospheric_neutrino.
        atmospheric_neutrino_probability_state``'s own per-particle
        result (``(n_angle, n_h, n_E, n_flavour)``), though any trailing
        shape after ``(n_angle, n_h, n_E)`` broadcasts through unchanged
        (e.g. ``atmospheric_neutrino_evolutor``'s own ``(n_angle, n_h,
        n_E, n_flavour, n_flavour)`` evolutors too, if a caller has a
        genuine use for a flux-weighted *evolutor* -- unlike a
        probability, that average is not itself a physical evolution
        operator, so this method takes no position on whether that is
        meaningful).

        Args:
            particle: Particle label, a key of ``phi_Eh``.
            weights: Propagated per-particle result. Axis 0 must have
                ``self.eta_rad.numel()`` entries, axis 1
                ``self.altitude_m.numel()``, axis 2
                ``self.energy_MeV.numel()`` -- ``atmospheric_neutrino_evolutor``'s/
                ``atmospheric_neutrino_probability_state``'s own axis
                order, unchanged.

        Returns:
            ``weights`` with its own height axis (axis 1) removed,
            shaped ``(n_angle, n_E, ...)``.

        Raises:
            KeyError: If ``particle`` is not in ``phi_Eh``.
            ValueError: If ``weights``'s leading three axes do not match
                ``(n_angle, n_h, n_E)``.
        """
        density = self.production_distribution(particle)  # (n_angle, n_E, n_h)
        expected = (density.shape[0], density.shape[2], density.shape[1])
        if weights.ndim < 3 or tuple(weights.shape[:3]) != expected:
            raise ValueError(
                f"weights must be shaped (n_angle, n_h, n_E, ...) = {expected + ('...',)}; "
                f"got {tuple(weights.shape)}."
            )
        density_angle_h_E = density.transpose(-1, -2)  # (n_angle, n_h, n_E)
        density_angle_h_E = density_angle_h_E.reshape(*density_angle_h_E.shape, *((1,) * (weights.ndim - 3)))
        return torch.trapezoid(weights * density_angle_h_E, x=self.altitude_m, dim=1)

    def particle_flux(self, particle: str) -> Flux:
        """The plain ``Flux`` of one particle species: this source energy grid and the species initial flavour.

        A pure basis state in the plain Standard Model ``FlavourSpace()`` (a source is not a BSM scenario by itself),
        ``antinu=True`` iff ``particle`` names an antineutrino. It carries no intensity, direction or production
        distribution; ``flux`` adds them.

        Args:
            particle: Particle label, a key of ``phi_Eh`` (e.g. ``"numu"``, ``"antinue"``).

        Returns:
            Flux with ``energy_MeV = self.energy_MeV``.

        Raises:
            ValueError: If ``particle`` is not in ``phi_Eh`` or has no known flavour mapping.
        """
        if particle not in self.phi_Eh:
            raise ValueError(f"particle={particle!r} is not one of this source's own particles: {sorted(self.phi_Eh)}.")
        key = particle.lower()
        if key not in _FLAVOUR_INDEX:
            raise ValueError(f"No flavour-index mapping for particle={particle!r}. Known: {sorted(_FLAVOUR_INDEX)}.")
        state = DensityMatrix.basis_state(_FLAVOUR_INDEX[key], FlavourSpace(), self.context)
        return Flux(energy_MeV=self.energy_MeV, state=state, antinu=key.startswith("anti"))

    @property
    def device(self) -> torch.device:
        """Return the device of this source tensors (``context.device``)."""
        return self.context.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the real dtype of this source tensors (``context.dtype``)."""
        return self.context.dtype

    @property
    def flux_unit(self) -> str:
        """Unit of the provider native ``phi_Eh`` (``flux`` converts it to ``cm^-2 s^-1 sr^-1 MeV^-1``)."""
        return _NATIVE_UNIT.get(self.provider, "unknown")

    def __str__(self) -> str:
        """Return a compact summary of the atmospheric source configuration."""
        return (
            f"AtmosphericSource | n_angle={self.eta_rad.numel()} | n_E={self.energy_MeV.numel()} | "
            f"n_h={self.altitude_m.numel()} | particles=[{', '.join(sorted(self.phi_Eh))}] | "
            f"provider={self.provider} | {self.device} / {self.dtype}"
        )

    __repr__ = __str__

    def flux(self, particles: Optional[Sequence[str]] = None) -> SourceFlux:
        """Present the atmospheric flux as a ``SourceFlux``: one ``Flux`` per particle species, with a direction axis.

        Each component has the initial flavour of its species and the nadir angles of this source as ``direction``;
        ``differential_flux`` is the height-integrated flux ``total_flux(particle)``, shaped ``(n_angle, n_E)``, in
        ``cm^-2 s^-1 sr^-1 MeV^-1`` (the provider's own unit, converted: MCEq writes ``cm^-2`` and Honda ``m^-2``, both
        per GeV). ``production`` is the distribution of the production altitude in metres,
        ``production_distribution(particle)``, shaped ``(n_angle, n_E, n_h)``, which the atmosphere medium averages
        over.

        Args:
            particles: Particle labels, keys of ``phi_Eh``; all of them if omitted.

        Returns:
            SourceFlux with one component per particle, in the requested order, all at ``location="source"``.

        Raises:
            ValueError: If a particle is not in ``phi_Eh`` or has no known flavour, or the provider's unit is unknown.
        """
        if self.provider not in _TO_PER_CM2_PER_MEV:
            raise ValueError(f"unknown flux provider {self.provider!r}; expected one of {sorted(_TO_PER_CM2_PER_MEV)}.")
        scale = _TO_PER_CM2_PER_MEV[self.provider]
        keys = list(self.phi_Eh) if particles is None else list(particles)
        components = {}
        for key in keys:
            base = self.particle_flux(key)
            production = ProductionDistribution(
                radius=self.altitude_m, weight=self.production_distribution(key), context=self.context,
            )
            components[key] = Flux(
                energy_MeV=base.energy_MeV, state=base.state, antinu=base.antinu,
                differential_flux=scale * self.total_flux(key), flux_units="cm^-2 s^-1 sr^-1 MeV^-1",
                production=production, direction=self.eta_rad,
            )
        return SourceFlux(components)

    @classmethod
    def default(
        cls,
        *,
        params: Optional["AtmosphericSourceParameters"] = None,
        context: RuntimeContext,
    ) -> "AtmosphericSource":
        """Load generated per-angle flux files, stacked into one source (the equivalent of the solar ``default``).

        Locates each file exactly as ``generate_mceq_flux``/``generate_honda_flux`` themselves name it
        (``tpeanuts.source.atmosphere.io.angle_output_path``, keyed by the surface/table angle ``alpha_deg``) and
        reshapes/stacks them into one dense ``AtmosphericSource``. Does **not** generate a missing file itself
        -- an MCEq solve or a Honda-table interpolation is comparatively expensive, so generation stays an explicit,
        separate step the caller controls.

        Args:
            params: Construction settings (zenith angles, provider, directory). None, or settings without angles,
                loads every ``flux_alpha_*deg.csv`` found in the provider output directory, in increasing
                ``alpha_deg``.
            context: RuntimeContext the returned source tensors are built on.

        Returns:
            AtmosphericSource stacking every resolved angle, in the order given (or found).

        Raises:
            ValueError: If both ``theta_deg`` and ``alpha_deg`` are given, if ``provider`` is not
                ``"mceq"``/``"honda"``, or if two loaded angles disagree on their own energy/altitude grid or particle
                set.
            FileNotFoundError: If no file exists at a resolved path, or no angle was given and none is found.
        """
        settings = params if params is not None else AtmosphericSourceParameters()
        theta_deg, alpha_deg, depth_m = settings.theta_deg, settings.alpha_deg, settings.depth_m
        provider, output_dir = settings.provider, settings.output_dir
        if theta_deg is not None and alpha_deg is not None:
            raise ValueError("Provide at most one of theta_deg or alpha_deg.")
        if provider not in _PROVIDER_FLUX_DIRNAME:
            raise ValueError(f"provider must be one of {sorted(_PROVIDER_FLUX_DIRNAME)}, got {provider!r}.")
        if output_dir is None:
            output_dir = default.ATMOSPHERE_DATA_DIR / _PROVIDER_FLUX_DIRNAME[provider]
        if theta_deg is None and alpha_deg is None:
            alpha_deg = _available_alphas(Path(output_dir), provider)

        requested_angles = theta_deg if alpha_deg is None else alpha_deg
        assert requested_angles is not None  # one of the two was given or discovered (above)
        angle_values = as_tensor(requested_angles, device=context.device, dtype=context.dtype).reshape(-1)

        energy_MeV: Optional[torch.Tensor] = None
        altitude_m: Optional[torch.Tensor] = None
        particles: Optional[set] = None
        eta_list = []
        phi_Eh_lists: Dict[str, list] = {}

        for angle_value in angle_values:
            if alpha_deg is None:
                alpha_value = float(theta_detector_to_alpha_surface(angle_value, depth_m, context=context).item())
            else:
                alpha_value = float(angle_value.item())

            path = angle_output_path(Path(output_dir), alpha_value)
            if not path.exists():
                raise FileNotFoundError(
                    f"No {provider} flux file for alpha_deg={alpha_value} at {path} -- generate it first with "
                    f"tpeanuts.source.atmosphere.{provider}.generate_{provider}_flux([{alpha_value}], 'alpha', ...)."
                )

            table = load_atmospheric_flux(path, context=context)
            e_grid, h_grid, phi_Eh_angle = _pivot_flat_table_to_grid(table, path=path)

            if energy_MeV is None:
                energy_MeV, altitude_m, particles = e_grid, h_grid, set(phi_Eh_angle)
            elif altitude_m is None or not (
                torch.allclose(energy_MeV, e_grid) and torch.allclose(altitude_m, h_grid)
            ):
                raise ValueError(f"{path}: energy/altitude grid differs from the first loaded angle.")
            elif particles is not None and set(phi_Eh_angle) != particles:
                raise ValueError(
                    f"{path}: particle set {sorted(phi_Eh_angle)} differs from the first loaded angle's "
                    f"{sorted(particles)}."
                )

            cos_zenith_values = table.cos_zenith
            if torch.any(torch.abs(cos_zenith_values - cos_zenith_values[0]) > 1.0e-6):
                raise ValueError(f"{path}: not every row shares the same cos_zenith value.")
            eta_list.append(torch.acos(torch.clamp(cos_zenith_values[0], -1.0, 1.0)))

            for particle, flux_Eh in phi_Eh_angle.items():
                phi_Eh_lists.setdefault(particle, []).append(flux_Eh)

        assert energy_MeV is not None and altitude_m is not None  # at least one angle was loaded
        return cls(
            energy_MeV=energy_MeV,
            altitude_m=altitude_m,
            eta_rad=torch.stack(eta_list),
            phi_Eh={particle: torch.stack(blocks, dim=0) for particle, blocks in phi_Eh_lists.items()},
            provider=provider,
            context=context,
        )


def build_atmospheric_source(
    atmospheric_source: Optional[AtmosphericSource],
    *,
    params: Optional[AtmosphericSourceParameters] = None,
    context: RuntimeContext,
) -> AtmosphericSource:
    """Return ``atmospheric_source`` if given, otherwise load the default one (like ``build_solar_source``).

    Args:
        atmospheric_source: An already-built source, returned unchanged (``params`` is then ignored).
        params: Construction settings for the loaded source.
        context: RuntimeContext for the loaded source.

    Returns:
        The given source, or ``AtmosphericSource.default(params=params, context=context)``.
    """
    if atmospheric_source is not None:
        return atmospheric_source
    return AtmosphericSource.default(params=params, context=context)

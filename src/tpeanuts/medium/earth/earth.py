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

"""A real Earth medium: a PREM density chord, for a detector at any depth, in any of the three PropagationMethod values.

``EarthMedium.profile()`` -- fully batchable in ``eta_rad``/``depth_m`` for
all three ``PropagationMethod`` values, mixing case-A and case-B
entries freely in the same call (see ``profile()``'s own docstring for
how) -- first picks a geometric case per element (see
``tpeanuts.medium.earth.geometry``'s own module docstring), then, for
case A ("Earth-crossing"), one of two structurally different
``tpeanuts.interfaces.medium.MediumProfile``s, chosen by ``self.method``:

    ``PropagationMethod.NUMERICAL``/``ANALYTIC``
        ``nsteps`` segments tiling the chord from the far-side surface
        entry point to the detector itself, each sampled at its own
        midpoint and cut at the radii where the PREM table jumps
        (``tpeanuts.medium.earth.geometry.aligned_chord_segments``): the
        segments are shared out among the shells crossed, at least one
        each and the rest in proportion to length, so none takes one
        density for both sides of a jump. With ``nsteps`` below
        ``2 * 8 + 1`` the chord is tiled by equal segments instead. ``ANALYTIC`` needs no different
        segment data at all -- it is an exact drop-in replacement for
        ``NUMERICAL``'s ``matrix_exp`` on a constant-density segment, so
        the same discretization already serves both.
    ``PropagationMethod.ANALYTIC_PERTURBATIVE``
        Up to ``2*n_shells - 1`` PREM-shell-aligned segments
        (``tpeanuts.medium.earth.geometry.crossed_shell_segments_batched``)
        instead of ``nsteps`` -- ``nsteps`` is ignored for this method.
        Each segment carries a genuine within-segment density
        *variation* (the shell's own even-power fit, shifted from
        radius to chord coordinate by ``tpeanuts.medium.earth.even_power.
        shift_shell_coefficients_to_chord``), not just a point sample --
        this is where the segment-count reduction that is the entire
        point of ``ANALYTIC_PERTURBATIVE`` actually comes from. Fewer
        than ``2*n_shells - 1`` shells crossed pads the remainder with
        zero-length (physically inert, ``matrix_exp(H*0) == I``) slots
        rather than needing a different output shape per element --
        this is what makes a batched ``eta_rad`` possible here at all (see
        ``crossed_shell_segments_batched``'s own docstring).

Case B (shallow, only reachable when ``depth_m > 0`` and ``eta_rad >=
pi/2``) is identical for all three methods: a single constant-density
segment (v1_0's own approximation -- see
``tpeanuts.medium.earth.geometry``'s module docstring for why there is
no real ``r(x)`` relationship to sample here), sourced from the
tabulated PREM table (not the coarser even-power fit -- there is no
"fewer segments" advantage to chase when it is already exactly one) --
expanded to as many equal sub-segments as case A's own segment count
(``nsteps``, or ``2*n_shells - 1``) so the two blend into one
consistently-shaped ``MediumProfile`` via ``torch.where``, exactly
reproducing the same physics (splitting one constant-``H`` segment into
equal pieces composes to the identical evolution operator).

Module contents:
    EarthMedium
        Frozen dataclass: a nadir angle, a detector depth, a
        propagation method, a segment count (``NUMERICAL``/``ANALYTIC``
        only), and a RuntimeContext, with a ``profile()`` method
        building the corresponding ``MediumProfile`` and an ``evolutor()``
        method building the evolution operator of the whole chord, and a
        ``propagate()`` method applying it to a ``Flux``, a ``transition_matrix()`` method giving the
        state-independent flavour-transition probabilities, and a ``from_flux()`` constructor taking the nadir
        angles from a flux with a direction axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch

from tpeanuts.config import default
from tpeanuts.interfaces.flux import Flux
from tpeanuts.interfaces.medium import MediumProfile
from tpeanuts.medium.common.propagation import check_compatible, evolve_flux
from tpeanuts.medium.earth.density import interpolate_prem_density, load_prem_table, prem_discontinuity_radii
from tpeanuts.medium.earth.even_power import load_even_power_shells, shift_shell_coefficients_to_chord
from tpeanuts.medium.earth.geometry import (
    aligned_chord_segments,
    crossed_shell_segments_batched,
    detector_x_coordinate_m,
    earth_chord_length_m,
    eta_prime_from_eta,
    radius_fraction_along_chord,
)
from tpeanuts.physics.common.terms import PropagationEnvironment
from tpeanuts.physics.evolution import PropagationMethod
from tpeanuts.physics.evolution.dispatch import evolve_path
from tpeanuts.physics.evolution.perturbative.profile import EvenPowerProfile
from tpeanuts.physics.model import PropagationModel
from tpeanuts.util.constant import R_E
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.geometry import detector_radius_fraction
from tpeanuts.util.tensor import TensorLike, as_tensor

_ZERO: Any = 0.0  # default depth; __post_init__ turns it into a tensor on the context


@dataclass(frozen=True)
class EarthMedium:
    """A source-to-detector path along a chord through the Earth, sampled from PREM.

    Attributes:
        eta_rad: Detector nadir angle, in radians, scalar or any batch
            shape; per-element domain depends on ``depth_m`` (see
            ``tpeanuts.medium.earth.geometry.earth_chord_length_m``).
            Freely batchable for every ``method``/``depth_m``
            combination, mixing case-A and case-B entries in the same
            call -- see ``profile()``'s own docstring.
        context: RuntimeContext the returned profile's tensors are built on.
        depth_m: Detector depth below the Earth's surface, in metres.
            Defaults to ``0.0`` (a surface detector, this package's
            original scope, fully backward compatible).
        method: Which ``PropagationMethod`` this profile is built for --
            see this module's own docstring for what each one produces.
            Defaults to ``PropagationMethod.NUMERICAL``.
        nsteps: Number of equal-length chord segments to discretize into,
            for ``NUMERICAL``/``ANALYTIC`` case-A paths only (ignored for
            ``ANALYTIC_PERTURBATIVE``, whose own segment count is fixed
            by the PREM shell geometry instead, and for case B, which is
            always exactly one segment). Defaults to
            ``tpeanuts.config.default.EARTH_NSTEPS``.

    Examples:
        >>> import math
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> from tpeanuts.medium import EarthMedium
        >>> medium = EarthMedium(eta_rad=torch.tensor(0.0, dtype=torch.float64), context=context)   # through the centre
        >>> profile = medium.profile()
        >>> profile.n_segments, round(float(profile.length_m.sum()) / 1e3)   # chord length in km: the Earth's diameter
        (100, 12742)
    """

    eta_rad: torch.Tensor
    context: RuntimeContext
    depth_m: torch.Tensor = _ZERO
    method: PropagationMethod = PropagationMethod.NUMERICAL
    nsteps: int = default.EARTH_NSTEPS

    if TYPE_CHECKING:  # the runtime __init__ is the dataclass one; tensor fields accept any TensorLike

        def __init__(
            self,
            eta_rad: TensorLike,
            context: RuntimeContext,
            depth_m: TensorLike = ...,
            method: PropagationMethod = ...,
            nsteps: int = ...,
        ) -> None: ...

    @classmethod
    def from_flux(cls, flux: Flux, *, depth_m: TensorLike = _ZERO, **options: Any) -> "EarthMedium":
        """The medium that carries ``flux``, which has a direction axis, along each of its nadir angles.

        ``eta_rad`` becomes the column of the flux's directions, so that with the energy as the last axis the evolutor
        is shaped ``(n_direction, n_energy, n_flavour, n_flavour)`` (see ``evolutor``).

        Args:
            flux: A Flux with a ``direction`` axis.
            depth_m: Detector depth below the surface, in metres.
            **options: Any other field of this class (``method``, ``nsteps``).

        Returns:
            An EarthMedium on the flux's context.

        Raises:
            ValueError: If the flux has no direction axis.
        """
        if flux.direction is None:
            raise ValueError(
                "the flux has no direction axis (flux.direction); build the medium with an eta_rad instead."
            )
        return cls(eta_rad=flux.direction.reshape(-1, 1), context=flux.context, depth_m=depth_m, **options)

    def profile(self) -> MediumProfile:
        """Build this trajectory's MediumProfile, shaped for ``self.method`` and this trajectory's own geometric case.

        ``eta_rad`` (and ``depth_m``) may be batched, mixing case-A and
        case-B entries (and, for ``depth_m == 0`` entries mixed in with
        buried ones, always case A) freely in the same call: both
        geometric cases are computed for the *whole* batch, then
        blended per element with ``torch.where`` on a genuine tensor
        mask -- not a Python ``if``/``bool()`` branch, which previously
        forced ``eta_rad``/``depth_m`` to be 0-dim whenever a case-B entry
        was even possible (``depth_m > 0``). Confirmed by direct
        experiment (see [[reference_batching_axis_collision]]'s own
        discipline) that this reproduces the old scalar-only behaviour
        bit-for-bit at every previously-supported input, and matches a
        manual per-element loop at newly-supported batched ones.

        Returns:
            See this module's own docstring for what each ``method``/
            case produces.

        Raises:
            ValueError: If ``eta_rad``/``depth_m`` is outside its valid
                domain (raised by ``earth_chord_length_m``/
                ``detector_radius_fraction``).
        """
        eta_t = as_tensor(self.eta_rad, device=self.context.device, dtype=self.context.dtype)
        r_d = detector_radius_fraction(self.depth_m, context=self.context)
        case_b_mask = torch.broadcast_to(
            (r_d != 1.0) & (eta_t >= math.pi / 2), torch.broadcast_shapes(eta_t.shape, r_d.shape)
        )

        if self.method is PropagationMethod.ANALYTIC_PERTURBATIVE:
            return self._profile_analytic_perturbative_blended(eta_t, r_d, case_b_mask)
        return self._profile_sampled_blended(eta_t, r_d, case_b_mask)

    def _profile_for_energy(self, E_MeV: torch.Tensor) -> "EarthMedium":
        """This medium, with ``eta_rad``/``depth_m`` broadcast against the energy when the method needs it in the
        profile.

        ``ANALYTIC_PERTURBATIVE`` keeps its density in a profile object whose tensors cannot be reshaped after the
        fact, so the energy's batch axes must already be in them: ``eta_rad`` and ``depth_m`` are expanded to the
        broadcast shape of all three (a view; the profile is then built per element). The sampled methods carry the
        density in plain tensors that ``evolutor`` aligns with the energy itself, and are returned unchanged.
        """
        if self.method is not PropagationMethod.ANALYTIC_PERTURBATIVE:
            return self
        eta_rad = as_tensor(self.eta_rad, device=self.context.device, dtype=self.context.dtype)
        depth = as_tensor(self.depth_m, device=self.context.device, dtype=self.context.dtype)
        shape = torch.broadcast_shapes(eta_rad.shape, depth.shape, E_MeV.shape)
        return replace(self, eta_rad=eta_rad.expand(shape), depth_m=depth.expand(shape))

    def evolutor(
        self,
        model: PropagationModel,
        energy_MeV: TensorLike,
        *,
        P: Optional[torch.Tensor] = None,
        P_nc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build the evolution operator of the whole chord for ``model`` at ``energy_MeV``.

        The chord's ``profile()`` is turned into a ``PropagationEnvironment`` and handed to
        ``tpeanuts.physics.evolution.dispatch.evolve_path``, which does all the physics. What is specific to this
        medium, and so lives here, is how the energy batch axis meets the chord's: a non-scalar energy is aligned
        against the profile's trailing segment axis with ``unsqueeze(-1)``, so ``energy_MeV`` and
        ``eta_rad``/``depth_m`` broadcast like ordinary tensors -- equal shapes pair element-wise (one angle per
        event), a scalar broadcasts against anything, and a full energy x angle grid is requested by passing the energy
        as a column, shape ``(n_E, 1)``, giving ``(n_E, *eta_shape, n_flavour, n_flavour)``.

        Args:
            model: Built PropagationModel (mixing, mass spectrum, and any active extensions/dissipators).
            energy_MeV: Neutrino energy in MeV, any shape (a scalar included).
            P: Optional matter-potential direction matrix, forwarded to ``evolve_path`` unchanged.
            P_nc: Optional neutral-current direction matrix, forwarded to ``evolve_path`` unchanged.

        Returns:
            The evolution operator for the full chord, shaped ``(..., n_flavour, n_flavour)`` (or the
            Liouville-space shape, if ``model`` has active dissipators).

        Raises:
            ValueError: Via ``profile()`` (an invalid or insufficiently scalar ``eta_rad``/``depth_m`` for ``method``),
            if
                ``energy_MeV`` cannot be broadcast against the ``eta_rad``/``depth_m`` batch, or via ``evolve_path`` (an
                incompatible ``method`` against active dissipators).
        """
        E_MeV = as_tensor(energy_MeV, device=self.context.device, dtype=self.context.dtype)
        medium_profile = self._profile_for_energy(E_MeV).profile() if E_MeV.ndim > 0 else self.profile()
        if E_MeV.ndim > 0:
            E_MeV = E_MeV.unsqueeze(-1)

        environment = PropagationEnvironment(
            E_MeV=E_MeV,
            n_e_mol_cm3=medium_profile.n_e_mol_cm3,
            n_n_mol_cm3=medium_profile.n_n_mol_cm3,
        )
        try:
            torch.broadcast_shapes(E_MeV.shape, medium_profile.n_e_mol_cm3.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"energy_MeV (shape {tuple(as_tensor(energy_MeV).shape)}) cannot be broadcast against the "
                f"eta_rad/depth_m batch (profile shape {tuple(medium_profile.n_e_mol_cm3.shape[:-1])}). Energy and "
                "eta_rad broadcast like ordinary tensors: equal shapes pair element-wise (e.g. one eta_rad per event); "
                "for a full energy x angle grid pass the energy as a column, shape (n_E, 1)."
            ) from exc

        return evolve_path(
            model,
            environment,
            medium_profile.length_m / R_E,
            method=medium_profile.method,
            perturbative_profile=medium_profile.perturbative_profile,
            evolution_scale_m=R_E,
            P=P,
            P_nc=P_nc,
        )

    def propagate(
        self,
        flux: Flux,
        model: PropagationModel,
        *,
        location: Optional[str] = None,
        P: Optional[torch.Tensor] = None,
        P_nc: Optional[torch.Tensor] = None,
    ) -> Flux:
        """Propagate ``flux`` through this chord and return it in its evolved state.

        The operator is ``evolutor(model, flux.energy_MeV)``, applied to ``flux.state`` (a Liouville evolution if
        ``model`` has dissipators). The energy grid, the intensity and the production distribution are carried over
        unchanged. Energy and angle batch axes follow ``evolutor``'s rules.

        Args:
            flux: The flux to propagate, on this medium's device/dtype and with the same neutrino/antineutrino
                convention as ``model``.
            model: Built PropagationModel.
            location: ``location`` of the returned flux; ``"detector"`` if omitted (the chord ends at the detector).
            P: Optional matter-potential direction matrix, forwarded to ``evolutor``.
            P_nc: Optional neutral-current direction matrix, forwarded to ``evolutor``.

        Returns:
            A new Flux, with ``"earth"`` appended to its ``history``.

        Raises:
            ValueError: If ``flux`` and ``model`` or this medium disagree (device, dtype, antinu), or as
                ``evolutor``.
        """
        check_compatible(flux, model, self.context)
        operator = self.evolutor(model, flux.energy_MeV, P=P, P_nc=P_nc)
        return evolve_flux(flux, model, operator, location=location or "detector", step="earth")

    def transition_matrix(
        self,
        model: PropagationModel,
        energy_MeV: TensorLike,
        *,
        P: Optional[torch.Tensor] = None,
        P_nc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The flavour-transition probabilities of the whole chord, independent of any initial state.

        ``P[..., beta, alpha] = |S[..., beta, alpha]|^2`` for the evolution operator ``S`` of ``evolutor``: the
        probability that a neutrino born as flavour ``alpha`` is detected as ``beta``, for every pair at once. It is
        what reweighting a Monte Carlo event by event needs, each event carrying its own known parent flavour as a
        label rather than a ``DensityMatrix``. Energy and angle batch axes follow ``evolutor``'s rules (equal shapes
        pair element-wise, one angle per event).

        Args:
            model: Built PropagationModel (mixing, mass spectrum and any active extensions); it must have no
                dissipators, the probabilities being those of a unitary evolution.
            energy_MeV: Neutrino energy in MeV, any shape (a scalar included).
            P: Optional matter-potential direction matrix, forwarded to ``evolutor``.
            P_nc: Optional neutral-current direction matrix, forwarded to ``evolutor``.

        Returns:
            Real tensor shaped ``(..., n_flavour, n_flavour)`` (final flavour, initial flavour), each column summing to
            one.

        Raises:
            ValueError: If ``model`` has dissipators, or as ``evolutor``.
        """
        if model.dissipators:
            raise ValueError(
                "transition_matrix is the probability of a unitary evolution; this model has dissipators "
                f"({model.dissipators!r}). Propagate a state with propagate() instead."
            )
        return self.evolutor(model, energy_MeV, P=P, P_nc=P_nc).abs() ** 2

    def _profile_sampled_blended(
        self, eta_t: torch.Tensor, r_d: torch.Tensor, case_b_mask: torch.Tensor
    ) -> MediumProfile:
        """Case A (``_profile_sampled``) blended with case B (expanded to ``nsteps`` equal sub-segments), per element.

        Two fast paths avoid ever expanding case B to ``nsteps``
        segments unless a genuine per-element blend needs it: all
        -case-A (``depth_m == 0`` is always this) skips case B
        entirely, and all-case-B (including every scalar case-B call,
        which is exactly this project's own pre-batching behaviour)
        returns case B's own natural single segment, not an
        artificially expanded one.
        """
        if torch.all(case_b_mask):
            chord_length_m, n_e_mid, n_n_mid = self._case_b_density(eta_t, r_d)
            return MediumProfile(
                length_m=chord_length_m.unsqueeze(-1), n_e_mol_cm3=n_e_mid.unsqueeze(-1),
                n_n_mol_cm3=n_n_mid.unsqueeze(-1), context=self.context, method=self.method,
            )

        profile_a = self._profile_sampled(eta_t, r_d)
        if not torch.any(case_b_mask):
            return profile_a

        chord_length_m, n_e_mid, n_n_mid = self._case_b_density(eta_t, r_d)
        batch_shape = chord_length_m.shape
        length_b = (chord_length_m / self.nsteps).unsqueeze(-1).expand(*batch_shape, self.nsteps)
        n_e_b = n_e_mid.unsqueeze(-1).expand(*batch_shape, self.nsteps)
        n_n_b = n_n_mid.unsqueeze(-1).expand(*batch_shape, self.nsteps)

        assert profile_a.n_n_mol_cm3 is not None  # the sampled PREM profile always carries neutrons
        mask = case_b_mask.unsqueeze(-1)
        return MediumProfile(
            length_m=torch.where(mask, length_b, profile_a.length_m),
            n_e_mol_cm3=torch.where(mask, n_e_b, profile_a.n_e_mol_cm3),
            n_n_mol_cm3=torch.where(mask, n_n_b, profile_a.n_n_mol_cm3),
            context=self.context,
            method=self.method,
        )

    def _case_b_density(
        self, eta_t: torch.Tensor, r_d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Case B's own chord length and constant midpoint density, batched -- shared by both blended dispatchers.

        Matches v1_0's own approximation exactly: density sampled once
        from the tabulated PREM table at the path's own midpoint radius
        ``r_mid = 1 - depth_m / (2*R_E)``, applied uniformly across the
        whole (short) chord -- see this module's own docstring.

        Returns:
            ``(chord_length_m, n_e_mol_cm3, n_n_mol_cm3)``, each shaped
            like the broadcast of ``eta_t``/``r_d``.
        """
        chord_length_m = earth_chord_length_m(eta_t, depth_m=self.depth_m, context=self.context)
        depth_t = as_tensor(self.depth_m, device=self.context.device, dtype=self.context.dtype)
        r_mid = torch.broadcast_to(1.0 - depth_t / (2.0 * R_E), chord_length_m.shape)
        table = load_prem_table(self.context)
        n_e_mol_cm3, n_n_mol_cm3 = interpolate_prem_density(r_mid, table)
        return chord_length_m, n_e_mol_cm3, n_n_mol_cm3

    def _profile_sampled(self, eta_t: torch.Tensor, r_d: torch.Tensor) -> MediumProfile:
        """Case A, ``NUMERICAL``/``ANALYTIC``: ``nsteps`` segments cut at the PREM jumps, each sampled at its midpoint.

        With fewer than ``2 * n_jumps + 1`` segments the chord is tiled by ``nsteps`` equal segments instead.
        """
        eta_prime = eta_prime_from_eta(eta_t, r_d)
        chord_length_m = earth_chord_length_m(eta_t, depth_m=self.depth_m, context=self.context)
        x_d = detector_x_coordinate_m(eta_t, r_d, context=self.context)
        x1 = x_d - chord_length_m  # far-side surface entry point (== -L/2 exactly when depth_m=0)

        jump_radii = prem_discontinuity_radii(self.context)
        if self.nsteps >= 2 * jump_radii.shape[-1] + 1:
            # nsteps segments cut at the density jumps of the table, so that none takes one density for both sides.
            midpoints_x_m, length_m = aligned_chord_segments(
                x1, x_d, eta_prime, jump_radii, self.nsteps, context=self.context
            )
        else:
            # Too few segments to give each shell its own: nsteps equal segments tiling [x1, x_d], sampled at midpoints.
            step_length_m = chord_length_m / self.nsteps
            step_index = torch.arange(self.nsteps, device=self.context.device, dtype=self.context.dtype)
            midpoints_x_m = x1.unsqueeze(-1) + (step_index + 0.5) * step_length_m.unsqueeze(-1)
            length_m = step_length_m.unsqueeze(-1).expand(*step_length_m.shape, self.nsteps).clone()

        radius_fraction = radius_fraction_along_chord(
            midpoints_x_m, eta_prime.unsqueeze(-1), context=self.context
        )
        table = load_prem_table(self.context)
        n_e_mol_cm3, n_n_mol_cm3 = interpolate_prem_density(radius_fraction, table)

        return MediumProfile(
            length_m=length_m,
            n_e_mol_cm3=n_e_mol_cm3,
            n_n_mol_cm3=n_n_mol_cm3,
            context=self.context,
            method=self.method,
        )

    def _analytic_perturbative_profile_object(self, eta_t: torch.Tensor, r_d: torch.Tensor) -> EvenPowerProfile:
        """The raw case-A ``EvenPowerProfile``, shared by the unblended and case-B-blended dispatchers.

        Always goes through ``crossed_shell_segments_batched`` (never
        the scalar-only ``crossed_shell_segments``) -- confirmed by
        direct experiment, element-for-element against the scalar
        function, that this reproduces it exactly (including for a
        genuinely scalar ``eta_rad``, where the padded slots are all
        zero-length and so contribute nothing): one code path serves
        both scalar and batched ``eta_rad`` here, not two.
        """
        eta_prime = eta_prime_from_eta(eta_t, r_d)
        shells = load_even_power_shells(self.context)

        x1_m, x2_m, shell_index, _valid = crossed_shell_segments_batched(
            eta_t, shells.rj, depth_m=self.depth_m, context=self.context
        )
        sin_eta_prime_sq = torch.sin(eta_prime).unsqueeze(-1) ** 2

        a, b, c = shift_shell_coefficients_to_chord(
            shells.alpha[shell_index], shells.beta[shell_index], shells.gamma[shell_index], sin_eta_prime_sq
        )
        a_n, b_n, c_n = shift_shell_coefficients_to_chord(
            shells.alpha_n[shell_index], shells.beta_n[shell_index], shells.gamma_n[shell_index], sin_eta_prime_sq
        )

        # EvenPowerProfile's own x1/x2 (and a/b/c) are dimensionless,
        # already normalized by profile_scale_m -- crossed_shell_segments_batched
        # returns physical metres, so divide by R_E (== profile_scale_m
        # below) first; a/b/c are already dimensionless (built from the
        # dimensionless radius-fraction fit and sin(eta_prime)**2), no
        # conversion needed for them.
        return EvenPowerProfile(
            x1=x1_m / R_E, x2=x2_m / R_E, a=a, b=b, c=c, a_n=a_n, b_n=b_n, c_n=c_n,
            profile_scale_m=R_E, evolution_scale_m=R_E,
            device=self.context.device, dtype=self.context.dtype,
        )

    def _profile_analytic_perturbative(self, eta_t: torch.Tensor, r_d: torch.Tensor) -> MediumProfile:
        """Case A, ``ANALYTIC_PERTURBATIVE``: PREM-shell-aligned segments, each with its own density variation."""
        profile = self._analytic_perturbative_profile_object(eta_t, r_d)
        return MediumProfile(
            length_m=profile.length * R_E,
            n_e_mol_cm3=profile.average,
            n_n_mol_cm3=profile.average_n,
            context=self.context,
            method=PropagationMethod.ANALYTIC_PERTURBATIVE,
            perturbative_profile=profile,
        )

    def _profile_analytic_perturbative_blended(
        self, eta_t: torch.Tensor, r_d: torch.Tensor, case_b_mask: torch.Tensor
    ) -> MediumProfile:
        """Case A (``ANALYTIC_PERTURBATIVE``) blended with case B (expanded to ``2*n_shells-1`` equal sub-segments), per
        element.

        Unlike ``_profile_sampled_blended`` (which only needs to blend
        the *derived* ``length_m``/density quantities), the perturbative
        evolver also consumes ``perturbative_profile``'s own raw
        polynomial coefficients directly -- so the blend has to build a
        genuine new ``EvenPowerProfile`` from per-slot ``torch.where``
        -blended ``x1``/``x2``/coefficients, not just blend two already
        -built profiles' derived outputs. Same two fast paths as
        ``_profile_sampled_blended`` (see its own docstring), sized to
        ``2*n_shells - 1`` instead of ``nsteps``.
        """
        if torch.all(case_b_mask):
            chord_length_m, n_e_mid, n_n_mid = self._case_b_density(eta_t, r_d)
            profile_b_single = EvenPowerProfile.constant(
                x1=torch.zeros_like(chord_length_m).unsqueeze(-1),
                x2=(chord_length_m / R_E).unsqueeze(-1),
                density=n_e_mid.unsqueeze(-1), density_n=n_n_mid.unsqueeze(-1),
                profile_scale_m=R_E, evolution_scale_m=R_E,
                device=self.context.device, dtype=self.context.dtype,
            )
            return MediumProfile(
                length_m=profile_b_single.length * R_E, n_e_mol_cm3=profile_b_single.average,
                n_n_mol_cm3=profile_b_single.average_n, context=self.context,
                method=PropagationMethod.ANALYTIC_PERTURBATIVE, perturbative_profile=profile_b_single,
            )

        profile_a = self._analytic_perturbative_profile_object(eta_t, r_d)
        if not torch.any(case_b_mask):
            return MediumProfile(
                length_m=profile_a.length * R_E, n_e_mol_cm3=profile_a.average, n_n_mol_cm3=profile_a.average_n,
                context=self.context, method=PropagationMethod.ANALYTIC_PERTURBATIVE, perturbative_profile=profile_a,
            )

        max_slots = profile_a.length.shape[-1]
        chord_length_m, n_e_mid, n_n_mid = self._case_b_density(eta_t, r_d)
        batch_shape = chord_length_m.shape
        length_b = (chord_length_m / max_slots).unsqueeze(-1).expand(*batch_shape, max_slots)
        profile_b = EvenPowerProfile.constant(
            x1=torch.zeros(*batch_shape, max_slots, device=self.context.device, dtype=self.context.dtype),
            x2=length_b / R_E,
            density=n_e_mid.unsqueeze(-1).expand(*batch_shape, max_slots),
            density_n=n_n_mid.unsqueeze(-1).expand(*batch_shape, max_slots),
            profile_scale_m=R_E, evolution_scale_m=R_E,
            device=self.context.device, dtype=self.context.dtype,
        )

        assert profile_a.coefficients_n is not None and profile_b.coefficients_n is not None
        mask = case_b_mask.unsqueeze(-1)
        blended = EvenPowerProfile.from_coefficients(
            x1=torch.where(mask, profile_b.x1, profile_a.x1),
            x2=torch.where(mask, profile_b.x2, profile_a.x2),
            coefficients=torch.where(mask.unsqueeze(-1), profile_b.coefficients, profile_a.coefficients),
            coefficients_n=torch.where(mask.unsqueeze(-1), profile_b.coefficients_n, profile_a.coefficients_n),
            profile_scale_m=R_E, evolution_scale_m=R_E,
            device=self.context.device, dtype=self.context.dtype,
        )
        return MediumProfile(
            length_m=blended.length * R_E,
            n_e_mol_cm3=blended.average,
            n_n_mol_cm3=blended.average_n,
            context=self.context,
            method=PropagationMethod.ANALYTIC_PERTURBATIVE,
            perturbative_profile=blended,
        )

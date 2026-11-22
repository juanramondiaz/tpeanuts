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

"""``ReactorSource``: a reactor, emitting pure electron antineutrinos with a Huber-Mueller spectrum.

Mirrors ``tpeanuts.source.solar.solar.SolarSource`` and
``tpeanuts.source.atmosphere.atmosphere.AtmosphericSource``: a frozen dataclass holding the tables the source is
built from, a ``default`` loader for the packaged ones (Daya Bay's cores, ``tpeanuts/data/reactor/dayabay/``), a
parameters dataclass with ``build_reactor_source`` and a ``flux`` method presenting the emission as a ``SourceFlux``.

A nuclear reactor's antineutrinos come from beta decays of fission fragments: overwhelmingly electron antineutrinos,
produced incoherently across a continuous spectrum, conventionally sampled over the ~1.8-9 MeV inverse-beta-decay
window. The spectrum is the fission-fraction-weighted sum of four isotope curves (Huber, Phys. Rev. C84, 024617
(2011) for U235/Pu239/Pu241; Mueller et al., Phys. Rev. C83, 054615 (2011) for U238), optionally corrected for the
non-equilibrium of long-lived fission daughters; scaled by the fission rate (thermal power over the mean energy per
fission) it is the emission rate of the core. All of that is the source's own data and behaviour: what stays in
``tpeanuts.experiments.dayabay`` is what belongs to its six cores and detectors (relative weights, spent-fuel
correction, baselines).

A reactor is a point source with one known flavour composition, so unlike the Sun or the atmosphere its flux has no
``production`` distribution and a single component, ``"nue_bar"``, in the ``SourceFlux`` that ``flux()`` returns.
Its state is the pure electron-flavour state of the plain Standard Model space: any sterile/NSI/magnetic-moment
physics is a property of the propagation model the flux is fed into, not of the source.

Module contents:
    ReactorSource
        Frozen dataclass: an energy grid, a RuntimeContext and (optionally) the isotope curves, fission fractions,
        energy per fission, thermal power and non-equilibrium correction. ``default`` loads the packaged ones;
        ``flux`` builds the ``SourceFlux``; ``huber_mueller_spectrum``/``weighted_spectrum_shape`` are its spectrum
        primitives (static), ``isotope_spectrum``/``spectrum_shape`` their use on the source's own curves;
        ``core_scaling`` (static) is the per-core weight and spent-fuel factor of a multi-reactor site.
    ReactorSourceParameters
        Frozen dataclass of the construction settings for ``ReactorSource.default`` (file overrides).
    build_reactor_source(...)
        Return the given source, or build the default one from ``ReactorSourceParameters``.
    ISOTOPES
        The 4 fissile isotopes every standard reactor antineutrino spectrum decomposes into.
    COMPONENT
        Name of the single component of the emitted ``SourceFlux``: ``"nue_bar"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

import torch

from tpeanuts.interfaces.flux import Flux, SourceFlux
from tpeanuts.physics.common.space import FlavourSpace
from tpeanuts.physics.common.state import DensityMatrix
from tpeanuts.source.reactor.io import (
    load_huber_mueller_spectra,
    load_nonequilibrium_correction,
    load_reactor_parameters,
)
from tpeanuts.util import constant
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.math import interp1d_linear
from tpeanuts.util.tensor import TensorLike, as_tensor

_ELECTRON_INDEX = 0

ISOTOPES: Tuple[str, ...] = ("U235", "U238", "Pu239", "Pu241")
COMPONENT = "nue_bar"

Curves = Mapping[str, Tuple[torch.Tensor, torch.Tensor]]

# Default true-antineutrino grid of ``ReactorSource.default``: from the inverse-beta-decay threshold up to where
# the tabulated Huber-Mueller flux is negligible (the grid Daya Bay's analysis uses).
_DEFAULT_E_MIN_MEV = 1.8
_DEFAULT_E_MAX_MEV = 9.0
_DEFAULT_E_POINTS = 145


@dataclass(frozen=True)
class ReactorSource:
    """A reactor: a source of pure, incoherent electron antineutrinos.

    Attributes:
        energy_MeV: Default energy grid to sample the emitted flux at, in MeV -- typically spanning the ~1.8-9 MeV
            inverse-beta-decay window, though nothing here enforces that range.
        context: RuntimeContext every tensor here is built on.
        curves: ``{isotope: (E_tab_MeV, N_tab_per_fission_per_MeV)}`` for every entry of ``ISOTOPES``: the tabulated
            Huber-Mueller spectra. None for a source without a spectrum (a bare flavour state and energy grid).
        fission_fractions: ``{isotope: fraction}`` for every entry of ``ISOTOPES`` (experiment-specific, e.g. real
            time-averaged values from a data release). Required together with ``curves``.
        energy_per_fission_MeV: ``{isotope: energy}`` released per fission, for every entry of ``ISOTOPES``. Needed,
            with ``fission_fractions``, to turn the thermal power into a fission rate.
        thermal_power_gw: Thermal power of the core in GW, scalar.
        nonequilibrium: ``{isotope: (E_MeV, C)}`` non-equilibrium correction (U235, Pu239, Pu241), applied to the
            matching ``curves`` as ``N (1 + C)``, with ``C`` interpolated onto each isotope's own energies (zero
            outside the correction's range). Isotopes without an entry are left unchanged.
        reactor_provider: Provenance label of the tables (``"dayabay"`` for the packaged ones); no physics.

    Computed on construction (not fields): ``corrected_curves``, the ``curves`` with the non-equilibrium correction
    folded in, which is what the spectrum is built from.

    Raises:
        ValueError: If ``curves`` and ``fission_fractions`` are not given together, if a given mapping does not cover
            exactly ``ISOTOPES``, if a curve is not one-dimensional with matching energy and value arrays, or if
            ``thermal_power_gw`` is negative.

    Examples:
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> from tpeanuts.source import ReactorSource
        >>> source = ReactorSource(energy_MeV=[2.0, 4.0, 6.0], context=context)
        >>> flux = source.flux()["nue_bar"]
        >>> flux.antinu, flux.state.probabilities().tolist()
        (True, [1.0, 0.0, 0.0])
    """

    energy_MeV: torch.Tensor
    context: RuntimeContext
    curves: Optional[Curves] = None
    fission_fractions: Optional[Mapping[str, TensorLike]] = None
    energy_per_fission_MeV: Optional[Mapping[str, float]] = None
    thermal_power_gw: Optional[float] = None
    nonequilibrium: Optional[Curves] = None
    reactor_provider: Optional[str] = None

    def __post_init__(self) -> None:
        context = self.context
        object.__setattr__(self, "energy_MeV", as_tensor(self.energy_MeV, device=context.device, dtype=context.dtype))
        if (self.curves is None) != (self.fission_fractions is None):
            raise ValueError("curves and fission_fractions must be given together.")
        if self.thermal_power_gw is not None and float(self.thermal_power_gw) < 0.0:
            raise ValueError("thermal_power_gw must be non-negative.")
        for name, mapping in (
            ("curves", self.curves), ("fission_fractions", self.fission_fractions),
            ("energy_per_fission_MeV", self.energy_per_fission_MeV),
        ):
            if mapping is not None and set(mapping) != set(ISOTOPES):
                raise ValueError(f"{name} must cover exactly {ISOTOPES}, got {sorted(mapping)}.")

        corrected = None
        if self.curves is not None:
            curves = {iso: self._normalised_curve(iso, curve) for iso, curve in self.curves.items()}
            object.__setattr__(self, "curves", curves)
            corrected = self._fold_nonequilibrium(curves)
        object.__setattr__(self, "_corrected_curves", corrected)

    def _normalised_curve(self, isotope: str, curve: Tuple[torch.Tensor, torch.Tensor]):
        """One isotope's ``(E, N)`` on this source's device/dtype, checked to be a matching pair of 1-D arrays."""
        E_tab = as_tensor(curve[0], device=self.context.device, dtype=self.context.dtype)
        N_tab = as_tensor(curve[1], device=self.context.device, dtype=self.context.dtype)
        if E_tab.ndim != 1 or N_tab.shape != E_tab.shape:
            raise ValueError(f"curves[{isotope!r}] must be two one-dimensional arrays of equal length.")
        return E_tab, N_tab

    def _fold_nonequilibrium(self, curves: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
        """The ``curves`` with ``nonequilibrium`` folded in; the same curves if there is none."""
        if not self.nonequilibrium:
            return curves
        corrected = dict(curves)
        for isotope, (E_corr, C_corr) in self.nonequilibrium.items():
            E_tab, N_tab = corrected[isotope]
            zero = torch.zeros((), device=E_tab.device, dtype=E_tab.dtype)
            E_corr = as_tensor(E_corr, device=self.context.device, dtype=self.context.dtype)
            C_corr = as_tensor(C_corr, device=self.context.device, dtype=self.context.dtype)
            corrected[isotope] = (E_tab, N_tab * (1.0 + interp1d_linear(E_tab, E_corr, C_corr, left=zero, right=zero)))
        return corrected

    @property
    def device(self) -> torch.device:
        """Return the device of this source's tensors."""
        return self.context.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the real dtype of this source's tensors."""
        return self.context.dtype

    @property
    def corrected_curves(self) -> Optional[Curves]:
        """The isotope curves with the non-equilibrium correction folded in (None without ``curves``)."""
        return self._corrected_curves  # type: ignore[attr-defined, no-any-return]

    @property
    def mean_energy_per_fission_MeV(self) -> float:
        """Fission-fraction-weighted mean energy released per fission, MeV.

        Raises:
            ValueError: If the source has no ``fission_fractions`` or ``energy_per_fission_MeV``.
        """
        if self.fission_fractions is None or self.energy_per_fission_MeV is None:
            raise ValueError("the mean energy per fission needs fission_fractions and energy_per_fission_MeV.")
        return sum(float(self.fission_fractions[iso]) * self.energy_per_fission_MeV[iso] for iso in ISOTOPES)

    @property
    def fissions_per_second(self) -> Optional[float]:
        """Fission rate of the core, ``P_th / <e_fission>``; None if the source has no ``thermal_power_gw``.

        Raises:
            ValueError: As ``mean_energy_per_fission_MeV``, if there is a thermal power but no energy per fission.
        """
        if self.thermal_power_gw is None:
            return None
        return (float(self.thermal_power_gw) * 1.0e9) / (self.mean_energy_per_fission_MeV * constant.MEV_TO_JOULE)

    @staticmethod
    def huber_mueller_spectrum(
        E_tab_MeV: torch.Tensor, N_tab_per_fission_per_MeV: torch.Tensor, E_grid_MeV: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate one isotope's tabulated spectrum onto ``E_grid_MeV``.

        Args:
            E_tab_MeV: Tabulated energy grid for this isotope, shape ``(n_tab,)``, strictly increasing.
            N_tab_per_fission_per_MeV: Tabulated spectrum values at ``E_tab_MeV``, antineutrinos/fission/MeV.
            E_grid_MeV: Requested true antineutrino energy grid, shape ``(n_E,)``.

        Returns:
            Real tensor shaped ``(n_E,)``, antineutrinos/fission/MeV, 0 outside the tabulated range.
        """
        zero = torch.zeros((), device=E_grid_MeV.device, dtype=E_grid_MeV.dtype)
        return interp1d_linear(E_grid_MeV, E_tab_MeV, N_tab_per_fission_per_MeV, left=zero, right=zero)

    @staticmethod
    def weighted_spectrum_shape(
        curves: Curves, fission_fractions: Mapping[str, TensorLike], E_grid_MeV: torch.Tensor,
    ) -> torch.Tensor:
        """Fission-fraction-weighted sum over ``ISOTOPES``, antineutrinos/fission/MeV.

        Args:
            curves: ``{isotope: (E_tab_MeV, N_tab_per_fission_per_MeV)}`` for every entry of ``ISOTOPES``.
            fission_fractions: ``{isotope: fraction}`` for every entry of ``ISOTOPES``.
            E_grid_MeV: True antineutrino energy grid, shape ``(n_E,)``.

        Returns:
            Real tensor shaped ``(n_E,)``, antineutrinos/fission/MeV.
        """
        total = torch.zeros_like(E_grid_MeV)
        for isotope in ISOTOPES:
            E_tab, N_tab = curves[isotope]
            total = total + fission_fractions[isotope] * ReactorSource.huber_mueller_spectrum(
                E_tab, N_tab, E_grid_MeV
            )
        return total

    def isotope_spectrum(self, isotope: str, energy_MeV: Optional[TensorLike] = None) -> torch.Tensor:
        """One isotope's spectrum on ``energy_MeV`` (default: the source's grid), antineutrinos/fission/MeV.

        Uses ``corrected_curves``, i.e. with the non-equilibrium correction.

        Raises:
            ValueError: If the source has no ``curves`` or ``isotope`` is not in ``ISOTOPES``.
        """
        if self.corrected_curves is None:
            raise ValueError("this source has no isotope curves.")
        if isotope not in self.corrected_curves:
            raise ValueError(f"isotope must be one of {ISOTOPES}, got {isotope!r}.")
        grid = self._grid(energy_MeV)
        return self.huber_mueller_spectrum(*self.corrected_curves[isotope], grid)

    def spectrum_shape(self, energy_MeV: Optional[TensorLike] = None) -> torch.Tensor:
        """The fission-fraction-weighted spectrum on ``energy_MeV`` (default: the source's grid), per fission per MeV.

        Raises:
            ValueError: If the source has no ``curves`` and ``fission_fractions``.
        """
        if self.corrected_curves is None or self.fission_fractions is None:
            raise ValueError("this source has no isotope curves and fission fractions.")
        return self.weighted_spectrum_shape(self.corrected_curves, self.fission_fractions, self._grid(energy_MeV))

    def _grid(self, energy_MeV: Optional[TensorLike]) -> torch.Tensor:
        """``energy_MeV`` as a tensor on this source's context, or the source's own grid if None."""
        if energy_MeV is None:
            return self.energy_MeV
        return as_tensor(energy_MeV, device=self.context.device, dtype=self.context.dtype)

    @classmethod
    def default(
        cls,
        *,
        energy_MeV: Optional[TensorLike] = None,
        params: Optional["ReactorSourceParameters"] = None,
        context: RuntimeContext,
    ) -> "ReactorSource":
        """Load the packaged reactor tables (Daya Bay's cores): spectra, fission fractions, power, correction.

        Args:
            energy_MeV: Energy grid of the source. None uses 145 points from 1.8 to 9 MeV.
            params: Construction settings (file overrides, whether to apply the non-equilibrium correction). None
                loads the package defaults.
            context: RuntimeContext the source's tensors are built on.

        Returns:
            ReactorSource with ``curves``, ``fission_fractions``, ``energy_per_fission_MeV``,
            ``thermal_power_gw`` and (unless disabled) ``nonequilibrium`` set.

        Raises:
            FileNotFoundError: If a data file is missing.
        """
        settings = params if params is not None else ReactorSourceParameters()
        if energy_MeV is None:
            energy_MeV = torch.linspace(
                _DEFAULT_E_MIN_MEV, _DEFAULT_E_MAX_MEV, _DEFAULT_E_POINTS, device=context.device, dtype=context.dtype,
            )
        fractions, energies, power = load_reactor_parameters(
            settings.parameters_path, power_path=settings.thermal_power_path,
        )
        if settings.thermal_power_gw is not None:
            power = settings.thermal_power_gw
        nonequilibrium = None
        if settings.apply_nonequilibrium:
            nonequilibrium = load_nonequilibrium_correction(settings.nonequilibrium_path, context=context)
        return cls(
            energy_MeV=energy_MeV, context=context,
            curves=load_huber_mueller_spectra(settings.spectra_path, context=context),
            fission_fractions=fractions, energy_per_fission_MeV=energies, thermal_power_gw=power,
            nonequilibrium=nonequilibrium, reactor_provider="dayabay",
        )

    @staticmethod
    def core_scaling(
        energy_MeV: torch.Tensor,
        relative_weight: Optional[TensorLike] = None,
        snf_correction: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Factor ``w * (1 + C_snf(E))`` that turns the plain emission into that of one core of a multi-reactor site.

        ``flux`` applies it when given ``relative_weight``/``snf_correction``; it is public so a site with several
        cores can compute the (expensive) emission once and rescale it per core.

        Args:
            energy_MeV: Energy grid, shape ``(n_E,)``; the result is on its device and dtype.
            relative_weight: The core's relative power/output weight, scalar; None means 1.
            snf_correction: ``(E_tab_MeV, C_snf)`` spent-nuclear-fuel correction, interpolated linearly and zero
                outside its table; None applies none.

        Returns:
            Tensor shaped ``(n_E,)`` (or a scalar if neither argument is given).
        """
        weight = 1.0 if relative_weight is None else relative_weight
        factor = as_tensor(weight, device=energy_MeV.device, dtype=energy_MeV.dtype)
        if snf_correction is not None:
            zero = torch.zeros((), device=energy_MeV.device, dtype=energy_MeV.dtype)
            factor = factor * (1.0 + interp1d_linear(energy_MeV, *snf_correction, left=zero, right=zero))
        return factor

    def flux(
        self,
        energy_MeV: Optional[TensorLike] = None,
        *,
        curves: Optional[Curves] = None,
        fission_fractions: Optional[Mapping[str, TensorLike]] = None,
        fissions_per_second: Optional[TensorLike] = None,
        relative_weight: Optional[TensorLike] = None,
        snf_correction: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> SourceFlux:
        """Present this reactor's emission as a ``SourceFlux`` with the single component ``"nue_bar"``.

        The component is a pure electron-antineutrino ``Flux`` (``antinu=True``, the plain Standard Model space).
        Without any spectrum -- neither on the source nor passed here -- it carries no intensity: the energy grid and
        the flavour state only. With the spectrum it carries ``differential_flux`` in antineutrinos per fission per
        MeV, and in antineutrinos per second per MeV if the fission rate is known too (from the source's thermal
        power, or ``fissions_per_second``). The arguments replace, for this call, the source's own tables; ``curves``
        replace the source's ``corrected_curves``, so pass them already corrected.

        A core of a multi-reactor site differs from the plain emission by a power/output weight and by the
        antineutrinos of its spent fuel: ``relative_weight`` and ``snf_correction`` scale the spectrum by
        ``w * (1 + C_snf(E))`` (``C_snf`` interpolated linearly, zero outside its table), the factor applied after the
        fission rate.

        Args:
            energy_MeV: Energy grid for this call; None uses the source's.
            curves: ``{isotope: (E_tab_MeV, N_tab_per_fission_per_MeV)}`` for every entry of ``ISOTOPES``.
            fission_fractions: ``{isotope: fraction}`` for every entry of ``ISOTOPES``. Required together with
                ``curves`` unless the source has them.
            fissions_per_second: Total fission rate of the core, scalar. Needs a spectrum.
            relative_weight: Relative power/output weight of this core, scalar; None leaves the spectrum unweighted.
                Needs a spectrum.
            snf_correction: ``(E_tab_MeV, C_snf)`` spent-nuclear-fuel correction of this core; None applies none.
                Needs a spectrum.

        Returns:
            SourceFlux with the one component ``"nue_bar"``.

        Raises:
            ValueError: If only one of the curves and the fission fractions is available, if ``fissions_per_second``,
                ``relative_weight`` or ``snf_correction`` is given without a spectrum, or if the fission rate is
                negative.
        """
        spectrum_curves = curves if curves is not None else self.corrected_curves
        spectrum_fractions = fission_fractions if fission_fractions is not None else self.fission_fractions
        if (spectrum_curves is None) != (spectrum_fractions is None):
            raise ValueError("curves and fission_fractions must be given together.")
        if fissions_per_second is not None and spectrum_curves is None:
            raise ValueError("fissions_per_second needs curves and fission_fractions.")
        if (relative_weight is not None or snf_correction is not None) and spectrum_curves is None:
            raise ValueError("relative_weight and snf_correction need curves and fission_fractions.")

        state = DensityMatrix.basis_state(_ELECTRON_INDEX, FlavourSpace(), self.context)
        grid = self._grid(energy_MeV)
        if spectrum_curves is None:
            return SourceFlux({COMPONENT: Flux(energy_MeV=grid, state=state, antinu=True)})

        assert spectrum_fractions is not None  # given together with the curves (checked above)
        spectrum = self.weighted_spectrum_shape(spectrum_curves, spectrum_fractions, grid)
        units = "antinu fission^-1 MeV^-1"
        rate = fissions_per_second if fissions_per_second is not None else self.fissions_per_second
        if rate is not None:
            rate_t = as_tensor(rate, device=self.context.device, dtype=self.context.dtype)
            if torch.any(rate_t < 0):
                raise ValueError("fissions_per_second must be non-negative.")
            spectrum = spectrum * rate_t
            units = "antinu s^-1 MeV^-1"
        if relative_weight is not None or snf_correction is not None:
            spectrum = self.core_scaling(grid, relative_weight, snf_correction) * spectrum
        return SourceFlux({COMPONENT: Flux(
            energy_MeV=grid, state=state, antinu=True, differential_flux=spectrum, flux_units=units,
        )})

    def __str__(self) -> str:
        """Return a compact summary of the reactor source configuration."""
        grid = self.energy_MeV.reshape(-1)
        power = f"{float(self.thermal_power_gw):g} GW" if self.thermal_power_gw is not None else "no power"
        return (
            f"ReactorSource | n_E={grid.numel()} | E=[{float(grid.min()):.3f}, {float(grid.max()):.3f}] MeV | "
            f"spectrum={'yes' if self.curves is not None else 'no'} | "
            f"nonequilibrium={'yes' if self.nonequilibrium else 'no'} | {power} | "
            f"provider={self.reactor_provider} | {self.device} / {self.dtype}"
        )

    __repr__ = __str__


@dataclass(frozen=True)
class ReactorSourceParameters:
    """Construction settings of ``ReactorSource.default`` (the equivalent of ``SolarSourceParameters``).

    Every field is optional; None means the packaged file or value.

    Attributes:
        spectra_path: CSV of the isotope spectra (``isotope, E_MeV, N_per_fission_per_MeV``).
        parameters_path: CSV of fission fractions and energy per fission
            (``isotope, fission_fraction, energy_per_fission_MeV``).
        thermal_power_path: Text file with the thermal power in GW.
        nonequilibrium_path: CSV of the non-equilibrium correction (``isotope, E_MeV, C``).
        thermal_power_gw: Thermal power in GW, replacing the file's value.
        apply_nonequilibrium: Whether to load and fold in the non-equilibrium correction (default True).
    """

    spectra_path: Optional[Union[str, Path]] = None
    parameters_path: Optional[Union[str, Path]] = None
    thermal_power_path: Optional[Union[str, Path]] = None
    nonequilibrium_path: Optional[Union[str, Path]] = None
    thermal_power_gw: Optional[float] = None
    apply_nonequilibrium: bool = True


def build_reactor_source(
    reactor_source: Optional[ReactorSource],
    *,
    params: Optional[ReactorSourceParameters] = None,
    energy_MeV: Optional[TensorLike] = None,
    context: RuntimeContext,
) -> ReactorSource:
    """Return ``reactor_source`` if given, otherwise build the default one (the equivalent of ``build_solar_source``).

    Args:
        reactor_source: An already-built source, returned unchanged (``params`` and ``energy_MeV`` are then ignored).
        params: Construction settings for the default source.
        energy_MeV: Energy grid of the default source.
        context: RuntimeContext for the default source.

    Returns:
        The given source, or ``ReactorSource.default(energy_MeV=…, params=…, context=…)``.
    """
    if reactor_source is not None:
        return reactor_source
    return ReactorSource.default(energy_MeV=energy_MeV, params=params, context=context)

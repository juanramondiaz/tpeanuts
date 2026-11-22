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

"""Generate initial atmospheric-neutrino flux tables ``Phi(E, h, angle)`` with MCEq.

Wraps the external MCEq (Matrix Cascade Equation) + crflux packages to
solve the atmospheric-shower cascade equations and reconstruct the
production-height-differential flux, then saves one CSV per angle in
the canonical schema ``tpeanuts.source.atmosphere.io`` already reads
(``energy_GeV, cos_zenith, particle, flux, altitude_km``) -- so the
files this module writes need no new loader, only
``load_atmospheric_flux`` pointed at them.

Ported from v1_0's ``external/mceq/{core,solver,depth,density,
smoothing,profiles,config}.py`` (~2100 lines across 7 files), with full
option parity deliberately kept, not trimmed to a minimal subset: every
interaction-model/primary-model/density-model choice v1_0 exposed, and
both of v1_0's depth-smoothing methods, are ported, consolidated into a
single ``MCEqConfig`` dataclass (v1_0 split the same information across
``MCEqModelConfig``/``GridConfig``/``SmoothingConfig``) whose field
defaults are sourced from ``tpeanuts.config.default`` rather than
hardcoded on the dataclass itself, matching this project's own
``SOLAR_*``/``ATMOSPHERE_*`` convention. v1_0's own parallel-execution
(``util/parallel.py``) and ``diagnostics.py`` are not ported -- not
requested, and orthogonal to the physics itself (a serial loop over
angles in ``generate_mceq_flux`` is the whole of what changes without
them).

**Why this module *is* wrapped in ``torch.no_grad()`` throughout**,
unlike every differentiable ``tpeanuts.physics``/``medium``/``source``
routine ([[feedback_pipeline_pure_composition]]'s sibling convention,
stated directly in ``tpeanuts.util.context``'s own module docstring:
"no such routine ever wraps itself in ``torch.no_grad()``"): MCEq
itself is a plain NumPy cascade-equation solver with no autograd
concept at all -- every tensor this module produces is built from a
NumPy array MCEq itself returned (``mceq.get_solution(...)``,
``mceq.density_model.get_mass_overburden(...)``), so it is a leaf
tensor with no gradient history to preserve regardless. Matching v1_0's
own blanket ``@torch.no_grad()`` on every function in this pipeline is
therefore a genuine performance choice at a real non-differentiable
external-library boundary, not a violation of the "caller decides"
principle that convention protects -- there is no caller who could ever
want gradients through an MCEq solve.

Two zenith-angle conventions are in play (see
``tpeanuts.medium.atmosphere.geometry``'s own module docstring for the
full geometric derivation): ``alpha`` (surface/MCEq zenith angle -- the
only one MCEq itself understands, ``0 <= alpha_deg < 90``) and
``theta`` (detector zenith angle -- what every other ``eta_rad``-based
propagation function in this project ultimately wants, at any detector
depth). ``generate_mceq_flux``'s ``angle_type`` argument lets a caller
supply either, converting via ``alpha_surface_to_theta_detector``/
``theta_detector_to_alpha_surface`` as needed; the file this module
writes always stores ``cos_zenith = cos(theta_deg)``, the detector
-frame convention ``load_atmospheric_flux`` already documents for every
other atmospheric flux table in this project.

Module contents:
    INTERACTION_MODELS, PRIMARY_MODELS, DENSITY_MODELS
        Full catalogs of MCEq/crflux model names this module accepts,
        mirroring v1_0's own ``external/mceq/config.py`` verbatim.
    MCEqConfig
        Single dataclass aggregating every MCEq run parameter (physics
        models, numerical grids, smoothing), with ``config.default``
        -sourced field defaults and a ``validate()`` method.
    generate_mceq_flux(...)
        Public entry point: solve and save one differential-flux CSV
        per requested angle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

from tpeanuts.config import default
from tpeanuts.source.atmosphere.io import angle_output_path, write_angle_flux_csv
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.geometry import alpha_surface_to_theta_detector, theta_detector_to_alpha_surface
from tpeanuts.util.math import searchsorted_contiguous
from tpeanuts.util.tensor import TensorLike, as_tensor

try:
    import crflux.models as _crflux_models
except ImportError:
    _crflux_models = None

try:
    from MCEq.core import MCEqRun as _MCEqRun
except ImportError:
    _MCEqRun = None


# ============================================================
# Available MCEq / crflux models (mirrors v1_0's own catalog verbatim)
# ============================================================

INTERACTION_MODELS = [
    "DPMJETIII193",
    "EPOSLHC",
    "EPOSLHCR",
    "QGSJETII04",
    "QGSJETIII",
    "SIBYLL21",
    "SIBYLL23D",
    "SIBYLL23E",
    "SIBYLL23ESTARBAR",
    "SIBYLL23ESTARMIXED",
    "SIBYLL23ESTARRHO",
    "SIBYLL23ESTARSTRANGE",
]

if _crflux_models is not None:
    PRIMARY_MODELS: Dict[str, Tuple[object, Optional[str]]] = {
        "HillasGaisser H3a": (_crflux_models.HillasGaisser2012, "H3a"),
        "HillasGaisser H4a": (_crflux_models.HillasGaisser2012, "H4a"),
        "GST 3-gen": (_crflux_models.GaisserStanevTilav, "3-gen"),
        "GST 4-gen": (_crflux_models.GaisserStanevTilav, "4-gen"),
        "PolyGonato": (_crflux_models.PolyGonato, None),
        "GlobalSplineFitBeta": (_crflux_models.GlobalSplineFitBeta, None),
        "GaisserHonda": (_crflux_models.GaisserHonda, None),
    }
else:
    PRIMARY_MODELS = {}

DENSITY_MODELS: Dict[str, Tuple[str, Tuple[Optional[str], Optional[str]]]] = {
    "CORSIKA": ("CORSIKA", ("USStd", None)),
    "NASA": ("MSIS00", ("SouthPole", "January")),
    "ICECUBE": ("MSIS00_IC", ("SouthPole", "January")),
    "ISOTHERMAL": ("Isothermal", ("", "")),
}

_SMOOTHING_METHODS = {None, "none", "spline", "gaussian"}

MCEQ_IMPORT_ERROR = (
    "MCEq and/or crflux could not be imported. Install the optional MCEq "
    "dependencies (e.g. `pip install MCEq crflux`) in this Python "
    "environment before calling tpeanuts.source.atmosphere.mceq.generate_mceq_flux()."
)


def _ensure_mceq_available() -> None:
    """Raise an actionable ``ImportError`` if MCEq/crflux are not installed."""
    if _MCEqRun is None or _crflux_models is None:
        raise ImportError(MCEQ_IMPORT_ERROR)


def _scalar_float(x: TensorLike) -> float:
    return float(as_tensor(x, dtype=torch.float64).reshape(-1)[0].item())


# ============================================================
# MCEqConfig: every MCEq run parameter, one dataclass
# ============================================================


@dataclass
class MCEqConfig:
    """Every parameter needed to run one MCEq cascade-equation solve and profile reconstruction.

    Consolidates what v1_0 split across ``MCEqModelConfig`` (physics
    models), ``GridConfig`` (numerical grids -- minus v1_0's own
    ``theta_grid_deg``, which is ``generate_mceq_flux``'s own
    ``angles_deg`` argument here, not a config field) and
    ``SmoothingConfig`` (depth-axis smoothing) into one object, per the
    "no simplification, full option parity, one config" instruction
    this module was built to. Every field default is read from
    ``tpeanuts.config.default`` at class-definition time, not
    hardcoded here.

    Attributes:
        interaction_model: Hadronic interaction model name; must be one
            of ``INTERACTION_MODELS``.
        primary_model: Primary cosmic-ray flux model name (a key of
            ``PRIMARY_MODELS``) or a raw ``(model_class, model_tag)``
            tuple already in the form MCEq expects.
        density_model: Atmosphere density/overburden model name; must
            be one of ``DENSITY_MODELS``.
        info: If True, MCEq's internal logger is left at INFO
            verbosity; otherwise restricted to ERROR.
        X_grid_gcm2: Strictly increasing 1-D slant-depth grid, in
            g/cm^2, MCEq's cascade equations are solved over.
        h_grid_km: Strictly increasing 1-D altitude grid, in km, the
            reconstructed production-height profile is reported on.
        X_obs_gcm2: Observation slant depth, in g/cm^2, at which the
            energy-differential flux is extracted; must lie within
            ``X_grid_gcm2``'s own range.
        smoothing_method: Depth-axis smoothing method applied to the
            solved flux before it is differentiated; one of ``None``,
            ``"none"``, ``"spline"`` (a log-domain moving average,
            despite the name -- see ``_smooth_flux_log_moving_average``)
            or ``"gaussian"``.
        smoothing: Smoothing strength for the ``"spline"`` method
            (larger -> wider moving-average window).
        gaussian_sigma: Standard deviation, in depth-grid-point units,
            of the ``"gaussian"`` method's kernel.
        positive_only: If True, the depth derivative dPhi/dX is clamped
            to non-negative values.

    Raises:
        ValueError: Via ``validate()``, if any model name is
            unrecognised, if ``X_grid_gcm2``/``h_grid_km`` are not
            strictly increasing 1-D tensors, if ``X_obs_gcm2`` falls
            outside ``X_grid_gcm2``'s range, or if ``smoothing_method``
            is not one of the allowed values.
    """

    interaction_model: str = default.ATMOSPHERE_MCEQ_INTERACTION_MODEL
    primary_model: Union[str, tuple] = default.ATMOSPHERE_MCEQ_PRIMARY_MODEL
    density_model: str = default.ATMOSPHERE_MCEQ_DENSITY_MODEL
    info: bool = default.ATMOSPHERE_MCEQ_INFO
    X_grid_gcm2: torch.Tensor = field(
        default_factory=lambda: torch.linspace(
            default.ATMOSPHERE_MCEQ_X_GRID_MIN_GCM2,
            default.ATMOSPHERE_MCEQ_X_GRID_MAX_GCM2,
            default.ATMOSPHERE_MCEQ_X_GRID_N,
            dtype=torch.float64,
        )
    )
    h_grid_km: torch.Tensor = field(
        default_factory=lambda: torch.linspace(
            default.ATMOSPHERE_MCEQ_H_GRID_MIN_KM,
            default.ATMOSPHERE_MCEQ_H_GRID_MAX_KM,
            default.ATMOSPHERE_MCEQ_H_GRID_N,
            dtype=torch.float64,
        )
    )
    X_obs_gcm2: float = default.ATMOSPHERE_MCEQ_X_OBS_GCM2
    smoothing_method: Optional[str] = default.ATMOSPHERE_MCEQ_SMOOTHING_METHOD
    smoothing: float = default.ATMOSPHERE_MCEQ_SMOOTHING
    gaussian_sigma: float = default.ATMOSPHERE_MCEQ_GAUSSIAN_SIGMA
    positive_only: bool = default.ATMOSPHERE_MCEQ_POSITIVE_ONLY

    def __post_init__(self) -> None:
        self.X_grid_gcm2 = as_tensor(self.X_grid_gcm2, dtype=torch.float64).reshape(-1)
        self.h_grid_km = as_tensor(self.h_grid_km, dtype=torch.float64).reshape(-1)

    def validate(self) -> None:
        """Check every field against its own valid domain.

        Raises:
            ValueError: See this dataclass's own docstring.
        """
        if self.interaction_model not in INTERACTION_MODELS:
            raise ValueError(
                f"Unknown interaction_model={self.interaction_model!r}. Available: {INTERACTION_MODELS}"
            )
        if isinstance(self.primary_model, str) and self.primary_model not in PRIMARY_MODELS:
            raise ValueError(
                f"Unknown primary_model={self.primary_model!r}. Available: {list(PRIMARY_MODELS.keys())}"
            )
        if self.density_model not in DENSITY_MODELS:
            raise ValueError(
                f"Unknown density_model={self.density_model!r}. Available: {list(DENSITY_MODELS.keys())}"
            )
        if self.X_grid_gcm2.ndim != 1:
            raise ValueError("X_grid_gcm2 must be one-dimensional.")
        if self.h_grid_km.ndim != 1:
            raise ValueError("h_grid_km must be one-dimensional.")
        if torch.any(torch.diff(self.X_grid_gcm2) <= 0.0):
            raise ValueError("X_grid_gcm2 must be strictly increasing.")
        if torch.any(torch.diff(self.h_grid_km) <= 0.0):
            raise ValueError("h_grid_km must be strictly increasing.")
        if not (float(self.X_grid_gcm2.min()) <= self.X_obs_gcm2 <= float(self.X_grid_gcm2.max())):
            raise ValueError(
                f"X_obs_gcm2={self.X_obs_gcm2} is outside the X grid range "
                f"[{float(self.X_grid_gcm2.min())}, {float(self.X_grid_gcm2.max())}]."
            )
        if self.smoothing_method not in _SMOOTHING_METHODS:
            raise ValueError(f"Unknown smoothing_method={self.smoothing_method!r}. Allowed: {_SMOOTHING_METHODS}")
        if self.smoothing < 0.0:
            raise ValueError("smoothing must be non-negative.")
        if self.gaussian_sigma < 0.0:
            raise ValueError("gaussian_sigma must be non-negative.")


# ============================================================
# MCEq initialization (the only place an MCEqRun is constructed)
# ============================================================


def _init_mceq_run(alpha_deg: float, config: MCEqConfig):
    """Build a fully configured ``MCEqRun`` instance for a surface zenith angle.

    The single point in this module that directly constructs an
    external ``MCEq.core.MCEqRun`` object, mirroring v1_0's own
    ``external/mceq/core.py::init_mceq``.

    Args:
        alpha_deg: Surface/MCEq zenith angle in degrees, ``0 <=
            alpha_deg < 90``.
        config: Already-validated ``MCEqConfig`` selecting the physics
            models.

    Returns:
        An initialized ``MCEqRun`` with its density model attached,
        ready to be solved.

    Raises:
        ImportError: If MCEq/crflux are not installed.
        ValueError: If ``alpha_deg`` is outside ``[0, 90)``.
    """
    if alpha_deg < 0.0 or alpha_deg >= 90.0:
        raise ValueError(f"alpha_deg must satisfy 0 <= alpha_deg < 90. Received alpha_deg={alpha_deg}.")

    _ensure_mceq_available()
    logging.getLogger().setLevel(logging.INFO if config.info else logging.ERROR)

    primary_model = (
        PRIMARY_MODELS[config.primary_model] if isinstance(config.primary_model, str) else config.primary_model
    )
    mceq = _MCEqRun(interaction_model=config.interaction_model, primary_model=primary_model, theta_deg=alpha_deg)
    mceq.set_density_model(DENSITY_MODELS[config.density_model])
    return mceq


# ============================================================
# Solve Phi(E, X, alpha) (direct MCEq calls only)
# ============================================================


@torch.no_grad()
def _solve_flux_vs_depth_grid(
    mceq, particle: str, X_grid_gcm2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve MCEq's cascade equations once over ``X_grid_gcm2`` and tabulate Phi(E, X, alpha).

    Direct driver of ``mceq.solve(int_grid=...)``/``mceq.get_solution(...)``
    -- mirrors v1_0's own ``external/mceq/solver.py``.

    Returns:
        Tuple ``(E_grid_GeV, flux_XE)``: the MCEq energy grid (CPU
        float64, shape ``(n_E,)``) and the solved flux, shape
        ``(n_X, n_E)``, clamped to non-negative values.
    """
    X_numpy = X_grid_gcm2.detach().cpu().to(torch.float64).numpy()
    mceq.solve(int_grid=X_numpy)
    flux_list = [
        torch.as_tensor(mceq.get_solution(particle, grid_idx=i), dtype=torch.float64, device="cpu").clamp_min(0.0)
        for i in range(len(X_grid_gcm2))
    ]
    E_grid = torch.as_tensor(mceq.e_grid, dtype=torch.float64, device="cpu")
    return E_grid, torch.stack(flux_list, dim=0)


# ============================================================
# Atmosphere depth utilities (generic tensor algebra + MCEq's own overburden)
# ============================================================


def _cos_deg(alpha_deg: float) -> float:
    return float(torch.cos(torch.deg2rad(torch.tensor(alpha_deg, dtype=torch.float64))))


@torch.no_grad()
def _atmosphere_slant_depth(X_vertical_gcm2: torch.Tensor, alpha_deg: float) -> torch.Tensor:
    """``X_slant = X_vertical / cos(alpha)`` (plane-parallel projection); mirrors v1_0's own ``atmosphere_slant_depth``.
    """
    cos_alpha = _cos_deg(alpha_deg)
    if cos_alpha <= 10.0 * torch.finfo(torch.float64).eps:
        raise ValueError("alpha_deg must satisfy cos(alpha) > 0.")
    return X_vertical_gcm2 / cos_alpha


@torch.no_grad()
def _atmosphere_vertical_depth_mceq(mceq, h_grid_km: torch.Tensor) -> torch.Tensor:
    """``X_vertical(h)`` from MCEq's own density model (``get_mass_overburden``, altitude in cm)."""
    h_cm = h_grid_km.detach().cpu().to(torch.float64).numpy() * 1.0e5
    X_vals = [float(mceq.density_model.get_mass_overburden(float(h))) for h in h_cm]
    return torch.tensor(X_vals, dtype=torch.float64)


@torch.no_grad()
def _interpolate_flux_at_Xobs(
    X_grid_gcm2: torch.Tensor, flux_XE: torch.Tensor, X_obs_gcm2: float, *, eps: float = 1.0e-300
) -> torch.Tensor:
    """Log-linear interpolation of ``flux_XE`` at a single ``X_obs_gcm2``; mirrors v1_0's own
    ``interpolate_flux_at_Xobs``.

    Returns:
        Tensor of shape ``(n_E,)``.
    """
    n_X = X_grid_gcm2.numel()
    idx = torch.clamp(
        searchsorted_contiguous(X_grid_gcm2, torch.tensor(X_obs_gcm2, dtype=X_grid_gcm2.dtype)), 1, n_X - 1
    )
    x0, x1 = X_grid_gcm2[idx - 1], X_grid_gcm2[idx]
    w = (X_obs_gcm2 - x0) / (x1 - x0)
    y0, y1 = flux_XE[idx - 1], flux_XE[idx]
    log_y0 = torch.log(torch.clamp(y0, min=eps))
    log_y1 = torch.log(torch.clamp(y1, min=eps))
    return torch.exp((1.0 - w) * log_y0 + w * log_y1)


@torch.no_grad()
def _interpolate_source_X_to_h(
    X_grid_gcm2: torch.Tensor, source_XE: torch.Tensor, X_of_h_gcm2: torch.Tensor
) -> torch.Tensor:
    """Interpolate a depth-tabulated source term onto ``X(h)``; mirrors v1_0's own ``interpolate_source_X_to_h``.

    Points in ``X_of_h_gcm2`` outside ``X_grid_gcm2``'s own range are
    set to zero (production above/below the tabulated depth grid).

    Returns:
        Tensor of shape ``(n_E, n_h)``.
    """
    n_X = X_grid_gcm2.numel()
    n_h = X_of_h_gcm2.numel()
    valid = (X_of_h_gcm2 >= X_grid_gcm2[0]) & (X_of_h_gcm2 <= X_grid_gcm2[-1])
    if not torch.any(valid):
        return torch.zeros(source_XE.shape[-1], n_h, dtype=source_XE.dtype)

    X_clamped = torch.clamp(X_of_h_gcm2, min=X_grid_gcm2[0], max=X_grid_gcm2[-1])
    idx = torch.clamp(searchsorted_contiguous(X_grid_gcm2, X_clamped), 1, n_X - 1)
    x0, x1 = X_grid_gcm2[idx - 1], X_grid_gcm2[idx]
    w = (X_clamped - x0) / (x1 - x0)

    y0 = source_XE[idx - 1, :]
    y1 = source_XE[idx, :]
    yq = (1.0 - w).unsqueeze(-1) * y0 + w.unsqueeze(-1) * y1
    yq = torch.where(valid.unsqueeze(-1), yq, torch.zeros_like(yq))
    return yq.movedim(-2, -1)


# ============================================================
# Smoothing + depth derivative (both v1_0 methods, full parity)
# ============================================================


def _gaussian_kernel1d(sigma: float, *, truncate: float = 4.0) -> torch.Tensor:
    radius = int(truncate * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=torch.float64)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _conv1d_along_depth(x_XE: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    n_X, n_E = x_XE.shape
    pad = kernel.numel() // 2
    x = x_XE.permute(1, 0).unsqueeze(0)
    x_pad = torch.nn.functional.pad(x, pad=(pad, pad), mode="replicate")
    weight = kernel.reshape(1, 1, -1).repeat(n_E, 1, 1)
    y = torch.nn.functional.conv1d(x_pad, weight, groups=n_E)
    return y.squeeze(0).permute(1, 0)


@torch.no_grad()
def _smooth_flux_gaussian(flux_XE: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian-kernel smoothing along the depth axis; mirrors v1_0's own ``smooth_flux_gaussian``."""
    if sigma <= 0.0:
        return flux_XE.clone()
    return _conv1d_along_depth(flux_XE, _gaussian_kernel1d(sigma))


@torch.no_grad()
def _smooth_flux_log_moving_average(flux_XE: torch.Tensor, window: int, *, eps: float = 1.0e-300) -> torch.Tensor:
    """Log-domain moving-average smoothing (v1_0's ``"spline"`` method); mirrors v1_0's own
    ``smooth_flux_log_moving_average``.
    """
    if window <= 1:
        return flux_XE.clone()
    if window % 2 == 0:
        window += 1
    log_flux = torch.log(torch.clamp(flux_XE, min=eps))
    kernel = torch.ones(window, dtype=torch.float64) / float(window)
    return torch.exp(_conv1d_along_depth(log_flux, kernel))


@torch.no_grad()
def _smooth_flux_in_depth(X_grid_gcm2: torch.Tensor, flux_XE: torch.Tensor, config: MCEqConfig) -> torch.Tensor:
    """Dispatch to the configured smoothing method; mirrors v1_0's own ``smooth_flux_in_depth``."""
    method = config.smoothing_method
    if method is None or method == "none":
        return flux_XE.clone()
    if method == "gaussian":
        return _smooth_flux_gaussian(flux_XE, config.gaussian_sigma)
    if method == "spline":
        n_X = X_grid_gcm2.numel()
        window = max(3, int(round(config.smoothing * n_X * 100.0)))
        if window % 2 == 0:
            window += 1
        return _smooth_flux_log_moving_average(flux_XE, window)
    raise ValueError(f"Unknown smoothing_method={method!r}. Allowed: {_SMOOTHING_METHODS}")


@torch.no_grad()
def _compute_depth_derivative(X_grid_gcm2: torch.Tensor, flux_XE: torch.Tensor, *, positive_only: bool) -> torch.Tensor:
    """Finite-difference ``dPhi/dX``; mirrors v1_0's own ``compute_depth_derivative``."""
    dPhi_dX = torch.gradient(flux_XE, spacing=(X_grid_gcm2,), dim=0)[0]
    if positive_only:
        dPhi_dX = torch.clamp(dPhi_dX, min=0.0)
    return dPhi_dX


@torch.no_grad()
def _smooth_and_differentiate_flux(
    X_grid_gcm2: torch.Tensor, flux_XE: torch.Tensor, config: MCEqConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Smooth then differentiate in one step; mirrors v1_0's own ``smooth_and_differentiate_flux``."""
    flux_smooth = _smooth_flux_in_depth(X_grid_gcm2, flux_XE, config)
    dPhi_dX = _compute_depth_derivative(X_grid_gcm2, flux_smooth, positive_only=config.positive_only)
    return flux_smooth, dPhi_dX


# ============================================================
# Production-height profile reconstruction (the 7-step pipeline)
# ============================================================


@torch.no_grad()
def _normalize_height_profiles(h_grid_km: torch.Tensor, source_Eh: torch.Tensor, *, eps: float = 0.0) -> torch.Tensor:
    """Normalize to ``f(h|E,alpha)`` integrating to 1 over h; mirrors v1_0's own ``normalize_height_profiles``."""
    norm_E = torch.trapezoid(source_Eh, x=h_grid_km, dim=-1)
    return torch.where(
        (norm_E > eps).unsqueeze(-1),
        source_Eh / norm_E.clamp_min(eps if eps > 0.0 else torch.finfo(source_Eh.dtype).tiny).unsqueeze(-1),
        torch.zeros_like(source_Eh),
    )


@torch.no_grad()
def _height_differential_flux(
    mceq, particle: str, alpha_deg: float, config: MCEqConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """End-to-end reconstruction of ``phi_Eh(E, h)`` for one particle species and surface zenith angle.

    The seven numbered steps mirror v1_0's own
    ``external/mceq/profiles.py::production_profiles_all_energies_from_flux_gradient``
    exactly.

    Returns:
        Tuple ``(E_grid_GeV, h_grid_km, phi_Eh)``, ``phi_Eh`` shaped
        ``(n_E, n_h)``, in ``(cm^2 s sr GeV km)^-1``.
    """
    X_grid = config.X_grid_gcm2.to(dtype=torch.float64)
    h_grid = config.h_grid_km.to(dtype=torch.float64)

    # 1. Solve Phi(E, X, alpha).
    E_grid, flux_XE = _solve_flux_vs_depth_grid(mceq, particle, X_grid)

    # 2. Phi(E, X_obs, alpha).
    phi_E_obs = _interpolate_flux_at_Xobs(X_grid, flux_XE, config.X_obs_gcm2)

    # 3-4. Smooth and differentiate.
    _, dPhi_dX = _smooth_and_differentiate_flux(X_grid, flux_XE, config)

    # 5. X(h, alpha) and dX/dh, from MCEq's own density model.
    X_vertical_of_h = _atmosphere_vertical_depth_mceq(mceq, h_grid)
    X_of_h = _atmosphere_slant_depth(X_vertical_of_h, alpha_deg)
    dXdh = torch.gradient(X_of_h, spacing=(h_grid,))[0]  # mirrors v1_0's own compute_dXdh

    # 6. Q(E, X) -> Q(E, h) via the |dX/dh| Jacobian.
    source_interp_Eh = _interpolate_source_X_to_h(X_grid, dPhi_dX, X_of_h)
    source_Eh = source_interp_Eh * torch.abs(dXdh).unsqueeze(0)

    # 7. Normalize f(h|E,alpha) and build phi_Eh = phi_E_obs * f_Eh.
    f_Eh = _normalize_height_profiles(h_grid, source_Eh)
    phi_Eh = phi_E_obs.unsqueeze(-1) * f_Eh

    return E_grid, h_grid, phi_Eh


# ============================================================
# Public entry point
# ============================================================


@torch.no_grad()
def generate_mceq_flux(
    angles_deg: TensorLike,
    angle_type: str,
    *,
    depth_m: TensorLike = 0.0,
    particles: Optional[Dict[str, str]] = None,
    config: Optional[MCEqConfig] = None,
    output_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> List[Path]:
    """Generate and save one differential-flux CSV per requested angle.

    For each angle in ``angles_deg``: resolves both the surface
    (``alpha``) and detector (``theta``) zenith angles via
    ``tpeanuts.medium.atmosphere.geometry`` (using ``depth_m`` only for
    that conversion -- MCEq itself has no notion of detector depth),
    solves and reconstructs ``phi_Eh(E, h)`` for every requested
    particle with a fresh ``MCEqRun`` (one per angle, since MCEq is
    initialized at a fixed zenith angle), and writes a single CSV
    holding every particle's rows, in the same schema
    ``tpeanuts.source.atmosphere.io.load_atmospheric_flux`` already
    reads.

    Args:
        angles_deg: Scalar or 1-D iterable of angles in degrees, in the
            convention named by ``angle_type``.
        angle_type: ``"alpha"`` (surface/MCEq zenith angle, ``0 <=
            alpha_deg < 90``) or ``"theta"`` (detector zenith angle, any
            value convertible to a valid ``alpha_deg`` at the given
            ``depth_m`` -- see
            ``tpeanuts.medium.atmosphere.geometry.theta_detector_to_alpha_surface``).
        depth_m: Detector depth below the surface, in metres, used only
            for the ``alpha``/``theta`` conversion. Defaults to
            ``0.0`` (a surface detector, where ``alpha == theta``).
        particles: Mapping from output flavour label to MCEq particle
            name (e.g. ``{"numu": "numu"}``). Defaults to
            ``tpeanuts.config.default.ATMOSPHERE_MCEQ_PARTICLES``.
        config: Optional ``MCEqConfig``; defaults to ``MCEqConfig()``.
        output_dir: Directory the per-angle CSVs are written into.
            Defaults to ``tpeanuts/data/atmosphere/<ATMOSPHERE_MCEQ_FLUX_DIRNAME>/``.
        overwrite: If False (default) and an angle's output file already
            exists, that angle is skipped without re-running MCEq (an
            MCEq solve is expensive -- seconds to minutes per angle).

    Returns:
        List of paths written (or already present, when
        ``overwrite=False`` skipped them), one per requested angle, in
        the same order as ``angles_deg``.

    Raises:
        ImportError: If MCEq/crflux are not installed.
        ValueError: If ``angle_type`` is not ``"alpha"``/``"theta"``, if
            ``config`` fails ``validate()``, or if any angle converts to
            an ``alpha_deg`` outside MCEq's own ``[0, 90)`` domain.
    """
    if angle_type not in ("alpha", "theta"):
        raise ValueError(f"angle_type must be 'alpha' or 'theta', got {angle_type!r}.")

    _ensure_mceq_available()

    if config is None:
        config = MCEqConfig()
    config.validate()

    if particles is None:
        particles = default.ATMOSPHERE_MCEQ_PARTICLES

    if output_dir is None:
        output_dir = default.ATMOSPHERE_DATA_DIR / default.ATMOSPHERE_MCEQ_FLUX_DIRNAME
    output_dir = Path(output_dir)

    angles_t = as_tensor(angles_deg, dtype=torch.float64).reshape(-1)
    depth_context = RuntimeContext(device=torch.device("cpu"), dtype=torch.float64)

    written_paths: List[Path] = []
    for angle_value in angles_t.tolist():
        if angle_type == "alpha":
            alpha_deg = float(angle_value)
            theta_deg = _scalar_float(
                alpha_surface_to_theta_detector(angle_value, depth_m, context=depth_context)
            )
        else:
            theta_deg = float(angle_value)
            alpha_deg = _scalar_float(
                theta_detector_to_alpha_surface(angle_value, depth_m, context=depth_context)
            )

        if not (0.0 <= alpha_deg < 90.0):
            raise ValueError(
                f"angle {angle_value!r} ({angle_type}) converts to a surface zenith angle "
                f"alpha_deg={alpha_deg}, outside MCEq's own valid domain [0, 90) -- MCEq "
                "cannot simulate a horizontal/upward-going shower at the surface."
            )

        cos_zenith = _cos_deg(theta_deg)
        out_path = angle_output_path(output_dir, alpha_deg)

        if out_path.exists() and not overwrite:
            written_paths.append(out_path)
            continue

        mceq = _init_mceq_run(alpha_deg, config)

        per_particle_phi_Eh: Dict[str, torch.Tensor] = {}
        E_grid = None
        h_grid = None
        for flavour_name, mceq_particle_name in particles.items():
            E_grid, h_grid, phi_Eh = _height_differential_flux(mceq, mceq_particle_name, alpha_deg, config)
            per_particle_phi_Eh[flavour_name] = phi_Eh

        assert E_grid is not None and h_grid is not None  # at least one particle was computed
        write_angle_flux_csv(
            out_path,
            energy_GeV=E_grid,
            altitude_km=h_grid,
            per_particle_flux=per_particle_phi_Eh,
            cos_zenith=cos_zenith,
        )
        written_paths.append(out_path)

    return written_paths

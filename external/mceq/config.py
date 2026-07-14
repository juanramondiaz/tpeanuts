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
Configuration objects for the mceq Atmosphere-flux pipeline.

This module is pure tpeanuts-native configuration: it only declares
constants and dataclasses, and never constructs or calls an external
MCEqRun instance. The constants here name (but do not import or
construct) MCEq physics models:

    - interaction_model: the hadronic interaction model used by MCEq to
      generate secondary particles in each air-shower interaction
      (e.g. SIBYLL, QGSJET, EPOS, DPMJET families). It controls the
      particle-production yields of the cascade-equation source terms.
    - primary_model: the primary cosmic-ray flux model from the external
      crflux package, giving the all-particle/per-nucleus flux injected
      at the top of the atmosphere as a function of energy. This is the
      boundary condition for MCEq's cascade-equation solve.
    - density_model: the Atmosphere density/overburden model used by
      MCEq to convert altitude h (km) into atmospheric slant depth X
      (g/cm^2), i.e. the column density of air traversed along the
      shower axis. Different models correspond to different
      site/season atmosphere profiles (US Standard Atmosphere via
      CORSIKA, NRLMSISE-00 at the South Pole for NASA/ICECUBE, or a
      simple isothermal profile).

This module also defines:

    - grid configuration (zenith angle, atmospheric depth, altitude
      grids, and the observation depth X_obs at which the flux is
      reported)
    - smoothing/profile-reconstruction configuration
    - output configuration
    - global run configuration aggregating all of the above plus the
      neutrino-flavour-to-MCEq-particle-name mapping

No mceq solver is initialized here.

Module functions:
    default_config:
        Build and validate a RunConfig populated with default values.
    make_config:
        Build and validate a RunConfig from individual keyword
        arguments, constructing the nested grid/model/smoothing/output/
        parallel sub-configs.
"""



from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any, TYPE_CHECKING
import torch

from tpeanuts.util.parallel import ParallelConfig
from tpeanuts.util.type import as_tensor

if TYPE_CHECKING:
    from tpeanuts.medium.atmosphere.io import OutputConfig


def _default_output_config():
    from tpeanuts.medium.atmosphere.io import OutputConfig

    return OutputConfig()


def _make_output_config(**kwargs):
    from tpeanuts.medium.atmosphere.io import OutputConfig

    return OutputConfig(**kwargs)
# ============================================================
# Optional CRflux import
# ============================================================

try:
    import crflux.models as pm
except ImportError:
    pm = None


# ============================================================
# Available mceq / CRflux models
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

DEFAULT_INTERACTION_MODEL = "SIBYLL23D"


if pm is not None:
    PRIMARY_MODELS = {
        "HillasGaisser H3a": (pm.HillasGaisser2012, "H3a"),
        "HillasGaisser H4a": (pm.HillasGaisser2012, "H4a"),
        "GST 3-gen": (pm.GaisserStanevTilav, "3-gen"),
        "GST 4-gen": (pm.GaisserStanevTilav, "4-gen"),
        "PolyGonato": (pm.PolyGonato, None),
        "GlobalSplineFitBeta": (pm.GlobalSplineFitBeta, None),
        "GaisserHonda": (pm.GaisserHonda,None)
    }

    DEFAULT_PRIMARY_MODEL = "HillasGaisser H3a"

else:
    PRIMARY_MODELS = {}
    DEFAULT_PRIMARY_MODEL = None


DENSITY_MODELS = {
    "CORSIKA": ("CORSIKA", ("USStd", None)),
    "NASA": ("MSIS00", ("SouthPole", "January")),
    "ICECUBE": ("MSIS00_IC", ("SouthPole", "January")),
    "ISOTHERMAL": ("Isothermal", ("", "")),
}

DEFAULT_DENSITY_MODEL = "CORSIKA"


# ============================================================
# Default flavour mapping
# ============================================================

DEFAULT_FLAVOURS = {
    "numu": "numu",
    "antinumu": "antinumu",
    "nue": "nue",
    "antinue": "antinue",
}


# ============================================================
# Config dataclasses
# ============================================================

@dataclass
class MCEqModelConfig:
    """
    Configuration of the physical mceq models used to build an MCEqRun.

    This dataclass only names the physics models; it does not import or
    construct MCEq/crflux objects itself (see
    tpeanuts.external.mceq.core.init_mceq, which resolves these names
    into actual MCEq/crflux objects and builds the MCEqRun instance).

    Attributes:
        interaction_model: Name of the hadronic interaction model used
            by MCEq for particle production in each cascade-equation
            interaction (e.g. "SIBYLL23D", "QGSJETII04", "EPOSLHC").
            Must be one of INTERACTION_MODELS.
        primary_model: Name of the primary cosmic-ray flux model (from
            the optional crflux package) that sets the boundary
            condition injected at the top of the atmosphere, or a raw
            (model_class, model_tag) tuple understood directly by MCEq.
            Must be a key of PRIMARY_MODELS when given as a string.
        density_model: Name of the Atmosphere density/overburden model
            used by MCEq to relate altitude h (km) to atmospheric slant
            depth X (g/cm^2). Must be one of DENSITY_MODELS.
        info: If True, MCEq's internal logger is set to INFO level
            (verbose); otherwise it is restricted to ERROR level.

    Raises:
        ValueError: Via validate(), if any of the above names is not
            recognised.
    """

    interaction_model: str = DEFAULT_INTERACTION_MODEL
    primary_model: Optional[Union[str, tuple]] = DEFAULT_PRIMARY_MODEL
    density_model: str = DEFAULT_DENSITY_MODEL
    info: bool = False

    def validate(self) -> None:
        """
        Check that interaction_model, primary_model and density_model
        are all recognised names.

        Raises:
            ValueError: If interaction_model is not in
                INTERACTION_MODELS, if primary_model is a string not in
                PRIMARY_MODELS, or if density_model is not in
                DENSITY_MODELS.
        """
        if self.interaction_model not in INTERACTION_MODELS:
            raise ValueError(
                f"Unknown interaction_model='{self.interaction_model}'. "
                f"Available models: {INTERACTION_MODELS}"
            )

        if isinstance(self.primary_model, str):
            if self.primary_model not in PRIMARY_MODELS:
                raise ValueError(
                    f"Unknown primary_model='{self.primary_model}'. "
                    f"Available primary models: {list(PRIMARY_MODELS.keys())}"
                )

        if self.density_model not in DENSITY_MODELS:
            raise ValueError(
                f"Unknown density_model='{self.density_model}'. "
                f"Available density models: {list(DENSITY_MODELS.keys())}"
            )


@dataclass
class GridConfig:
    """
    Numerical grids used in the height-flux reconstruction pipeline.

    These grids are tpeanuts-native (plain torch tensors); MCEq is
    queried at the points they define, but the grids themselves are not
    MCEq objects.

    theta_grid_deg:
        Grid of zenith angles in degrees, 0 <= alpha < 90. alpha=0 is a
        vertically downward-going trajectory (shortest atmospheric
        path); larger alpha corresponds to more inclined trajectories
        with a longer atmospheric slant depth for a given altitude.

    X_grid_gcm2:
        Atmosphere depth grid in g/cm^2, i.e. the slant column density
        of air (integral of mass density along the line of sight) at
        which the cascade-equation flux Phi(E, X, alpha) is evaluated
        by MCEq. Must be strictly increasing.

    h_grid_km:
        Altitude-above-sea-level grid in km at which the reconstructed
        height-differential production profile f(h | E, alpha) and flux
        Phi(E, h, alpha) are reported. Must be strictly increasing.

    X_obs_gcm2:
        Observation slant depth in g/cm^2 (e.g. the depth of a detector
        or the ground) at which the energy-differential flux Phi(E,
        X_obs, alpha) is extracted/interpolated from the MCEq solution.
        Must lie within the range covered by X_grid_gcm2.
    """

    theta_grid_deg: torch.ndarray = field(
        default_factory=lambda: torch.linspace(0.0, 85.0, 18)
    )

    X_grid_gcm2: torch.ndarray = field(
        default_factory=lambda: torch.linspace(1.0, 1030.0, 220)
    )

    h_grid_km: torch.ndarray = field(
        default_factory=lambda: torch.linspace(0.0, 80.0, 300)
    )

    X_obs_gcm2: float = 1030.0

    def __post_init__(self) -> None:
        self.theta_grid_deg = as_tensor(self.theta_grid_deg, dtype=torch.float64)
        self.X_grid_gcm2 = as_tensor(self.X_grid_gcm2, dtype=torch.float64)
        self.h_grid_km = as_tensor(self.h_grid_km, dtype=torch.float64)

    def validate(self) -> None:
        """
        Coerce grids to 1-D float64 tensors and check physical bounds.

        Raises:
            ValueError: If any grid is not one-dimensional, if
                theta_grid_deg has values outside [0, 90) degrees, if
                X_grid_gcm2 or h_grid_km is not strictly increasing, or
                if X_obs_gcm2 falls outside the range of X_grid_gcm2.
        """
        self.theta_grid_deg = as_tensor(self.theta_grid_deg, dtype=torch.float64)
        self.X_grid_gcm2 = as_tensor(self.X_grid_gcm2, dtype=torch.float64)
        self.h_grid_km = as_tensor(self.h_grid_km, dtype=torch.float64)

        if self.theta_grid_deg.ndim != 1:
            raise ValueError("theta_grid_deg must be one-dimensional.")

        if self.X_grid_gcm2.ndim != 1:
            raise ValueError("X_grid_gcm2 must be one-dimensional.")

        if self.h_grid_km.ndim != 1:
            raise ValueError("h_grid_km must be one-dimensional.")

        if torch.any(self.theta_grid_deg < 0.0) or torch.any(self.theta_grid_deg >= 90.0):
            raise ValueError("All theta values must satisfy 0 <= alpha < 90 degrees.")

        if torch.any(torch.diff(self.X_grid_gcm2) <= 0.0):
            raise ValueError("X_grid_gcm2 must be strictly increasing.")

        if torch.any(torch.diff(self.h_grid_km) <= 0.0):
            raise ValueError("h_grid_km must be strictly increasing.")

        if not (self.X_grid_gcm2.min() <= self.X_obs_gcm2 <= self.X_grid_gcm2.max()):
            raise ValueError(
                f"X_obs_gcm2={self.X_obs_gcm2} is outside the X grid range "
                f"[{self.X_grid_gcm2.min()}, {self.X_grid_gcm2.max()}]."
            )


@dataclass
class SmoothingConfig:
    """
    Configuration for flux smoothing and depth-derivative extraction.

    This config is purely tpeanuts-native (consumed by
    tpeanuts.external.mceq.smoothing) and does not call MCEq. It
    controls how the raw MCEq flux Phi(E, X, alpha), which can be noisy
    along the depth axis X, is smoothed before the numerical derivative
    dPhi/dX is taken. dPhi/dX is the depth-differential particle
    production source term used downstream to reconstruct the
    height-dependent production profile f(h | E, alpha).

    Attributes:
        method: Smoothing method applied along the depth axis before
            differentiation. One of {None, "none", "spline",
            "gaussian"}. "spline" is implemented as a log-domain moving
            average (see smooth_flux_log_moving_average), "gaussian"
            applies a Gaussian kernel convolution, and None/"none"
            disables smoothing.
        smoothing: Smoothing strength used by the "spline"/
            "log_moving_average" method to determine the moving-average
            window size; must be non-negative.
        gaussian_sigma: Standard deviation, in grid-point units, of the
            Gaussian kernel used by the "gaussian" method; must be
            non-negative.
        positive_only: If True, the computed dPhi/dX derivative is
            clamped to be non-negative (unphysical negative production
            rates from numerical noise are zeroed out).
    """

    method: Optional[str] = "spline"
    smoothing: float = 1.0e-4
    gaussian_sigma: float = 2.0
    positive_only: bool = True

    def validate(self) -> None:
        """
        Check that method, smoothing and gaussian_sigma have valid
        values.

        Raises:
            ValueError: If method is not one of the allowed smoothing
                methods, or if smoothing/gaussian_sigma is negative.
        """
        allowed = {None, "none", "spline", "gaussian"}

        if self.method not in allowed:
            raise ValueError(
                f"Unknown smoothing method='{self.method}'. "
                f"Allowed: {allowed}"
            )

        if self.smoothing < 0.0:
            raise ValueError("smoothing must be non-negative.")

        if self.gaussian_sigma < 0.0:
            raise ValueError("gaussian_sigma must be non-negative.")


@dataclass
class RunConfig:
    """
    Full configuration object for the mceq height-flux pipeline.

    Aggregates the physics-model selection (model), the numerical grids
    (grid), the flux smoothing/derivative settings (smoothing), the
    output file settings (output), the parallel-execution settings
    (parallel), and the mapping from tpeanuts neutrino-flavour names to
    the corresponding MCEq particle names (flavours, e.g.
    {"numu": "numu", "antinumu": "antinumu", ...}). This object is
    consumed by the MCEq generator/orchestration utilities to drive
    Atmosphere height-flux dataset generation.

    Attributes:
        model: MCEqModelConfig selecting the interaction model, primary
            cosmic-ray model and Atmosphere density model.
        grid: GridConfig with the theta, X and h grids and the
            observation depth X_obs_gcm2.
        smoothing: SmoothingConfig controlling flux smoothing and
            depth-derivative extraction.
        output: OutputConfig controlling where/how results are saved.
        parallel: ParallelConfig controlling whether/how the build loop
            is parallelized across angles.
        flavours: Mapping from tpeanuts flavour name to the MCEq
            particle name used in calls such as mceq.get_solution(...).
    """

    model: MCEqModelConfig = field(default_factory=MCEqModelConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    output: "OutputConfig" = field(default_factory=_default_output_config)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)

    flavours: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_FLAVOURS)
    )

    def validate(self) -> None:
        """
        Validate every nested sub-config and the flavours mapping.

        Raises:
            ValueError: If any nested sub-config fails its own
                validate(), if flavours is not a non-empty dict, or if
                any flavour/particle name in it is not a string.
        """
        self.model.validate()
        self.grid.validate()
        self.smoothing.validate()
        self.output.validate()
        self.parallel.validate()

        if not isinstance(self.flavours, dict) or len(self.flavours) == 0:
            raise ValueError("flavours must be a non-empty dictionary.")

        for flavour_name, mceq_particle_name in self.flavours.items():
            if not isinstance(flavour_name, str):
                raise ValueError("All flavour names must be strings.")

            if not isinstance(mceq_particle_name, str):
                raise ValueError("All mceq particle names must be strings.")


# ============================================================
# Helper constructors
# ============================================================

def default_config() -> RunConfig:
    """
    Build a validated RunConfig populated entirely with default values.

    Returns:
        A RunConfig using DEFAULT_INTERACTION_MODEL,
        DEFAULT_PRIMARY_MODEL, DEFAULT_DENSITY_MODEL, the default
        theta/X/h grids and X_obs_gcm2, default smoothing settings,
        default output settings, parallel execution disabled, and
        DEFAULT_FLAVOURS as the flavour mapping.
    """
    config = RunConfig()
    config.validate()
    return config


def make_config(
    *,
    theta_grid_deg: Optional[torch.ndarray] = None,
    X_grid_gcm2: Optional[torch.ndarray] = None,
    h_grid_km: Optional[torch.ndarray] = None,
    X_obs_gcm2: float = 1030.0,
    interaction_model: str = DEFAULT_INTERACTION_MODEL,
    primary_model: Optional[Union[str, tuple]] = DEFAULT_PRIMARY_MODEL,
    density_model: str = DEFAULT_DENSITY_MODEL,
    smoothing_method: Optional[str] = "spline",
    smoothing: float = 1.0e-4,
    positive_only: bool = True,
    output_dir: str = "mceq_height_flux_outputs",
    filename: str = "phi_E_theta_h_from_mceq_profiles.npz",
    dtype: Any = torch.float64,
    compressed: bool = True,
    parallel: bool = False,
    n_jobs: int = 4,
    flavours: Optional[Dict[str, str]] = None,
) -> RunConfig:
    """
    Build a validated RunConfig from individual keyword arguments.

    Convenience constructor that assembles the nested GridConfig,
    MCEqModelConfig, SmoothingConfig, OutputConfig and ParallelConfig
    from flat keyword arguments, rather than requiring the caller to
    build each sub-config object explicitly.

    Args:
        theta_grid_deg: Zenith-angle grid in degrees (0 <= alpha < 90);
            defaults to 18 points linearly spaced in [0, 85].
        X_grid_gcm2: Atmospheric slant-depth grid in g/cm^2 at which
            MCEq's cascade-equation flux is solved; defaults to 220
            points linearly spaced in [1, 1030].
        h_grid_km: Altitude grid in km at which the reconstructed
            height-differential profile/flux is reported; defaults to
            300 points linearly spaced in [0, 80].
        X_obs_gcm2: Observation slant depth in g/cm^2 at which the
            energy-differential flux Phi(E, X_obs, alpha) is extracted.
        interaction_model: Name of the MCEq hadronic interaction model
            (see INTERACTION_MODELS).
        primary_model: Name of the crflux primary cosmic-ray flux model
            (see PRIMARY_MODELS), or a raw (class, tag) tuple.
        density_model: Name of the MCEq Atmosphere density model (see
            DENSITY_MODELS).
        smoothing_method: Flux-smoothing method passed to
            SmoothingConfig.method.
        smoothing: Smoothing strength passed to
            SmoothingConfig.smoothing.
        positive_only: Whether the depth derivative dPhi/dX is clamped
            to non-negative values; passed to
            SmoothingConfig.positive_only.
        output_dir: Directory in which generated flux files are saved.
        filename: Output file name template for generated flux files.
        dtype: Torch dtype used when persisting tensors to disk.
        compressed: Whether the saved output archive is compressed.
        parallel: Whether the build loop over zenith angles is
            parallelized across worker processes/threads.
        n_jobs: Number of parallel workers used when parallel is True.
        flavours: Mapping from tpeanuts neutrino-flavour name to MCEq
            particle name; defaults to a copy of DEFAULT_FLAVOURS.

    Returns:
        A validated RunConfig built from the above arguments.
    """
    grid = GridConfig(
        theta_grid_deg=(
            torch.linspace(0.0, 85.0, 18)
            if theta_grid_deg is None
            else theta_grid_deg
        ),
        X_grid_gcm2=(
            torch.linspace(1.0, 1030.0, 220)
            if X_grid_gcm2 is None
            else X_grid_gcm2
        ),
        h_grid_km=(
            torch.linspace(0.0, 80.0, 300)
            if h_grid_km is None
            else h_grid_km
        ),
        X_obs_gcm2=X_obs_gcm2,
    )

    config = RunConfig(
        model=MCEqModelConfig(
            interaction_model=interaction_model,
            primary_model=primary_model,
            density_model=density_model,
        ),
        grid=grid,
        smoothing=SmoothingConfig(
            method=smoothing_method,
            smoothing=smoothing,
            positive_only=positive_only,
        ),
        output=_make_output_config(
            output_dir=output_dir,
            filename=filename,
            dtype=dtype,
            compressed=compressed,
        ),
        parallel=ParallelConfig(
            parallel=parallel,
            n_jobs=n_jobs,
        ),
        flavours=dict(DEFAULT_FLAVOURS) if flavours is None else flavours,
    )

    config.validate()
    return config

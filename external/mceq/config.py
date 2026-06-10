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
Configuration objects for the mceq atmospheric-flux pipeline.

This module defines:

    - available interaction models
    - available primary cosmic-ray models
    - available atmospheric density models
    - grid configuration
    - smoothing/profile configuration
    - output configuration
    - global run configuration

No mceq solver is initialized here.
"""



from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any
import torch

from tpeanuts.io.io_atmosphere import OutputConfig
from tpeanuts.util.parallel import ParallelConfig
from tpeanuts.util.type import _as_tensor
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
    Configuration of the physical mceq models.
    """

    interaction_model: str = DEFAULT_INTERACTION_MODEL
    primary_model: Optional[Union[str, tuple]] = DEFAULT_PRIMARY_MODEL
    density_model: str = DEFAULT_DENSITY_MODEL
    info: bool = False

    def validate(self) -> None:
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
    Numerical grids used in the reconstruction.

    X_grid_gcm2:
        Atmospheric depth grid in g/cm^2.

    h_grid_km:
        Altitude grid in km.

    theta_grid_deg:
        Zenith-angle grid in degrees.

    X_obs_gcm2:
        Observation depth in g/cm^2.
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
        self.theta_grid_deg = _as_tensor(self.theta_grid_deg, dtype=torch.float64)
        self.X_grid_gcm2 = _as_tensor(self.X_grid_gcm2, dtype=torch.float64)
        self.h_grid_km = _as_tensor(self.h_grid_km, dtype=torch.float64)

    def validate(self) -> None:
        self.theta_grid_deg = _as_tensor(self.theta_grid_deg, dtype=torch.float64)
        self.X_grid_gcm2 = _as_tensor(self.X_grid_gcm2, dtype=torch.float64)
        self.h_grid_km = _as_tensor(self.h_grid_km, dtype=torch.float64)

        if self.theta_grid_deg.ndim != 1:
            raise ValueError("theta_grid_deg must be one-dimensional.")

        if self.X_grid_gcm2.ndim != 1:
            raise ValueError("X_grid_gcm2 must be one-dimensional.")

        if self.h_grid_km.ndim != 1:
            raise ValueError("h_grid_km must be one-dimensional.")

        if torch.any(self.theta_grid_deg < 0.0) or torch.any(self.theta_grid_deg >= 90.0):
            raise ValueError("All theta values must satisfy 0 <= theta < 90 degrees.")

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
    Configuration for flux smoothing and derivative extraction.
    """

    method: Optional[str] = "spline"
    smoothing: float = 1.0e-4
    gaussian_sigma: float = 2.0
    positive_only: bool = True

    def validate(self) -> None:
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
    """

    model: MCEqModelConfig = field(default_factory=MCEqModelConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)

    flavours: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_FLAVOURS)
    )

    def validate(self) -> None:
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
        output=OutputConfig(
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

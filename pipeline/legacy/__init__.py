"""Explicit NumPy/Numba legacy backend used only for validation."""

from tpeanuts.pipeline.legacy.solar import (
    LegacySolarSurfaceResult,
    propagate_legacy_solar_to_surface,
)
from tpeanuts.pipeline.legacy.earth import (
    LegacyEarthDetectorResult,
    propagate_legacy_earth_to_detector,
    propagate_legacy_earth_to_detector_integrated,
)
from tpeanuts.pipeline.legacy.solar_earth import (
    LegacySolarEarthDetectorResult,
    propagate_legacy_solar_to_earth_detector,
)

__all__ = [
    "LegacySolarSurfaceResult",
    "LegacyEarthDetectorResult",
    "LegacySolarEarthDetectorResult",
    "propagate_legacy_solar_to_surface",
    "propagate_legacy_earth_to_detector",
    "propagate_legacy_earth_to_detector_integrated",
    "propagate_legacy_solar_to_earth_detector",
]

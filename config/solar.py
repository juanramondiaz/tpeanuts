"""Composed solar source and medium construction settings."""

from dataclasses import dataclass, field

from tpeanuts.medium.solar.profile import SolarMediumParameters
from tpeanuts.source.solar import SolarSourceParameters


@dataclass(frozen=True)
class SolarParameters:
    """Keep solar-medium and solar-source configuration separate but grouped."""

    medium: SolarMediumParameters = field(default_factory=SolarMediumParameters)
    source: SolarSourceParameters = field(default_factory=SolarSourceParameters)

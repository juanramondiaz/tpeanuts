"""High-level configuration utilities with cycle-safe lazy exports."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tpeanuts.config.propagation import PropagationConfig
    from tpeanuts.config.solar import SolarParameters

__all__ = [
    "PropagationConfig",
    "SolarParameters",
]


def __getattr__(name: str):
    if name == "PropagationConfig":
        from tpeanuts.config.propagation import PropagationConfig

        return PropagationConfig
    if name == "SolarParameters":
        from tpeanuts.config.solar import SolarParameters

        return SolarParameters
    raise AttributeError(name)

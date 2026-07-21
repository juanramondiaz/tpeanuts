"""High-level configuration utilities with cycle-safe lazy exports."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tpeanuts.config.propagation import PropagationConfig

__all__ = [
    "PropagationConfig",
]


def __getattr__(name: str):
    if name == "PropagationConfig":
        from tpeanuts.config.propagation import PropagationConfig

        return PropagationConfig
    raise AttributeError(name)

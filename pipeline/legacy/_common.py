"""Shared validation guards for the legacy Peanuts backend."""

import torch

from tpeanuts.config.propagation import PropagationConfig


def validate_legacy_configuration(config: PropagationConfig) -> None:
    """Reject model extensions unsupported by the original Peanuts code."""
    if int(config.oscillation.pmns.n_flavours) != 3:
        raise ValueError("Legacy Peanuts supports only the three-flavour model.")
    if config.oscillation.nsi is not None:
        raise ValueError("Legacy Peanuts does not support NSI configurations.")
    antinu = config.oscillation.antinu
    if torch.is_tensor(antinu) and antinu.numel() != 1:
        raise ValueError("Legacy Peanuts requires a scalar antinu selection.")

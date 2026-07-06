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

"""Input/output helpers for even-power perturbative profiles.

This module reads the legacy Earth-density CSV format whose columns encode an
even-power radial model,

    n_e(r) = alpha + beta*r**2 + gamma*r**4 + delta1*r**6 + ...

and converts it into shell radii and model coefficients. Medium-level objects
such as ``EarthProfile`` are built outside this module.

Module functions:
    parse_density_table(...): Convert a pandas table into ``(rj, coefficients)``.
    load_earth_density_from_csv(...): Read an even-power density CSV file into
        ``(rj, coefficients)``.
"""

from __future__ import annotations

from typing import Union

import os

import pandas as pd
import torch

import tpeanuts.util.default as default
from tpeanuts.util.torch_util import default_device
from tpeanuts.util.type import as_tensor


def parse_density_table(
    table: pd.DataFrame,
    *,
    tabulated_density: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build shell radii and coefficients from an even-power density table.

    Args:
        table: DataFrame with columns ``rj``, ``alpha`` and optionally
            ``beta``, ``gamma`` and ``delta*``.
        tabulated_density: If True, keep only the constant ``alpha`` term.
        device: Torch device for the resulting profile tensors.
        dtype: Real dtype for the resulting profile tensors.

    Returns:
        Tuple ``(rj, coefficients)``. ``rj`` contains dimensionless shell radii
        and ``coefficients`` has shape ``(n_layers, n_coefficients)`` ordered as
        ``alpha, beta, gamma, delta1, ...``.
    """
    rj = as_tensor(
        table["rj"],
        device=device,
        dtype=dtype,
    )

    alpha = as_tensor(
        table["alpha"],
        device=device,
        dtype=dtype,
    )

    if tabulated_density:

        beta = torch.zeros_like(
            rj,
            device=device,
            dtype=dtype,
        )

        gamma = torch.zeros_like(
            rj,
            device=device,
            dtype=dtype,
        )

        deltas = torch.zeros(
            (0, rj.numel()),
            device=device,
            dtype=dtype,
        )

    else:

        beta = as_tensor(
            table.get("beta", torch.zeros_like(rj)),
            device=device,
            dtype=dtype,
        )

        gamma = as_tensor(
            table.get("gamma", torch.zeros_like(rj)),
            device=device,
            dtype=dtype,
        )

        delta_names = [
            column_name
            for column_name in table.columns
            if column_name.startswith("delta")
        ]

        if len(delta_names) == 0:

            deltas = torch.zeros(
                (0, rj.numel()),
                device=device,
                dtype=dtype,
            )

        else:

            deltas = torch.stack(
                [
                    as_tensor(
                        table[name],
                        device=device,
                        dtype=dtype,
                    )
                    for name in delta_names
                ],
                dim=0,
            )

    coefficients = torch.cat(
        [
            alpha[..., None],
            beta[..., None],
            gamma[..., None],
            deltas.transpose(0, 1),
        ],
        dim=-1,
    )

    return rj, coefficients


def load_earth_density_from_csv(
    density_file: str,
    *,
    tabulated_density: bool = default.earth_tabulated_density,
    device: Union[str, torch.device, None] = default.device,
    dtype: torch.dtype = default.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read an even-power Earth-density CSV file.

    Args:
        density_file: Path to the CSV file.
        tabulated_density: If True, keep only the constant ``alpha`` term.
        device: Torch device for the resulting profile tensors.
        dtype: Real dtype for the resulting profile tensors.

    Returns:
        Tuple ``(rj, coefficients)`` suitable for ``EvenPowerProfileLayered``.
    """
    device = default_device(device)

    if not os.path.isfile(density_file):
        raise FileNotFoundError(
            f"density file not found: {density_file}"
        )

    table = pd.read_csv(density_file)

    return parse_density_table(
        table,
        tabulated_density=tabulated_density,
        device=device,
        dtype=dtype,
    )

"""Input tables used by vacuum propagation between astronomical bodies."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

import tpeanuts.config.default as default
from tpeanuts.util.io import package_dir
from tpeanuts.util.type import as_tensor


def load_sun_earth_distance(
    path: str | Path | None = None,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, list[str] | torch.Tensor]:
    """Load the date-resolved Sun-Earth distance table."""
    if path is None:
        path = package_dir() / default.solar_data_dir / default.solar_sun_earth_distance_filename
    table = pd.read_csv(path)
    required = {"date", "distance_km", "distance_AU"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            "Sun-Earth distance table is missing required columns: "
            + ", ".join(sorted(missing))
        )
    return {
        "date": [str(value) for value in table["date"].to_list()],
        "distance_km": as_tensor(table["distance_km"].to_numpy(), device=device, dtype=dtype),
        "distance_AU": as_tensor(table["distance_AU"].to_numpy(), device=device, dtype=dtype),
    }

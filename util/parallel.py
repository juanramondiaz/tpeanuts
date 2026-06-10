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
Shared parallel execution utilities.

This module contains the common parallel-execution configuration and helper
functions used by TPeanuts pipelines. It intentionally contains no physics,
domain-specific MCEq logic, or file-format handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional

from tqdm import tqdm


try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None
    delayed = None


@dataclass
class ParallelConfig:
    """
    Parallel execution configuration.
    """

    parallel: bool = False
    n_jobs: int = 4
    backend: str = "loky"

    def validate(self) -> None:
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be 0.")

        if self.backend not in {"loky", "threading", "multiprocessing"}:
            raise ValueError(
                "backend must be one of: 'loky', 'threading', 'multiprocessing'."
            )


def require_joblib() -> None:
    if Parallel is None or delayed is None:
        raise ImportError(
            "joblib is required for parallel execution. "
            "Install it with: pip install joblib"
        )


def parallel_map(
    func: Callable,
    items: Iterable,
    config: Optional[ParallelConfig] = None,
    *,
    show_progress: bool = True,
    desc: str = "Parallel jobs",
    **fixed_kwargs,
) -> List[Any]:
    require_joblib()

    if config is None:
        config = ParallelConfig()

    config.validate()

    items = list(items)
    iterator = items

    if show_progress:
        iterator = tqdm(items, desc=desc)

    return Parallel(
        n_jobs=config.n_jobs,
        backend=config.backend,
    )(
        delayed(func)(
            item,
            **fixed_kwargs,
        )
        for item in iterator
    )


def serial_map(
    func: Callable,
    items: Iterable,
    *,
    show_progress: bool = True,
    desc: str = "Jobs",
    **fixed_kwargs,
) -> List[Any]:
    items = list(items)
    iterator = items

    if show_progress:
        iterator = tqdm(items, desc=desc)

    results = []
    for item in iterator:
        results.append(func(item, **fixed_kwargs))

    return results


def run_map(
    func: Callable,
    items: Iterable,
    config: Optional[ParallelConfig] = None,
    *,
    show_progress: bool = True,
    desc: str = "Jobs",
    **fixed_kwargs,
) -> List[Any]:
    if config is None:
        config = ParallelConfig()

    config.validate()

    if config.parallel:
        return parallel_map(
            func=func,
            items=items,
            config=config,
            show_progress=show_progress,
            desc=desc,
            **fixed_kwargs,
        )

    return serial_map(
        func=func,
        items=items,
        show_progress=show_progress,
        desc=desc,
        **fixed_kwargs,
    )


def run_task_dicts(
    func: Callable,
    tasks: Iterable[dict],
    config: Optional[ParallelConfig] = None,
    *,
    show_progress: bool = True,
    desc: str = "Tasks",
) -> List[Any]:
    def _worker(task: dict):
        return func(**task)

    return run_map(
        func=_worker,
        items=list(tasks),
        config=config,
        show_progress=show_progress,
        desc=desc,
    )

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

Module functions:
    require_joblib(...): Raise ImportError if joblib is not installed.
    parallel_map(...): Apply a function to items in parallel using joblib.
    serial_map(...): Apply a function to items sequentially, with an
        optional progress bar.
    run_map(...): Dispatch to parallel_map or serial_map based on config.
    run_task_dicts(...): Run a function over a sequence of keyword-argument
        dicts via run_map.

Module classes:
    ParallelConfig: Dataclass describing parallel-execution settings
        (enabled flag, worker count, joblib backend).
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
    Parallel execution configuration shared by TPeanuts batch helpers.

    Attributes:
        parallel: Whether run_map should dispatch to joblib-based parallel
            execution (True) or a plain sequential loop (False).
        n_jobs: Number of joblib worker processes/threads to use. Joblib
            conventions apply: -1 uses all available CPUs.
        backend: Joblib backend name; one of "loky" (separate processes),
            "threading", or "multiprocessing".
    """

    parallel: bool = False
    n_jobs: int = 4
    backend: str = "loky"

    def validate(self) -> None:
        """Validate the configuration, raising ValueError if inconsistent.

        Raises:
            ValueError: If n_jobs is 0, or backend is not one of "loky",
                "threading", or "multiprocessing".
        """
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be 0.")

        if self.backend not in {"loky", "threading", "multiprocessing"}:
            raise ValueError(
                "backend must be one of: 'loky', 'threading', 'multiprocessing'."
            )


def require_joblib() -> None:
    """Raise ImportError if joblib is not installed.

    Raises:
        ImportError: If the joblib package could not be imported.
    """
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
    """
    Apply a function to each item in parallel using joblib.

    Args:
        func: Callable invoked as func(item, **fixed_kwargs) for each item.
        items: Iterable of items to process; consumed into a list.
        config: Parallel execution settings. None uses ParallelConfig()
            defaults.
        show_progress: Whether to display a tqdm progress bar.
        desc: Progress bar description.
        **fixed_kwargs: Additional keyword arguments passed to every call of
            func.

    Returns:
        List of results in the same order as items.

    Raises:
        ImportError: If joblib is not installed.
        ValueError: If config is invalid (see ParallelConfig.validate).
    """
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
    """
    Apply a function to each item sequentially.

    Args:
        func: Callable invoked as func(item, **fixed_kwargs) for each item.
        items: Iterable of items to process; consumed into a list.
        show_progress: Whether to display a tqdm progress bar.
        desc: Progress bar description.
        **fixed_kwargs: Additional keyword arguments passed to every call of
            func.

    Returns:
        List of results in the same order as items.
    """
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
    """
    Apply a function to items, dispatching to parallel or serial execution.

    Args:
        func: Callable invoked as func(item, **fixed_kwargs) for each item.
        items: Iterable of items to process.
        config: Parallel execution settings. None uses ParallelConfig()
            defaults (sequential execution).
        show_progress: Whether to display a tqdm progress bar.
        desc: Progress bar description.
        **fixed_kwargs: Additional keyword arguments passed to every call of
            func.

    Returns:
        List of results in the same order as items.

    Raises:
        ValueError: If config is invalid (see ParallelConfig.validate).
        ImportError: If config.parallel is True and joblib is not installed.
    """
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
    """
    Run a function over a sequence of keyword-argument dictionaries.

    Each task dict is expanded as keyword arguments to func, i.e.
    func(**task). Useful for batch jobs where each unit of work is
    naturally described by a dict of named parameters.

    Args:
        func: Callable invoked as func(**task) for each task dict.
        tasks: Iterable of dicts, each containing the keyword arguments for
            one call to func.
        config: Parallel execution settings forwarded to run_map. None uses
            ParallelConfig() defaults.
        show_progress: Whether to display a tqdm progress bar.
        desc: Progress bar description.

    Returns:
        List of results in the same order as tasks.
    """
    def _worker(task: dict):
        return func(**task)

    return run_map(
        func=_worker,
        items=list(tasks),
        config=config,
        show_progress=show_progress,
        desc=desc,
    )

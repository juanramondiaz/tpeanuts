#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
#
# TPeanuts — Fast Simulation of Neutrino Oscillations in Matter.
#
# Developed as part of the Master's Thesis for the MSc in Advanced Physics,
# specialization in Theoretical Physics, at the Universitat de València.
#
# Author
# ------
# Juan Ramón Díaz Santos
# diazjuan@alumni.uv.es
#
# Supervisors
# -----------
# Roberto Ruiz de Austri Bazán
# rruiz@ific.uv.es
#
# Michele Lucente
# michele.lucente@unibo.it
#
# Universitat de València, Spain
# September 2026
# =============================================================================

"""Lazy re-exports: public names importable from a package without importing the package's submodules up front.

``import tpeanuts`` must stay instantaneous (no torch import), yet ``tpeanuts.PropagationModel`` should work. A package
declares ``_EXPORTS = {name: "defining.module"}``, ``__all__ = list(_EXPORTS)`` and
``__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)``; the defining module is imported the first time the name is
read (PEP 562) and the result is cached on the package.

Module contents:
    lazy_exports(...)
        Build the module-level ``__getattr__`` and ``__dir__`` for a package.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, Callable, Dict, List, Tuple


def lazy_exports(package: str, exports: Dict[str, str]) -> Tuple[Callable[[str], Any], Callable[[], List[str]]]:
    """Return the ``(__getattr__, __dir__)`` pair that serves ``exports`` lazily for ``package``.

    Args:
        package: The package's ``__name__``.
        exports: Maps each public name to the dotted path of the module that defines it.

    Returns:
        ``__getattr__(name)``, which imports the defining module on first access, caches the value in the package
        namespace and raises ``AttributeError`` for unknown names, and ``__dir__()``, which lists the package's names
        together with the lazy exports.
    """

    def __getattr__(name: str) -> Any:
        if name not in exports:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        value = getattr(importlib.import_module(exports[name]), name)
        vars(sys.modules[package])[name] = value
        return value

    def __dir__() -> List[str]:
        return sorted(set(vars(sys.modules[package])) | set(exports))

    return __getattr__, __dir__

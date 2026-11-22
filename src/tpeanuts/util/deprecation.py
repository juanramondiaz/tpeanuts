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

"""Deprecation policy and helpers: a typed warning, a decorator, and a function for renamed arguments.

Policy (tpeanuts follows semantic versioning): anything documented as public is deprecated for **at least one minor
release** before it is removed, and the removal happens only in a major release (or, before 1.0, a minor one that says
so in the changelog). A deprecated object keeps working and emits :class:`TPeanutsDeprecationWarning`, which names the
version that deprecated it, the version that removes it, and the replacement. The test suite turns these warnings into
errors, so the library itself never calls a deprecated name.

Module contents:
    TPeanutsDeprecationWarning
        ``DeprecationWarning`` subclass raised by everything below; filter on it to silence or escalate.
    warn_deprecated(...)
        Emit the warning with the standard message, attributed to the caller's caller.
    deprecated(...)
        Decorator that warns every time a function, method or class is called.
    renamed_argument(...)
        Map an old keyword argument onto its new name, warning when the old one is used.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, Dict, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class TPeanutsDeprecationWarning(DeprecationWarning):
    """A tpeanuts feature is deprecated and will be removed in a later release.

    Examples:
        >>> import warnings
        >>> from tpeanuts.util.deprecation import TPeanutsDeprecationWarning, warn_deprecated
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter("always")
        ...     warn_deprecated("old()", since="1.1", removal="2.0", alternative="new()")
        >>> caught[0].category is TPeanutsDeprecationWarning
        True
        >>> str(caught[0].message)
        'old() is deprecated since tpeanuts 1.1 and will be removed in 2.0; use new() instead.'
    """


def _message(name: str, since: str, removal: str, alternative: Optional[str]) -> str:
    text = f"{name} is deprecated since tpeanuts {since} and will be removed in {removal}"
    return text + (f"; use {alternative} instead." if alternative else ".")


def warn_deprecated(
    name: str, *, since: str, removal: str, alternative: Optional[str] = None, stacklevel: int = 2
) -> None:
    """Emit a :class:`TPeanutsDeprecationWarning` for ``name``.

    Args:
        name: What is deprecated, as the user writes it (``"foo(bar=...)"``, ``"tpeanuts.x.y"``).
        since: Version that deprecated it.
        removal: Version in which it will be removed.
        alternative: What to use instead, if anything.
        stacklevel: Frames to skip so the warning points at user code; ``2`` is the caller of the function that calls
            this one.
    """
    warnings.warn(_message(name, since, removal, alternative), TPeanutsDeprecationWarning, stacklevel=stacklevel + 1)


def deprecated(*, since: str, removal: str, alternative: Optional[str] = None) -> Callable[[F], F]:
    """Decorate a function, method or class so every call warns.

    Args:
        since: Version that deprecated it.
        removal: Version in which it will be removed.
        alternative: What to use instead, if anything.

    Returns:
        A decorator; the decorated callable behaves exactly as before and its docstring gains a ``deprecated`` note.

    Examples:
        >>> import warnings
        >>> from tpeanuts.util.deprecation import deprecated
        >>> @deprecated(since="1.1", removal="2.0", alternative="new_function")
        ... def old_function(x):
        ...     return 2 * x
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter("always")
        ...     result = old_function(3)
        >>> result, len(caught)
        (6, 1)
    """

    def decorator(target: F) -> F:
        name = f"{target.__module__}.{target.__qualname__}"
        note = _message(name, since, removal, alternative)

        @functools.wraps(target)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warn_deprecated(name, since=since, removal=removal, alternative=alternative, stacklevel=2)
            return target(*args, **kwargs)

        wrapper.__doc__ = f"{target.__doc__ or ''}\n\n.. deprecated:: {since}\n   {note}"
        return wrapper  # type: ignore[return-value]

    return decorator


def renamed_argument(
    kwargs: Dict[str, Any], old: str, new: str, *, since: str, removal: str, function: str = ""
) -> Any:
    """Pop ``old`` from ``kwargs`` (warning) and return its value, or ``None`` if it was not given.

    Use it inside a function that still accepts ``**kwargs`` for the old spelling::

        value = renamed_argument(kwargs, "eps", "epsilon", since="1.2", removal="2.0", function="fit")
        if value is not None:
            epsilon = value

    Args:
        kwargs: The keyword arguments received by the function; ``old`` is removed from it in place.
        old: The deprecated keyword.
        new: The keyword that replaces it.
        since: Version that deprecated ``old``.
        removal: Version in which ``old`` will be removed.
        function: Name of the function, for the message.

    Returns:
        The value passed as ``old``, or ``None`` when ``old`` was not given.
    """
    if old not in kwargs:
        return None
    warn_deprecated(f"{function}({old}=...)", since=since, removal=removal, alternative=f"{new}=", stacklevel=2)
    return kwargs.pop(old)

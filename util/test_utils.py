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
Minimal testing utilities for Spyder execution.

No pytest required. These helpers provide lightweight printing/formatting
for ad-hoc test scripts run interactively (e.g. in Spyder), plus simple
assertion and test-runner helpers comparable to a small subset of pytest.

Module functions:
    printoptions(...): Configure torch tensor print formatting.
    banner(...), section(...), step(...), print_header(...): Print
        decorative section headers to delimit test output.
    print_ok(...), print_fail(...): Print a single pass/fail status line.
    print_test_start(...), print_test_pass(...), print_test_fail(...): Print
        per-test status lines used by run_test_function.
    max_abs_error(...): Compute the maximum absolute error between two
        scalars or tensors.
    assert_true(...): Raise AssertionError if a condition is falsy.
    assert_close(...): Raise AssertionError if two values are not
        numerically close.
    assert_raises(...): Assert that calling a function raises a specific
        exception type.
    run_test_function(...): Run a single test function, catching and
        reporting exceptions.
    run_test_suite(...): Run a sequence of test functions and print a
        pass/fail summary.
    build_pmns(...): Build a representative PMNS matrix for testing.
    default_inputs(...): Build a dictionary of representative oscillation
        input tensors for testing.
"""



import traceback
import inspect

import torch

from tpeanuts.core.common.pmns import PMNSParams
from tpeanuts.core.SM.pmns import PMNS_SM
from tpeanuts.util.context import RuntimeContext


# Default absolute tolerance used by assert_close when not overridden.
ATOL = 1.0e-10
# Default relative tolerance used by assert_close when not overridden.
RTOL = 1.0e-8


def printoptions(precision=10, sci_mode=True, linewidth=160):
    """Configure torch tensor print formatting for readable test output.

    Args:
        precision: Number of digits printed after the decimal point.
        sci_mode: Whether to print floating values in scientific notation.
        linewidth: Maximum characters per line before wrapping.
    """
    torch.set_printoptions(
        precision=precision,
        sci_mode=sci_mode,
        linewidth=linewidth,
    )


def banner(title: str):
    """Print a title surrounded by '=' separator lines.

    Args:
        title: Text to print between the separator lines.
    """
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def section(title: str):
    """Print a title surrounded by '-' separator lines.

    Args:
        title: Text to print between the separator lines.
    """
    print("\n" + "-" * 90)
    print(title)
    print("-" * 90)


def print_ok(msg: str):
    """Print a message prefixed with an "[OK]" status tag.

    Args:
        msg: Message text to print.
    """
    print(f"[OK]     {msg}")


def print_fail(msg: str):
    """Print a message prefixed with a "[FAILED]" status tag.

    Args:
        msg: Message text to print.
    """
    print(f"[FAILED] {msg}")


def step(number: int, title: str):
    """Print a numbered step header surrounded by '#' separator lines.

    Args:
        number: Step number to display.
        title: Step description.
    """
    print("\n" + "#" * 100)
    print(f"STEP {number}: {title}")
    print("#" * 100)


def print_header(title):
    """Print a title surrounded by '=' separator lines (80 columns wide).

    Args:
        title: Text to print between the separator lines.
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_test_start(name):
    """Print a "[RUNNING]" status line for a test about to execute.

    Args:
        name: Test name to display.
    """
    print(f"\n[RUNNING] {name}")


def print_test_pass(name):
    """Print a "[PASS]" status line for a test that succeeded.

    Args:
        name: Test name to display.
    """
    print(f"[PASS]    {name}")


def print_test_fail(name, error):
    """Print a "[FAIL]" status line with the exception type and message.

    Args:
        name: Test name to display.
        error: Exception instance raised by the test.
    """
    print(f"[FAIL]    {name}")
    print(f"         {type(error).__name__}: {error}")


def max_abs_error(a, b) -> float:
    """
    Compute the maximum absolute error between two scalars or tensors.

    Args:
        a: First value; scalar or torch tensor.
        b: Second value; scalar or torch tensor. Converted to match a's
            device/dtype when either input is a tensor.

    Returns:
        Maximum absolute element-wise difference as a Python float.
    """
    if torch.is_tensor(a) or torch.is_tensor(b):
        a_t = torch.as_tensor(a)
        b_t = torch.as_tensor(b, device=a_t.device, dtype=a_t.dtype)
        return torch.max(torch.abs(a_t - b_t)).item()

    return abs(a - b)


def assert_true(condition, message="Condition is not True", name=None):
    """
    Raise AssertionError if condition is falsy.

    Args:
        condition: Value evaluated for truthiness.
        message: Error message used when name is not provided.
        name: Optional label that overrides message when provided.

    Raises:
        AssertionError: If condition is falsy.
    """
    if name is not None:
        message = name
    if not condition:
        raise AssertionError(message)


def assert_close(
    value,
    expected,
    atol=1.0e-12,
    rtol=1.0e-12,
    message=None,
    name=None,
):
    """
    Raise AssertionError if value and expected are not numerically close.

    Tensor inputs are compared with torch.allclose(atol, rtol); scalar
    inputs use the equivalent abs(value - expected) <= atol + rtol *
    abs(expected) test.

    Args:
        value: Candidate scalar or tensor value.
        expected: Reference scalar or tensor value.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        message: Optional error message used when name is not provided.
        name: Optional label that overrides message when provided.

    Raises:
        AssertionError: If value and expected are not within tolerance.
    """
    label = name if name is not None else message

    if torch.is_tensor(value) or torch.is_tensor(expected):
        value_t = torch.as_tensor(value)
        expected_t = torch.as_tensor(
            expected,
            device=value_t.device,
            dtype=value_t.dtype,
        )
        ok = torch.allclose(value_t, expected_t, atol=atol, rtol=rtol)
        diff = max_abs_error(value_t, expected_t)

        if not ok:
            if label is None:
                label = (
                    f"Tensors are not close: max_abs={diff}, "
                    f"atol={atol}, rtol={rtol}"
                )
            raise AssertionError(label)
        return

    diff = abs(value - expected)
    tol = atol + rtol * abs(expected)

    if diff > tol:
        if label is None:
            label = (
                f"Values are not close: value={value}, "
                f"expected={expected}, diff={diff}, tol={tol}"
            )
        raise AssertionError(label)


def assert_raises(expected_exception, func, *args, **kwargs):
    """
    Assert that calling func(*args, **kwargs) raises expected_exception.

    Args:
        expected_exception: Exception type expected to be raised.
        func: Callable to invoke.
        *args: Positional arguments passed to func.
        **kwargs: Keyword arguments passed to func.

    Raises:
        AssertionError: If func raises a different exception type, or
            raises no exception at all.
    """
    try:
        func(*args, **kwargs)
    except expected_exception:
        return
    except Exception as exc:
        raise AssertionError(
            f"Expected {expected_exception.__name__}, "
            f"but got {type(exc).__name__}: {exc}"
        ) from exc

    raise AssertionError(
        f"Expected {expected_exception.__name__}, but no exception was raised."
    )


def run_test_function(func, verbose_traceback=False):
    """
    Run a single zero-argument test function, catching and reporting errors.

    If func accepts a "savefig" parameter, it is called with savefig=True
    so plotting tests save their figures instead of blocking on display.

    Args:
        func: Test function to run. Must take no required arguments other
            than an optional "savefig" keyword.
        verbose_traceback: Whether to print the full traceback on failure.

    Returns:
        True if func completed without raising; False if it raised an
        exception (the exception is caught, not re-raised).
    """
    name = func.__name__
    print_test_start(name)

    try:
        if "savefig" in inspect.signature(func).parameters:
            func(savefig=True)
        else:
            func()
        print_test_pass(name)
        return True

    except Exception as exc:
        print_test_fail(name, exc)

        if verbose_traceback:
            traceback.print_exc()

        return False


def run_test_suite(test_functions, suite_name="Test suite", verbose_traceback=False):
    """
    Run a sequence of test functions and print a pass/fail summary.

    Args:
        test_functions: Iterable of zero-argument test functions, each run
            via run_test_function.
        suite_name: Title printed in the suite header.
        verbose_traceback: Whether to print full tracebacks for failures.

    Returns:
        True if every test passed; False if at least one test failed.
    """
    print_header(suite_name)

    passed = 0
    failed = 0

    for func in test_functions:
        ok = run_test_function(
            func,
            verbose_traceback=verbose_traceback,
        )

        if ok:
            passed += 1
        else:
            failed += 1

    print("\n" + "-" * 80)
    print(f"SUMMARY: {passed} passed | {failed} failed | {passed + failed} total")
    print("-" * 80)

    return failed == 0


def build_pmns():
    """
    Build a representative PMNS matrix for use in tests.

    Uses fixed, illustrative (not best-fit) mixing angles theta12, theta13,
    theta23, and CP-violating phase delta, all in radians, at float64
    precision. The resulting matrix exercises the same code paths as
    physical PMNS matrices without depending on global oscillation
    parameter defaults.

    Returns:
        PMNS_SM instance built from the fixed test angles.
    """
    context = RuntimeContext.resolve(None, torch.float64)
    return PMNS_SM(
        PMNSParams(
            theta12=torch.tensor(0.59, dtype=torch.float64),
            theta13=torch.tensor(0.15, dtype=torch.float64),
            theta23=torch.tensor(0.78, dtype=torch.float64),
            delta=torch.tensor(1.20, dtype=torch.float64),
            context=context,
        )
    )


def default_inputs():
    """
    Build a dictionary of representative oscillation input tensors for tests.

    All values are float64 torch scalars (or a length-3 tensor for
    flux_initial) with fixed, illustrative magnitudes: mixing angles
    theta12/theta13/theta23 and CP phase delta in radians; DeltamSq21,
    DeltamSq3l_NO, and DeltamSq3l_IO are mass-squared splittings in eV^2 for
    normal/inverted ordering; E_MeV is the neutrino energy in MeV; x1/x2 are
    dimensionless path-fraction markers; a/b/c are generic auxiliary
    coefficients used by some analytic test formulae; flux_initial is an
    unnormalized 3-flavour flux vector (nue, numu, nutau).

    Returns:
        Dictionary mapping parameter names to float64 torch tensors.
    """
    return {
        "theta12": torch.tensor(0.59, dtype=torch.float64),
        "theta13": torch.tensor(0.15, dtype=torch.float64),
        "theta23": torch.tensor(0.78, dtype=torch.float64),
        "delta": torch.tensor(1.20, dtype=torch.float64),
        "DeltamSq21": torch.tensor(7.42e-5, dtype=torch.float64),
        "DeltamSq3l_NO": torch.tensor(2.517e-3, dtype=torch.float64),
        "DeltamSq3l_IO": torch.tensor(-2.498e-3, dtype=torch.float64),
        "E_MeV": torch.tensor(1000.0, dtype=torch.float64),
        "x1": torch.tensor(0.20, dtype=torch.float64),
        "x2": torch.tensor(0.70, dtype=torch.float64),
        "a": torch.tensor(1.10, dtype=torch.float64),
        "b": torch.tensor(0.20, dtype=torch.float64),
        "c": torch.tensor(0.05, dtype=torch.float64),
        "flux_initial": torch.tensor([1.0, 2.0, 0.1], dtype=torch.float64),
    }

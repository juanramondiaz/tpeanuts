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

No pytest required.
"""



import traceback
import inspect

import torch

from tpeanuts.core.pmns import PMNS


ATOL = 1.0e-10
RTOL = 1.0e-8


def printoptions(precision=10, sci_mode=True, linewidth=160):
    torch.set_printoptions(
        precision=precision,
        sci_mode=sci_mode,
        linewidth=linewidth,
    )


def banner(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def section(title: str):
    print("\n" + "-" * 90)
    print(title)
    print("-" * 90)


def print_ok(msg: str):
    print(f"[OK]     {msg}")


def print_fail(msg: str):
    print(f"[FAILED] {msg}")


def step(number: int, title: str):
    print("\n" + "#" * 100)
    print(f"STEP {number}: {title}")
    print("#" * 100)


def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_test_start(name):
    print(f"\n[RUNNING] {name}")


def print_test_pass(name):
    print(f"[PASS]    {name}")


def print_test_fail(name, error):
    print(f"[FAIL]    {name}")
    print(f"         {type(error).__name__}: {error}")


def max_abs_error(a, b) -> float:
    if torch.is_tensor(a) or torch.is_tensor(b):
        a_t = torch.as_tensor(a)
        b_t = torch.as_tensor(b, device=a_t.device, dtype=a_t.dtype)
        return torch.max(torch.abs(a_t - b_t)).item()

    return abs(a - b)


def assert_true(condition, message="Condition is not True", name=None):
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
    return PMNS(
        theta12=0.59,
        theta13=0.15,
        theta23=0.78,
        delta=1.20,
        real_dtype=torch.float64,
    )


def default_inputs():
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

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

"""Harness shared by every benchmark: timer, resources, environment record, result schema and scenarios.

It must stay free of any reference-package import, so that it can be used from each package's own environment (the
nuSQuIDS one has no ``torch``): ``torch`` and ``psutil`` are imported only locally and only if installed.

Module contents:
    ``timing``
        Wall-clock timer with min/median/IQR statistics and a device-synchronisation hook.
    ``resources``
        CUDA synchronisation and peak memory, process RAM and its budget, log-log scaling fit.
    ``environment``
        Record of the machine and software of a run, and the CPU thread control.
    ``results``
        JSON schema of the records, with the correctness-first rule.
    ``scenarios``
        Scenario catalogue and the standard energy and nadir grids.
    ``spec``
        YAML specification of one benchmark: validation, ``spec_hash``, runner command line, output CSV path.
    ``cache``
        ``load_or_run``: read the CSV of a specification if it exists and is current, measure it only where allowed.
    ``view``
        The tables, figures and structural checks that the benchmark notebooks show from the loaded results.
"""

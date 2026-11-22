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

"""Benchmarks of tpeanuts against nuSQuIDS, run inside WSL in nuSQuIDS's own environment (step B3).

Each case is checked against the tpeanuts reference before it is timed; the scenarios nuSQuIDS has no equivalent
observable for are recorded as unavailable, never as zero. The code lives on the Windows drive and runs on a native WSL
copy that ``sync_wsl`` refreshes and fingerprints.

Module contents:
    ``adapter``
        nuSQuIDS solver and bodies with the shared parameters, and the scenarios over a grid.
    ``run``
        Command-line runner: checks and times the scenarios and writes a ``ResultSet`` JSON.
    ``sync_wsl``
        Copies the needed files to WSL and writes their SHA-256 manifest.
"""

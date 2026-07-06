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
Created on Sat May  9 00:03:45 2026

@author: juanr
"""


import matplotlib.pyplot as plt
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parents[3]


from tpeanuts.util.test_utils import assert_true, run_test_suite

from tpeanuts.external.mceq.core import (
    init_mceq,
)

def test_mceq_solver_plot():
    try:
        mceq = init_mceq(
                interaction_model='SIBYLL23D',
                primary_model='HillasGaisser H3a',
                theta_deg=0.0,
                info=True,
                )
        mceq.solve()

        E = mceq.e_grid
        phi = mceq.get_solution('total_numu')

        plt.loglog(E, phi)
        plt.xlabel("Energy [GeV]")
        plt.ylabel(r"$\Phi_{\nu_\mu}$")
        plt.title(r"Energy Espectru of Neutrino flux $\phi_\mu$ at $X_{obs}$ ")
        plt.grid(True)
        plt.show()

    except ImportError as exc:
        print("mceq is not available in this environment.")
        print(f"Reason: {exc}")
        print("The tpeanuts.external.mceq import path is valid; install mceq to run the solver plot.")


def run_atmosphere_tests(verbose_traceback=False):
    return run_test_suite([test_mceq_solver_plot], suite_name="atmosphere MCEQ tests", verbose_traceback=verbose_traceback)


if __name__ == "__main__":
    run_atmosphere_tests(verbose_traceback=True)

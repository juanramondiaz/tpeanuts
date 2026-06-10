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


# Constant factor 
_VK_PREFAC = 3.868e-7 / 2.533


# earth radius in meters 
R_E = 6.371e6  # m

# earth radius in kilometers 
R_E_KM = float(R_E) / 1.0e3

# solar radius in meters.
R_SUN = 6.957e8  # m

# solar radius in kilometers.
R_SUN_KM = float(R_SUN) / 1.0e3

# Astronomical unit in kilometers.
AU_KM = 149_597_870.7  # km

# Astronomical unit in meters.
AU_M = AU_KM * 1.0e3

# Default mean Sun-earth distance.
SUN_EARTH_DISTANCE_KM = AU_KM
SUN_EARTH_DISTANCE_AU = 1.0


# hbar*c in MeV*m 
HBARC_MeV_m = 197.3269804e-15

# proton mass in kilograms.
M_PROTON_KG = 1.67262192369e-27

# Avogadro constant in mol^-1.
N_A = 6.02214076e23

# Conversion from mass density [g/cm^3] to nucleon molar density [mol/cm^3]
# using the proton mass as the nucleon mass scale.
GCM3_TO_NUCLEON_MOLCM3 = 1.0e-3 / (M_PROTON_KG * N_A)

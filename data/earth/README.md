# Radial Earth reference models

`prem`, `ak135`, and `legacy` contain canonical pointwise density tables and
provenance metadata. PREM is the configured default. The legacy provider also
contains the original five-layer even-power fit used by PEANUTS and canonical
electron/neutron fit tables under `legacy/fit`.

`prem/density/prem_density_dense.csv` is a 1 km resampling of each continuous
PREM branch. Both sides of density discontinuities are retained. It is useful
as a numerical grid but contains no additional physical information relative
to the original PREM table/parametrisation.

The seismic models determine mass density, not chemical composition. The
canonical `electron_fraction`, `electron_density_mol_cm3` and
`neutron_density_mol_cm3` columns are therefore documented neutrino-physics
extensions. They must not be cited as original PREM or ak135 observables.

These pointwise reference tables coexist with provider-local `fit/` tables,
which contain the coefficients used by the perturbative Earth propagator.

Select the converted PEANUTS model with
`EarthParameters(density_provider="legacy")`. Its pointwise table is sampled
from the exact five-layer polynomial, while `legacy/fit` retains the compact
coefficients used by analytical propagation. `legacy_data_dir` remains
`data/peanuts` and is unrelated to provider selection.

# Canonical solar providers

Runtime providers use separate `density/`, `production/`, and `flux/` CSV
products. Optional spectra, probabilities, observations, source metadata, and
unaltered downloads live in their corresponding directories.

- `zenodo` (default): SF-III AGSS09 density, production, and total fluxes.
- `bahcall`: BP2000/BP2004 canonical products.
- `legacy`: B16 AGSS09met canonical products converted from the files bundled
  with legacy PEANUTS. GS98 is retained as an explicit alternative table.
- `sno` and `borexino`: detector observations/probability products; they are
  not complete solar-structure providers.

Select the legacy runtime model with
`SolarParameters(provider="legacy")`. The separate configuration value
`legacy_data_dir = "data/peanuts"` is intentionally reserved for direct
validation against the old Python package and is not a provider path.

Production spectra are selected independently from the structural provider.
The default is `spectrum_provider="legacy"`, so the default Zenodo SF-III
profile automatically carries the legacy pp, hep, 7Be, 8B and CNO spectra.
The default 8B variant is Winter (`"ortiz"` is available explicitly); the
default 7Be spectrum combines its two line shapes. No pep spectrum is bundled,
and requesting its differential flux without an explicit override raises a
clear missing-spectrum error.

The B16 source files do not contain a neutron-density profile. Consequently,
the canonical legacy density contains electron density only. Standard-model
propagation works directly; a sterile calculation requiring the NC potential
must provide an explicitly chosen, composition-compatible neutron profile.

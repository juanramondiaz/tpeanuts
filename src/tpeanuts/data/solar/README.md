# Solar-neutrino input data

Ported from this project's frozen v1_0 reference (`v1_0/data/solar/`).
Phase 1 of the solar migration (`tpeanuts.source.solar`): production and
total-flux tables, plus production-energy spectra. Phase 2
(`tpeanuts.medium.solar`): one electron/neutron density table per
provider (below) -- no composition table (`raw/struct+nu_*.dat`) is
ported, since none of the three density tables actually ported needs its
derived-neutron-density fallback (see below); a future phase would port
it only if a provider genuinely needing it is added. Only the files a
v1_0 loader actually references are copied; v1_0's own `legacy/spectrum/`
directory has three orphaned CSVs (`spectrum_7Be.csv`,
`spectrum_7Be_excited.csv`, `spectrum_7Be_ground.csv`) that no
`_SPECTRUM_FILES` entry in `source/solar/io.py` uses -- not ported.

Three independent providers, mirroring v1_0's own three-provider split
(`source/solar/io.py::_PROVIDER_FILES`/`_SPECTRUM_FILES`):

- **`zenodo/`** -- B23/SF-III (DOI `10.5281/zenodo.10822316`, v1.2), the
  configured default provider (`config.default.SOLAR_PROVIDER`). Only the
  **AGSS09** model is ported (`production_SF3_AGSS09.csv`,
  `fluxes_SF3_AGSS09.csv`, `density/density_SF3_AGSS09.csv`) -- v1_0
  ships five further composition variants (AAG21, C11, GS98, MB22m,
  MB22p) that no default or prior request uses. No production spectra
  (`energy_spectra_available: false` in `metadata/source.json`).
  `density_SF3_AGSS09.csv` carries `electron_density_mol_cm3` and
  `neutron_density_mol_cm3` directly (its `mass_density_g_cm3` column is
  ignored by the runtime loader).
- **`bahcall/`** -- Bahcall Solar Neutrino Software and Data
  (`https://www.sns.ias.edu/~jnb/SNdata/sndata.html`), ported in full:
  `bp2004_production.csv` (per-shell production fractions, extra
  structural columns such as `temperature_MK`/`density_log_10` are
  present in the file but ignored by the runtime loader),
  `fluxes_bahcall_bp2004.csv`, the full 11-file production-spectrum set
  under `spectrum/` (continuous pp/hep/8B/13N/15O/17F shapes plus the
  discrete 7Be/pep lines, with `spectrum/metadata.json` documenting the
  two lines' own provenance -- Bahcall, Phys. Rev. D 49, 3923 (1994)),
  and `density/bp2000_density_electron_neutron_monotonic.csv` (also
  carries both electron and neutron density directly).
- **`legacy/`** -- B16 (Vinyoles et al., Astrophys. J. 835, 202 (2017),
  arXiv:1611.09867), the configured default *spectrum* provider
  (`config.default.SOLAR_SPECTRUM_PROVIDER`). Only the **AGSS09met**
  production/flux/density variant is ported (not `GS98`), but the
  **full** 11-file referenced spectrum set is (legacy is the default
  spectrum provider, and its `winter`/`ortiz` 8B and `ground`/`excited`
  7Be variant mechanism must stay intact).
  `density/density_b16_agss09_extended.csv` carries only
  `electron_density_mol_cm3` -- no neutron column, and (see above) no
  composition fallback is ported to derive one, so
  `tpeanuts.medium.solar.solar.SolarMedium.density_n` is `None`
  for this provider, matching v1_0 exactly (its own `composition_path`
  is opt-in with no default resolution). Note: v1_0's own
  `medium/solar/io.py` still points at the *plain*
  `density_b16_agss09.csv`, but that file only tabulates radius up to
  0.5 R_sun (confirmed directly) and so cannot satisfy
  `SolarMedium`'s own "must reach the solar surface" invariant --
  `..._extended.csv` (present in the same v1_0 folder, reaching
  radius=1.0) is ported here instead, user-confirmed 2026-09-23.

Each provider's `metadata/source.json` (provider, DOI/reference,
default-model flag) is copied verbatim. Directory layout mirrors v1_0's
own `{production,flux,spectrum,density,metadata}/` split; consumed by
`tpeanuts.source.solar.io` (production/flux/spectrum) and
`tpeanuts.medium.solar.density` (density).

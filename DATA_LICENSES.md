# Third-party data: provenance, licenses and redistribution

`tpeanuts` itself is released under GPL-3.0-or-later (see `LICENSE`). The **data** it reads come from the
sources below, each with its own terms. This file records, per dataset, where it comes from, what is known
about its license, how that was checked, and whether it is shipped in the package.

Last verification: 2026-10-22. "Verified" means the terms were read on the publisher's record page (Zenodo,
Harvard Dataverse) or in the project's own repository on that date. Re-check before each release.

## Status categories

| Status | Meaning | Policy |
|---|---|---|
| **A. Open license verified** | An explicit open license is stated by the data owner. | May be redistributed with the attribution the license requires. |
| **B. Published numbers, no explicit license** | Values tabulated or plotted in a journal article or on a collaboration page, transcribed or converted by this project. No license text found. | Attribute and cite. Shipped only after the maintainer's explicit sign-off (see "Decision needed"). |
| **C. Terms not found** | No statement of terms could be located, or the source could not be inspected. | **Not shipped in the wheel.** Fetched by the user from the original source, or shipped only after written permission. |

## Summary

| Dataset | Location (packaged: relative to `tpeanuts/data`; others: development sandbox `data/`) | Size | Status | Shipped in the wheel |
|---|---|---|---|---|
| Daya Bay full data release | `experiments/dayabay/` and `reactor/dayabay/` (the reactor-source tables: spectra, fission fractions, thermal power, non-equilibrium correction) | 1.3 MB | A (CC BY 4.0) | yes |
| Solar model B23 / SF-III (AGSS09) | `solar/zenodo/` | <0.8 MB (all solar) | A (CC BY 4.0) | yes |
| IceCube DeepCore 9-year release | `datasets/icecube/` (sandbox) | 64 MB | A (CC0 1.0) | **no** (size) - downloaded |
| PREM Earth model | `earth/` | 32 KB | B | yes (sign-off 2026-10-22) |
| Solar model B16 (legacy tables) | `solar/legacy/` | part of 0.8 MB | B (via PEANUTS, GPL-3.0) | yes (sign-off 2026-10-22) |
| Bahcall solar data (BP2000/BP2004) | `solar/bahcall/` | part of 0.8 MB | B | yes (sign-off 2026-10-22) |
| Nakamura nu-d cross sections | `experiments/sno/nakamura/` | <0.9 MB | B | yes (sign-off 2026-10-22) |
| SNO observations | `experiments/sno/observation/` | <0.9 MB | B | yes (sign-off 2026-10-22) |
| SNO salt-phase (SNO-II) tables | `experiments/sno_ii/` | 22 KB | B (manual transcription) | yes (sign-off 2026-10-22) |
| Borexino 2018 spectra | `experiments/borexino/` | 49 KB | B | yes (sign-off 2026-10-22) |
| Honda/HKKM flux, derived table | `generated/atmosphere/honda/flux/honda_flux.csv` | 3.6 MB | **C** | **no** - generated locally (maintainer decision 2026-10-22) |
| Honda/HKKM raw tables | `datasets/honda/` | 289 MB | **C** | **no** - downloaded |
| nuSQuIDS `astro/*.dat` | not in the repository | - | n/a | no - fetched by `validation.nusquids.fetch_data` |
| MCEq / crflux databases | not in the repository | - | n/a | no - downloaded by MCEq |

## Per-dataset detail

### Daya Bay full data release - status A
- **Source:** Daya Bay Collaboration, *Full Data Release of the Daya Bay Reactor Neutrino Experiment*, v1.0.0
  (15 Nov 2025), Zenodo, DOI [10.5281/zenodo.17587229](https://doi.org/10.5281/zenodo.17587229). Obtained from
  the official GitHub mirror `dayabay-experiment/dayabay-data-official`.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0), read on the Zenodo record.
- **Conditions:** credit the creator; cite both the release and F. P. An et al. (Daya Bay Collaboration),
  *Precision Measurement of Reactor Antineutrino Oscillation at Kilometer-Scale Baselines by Daya Bay*,
  Phys. Rev. Lett. 130, 161802 (2023), DOI 10.1103/PhysRevLett.130.161802. A copy should be accompanied by the
  current dataset description.
- **Layout:** the tables that describe the reactor itself (`huber_mueller_spectra.csv`, `nonequilibrium_correction.csv`,
  `reactor_parameters.csv`, `reactor_thermal_power_gw.txt`) are packaged under `reactor/dayabay/` since 2026-10-28; the
  experiment's own tables stay under `experiments/dayabay/`. Contents unchanged (same SHA-256).
- **Changes made by this project:** the `tsv.bz2` tier was converted to tidy `.csv`; no values were altered
  (this must be stated wherever the files are redistributed, as CC BY requires).
- **Not checked:** which of the files (e.g. `huber_mueller_spectra.csv`) are part of the release itself and which were added by this project from other references; verify before redistribution.

### Solar model B23 / SF-III (AGSS09) - status A
- **Source:** Herrera, Y. and Serenelli, A., *Standard Solar Models B23 / SF-III*, v1.2 (21 Nov 2023), Zenodo,
  DOI [10.5281/zenodo.10822316](https://doi.org/10.5281/zenodo.10822316); the record points to Solar Fusion III
  (arXiv:2405.06470) for the nuclear rates and opacities.
- **License:** CC BY 4.0, read on the Zenodo record. Only the AGSS09 variant is used.

### IceCube DeepCore 9-year golden-event release - status A
- **Source:** IceCube Collaboration, *Replication Data for: Measurement of atmospheric neutrino mixing with
  improved IceCube DeepCore calibration and data processing*, Harvard Dataverse, V1 (29 Jul 2025),
  DOI [10.7910/DVN/B4RITM](https://doi.org/10.7910/DVN/B4RITM); accompanies Phys. Rev. D 108, 012014 (2023),
  arXiv:2304.12236.
- **License:** CC0 1.0 (public-domain dedication), read through the Dataverse API. CC0 imposes no citation duty; citing the release is good practice.
- **Integrity:** the file sizes reported by Dataverse match the local files (e.g. `mc_numu_cc.tab`, 44 626 533 B).
- **Policy:** not shipped (64 MB, only needed by the IceCube detector module); fetched with `tpeanuts-data`.

### PREM Earth model - status B
- **Source:** A. M. Dziewonski and D. L. Anderson, *Preliminary reference Earth model*, Phys. Earth Planet.
  Inter. 25, 297-356 (1981), DOI 10.1016/0031-9201(81)90046-7. The table was taken from a third-party copy
  (`ondrejsramek.net/.../prem_table_full.txt`, see `earth/metadata/source.json`).
- **Terms:** published numerical model; no license statement located.
- **Project additions:** the electron/neutron density columns and the even-power polynomial fits are this
  project's own derivations (they assume electron fractions per layer; see `earth/README.md`) and are covered by
  the project license.

### Solar model B16 ("legacy" tables) - status B
- **Source:** N. Vinyoles et al., *A new generation of standard solar models*, Astrophys. J. 835, 202 (2017),
  arXiv:1611.09867. The tables used here were converted from files bundled with PEANUTS.
- **Terms:** PEANUTS is distributed under GPL-3.0 (Gonzalo and Lucente, Eur. Phys. J. C (2024),
  arXiv:2303.15527, <https://github.com/michelelucente/PEANUTS>), which is compatible with this project. The
  terms set by the B16 authors for their own tables were **not** located.
- **Note:** this is the *default spectrum provider* (`SOLAR_SPECTRUM_PROVIDER`); see "Decision needed".

### Bahcall solar data - status B
- **Source:** Bahcall Solar Neutrino Software and Data, <https://www.sns.ias.edu/~jnb/SNdata/sndata.html>
  (BP2000 densities, BP2004 production and fluxes, production spectra; Bahcall, Phys. Rev. D 49, 3923 (1994)
  for the 7Be/pep lines).
- **Terms:** no terms of use located. Not the default provider.

### Nakamura neutrino-deuteron cross sections - status B
- **Source:** Nakamura et al., Nucl. Phys. A707, 561 (2002). Tabulated cross sections converted to CSV
  (`experiments/sno/nakamura/`). No license statement located.

### SNO observations and SNO salt-phase (SNO-II) tables - status B
- **SNO observations:** SNO collaboration public page, <https://sno.phy.queensu.ca/sno/prlwebpage/>.
- **SNO-II:** B. Aharmim et al. (SNO), Phys. Rev. C 72, 055502 (2005), arXiv:nucl-ex/0502021, *manually
  transcribed from the rendered PDF* (no machine-readable release exists); the transcription was cross-checked
  against the paper's totals (`experiments/sno_ii/metadata/source.json`). Verify the transcription before relying
  on it for publication-grade results.
- **Neutral-current capture parameters:** SNO Collaboration, Phys. Rev. Lett. 89, 011301 (2002).

### Borexino 2018 - status B
- **Source:** Borexino Collaboration, Nature (2018), DOI 10.1038/s41586-018-0624-y; figure data from
  the collaboration page. No license statement located.

### Honda/HKKM atmospheric flux - status C
- **Source:** M. Honda et al. (2015), arXiv:1502.03916 (the paper is kept in `external/Honda/`);
  tables at `http://www-rccn.icrr.u-tokyo.ac.jp/mhonda/public/nflx2014/` (downloaded 2026-06-05, 591 files
  listed, 2 failed; see `external/Honda/README_download.txt`).
- **Terms:** none found. The page could not be inspected (its TLS certificate does not match the host) and a
  web search found no usage statement.
- **Affected files:** the raw tables (289 MB, never shipped), the single raw file kept for provenance
  (`atmosphere/honda/raw/frj-0608-20-12-solmin.d.gz`) and the **derived** `honda_flux.csv`.
- **Upstream availability (checked 2026-10-22):** the registered URL
  `http://www-rccn.icrr.u-tokyo.ac.jp/mhonda/public/nflx2014/` now answers `301` to HTTPS with a TLS certificate that
  does not match the host (it belongs to `icrr2.jp`), and the file path returns `404` even with certificate
  verification disabled; `icrr2.jp` does not serve the path either. The tables could not be fetched automatically.
  The local copy used to validate `tpeanuts` was downloaded on 2026-06-05 (see `README_download.txt`).
  `tpeanuts-data fetch honda --from DIR_OR_URL` accepts a local copy or a mirror and verifies every file against the
  registry checksums.
- **Policy:** neither the raw nor the derived table is shipped until written permission is obtained. The user
  downloads the original table and the derived CSV is generated locally. Citation of the HKKM papers is required.

## Decisions (maintainer)

Taken on 2026-10-22:

1. **Honda/HKKM: local generation.** Neither the raw tables nor the derived `honda_flux.csv` are shipped. The
   default atmospheric surface-flux table (`honda/flux/honda_flux.csv` under the generated-data directory) must be
   produced on the user's machine from the original Honda tables, or an explicit table passed to
   `load_atmospheric_flux`. Reading it when absent raises `DataNotAvailableError` with this pointer.
2. **Status-B datasets are shipped in the wheel with attribution** (PREM, Bahcall and B16 solar tables, Nakamura,
   SNO, SNO-II, Borexino). The sources and citations above travel with the package (`DATA_LICENSES.md`, `NOTICE`,
   and the `README`/`metadata` files kept next to the data).

Still open:

3. **Default solar spectra (B16 via PEANUTS):** confirm the B16 authors' terms, or switch the default spectrum
   provider to B23 (status A, but B23 carries no production spectra).
4. **SNO-II:** the manual PDF transcription should be re-verified before it backs publication-grade results.

## Attribution required by CC BY (for redistributed data)

Daya Bay (CC BY 4.0) and B23 / SF-III (CC BY 4.0): attribution, link to the license, and an indication of the
changes made. These notices are repeated in `NOTICE`.

# Earth reference data (PREM)

Ported from this project's frozen v1_0 reference
(`v1_0/data/earth/prem/`). Only PREM (Dziewonski & Anderson, Phys. Earth
Planet. Inter. 25, 297-356 (1981), DOI 10.1016/0031-9201(81)90046-7) is
ported -- v1_0 also has AK135 and a "legacy" density model, neither of
which has any consumer in `tpeanuts.medium.earth` yet, so they are not
copied here (same "no speculative data ahead of a real need" policy as
`tpeanuts.config.default`/`tpeanuts.util.constant`). `prem_density_dense.csv`
(a 1 km-resampled grid of the same table, per v1_0's own
`data/earth/README.md`) is also not ported: it "contains no additional
physical information relative to the original PREM table," so there is
nothing it can provide that interpolating the sparse table directly
cannot.

Directory layout (mirrors v1_0's own `prem/{raw,density,fit,metadata}/`
split, minus the now-unnecessary `prem/` provider-name layer since this
is the only Earth density model here):

- `raw/prem_table_full.txt` -- the original published PREM table
  (radius, mass density, seismic velocities, elastic moduli, gravity,
  pressure, depth), unmodified, for full provenance.
- `density/prem_density.csv` -- the same table plus derived
  `radius_fraction`/`electron_fraction`/`electron_density_mol_cm3`/
  `neutron_density_mol_cm3` columns. **These four are neutrino-physics
  extensions, not original PREM observables** -- PREM determines mass
  density, not chemical composition; the electron fraction per layer
  (`composition_extension` in `metadata/source.json`) is a separate,
  explicitly model-dependent assumption. Consumed today by
  ``tpeanuts.medium.earth.density.load_prem_table`` (electron/neutron
  density only).
- `fit/even_power_electron.csv`, `fit/even_power_neutron.csv` -- per
  -shell even-power polynomial coefficients (`rj,alpha,beta,gamma`:
  `n(r) = alpha + beta*r^2 + gamma*r^4` for `r <= rj`), a compact
  closed-form fit to the same electron/neutron density columns above.
  Consumed by ``tpeanuts.medium.earth.even_power``/
  ``tpeanuts.medium.earth.earth.EarthMedium``'s
  ``PropagationMethod.ANALYTIC_PERTURBATIVE`` path (see
  ``tpeanuts.physics.evolution.perturbative.profile.EvenPowerProfile``), instead of
  the raw tabulated samples ``density/prem_density.csv`` provides for
  ``NUMERICAL``/``ANALYTIC``.

  **Not a verbatim copy of v1_0's own file.** v1_0's 5-shell fit
  (`rj` = 0.192, 0.546, 0.895, 0.937, 1.0) collapses the entire
  mantle-to-crust-to-ocean transition (the outermost ~400 km) into one
  shell -- confirmed directly that this makes the fit's last shell miss
  the crust/ocean density jump entirely (predicting ~1.67 mol/cm^3 at
  the surface where the tabulated data drops to 0.567, a ~3x error,
  since a 3 km-thick ocean layer's contribution to a least-squares fit
  over a ~400 km shell is negligible). The last shell was re-fit here as
  two: `rj = 0.9995291163082719` (mantle+crust, least-squares over the
  tabulated points in that range) and `rj = 1.0` (ocean, an exact
  2-point fit through the table's own last two rows, `gamma = 0` --
  only two tabulated points exist for this layer, so an exact linear
  -in-`r^2` fit is the only well-posed choice -- fitting a full 3
  -parameter `alpha,beta,gamma` there would be underdetermined). The other four shells (`rj` <= 0.937) are v1_0's own values,
  unchanged.

  Even after this split, the mantle+crust shell (now `rj=0.9995...`)
  still has real internal structure it does not fully capture (PREM's
  own Moho and upper/lower-crust discontinuities sit inside it, up to
  ~13% pointwise error confirmed directly) -- deliberately not split
  further: chasing every remaining PREM discontinuity with its own
  shell would keep eroding the "few segments" property that is the
  entire point of `ANALYTIC_PERTURBATIVE` over `NUMERICAL`. The
  resulting ~0.5-1% level discrepancy in `P_ee` near a trajectory's
  surface entry/exit (confirmed directly against `NUMERICAL` at high
  `nsteps`) is accepted as an inherent property of a compact 6-shell
  polynomial representation, not a bug -- see
  ``tpeanuts.medium.earth.test.test_earth``'s own convergence test for
  the actual measured tolerance.
- `metadata/source.json` -- citation, DOI, source URL, and the
  electron-fraction constants used to derive `density/prem_density.csv`
  from the raw mass-density table.

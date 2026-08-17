# Data release of the IceCube/DeepCore 9-year golden event oscillation analysis
This sample contains the data used in the [Measurement of atmospheric neutrino mixing with improved IceCube DeepCore calibration and data processing](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.108.012014), published in Phys. Rev. D 108, 012014 (2023). Please refer to the publication for a detailed explanation of the sample and analysis.

---

## Files
Along with this readme, you'll find nine `.csv` files and one ipython notebook:
* `data.csv`: count of data events per analysis bin.
* `mc_nu*.csv`: event-by-event information of Monte-Carlo (MC) neutrinos; split between neutral current (NC) and neutrino flavor (electron, muon, and tau) + charged current (CC).
* `mc_mu.csv`: per-bin information from the MC muon background.
* `hs_nu*.csv`: correction to be applied to the neutrino MC expected count per bin to account for detector systematics; split between NC + nu_e CC, nu_mu CC, and nu_tau CC.
* `example.ipynb`: ipython notebook presenting a simple fit to the data.

---

## Binning
The reconstructed variables are provided in the analysis binning, such that events fall into the mid-point of their bin.
* Reconstructed energy, or `reco_energy`, in GeV: `[6.31, 8.45862141, 11.33887101, 15.19987592, 20.37559363, 27.3136977, 36.61429921, 49.08185342, 65.79474104, 88.19854278, 158.49]`.
* Reconstructed cosine of the particle's zenith angle, or `reco_coszen`: `[-1., -0.89, -0.78, -0.67, -0.56, -0.45, -0.34, -0.23, -0.12, -0.01, 0.1]`. The convention is such that -1 corresponds to straight "upgoing" trajectories (earth crossing), +1 to straight "downgoing" (directly from the sky above), and 0 is horizontal, i.e. from the horizon.
* Reconstructed particle ID, or `pid`: `[0.55, 0.75, 1.]`. Represents the likelihood that an event is a cascade or a track, where 0 corresponds to a cascade and 1 to a track, to the best of our knowledge. Since this sample focuses on tracks, the cascade bin `[0., 0.55]` is omitted.

---

## data.csv
The counts of observed data in the analysis binning. Each row corresponds to one bin, such that
* `count`: number of data events in the corresponding bin;
* `pid`: reconstructed particle ID in analysis binning;
* `reco_coszen`: reconstructed cosine(zenith) in analysis binning; and
* `reco_energy`: reconstructed energy (GeV) in analysis binning.

---

## mc_nu*.csv
MC neutrinos with one event per row. The four files included are
* `mc_nu_nc.csv`: neutral-current (NC) neutrino events, all flavors;
* `mc_nue_cc.csv`: charged-current (CC) electron neutrino events;
* `mc_numu_cc.csv`: CC muon neutrino events; and
* `mc_nutau_cc.csv`: CC tau neutrino events.

Each row corresponds to one event. The reconstructed variables follow the same convention as `data.csv`, where
* `reco_energy` is the reconstructed energy (GeV) in analysis binning;
* `reco_coszen` is the reconstructed cosine(zenith) in analysis binning; and
* `pid` is the reconstructed particle ID in analysis binning.

Then, the true information is
* `pdg` stands for the PDG code of the neutrino interacting in the detector (see the [PDG summary tables](https://pdg.lbl.gov/2024/tables/contents_tables.html));
* `true_energy` is the simulated energy (GeV) of the neutrino; and
* `true_coszen` is the simulated cosine(zenith) of the neutrino.

Due to the way neutrinos are simulated in IceCube, each MC event has an associated weight, encoded by
* `weight`, in (GeV cm^2 sr). This should be multiplied by the atmospheric flux in (cm^-2 sr^-1 s^-1) and by lifetime in (s). For the sample's lifetime, please see the analysis publication.

There are two reaction types and a few interaction types for the simulated neutrinos, which are encoded as
* `type` for the reaction type, where 0 stands for NC and 1 for CC; and
* `interaction` for the interaction type, where
    * 0 stands for quasi-elastic (QE) events;
    * 1 for resonant (RES) events;
    * 2 for deep inelastic scattering (DIS) events;
    * 3 for coherent interactions; and
    * -1 stands for other types of interactions. These are very rare (< 0.08% of the total weighted events).

For non-DIS CC events, we encode the interaction uncertainty parameters through quad and linear fit terms, the parameters of which are
* `MaCCQE_linear` and `MaCCQE_quad` for the axial mass CC QE events; and
* `MaCCRES_linear` and `MaCCRES_quad` for the axial mass CC RES events.

Please find a brief explanation on how to use these in Section VI.C.2. in the publication.

Then, for DIS events, we encode the experiment-like kinematic variables. All of these neglect Fermi momentum / off-shellness.
* `Q2` (GeV^2) is the momentum transfer Q^2;
* `W` (GeV) is the hadronic invariant mass;
* `x` is the Bjorken scaling variable; and
* `y` the inelasticity.

---

## mc_mu.csv
Counts and absolute uncertainties for the atmospheric muon background in the analysis binning. Due to the high efficiency of the event selection, the produced histogram is sparsely populated, which we overcome by applying a variable bandwidth kernel density estimator (KDE).
* `count`: number of muons in the corresponding bin.
* `abs_uncertainty`: absolute uncertainty on count, accounting for statistical and shape uncertainty.
* `pid`: reconstructed particle ID in analysis binning.
* `reco_coszen`: reconstructed cosine(zenith) in analysis binning.
* `reco_energy`: reconstructed energy (GeV) in analysis binning.

For a more details explanation of the KDE procedure used, please see Section VI.D. in the publication.

---

## hs_nu*.csv
Hypersurfaces needed to correct the histograms from detector uncertainties. The three files included are
* `hs_nu_nc_nue_cc.csv`: all NC neutrinos and CC electron neutrinos hypersurfaces;
* `hs_numu_cc.csv`: CC muon neutrinos hypersurfaces; and
* `hs_nutau_cc.csv`: CC tau neutrinos hypersurfaces.

Each bin in the MC neutrino histogram is to be corrected by a multiplicative factor obtained with these hypersurfaces. This factor is given by the intercept `c` plus a slope `m_n` multiplied by the difference between the parameter fit and the nominal value for the parameter `p_n`, or

    f(p_1, ..., p_N) = c + sum_{n=1}^N m_n * Delta p_n.

The nominal value for each parameter is
* DOM efficiency (`dom_eff`): 1.00;
* hole ice p0 (`hole_ice_p0`): 0.10;
* hole ice p1 (`hole_ice_p1`): -0.05;
* bulk ice absorption (`bulk_ice_abs`): 1.00; and
* bulk ice scattering (`bulk_ice_scat`): 1.00.

For the best-fit parameters values, please see the publication.

Each hypersurfaces file encodes the values for the intercept and each of the slopes, along with their sigma. Like the other files, hypersurfaces share the same PID, reconstructed cosine zenith, and reconstructed energy bins. In addition, since the hypersurface fits are sensitive to the choice of Delta m31, they are also binned in this parameter. To account for hypersurfaces between two Delta m31 bins, a piece-wise linear interpolation can be used. Please find its implementation in the included iPython notebook.

For a more detailed explanation of the hypersurfaces procedure, see Section VI.A. in the publication.

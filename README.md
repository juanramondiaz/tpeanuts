# tpeanuts

**A modular, GPU-capable neutrino propagation engine in PyTorch.**

tpeanuts computes neutrino flavour evolution in vacuum and in matter — through the Earth, inside the Sun, in the
atmosphere and from reactors — for the Standard Model and several extensions, on CPU or GPU, with batched inputs and
automatic differentiation. Every ingredient (mixing, mass spectrum, new-physics term, medium, source, detector) is a
separate object composed into one `PropagationModel`, so adding a new effect never means forking the propagation code.

- **Physics:** three-flavour oscillations with the MSW effect; 3+N sterile neutrinos; non-standard interactions (NSI);
  Dirac and Majorana neutrinos; neutrino magnetic moment; mass-basis decoherence. All share one density-matrix
  formalism.
- **Propagation methods:** numerical (exact), perturbative analytic, adiabatic with Landau–Zener, and open-system
  (Lindblad) evolution.
- **Media and sources:** Earth (PREM, any nadir angle and detector depth), Sun (production profiles and fluxes),
  atmosphere (exponential, tabulated or NRLMSIS density; MCEq or Honda fluxes), reactors.
- **Detectors and fits:** event rates for Daya Bay, SNO, Borexino and IceCube, and differentiable likelihoods and
  profile scans built on top (experimental API).
- **Engineering:** batched over energies, angles and parameters; runs on CPU and CUDA at float32 or float64; unit,
  integration, regression and cross-validation tests (including against nuSQuIDS, with agreement at the few-`1e-3` level).

## Installation

```bash
pip install tpeanuts
```

This installs the CPU build of PyTorch on Windows and macOS. For CUDA, install PyTorch first from the matching index and
then tpeanuts:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install tpeanuts
```

Python 3.11 or later is required (Linux is the reference platform; Windows is supported; macOS is untested).

Optional extras: `pip install "tpeanuts[msis]"` (NRLMSIS atmosphere), `"tpeanuts[mceq]"` (atmospheric fluxes with MCEq),
`"tpeanuts[nusquids]"` (cross-checks against nuSQuIDS), `"tpeanuts[test]"` (test tooling), `"tpeanuts[all]"`.

### Data

The small tables the physics needs (PREM, solar models, cross sections, Daya Bay inputs, …, 3 MB) ship inside the package.
Two larger datasets are downloaded on demand, with checksums:

```bash
tpeanuts-data list                 # what is available and what is installed
tpeanuts-data fetch icecube        # public IceCube data (CC0)
tpeanuts-data build honda-flux     # Honda surface flux table, generated locally from the upstream raw file
```

Set `TPEANUTS_DATA_DIR` and `TPEANUTS_USER_DATA_DIR` to relocate the downloaded and generated data. Licences and
attributions of every dataset are in [`DATA_LICENSES.md`](DATA_LICENSES.md).

## A first calculation

Probability that a muon neutrino is detected as an electron neutrino after crossing the Earth through its centre:

```python
import math

import torch
import tpeanuts as tp

context = tp.RuntimeContext.resolve("cpu", torch.float64)   # or ("cuda", torch.float32)

space = tp.FlavourSpace()
mixing = tp.MixingMatrix(
    space=space,
    standard=tp.StandardMixingParameters(
        theta12=math.radians(33.4), theta13=math.radians(8.6), theta23=math.radians(42.0),
        delta13=math.radians(230.0), context=context,
    ),
    nature=tp.Dirac(),
    context=context,
)
masses = tp.MassSpectrum(space=space, DeltamSq21=7.4e-5, DeltamSq3l=2.5e-3, context=context)
model = tp.PropagationModel(mixing=mixing, mass_spectrum=masses)

energies_MeV = torch.logspace(2.5, 4.5, 5, dtype=torch.float64)          # 0.3 - 30 GeV
nadir = torch.tensor(0.0, dtype=torch.float64)                            # through the Earth's centre
earth = tp.EarthMedium(eta_rad=nadir, context=context, method=tp.PropagationMethod.NUMERICAL)
P = earth.transition_matrix(model, energies_MeV)

# P[..., final, initial]: probability of a muon neutrino being detected as an electron neutrino
print(P[:, 0, 1])
# tensor([3.3397e-03, 4.0993e-02, 3.2896e-01, 6.3611e-02, 3.3077e-05], dtype=torch.float64)
assert torch.allclose(P.sum(dim=-2), torch.ones(5, 3, dtype=torch.float64))   # probability is conserved
```

## Documentation

The documentation in [`docs/`](docs/index.md) (build it with `pip install -e ".[docs]"` and `make -C docs html`) has a
quickstart, a user guide (concepts, units and conventions, batching rules, precision, new physics, extending the
engine), eight tutorials, the API reference and a page of known limitations. Start with the
[conventions](docs/guide/conventions.md).

## Notebooks

[`notebooks/`](notebooks/README.md) holds tutorial notebooks written as self-contained research documents, with
theory, references, checks and an interpretation of the results; start with the `notebooks/1_physics/1_foundation/` series,
from `01_getting_started.ipynb` (`pip install "tpeanuts[notebooks]"`).

## Examples

[`examples/`](examples/README.md) has self-contained, tested scripts (solar MSW, Earth batching, …); each ends with
physical checks and can draw a figure with `--plot` (`pip install "tpeanuts[examples]"`).

## Public API and stability

The top-level names (`import tpeanuts as tp`) and the names listed in `__all__` of `tpeanuts.util`, `physics`,
`physics.extensions`, `medium`, `source`, `pipeline` and `interfaces` are the **stable API**: they follow semantic
versioning, and anything removed is deprecated for at least one minor release first
(`tpeanuts.util.deprecation`). `tpeanuts.detector`, `tpeanuts.experiments` and `tpeanuts.inference` are **experimental** in 1.0 and may change
between minor versions. Everything under a name starting with an underscore is private. The package ships type
information (`py.typed`).

## Precision and known limits

- Use **float64** for the numerical solar propagation: in float32 it is off by up to about 10 % at 8 MeV (phases
  accumulated over the whole solar radius exceed float32 resolution). Earth and adiabatic calculations stay within
  about `1e-4` of float64 in float32 (worst case over a 10 000-path grid; see `examples/06_cpu_vs_gpu.py`).
- `PropagationMethod.ANALYTIC_PERTURBATIVE` evaluates one energy at a time; use `NUMERICAL` for energy grids.
- The Honda upstream tables are fetched from a third party; if the site is unreachable use
  `tpeanuts-data build honda-flux --from <directory-or-URL>`.

## Testing

```bash
pip install "tpeanuts[test]"
pytest -m "not slow and not gpu"          # the fast level, a couple of minutes
python tools/run_reference_tests.py       # the offline, deterministic reference level, with an archived report
```

[`tests/README.md`](tests/README.md) describes the levels, markers, tolerances and coverage policy.

## Citing

If you use tpeanuts in your work, please cite it using [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this
repository" button reads it) and, once available, the accompanying paper in *Computer Physics Communications*.

## Contributing

Bug reports, questions and pull requests are welcome; see [`CONTRIBUTING.md`](CONTRIBUTING.md). Please follow the
[code of conduct](CODE_OF_CONDUCT.md).

## Licence

The code is released under the GNU General Public Licence v3 or later ([`LICENSE`](LICENSE)). Third-party notices are in
[`NOTICE`](NOTICE) and the terms of each dataset in [`DATA_LICENSES.md`](DATA_LICENSES.md).

## Author

Juan Ramón Díaz Santos, with supervision by Roberto Ruiz de Austri Bazán and Michele Lucente.
Developed as part of an MSc thesis in Theoretical Physics at the Universitat de València.

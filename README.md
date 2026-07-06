# tpeanuts

Torch-based neutrino oscillation and flux propagation utilities, with a bundled
legacy `peanuts` compatibility package.

## Installation

Create or activate your Python environment, install the base dependencies, and
install the project in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

Optional workflows may also need `pymsis`, `MCEq`, and `crflux`.

## Layout

- `core/`: Hamiltonians, potentials, spectral helpers, evolution operators, and probabilities.
- `earth/`: Earth density, geometry, exposure, evolution, and integration utilities.
- `solar/`: Solar profiles, matter mixing, probabilities, validation, and IO.
- `atmosphere/`: Atmosphere geometry, density, propagation, and flux helpers.
- `mceq/`: MCEq configuration, solving, profile reconstruction, IO, and generation.
- `pipeline/`: Higher-level solar, Atmosphere, coherent, incoherent, and legacy pipelines.
- `peanuts/`: Legacy NumPy/Numba implementation kept for validation and compatibility.
- `tests/`: Executable test and diagnostic scripts.
- `data/`: Reference input data and generated flux data.
- `outputs/`: Generated figures, reports, and analysis artifacts.

Test figures are written under `outputs/tests/figures/`.

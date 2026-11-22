# Benchmarks

Comparative performance of tpeanuts against the original peanuts and nuSQuIDS. Correctness is checked in
[`validation/`](../validation); here only the cost is measured. The plan and its steps are in
[`doc/P.Paper/plan_benchmarks.md`](../doc/P.Paper/plan_benchmarks.md).

```
common/      shared harness: timer, environment record, result schema, scenarios
tpeanuts/    the tpeanuts scenarios, timed alone (the baseline)
peanuts/     tpeanuts vs the original peanuts
nusquids/    tpeanuts vs nuSQuIDS (WSL)
baselines/   versioned baselines (JSON), one per machine and scenario
results/     output of each run (not versioned)
```

This directory is not part of the installed package. `tools/bench_pipelines.py` and `reports/perf/` remain the
tpeanuts-only regression benchmark (before/after a refactor).

## Running the tpeanuts baseline

From the repository root:

```
python -m benchmarks.tpeanuts.run --smoke --label quick                      # reduced sizes, seconds
python -m benchmarks.tpeanuts.run --methods numerical analytic --threads 8   # the catalogue, CPU
python -m benchmarks.tpeanuts.run --device cuda --max-points 0               # the full grid on the GPU
```

Each case is timed in two phases (`build`: model, media and legs; `evaluate`: the whole grid) and written to
`results/<label>.json` (schema in `common/results.py`). Grids above `--max-points` (1e6 by default) are recorded as
`skipped`, and a method that cannot handle a size as `failed`, never dropped. The atmosphere scenario uses a synthetic
flux, so it needs no downloaded data and times only the propagation.

## Running the comparison with the original PEANUTS

The original PEANUTS (NumPy/Numba, a separate git repository) runs in its own environment, so the two backends only
exchange files. From the repository root:

```
python -m benchmarks.tpeanuts.export_reference                      # tpeanuts environment: the reference .npz
python -s -m benchmarks.peanuts.run --smoke --label quick           # peanuts environment: check, then time
python -s -m benchmarks.peanuts.run --max-points 100000 --label peanuts_cpu_float64
python -s -m benchmarks.peanuts.run --modes analytical numerical    # adds the Earth ODE mode (about 100 s per point)
```

The original is looked up from `--peanuts-path`, then `PEANUTS_PATH`, then a `PEANUTS` directory next to this repository;
its git commit is recorded in the result. It is not `v1_0/peanuts`, which is a modified fork. A case is timed only if it
agrees with the tpeanuts reference within its tolerance (`benchmarks/peanuts/run.py`, `TOLERANCES`, each with its
reason); a case that does not is recorded as `disagrees`. PEANUTS has no atmosphere (`unavailable`). Its Earth modes are
paired with the tpeanuts method that uses the same algorithm: `analytical` with `analytic_perturbative`, `numerical`
with `numerical`.

## Running the comparison with nuSQuIDS (WSL)

nuSQuIDS only installs inside WSL, and the repository is on a synchronised Windows drive that WSL writes to unreliably,
so the run happens on a native copy that a script refreshes and fingerprints:

```
python -m benchmarks.tpeanuts.export_reference                      # Windows, tpeanuts environment: the reference .npz
python -m benchmarks.nusquids.sync_wsl                              # copy the needed files to ~/tpeanuts_bench (WSL)
# then, in WSL:
source ~/.venvs/nusquids-py312/bin/activate
export NUSQUIDS_DATA_DIR=~/nusquids_data_fetched                    # the wheel ships no astro data (fetch_data)
cd ~/tpeanuts_bench && python -m benchmarks.nusquids.run --smoke --label quick
```

`sync_wsl` writes `MANIFEST.json` (SHA-256 of every file and of the tree, source commit, dirty flag), and the result
records it. Scenarios and pairing: `vacuum` and `earth` (PREM, `nue`) are paired with the tpeanuts `numerical` method;
`atmosphere` uses `EarthAtm` (`numu`), whose atmosphere is fixed at a 22 km production height, so it is checked against
tpeanuts computed with that single height (reference key `atmosphere_earthatm`), not the four-height average of the
tpeanuts baseline. `solar` and `solar_earth` are `unavailable`: tpeanuts integrates the 8B flux over its production
distribution in the Sun, whereas nuSQuIDS evolves from one production radius. The solver is used as a user uses it
(body created once, one track per angle, one evolution per energy), single-threaded; creating it is the `build` phase.

## Running on a cluster (HTCondor)

`benchmarks/condor/` holds what the Artemisa runs need; nothing in it is specific to one cluster except the resource lines.

1. **Environments** (one per backend, as above): `tpeanuts` with a CUDA torch that the node driver supports and
   `pip install -e .`; `peanuts` without torch; `nusquids` with the wheel, plus its astro data
   (`python -m validation.nusquids.fetch_data`). The cluster has internet access, so these installs can be done there.
2. **Code**: clone or `rsync` the repository (and the original `PEANUTS`) to a filesystem the workers see, then run
   `python -m benchmarks.nusquids.sync_wsl --manifest-only` there (or before uploading) so that every result carries the
   fingerprint of the code that produced it.
3. **Configure**: `cp benchmarks/condor/env.sh.example benchmarks/condor/env.sh` and edit the paths and the activation
   functions (`env.sh` is ignored by git).
4. **Render the submit files** from the specifications in `benchmarks/suites/<series>/<name>.yaml` (one YAML per benchmark:
   what is measured, the HTCondor resources, the order in `condor.after`, and where the CSV goes, by default next to the
   YAML). From the repository root:

   ```
   python benchmarks/condor/render.py --all --chain      # benchmarks/condor/generated/*.sub and submit_all.sh
   python benchmarks/condor/render.py --check            # exit 1 if a generated file no longer matches its YAML
   ```

   The generated files are not versioned; the YAML is the only description. DAGMan is not used: `submit_all.sh` submits one
   job, waits for it with `condor_wait`, checks that its output exists and goes on, so two measurements never share a
   machine and a failed job stops the sequence (an earlier output is kept as `<file>.previous`). The order is
   `0_reference/reference` (the tpeanuts reference `.npz`), then `tpeanuts_cpu_float64`, `tpeanuts_cuda_float64`,
   `tpeanuts_cuda_float32`, `peanuts_analytical_cpu_float64` and `nusquids_cpu_float64`. `peanuts_numerical_cpu_float64`
   (about 100 s per point) is `optional`: it is left out of the chain (`--include-optional` adds it) and is submitted alone
   with `condor_submit benchmarks/condor/generated/3_peanuts_peanuts_numerical_cpu_float64.sub` when wanted. One job by
   hand: `python benchmarks/condor/render.py 1_intrinsic/tpeanuts_cpu_float64` and `condor_submit` the file it writes.
5. **Collect**: the CSV files appear in `benchmarks/suites/<series>/` (shared filesystem); commit them, and read them with
   `python -m benchmarks.report --inputs benchmarks/suites/*/*.csv`. A result taken with a dirty checkout records
   `dirty = True` and the notebooks will say so.

Each result records the HTCondor job and machine attributes (`environment["scheduler"]`), the CPU model, memory and GPUs
(`environment["hardware"]`) and the code fingerprint (`environment["manifest"]`). The GPU nodes have driver 580 (CUDA 13.0), so a torch wheel built for CUDA 12.x (cu126) runs on them. The
schedd shows no run-time limit attribute, so the submit files do not set `+MaxRuntime`. On Artemisa the filesystem is shared with the workers (`should_transfer_files = NO`) and a GPU is requested with
`request_gpus = 1`.

## Reading the results

```
python -m benchmarks.report                                         # every JSON in results/, tables to stdout
python -m benchmarks.report --inputs a.json b.json --output-dir benchmarks/report --figures
```

`benchmarks/report.py` only reads the result files. It prints and optionally writes as CSV (and PNG with `--figures`)
the **speed-ups** of tpeanuts over each reference (matched on scenario, sizes and paired algorithm; one entry per
tpeanuts device, so GPU and CPU are separate and labelled), the **scaling exponents** (log-log slope of time against
points) and the **absences** (every record without a timing, with its status and reason). A pairing in which the two
sides do different amounts of work carries a caveat in the table (the nuSQuIDS atmosphere, where tpeanuts averages four
production heights).

## Protocol

- **Precision:** float64 in every backend that supports it; otherwise the result is labelled.
- **Repetitions:** one warm-up call and N repeats (7 by default); minimum, median and interquartile range are recorded.
- **Devices:** CUDA is synchronised around each call; peak CUDA memory and process RAM are recorded.
- **Threads:** CPU threads are fixed and recorded (`torch`, OpenMP/MKL, nuSQuIDS).
- **Phases:** object construction and evaluation are timed separately.
- **Environment:** every result carries versions, device, threads, git hash and date.
- **Fairness:** CPU against CPU and GPU against CPU are both reported and always labelled; results from different
  machines are never mixed.
- **Isolation:** each backend runs in its own process and environment and exchanges only JSON files; two packages are
  never imported in the same interpreter.
- **Correctness first:** a timing is only recorded for a case whose result agrees with the reference.

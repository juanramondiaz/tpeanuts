#!/bin/bash
# Run one benchmark backend inside an HTCondor job (or by hand on a login node for a short check).
#
# Usage:  run.sh <backend> [arguments of the runner...]
#   backend: export | tpeanuts | peanuts | nusquids
#
# Each backend uses its own environment, defined in benchmarks/condor/env.sh (copy env.sh.example and edit it); the
# runners exchange only files under benchmarks/results. BENCH_THREADS (set by the submit files from request_cpus) fixes the
# CPU threads of tpeanuts; PEANUTS and nuSQuIDS are single-threaded by construction and always get one.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${BENCH_ENV_FILE:-$HERE/env.sh}"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "run.sh: $ENV_FILE not found; copy env.sh.example to env.sh and edit it" >&2
    exit 2
fi
# shellcheck source=/dev/null
source "$ENV_FILE"

if [[ $# -lt 1 ]]; then
    echo "usage: run.sh <export|tpeanuts|peanuts|nusquids> [runner arguments...]" >&2
    exit 2
fi
backend="$1"
shift

cd "$BENCH_REPO"
export PYTHONPATH="$BENCH_REPO"
export PYTHONUNBUFFERED=1
threads="${BENCH_THREADS:-1}"

echo "== $(date -u +%FT%TZ) backend=$backend host=$(hostname) threads=$threads"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
fi

if [[ -n "${_CONDOR_MACHINE_AD:-}" && -r "$_CONDOR_MACHINE_AD" ]]; then
    grep -i -E '^(gpus|cuda|assignedgpus|totalgpus|machine) ?' "$_CONDOR_MACHINE_AD" | head -30 || true
fi

case "$backend" in
    export)
        bench_activate_tpeanuts
        exec python -s -m benchmarks.tpeanuts.export_reference "$@"
        ;;
    tpeanuts)
        bench_activate_tpeanuts
        exec python -s -m benchmarks.tpeanuts.run --threads "$threads" "$@"
        ;;
    peanuts)
        bench_activate_peanuts
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
        exec python -s -m benchmarks.peanuts.run "$@"
        ;;
    nusquids)
        bench_activate_nusquids
        exec python -m benchmarks.nusquids.run --threads 1 "$@"
        ;;
    *)
        echo "run.sh: unknown backend '$backend'" >&2
        exit 2
        ;;
esac

#!/bin/bash
# Create the three benchmark environments (and fetch what they need) on the cluster. Idempotent: an existing
# environment or checkout is kept unless --recreate is given.
#
# Usage (from the repository root, on a login node; Artemisa has internet access):
#   bash benchmarks/condor/setup_envs.sh [--only tpeanuts|peanuts|nusquids|data] [--recreate] [--no-verify]
#
# What it does, in order:
#   1. miniconda in $CONDA_DIR, if conda is not there yet
#   2. env `tpeanuts`: Python 3.12, torch (CUDA 12.6 wheel: the nodes have driver 580 = CUDA 13.0), this repository
#      installed editable (src/ layout)
#   3. env `peanuts`: Python 3.12, numpy/scipy/numba/pandas/mpmath, NO torch; plus a clone of the original PEANUTS
#   4. env `nusquids`: Python 3.12 (the wheel is py3.12 manylinux), nusquids + numpy, plus its astro data
#   5. benchmarks/condor/env.sh written from the paths used here (only if it does not exist)
#   6. a short import check per environment (MANIFEST.json is written later by `run.sh export`, after the reference)
#
# Variables (override from the shell): CONDA_DIR, PEANUTS_DIR, PEANUTS_URL, NUSQUIDS_DATA_DIR, TORCH_INDEX, PYTHON_VERSION.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
PEANUTS_DIR="${PEANUTS_DIR:-$HOME/PEANUTS}"
PEANUTS_URL="${PEANUTS_URL:-https://github.com/michelelucente/PEANUTS}"
NUSQUIDS_DATA_DIR="${NUSQUIDS_DATA_DIR:-$HOME/nusquids_data}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

only="all"; recreate=0; verify=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) only="$2"; shift 2 ;;
        --recreate) recreate=1; shift ;;
        --no-verify) verify=0; shift ;;
        -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "setup_envs.sh: unknown argument '$1'" >&2; exit 2 ;;
    esac
done
want() { [[ "$only" == "all" || "$only" == "$1" ]]; }

log() { echo "== [$(date +%H:%M:%S)] $*"; }

# ---- 1. conda ----------------------------------------------------------------------------------------------------
if [[ ! -x "$CONDA_DIR/bin/conda" ]]; then
    log "installing miniconda in $CONDA_DIR"
    installer="$(mktemp -d)/miniconda.sh"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$installer"
    bash "$installer" -b -p "$CONDA_DIR"
fi
# shellcheck source=/dev/null
source "$CONDA_DIR/etc/profile.d/conda.sh"
# Accept the channel terms non-interactively on conda versions that require it (harmless otherwise).
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

make_env() {   # make_env <name>: create the environment (python only), recreating it on request
    local name="$1"
    if conda env list | awk '{print $1}' | grep -qx "$name"; then
        if [[ $recreate -eq 1 ]]; then
            log "removing environment $name"
            conda env remove -y -n "$name"
        else
            log "environment $name exists (kept; --recreate to rebuild)"
            return 0
        fi
    fi
    log "creating environment $name (python $PYTHON_VERSION)"
    conda create -y -n "$name" "python=$PYTHON_VERSION" pip
}

# ---- 2. tpeanuts -------------------------------------------------------------------------------------------------
if want tpeanuts; then
    make_env tpeanuts
    conda activate tpeanuts
    export PYTHONNOTUSERSITE=1          # never pick up ~/.local packages (they break the torch/CUDA stack)
    log "tpeanuts: torch from $TORCH_INDEX"
    python -m pip install --upgrade pip
    python -m pip install torch --index-url "$TORCH_INDEX"
    log "tpeanuts: repository (editable) from $REPO"
    python -m pip install -e "$REPO"
    conda deactivate
fi

# ---- 3. peanuts --------------------------------------------------------------------------------------------------
if want peanuts; then
    make_env peanuts
    conda activate peanuts
    export PYTHONNOTUSERSITE=1
    log "peanuts: numerical stack (no torch)"
    python -m pip install --upgrade pip
    python -m pip install numpy scipy numba pandas mpmath
    conda deactivate
    if [[ -d "$PEANUTS_DIR/.git" ]]; then
        log "PEANUTS checkout exists in $PEANUTS_DIR ($(git -C "$PEANUTS_DIR" rev-parse --short HEAD)); kept"
    else
        log "cloning the original PEANUTS into $PEANUTS_DIR"
        git clone "$PEANUTS_URL" "$PEANUTS_DIR"
    fi
fi

# ---- 4. nusquids -------------------------------------------------------------------------------------------------
if want nusquids || want data; then
    if want nusquids; then
        make_env nusquids
    fi
    conda activate nusquids
    export PYTHONNOTUSERSITE=1
    if want nusquids; then
        log "nusquids: wheel"
        python -m pip install --upgrade pip
        python -m pip install numpy nusquids
    fi
    # The wheel ships no astro data; fetch_data extracts it from the PyPI sdist (needs only the stdlib).
    log "nusquids: astro data in $NUSQUIDS_DATA_DIR"
    (cd "$REPO" && PYTHONPATH="$REPO" python -m validation.nusquids.fetch_data "$NUSQUIDS_DATA_DIR")
    conda deactivate
fi

# ---- 5. env.sh ---------------------------------------------------------------------------------------------------
ENV_FILE="$HERE/env.sh"
if [[ "$only" == "all" ]]; then
    if [[ -f "$ENV_FILE" ]]; then
        log "$ENV_FILE exists; not overwritten (compare with env.sh.example if a path changed)"
    else
        log "writing $ENV_FILE"
        cat > "$ENV_FILE" <<EOF
# Written by setup_envs.sh. Sourced by run.sh.
export BENCH_REPO="$REPO"
export PEANUTS_PATH="$PEANUTS_DIR"
export NUSQUIDS_DATA_DIR="$NUSQUIDS_DATA_DIR"

bench_activate_tpeanuts() { source "$CONDA_DIR/etc/profile.d/conda.sh"; conda activate tpeanuts; export PYTHONNOTUSERSITE=1; }
bench_activate_peanuts()  { source "$CONDA_DIR/etc/profile.d/conda.sh"; conda activate peanuts;  export PYTHONNOTUSERSITE=1; }
bench_activate_nusquids() { source "$CONDA_DIR/etc/profile.d/conda.sh"; conda activate nusquids; export PYTHONNOTUSERSITE=1; }
EOF
    fi

    # MANIFEST.json (the code fingerprint) includes the reference .npz, which does not exist yet: `run.sh export`
    # (the first job of the chain) writes both, in that order.
fi

# ---- verification ------------------------------------------------------------------------------------------------
if [[ $verify -eq 1 ]]; then
    log "verification (a login node has no GPU: CUDA is only reported here, tested on a GPU node)"
    if want tpeanuts; then
        conda activate tpeanuts
        (cd "$REPO" && python -s - <<'PY'
import torch, tpeanuts
print("tpeanuts", getattr(tpeanuts, "__version__", "?"), "| torch", torch.__version__,
      "| built for CUDA", torch.version.cuda, "| cuda available here:", torch.cuda.is_available())
PY
        )
        conda deactivate
    fi
    if want peanuts; then
        conda activate peanuts
        PEANUTS_PATH="$PEANUTS_DIR" python -s - <<'PY'
import importlib.util, os, sys
for mod in ("numpy", "scipy", "numba", "pandas", "mpmath"):
    __import__(mod)
print("peanuts env: numerical stack ok; torch absent:", importlib.util.find_spec("torch") is None)
sys.path.insert(0, os.environ["PEANUTS_PATH"])
from peanuts.earth import EarthDensity, Pearth      # noqa: F401
from peanuts.solar import SolarModel                # noqa: F401
print("original PEANUTS imports from", os.environ["PEANUTS_PATH"])
PY
        conda deactivate
    fi
    if want nusquids; then
        conda activate nusquids
        NUSQUIDS_DATA_DIR="$NUSQUIDS_DATA_DIR" python -s -c "import nuSQuIDS; print('nuSQuIDS imports; data dir:', '$NUSQUIDS_DATA_DIR')"
        conda deactivate
    fi
fi

log "done. Next: bash benchmarks/condor/run.sh export  (then a --smoke run), see benchmarks/README.md"

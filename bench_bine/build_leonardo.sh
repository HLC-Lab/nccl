#!/bin/bash
# Build BOTH NCCL variants (original PAT baseline + Bine) and nccl-tests on
# Leonardo Booster (A100, sm_80). Run from anywhere: bash bench_bine/build_leonardo.sh
#
# Layout produced (siblings of the repo):
#   <repo>/build/lib/libnccl.so.2            <- Bine   (this checkout)
#   <repo>/../nccl-baseline/build/lib/...     <- PAT    (clean checkout @ BASELINE_REF)
#   <repo>/../nccl-tests/build/all_gather_perf
# Both libnccl share SONAME libnccl.so.2, so one nccl-tests binary runs against
# either by swapping LD_LIBRARY_PATH (see run_compare.slurm).
set -euo pipefail

module purge
module load cuda hpcx-mpi
# If the NCCL/nccl-tests build complains about the C++ compiler being too old,
# also `module load gcc/<recent>` (the system gcc on the login nodes may be old).

export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}
export MPI_HOME=${MPI_HOME:-$(dirname "$(dirname "$(command -v mpicc)")")}
GENCODE="-gencode=arch=compute_80,code=sm_80"     # A100 = compute capability 8.0

REPO="$(cd "$(dirname "$0")/.." && pwd)"          # the Bine checkout
BASELINE_REF="${BASELINE_REF:-5067397}"           # pre-Bine commit == original PAT
BASELINE_DIR="$REPO/../nccl-baseline"

# --- baseline (original PAT) in a detached worktree ---
cd "$REPO"
if [ ! -d "$BASELINE_DIR" ]; then
  git worktree add --detach "$BASELINE_DIR" "$BASELINE_REF"
fi
cd "$BASELINE_DIR"
make -j"$(nproc)" src.build CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="$GENCODE"
echo ">>> baseline (PAT) NCCL: $BASELINE_DIR/build/lib"

# --- Bine (this checkout, with the uncommitted/branch changes) ---
cd "$REPO"
make -j"$(nproc)" src.build CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="$GENCODE"
echo ">>> bine NCCL: $REPO/build/lib"

# --- nccl-tests (built once; libnccl swapped at runtime) ---
cd "$REPO/.."
[ -d nccl-tests ] || git clone https://github.com/NVIDIA/nccl-tests
cd nccl-tests
make -j"$(nproc)" MPI=1 MPI_HOME="$MPI_HOME" NCCL_HOME="$REPO/build" \
     CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="$GENCODE"
echo ">>> nccl-tests: $PWD/build"
echo
echo "Done. Edit ROOT in run_compare.slurm if the repos are not under \$HOME/work, then sbatch it."

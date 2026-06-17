#!/bin/bash
# AllGather: Bine vs original PAT vs Ring in ONE allocation (three srun steps).
# NOTE: three back-to-back srun steps can hit launch timeouts at scale -- prefer
# submit_all.sh (one job per algorithm) for >2 nodes. Kept for same-node runs.
# Submit: sbatch bench_bine/run_compare.slurm
#SBATCH -A IscrB_DEFT
#SBATCH -p boost_usr_prod
#SBATCH -N 16                        # nodes == ranks == GPUs. POWER OF TWO (Bine requirement).
#SBATCH --ntasks-per-node=1          # 1 rank/node -> nNodes==nRanks (PAT/Bine requirement)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH -t 00:30:00
#SBATCH -o ag_compare_%j.out

module purge
module load cuda hpcx-mpi

ROOT=/leonardo_scratch/large/userexternal/$USER
BINE_LIB=$ROOT/nccl/build/lib
BASE_LIB=$ROOT/nccl-baseline/build/lib
TESTS=$ROOT/nccl-tests/build/all_gather_perf
SWEEP="-b 8 -e 1G -f 2 -g 1 -c 1"                # -c 1 = correctness check every size

export NCCL_PROTO=Simple
export NCCL_DEBUG=WARN                            # flip to INFO once to confirm algo+IB selection

run() {  # label  libdir  algo  -- all three steps share THIS single allocation (same nodes)
  echo "############## $1 (NCCL_ALGO=$3) ##############"
  local try=0
  until LD_LIBRARY_PATH=$2:$LD_LIBRARY_PATH NCCL_ALGO=$3 \
          srun --cpu-bind=cores --mpi=pmix $TESTS $SWEEP; do
    try=$((try+1))
    [ $try -ge 3 ] && { echo "!! $1: srun failed after $try attempts, skipping"; break; }
    echo "!! $1: srun launch failed, retry $try/3 after 20s..."; sleep 20
  done
  echo
  sleep 5   # let slurmstepd from the previous step tear down before the next launch
}

run BINE $BINE_LIB PAT     # this build: PAT == Bine schedule
run PAT  $BASE_LIB PAT     # baseline build: original PAT
run RING $BASE_LIB RING    # Ring (unchanged; baseline or bine identical)

# Compare the 'busbw' (GB/s) column across the three blocks at matching sizes.
# 'Out of bounds values: 0 OK' on the BINE block = correctness pass.
# If srun rejects --mpi=pmix, try --mpi=pmi2.

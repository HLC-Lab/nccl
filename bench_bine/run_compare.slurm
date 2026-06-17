#!/bin/bash
# AllGather: Bine vs original PAT vs Ring on Leonardo Booster (1 GPU/node).
# Submit: sbatch bench_bine/run_compare.slurm
#SBATCH -A <YOUR_ACCOUNT>            # CINECA project, e.g. -A IscrC_xxxxx
#SBATCH -p boost_usr_prod
#SBATCH --qos boost_qos_dbg          # debug: <=2 nodes, 30 min. Remove for >2 nodes.
#SBATCH -N 2                         # nodes == ranks == GPUs. POWER OF TWO (Bine requirement).
#SBATCH --ntasks-per-node=1          # 1 rank/node -> nNodes==nRanks (PAT/Bine requirement)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -o ag_compare_%j.out

module purge
module load cuda/12.1 gcc/12.2.0 openmpi/4.1.6--gcc--12.2.0

ROOT=$HOME/work                                  # adjust if repos live elsewhere
BINE_LIB=$ROOT/nccl/build/lib
BASE_LIB=$ROOT/nccl-baseline/build/lib
TESTS=$ROOT/nccl-tests/build/all_gather_perf
SWEEP="-b 8 -e 1G -f 2 -g 1 -c 1"                # -c 1 = correctness check every size

export NCCL_PROTO=Simple
export NCCL_DEBUG=WARN                            # flip to INFO once to confirm algo+IB selection

run() {  # label  libdir  algo
  echo "############## $1 (NCCL_ALGO=$3) ##############"
  LD_LIBRARY_PATH=$2:$LD_LIBRARY_PATH NCCL_ALGO=$3 \
    srun --cpu-bind=cores --mpi=pmix $TESTS $SWEEP
  echo
}

run BINE $BINE_LIB PAT     # this build: PAT == Bine schedule
run PAT  $BASE_LIB PAT     # baseline build: original PAT
run RING $BASE_LIB RING    # Ring (unchanged; baseline or bine identical)

# Compare the 'busbw' (GB/s) column across the three blocks at matching sizes.
# 'Out of bounds values: 0 OK' on the BINE block = correctness pass.
# If srun rejects --mpi=pmix, try --mpi=pmi2.

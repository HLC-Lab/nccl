#!/bin/bash
# AllGather benchmark on Leonardo Booster. NCCL_ALGO=PAT runs the Bine schedule.
# Submit: sbatch bench_bine/run_allgather.slurm
#SBATCH -A <YOUR_ACCOUNT>            # CINECA project account, e.g. -A IscrC_xxxxx
#SBATCH -p boost_usr_prod
#SBATCH --qos boost_qos_dbg          # debug QOS: <=2 nodes, 30 min. Remove for >2 nodes (normal QOS).
#SBATCH -N 2                         # nodes == ranks == GPUs. MUST be a power of two for Bine.
#SBATCH --ntasks-per-node=1          # 1 rank per node  ->  nNodes==nRanks  (PAT/Bine requirement)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -o ag_bine_%j.out

module purge
module load cuda/12.1 gcc/12.2.0 openmpi/4.1.6--gcc--12.2.0

ROOT=$HOME/work                      # adjust if the repos live elsewhere
export NCCL_HOME=$ROOT/nccl/build
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH

# --- algorithm under test ---
export NCCL_ALGO=PAT                 # this build: PAT == Bine AllGather schedule
export NCCL_PROTO=Simple

# --- visibility: confirm PAT was actually picked and IB (not socket) is used ---
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING,NET

# all_gather_perf: -b/-e size sweep, -f 2 doubling, -g 1 GPU/rank, -c 1 correctness check.
srun --cpu-bind=cores --mpi=pmix \
  $ROOT/nccl-tests/build/all_gather_perf -b 8 -e 1G -f 2 -g 1 -c 1

# If srun rejects --mpi=pmix, try --mpi=pmi2.
# Read the 'busbw' (GB/s) column for bandwidth; 'Out of bounds values: 0 OK' = correctness pass.

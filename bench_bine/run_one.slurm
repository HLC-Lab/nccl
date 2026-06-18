#!/bin/bash
# Single-algorithm AllGather run (one srun step per job — robust at scale).
# Submit three times with different ALGO/LIB (see submit_all.sh):
#   sbatch -J BINE -o ag_bine_%j.out --export=ALL,ALGO=PAT,LIB=$ROOT/nccl/build/lib          run_one.slurm
#   sbatch -J PAT  -o ag_pat_%j.out  --export=ALL,ALGO=PAT,LIB=$ROOT/nccl-baseline/build/lib  run_one.slurm
#   sbatch -J RING -o ag_ring_%j.out --export=ALL,ALGO=RING,LIB=$ROOT/nccl-baseline/build/lib run_one.slurm
#SBATCH -A IscrB_DEFT
#SBATCH -p boost_usr_prod
#SBATCH -N 16                        # POWER OF TWO for Bine.
#SBATCH --ntasks-per-node=1          # 1 GPU/node -> nNodes==nRanks
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH -t 00:30:00

module purge
module load cuda hpcx-mpi

ROOT=/leonardo_scratch/large/userexternal/$USER
: "${ALGO:=PAT}"
: "${LIB:=$ROOT/nccl/build/lib}"
export LD_LIBRARY_PATH=$LIB:$LD_LIBRARY_PATH
export NCCL_ALGO=$ALGO
export NCCL_PROTO=Simple

# Tag the run so the output is self-describing for extract_results.py.
echo "############## ${SLURM_JOB_NAME} (NCCL_ALGO=$ALGO  LIB=$LIB  N=$SLURM_NNODES) ##############"
srun $ROOT/nccl-tests/build/all_gather_perf -b 8 -e 1G -f 2 -g 1 -c 1

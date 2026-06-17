#!/bin/bash
# Submit the three AllGather configs as SEPARATE jobs (avoids multi-srun-step
# launch timeouts). Pass node count as $1 (power of two), default 16.
#   bash bench_bine/submit_all.sh        # 16 nodes
#   bash bench_bine/submit_all.sh 8
set -euo pipefail
N=${1:-16}
ROOT=/leonardo_scratch/large/userexternal/$USER
S=bench_bine/run_one.slurm

sbatch -N "$N" -J BINE -o "ag_bine_${N}n_%j.out" \
  --export=ALL,ALGO=PAT,LIB=$ROOT/nccl/build/lib            "$S"
sbatch -N "$N" -J PAT  -o "ag_pat_${N}n_%j.out"  \
  --export=ALL,ALGO=PAT,LIB=$ROOT/nccl-baseline/build/lib   "$S"
sbatch -N "$N" -J RING -o "ag_ring_${N}n_%j.out" \
  --export=ALL,ALGO=RING,LIB=$ROOT/nccl-baseline/build/lib  "$S"
echo "submitted BINE/PAT/RING at N=$N"

#!/bin/bash
# Final AllGather comparison: Bine vs PAT vs Ring, same allocation (fair).
#
#   Bine = $ROOT/nccl           (this work)      NCCL_ALGO=PAT
#   PAT  = $ROOT/nccl-baseline  (upstream 5067397) NCCL_ALGO=PAT
#   Ring = $ROOT/nccl-baseline  (upstream)        NCCL_ALGO=Ring
#
# Usage, inside an salloc (uses the allocated nodes):
#   bash final_compare.sh                 # N=all alloc nodes, 3 reps, -n 50, default channels
#   bash final_compare.sh 16 5 50         # N=16, 5 reps, 50 iters
#   NCCL_FORCE_CH=16 bash final_compare.sh   # force 16 channels for all three
#
# Or submit with sbatch (set -N / --ntasks-per-node in the sbatch line).
set -u

ROOT=${ROOT:-/leonardo_scratch/large/userexternal/$USER}
N=${1:-${SLURM_NNODES:-8}}
REPS=${2:-3}
ITERS=${3:-50}
TEST=$ROOT/nccl-tests/build/all_gather_perf
ARGS="-b 8 -e 1G -f 2 -g 1 -c 1 -n $ITERS"

chenv=""
if [ "${NCCL_FORCE_CH:-0}" -gt 0 ]; then
  chenv="NCCL_MIN_NCHANNELS=$NCCL_FORCE_CH NCCL_MAX_NCHANNELS=$NCCL_FORCE_CH"
fi

OUT=$ROOT/nccl/bench_bine/final_$(date +%Y%m%d_%H%M%S)_N${N}_ch${NCCL_FORCE_CH:-def}
mkdir -p "$OUT"

run() { # name lib algo rep
  local name=$1 lib=$2 algo=$3 rep=$4
  local log="$OUT/${name}_rep${rep}.log"
  env $chenv NCCL_ALGO=$algo LD_LIBRARY_PATH=$ROOT/$lib/build/lib:${LD_LIBRARY_PATH:-} \
    srun -N "$N" -n "$N" --ntasks-per-node=1 --unbuffered --export=ALL "$TEST" $ARGS > "$log" 2>&1
  local avg wrong
  avg=$(grep "Avg bus" "$log" | awk '{print $NF}')
  wrong=$(awk '$3=="float"{w+=$9+$13} END{print w+0}' "$log")
  echo "  $name rep$rep: avg busbw=${avg:-FAIL}  #wrong=$wrong"
}

echo "=== Final compare: N=$N reps=$REPS iters=$ITERS channels=${NCCL_FORCE_CH:-default} ==="
echo "=== nodes: $(echo ${SLURM_NODELIST:-?}) ==="
echo "=== logs -> $OUT ==="
for r in $(seq 1 "$REPS"); do
  echo "--- rep $r ---"
  run Bine nccl          PAT  "$r"
  run PAT  nccl-baseline PAT  "$r"
  run Ring nccl-baseline Ring "$r"
done

echo ""
echo "=== SUMMARY: out-of-place busbw (GB/s) per size, mean over $REPS reps ==="
# Portable awk (no gawk/python needed). Algo from filename; busbw_oop is field 8.
awk '
  FNR==1 { fn=FILENAME; sub(/.*\//,"",fn); sub(/_rep.*/,"",fn); algo=fn }
  $3=="float" && ($1+0)>0 {
    sz=$1+0; k=algo SUBSEP sz; sum[k]+=$8; cnt[k]++;
    if(!(sz in seen)){ seen[sz]=1; szl[++nsz]=sz }
  }
  function av(a,z,   key){ key=a SUBSEP z; return (cnt[key]>0 ? sum[key]/cnt[key] : 0) }
  END {
    for(i=2;i<=nsz;i++){ v=szl[i]; j=i-1; while(j>=1 && szl[j]>v){ szl[j+1]=szl[j]; j-- } szl[j+1]=v }
    printf "%14s %8s %8s %8s %9s %10s\n","size(B)","Ring","PAT","Bine","Bine/PAT","Bine/Ring"
    for(i=1;i<=nsz;i++){ z=szl[i]; r=av("Ring",z); p=av("PAT",z); b=av("Bine",z)
      printf "%14d %8.2f %8.2f %8.2f %9.2f %10.2f\n", z, r, p, b, (p>0?b/p:0), (r>0?b/r:0)
    }
  }
' "$OUT"/*.log | tee "$OUT/summary.txt"

echo ""
echo "Done. Full logs + summary.txt in: $OUT"

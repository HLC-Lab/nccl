#!/bin/bash
# FAIR best-of-breed comparison of Bine vs PAT vs Ring for AllGather.
#
# The auto-channel runs were biased: the Bine build carries a channel floor (2->16)
# that the baseline (PAT/Ring) build does not, so they compared tuned-Bine vs
# default-Ring/PAT. This removes that: it sweeps the channel count for ALL THREE in ONE
# allocation (NCCL_MIN=MAX=C forces exactly C channels and neutralizes the Bine floor,
# which is clamped by ncclMaxNchannels()), then reports EACH ALGORITHM AT ITS OWN BEST
# CHANNEL COUNT per size. The post-hoc best-C pick is optimistic but symmetric -> fair.
#
# Usage (inside ONE salloc; do NOT split across allocations -- placement variance):
#   bash bench_bine/fair_envelope.sh 128 2 50
#   CHANS="2 8 16 32" bash bench_bine/fair_envelope.sh 128 3 50
set -u

ROOT=${ROOT:-/leonardo_scratch/large/userexternal/$USER}
N=${1:-${SLURM_NNODES:-8}}
REPS=${2:-2}
ITERS=${3:-50}
CHANS=${CHANS:-"2 4 8 16 32"}
TEST=$ROOT/nccl-tests/build/all_gather_perf
ARGS="-b 8 -e 1G -f 2 -g 1 -c 1 -n $ITERS"
OUT=$ROOT/nccl/bench_bine/fair_$(date +%Y%m%d_%H%M%S)_N${N}
mkdir -p "$OUT"

run() { # name lib algo C rep
  local name=$1 lib=$2 algo=$3 C=$4 rep=$5 log="$OUT/${name}_c${C}_rep${rep}.log"
  env NCCL_MIN_NCHANNELS=$C NCCL_MAX_NCHANNELS=$C NCCL_ALGO=$algo \
    LD_LIBRARY_PATH=$ROOT/$lib/build/lib:${LD_LIBRARY_PATH:-} \
    srun -N "$N" -n "$N" --ntasks-per-node=1 --unbuffered --export=ALL "$TEST" $ARGS > "$log" 2>&1
  local wrong; wrong=$(awk '$3=="float"{w+=$9+$13} END{print w+0}' "$log")
  [ "${wrong:-0}" != 0 ] && echo "    !! $name c$C rep$rep: #wrong=$wrong"
}

echo "=== FAIR envelope: N=$N reps=$REPS iters=$ITERS  channels swept: $CHANS ==="
echo "=== nodes: ${SLURM_NODELIST:-?} ==="
echo "=== logs -> $OUT ==="
for r in $(seq 1 "$REPS"); do
  for C in $CHANS; do
    echo "-- rep $r, $C channels --"
    run Bine nccl          PAT  "$C" "$r"
    run PAT  nccl-baseline PAT  "$C" "$r"
    run Ring nccl-baseline Ring "$C" "$r"
  done
done

echo ""
echo "=== BEST-OF-BREED: each algorithm at its OWN best channel count per size (mean/$REPS reps) ==="
awk '
  # filename: <algo>_c<C>_rep<r>.log
  FNR==1 { f=FILENAME; sub(/.*\//,"",f); sub(/\.log$/,"",f)
           algo=f; sub(/_c.*/,"",algo)
           cc=f; sub(/.*_c/,"",cc); sub(/_rep.*/,"",cc) }
  $3=="float" && ($1+0)>0 {
    z=$1+0; k=algo SUBSEP cc SUBSEP z; s[k]+=$8; n[k]++
    if(!(z in szseen)){szseen[z]=1; szl[++nsz]=z}
    if(!(cc in cseen)){cseen[cc]=1; cl[++ncl]=cc}
  }
  function mean(a,c,z,  k){ k=a SUBSEP c SUBSEP z; return n[k]>0 ? s[k]/n[k] : -1 }
  function bestC(a,z,  i,m,bm,bc){ bm=-1; bc=0
    for(i=1;i<=ncl;i++){ m=mean(a,cl[i],z); if(m>bm){bm=m; bc=cl[i]} }
    BM=bm; BC=bc; return bm }
  END {
    for(i=2;i<=nsz;i++){v=szl[i];j=i-1;while(j>=1&&szl[j]>v){szl[j+1]=szl[j];j--}szl[j+1]=v}
    printf "%12s  %14s %14s %14s   %-6s %s\n","size(B)","Ring","PAT","Bine","best","(ch: R/P/B)"
    for(si=1;si<=nsz;si++){ z=szl[si]
      r=bestC("Ring",z); rc=BC; p=bestC("PAT",z); pc=BC; b=bestC("Bine",z); bc=BC
      win = (b>=r && b>=p) ? "BINE" : (r>=p ? "Ring" : "PAT")
      printf "%12d  %8.2f@%-4s %8.2f@%-4s %8.2f@%-4s   %-6s (%s/%s/%s)\n",
             z, r,rc, p,pc, b,bc, win, rc,pc,bc
    }
  }
' "$OUT"/*.log | tee "$OUT/envelope.txt"

echo ""
echo "Each cell = that algorithm's BEST busbw over the swept channel counts (the @C shows"
echo "which count won). 'best' = fastest algorithm at that size when each is optimally"
echo "channelled. This is the fair algorithm-vs-algorithm envelope. Full logs in: $OUT"

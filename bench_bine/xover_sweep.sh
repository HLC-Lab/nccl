#!/bin/bash
# Sweep NCCL_BINE_XOVER to place the Bine AllGather butterfly/relay crossover.
# ONE build: the Bine lib ($ROOT/nccl, NCCL_ALGO=PAT) is run at several XOVER values
# (per-channel per-rank bytes) and a single per-size busbw table is printed so the
# crossover is obvious. XOVER=0 => pure relay, XOVER=2000000000 => pure butterfly (the
# two envelopes); the intermediate values are candidate thresholds. PAT (upstream
# baseline) is run once per rep as a reference column.
#
# Usage (inside an salloc; forces channels like final_compare.sh):
#   NCCL_FORCE_CH=16 bash bench_bine/xover_sweep.sh                 # N=alloc, 2 reps, 20 iters
#   NCCL_FORCE_CH=16 bash bench_bine/xover_sweep.sh 64 3 50
#   XOVERS="0 16384 65536 2000000000" NCCL_FORCE_CH=16 bash bench_bine/xover_sweep.sh 64 2 20
set -u

ROOT=${ROOT:-/leonardo_scratch/large/userexternal/$USER}
N=${1:-${SLURM_NNODES:-8}}
REPS=${2:-2}
ITERS=${3:-20}
TEST=$ROOT/nccl-tests/build/all_gather_perf
ARGS="-b 8 -e 1G -f 2 -g 1 -c 1 -n $ITERS"
# 0 = relay envelope, 2000000000 = butterfly envelope, rest = candidate crossovers.
XOVERS=${XOVERS:-"0 16384 32768 65536 131072 2000000000"}

chenv=""
if [ "${NCCL_FORCE_CH:-0}" -gt 0 ]; then
  chenv="NCCL_MIN_NCHANNELS=$NCCL_FORCE_CH NCCL_MAX_NCHANNELS=$NCCL_FORCE_CH"
fi

OUT=$ROOT/nccl/bench_bine/xover_$(date +%Y%m%d_%H%M%S)_N${N}_ch${NCCL_FORCE_CH:-def}
mkdir -p "$OUT"

srun_one() { # logfile  env-assignments... -- lib
  local log=$1; shift
  local lib=$1; shift
  env "$@" LD_LIBRARY_PATH="$lib":${LD_LIBRARY_PATH:-} \
    srun -N "$N" -n "$N" --ntasks-per-node=1 --unbuffered --export=ALL "$TEST" $ARGS > "$log" 2>&1
}

runbine() { # xover rep
  local x=$1 rep=$2 log="$OUT/x${1}_rep${2}.log"
  srun_one "$log" "$ROOT/nccl/build/lib" $chenv NCCL_BINE_XOVER=$x NCCL_ALGO=PAT
  local wrong; wrong=$(awk '$3=="float"{w+=$9+$13} END{print w+0}' "$log")
  local avg;   avg=$(grep "Avg bus" "$log" | awk '{print $NF}')
  echo "  XOVER=$x rep$rep: avg=${avg:-FAIL} #wrong=${wrong}$( [ "${wrong:-0}" != 0 ] && echo '  <<< WRONG' )"
}

runpat() { # rep
  local rep=$1 log="$OUT/pat_rep${1}.log"
  if [ ! -e "$ROOT/nccl-baseline/build/lib/libnccl.so.2" ]; then return; fi
  srun_one "$log" "$ROOT/nccl-baseline/build/lib" $chenv NCCL_ALGO=PAT
}

echo "=== NCCL_BINE_XOVER sweep: N=$N reps=$REPS iters=$ITERS channels=${NCCL_FORCE_CH:-default} ==="
echo "=== XOVERS: $XOVERS   (0 = relay envelope, 2000000000 = butterfly envelope) ==="
echo "=== nodes: ${SLURM_NODELIST:-?} ==="
echo "=== logs -> $OUT ==="
for r in $(seq 1 "$REPS"); do
  echo "--- rep $r ---"
  for x in $XOVERS; do runbine "$x" "$r"; done
  runpat "$r"
done

echo ""
echo "=== out-of-place busbw (GB/s) per size, mean over $REPS reps; * = best XOVER at that size ==="
awk -v xovers="$XOVERS" '
  FNR==1 { fn=FILENAME; sub(/.*\//,"",fn); sub(/_rep.*/,"",fn); tag=fn }   # tag: x<val> | pat
  $3=="float" && ($1+0)>0 {
    sz=$1+0; k=tag SUBSEP sz; sum[k]+=$8; cnt[k]++;
    if(!(sz in seen)){seen[sz]=1; szl[++nsz]=sz}
  }
  function av(t,z,  key){key=t SUBSEP z; return cnt[key]>0?sum[key]/cnt[key]:0}
  function lab(v){ if(v==0)return "relay"; if(v+0>=2000000000)return "bfly"; return "x"v }
  END{
    nx=split(xovers,xv," ")
    for(i=2;i<=nsz;i++){v=szl[i];j=i-1;while(j>=1&&szl[j]>v){szl[j+1]=szl[j];j--}szl[j+1]=v}
    printf "%12s","size(B)"
    for(i=1;i<=nx;i++) printf " %9s",lab(xv[i])
    printf " %9s %11s\n","PAT","best/PAT"
    for(s=1;s<=nsz;s++){ z=szl[s]; printf "%12d",z
      best=-1; for(i=1;i<=nx;i++){b=av("x"xv[i],z); if(b>best){best=b;bx=xv[i]}}
      for(i=1;i<=nx;i++){ b=av("x"xv[i],z); printf " %8.2f%s",b,(xv[i]==bx?"*":" ") }
      p=av("pat",z); printf " %9.2f %11.2f\n",p,(p>0?best/p:0)
    }
    printf "%12s","AVG"
    for(i=1;i<=nx;i++){ t="x"xv[i]; sm=0;c=0; for(s=1;s<=nsz;s++){b=av(t,szl[s]); if(b>0){sm+=b;c++}} printf " %9.2f",(c>0?sm/c:0) }
    sm=0;c=0; for(s=1;s<=nsz;s++){b=av("pat",szl[s]); if(b>0){sm+=b;c++}} printf " %9.2f\n",(c>0?sm/c:0)
  }
' "$OUT"/*.log | tee "$OUT/summary.txt"

echo ""
echo "READ: at each size the * marks the best XOVER. relay(0) should win large, bfly small;"
echo "the crossover is where they swap. Pick the single finite XOVER whose column best tracks"
echo "the per-size max AND has the best AVG -> that is your NCCL_BINE_XOVER (bake into the"
echo "collectives.h default + init.cc NCCL_PARAM). Full logs + summary.txt in: $OUT"

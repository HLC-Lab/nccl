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
echo "=== SUMMARY: out-of-place busbw (GB/s), mean over $REPS reps ==="
python3 - "$OUT" <<'PY' | tee "$OUT/summary.txt"
import sys, glob, os
from collections import defaultdict
d = sys.argv[1]
data = defaultdict(lambda: defaultdict(list))   # algo -> size -> [busbw_oop]
for f in glob.glob(os.path.join(d, "*_rep*.log")):
    algo = os.path.basename(f).split("_rep")[0]
    for line in open(f):
        p = line.split()
        if len(p) >= 9 and p[2] == "float":
            try:
                sz = int(p[0]); bw = float(p[7])
            except ValueError:
                continue
            if sz > 0:
                data[algo][sz].append(bw)
def avg(l): return sum(l)/len(l) if l else 0.0
sizes = sorted({s for a in data for s in data[a]})
print(f"{'size(B)':>13} {'Ring':>7} {'PAT':>7} {'Bine':>7} {'Bine/PAT':>9} {'Bine/Ring':>10}")
for s in sizes:
    r = avg(data['Ring'].get(s, [])); pt = avg(data['PAT'].get(s, [])); b = avg(data['Bine'].get(s, []))
    print(f"{s:>13} {r:>7.2f} {pt:>7.2f} {b:>7.2f} {(b/pt if pt else 0):>9.2f} {(b/r if r else 0):>10.2f}")
# overall avg-busbw (mean of per-run Avg lines) for a single headline number
import re
def avgline(algo):
    vals=[]
    for f in glob.glob(os.path.join(d, f"{algo}_rep*.log")):
        for line in open(f):
            m=re.search(r"Avg bus bandwidth\s*:\s*([\d.]+)", line)
            if m: vals.append(float(m.group(1)))
    return sum(vals)/len(vals) if vals else 0.0
print(f"\nAvg-busbw headline:  Ring={avgline('Ring'):.3f}  PAT={avgline('PAT'):.3f}  Bine={avgline('Bine'):.3f}"
      f"   (Bine/PAT={ (avgline('Bine')/avgline('PAT')) if avgline('PAT') else 0:.3f})")
PY

echo ""
echo "Done. Full logs + summary.txt in: $OUT"

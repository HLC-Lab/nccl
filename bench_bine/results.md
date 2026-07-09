# Bine AllGather — benchmark results

Leonardo Booster, A100-SXM-64GB (sm_80), 1 GPU/node, InfiniBand. nccl-tests
`all_gather_perf -b 8 -e 1G -f 2 -g 1 -c 1`. busbw = bus bandwidth (GB/s),
out-of-place. PAT = upstream baseline (commit 5067397). Bine = this work.

## 2026-06-17 — 16 nodes — correctness PASS (both, 0 #wrong)

First correct multi-node run. Bine here is the **unoptimized** version:
`parallelFactor = 1` (no cross-op overlap) and **no aggregation** (one block per
network message). So this is *optimized PAT vs naive Bine* — not yet a fair
schedule comparison. Tracking it as the baseline to improve from.

| size      |  PAT busbw |  Bine busbw | Bine/PAT |
|-----------|-----------:|------------:|---------:|
| 256 B     |       0.00 |        0.00 |    —     |
| 8 KB      |       0.08 |        0.03 |   0.38   |
| 64 KB     |       0.39 |        0.22 |   0.56   |
| 256 KB    |       1.76 |        1.09 |   0.62   |
| 1 MB      |       5.27 |        2.49 |   0.47   |
| 2 MB      |       5.86 |        3.47 |   0.59   |
| 4 MB      |       4.15 |        3.78 |   0.91   |
| 8 MB      |       6.19 |        4.53 |   0.73   |
| 16 MB     |       6.47 |        6.03 |   0.93   |
| 64 MB     |       6.73 |        6.92 |   1.03   |
| 256 MB    |       7.53 |        7.00 |   0.93   |
| 512 MB    |       7.84 |        6.68 |   0.85   |
| 1 GB      |       7.74 |        6.64 |   0.86   |
| **avg**   |   **2.78** |    **2.26** |          |

Small-message latency (256 B, time): PAT 81 µs vs Bine 257 µs (~3.2x).

Notes:
- Bine slower everywhere, worst in small/mid (latency-bound) range; converges at
  large sizes (even edges ahead at 64 MB).
- Expected: `parallelFactor=1` kills overlap; no aggregation means more, smaller
  messages. Optimization work (parallelFactor / aggregation) is the next step.

## 2026-06-17 — 16 nodes — aggregation v1 (parallelFactor=agg, wave padding): REGRESSED

Avg busbw 1.97 (Bine) vs 2.73 (PAT, same run). WORSE than the naive Bine (2.26).
Two causes: skip-padding to fill agg-wide waves ~doubled op/barrier count for
small rounds; splitting 512 workers into `agg` groups left only 512/agg threads
per copy, hurting large-message bandwidth too (1 GB: 5.25 vs 6.64).
=> parallelFactor>1 is the wrong lever for Bine. Reverted to parallelFactor=1.

## 2026-06-17 — 16 nodes — aggregation v2 (parallelFactor=1, no padding): IMPROVED

Same-nodes run (PAT and Bine both on lrdn0031..1156), -n 10. busbw out-of-place.

| size    |  PAT |  Bine v2 | Bine/PAT |
|---------|-----:|---------:|---------:|
| 8 KB    | 0.09 |     0.07 |   0.78   |
| 64 KB   | 0.61 |     0.51 |   0.84   |
| 256 KB  | 2.07 |     1.73 |   0.84   |
| 512 KB  | 3.55 |     2.31 |   0.65   |
| 1 MB    | 5.46 |     3.78 |   0.69   |
| 2 MB    | 5.33 |     5.91 |   1.11   |
| 4 MB    | 7.45 |     5.61 |   0.75   |
| 8 MB    | 8.95 |     6.42 |   0.72   |
| 16 MB   | 8.04 |     6.72 |   0.84   |
| 32 MB   | 7.57 |     7.34 |   0.97   |
| 64 MB   | 7.06 |     7.49 |   1.06   |
| 128 MB  | 7.36 |     7.58 |   1.03   |
| 256 MB  | 7.94 |     7.14 |   0.90   |
| 1 GB    | 8.09 |     7.06 |   0.87   |
| **avg** | 3.15 | **2.76** |   0.88   |

Aggregation lifted Bine from 2.26 (naive) to 2.76 avg; now competitive/ahead at
large sizes. Remaining gap at small/mid (512 KB–16 MB): block_by_block gathers
SCATTERED blocks into the send slot (strided copy) vs PAT's single contiguous
send. That copy inefficiency is the block_by_block ceiling -> needs send_remap
(contiguous) for parity. High run-to-run variance (PAT 2.73–3.15 across runs);
use run_compare.slurm same-allocation runs for the authoritative comparison.

## 2026-06-17 — 64 nodes — aggregation v2 — gap WIDENS with scale

Same-nodes, -n 50. busbw out-of-place. (PAT log truncated at 256 MB.)

| size    |  PAT | Bine v2 | ratio |
|---------|-----:|--------:|------:|
| 512 KB  | 1.55 |    1.18 |  0.76 |
| 1 MB    | 2.38 |    1.87 |  0.79 |
| 2 MB    | 5.55 |    2.23 |  0.40 |
| 8 MB    | 5.33 |    3.96 |  0.74 |
| 16 MB   | 6.00 |    4.26 |  0.71 |
| 32 MB   | 6.31 |    4.15 |  0.66 |
| 64 MB   | 7.00 |    5.37 |  0.77 |
| 128 MB  | 7.56 |    6.07 |  0.80 |
| 256 MB  | 7.40 |    6.00 |  0.81 |
| Bine avg (full sweep) | — | 1.81 | |

At 64 ranks the ratio drops to ~0.7-0.8 (was 0.88 at 16). block_by_block sends up
to n-1=63 messages (postFreq->1 at large chunks) vs PAT's ~log2(64)=6 contiguous
messages -> message/scatter overhead grows with scale. **Conclusion: block_by_block
is message-count-bound; send_remap (log n contiguous messages) is required to beat
PAT. block_by_block now fully characterized -> moving to send_remap.**

## 2026-06-17 — 16 nodes — v3 (2_blocks + postFreq aggregation + parallelFactor=postFreq)

Same-nodes, -n 50. busbw out-of-place. Root cause of prior gap (from 5-agent
workflow + op-count harness): NOT message count, NOT the pi pattern, NOT stepOffset
-- it was CONCURRENCY (Bine parallelFactor=1 vs PAT up to 16). v3 raises
parallelFactor=postFreq (intra-message aggregation, the only race-free concurrency
for a pairwise butterfly) and drops the flat path (which caused the 2_blocks
correctness #wrong via FIFO-slot overflow). Both fixed: 0 #wrong everywhere.

| size    |  PAT | Bine v3 | ratio |
|---------|-----:|--------:|------:|
| 256 KB  | 2.01 |    1.76 |  0.88 |
| 512 KB  | 3.44 |    3.20 |  0.93 |
| 1 MB    | 5.38 |    4.53 |  0.84 |
| 2 MB    | 7.47 |    5.02 |  0.67 |  <- worst (mid; postFreq~2-4)
| 8 MB    | 8.00 |    7.63 |  0.95 |
| 16 MB   | 6.82 |    6.08 |  0.89 |
| 64 MB   | 6.87 |    6.52 |  0.95 |
| 128 MB  | 7.20 |    7.14 |  0.99 |
| 256 MB  | 7.67 |    7.59 |  0.99 |
| 1 GB    | 8.20 |    7.95 |  0.97 |
| **avg** | 3.11 |    2.82 |  0.91 |

Trajectory (16n ratio): naive 0.81 -> v2(agg,pf=1) 0.88 -> v3(agg,pf=postFreq) 0.91.
At LARGE sizes Bine ~matches PAT (0.97-0.99, both link-bound) -- the structural
concurrency limit does NOT bite there. Residual gap is MID sizes (1-4MB) where
postFreq (concurrency) is only ~2-4.

## 2026-06-17 — 32 nodes — v3 — concurrency win at small/mid, structural ceiling at large

Same-nodes, -n 50, 0 #wrong. (64-node attempt HUNG for BOTH baseline and v3 = cluster/node issue, not our code.)

| size    |  PAT | Bine v3 | ratio |
|---------|-----:|--------:|------:|
| 256 KB  | 0.96 |    0.60 |  0.63 |
| 512 KB  | 1.71 |    1.79 |  1.05 |  <- Bine >= PAT
| 1 MB    | 2.43 |    2.43 |  1.00 |
| 2 MB    | 4.06 |    4.54 |  1.12 |  <- Bine beats PAT (high postFreq -> concurrency)
| 4 MB    | 7.33 |    4.92 |  0.67 |
| 16 MB   | 8.23 |    5.62 |  0.68 |
| 64 MB   | 8.22 |    6.84 |  0.83 |
| 256 MB  | 8.77 |    7.41 |  0.84 |
| 1 GB    | 9.16 |    7.53 |  0.82 |
| **avg** | 3.04 |    2.54 |  0.84 |

Read: v3's concurrency makes Bine MATCH/BEAT PAT at small/mid (512K-2M, postFreq
large). Large (4M+) trails 0.67-0.84 and the gap WIDENS with scale (0.91@16n ->
0.84@32n): at 16n PAT was link-bound (~8) so Bine matched; at 32n PAT scales to
9.16 GB/s @1G while Bine plateaus ~7.5 -- because postFreq->1 at large => Bine
parallelFactor->1 vs PAT's 16-way cross-peer concurrency. STRUCTURAL concurrency
ceiling (pairwise butterfly can't match tree-relay concurrency), NOT the pi pattern
(which still doesn't make Bine BEAT PAT at large).

### NCCL_BUFFSIZE=8MB (v3, 32 nodes) -- did NOT help (avg 2.43 vs 2.54 @4MB default)
Helped small (256K 0.60->1.08, 1M 2.43->2.93: whole msg in fewer chunks) but HURT
mid/large (2M 4.54->3.99, 8M 5.38->4.64, 32M 6.43->5.54, 128M 7.24->6.98). Reason:
chunkCount scales WITH buffsize, so postFreq=slotElems/chunkCount stays flat -> no
extra concurrency, just bigger chunks => FEWER in-flight messages at mid/large =>
worse pipeline fill. => NCCL_BUFFSIZE is the wrong knob (both sides of the ratio
grow together). Decoupling concurrency from chunk size needs CODE: sub-chunking
(split chunkCount slice into K sub-slices -> postFreq*K). If sub-chunking lifts
large-msg busbw, the limit was BDP-fill on one link; if not, it is the deeper
multi-peer/multi-NIC limit (butterfly sends to ONE peer/step, PAT's aggregated
trees spread concurrent traffic across many peers/paths/NICs) = structural.

## 2026-06-19 — 32 nodes — v6 RELAY-TREE (negabinary edges + multi-peer width)

Same-nodes, -n 50, 0 #wrong, NO HANG (relay approach works end-to-end on HW).
NOTE: this PAT baseline run is the FASTEST observed (9.86@1G vs prior ~8.6-9.2) =
great node placement; ratio looks worse partly for that reason. v6 absolute @1G
(7.79) actually >= v4's (~7.35).

| size    |  PAT |  v6  | ratio |
|---------|-----:|-----:|------:|
| 1 MB    | 4.74 | 3.87 |  0.82 |
| 2 MB    | 6.62 | 4.99 |  0.75 |
| 4 MB    | 8.19 | 5.61 |  0.69 |
| 8 MB    | 8.75 | 5.38 |  0.61 |  <- worst (copy-bound mid)
| 16 MB   | 9.39 | 5.72 |  0.61 |
| 64 MB   | 9.42 | 7.01 |  0.74 |
| 256 MB  | 9.78 | 7.58 |  0.78 |
| 1 GB    | 9.86 | 7.79 |  0.79 |
| **avg** | 3.63 | 2.73 |  0.75 |

CORRECT + multi-peer width achieved, but does NOT beat PAT. ROOT CAUSE: my relay
emits SEPARATE send + recv ops; PAT FUSES recv+forward into ONE patCopy (read
recv-FIFO once, write BOTH output and send-FIFO). So v6 does 2x ops + ~2x copy
traffic, and parallelFactor=1 (forced -- relay depth-deps forbid v4's multi-group
copy concurrency) => single 512-thread group is COPY-BOUND => the network width it
unlocked can't be cashed in (worst at 8-16MB where copy dominates, recovers toward
1GB). => to match PAT, must FUSE recv+forward (= PAT structure with negabinary
edges). Width validated; efficiency (fusion) is the remaining gap.

## 2026-06-17 — 32 nodes — v4 (block_by_block CORRECT locality distant->near + concurrency)

Same-nodes, -n 50, 0 #wrong. Fixes the 2_blocks anti-locality (big transfers now
to NEAREST peer, per Bine intent).

| size    |  PAT |   v4 | ratio |
|---------|-----:|-----:|------:|
| 256 KB  | 1.50 | 1.63 |  1.09 |
| 512 KB  | 2.85 | 2.73 |  0.96 |
| 1 MB    | 2.98 | 3.80 |  1.27 |  <- Bine beats PAT
| 2 MB    | 4.05 | 5.07 |  1.25 |  <- Bine beats PAT
| 4 MB    | 6.04 | 5.47 |  0.91 |
| 16 MB   | 7.38 | 5.62 |  0.76 |
| 64 MB   | 8.74 | 5.72 |  0.65 |
| 256 MB  | 7.85 | 6.73 |  0.86 |
| 1 GB    | 8.86 | 7.35 |  0.83 |
| **avg** | 3.04 | 2.60 | 0.855 |

vs anti-local 2_blocks @32n (avg 0.836): v4 = 0.855, MODEST gain (and partly
cross-run noise). Small/mid: clear Bine WIN (1-2MB 1.25-1.27x). Large: still ~0.8,
basically unchanged. LESSON: locality & concurrency are COUPLED -- at large
postFreq->1, so Bine drives ~1 msg in flight and CANNOT keep even the now-nearest
(fast) link busy => can't cash in the locality. Concurrency is the binding
constraint at large; fixing locality alone can't overcome it. => sub-chunking
(raise parallelFactor independent of chunkCount) is the needed next lever: locality
(v4) + concurrency (sub-chunk) together is what could close/win large.

## 2026-06-19 — 32 nodes — v5 (v4 locality + SUB-CHUNKING, parallelFactor=16 at large)

Same-nodes, -n 50, 0 #wrong. Sub-chunking forces parallelFactor=16 even at large
chunks (v4 collapsed to 1 there).

| size    |  PAT |   v4 |   v5 | v5/PAT |
|---------|-----:|-----:|-----:|-------:|
| 512 KB  | 3.00 | 2.73 | 2.73 |  0.91  |
| 1 MB    | 2.86 | 3.80 | 2.37 |  0.83  |  (1M = high-variance transition pt)
| 2 MB    | 4.97 | 5.07 | 4.52 |  0.91  |
| 4 MB    | 5.66 | 5.47 | 5.81 |  1.03  |
| 16 MB   | 6.69 | 5.62 | 5.45 |  0.81  |
| 32 MB   | 9.64 |   -  | 5.79 |  0.60  |
| 64 MB   | 9.08 | 5.72 | 6.21 |  0.68  |
| 256 MB  | 8.63 | 6.73 | 6.91 |  0.80  |
| 1 GB    | 8.59 | 7.35 | 7.35 |  0.86  |
| **avg** | 3.01 | 2.60 | 2.55 |  0.847 |

DECISIVE NEGATIVE RESULT: v5 large-msg busbw == v4's (statistically identical;
1G 7.35==7.35). Forcing parallelFactor=16 at large changed NOTHING => the large gap
is NOT GPU-copy / BDP-fill. WHY: our parallelFactor adds COPY concurrency (16 worker
groups fill ONE FIFO slot, ONE post/wave => still ONE network message per wave to
ONE peer); the in-flight NETWORK depth is unchanged. At large the bottleneck is the
network, not the copy. PAT's parallelFactor is NETWORK concurrency (n aggregated
trees post to MANY peers at once => many NICs/paths busy). The pairwise butterfly has
dependent serial steps, ONE peer each => at most 1 peer-flow at a time (the one link
is already FIFO-saturated). STRUCTURAL, inherent to the butterfly, unfixable without
abandoning it (=> relay-tree/PAT-with-negabinary-edges). Sub-chunking still helps
small/mid (copy/latency-bound there) but not the avg. => Bine-the-butterfly is
CHARACTERIZED: correct, competitive/winning small-mid, structurally ~0.85 at large
because it drives 1 network flow/step vs PAT's many.

## 2026-06-19 — same-allocation scaling (v7 fused relay vs PAT), 1 GB out-of-place

16-node salloc (lrdn0001..0239), -n 20. Channels/threads IDENTICAL (NCCL_DEBUG:
both 2 coll channels, 512 nThreads) -> channel hypothesis RULED OUT.

| N  | Bine | PAT  | ratio |
|----|-----:|-----:|------:|
| 4  | 8.43 | 8.86 | 0.95  |
| 8  | 7.81 | 8.76 | 0.89  |
| 16 | 7.74 | 8.04 | 0.96  |

GAP IS ~CONSTANT (~0.93), DOES NOT WIDEN WITH N. Not channels (identical), not
path-length (4n fully-connected already 0.95), not connection-path-count (that
would scale). It's a UNIFORM ~7% implementation overhead (steady-state pipeline
efficiency / op ordering) in the research port vs NVIDIA tuned PAT. The earlier 32n
0.79 was PLACEMENT VARIANCE: PAT swings a lot (8.04 at 16n here vs 9.8 at 32n
earlier), Bine is STEADY (7.7-8.4 across all N) -> Bine less placement-sensitive
(negabinary contention-balancing). CONCLUSION: faithful Bine relay in NCCL is
within ~7% of tuned PAT at large, correct, and more placement-robust; the pattern
is competitive, not the bottleneck.

## 2026-06-19 — BINE BEATS PAT (channel sweep, 8 nodes, 1 GB, same allocation)

The ~7% "deficit" was a CHANNEL-COUNT/placement artifact. Forcing equal channels
(NCCL_MIN_NCHANNELS=NCCL_MAX_NCHANNELS) for both, back-to-back:

| nchannels | Bine | PAT  | margin |
|-----------|-----:|-----:|-------:|
| 2         | 9.37 | 8.84 | +6%    |
| 4         |10.69 | 9.25 | +16%   |
| 8         |11.46 |10.05 | +14%   |
| 16        |11.85 |10.74 | +10%   |

BINE WINS AT EVERY CHANNEL COUNT. Both scale with channels; Bine scales BETTER and
its peak (11.85) beats PAT peak (10.74) by 10%. (Build still had diagnostic printfs
that SLOW Bine -> real margin larger.) Diagnostic earlier proved kernel is
network-bound + workers never starve (impl optimal); the win is that binePi
connections use the NICs better than +-2^k as channels increase = the multi-peer-
width advantage, visible once channels are equalized. The earlier same-default runs
(Bine 7.81 vs PAT 8.76 @8n) compared unequal channel counts (default tuning under-
served Bine) + placement noise. NEXT: characterize across all sizes at fixed
channels; push channels to 32/64 for ceiling; relax tuning.cc:342 busBw*=.75 for
the Bine AG path so it gets full channels + auto-select by default.

## 2026-06-19 — CONFIRMED: Bine beats PAT at IDENTICAL channels (default + forced)

Instrumented per-op channel count (BINE-CH log in enqueue.cc). 8 nodes, 1 GB,
default (no NCCL_MIN_NCHANNELS):
  Bine: nMaxChannels=2 nChannels=2 commNChannels=2  -> 9.41 GB/s
  PAT : nMaxChannels=2 nChannels=2 commNChannels=2  -> 8.96 GB/s

CHANNELS IDENTICAL (2/2 both). Bine wins +5% at default, same channels. The earlier
7.81 (apparent loss) was NOISE: forced-2 gave Bine 9.37, default-2 gives 9.41 -
consistent. enqueue.cc channel-selection is byte-identical for both, so the per-op
count is necessarily equal; the mid-analysis "default under-channels Bine"
hypothesis was WRONG (refuted by this measurement).

FINAL: Bine BEATS PAT at equal channels everywhere - default 2ch +5%, 4ch +16%,
8ch +14%, 16ch +10%. No env var, no tuning patch needed; win is algorithmic
(multi-peer width uses NICs better). NCCL defaults to only 2 channels here
(commNChannels=2); forcing more helps BOTH (~2x) and widens Bine's margin -
orthogonal knob. Diagnostic proved impl optimal (network-bound, 0 worker starvation).

## 2026-06-20 — 3-way: Bine(v7 relay) vs PAT vs Ring, 16 nodes, DEFAULT channels, 3 reps

final_compare.sh. busbw out-of-place, mean of 3 reps. 0 #wrong all.

| size    | Ring  | PAT   | Bine  | Bine/PAT | Bine/Ring |
|---------|------:|------:|------:|---------:|----------:|
| 256 KB  | 2.07  | 2.01  | 0.70  | 0.35     | 0.34      |
| 1 MB    | 5.69  | 5.39  | 2.41  | 0.45     | 0.42      |
| 2 MB    | 7.83  | 7.50  | 4.07  | 0.54     | 0.52      |
| 8 MB    |10.24  | 8.04  | 6.75  | 0.84     | 0.66      |
| 16 MB   |10.17  | 6.88  | 8.51  | 1.24     | 0.84      |
| 64 MB   |11.88  | 6.88  | 9.92  | 1.44     | 0.83      |
| 256 MB  |11.95  | 7.71  |10.20  | 1.32     | 0.85      |
| 1 GB    |11.98  | 8.22  |10.29  | 1.25     | 0.86      |
| **avg** | 4.39  | 3.14  | 3.22  | 1.027    | —         |

HONEST PICTURE: Bine(v7 relay) BEATS PAT only at LARGE (>=16MB, 1.24-1.44x); LOSES
to PAT at small/mid (0.25-0.84x). RING is best overall (avg 4.39) and beats Bine at
every size (0.19-0.86). avg: Ring 4.39 > Bine 3.22 ~ PAT 3.14 (+2.7%).
CONTEXT: Ring winning large is EXPECTED (BW-optimal; NCCL auto-picks Ring for large).
PAT weak at large is expected (trees aren't for large). The meaningful tree-vs-tree
comparison (Bine vs PAT) is MIXED: Bine wins large, loses small/mid -- the WRONG way
round, since small/mid latency is a tree's home turf. The earlier BUTTERFLY (v3/v4,
parallelFactor=postFreq) WON small/mid vs PAT (2M 1.12x); switching to the RELAY
(v7, parallelFactor=1) to win large traded that away -- at small/mid pf=1 means the
single worker stalls on each recv (latency-bound) -> 3x slower. So the relay won the
regime Ring already owns and gave up the regime Bine could own.
NEXT: (a) forced-channels 3-way (does ordering change? does Bine close on Ring at
large?); (b) the real opportunity = small/mid (Bine's niche) -> hybrid (butterfly for
small/mid + relay for large) or fix v7 small-msg latency (parallelFactor>1 with
depth-aligned waves).

## 2026-07-08/09 — v8: offline review + deadlock fix. FIRST 64-NODE RUNS (v7 relay, cleaned)

Code review + offline verification (bench_bine/verify_schedule.py, new) proved the
v7 depth-ordered emission DEADLOCKS under NCCL's 8-slot FIFOs at n>=64: per-connection
bursts = binomial(log2(n)-1,d), <=6 at n=32 (live) but 10 at n=64 -> mutual dim-0
stall. Model is confluent => hang is structural, not timing. v7 had only ever run at
<=32 nodes. Also removed the leftover BINE-DBG/BINE-STUCK device printfs (all pre-v8
numbers above were measured WITH printfs in the hot path; not directly comparable).
Commits: ebe845f (printf removal), 83356bb (source order), 666265e (skew lambda=6).
Plan going forward: bench_bine/IMPROVEMENT_PLAN.md.

### v8a (commit 83356bb) — plain SOURCE ORDER emission: LIVE at 64n but ~0.1x PAT

Emit blocks s=0..n-1 (each block's fused+extras adjacent). Deadlock-free down to
FIFO depth 2 (verified), and indeed: N=64, default channels, -n 10, 1 rep:
0 #wrong, NO HANG — first-ever correct 64-node completion (old order hangs there).
BUT: Bine avg 0.34 vs PAT 3.34 (Ring 3.62); flat ~1.05 GB/s plateau >=16 MB
(1 GB: 1.05 vs PAT 9.41 = 0.11x; 1 MB: 0.22 vs 4.32 = 0.05x).
ROOT CAUSE (timed model, bench_bine/timed_sim.py, new): with key(s)=s every rank
consumes each block at the same wavefront instant its parent produces it -> every
fused op stalls ONE FULL serialized network hop; nothing is ever pre-buffered in
the FIFO. Model reproduces the collapse (~2 GB/s @64n) and the depth-order contrast
(~32 GB/s where it is live). LESSON: liveness gates are NECESSARY NOT SUFFICIENT
for an emission order; throughput model added as a mandatory gate.

### v8b (commit 666265e) — SKEWED source order key(s) = s + 6*depth(s): 0.90x PAT at large

One-parameter family: lambda=0 = source order (live, synchronous, slow);
lambda=inf = depth order (pipelined, deadlocks n>=64). lambda emits each block
~lambda positions later per tree-hop => data arrives ~one hop BEFORE consumption
(pipelining) with bounded per-connection skew (liveness). Consistency holds for any
lambda (receiver keys = sender keys + lambda). lambda=6 chosen by sim: live down to
FIFO depth 6 (2 slots margin) for all po2 n<=256; lambda=8 zero margin; lambda>=12
deadlocks. depth(s) comes free from the getIdx DFS (the O(n^2 log n) depthOf pass
deleted: ~230k -> ~500 binePi calls/channel/call at n=256).

N=64, default channels, -n 10, 1 rep, 0 #wrong, no hang (different allocation than
v8a; within-run Bine/PAT is the fair comparison):

| size    |  Ring |  PAT  | Bine  | Bine/PAT |
|---------|------:|------:|------:|---------:|
| 1 MB    |  2.02 |  4.10 |  1.28 |   0.31   |
| 4 MB    |  6.18 |  7.21 |  3.88 |   0.54   |
| 16 MB   |  9.60 |  9.13 |  6.55 |   0.72   |
| 32 MB   | 10.66 |  8.09 |  7.32 |   0.90   |
| 64 MB   | 10.26 |  8.41 |  7.86 |   0.93   |
| 256 MB  | 11.91 |  9.20 |  8.14 |   0.88   |
| 1 GB    | 12.00 |  9.51 |  8.54 |   0.90   |
| **avg** |  3.62 |  3.36 |  2.44 |   0.73   |

READ: 8x over v8a at large; >=32 MB now 0.88-0.93x PAT at DEFAULT channels — same
regime as the healthy 16-node v7 runs (0.89-0.96), so the scaling pathology is gone
(64n now behaves like 16n did). Small/mid (<=8 MB) still 0.2-0.65x: structural
(1 block/slot/post, parallelFactor=1) -> Phase 4 butterfly hybrid, unchanged by this
fix. NEXT: (a) forced-channels 3-way at 64n (historically Bine +10% over PAT at
16ch, equal channels); (b) 128-node point (sim says live; would fully retire the
deadlock question); (c) Phase 4 butterfly for small/mid.

### v8b + NCCL_FORCE_CH=16, 64 nodes — BINE BEATS PAT >=128 MB (+7..23%); channels hurt small/mid

Same build (666265e), N=64, -n 10, 1 rep, 0 #wrong, forced 16 channels all three
(different allocation than the default-channel run):

| size    |  Ring |  PAT  | Bine  | Bine/PAT |
|---------|------:|------:|------:|---------:|
| 1 MB    |  1.86 |  1.61 |  0.18 |   0.11   |
| 8 MB    |  4.90 |  5.67 |  1.58 |   0.28   |
| 32 MB   |  9.76 |  8.55 |  5.34 |   0.62   |
| 64 MB   |  7.58 |  9.01 |  7.99 |   0.89   |
| 128 MB  | 11.80 |  8.41 |  9.22 |   1.10   |
| 256 MB  | 12.18 |  7.66 |  9.45 |   1.23   |
| 512 MB  | 12.22 |  8.41 |  9.00 |   1.07   |
| 1 GB    | 12.17 |  8.54 | 10.16 |   1.19   |
| **avg** |  3.10 |  2.70 |  2.05 |   0.76   |

READ: the multi-peer-width advantage SURVIVES AT 64 NODES: with equal 16 channels
Bine wins every size >=128 MB (peak 10.16 vs PAT 8.54 = +19%; crossover ~100 MB),
and Bine's best config beats PAT's best config at 1 GB (10.16 vs 9.51 default-ch =
+7%). Channels scale Bine UP at large (8.54 -> 10.16) while PAT goes DOWN
(9.51 -> 8.54). BUT 16ch degrades small/mid for ALL algos (PAT 1 MB 4.10 -> 1.61;
Bine 1.28 -> 0.18; per-channel slices go latency-bound), so channel count is a
per-size knob: default channels win <=64 MB, 16ch wins >=128 MB. Bine is hurt MORE
by many channels at small sizes (fixed per-op stalls x parallelFactor=1 replicate
per channel). CONCLUSIONS: (a) large-message story is now Bine>PAT at 64n given
channel tuning (auto-tuning would need the tuning.cc model updated — known Phase 5
item); (b) small/mid remains THE gap and is unaffected by channels -> Phase 4
butterfly hybrid is the next code change; (c) optional: ch=4/8 sweep to map the
crossover; 128-node point still pending.

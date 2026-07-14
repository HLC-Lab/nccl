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

## 2026-07-09 — v9: Phase 4 relay+butterfly HYBRID, 64 nodes

Hybrid in one PatAGAlgorithm (commit fbc90c5): butterfly (packed pairwise, pico
allgather_bine_block_by_block) for small/mid, skewed relay for large. First cut
selected the mode by postFreq.

### v9 default channels — butterfly HELPS small/mid, large unchanged (clean win)

N=64, -n 10, 1 rep, 0 #wrong. vs the v8b relay-only default-channel run:

| size    | PAT  | v8b relay | v9 hybrid | v9/PAT | v8b/PAT |
|---------|-----:|----------:|----------:|-------:|--------:|
| 1 MB    | 4.42 |      1.28 |      2.00 |  0.45  |  0.31   |
| 4 MB    | 7.26 |      3.88 |      5.45 |  0.75  |  0.54   |
| 8 MB    | 8.40 |     ~5.4  |      6.10 |  0.73  |  ~0.65  |
| 16 MB   | 9.15 |      6.55 |      6.54 |  0.71  |  0.72   |
| 32 MB   | 8.79 |      7.32 |      7.29 |  0.83  |  0.90   |
| 1 GB    | 9.33 |      8.54 |      8.21 |  0.88  |  0.90   |

Butterfly lifts 1 MB 0.31->0.45 and 4 MB 0.54->0.75; large (>=16 MB) unchanged
(relay). Net improvement at default channels, no regression.

### v9 FORCED 16 channels — REGRESSION at large: postFreq gate is channel-sensitive

N=64, NCCL_FORCE_CH=16, -n 10, 1 rep, 0 #wrong. vs v8b relay-only forced-16ch:

| size    | PAT  | v8b relay-16ch | v9 hybrid-16ch |
|---------|-----:|---------------:|---------------:|
| 1 MB    | 1.32 |           0.18 |           1.31 |   <- butterfly: 7x small win
| 128 MB  | 4.49 |           9.22 |           4.72 |
| 1 GB    | 8.22 |          10.16 |           5.98 |   <- REGRESSION (10.2 -> 6.0)

DIAGNOSIS: the relay path is byte-identical to v8b, so 1 GB = 5.98 (not ~10) proves
the BUTTERFLY was selected at 1 GB. Root cause: postFreq = slotBytes/chunkBytes is
per-CHANNEL; 16 channels shrink the chunk, raise postFreq, and trip the butterfly
switch even for a 1 GB message -> butterfly's pairwise bandwidth limit bites (the
regime the relay exists for). Small win is real and large (butterfly-mode 1 GB 5.98)
matches the old pure-butterfly numbers.

### FIX ATTEMPT 1 (commit 68894ae): gate on full PER-RANK size -- CAUSED A DEADLOCK

useButterfly = (count*sizeof(T) <= THRESH) && (postFreq >= minPost). INTENT: count is
channel-invariant. BUG: 'count' is NOT host/device-consistent -- on the device it is the
FULL per-rank size, but the host proxy (proxy.cc:718-722) passes size=nbytes/nRanks
(PER-CHANNEL) as count. So above 2 channels the proxy and kernel pick DIFFERENT modes ->
different per-dim step counts -> NETWORK HANG. (2 channels survived only because the
postFreq safety floor forced relay on both sides for the sizes that would otherwise
straddle the threshold.) Reported: ">2 channels takes forever." fbc90c5's postFreq gate
did NOT hang because postFreq derives from the chunk size, which IS consistent.

### FIX ATTEMPT 2 (current): gate on THIS CHANNEL's per-rank bytes (end-offset)*sizeof(T)

useButterfly = ((end-offset)*sizeof(T) <= BINE_BUTTERFLY_MAX_BYTES) && (postFreq >= minPost).
(end-offset) = channelCount on the device and 'size' on the host -- the exact quantity the
chunk loop already relies on being equal, so host and device agree by construction (no
hang). It also scales the right way (less data/channel -> more latency-bound -> butterfly),
so forced-16ch 1 GB -> ~1 MB/rank per channel -> above threshold -> RELAY (fixes the
fbc90c5 regression too). BINE_BUTTERFLY_MAX_BYTES = 128 KB per-channel-per-rank (first cut;
tune with BINE_FORCE_* builds). Offline gates pass (byte-identical op lists + host/device
mode-consistency check added). NEEDS HARDWARE RE-RUN: expect no hang at any channel count,
default small/mid wins preserved, forced-16ch 1 GB back to ~10 (relay).

SAFETY VALVE for benchmarking: BINE_FORCE_RELAY / BINE_FORCE_BUTTERFLY builds skip the auto
mode decision entirely (force-relay = always relay; force-butterfly = butterfly wherever
postFreq is safe), so they CANNOT hit any mode-mismatch hang. Use BINE_FORCE_RELAY to get
the large-message forced-channel win independently of the auto gate.

### v9 fix CONFIRMED on hardware: 64 nodes, FORCED 8 channels, 3 reps, NO HANG

commit 28e50f6, N=64, NCCL_FORCE_CH=8, -n 50, 3 reps, 0 #wrong. The count-gate build
(68894ae) HUNG at >2 channels; this per-channel-gate build completes cleanly at 8 -> the
host/device mode-mismatch deadlock is fixed. Means over 3 reps:

| size    | Ring |  PAT | Bine | Bine/PAT | Bine/Ring |
|---------|-----:|-----:|-----:|---------:|----------:|
| 1 MB    | 0.57 | 1.61 | 0.88 |   0.54   |   1.53    |
| 8 MB    | 1.10 | 4.41 | 4.22 |   0.96   |   3.82    |
| 16 MB   | 2.05 | 5.20 | 4.74 |   0.91   |   2.31    |
| 64 MB   | 4.05 | 6.34 | 6.79 |   1.07   |   1.68    |
| 128 MB  | 6.20 | 6.53 | 6.31 |   0.97   |   1.02    |
| 256 MB  | 8.52 | 6.75 | 7.62 |   1.13   |   0.89    |
| 512 MB  | 9.29 | 7.11 | 7.85 |   1.10   |   0.84    |
| 1 GB    |10.08 | 7.56 | 8.24 |   1.09   |   0.82    |
| **avg** | 1.70 | 2.14 | 2.00 |   0.94   |   1.18    |

READ: (1) NO HANG at 8 channels = deadlock fix validated. (2) No large-message regression:
1 GB picks the relay (8.24), Bine BEATS PAT for every size >=64 MB (1.07-1.13x) and 8-16 MB
is ~parity (0.91-0.96). (3) Channel-robustness confirmed again: PAT 1 GB DROPS with more
channels (9.3 @2ch default -> 7.56 @8ch) while Bine holds (~8.2), so Bine overtakes PAT at
8ch. (4) Bine crushes Ring at small/mid (1.5-4x) and Ring only wins the >=256 MB tail. (5)
Small/mid <=4 MB still 0.5-0.7x PAT -> the parallelFactor=1 gap (Phase 4b).

### v9 fix, 64 nodes, FORCED 16 channels, 3 reps (commit 28e50f6, HEAD confirmed)

N=64, NCCL_FORCE_CH=16, -n 50, 3 reps, 0 #wrong, NO HANG.

| size    | Ring |  PAT | Bine | Bine/PAT |
|---------|-----:|-----:|-----:|---------:|
| 1 MB    | 0.59 | 1.04 | 0.87 |   0.84   |
| 4 MB    | 1.26 | 2.45 | 2.44 |   1.00   |
| 8 MB    | 1.18 | 3.88 | 4.05 |   1.04   |
| 16 MB   | 2.41 | 4.88 | 4.51 |   0.92   |
| 32 MB   | 4.34 | 5.63 | 4.63 |   0.82   |  <- DIP
| 64 MB   | 3.57 | 5.70 | 4.66 |   0.82   |  <- DIP
| 128 MB  | 6.82 | 6.36 | 7.19 |   1.13   |
| 256 MB  | 8.37 | 6.57 | 7.60 |   1.16   |
| 1 GB    | 9.97 | 7.63 | 8.58 |   1.12   |
| **avg** | 1.74 | 1.95 | 1.93 |   0.99   |

READ: (1) NO HANG at 16 channels + 1 GB = 8.58 (relay), so the deadlock AND the
fbc90c5 large-regression are both gone at 16ch too. (2) Bine BEATS PAT at large
(>=128 MB, 1.12-1.16x) and at 4-8 MB (1.00-1.04x); ~parity overall (avg 0.99).
(3) The absolute 1 GB is 8.58, not the earlier 1-rep 10.16 -- that was a
good-placement allocation; THIS allocation is ~15% slower for ALL algos (PAT
7.63 vs 8.54, Ring 9.97 vs 12). The stable, honest claim is the RATIO: Bine ~1.12x
PAT at large. (4) NEW FINDING -- a DIP at 16-64 MB (0.82-0.92x): at 16 channels
these sizes fall UNDER the 128 KB/channel butterfly threshold (32 MB -> 32 KB/rank
per channel) so they run the BUTTERFLY, which underperforms the relay in that band
(cf. 8ch where 64 MB used relay and hit 1.07x). => BINE_BUTTERFLY_MAX_BYTES=128 KB
is too high; the butterfly should hand 16-64 MB back to the relay. This is the
crossover-tuning item: force-sweep (BINE_FORCE_RELAY vs BINE_FORCE_BUTTERFLY builds)
to place the threshold, expected to lift the 16-64 MB dip to relay's ~1.0x+.
(5) Small/mid <=2 MB still 0.4-0.84x PAT (parallelFactor=1, Phase 4b). Bine > Ring
everywhere except the >=256 MB tail.

## 2026-07-10 — v9 at 128 NODES: no hang at any channel count (MILESTONE)

commit 28e50f6. N=128, -n as noted, 1 rep each (INDICATIVE, noisy -- multi-rep TODO), 0
#wrong everywhere. 128 nodes is where the pre-fix (depth-order) code deadlocked; it now
runs clean at 4, 8, and 16 channels -> deadlock retired on HW to 128n (guard allows 256).

Bine/PAT at large (across ch4/ch8/ch16, 1 rep so +/- noise):

| size    | ch4  | ch8  | ch16(-n40) |
|---------|-----:|-----:|-----------:|
| 64 MB   | 1.17 | 0.83 | 0.86       |
| 128 MB  | 1.01 | 1.11 | 1.14       |
| 256 MB  | 1.12 | 1.13 | 1.09       |
| 512 MB  | 0.80 | 0.96 | 1.15       |
| 1 GB    | 1.15 | 1.11 | 1.11       |

READ: (1) NO HANG at 128 nodes, all channel counts = deadlock fully retired on HW.
(2) Bine BEATS PAT at large (>=128 MB ~1.09-1.15x, 1 GB 1.11-1.15x) at 128 nodes too --
the multi-peer-width win scales. 1 GB abs: Bine 7.12(ch4)/8.23(ch8)/8.72(ch16) vs PAT
6.21/7.43/7.83; both scale up with channels here, Bine stays ~1.1x ahead. (3) The MID-SIZE
DIP (16-64 MB, butterfly overused) is present and a bit worse than 64n (minPost=divUp(64,8)
=8 at n=128, so the butterfly region is larger): ch16 32 MB 0.79, 64 MB 0.86; ch8 32 MB
0.42. Confirms the BINE_BUTTERFLY_MAX_BYTES tuning is needed (threshold too high) -- now seen
at BOTH 64n and 128n. (4) Small/mid <=4 MB 0.4-0.75x (parallelFactor=1). (5) Bine >> Ring
small/mid (2-4.6x); Ring wins only >=512 MB. avg Bine ~0.90x PAT (dragged by small/mid+dip).
CAVEAT: 1 rep -> the 512 MB ch4 0.80 and 32 MB ch8 0.42 are likely noise; re-run >=3 reps
for any quoted number. NEXT: force-sweep to fix the mid-dip threshold; then 3-rep clean sweep.

## 2026-07-10 — NCCL_BINE_XOVER runtime knob + crossover sweep (64n, 16ch)

commit 9868c2f made the butterfly/relay crossover a runtime env var NCCL_BINE_XOVER
(per-channel per-rank bytes; 0=relay, huge=butterfly, else crossover), plumbed like
NCCL_BUFFSIZE (host comm->bineXover -> device, set once = host/device consistent).
xover_sweep.sh runs all XOVER values + PAT in one allocation.

### Sweep (64n, 16ch, 2 reps): AVG busbw by XOVER

| XOVER  | relay(0) | 16K  | 32K  | 64K  | 128K | bfly | PAT  |
|--------|---------:|-----:|-----:|-----:|-----:|-----:|-----:|
| avg    |    3.03  | 3.47 | 3.53 |3.57* | 3.49 | 3.43 | 3.39 |

64K best avg (beats PAT avg 3.39), and owns the 32K-16M mid-range per-size. 128K (the
old default) was 3.49. => default lowered 128K -> 64K (commit pending). NOTE the mid-large
dip is NOT threshold-fixable: at 33M ALL Bine modes (~5.5-5.8) trail PAT (~8.5); pure relay
is actually WORST there (4.85) because relay hasn't ramped to plateau yet while PAT has.

### Clean confirm (64n, 16ch, NCCL_BINE_XOVER=65536, 3 reps, 0 #wrong)

| size    | Ring  |  PAT  | Bine  | Bine/PAT |
|---------|------:|------:|------:|---------:|
| 1 MB    | 1.62 |  0.85 |  0.83 |   0.98   |
| 2 MB    | 1.67 |  1.62 |  1.32 |   0.81   |
| 4 MB    | 1.75 |  2.03 |  2.55 |   1.26   |  <- Bine wins
| 8 MB    | 2.40 |  2.82 |  4.01 |   1.42   |  <- Bine wins
| 16 MB   | 6.46 |  4.49 |  4.70 |   1.05   |
| 33 MB   | 8.35 |  7.66 |  5.00 |   0.65   |  <- dip (structural, not threshold)
| 67 MB   | 7.52 |  6.82 |  4.77 |   0.70   |  <- dip
| 128 MB  |10.28 |  7.62 |  7.53 |   0.99   |
| 256 MB  |10.52 |  7.36 |  8.73 |   1.19   |  <- Bine wins
| 1 GB    |10.65 |  7.94 |  9.07 |   1.14   |  <- Bine wins
| **avg** | 2.56 |  2.10 |  2.06 |   0.98   |

HONEST PICTURE at 16ch/64n (clean 3-rep): Bine ~= PAT overall (0.98). Bine WINS 4-16 MB
(butterfly, 1.05-1.42x) and >=256 MB (relay, 1.14-1.19x); LOSES <=2 MB (parallelFactor=1)
and 33-67 MB (transitional: relay not yet at plateau, butterfly below PAT -- both modes
lose, threshold can't fix). Ring is best overall here (2.56) but that is the BW-optimal
baseline; the tree-vs-tree result (Bine vs PAT) is PARITY-with-band-wins. To beat PAT at
MOST sizes needs Phase 4b (parallelFactor>1) for <=2 MB and a fix for the 33-67 MB ramp
(sub-chunking / pipeline fill), NOT more threshold tuning.

### 128 nodes, 16ch, XOVER=65536, 1 rep -n50 (INDICATIVE) -- large win HOLDS, mid erodes

N=128 (1 GPU/node), 0 #wrong, no hang. 1 rep on a 10-min QOS alloc -> magnitudes noisy.

| size    | Ring  |  PAT  | Bine  | Bine/PAT |
|---------|------:|------:|------:|---------:|
| 4 MB    | 0.66 |  4.15 |  1.12 |   0.27   |  <- butterfly, LOST (won at 64n!)
| 8 MB    | 0.71 |  4.28 |  2.40 |   0.56   |
| 16 MB   | 1.34 |  4.26 |  3.44 |   0.81   |
| 33 MB   | 2.10 |  4.04 |  5.64 |   1.40   |  (noisy: adjacent 67M dips)
| 67 MB   | 4.29 |  5.92 |  3.32 |   0.56   |
| 128 MB  | 6.26 |  6.23 |  7.47 |   1.20   |  <- Bine wins
| 256 MB  | 5.78 |  6.48 |  6.83 |   1.05   |
| 512 MB  | 7.99 |  6.77 |  7.60 |   1.12   |
| 1 GB    | 9.05 |  6.76 |  8.00 |   1.18   |  <- Bine wins
| **avg** | 1.39 |  1.93 |  1.70 |   0.88   |

READ: (1) LARGE WIN HOLDS AT 128n: >=128 MB Bine 1.05-1.20x PAT (four consecutive sizes,
robust even at 1 rep) -- the multi-peer-width advantage scales. (2) BUTTERFLY ERODES WITH
SCALE: the 4-16 MB butterfly wins seen at 64n did NOT reproduce at 128n (4M 0.27, 8M 0.56)
-- log2(n)=7 rounds + smaller per-block data weaken the pairwise butterfly at 128 nodes.
=> the 64 KB default is 64n-tuned; 128n likely wants its own (lower) crossover, BUT chasing
it per-node-count is a losing game -- the real small/mid lever is Phase 4b (parallelFactor
>1). (3) avg 0.88x (vs 0.98 at 64n): Bine relatively weaker at 128n, dragged by small/mid.
CAVEAT: 1 rep; 33M 1.40 / 67M 0.56 are adjacent opposite spikes = noise; re-run 3 reps for
any quoted small/mid number. Large (>=128M) is trustworthy.

## 2026-07-10 — 128 NODES, 3-REP CONFIRMATION (16ch + 8ch) + butterfly envelope

Same allocation, N=128 (1 GPU/node), -n 50, 0 #wrong everywhere, no hangs.
NOTE on the default XOVER in these runs: binary default matters only where the postFreq
safety floor allows the butterfly at all; at 128n (minPost=8, slot 512KB) that region ends
far below where the 64KB-vs-128KB defaults differ, so these numbers are identical either way.

### 16ch, 3 reps (default XOVER):

| size    | Ring |  PAT | Bine | Bine/PAT |
|---------|-----:|-----:|-----:|---------:|
| 1 MB    | 0.33 | 1.01 | 0.62 |   0.61   |
| 4 MB    | 1.26 | 3.84 | 1.74 |   0.45   |
| 16 MB   | 1.93 | 3.89 | 3.79 |   0.97   |
| 33 MB   | 4.27 | 5.13 | 4.63 |   0.90   |
| 67 MB   | 5.58 | 6.99 | 4.84 |   0.69   |
| 128 MB  | 4.81 | 6.96 | 6.75 |   0.97   |
| 256 MB  | 6.99 | 6.87 | 7.71 |   1.12   |  <- Bine wins (3-rep)
| 512 MB  | 8.90 | 6.97 | 7.87 |   1.13   |  <- Bine wins
| 1 GB    |10.09 | 7.03 | 8.04 |   1.14   |  <- Bine wins
| **avg** | 1.67 | 2.01 | 1.78 |   0.89   |

### 8ch, 3 reps (default XOVER): large win thinner but present

67 MB 1.05 / 256 MB 1.03 / 512 MB 1.11 / 1 GB 1.05; small/mid 0.25-0.79; avg 0.88.
(Same pattern as 64n: more channels -> bigger Bine edge at large.)

### 16ch, XOVER=2e8 ("butterfly wherever safe"), 1 rep:

Small/mid STILL 0.17-0.59 -- maxing the butterfly region does NOT fix small at 128n
(and at large the postFreq floor forces relay anyway: 512 MB 1.19, 1 GB 1.16, same as
default). Confirms with data: no XOVER value helps <=8 MB at 128 nodes.

BOTTOM LINE 128n (3-rep): Bine BEATS PAT >=256 MB at 16ch (1.12-1.14x) and >=512 MB at 8ch
(1.05-1.11x); ~parity 16-128 MB except a persistent 67 MB dip (0.69); LOSES <=8 MB
(0.25-0.66) at every XOVER -- the butterfly's 64n small/mid wins do not survive at 128n
(7 rounds, smaller blocks). avg ~0.89 both channel counts. CONCLUSION: threshold tuning is
EXHAUSTED as a lever; the small/mid gap at scale is parallelFactor=1 (Phase 4b) territory,
plus the 33-67 MB relay ramp (sub-chunking/pipeline-fill).

## 2026-07-13 — v10: slice-packing fix VALIDATED (64n, 16ch, 3 reps, default XOVER=64KB)

Rebuilt with d4339d8 (butterfly packs by actual slice size, not the 64KB-floored chunk
capacity). 0 #wrong, 3 reps. This allocation runs ~6-9% faster overall than the 07-10 one
(Ring avg 2.72 vs 2.56, PAT 2.27 vs 2.10) -- gains beyond that are the fix.

| size    | PAT   | Bine  | Bine/PAT | vs 07-10 run: Bine abs / ratio |
|---------|------:|------:|---------:|--------------------------------|
| 16 KB   |  0.06 |  0.03 |   0.50   | 0.02->0.03 (+50%) / 0.37->0.50 |
| 32 KB   |  0.11 |  0.06 |   0.53   | 0.04->0.06 (+50%) / 0.41->0.53 |
| 64 KB   |  0.20 |  0.13 |   0.64   | 0.09->0.13 (+44%) / 0.52->0.64 |
| 128 KB  |  0.31 |  0.19 |   0.61   | 0.15->0.19 (+27%) / 0.54->0.61 |
| 256 KB  |  0.44 |  0.30 |   0.67   | 0.25->0.30 (+20%)              |
| 2 MB    |  1.66 |  1.54 |   0.93   | 1.32->1.54 (+17%) / 0.81->0.93 |
| 4 MB    |  2.72 |  2.86 |   1.05   | (PAT much higher this alloc)   |
| 8 MB    |  4.57 |  4.13 |   0.90   |                                |
| 16 MB   |  5.61 |  5.42 |   0.97   |                                |
| 33 MB   |  6.72 |  5.54 |   0.82   | dip persists (structural)      |
| 67 MB   |  7.85 |  5.40 |   0.69   | dip persists                   |
| 128 MB  |  7.71 |  8.61 |   1.12   | <- Bine wins                   |
| 256 MB  |  7.66 |  8.82 |   1.15   | <- Bine wins                   |
| 512 MB  |  7.61 |  8.81 |   1.16   | <- Bine wins                   |
| 1 GB    |  8.17 |  9.33 |   1.14   | <- Bine wins                   |
| **avg** |  2.27 |  2.24 |   0.99   | parity (rep3: Bine > PAT)      |

READ: (1) PACKING FIX CONFIRMED: tiny sizes up 44-50% absolute (16-64 KB), way beyond the
~8% allocation effect, matching the timed model's 1.34-1.37x prediction; ratios at <=256 KB
lifted from 0.37-0.54 to 0.50-0.67. (2) Large headline reconfirmed a THIRD time at 64n:
>=128 MB = 1.12-1.16x PAT. (3) avg parity (0.99). (4) The 33-67 MB dip persists as expected
(both these sizes run the butterfly at the 64 KB default -- 67 MB sits exactly on the
boundary; the sweep showed butterfly >= relay there, so it is structural, not mode choice).
Remaining known gaps: <=1 MB (0.5-0.8x, forced-16ch config auto-tune would never pick) and
33-67 MB ramp. NEXT (optional): re-run xover_sweep.sh (threshold was tuned pre-fix; optimum
can only move UP from 64 KB), and a 128n rep to see how much the fix recovers there
(packing gain is largest at 128n: 64 blocks/slot vs 8).

### v10 at 128n/16ch (1 rep, -n 50, default XOVER=64KB): mid recovered; 128MB mode flip found

0 #wrong. vs the pre-fix 07-10 3-rep run: 4 MB 0.45->0.69, 8 MB 0.55->0.83, 16 MB 0.84,
33 MB 0.90->0.93 (Bine abs +21-43% across 1-33 MB) => the packing fix substantially
reverses the 128n small/mid erosion. Large: 256 MB 1.13, 512 MB 1.20, 1 GB 1.18 -- headline
holds. BUT 128 MB fell 0.97 -> 0.69 (Bine abs 6.75 -> 5.10): MECHANISTIC, not noise. At
128n/16ch, 128 MB has perChan = exactly 64 KB and chunk 256 KB; pre-fix pf = 512K/256K = 2
< minPost 8 -> RELAY (6.75); post-fix slice-based pf = 8 -> BUTTERFLY (5.10, worse). 67 MB
made the same relay->butterfly flip and IMPROVED (4.84 -> 5.42). Combined with the 64n sweep
(67 MB: relay 7.36 > bfly 5.47; 33 MB: bfly 5.76 > relay 4.85), both scales agree:
butterfly wins at <=32 KB perChan slices, relay wins at >=64 KB perChan.
=> DEFAULT MOVED 64 KB -> 48 KB (between the two boundary cases): keeps 33 MB@64n and
67 MB@128n on the butterfly, returns 67 MB@64n and 128 MB@128n to the relay (each worth
+25-35% at its size per the measurements above).

### 32-CHANNEL PROBE (128n, XOVER=49152, 1 rep, Bine only -- job died before PAT/Ring):
### channel scaling is EXHAUSTED at 16ch

Bine 1 GB: 8.53 @32ch vs 8.88/8.70 @16ch (same allocation) -- FLAT. 512 MB: 8.12 vs
8.44/8.24. So the large-message ceiling (~8.5-9.3 GB/s at both 64n and 128n) is
CHANNEL-COUNT-INDEPENDENT from 8ch up; more channels cannot push the Ring crossover
(Ring plateaus ~12 = single-rail saturation). The remaining ~25-30% gap to Ring is a
channel-independent bottleneck (candidates: dim-0 pairing imbalance -- half of all
forwarded traffic on one connection; pipeline bubbles in the skew order; proxy/post
overhead). NOTE 128 MB read 4.89 here: NOT a regression -- at 32ch its per-channel slice
halves to 32 KB <= 48 KB so it flips to the butterfly (the mode gate is per-channel by
design); at 32ch the relay/butterfly boundary sits one octave higher in message size.
Also small sizes degrade further at 32ch (1 MB 0.52 abs vs 0.71-0.87 @16ch) -- consistent
with the forced-channel tax. => 16ch is Bine's sweet spot at 128n; "delay Ring" needs a
structural fix (channel-rotation v11 idea, or understanding the ~8.8 ceiling), not more
channels.

### AUTO-CHANNEL run (64n, default channels = 2, XOVER=49152, 1 rep): the deployed-default
### picture + the crossover is CHANNEL-BUDGET-DEPENDENT

| size    | Ring |  PAT | Bine | B/PAT | mode @2ch,48K | July-9 default run (bfly there) |
|---------|-----:|-----:|-----:|------:|---------------|---------------------------------|
| 1 MB    | 0.40 | 2.62 | 1.47 | 0.56  | bfly          | 0.50                            |
| 4 MB    | 2.45 | 4.14 | 4.03 | 0.97  | bfly (32K/ch) | 0.71                            |
| 8 MB    | 2.72 | 7.34 | 3.20 | 0.44  | RELAY (64K/ch)| 0.80 (Bine 5.13, was BFLY)      |
| 16 MB   | 9.00 | 7.54 | 3.73 | 0.49  | RELAY (128K)  | 0.77 (Bine 5.97, was BFLY)      |
| 33 MB   | 9.95 | 6.90 | 6.93 | 1.00  | relay         | 0.86                            |
| 67 MB   | 9.51 | 6.78 | 7.76 | 1.14  | relay         | 0.92                            |
| 1 GB    |10.00 | 7.56 | 7.74 | 1.02  | relay         | 0.95                            |

READS: (1) 8-16 MB CRATERED (0.44/0.49; abs 3.20/3.73 vs 5.13/5.97 on July 9): the 48 KB
crossover -- tuned at 16 CHANNELS -- flips these sizes to the RELAY at 2 channels, but with
only 2 channel-pipelines the relay cannot hide latency at 64-128 KB slices; the butterfly
(safe there: pf 8/4 >= minPost 4) was right, as July 9 shows. => THE OPTIMAL CROSSOVER
DEPENDS ON THE CHANNEL BUDGET, not just per-channel slice bytes: ~48 KB at >=8-16ch,
>=128 KB (butterfly-where-safe) at 2-4ch. Workaround for default-channel runs:
NCCL_BINE_XOVER=131072. (2) DEPLOYED DEFAULT = 2 CHANNELS: Bine large = 7.4-7.8 = PAT
parity (1.00-1.14 this alloc, 0.86-0.97 July 9) -- the +12-16% large win REQUIRES the
8-16ch budget NCCL never grants by default. PAT is tuned for 2ch; Bine is the only algo
here that gains from more. => the Phase-5 init-time channel floor (NVLS-style) for
Bine-capable comms is THE product fix: it would make the entire forced-16ch result set the
default behavior AND make the 48 KB crossover correct in deployment. (3) At 64n default,
Ring owns >=16 MB outright (9-10.4) -- Bine's deployed niche at 64n/2ch is thin; the 128n
picture (Ring weak at mid under forced ch) still needs an auto-channel run at 128n.

## 2026-07-13 — v11: CHANNEL FLOOR VALIDATED (64n, AUTO channels, 3 reps) — large win is
## now the OUT-OF-THE-BOX default

Build 188acdf (floor NCCL_BINE_NCHANNELS=16 + MINSIZE=128MB + 48KB xover compiled in).
Auto channels (base budget 2; INFO line confirmed "raised channel budget 2 -> 16, other
collectives keep 2"). 0 #wrong, 3 reps. vs the pre-floor auto run (2ch everywhere):

| size    | Ring  |  PAT  | Bine  | B/PAT | pre-floor Bine | delta       |
|---------|------:|------:|------:|------:|---------------:|-------------|
| 8 MB    |  3.54 |  6.61 |  3.70 | 0.56  |  3.20          | ~same (base)|
| 16 MB   |  7.75 |  6.88 |  4.56 | 0.66  |  3.73          | ~same (base)|
| 33 MB   |  9.57 |  7.58 |  5.94 | 0.78  |  6.93          | ~same (base)|
| 67 MB   |  8.43 |  7.35 |  7.47 | 1.02  |  7.76          | ~same (base)|
| 128 MB  | 11.44 |  7.92 |  8.74 | 1.10  |  7.55          | +1.2 (FLOOR)|
| 256 MB  | 10.43 |  8.26 |  9.00 | 1.09  |  7.70          | +1.3 (FLOOR)|
| 512 MB  | 11.13 |  8.61 |  8.82 | 1.02  |  7.43          | +1.4 (FLOOR)|
| 1 GB    | 11.14 |  8.81 |  9.59 | 1.09  |  7.74          | +1.85(FLOOR)|

READ: (1) FLOOR WORKS AS DESIGNED: >=128 MB ops silently ran the 16-channel relay and
gained +1.2..+1.85 GB/s (1 GB 9.59 = best default-config Bine number recorded); <128 MB
(incl. PAT/Ring at ALL sizes) unchanged = the base-budget clamp works. (2) Bine now beats
PAT >=128 MB (1.02-1.10x here; PAT ran hot in this allocation at 8.3-8.8 -- typical
ratios 1.1-1.2x) WITH ZERO CONFIGURATION. (3) Known gaps unchanged and documented:
<=2 MB (pf=1), 8-33 MB soft band (2ch relay/ramp), and at 64n Ring still owns >=16 MB
absolutely (11+ GB/s) -- the tree-vs-tree (PAT-slot) win is the claim, plus auto-SELECT
model constants remain TODO for the tuner to actually pick Bine over Ring where it wins
(128n mid-range). DEPLOYMENT STORY CLOSED: correct + hang-free to 128n, wins >=128 MB
out of the box, all knobs (NCCL_BINE_NCHANNELS/_MINSIZE, NCCL_BINE_XOVER) runtime-tunable.

## 2026-07-14 — v11 at 128 NODES, AUTO CHANNELS (first deployed-default 128n run, 1 rep):
## BINE IS THE BEST ALGORITHM OVERALL; RING WINS NOTHING

Build 188acdf+, no channel forcing (base 2, floor->16 for Bine >=128 MB). 0 #wrong.

| size    | Ring |  PAT | Bine | B/PAT | B/Ring | best  |
|---------|-----:|-----:|-----:|------:|-------:|-------|
| 1 MB    | 0.27 | 0.72 | 0.52 | 0.72  |  1.93  | PAT   |
| 2 MB    | 0.20 | 1.04 | 1.09 | 1.05  |  5.45  | BINE  |
| 4 MB    | 0.40 | 1.64 | 1.81 | 1.10  |  4.52  | BINE  |
| 8 MB    | 0.65 | 2.17 | 1.56 | 0.72  |  2.40  | PAT   |
| 16 MB   | 1.25 | 3.19 | 1.12 | 0.35  |  0.90  | PAT (Bine soft band) |
| 33 MB   | 1.90 | 4.55 | 2.53 | 0.56  |  1.33  | PAT   |
| 67 MB   | 3.53 | 5.07 | 3.96 | 0.78  |  1.12  | PAT   |
| 128 MB  | 3.14 | 5.74 | 6.59 | 1.15  |  2.10  | BINE  |
| 256 MB  | 5.24 | 5.82 | 6.09 | 1.05  |  1.16  | BINE  |
| 512 MB  | 6.29 | 5.26 | 8.56 | 1.63  |  1.36  | BINE  |
| 1 GB    | 6.95 | 5.98 | 9.38 | 1.57  |  1.35  | BINE  |
| **avg** | 1.09 | 1.56 | 1.60 |       |        | BINE  |

READS: (1) THE SCALE THESIS CONFIRMED AT DEPLOYED DEFAULTS: at 64n Ring owned everything
>=16 MB; at 128n Ring wins NOT A SINGLE SIZE (127-hop pipeline cannot fill; 1 GB only 6.95).
Bine is the best algorithm at 2-4 MB and everything >=128 MB, and best ON AVERAGE
(1.60 > PAT 1.56 > Ring 1.09) -- out of the box, zero configuration. (2) The channel floor
delivers at scale: 512 MB-1 GB at 8.56-9.38 GB/s = 1.57-1.63x PAT (PAT at 2ch is weak at
128n large: 5.3-6.0). 1 GB 9.38 is the best 128n Bine number recorded in any config.
(3) Soft band persists, shifted with n: 16 MB craters at 0.35 (slice 64 KB -> relay at
2ch; the known channel-budget-dependent crossover artifact), 8-67 MB belongs to PAT.
(4) 1 rep -- ratios >=1.5 at large are far beyond noise; a 3-rep re-run would make it
citable. NEXT: 256n (guard boundary; predicted: window widens further).

### 3-REP CONFIRMATION (same allocation): the capstone is citable

avg per rep: Bine 1.83/1.67/1.74 vs PAT 1.61/1.49/1.54 vs Ring 1.18/1.14/1.21 -- Bine best
on average in EVERY rep. 3-rep means: 2 MB 1.09, 4 MB 1.61, 8 MB 1.73 (butterfly band,
best-of-all); 128 MB 1.13, 256 MB 1.38, 512 MB 1.45, 1 GB 1.45 (floor/relay band,
best-of-all); Ring wins nothing. DIP CONFIRMED at 16-67 MB (0.50/0.57/0.82): exactly the
sizes that run the RELAY on the BASE 2-channel budget -- bounded below by the 48 KB
crossover (16 MB slice = 64 KB > 48 KB ends the butterfly) and above by the 128 MB channel
floor (16ch begins). In that window the relay has 2 pipelines and sub-plateau slices
(64-256 KB) while PAT sits at its design center (2ch + aggregation). Butterfly rescue is
SAFETY-blocked at 33-67 MB (2ch slices 128-256 KB pack only 4/2 per slot < minPost 8);
threshold rescue works only for 16 MB; extending the floor downward is n-dependent (helps
33 MB at 128n per the 16ch data, HURTS the same band at 64n). Documented as the known
structural band; per-job mitigation NCCL_BINE_XOVER=131072 (fixes 16 MB only).

### 48 KB CONFIRMED on HW (128n/16ch, NCCL_BINE_XOVER=49152 at runtime, 1 rep, same alloc)

Prediction hit exactly: 128 MB 5.12 -> 7.21 GB/s (0.77 -> 1.05, relay) with 67 MB still
butterfly; everything else within noise of the 64 KB runs. Full >=128 MB band now wins:
1.05 / 1.13 / 1.16 / 1.19 (128M/256M/512M/1G). This run also had 8-16 MB at 1.05 and
avg Bine 2.00 > PAT 1.94 -- FIRST avg win at 128 nodes (1 rep; the stable claim remains
"parity avg, wins >=128 MB"). Runtime knob validated end-to-end (no rebuild needed; note
the env var must be NCCL_BINE_XOVER -- plain BINE_XOVER is silently ignored, tested).
Remaining soft spots unchanged: <=4 MB (0.25-0.77, pf=1 + forced-ch artifact) and the
33-67 MB butterfly band (0.73-0.90).

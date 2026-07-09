# Bine AllGather — correctness & performance plan

Execution plan for the Bine AllGather implementation in this NCCL fork
(fork point: upstream `5067397`). Written to be executed phase by phase by a
Claude Code session. Every phase has a **gate**; do not start the next phase
until the gate passes. Design decisions in this plan were pre-verified with
the simulator in `bench_bine/verify_schedule.py` — run it before touching
anything to see the expected baseline output.

**Goals**
1. Remove leftover debug code (it contaminates every published benchmark).
2. Fix the FIFO deadlock so Bine runs correctly at 64–256 ranks.
3. Make Bine competitive with PAT across the whole size sweep (it already
   wins at ≥16 MB; the small/mid range is the gap).

---

## Architecture recap (read first)

`NCCL_ALGO=PAT` for AllGather runs the Bine schedule. One class drives
everything: `PatAGAlgorithm<T>` in `src/include/collectives.h` (~line 697).

- It builds, at construction, an **op list** `(opKind, opRdim, opSdim, opSrc)`
  that depends ONLY on `(rank, nranks)`. Kinds: `0` INIT (copy own block
  input→output), `1` SEND-only (reads userOutput, forwards a block), `2`
  RECV-only (leaf), `3` FUSED (receive + forward in one copy).
- `getNextOp()` replays the list once per chunk (`offset` advances by
  `chunkCount` until `end`), filling `ncclPatStep` (`inpIx/outIx/recvOffset/
  sendOffset/postRecv/postSend/stepOffset/nelem/last`).
- **Device**: `src/device/all_gather.h` (~117–189) — one compute thread
  constructs the class and emits ops into a shmem ring; worker group(s)
  execute `prims.patCopy(ps, shmem)` per op (`src/device/prims_simple.h`
  ~1068). `parallelFactor` is 1: a single worker group executes ops strictly
  in order.
- **Host proxy**: `src/proxy.cc` (~715, `ncclPatternPatDown`) replays the SAME
  class with `T=char` to count network steps per dim. Anything that changes
  the op list automatically stays host/device-consistent because both sides
  run the same code.
- **Connections**: dim `k` ⇔ peer `binePi(rank,k,nranks)` for BOTH send and
  recv (involution ⇒ symmetric). Wired in `src/transport/generic.cc` (~87)
  and `src/proxy.cc` (~736). `src/graph/tuning.cc` (~308) restricts the algo
  to power-of-two `nRanks ≤ 256`.

### Invariants — breaking any of these corrupts data or hangs

1. **Per-connection sequence equality**: for every rank `r` and dim `k`, the
   ordered list of blocks `r` sends on dim `k` must equal the ordered list of
   blocks `binePi(r,k)` expects to receive on its dim `k`. (FIFOs are ordered;
   there is no tag matching.)
2. **Read-after-gather**: a kind-1 op for block `s` reads `userOutput`; the op
   that gathers `s` (INIT or recv) must come EARLIER in the same list.
3. **Bounded-FIFO liveness**: each connection holds `NCCL_STEPS = 8` slots per
   side. Ranks execute ops sequentially, so the emission order must never let
   a send get more than the FIFO depth ahead of the partner's matching
   consumption. This is what `verify_schedule.py`'s FIFO simulation checks —
   the model is confluent, so a deadlock it finds WILL occur on hardware.
4. **Host/device lockstep**: the op list and post flags must be identical in
   the device kernel and the host proxy. Keep all schedule logic inside
   `PatAGAlgorithm`; never fork it between the two call sites
   (`all_gather.h:138`, `proxy.cc:721`).
5. **One post per slot**: `postSend/postRecv` advance the FIFO step. Blocks
   sharing a slot must post exactly once (on the pack's last block), and pack
   boundaries must be identical on both endpoints of the connection.
6. `nOps ≤ RMAXOPS (520)` for every mode at every supported `nranks`.

### The verifier

```
cd bench_bine
python3 verify_schedule.py baseline   # pre-fix depth order: static OK, fifo DEADLOCK at n>=64 (the bug)
python3 verify_schedule.py phase2     # gate for Phase 2 (must print RESULT: PASS)
python3 verify_schedule.py phase4     # gate for Phase 4 (must print RESULT: PASS)
python3 timed_sim.py                  # throughput model: candidate emission orders side by side
```

`timed_sim.py` exists because liveness is necessary but NOT sufficient: plain
source order passes every liveness gate yet measured ~0.1x PAT on hardware.
Any emission-order change must look sane in the timed model too.

The Python classes mirror the C++. **Whenever you edit the C++ schedule, make
the identical edit to the mirror class and re-run.** If in doubt, print both
op lists (e.g. add a temporary host-side dump using `PatAGAlgorithm<char>`,
which already compiles on the host — `proxy.cc` proves it) and diff them.

---

## Phase 1 — strip debug code  (trivial, do first)

All are additions of this fork; upstream (`git show 5067397:<file>`) shows the
clean form.

1. `src/device/all_gather.h`: delete the four BINE-DBG printf lines
   (~142 `CONSTRUCT-DONE`, ~152 `COMPUTE-LOOPING`, ~155 `COMPUTE-DONE`,
   ~185 `WORKER-DONE`). Keep the surrounding logic (in particular
   `shmem->parallelFactor = patAlgo.getParallelFactor();` stays).
2. `src/device/prims_simple.h`, `patCopy`: in BOTH spin-wait loops
   (recv ~1095–1101, send ~1110–1116) remove `long long dbgs = 0;` and the
   `if (++dbgs == 500000000 ...) printf("BINE-STUCK ...")` lines, restoring:
   ```c
   int spins = 0;
   while (...) {
     peer->stepCache = loadStepValue(...);
     if (checkAbort(flags, Aborted, spins)) break;
   }
   ```
3. Grep to confirm no stragglers: `grep -rn "BINE-DBG\|BINE-STUCK\|dbgs" src/`
   must return nothing.

**Gate**: build succeeds (`make -j src.build` or the usual build), grep clean.

---

## Phase 2 — deadlock fix: SKEWED source-order emission (+ delete `depthOf`)

> **Amended 2026-07-09 after a 64-node Leonardo run.** The first version of
> this phase prescribed plain source order (`s = 0..n-1`). It is deadlock-free
> (0 #wrong, no hang at 64 nodes — the liveness fix worked) but measured
> **~0.1x PAT at every size** (~1 GB/s plateau): every rank consumes each
> block at the same wavefront instant its parent produces it, so every fused
> op stalls one full network hop, serialized. Liveness gates alone are NOT
> sufficient for an emission order — it must also pass the throughput model
> in `bench_bine/timed_sim.py`.

**Why**: the original emission groups ops by tree depth. Simulation shows this
deadlocks with 8-slot FIFOs at `n ≥ 64`: its per-connection bursts equal
binomial(log2(n)−1, d) — ≤6 at n=32 but 10 at n=64 — so dim-0 partners stall
symmetrically mid-depth-group. Plain source order is live but slow (above).
The fix is the one-parameter family between them: emit blocks in ascending

```
key(s) = s + λ · depth(s)        (ties by s;  λ = BINE_SKEW_LAMBDA = 6)
```

where `depth(s)` is this rank's depth in block s's broadcast tree (recorded
for free during the `getIdx` DFS — the separate O(n²·log n) `depthOf` pass,
~230k `binePi` calls per channel per call at n=256, is deleted). λ places a
block's ops ~λ positions later per tree-hop, so data arrives ~one hop before
it is consumed (throughput ≈ depth order's) while per-connection send/recv
skew stays bounded (liveness). Per-connection order consistency holds for ANY
λ: the receiver's depth is the sender's +1, so its keys are the sender's
shifted by exactly +λ — same relative order at both endpoints.

λ chosen by simulation: λ=6 is live down to FIFO depth 6 (two slots of margin
below the real 8) for all po2 n ≤ 256 and within ~5% of the model's best
throughput; λ=8 has zero margin; λ≥12 deadlocks at n≥64; λ=0 is the measured
10x regression.

**Edits in `src/include/collectives.h`, `PatAGAlgorithm`** (IMPLEMENTED):

1. Delete `depthOf()` and `inArr()` entirely. Instead, extend `getIdx()` with
   an optional `int* dep` output: `dep[i]` = DFS depth of `out[i]` below the
   subtree root (`e0` has depth 1). When called with `start == rank`, this is
   exactly rank's depth in `out[i]`'s broadcast tree, so the krecv pass fills
   `dof[s]` for free.
2. Emit blocks by an O(n²) min-key selection scan over
   `key(x) = x + BINE_SKEW_LAMBDA * dof[x]` (strict `<`, so ties resolve to
   the smaller block id — this must match the mirror's stable
   `sorted(..., key=(key, s))`). Per-block inner logic unchanged: INIT +
   own-block forwards for `s == rank`; FUSED + extra forwards, or leaf recv,
   otherwise.
3. `push()` overflow is loud (`__trap()` on device, `abort()` on host), and
   `#include <stdlib.h>` was added for the host path.
4. `BINE_SKEW_LAMBDA` (=6) is a `#define` above the class; it MUST stay in
   lockstep with `BINE_SKEW_LAMBDA` in `verify_schedule.py`.

**Notes**: `getNextOp`, `patCopy`, `proxy.cc`, `generic.cc`, and the tuning
guard need NO changes in this phase. The op multiset is unchanged (verified by
the harness), so per-dim step counts and hence proxy behavior are identical.

**Gate** (all three, after ANY change to the emission):
1. `python3 verify_schedule.py phase2` → `RESULT: PASS` (static invariants +
   FIFO liveness at depth 8 and at depth 6 — the margin — for 1 and 3 chunks).
2. `python3 timed_sim.py` → the shipped order must be within ~2x of the best
   LIVE candidate at every n (this is the gate plain source order failed).
3. Byte-compare the real C++ op lists against the mirror (standalone host
   compile of the constructor logic; `proxy.cc` proves the header compiles on
   host) for all po2 n ≤ 256, all ranks.

---

## Phase 3 — rebuild, hardware validation, clean re-benchmark

1. Rebuild both trees per `bench_bine/README.md` / `build_leonardo.sh`.
2. Correctness sweep (`all_gather_perf -b 8 -e 1G -f 2 -c 1`), ascending node
   counts: 4, 8, 16, 32, then 64 and 128 (the previously-deadlocking regime).
   Required: `0` wrong everywhere, no hangs. If ANY hang occurs, stop and
   bisect with the simulator — do not tweak timing-related knobs to paper
   over it.
3. Performance sweep vs PAT baseline and Ring (existing `run_compare.slurm` /
   `final_compare.sh`), same allocation, ≥3 reps, default channels plus
   forced-equal channels (`NCCL_MIN/MAX_NCHANNELS` ∈ {2,4,8,16}).
4. Record results in a NEW dated section of `bench_bine/results.md`; note
   explicitly that this is the first printf-free, deadlock-fixed build.
   While editing: delete the truncated duplicate section (the first
   "CONFIRMED: Bine beats PAT at IDENTICAL channels" copy, ~lines 295–309)
   and `git add bench_bine/results.md` (it is currently untracked!).

**Gate**: 0 wrong at all node counts including 128; large-size (≥64 MB) busbw
at least matches the pre-fix build. Expect small/mid to improve (printf +
construction cost removed) but likely still trail PAT — that is Phase 4.

---

## Phase 4 — small/mid performance: butterfly mode (hybrid)

**Why**: the relay posts one block per FIFO slot and per network message;
upstream PAT packs many small blocks per slot (`postFreq`) — that is the
small-message gap. Fixed-size packing on the RELAY is **proven unsafe** (see
Do-NOTs). The safe home for packing is the **butterfly** schedule — the
`allgather_bine_block_by_block` from pico (the reference implementation this
fork follows), which this fork's own v2/v3 experiments already showed winning
small/mid against PAT. Butterfly and relay use the SAME per-step partner
`binePi(rank,t)`, so transport wiring, proxy, and `patCopy` all work
unchanged: the butterfly is just a different op list in the same class.

**Schedule** (mirror class `Butterfly` in `verify_schedule.py`, pre-verified):

- Op 0: INIT (kind 0).
- For round `t = nsteps-1` down to `0`, partner `p = binePi(rank, t)`:
  - SEND ops (kind 1, `sendDim = t`) for blocks `getIdx(p, t)` in ascending
    order — all already gathered (they are exactly own block ∪ the blocks
    received in rounds `t' > t`).
  - RECV ops (kind 2, `recvDim = t`) for blocks `getIdx(rank, t)` ascending.
- Packing: within a round's send (resp. recv) run, block `j` goes to slot
  offset `(j % postFreq) * nelem`; `postSend`/`postRecv` = 1 only on the last
  block of each pack and on the round's last block. `stepOffset` stays 0.
- `nOps = 2*nranks − 1` (511 at n=256 — fits RMAXOPS=520, do not add ops).

**Safety condition** (hard requirement, sim-verified: below it the butterfly
deadlocks — both partners stuff their send FIFOs): the largest round (`t=0`,
`n/2` blocks) must fit in the FIFO:

```
postFreq >= divUp(nranks/2, NCCL_STEPS)
```

**Implementation:**

1. Constructor: compute `postFreq` from the already-passed (currently voided)
   first argument: `postFreq = slotBytes / (chunkCount * sizeof(T))`
   (device: elements×sizeof(T); host proxy: T=char and chunkCount is
   `op->chunkSize` in bytes — the ratio is identical on both sides, which is
   what keeps them in lockstep). Clamp: `if (postFreq > nranks/2) postFreq =
   nranks/2`.
2. Mode select (identical on host and device by construction):
   `useButterfly = (postFreq >= divUp(nranks/2, NCCL_STEPS))`. Butterfly for
   small chunks, source-order relay otherwise. Add a compile-time override
   for benchmarking: `#define BINE_FORCE_RELAY` / `BINE_FORCE_BUTTERFLY`.
3. Per-op metadata: add two arrays alongside `opSrc/opRdim/opSdim/opKind`:
   `unsigned char opSlotPos[RMAXOPS]` (block's position in its pack, ≤127)
   and `unsigned char opPost[RMAXOPS]` (bit0 = postSend, bit1 = postRecv).
   Relay mode fills `opSlotPos = 0`, `opPost = both bits` (today's behavior).
4. `getNextOp()`: `recvOffset = opSlotPos * nelem` (recv ops), `sendOffset =
   opSlotPos * nelem` (send ops); `postRecv/postSend` from `opPost` instead
   of constant 1. Everything else unchanged. (`patCopy` already consumes
   these fields — the machinery is upstream PAT's own packing support. The
   `connFifo[...].size = (sendOffset + nelem) * sizeof(T)` line already
   yields the correct pack size because the posting op is the pack's last.)
5. `parallelFactor` stays 1 in both modes.
6. Keep the Python mirror (`Butterfly` class) in lockstep with the C++.

**Gate**: `python3 verify_schedule.py phase4` → PASS. Then hardware: `-c 1`
correctness at 4–128 nodes across the FULL size sweep (the mode boundary must
be crossed: verify sizes just below and above the butterfly/relay switch).
Then benchmark small/mid (8 KB–8 MB) vs PAT; record the crossover and, if the
auto threshold is not optimal, adjust the mode condition (e.g. also require
`chunkCount * nranks ≤ X`) based on data, re-running both gates after.

**Stretch (Phase 4b, optional — only if small/mid still trails PAT):**
restore worker-group concurrency (`parallelFactor > 1`, upstream used up to
16). This is the remaining structural gap vs PAT at small sizes, but it is
NOT safe to just flip on. Races to solve first (enumerate + design before
coding): (a) kind-1 extra-forwards read `userOutput` written by a fused op a
few positions earlier — concurrent groups race unless producer/consumer are
≥ nGroups apart in the list; (b) `peer->step` in the shared shmem peer struct
is read/written by whichever group executes an op on that dim — same-dim
posting ops must be ≥ nGroups apart; (c) in butterfly packs, the pack's
posting op must not run before other groups finished writing the same slot.
Any design must pass a grouped-execution extension of the simulator (model:
`nGroups` cursors, group `g` executes ops `g, g+nGroups, ...` sequentially)
plus hardware `-c 1` at every node count. If this proves invasive, stop —
Phases 1–4 already deliver correct scaling + a competitive sweep.

---

## Phase 5 — optional hygiene (any order, low risk)

- **ReduceScatter landmine** (1-line fix): this fork removed the connections
  RS-PAT needs (upstream's `// ReduceScatter` / `// AllGather` comments are
  swapped relative to actual use). In `src/transport/generic.cc` (~84) change
  `ncclTransportP2pConnect(comm, c, 1, &prevPeer, 1, &nextPeer, 0)` to
  `... 1, &nextPeer, 1, &prevPeer, 0` (RS receives from `rank−2^i`, sends to
  `rank+2^i`) and fix the comment. Until then, add a README warning that RS
  with `NCCL_ALGO=PAT` is broken on this branch.
- README: document that non-po2/>256 with FORCED `NCCL_ALGO=PAT` errors out
  ("No algorithm/protocol available") rather than falling back to Ring —
  only automatic selection falls back. Document the 256-rank cap.
- Remove the now-stale `(void)stepDepth; (void)maxParallelFactor;` params or
  put them to use (Phase 4 uses `slotBytes`).
- Squash/rename the 17 identical "[ADD] Trying to implement Bine." commits.
- If auto-selection ever matters (today you force PAT): revisit
  `tuning.cc:342` (`busBw *= .75`) and the PAT latency constants with the
  Phase 3/4 measurements.

---

## Do-NOTs (each was tested; do not re-litigate without new evidence)

1. **Do not add fixed-size slot packing to the relay.** Simulated: deadlocks
   for every pack size ≥ 2, even at n=32 — a receiver stalls on a full-pack
   post while the sender, mid-pack, needs a block back from that receiver on
   the same pairwise dim. Packing is only safe in butterfly rounds under the
   `postFreq` condition.
2. **Do not "fix" the deadlock by raising `NCCL_STEPS`** — required depth
   grows ~n/8 (needs 64 slots at n=256), and it doubles buffer memory
   globally. The emission order is the correct fix.
3. **Do not enable `parallelFactor > 1` casually** — see Phase 4b races.
4. **Do not reorder ops** (any mode) without re-running BOTH
   `verify_schedule.py` (liveness, incl. the depth-6 margin runs) AND
   `timed_sim.py` (throughput) and updating the mirror class. A live order
   can still be 10x slow — see #6.
5. **Do not compare against the old results.md numbers** — they were measured
   with device printfs in the hot path.
6. **Do not use plain source order (λ=0)** — deadlock-free but MEASURED
   ~0.1x PAT at 64 nodes on Leonardo (2026-07-09): consumption is synchronous
   with production, so every fused op eats a serialized network hop. And do
   not raise `BINE_SKEW_LAMBDA` past 8: λ=8 already has zero FIFO-depth
   margin, λ≥12 deadlocks at n≥64.

## Acceptance criteria (project done when all hold)

1. `python3 bench_bine/verify_schedule.py all` → `RESULT: PASS`.
2. `grep -rn "BINE-DBG\|BINE-STUCK" src/` → empty.
3. `all_gather_perf -c 1`: 0 wrong, no hang, at 4/8/16/32/64/128 nodes
   (and 256 if obtainable), full 8 B–1 GB sweep, both modes exercised.
4. busbw ≥ PAT at every size ≥ 16 MB; ≥ 0.9× PAT at 64 KB–8 MB (stretch:
   ≥ 1.0× — Phase 4b territory); results recorded in results.md with the
   build/commit noted, and results.md committed to git.

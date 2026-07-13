#!/usr/bin/env python3
"""How far past the current 256-rank guard does the Bine schedule scale?

Checks, for n = 512, 1024, 2048 (all beyond today's tuning.cc cap):
  1. STATIC schedule math at lambda=6: partition/coverage, single reception,
     read-after-gather, per-connection send/recv sequence equality (the FIFO
     wire-order invariant), and the required op-list size (vs RMAXOPS=520).
  2. RELAY LIVENESS: bounded-FIFO deadlock-freedom of the skewed source order
     at FIFO depth 8 (real) and 6 (margin), for lambda in {4, 6, 8} -- the
     lambda=6 proof currently extends only to n=256, and its safety margin is
     known to narrow as n grows (lambda>=12 already deadlocks at n>=64).
  3. BUTTERFLY LIVENESS at the packing-safety floor minPost = divUp(n/2, 8)
     (and one below it, expecting deadlock), via the pack-aware engine.

Liveness engines: verify_schedule.fifo_sim (greedy, exact) where affordable,
group_sim.timed_grouped (event-driven, same FIFO semantics, P=1) for large n.
Both are confluent models: a deadlock found is schedule-independent.

Run: python3 scale_study.py [maxn]   (default 2048)
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_schedule import (RelaySkewOrder, Butterfly, binePi, fifo_sim,
                             NCCL_STEPS)
from group_sim import timed_grouped

RMAXOPS_TODAY = 520


def static_checks_fast(n, lam=6):
    """O(n^2)-ish single-pass version of the static invariants."""
    t0 = time.time()
    errs = []
    max_ops = 0
    seqs = {}
    for r in range(n):
        a = RelaySkewOrder(0, 1, 1, 1, r, n, lam=lam)
        max_ops = max(max_ops, len(a.ops))
        if a.krecv_multi:
            errs.append(f"r{r}: krecv overwritten (partition violated)")
        have = set()
        seen = set()
        snd = {}
        rcv = {}
        for (K, rd, sd, s) in a.ops:
            if K == 1 and s not in have:
                errs.append(f"r{r}: sends {s} before gathered")
            if rd >= 0:
                if s in seen:
                    errs.append(f"r{r}: recv {s} twice")
                seen.add(s)
                rcv.setdefault(rd, []).append(s)
            if rd >= 0 or K == 0:
                have.add(s)
            if sd >= 0:
                snd.setdefault(sd, []).append(s)
        if len(seen) != n - 1:
            errs.append(f"r{r}: coverage {len(seen)} != {n - 1}")
        seqs[r] = (snd, rcv)
    nst = (n - 1).bit_length()
    for r in range(n):
        for k in range(nst):
            p = binePi(r, k, n)
            if seqs[r][0].get(k, []) != seqs[p][1].get(k, []):
                errs.append(f"conn r{r}->r{p} dim{k}: sequence mismatch")
                break
    return errs, max_ops, time.time() - t0


def relay_liveness(n, lam, depth):
    """Event-driven liveness (None = deadlock). Confluent model."""
    t0 = time.time()
    ms = timed_grouped(n, RelaySkewOrder, {"lam": lam}, depth=depth,
                       o_exec=1.0, l_hop=5.0, l_cred=2.0)
    return ('OK' if ms is not None else 'DEADLOCK'), time.time() - t0


def butterfly_liveness(n, P, depth=NCCL_STEPS):
    t0 = time.time()
    ms = timed_grouped(n, Butterfly, {"postFreq": P}, depth=depth,
                       o_exec=1.0, l_hop=5.0, l_cred=2.0)
    return ('OK' if ms is not None else 'DEADLOCK'), time.time() - t0


def main():
    maxn = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    sizes = [s for s in (512, 1024, 2048, 4096) if s <= maxn]
    for n in sizes:
        print(f"===== n = {n} =====", flush=True)
        errs, mx, dt = static_checks_fast(n)
        need_bfly = 2 * n - 1
        print(f"static (lam=6): {'OK' if not errs else 'FAIL: ' + errs[0]}"
              f"  [{dt:.0f}s]", flush=True)
        print(f"op-list size: relay max {mx}, butterfly {need_bfly}"
              f"  (today RMAXOPS={RMAXOPS_TODAY} -> need >= {max(mx, need_bfly)})",
              flush=True)
        for lam in (4, 6, 8):
            row = []
            for depth in (8, 6):
                res, dt = relay_liveness(n, lam, depth)
                row.append(f"d{depth}:{res}[{dt:.0f}s]")
            print(f"relay lam={lam}: " + "  ".join(row), flush=True)
        minP = (n // 2 + NCCL_STEPS - 1) // NCCL_STEPS
        res, dt = butterfly_liveness(n, minP)
        resb, dtb = butterfly_liveness(n, max(1, minP // 2))
        print(f"butterfly: minPost={minP} -> {res}[{dt:.0f}s];"
              f" below floor ({minP // 2}) -> {resb}[{dtb:.0f}s] (expect DEADLOCK)",
              flush=True)
        # exact greedy cross-check where affordable
        if n <= 512:
            r = fifo_sim(n, RelaySkewOrder, nchunks=1, depth=8, lam=6)
            r6 = fifo_sim(n, RelaySkewOrder, nchunks=1, depth=6, lam=6)
            print(f"greedy cross-check (exact): d8={r} d6={r6}", flush=True)
        print(flush=True)


if __name__ == '__main__':
    main()

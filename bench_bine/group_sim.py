#!/usr/bin/env python3
"""Phase 4b design study: grouped execution (parallelFactor > 1) for Bine AllGather.

Device semantics being modeled (src/device/all_gather.h):
  - P worker groups; op i is executed by group i % P; each group is sequential.
  - Groups are NOT synchronized with each other. Two ops in different groups may
    overlap or reorder arbitrarily, bounded only by the shmem op ring:
    NCCL_SHMEM_PAT_STEPS = 32 slots, so op j cannot even be dispatched until op
    j-32 has completed (this also provides the ONLY cross-group happens-before).

Part 1 - RACE CHECK on today's op lists if P were simply raised:
  hazard A (output RAW): a SEND-only op reads userOutput[s] written by the
    earlier op that gathered s. Different group + index gap < 32 => can overlap
    or reorder => corruption.
  hazard B (per-dim FIFO state): ops sharing a send-dim (or recv-dim) read-
    modify-write the same shmem peer struct (peer->step etc.). Different group +
    gap < 32 => race/deadlock. (Same-dim posts must also stay in order.)

Part 2 - TIMED comparison of candidates in the small/mid (latency-bound) regime:
  baseline : today's fused relay, P=1
  bfly     : butterfly with slice-based packing (current small-message mode)
  dimpar   : PHASE 4b CANDIDATE: UNFUSED relay (recv ops and forward ops are
             separate, like v6), groups own dim-SIDES (send side of dim k /
             recv side of dim k), forwards wait for their block's recv via an
             explicit cross-group dependency (device would need a per-group
             completed-step counter in shmem -- small, safe addition).
             Unfusing costs an extra output read per forward (~2x copy traffic)
             but copy time is negligible in this regime.

Run: python3 group_sim.py
"""
import heapq
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_schedule import (RelayBase, RelaySkewOrder, Butterfly, binePi,
                             NCCL_STEPS)

RING = 32  # NCCL_SHMEM_PAT_STEPS


# --------------------------- Part 1: race check ----------------------------

def race_check(n, P, ops):
    """Count hazard-A and hazard-B pairs if ops ran with P groups (op i -> i%P)."""
    gathered_at = {}  # block -> op index that wrote userOutput[s]
    a = b = 0
    last_dim_op = {}  # ('s'|'r', dim) -> last op index using that dim side
    for i, (K, rd, sd, s) in enumerate(ops):
        if K in (0, 2, 3) or K == 0:  # INIT, RECV-only, FUSED write output[s]
            pass
        # hazard A: SEND-only (kind 1) reads output[s]
        if K == 1:
            w = gathered_at.get(s)
            if w is not None and (w % P) != (i % P) and (i - w) < RING:
                a += 1
        if K == 0 or rd >= 0:
            gathered_at[s] = i
        # hazard B: same dim-side within RING in different groups
        for side, d in (('s', sd), ('r', rd)):
            if d < 0:
                continue
            j = last_dim_op.get((side, d))
            if j is not None and (j % P) != (i % P) and (i - j) < RING:
                b += 1
            last_dim_op[(side, d)] = i
    return a, b


# ----------------------- unfused relay (candidate) -------------------------

class RelayUnfused(RelayBase):
    """v6-style unfused relay in skew order: for each block (skew order),
    a RECV op (or INIT), then one SEND op per child dim (reads output).
    Used only by the Phase 4b study; not implemented in C++."""

    def __init__(self, offset, end, count, chunkCount, rank, nranks, lam=6):
        super().__init__(offset, end, count, chunkCount, rank, nranks)
        n = nranks
        dof = [0] * n
        for k in range(self.nsteps):
            e0 = binePi(rank, k, n)
            dof[e0] = 1
            node = [e0]; sval = [k + 1]; dstk = [1]
            while node:
                y = node[-1]; s_ = sval[-1]; d = dstk[-1]
                if s_ >= self.nsteps:
                    node.pop(); sval.pop(); dstk.pop(); continue
                sval[-1] = s_ + 1
                c = binePi(y, s_, n)
                dof[c] = d + 1
                node.append(c); sval.append(s_ + 1); dstk.append(d + 1)
        order = sorted(range(n), key=lambda s: (s + lam * dof[s], s))
        for s in order:
            if s == rank:
                self.ops.append((0, -1, -1, s))          # INIT
            else:
                self.ops.append((2, self.krecv[s], -1, s))  # RECV-only
            for k in range(self.nsteps):
                if self.cmask[s] & (1 << k):
                    self.ops.append((1, -1, k, s))          # SEND-only fwd


def dimside_groups(algo):
    """Group assignment for the dim-partition candidate: each op is owned by its
    dim-side; INIT joins the busiest send side it feeds (dim of first child).
    Returns (group id per op, number of groups)."""
    gid = {}
    nxt = 0
    out = []
    for (K, rd, sd, s) in algo.ops:
        key = ('r', rd) if rd >= 0 else ('s', sd if sd >= 0 else -1)
        if key not in gid:
            gid[key] = nxt
            nxt += 1
        out.append(gid[key])
    return out, nxt


# ------------------------ Part 2: timed grouped sim ------------------------

def timed_grouped(n, algo_cls, kw, group_map_fn=None, o_exec=2.0, l_hop=22.0,
                  l_cred=15.0, depth=NCCL_STEPS, ring=RING, dep_wait=True,
                  max_us=30_000_000.0):
    """Event-driven grouped execution. group_map_fn(algo) -> (groups, P);
    default = P=1. Butterfly packing honored via algo.meta if present.
    Returns makespan us or None (deadlock/livelock)."""
    algos = {r: algo_cls(0, 1, 1, 1, r, n, **(kw or {})) for r in range(n)}
    groups = {}
    nG = {}
    for r, a in algos.items():
        if group_map_fn is None:
            groups[r] = [0] * len(a.ops)
            nG[r] = 1
        else:
            groups[r], nG[r] = group_map_fn(a)
    nst = algos[0].nsteps
    # per-connection: posted packs (arrival times of full packs) + credits
    arrivals = {(r, k): [] for r in range(n) for k in range(nst)}
    consumed = {(r, k): 0 for r in range(n) for k in range(nst)}
    credits = {(r, k): depth for r in range(n) for k in range(nst)}
    openpack = {(r, k): 0 for r in range(n) for k in range(nst)}  # blocks in open pack
    done_op = {r: [False] * len(algos[r].ops) for r in range(n)}
    done_pfx = {r: 0 for r in range(n)}       # ops [0, pfx) all complete (ring cap)
    have_block = {r: {} for r in range(n)}    # block -> completion time on r
    cursor = {r: {} for r in range(n)}        # group -> next op LIST position
    oplist_by_group = {r: {} for r in range(n)}
    for r, a in algos.items():
        for pos, g in enumerate(groups[r]):
            oplist_by_group[r].setdefault(g, []).append(pos)
        for g in oplist_by_group[r]:
            cursor[r][g] = 0
    free_at = {(r, g): 0.0 for r in range(n) for g in oplist_by_group[r]}
    total = sum(len(a.ops) for a in algos.values())
    ndone = 0
    finish = 0.0
    seq = 0
    pq = [(0.0, i, (r, g)) for i, (r, g) in
          enumerate((r, g) for r in range(n) for g in oplist_by_group[r])]
    heapq.heapify(pq)
    seq = len(pq)

    def meta(a, i):
        m = getattr(a, 'meta', None)
        return m[i] if m is not None else (0, True)  # relay: post every op

    def try_group(r, g, t):
        nonlocal ndone, finish, seq
        a = algos[r]
        lst = oplist_by_group[r][g]
        while cursor[r][g] < len(lst):
            i = lst[cursor[r][g]]
            K, rd, sd, s = a.ops[i]
            _, posts = meta(a, i)
            t_start = max(t, free_at[(r, g)])
            if i - done_pfx[r] >= ring:
                return  # op-ring full; retried when prefix advances
            if K == 1 and dep_wait:
                dep = have_block[r].get(s)
                if dep is None:
                    return  # producer not done; producer completion re-wakes
                t_start = max(t_start, dep)
            if rd >= 0:
                conn = (binePi(r, rd, n), rd)
                ci = consumed[conn]
                if ci >= len(arrivals[conn]):
                    return
                if arrivals[conn][ci] > t_start:
                    heapq.heappush(pq, (arrivals[conn][ci], seq, (r, g))); seq += 1
                    return
            if sd >= 0 and posts and credits[(r, sd)] <= 0:
                return
            t_end = t_start + o_exec
            if rd >= 0:
                conn = (binePi(r, rd, n), rd)
                if posts:  # pack fully consumed -> credit home
                    consumed[conn] += 1
                    heapq.heappush(pq, (t_end + l_cred, seq, ('cred', conn))); seq += 1
                have_block[r][s] = t_end
            if K == 0:
                have_block[r][s] = t_end
            if sd >= 0:
                openpack[(r, sd)] += 1
                if posts:
                    credits[(r, sd)] -= 1
                    heapq.heappush(pq, (t_end + l_hop, seq,
                                        ('arr', (r, sd), t_end + l_hop))); seq += 1
                    openpack[(r, sd)] = 0
            done_op[r][i] = True
            while done_pfx[r] < len(done_op[r]) and done_op[r][done_pfx[r]]:
                done_pfx[r] += 1
            free_at[(r, g)] = t_end
            finish = max(finish, t_end)
            cursor[r][g] += 1
            ndone += 1
            t = t_end
        return

    def wake_all_of_rank(r, t):
        # loop to fixpoint: a completion in one group can unblock another group
        # of the same rank (ring advance / producer dep) already swept this pass
        while True:
            before = ndone
            for g in oplist_by_group[r]:
                try_group(r, g, t)
            if ndone == before:
                break

    while pq and ndone < total:
        t, _, ev = heapq.heappop(pq)
        if t > max_us:
            return None
        if isinstance(ev, tuple) and ev[0] == 'cred':
            credits[ev[1]] += 1
            wake_all_of_rank(ev[1][0], t)
        elif isinstance(ev, tuple) and ev[0] == 'arr':
            (snd, k) = ev[1]
            arrivals[(snd, k)].append(ev[2])
            wake_all_of_rank(binePi(snd, k, n), t)
        else:
            r, g = ev
            try_group(r, g, t)
            wake_all_of_rank(r, t)   # cross-group deps within the rank
    return finish if ndone == total else None


def main():
    print("== Part 1: races if parallelFactor were simply raised on TODAY's lists ==")
    print(f"{'list':16s} {'n':>4s} {'P':>3s} {'RAW(out)':>9s} {'dim-state':>10s}")
    for n in (64, 128):
        for P in (2, 4, 8, 16):
            a = RelaySkewOrder(0, 1, 1, 1, 1, n)
            ra, rb = race_check(n, P, a.ops)
            print(f"{'relay(skew)':16s} {n:4d} {P:3d} {ra:9d} {rb:10d}")
    print("=> nonzero everywhere: naive P>1 corrupts (RAW) and races the FIFO state.")
    print()

    print("== Part 2: timed, small/mid latency-bound regime (o=2us hop=22us cred=15us) ==")
    print("   makespan per chunk -> effective speedup vs today's P=1 fused relay")
    print(f"{'n':>4s} {'fusedP1':>9s} {'bfly(pack)':>11s} {'dimpar':>9s}"
          f" {'bfly speedup':>13s} {'dimpar speedup':>15s} {'dimpar groups':>14s}")
    for n in (64, 128):
        base = timed_grouped(n, RelaySkewOrder, None)
        minP = max(1, math.ceil((n // 2) / NCCL_STEPS))
        pack = min(n // 2, 64)  # slice-based packing (post-fix) at tiny sizes
        bf = timed_grouped(n, Butterfly, {'postFreq': pack})
        _, ng = dimside_groups(RelayUnfused(0, 1, 1, 1, 1, n))
        dp = timed_grouped(n, RelayUnfused, None, group_map_fn=dimside_groups)
        fmt = lambda x: f"{x:9.0f}" if x else "  DEADLK "
        print(f"{n:4d} {fmt(base)} {fmt(bf):>11s} {fmt(dp)}"
              f" {base / bf if bf else 0:13.2f} {base / dp if dp else 0:15.2f} {ng:14d}")
    print()
    print("Load-balance ceiling for dim-partition (op counts per dim-side, n=128):")
    a = RelayUnfused(0, 1, 1, 1, 1, 128)
    from collections import Counter
    c = Counter()
    for (K, rd, sd, s) in a.ops:
        c[('r', rd) if rd >= 0 else ('s', sd)] += 1
    tot = sum(c.values())
    mx = max(c.values())
    print(f"  total ops {tot}, largest dim-side {mx} -> speedup ceiling {tot / mx:.1f}x"
          f" (dim-0 side owns half of one direction)")


if __name__ == '__main__':
    main()

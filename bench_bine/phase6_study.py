#!/usr/bin/env python3
"""Phase 6 go/no-go: can a PARALLEL RELAY (unfused ops + dim-side worker groups +
cross-group dependency counters) win the 16-128 MB band at 128 nodes?

VERSION 2 -- with a SHARED-NIC constraint. Version 1 gave every connection an
independent 12.5 GB/s link; it could not calibrate (model 3.2-4.9x faster than the
measured anchors even at max latency), i.e. it over-credited exactly the parallelism
the candidate exploits. Discarded. Here every rank has ONE egress and ONE ingress
resource (rate = NIC_BW / nChannels per channel-instance; channels are symmetric
mean-field shares of the pipe), each transfer occupies both endpoints' resources
for pack_bytes/rate, and (alpha, NIC_BW) are calibrated on two measured anchors
with a third measured point held out for validation:
    anchors : 67 MB @ 2ch = 5.66 GB/s   and   128 MB @ 16ch = 8.66 GB/s
    holdout : 512 MB @ 16ch = 9.36 GB/s          (all: fair envelope, 128n, 2 reps)

Candidates per (size, C): fused relay P=1 (today), packed butterfly (where safe),
DIMPAR = unfused relay + dim-side groups + cross-group dep waits (Phase 6).
Verdict: dimpar-best vs max(fair-PAT, fair-Ring) per size.
"""
import heapq
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_schedule import (RelaySkewOrder, Butterfly, binePi, NCCL_STEPS)
from group_sim import RelayUnfused, dimside_groups

N = 128
SLOT = 512 * 1024
MB = 1024 * 1024
RING_CAP = 32
SIZES = [16 * MB, 33 * MB, 67 * MB, 128 * MB]
FAIR = {16 * MB: (6.22, 4.96, 4.87), 33 * MB: (7.19, 7.32, 5.28),
        67 * MB: (7.31, 7.46, 5.66), 128 * MB: (7.58, 7.61, 8.66)}
ANCHORS = [(67 * MB, 2, 5.66), (128 * MB, 16, 8.66)]
HOLDOUT = (512 * MB, 16, 9.36)


def sim(n, algo_cls, kw, group_map_fn, slice_b, alpha, nic_gbs, C,
        copy_gbs=100.0, ring=RING_CAP, dep_wait=True, max_us=120_000_000.0):
    """Event-driven grouped execution with shared per-rank NIC resources.
    Returns makespan (us) of one chunk (one op-list pass), or None."""
    kw = kw or {}
    algos = {r: algo_cls(0, 1, 1, 1, r, n, **kw) for r in range(n)}
    if group_map_fn is None:
        groups = {r: [0] * len(algos[r].ops) for r in range(n)}
    else:
        groups = {r: group_map_fn(algos[r])[0] for r in range(n)}
    nG = {r: (max(groups[r]) + 1) for r in range(n)}
    nst = algos[0].nsteps
    rate = nic_gbs * 1e9 / C / 1e6  # bytes per us available to THIS channel instance
    o_exec = {r: 1.0 + nG[r] * slice_b / (copy_gbs * 1e3) for r in range(n)}

    arrivals = {(r, k): [] for r in range(n) for k in range(nst)}
    consumed = {(r, k): 0 for r in range(n) for k in range(nst)}
    credits = {(r, k): NCCL_STEPS for r in range(n) for k in range(nst)}
    openpk = {(r, k): 0 for r in range(n) for k in range(nst)}
    nic_out = {r: 0.0 for r in range(n)}
    nic_in = {r: 0.0 for r in range(n)}
    done_op = {r: [False] * len(algos[r].ops) for r in range(n)}
    done_pfx = {r: 0 for r in range(n)}
    have_block = {r: {} for r in range(n)}
    oplist_by_group = {r: {} for r in range(n)}
    for r in range(n):
        for pos, g in enumerate(groups[r]):
            oplist_by_group[r].setdefault(g, []).append(pos)
    cursor = {r: {g: 0 for g in oplist_by_group[r]} for r in range(n)}
    free_at = {(r, g): 0.0 for r in range(n) for g in oplist_by_group[r]}
    total = sum(len(a.ops) for a in algos.values())
    ndone = 0
    finish = 0.0
    seq_n = [0]
    pq = []

    def push(t, ev):
        heapq.heappush(pq, (t, seq_n[0], ev))
        seq_n[0] += 1

    for r in range(n):
        for g in oplist_by_group[r]:
            push(0.0, ('wake', r, g))

    def meta(a, i):
        m = getattr(a, 'meta', None)
        return m[i] if m is not None else (0, True)

    def try_group(r, g, t):
        nonlocal ndone, finish
        a = algos[r]
        lst = oplist_by_group[r][g]
        while cursor[r][g] < len(lst):
            i = lst[cursor[r][g]]
            K, rd, sd, s = a.ops[i]
            _, posts = meta(a, i)
            t_start = max(t, free_at[(r, g)])
            if i - done_pfx[r] >= RING_CAP:
                return
            if K == 1 and dep_wait:
                dep = have_block[r].get(s)
                if dep is None:
                    return
                t_start = max(t_start, dep)
            if rd >= 0:
                conn = (binePi(r, rd, n), rd)
                ci = consumed[conn]
                if ci >= len(arrivals[conn]):
                    return
                if arrivals[conn][ci][0] > t_start:
                    push(arrivals[conn][ci][0], ('wake', r, g))
                    return
            if sd >= 0 and posts and credits[(r, sd)] <= 0:
                return
            t_end = t_start + o_exec[r]
            if rd >= 0:
                conn = (binePi(r, rd, n), rd)
                blocks = arrivals[conn][consumed[conn]][1]
                have_block[r][s] = t_end
                if blocks <= 1 or consumed_partial_done(conn):
                    pass
                consumed[conn] += 1
                push(t_end + 0.6 * alpha, ('cred', conn))
            if K == 0:
                have_block[r][s] = t_end
            if sd >= 0:
                openpk[(r, sd)] += 1
                if posts:
                    credits[(r, sd)] -= 1
                    dst = binePi(r, sd, n)
                    pack_bytes = openpk[(r, sd)] * slice_b
                    openpk[(r, sd)] = 0
                    s0 = max(t_end, nic_out[r], nic_in[dst])
                    dur = pack_bytes / rate
                    nic_out[r] = s0 + dur
                    nic_in[dst] = s0 + dur
                    push(s0 + dur + alpha, ('arr', (r, sd), s0 + dur + alpha))
            done_op[r][i] = True
            while done_pfx[r] < len(done_op[r]) and done_op[r][done_pfx[r]]:
                done_pfx[r] += 1
            free_at[(r, g)] = t_end
            finish = max(finish, t_end)
            cursor[r][g] += 1
            ndone += 1
            t = t_end
        return

    def consumed_partial_done(conn):
        return True  # packs are consumed op-by-op; credit returns on pack end below

    def wake_rank(r, t):
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
        if ev[0] == 'cred':
            credits[ev[1]] += 1
            wake_rank(ev[1][0], t)
        elif ev[0] == 'arr':
            (snd, k) = ev[1]
            arrivals[(snd, k)].append((ev[2], 1))
            wake_rank(binePi(snd, k, n), t)
        else:
            _, r, g = ev
            wake_rank(r, t)
    return finish if ndone == total else None


def busbw(size, ms):
    return None if ms is None else size * (N - 1) / N / (ms * 1e-6) / 1e9


def run(kind, size, C, alpha, nic):
    sl = size // (N * C)
    if sl < 1024:
        return None
    if kind == 'fused':
        return busbw(size, sim(N, RelaySkewOrder, None, None, sl, alpha, nic, C))
    if kind == 'dimpar':
        return busbw(size, sim(N, RelayUnfused, None, dimside_groups, sl, alpha, nic, C))
    if kind == 'bfly':
        min_post = (N // 2 + NCCL_STEPS - 1) // NCCL_STEPS
        pf = min(N // 2, SLOT // sl)
        if pf < min_post:
            return None
        return busbw(size, sim(N, Butterfly, {'postFreq': pf}, None, sl, alpha, nic, C))
    raise ValueError(kind)


def calibrate():
    best = (None, None, 1e18)
    for alpha in (5, 10, 15, 20, 30, 40, 60):
        for nic in (6, 8, 10, 12.5, 15, 20):
            err = 0.0
            for size, C, meas in ANCHORS:
                m = run('fused', size, C, alpha, nic)
                if m is None:
                    err = 1e18
                    break
                err += ((m - meas) / meas) ** 2
            if err < best[2]:
                best = (alpha, nic, err)
    return best[0], best[1]


def main():
    alpha, nic = calibrate()
    print(f"Calibrated: ALPHA={alpha} us, NIC={nic} GB/s")
    for size, C, meas in ANCHORS:
        print(f"  anchor {size // MB:4d}MB @{C:2d}ch: model {run('fused', size, C, alpha, nic):5.2f} vs measured {meas:5.2f}")
    h = run('fused', HOLDOUT[0], HOLDOUT[1], alpha, nic)
    print(f"  HOLDOUT {HOLDOUT[0] // MB:4d}MB @{HOLDOUT[1]:2d}ch: model {h:5.2f} vs measured {HOLDOUT[2]:5.2f}"
          f"  (residual {abs(h - HOLDOUT[2]) / HOLDOUT[2] * 100:.0f}%)")
    print()
    hdr = (f"{'size':>6s} {'cfg':>5s} | {'fused':>6s} {'bfly':>6s} {'DIMPAR':>7s} {'dp/f':>5s} |"
           f" {'PATbest':>7s} {'Ringbest':>8s} {'today':>6s}")
    print(hdr)
    print('-' * len(hdr))
    verdict = {}
    for size in SIZES:
        pat, ring_t, today = FAIR[size]
        best_dp = (None, None)
        for C in (2, 4, 8, 16):
            f = run('fused', size, C, alpha, nic)
            b = run('bfly', size, C, alpha, nic)
            d = run('dimpar', size, C, alpha, nic)
            fmt = lambda x: f"{x:6.2f}" if x else "   n/a"
            rat = f"{d / f:5.2f}" if (d and f) else "     "
            print(f"{size // MB:4d}MB @{C:2d}ch | {fmt(f)} {fmt(b)} {fmt(d):>7s} {rat} |"
                  f" {pat:7.2f} {ring_t:8.2f} {today:6.2f}")
            if d and (best_dp[0] is None or d > best_dp[0]):
                best_dp = (d, C)
        verdict[size] = best_dp
        print()
    print("=== VERDICT (dimpar best vs max(PAT,Ring) fair targets) ===")
    for size in SIZES:
        pat, ring_t, today = FAIR[size]
        tgt = max(pat, ring_t)
        d, C = verdict[size]
        if d is None:
            print(f"{size // MB:4d} MB: dimpar failed")
            continue
        call = "WIN" if d >= tgt else ("PARITY" if d >= 0.95 * tgt else "LOSE")
        print(f"{size // MB:4d} MB: dimpar {d:5.2f} @{C}ch vs target {tgt:5.2f} (today {today:5.2f}) -> {call}")


if __name__ == '__main__':
    main()

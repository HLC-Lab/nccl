#!/usr/bin/env python3
"""Timed (latency-aware) simulator for Bine relay op orders.

Companion to verify_schedule.py: that one gates LIVENESS (deadlock-freedom);
this one models THROUGHPUT, because a live order can still be slow -- plain
source order (lambda=0) is deadlock-free at FIFO depth 2 yet measured ~0.1x
PAT at 64 nodes on Leonardo (every fused op stalls one network hop,
serialized). Any change to the emission order must pass BOTH gates.

Model: per-rank strictly sequential op execution with
  - per-op execution cost  o_exec (copy + kernel overhead)
  - network hop latency    l_hop  (alpha + bytes/BW)
  - credit-return latency  l_cred (head-pointer update back to the sender)
  - NCCL_STEPS=8 credits per directed connection

Usage: python3 timed_sim.py    (prints the candidate-order comparison table)
"""
import heapq
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_schedule import (RelayDepthOrder, RelaySrcOrder, RelaySkewOrder,
                             binePi, fifo_sim, NCCL_STEPS)


def timed_sim(n, cls, kw=None, o_exec=5.0, l_hop=66.0, l_cred=25.0,
              depth=NCCL_STEPS, max_us=60_000_000.0):
    """Event-driven; returns makespan (us) of one chunk, or None on deadlock."""
    kw = kw or {}
    algos = {r: cls(0, 1, 1, 1, r, n, **kw) for r in range(n)}
    ipos = {r: 0 for r in range(n)}
    nst = algos[0].nsteps
    arrivals = {(r, k): [] for r in range(n) for k in range(nst)}
    consumed = {(r, k): 0 for r in range(n) for k in range(nst)}
    credits = {(r, k): depth for r in range(n) for k in range(nst)}
    free_at = {r: 0.0 for r in range(n)}
    total = sum(len(algos[r].ops) for r in range(n))
    done = 0
    finish = 0.0
    seq = 0
    pq = []   # (time, seq, kind, payload): kind 0 = wake rank, 1 = credit conn
    for r in range(n):
        pq.append((0.0, seq, 0, r)); seq += 1
    heapq.heapify(pq)

    def try_advance(r, t):
        nonlocal done, finish, seq
        while ipos[r] < len(algos[r].ops):
            K, rd, sd, s = algos[r].ops[ipos[r]]
            t_start = max(t, free_at[r])
            if rd >= 0:
                conn = (binePi(r, rd, n), rd)
                i = consumed[conn]
                if i >= len(arrivals[conn]):
                    return  # not sent yet; the sender's send event wakes us
                if arrivals[conn][i] > t_start:
                    heapq.heappush(pq, (arrivals[conn][i], seq, 0, r)); seq += 1
                    return
            if sd >= 0 and credits[(r, sd)] <= 0:
                return  # the credit-return event wakes us
            t_end = t_start + o_exec
            if rd >= 0:
                conn = (binePi(r, rd, n), rd)
                consumed[conn] += 1
                heapq.heappush(pq, (t_end + l_cred, seq, 1, conn)); seq += 1
            if sd >= 0:
                credits[(r, sd)] -= 1
                arrivals[(r, sd)].append(t_end + l_hop)
                heapq.heappush(pq, (t_end + l_hop, seq, 0, binePi(r, sd, n))); seq += 1
            free_at[r] = t_end
            finish = max(finish, t_end)
            ipos[r] += 1
            done += 1
            t = t_end

    while pq and done < total:
        t, _, kind, payload = heapq.heappop(pq)
        if t > max_us:
            return None
        if kind == 1:
            credits[payload] += 1
            try_advance(payload[0], t)
        else:
            try_advance(payload, t)
    return finish if done == total else None


def run():
    print("Effective per-rank GB/s, one 512KB-slot chunk "
          "(o=5us, hop=66us, cred=25us, fifo=8):")
    hdr = f"{'order':20s}" + "".join(f" {('n=%d' % n):>10s}" for n in (16, 32, 64, 128, 256))
    print(hdr + "   liveness(d8)")
    cands = [
        ("depth (pre-fix)", RelayDepthOrder, None),
        ("source (lam=0)", RelaySrcOrder, None),
    ] + [(f"skew lam={lam}", RelaySkewOrder, {"lam": lam}) for lam in (2, 4, 6, 8, 12)]
    for name, cls, kw in cands:
        row = []
        for n in (16, 32, 64, 128, 256):
            ms = timed_sim(n, cls, kw)
            row.append(f"{(n - 1) * 512e3 / (ms * 1e-6) / 1e9:9.2f}" if ms else "  DEADLK")
        live = all(fifo_sim(n, cls, nchunks=1, depth=8, **(kw or {})) == 'OK'
                   for n in (16, 32, 64, 128, 256))
        print(f"{name:20s}" + "".join(f" {x:>10s}" for x in row)
              + f"   {'LIVE' if live else 'DEADLOCKS'}")


if __name__ == "__main__":
    run()

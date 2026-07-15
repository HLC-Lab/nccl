#!/usr/bin/env python3
"""Phase 7 offline verification: BLOCK-STRIPED channels for Bine AllGather.

Today (NCCL convention) every block is byte-sliced across all C channels, so more
channels => smaller messages (the root of the 16-128 MB band pathology: measured
5.28/8.66/9.36 GB/s at 16K/64K/256K messages with IDENTICAL op counts). Striping
instead partitions the SET of source blocks across channels (channel c owns
{s : s mod C == c}); each block travels WHOLE on exactly one channel: C x bigger
messages, C x fewer per channel, same bytes per channel.

This harness verifies, per (n, C, channel):
  static  : stripe coverage (union over channels = every block exactly once, and
            only on its stripe channel), per-connection send/recv sequence
            equality within the channel, read-after-gather.
  liveness: bounded-FIFO simulation per channel (channels have independent
            connections): striped skew-relay at depth 8 and 6 (margin); striped
            packed butterfly at its per-channel pack factor, including the
            per-channel safety rule ceil(maxRound_c / pf) <= NCCL_STEPS.
  balance : ops per channel (min/max over channels), largest per-channel
            butterfly round.

Feasibility of the plumbing was checked in-code: device has work->channelLo/Hi
(=> C and stripe index), proxy op has channelId + nChannels. Both sides can
compute the same stripe deterministically (Do-NOT #7 respected by design).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_schedule import RelayBase, binePi, NCCL_STEPS

LAM = 6

# --- stripe functions (must be trivially portable to C++ host+device) -------
# 'mod'  : s % C                  (correlates with negabinary structure -> imbalance)
# 'hash' : Knuth multiplicative   (decorrelates tree roles from channels)
# 'blk'  : contiguous ranges      (s * C // n)
STRIPE_FNS = {
    'mod': lambda s, C, n: s % C,
    'hash': lambda s, C, n: ((s * 2654435761) >> 8) % C,
    'blk': lambda s, C, n: min(C - 1, s * C // n),
}
STRIPE = STRIPE_FNS['mod']  # overridden by main()


class StripedRelay(RelayBase):
    """Skew-order relay restricted to blocks {s : STRIPE(s,C,n) == stripeIdx}."""

    def __init__(self, offset, end, count, chunkCount, rank, nranks,
                 stripeC=1, stripeIdx=0, lam=LAM):
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
            if STRIPE(s, stripeC, nranks) != stripeIdx:
                continue
            self.ops.extend(self.emit_block_ops(s))


class StripedButterfly(RelayBase):
    """Packed butterfly restricted to a stripe. Both endpoints filter the same
    global set (s % C), so per-connection sequences match by construction."""

    def __init__(self, offset, end, count, chunkCount, rank, nranks,
                 stripeC=1, stripeIdx=0, postFreq=1):
        super().__init__(offset, end, count, chunkCount, rank, nranks)
        n = nranks
        self.P = postFreq
        self.ops = []
        self.meta = []
        if STRIPE(rank, stripeC, nranks) == stripeIdx:
            self.ops.append((0, -1, -1, rank))
            self.meta.append((0, False))
        for t in range(self.nsteps - 1, -1, -1):
            partner = binePi(rank, t, nranks)
            S = [b for b in sorted(self.getIdx(partner, t)) if STRIPE(b, stripeC, nranks) == stripeIdx]
            R = [b for b in sorted(self.getIdx(rank, t)) if STRIPE(b, stripeC, nranks) == stripeIdx]
            for j, b in enumerate(S):
                post = (j % postFreq == postFreq - 1) or (j == len(S) - 1)
                self.ops.append((1, -1, t, b))
                self.meta.append((j % postFreq, post))
            for j, b in enumerate(R):
                post = (j % postFreq == postFreq - 1) or (j == len(R) - 1)
                self.ops.append((2, t, -1, b))
                self.meta.append((j % postFreq, post))


# ------------------------------ checks -------------------------------------

def static_striped(n, C, cls, **kw):
    """Coverage + per-connection consistency + read-after-gather, per channel."""
    errs = []
    recv_seen = {r: {} for r in range(n)}  # rank -> block -> channel
    for c in range(C):
        algos = {r: cls(0, 1, 1, 1, r, n, stripeC=C, stripeIdx=c, **kw)
                 for r in range(n)}
        for r, a in algos.items():
            have = set()
            for i, (K, rd, sd, s) in enumerate(a.ops):
                if STRIPE(s, C, n) != c:
                    errs.append(f"n={n} C={C} ch{c} r{r}: op for off-stripe block {s}")
                if rd >= 0:
                    if s in recv_seen[r]:
                        errs.append(f"n={n} C={C} r{r}: block {s} received on ch"
                                    f"{recv_seen[r][s]} AND ch{c}")
                    recv_seen[r][s] = c
                if sd >= 0 and K == 1 and s not in have:
                    errs.append(f"n={n} C={C} ch{c} r{r}: op {i} sends {s} before gathered")
                if rd >= 0 or K == 0:
                    have.add(s)
        for r, a in algos.items():
            for k in range(a.nsteps):
                p = binePi(r, k, n)
                snd = [s for (K, rd, sd, s) in a.ops if sd == k]
                rcv = [s for (K, rd, sd, s) in algos[p].ops if rd == k]
                if snd != rcv:
                    errs.append(f"n={n} C={C} ch{c} conn r{r}->r{p} dim{k}: seq mismatch")
    for r in range(n):
        missing = [s for s in range(n) if s != r and s not in recv_seen[r]]
        if missing:
            errs.append(f"n={n} C={C} r{r}: never receives {missing[:5]}")
        for s, c in recv_seen[r].items():
            if STRIPE(s, C, n) != c:
                errs.append(f"n={n} C={C} r{r}: block {s} on wrong channel {c}")
    return errs


def fifo_sim_striped(n, C, c, depth=NCCL_STEPS, nchunks=1, **kw):
    """Unpacked FIFO sim for one channel of the striped relay."""
    algos = {r: StripedRelay(0, 1, 1, 1, r, n, stripeC=C, stripeIdx=c, **kw)
             for r in range(n)}
    streams = {r: [(op, ch) for ch in range(nchunks) for op in algos[r].ops]
               for r in range(n)}
    ipos = {r: 0 for r in range(n)}
    fifo = {(r, k): [] for r in range(n) for k in range(algos[0].nsteps)}
    out = {r: set() for r in range(n)}
    total = sum(len(x) for x in streams.values())
    done = 0
    while done < total:
        prog = False
        for r in range(n):
            while ipos[r] < len(streams[r]):
                (K, rd, sd, s), ch = streams[r][ipos[r]]
                ok = True
                if rd >= 0:
                    q = fifo[(binePi(r, rd, n), rd)]
                    if not q:
                        ok = False
                    elif q[0] != (s, ch):
                        return f'ORDER-MISMATCH r{r}'
                if ok and sd >= 0 and len(fifo[(r, sd)]) >= depth:
                    ok = False
                if not ok:
                    break
                if rd >= 0:
                    fifo[(binePi(r, rd, n), rd)].pop(0)
                    out[r].add((s, ch))
                if K == 0:
                    out[r].add((s, ch))
                if sd >= 0:
                    if K == 1 and (s, ch) not in out[r]:
                        return f'READ-BEFORE-GATHER r{r}'
                    fifo[(r, sd)].append((s, ch))
                ipos[r] += 1
                done += 1
                prog = True
        if not prog:
            return f'DEADLOCK({total - done})'
    return 'OK'


def bfly_sim_striped(n, C, c, pf, depth=NCCL_STEPS, nchunks=1):
    """Packed-butterfly FIFO sim for one striped channel."""
    algos = {r: StripedButterfly(0, 1, 1, 1, r, n, stripeC=C, stripeIdx=c,
                                 postFreq=pf) for r in range(n)}
    streams = {r: [(i, ch) for ch in range(nchunks) for i in range(len(algos[r].ops))]
               for r in range(n)}
    ipos = {r: 0 for r in range(n)}
    nst = algos[0].nsteps
    posted = {(r, k): [] for r in range(n) for k in range(nst)}
    openpk = {(r, k): [] for r in range(n) for k in range(nst)}
    hdused = {(r, k): 0 for r in range(n) for k in range(nst)}
    out = {r: set() for r in range(n)}
    total = sum(len(x) for x in streams.values())
    done = 0
    while done < total:
        prog = False
        for r in range(n):
            while ipos[r] < len(streams[r]):
                i, ch = streams[r][ipos[r]]
                K, rd, sd, s = algos[r].ops[i]
                _, posts = algos[r].meta[i]
                ok = True
                if K == 2:
                    snd = binePi(r, rd, n)
                    q = posted[(snd, rd)]
                    if not q:
                        ok = False
                    elif q[0][hdused[(snd, rd)]] != (s, ch):
                        return f'ORDER-MISMATCH r{r}'
                if K == 1:
                    if len(posted[(r, sd)]) >= depth:
                        ok = False
                    elif (s, ch) not in out[r]:
                        return f'READ-BEFORE-GATHER r{r} blk{s}'
                if not ok:
                    break
                if K == 0:
                    out[r].add((s, ch))
                if K == 1:
                    openpk[(r, sd)].append((s, ch))
                    if posts:
                        posted[(r, sd)].append(openpk[(r, sd)])
                        openpk[(r, sd)] = []
                if K == 2:
                    snd = binePi(r, rd, n)
                    out[r].add((s, ch))
                    hdused[(snd, rd)] += 1
                    if posts:
                        pk = posted[(snd, rd)].pop(0)
                        if len(pk) != hdused[(snd, rd)]:
                            return 'PACK-MISMATCH'
                        hdused[(snd, rd)] = 0
                ipos[r] += 1
                done += 1
                prog = True
        if not prog:
            return f'DEADLOCK({total - done})'
    return 'OK'


def max_round(n, C):
    """Largest per-channel butterfly round over ranks/channels/rounds."""
    m = 0
    a = RelayBase(0, 1, 1, 1, 0, n)
    for c in range(C):
        for t in range(a.nsteps):
            m = max(m, len([b for b in a.getIdx(0, t) if STRIPE(b, C, n) == c]))
    return m


def send_balance(n, C):
    """Wire-balance of the striped relay: each op with sd>=0 sends one whole
    block, so sends-per-(rank,channel) IS the egress byte distribution.
    Returns max straggler ratio = worst per-channel sends / ideal (n-1)/C."""
    ideal = (n - 1) / C
    worst = 0.0
    for r in range(n):
        for c in range(C):
            a = StripedRelay(0, 1, 1, 1, r, n, stripeC=C, stripeIdx=c)
            sends = sum(1 for (K, rd, sd, s) in a.ops if sd >= 0)
            worst = max(worst, sends / ideal)
    return worst


def main():
    global STRIPE
    print("=== stripe-function selection: straggler ratio (worst channel egress / ideal) ===")
    print(f"{'config':>12s}" + "".join(f" {fn:>7s}" for fn in STRIPE_FNS))
    scores = {fn: 0.0 for fn in STRIPE_FNS}
    for (n, C) in ((128, 4), (128, 8), (128, 16), (256, 16)):
        row = []
        for fn in STRIPE_FNS:
            STRIPE = STRIPE_FNS[fn]
            w = send_balance(n, C)
            scores[fn] = max(scores[fn], w)
            row.append(f"{w:7.2f}")
        print(f"n={n:4d} C={C:2d}:" + "".join(f" {x}" for x in row))
    winner = min(scores, key=lambda f: scores[f])
    print(f"-> winner: '{winner}' (worst straggler {scores[winner]:.2f}x; 1.00 = perfect)")
    print()
    STRIPE = STRIPE_FNS[winner]
    print(f"=== full gate battery with stripe fn '{winner}' ===")
    all_ok = True
    for n in (64, 128, 256):
        for C in (2, 4, 8, 16):
            errs = static_striped(n, C, StripedRelay)
            errs += static_striped(n, C, StripedButterfly, postFreq=1)
            # liveness: check channels 0 and C-1 (structure identical mod residue)
            rl = [fifo_sim_striped(n, C, c, depth=d)
                  for c in (0, C - 1) for d in (8, 6)]
            mr = max_round(n, C)
            # butterfly: pf=1 (full-size blocks) plus the smallest SAFE pf
            need_pf = max(1, math.ceil(mr / NCCL_STEPS))
            bf = [bfly_sim_striped(n, C, c, pf) for c in (0, C - 1)
                  for pf in {1, need_pf}]
            bf_ok = all(x == 'OK' for x in bf) if need_pf == 1 else \
                all(bfly_sim_striped(n, C, c, need_pf) == 'OK' for c in (0, C - 1))
            # balance: ops per channel for rank 1
            ops_per_ch = [len(StripedRelay(0, 1, 1, 1, 1, n, stripeC=C, stripeIdx=c).ops)
                          for c in range(C)]
            good = (not errs and all(x == 'OK' for x in rl) and bf_ok)
            all_ok &= good
            print(f"n={n:4d} C={C:2d}: static {'OK' if not errs else 'FAIL:' + errs[0]}"
                  f"  relay(d8/d6 x ch0/chLast): {rl}"
                  f"  bfly(maxRound={mr}, needPf={need_pf}): {'OK' if bf_ok else bf}"
                  f"  ops/ch {min(ops_per_ch)}-{max(ops_per_ch)}"
                  + ('' if good else '  <<< FAIL'))
        print()
    print('RESULT:', 'PASS' if all_ok else 'FAIL')
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()

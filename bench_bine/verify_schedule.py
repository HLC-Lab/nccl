#!/usr/bin/env python3
"""Offline verifier for the Bine AllGather schedule in src/include/collectives.h.

This file is the ground-truth gate for ANY change to the schedule (op-list
construction / emission order / packing). It mirrors the C++ PatAGAlgorithm
logic in Python and checks, for every power-of-two nranks <= 256 and EVERY
rank:

  static:  - partition/coverage: each block received exactly once (krecv
             defined once), involution pi(pi(r,k),k)==r
           - per-connection consistency: the ordered block sequence rank r
             sends on dim k EQUALS the sequence binePi(r,k) receives on dim k
           - SEND-only ops (kind 1, read userOutput) come after the op that
             gathered that block
           - nOps <= RMAXOPS
  dynamic: - event-driven FIFO simulation: every rank executes its op list
             strictly sequentially (parallelFactor=1); each directed
             connection (r -> binePi(r,k), dim k) is a bounded FIFO.
             Completion without deadlock must hold at FIFO depth 8 (NCCL_STEPS)
             AND depth 2 (margin), for 1 and 3 chunks.
           - chunk arithmetic of getNextOp (tails, empty channel, last=2)

IMPORTANT: the classes below must be kept in lockstep with
src/include/collectives.h. If you change the C++ emission, change the mirror
here the same way and re-run. The FIFO model is *confluent* (each rank has one
next op, each FIFO one consumer), so a deadlock found here is
schedule-independent - it WILL happen on hardware; timing cannot avoid it.

Usage:
  python3 verify_schedule.py baseline   # mirror of the ORIGINAL depth-order
                                        # emission: static passes, FIFO sim is
                                        # EXPECTED to deadlock at n>=64
  python3 verify_schedule.py phase2     # skewed source-order relay (the code
                                        # in collectives.h): ALL gates must pass
  python3 verify_schedule.py phase4     # butterfly mode: ALL gates must pass
  python3 verify_schedule.py all        # run everything
Exit code 0 = all required gates passed.
"""
import math
import sys

RMAXOPS = 520
NCCL_STEPS = 8


def log2Up(n):
    l = 0
    while (1 << l) < n:
        l += 1
    return l


def bineRho(step):
    r = 1
    p = 1
    for _ in range(1, step + 1):
        p *= -2
        r += p
    return r


def binePi(rank, step, nranks):
    rho = bineRho(step)
    # C '%' truncates toward zero then the code adds nranks if negative; for
    # |arg| < 2*nranks this equals mathematical mod == Python's '%'.
    return (rank + rho) % nranks if (rank & 1) == 0 else (rank - rho) % nranks


class RelayBase:
    """Shared machinery: krecv/cmask precompute + getNextOp chunk arithmetic.

    Subclasses fill self.ops = [(kind, rdim, sdim, srcBlock), ...] with
    kind: 0=INIT (input->output), 1=SEND-only (reads output), 2=RECV-only,
    3=FUSED (recv + forward)."""

    def __init__(self, offset, end, count, chunkCount, rank, nranks):
        self.offset = offset
        self.end = end
        self.count = count
        self.chunkCount = chunkCount
        self.rank = rank
        self.nranks = nranks
        self.nsteps = log2Up(nranks)
        n = nranks
        self.krecv = [-1] * n
        self.cmask = [0] * n
        self.krecv_multi = 0
        for k in range(self.nsteps):
            for i in self.getIdx(rank, k):
                if self.krecv[i] != -1:
                    self.krecv_multi += 1
                self.krecv[i] = k
            for i in self.getIdx(binePi(rank, k, n), k):
                self.cmask[i] |= (1 << k)
        self.ops = []
        self.ip = 0

    def getNelem(self):
        return min(self.chunkCount, self.end - self.offset)

    def getIdx(self, start, st):
        """DFS pre-order of the subtree entered at step st (mirror of C getIdx)."""
        if st >= self.nsteps:
            return []
        out = []
        e0 = binePi(start, st, self.nranks)
        out.append(e0)
        node = [e0]
        sval = [st + 1]
        while node:
            y = node[-1]
            s = sval[-1]
            if s >= self.nsteps:
                node.pop()
                sval.pop()
                continue
            sval[-1] = s + 1
            c = binePi(y, s, self.nranks)
            out.append(c)
            node.append(c)
            sval.append(s + 1)
        return out

    def emit_block_ops(self, s):
        """Ops for source block s: INIT+own-forwards, or fused+extras, or leaf."""
        ops = []
        if s == self.rank:
            ops.append((0, -1, -1, self.rank))
            for k in range(self.nsteps):
                if self.cmask[s] & (1 << k):
                    ops.append((1, -1, k, s))
        else:
            first = -1
            for k in range(self.nsteps):
                if self.cmask[s] & (1 << k):
                    first = k
                    break
            if first >= 0:
                ops.append((3, self.krecv[s], first, s))
                for k in range(first + 1, self.nsteps):
                    if self.cmask[s] & (1 << k):
                        ops.append((1, -1, k, s))
            else:
                ops.append((2, self.krecv[s], -1, s))
        return ops

    def getNextOp(self):
        ps = dict(last=0, recvDim=-1, sendDim=-1, postRecv=0, postSend=0,
                  inpIx=0, outIx=0)
        nelem = self.getNelem()
        ps['nelem'] = nelem
        K, rd, sd, s = self.ops[self.ip]
        if K == 0:
            ps['inpIx'] = self.offset
            ps['outIx'] = self.rank * self.count + self.offset
        else:
            if rd >= 0:
                ps['recvDim'] = rd
                ps['outIx'] = s * self.count + self.offset
                ps['postRecv'] = 1
            if sd >= 0:
                ps['sendDim'] = sd
                ps['inpIx'] = s * self.count + self.offset
                ps['postSend'] = 1
        self.ip += 1
        if self.ip >= len(self.ops):
            if self.offset + self.chunkCount >= self.end:
                ps['last'] = 2
            else:
                self.offset += self.chunkCount
                self.ip = 0
        return ps


class RelayDepthOrder(RelayBase):
    """Mirror of the ORIGINAL (pre-fix) emission: ops grouped by tree depth.
    Kept as the baseline; its FIFO sim deadlocks at n>=64 - that is the bug."""

    def __init__(self, *a):
        super().__init__(*a)
        n = self.nranks

        def depthOf(s):
            if s == self.rank:
                return 0
            x, d, guard = self.rank, 0, 0
            while x != s:
                guard += 1
                if guard > 2 * n:
                    break
                f = -1
                for k in range(self.nsteps):
                    if s in self.getIdx(x, k):
                        f = k
                        break
                if f < 0:
                    return -1
                x = binePi(x, f, n)
                d += 1
            return d

        dof = [depthOf(s) for s in range(n)]
        for d in range(0, max(dof) + 1):
            for s in range(n):
                if dof[s] == d:
                    self.ops.extend(self.emit_block_ops(s))


class RelaySrcOrder(RelayBase):
    """Plain source order s = 0..n-1 (lambda=0). Deadlock-free at FIFO depth 2
    BUT measured ~0.1x PAT at 64 nodes on Leonardo: every rank consumes each
    block at the same wavefront instant its parent produces it, so every fused
    op stalls one full network hop, serialized. Kept as a reference; DO NOT
    ship. The shipping order is RelaySkewOrder below."""

    def __init__(self, *a):
        super().__init__(*a)
        for s in range(self.nranks):
            self.ops.extend(self.emit_block_ops(s))


BINE_SKEW_LAMBDA = 6  # must match src/include/collectives.h


class RelaySkewOrder(RelayBase):
    """Phase-2 target (mirror of the C++): emit blocks in ascending
    key(s) = s + lam*depth(s), ties by s, where depth(s) is this rank's depth
    in block s's broadcast tree (recorded during the getIdx DFS -- equals the
    old depthOf()). lam interpolates source order (lam=0, live but slow) and
    depth order (lam=inf, fast but deadlocks at n>=64). Per-connection order
    consistency holds for any lam because the receiver's depth is the
    sender's +1, shifting all keys by the same +lam. lam=6: deadlock-free
    down to FIFO depth 6 for all po2 n <= 256 (2 slots of margin), and near
    the throughput optimum of the timed model (timed_sim.py)."""

    def __init__(self, offset, end, count, chunkCount, rank, nranks,
                 lam=BINE_SKEW_LAMBDA):
        super().__init__(offset, end, count, chunkCount, rank, nranks)
        n = nranks
        self.dof = dof = [0] * n
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
        # equivalent to the C++ O(n^2) min-key selection scan (strict '<',
        # first smallest s wins ties)
        order = sorted(range(n), key=lambda s: (s + lam * dof[s], s))
        for s in order:
            self.ops.extend(self.emit_block_ops(s))


class Butterfly(RelayBase):
    """Phase-4 target (mirror of pico allgather_bine_block_by_block):
    rounds t = nsteps-1 .. 0 with partner binePi(rank,t) (conn dim t):
    send blocks getIdx(partner,t) ascending, then recv blocks getIdx(rank,t)
    ascending; postFreq blocks share one FIFO slot; post on last-of-pack or
    last-of-round. SAFETY REQUIRES postFreq >= ceil((n/2)/NCCL_STEPS)."""

    def __init__(self, offset, end, count, chunkCount, rank, nranks, postFreq):
        super().__init__(offset, end, count, chunkCount, rank, nranks)
        self.P = postFreq
        self.ops = [(0, -1, -1, rank)]
        self.meta = [(0, False)]  # (slotPos, posts)
        for t in range(self.nsteps - 1, -1, -1):
            partner = binePi(rank, t, nranks)
            S = sorted(self.getIdx(partner, t))
            R = sorted(self.getIdx(rank, t))
            for j, b in enumerate(S):
                post = (j % postFreq == postFreq - 1) or (j == len(S) - 1)
                self.ops.append((1, -1, t, b))
                self.meta.append((j % postFreq, post))
            for j, b in enumerate(R):
                post = (j % postFreq == postFreq - 1) or (j == len(R) - 1)
                self.ops.append((2, t, -1, b))
                self.meta.append((j % postFreq, post))


# --------------------------- checks ---------------------------------------

def static_checks(n, cls, **kw):
    errs = []
    algos = {r: cls(0, 1, 1, 1, r, n, **kw) for r in range(n)}
    for r, a in algos.items():
        if a.krecv_multi:
            errs.append(f"n={n} r={r}: krecv overwritten (partition violated)")
        if len(a.ops) > RMAXOPS:
            errs.append(f"n={n} r={r}: nOps {len(a.ops)} > RMAXOPS {RMAXOPS}")
        for s in range(n):
            if s != r and a.krecv[s] == -1:
                errs.append(f"n={n} r={r}: never receives block {s}")
        nrecv = sum(1 for (K, rd, sd, s) in a.ops if rd >= 0)
        if nrecv != n - 1:
            errs.append(f"n={n} r={r}: recv-op count {nrecv} != {n - 1}")
        seen = set()
        for i, (K, rd, sd, s) in enumerate(a.ops):
            if rd >= 0:
                if s in seen:
                    errs.append(f"n={n} r={r}: block {s} received twice")
                seen.add(s)
        have = set()
        for i, (K, rd, sd, s) in enumerate(a.ops):
            if sd >= 0 and K == 1 and s not in have:
                errs.append(f"n={n} r={r}: op {i} sends block {s} before gathered")
            if rd >= 0 or K == 0:
                have.add(s)
        for k in range(a.nsteps):
            if binePi(binePi(r, k, n), k, n) != r:
                errs.append(f"n={n} r={r} k={k}: involution broken")
    for r, a in algos.items():
        for k in range(a.nsteps):
            p = binePi(r, k, n)
            snd = [s for (K, rd, sd, s) in a.ops if sd == k]
            rcv = [s for (K, rd, sd, s) in algos[p].ops if rd == k]
            if snd != rcv:
                errs.append(f"n={n} conn r{r}->r{p} dim{k}: send/recv sequences differ")
    return errs


def fifo_sim(n, cls, nchunks=1, depth=NCCL_STEPS, **kw):
    """Unpacked model (one FIFO slot per op, matches postRecv=postSend=1)."""
    algos = {r: cls(0, nchunks, nchunks, 1, r, n, **kw) for r in range(n)}
    streams = {r: [(op, c) for c in range(nchunks) for op in algos[r].ops] for r in range(n)}
    ipos = {r: 0 for r in range(n)}
    fifo = {(r, k): [] for r in range(n) for k in range(algos[r].nsteps)}
    out = {r: set() for r in range(n)}
    total = sum(len(x) for x in streams.values())
    done = 0
    while done < total:
        prog = False
        for r in range(n):
            while ipos[r] < len(streams[r]):
                (K, rd, sd, s), c = streams[r][ipos[r]]
                ok = True
                if rd >= 0:
                    q = fifo[(binePi(r, rd, n), rd)]
                    if not q:
                        ok = False
                    elif q[0] != (s, c):
                        return f'ORDER-MISMATCH r{r}'
                if ok and sd >= 0 and len(fifo[(r, sd)]) >= depth:
                    ok = False
                if not ok:
                    break
                if rd >= 0:
                    fifo[(binePi(r, rd, n), rd)].pop(0)
                    out[r].add((s, c))
                if K == 0:
                    out[r].add((r, c))
                if sd >= 0:
                    if K == 1 and (s, c) not in out[r]:
                        return f'READ-BEFORE-GATHER r{r} blk{s}'
                    fifo[(r, sd)].append((s, c))
                ipos[r] += 1
                done += 1
                prog = True
        if not prog:
            return f'DEADLOCK({total - done} ops pending)'
    for r in range(n):
        if out[r] != {(s, c) for s in range(n) for c in range(nchunks)}:
            return f'INCOMPLETE r{r}'
    if any(fifo.values()):
        return 'FIFO-RESIDUE'
    return 'OK'


def butterfly_sim(n, P, depth=NCCL_STEPS, nchunks=1):
    """Packed model: P blocks share a slot; slot visible to the receiver only
    when posted; slot freed when its last block is consumed."""
    algos = {r: Butterfly(0, 1, 1, 1, r, n, P) for r in range(n)}
    streams = {r: [(i, c) for c in range(nchunks) for i in range(len(algos[r].ops))] for r in range(n)}
    ipos = {r: 0 for r in range(n)}
    posted = {(r, k): [] for r in range(n) for k in range(algos[r].nsteps)}
    openpk = {(r, k): [] for r in range(n) for k in range(algos[r].nsteps)}
    head_used = {(r, k): 0 for r in range(n) for k in range(algos[r].nsteps)}
    out = {r: set() for r in range(n)}
    total = sum(len(x) for x in streams.values())
    done = 0
    while done < total:
        prog = False
        for r in range(n):
            while ipos[r] < len(streams[r]):
                i, c = streams[r][ipos[r]]
                K, rd, sd, s = algos[r].ops[i]
                _, posts = algos[r].meta[i]
                ok = True
                if K == 2:
                    snd = binePi(r, rd, n)
                    q = posted[(snd, rd)]
                    if not q:
                        ok = False
                    elif q[0][head_used[(snd, rd)]] != (s, c):
                        return f'ORDER-MISMATCH r{r} op{i}'
                if K == 1:
                    if len(posted[(r, sd)]) >= depth:
                        ok = False
                    elif (s, c) not in out[r]:
                        return f'READ-BEFORE-GATHER r{r} blk{s}'
                if not ok:
                    break
                if K == 0:
                    out[r].add((r, c))
                if K == 1:
                    openpk[(r, sd)].append((s, c))
                    if posts:
                        posted[(r, sd)].append(openpk[(r, sd)])
                        openpk[(r, sd)] = []
                if K == 2:
                    snd = binePi(r, rd, n)
                    out[r].add((s, c))
                    head_used[(snd, rd)] += 1
                    if posts:
                        pk = posted[(snd, rd)].pop(0)
                        if len(pk) != head_used[(snd, rd)]:
                            return 'PACK-SIZE-MISMATCH'
                        head_used[(snd, rd)] = 0
                ipos[r] += 1
                done += 1
                prog = True
        if not prog:
            return f'DEADLOCK({total - done} ops pending)'
    for r in range(n):
        if out[r] != {(s, c) for s in range(n) for c in range(nchunks)}:
            return f'INCOMPLETE r{r}'
    return 'OK'


def chunk_checks(n, cls, **kw):
    errs = []
    r = min(1, n - 1)
    for (off, end, cc) in [(0, 10, 4), (5, 12, 4), (0, 4, 4), (0, 3, 8), (7, 7, 4)]:
        a = cls(off, end, 100, cc, r, n, **kw)
        nops = len(a.ops)
        exp_chunks = 1 if end <= off else -(-(end - off) // cc)
        emitted = 0
        while True:
            ps = a.getNextOp()
            emitted += 1
            chunk = (emitted - 1) // nops
            expect = min(cc, end - (off + chunk * cc))
            if ps['nelem'] != expect:
                errs.append(f"n={n} off={off} end={end} cc={cc}: op {emitted - 1} nelem "
                            f"{ps['nelem']} != {expect}")
                break
            if ps['last'] == 2:
                if emitted != nops * exp_chunks:
                    errs.append(f"n={n} off={off} end={end} cc={cc}: last=2 at op "
                                f"{emitted}, expected {nops * exp_chunks}")
                break
            if emitted > nops * (exp_chunks + 2) + 10:
                errs.append(f"n={n} off={off} end={end} cc={cc}: last=2 never fired")
                break
    return errs


SIZES = [2, 4, 8, 16, 32, 64, 128, 256]


def run_baseline():
    print("== baseline (depth-order mirror of the pre-fix code) ==")
    ok = True
    for n in SIZES:
        e = static_checks(n, RelayDepthOrder)
        r8 = fifo_sim(n, RelayDepthOrder, depth=8)
        print(f"n={n:4d}: static {'OK' if not e else 'FAIL'}  fifo(8): {r8}")
        ok &= not e
    print("NOTE: fifo DEADLOCK at n>=64 is the known bug this plan fixes;")
    print("      static checks must still pass.")
    return ok


def run_phase2():
    print(f"== phase2 (skewed source-order relay, lambda={BINE_SKEW_LAMBDA}) - ALL gates must pass ==")
    ok = True
    for n in SIZES:
        e = static_checks(n, RelaySkewOrder)
        # multiset equality with depth-order (same work, different order)
        for r in range(n):
            if sorted(RelaySkewOrder(0, 1, 1, 1, r, n).ops) != \
               sorted(RelayDepthOrder(0, 1, 1, 1, r, n).ops):
                e.append(f"n={n} r={r}: op multiset differs from depth-order")
                break
        # depth 8 = real FIFO; depth 6 = required liveness margin
        res = [fifo_sim(n, RelaySkewOrder, nchunks=c, depth=d)
               for c in (1, 3) for d in (8, 6)]
        cerrs = chunk_checks(n, RelaySkewOrder)
        good = not e and all(x == 'OK' for x in res) and not cerrs
        ok &= good
        print(f"n={n:4d}: static {'OK' if not e else 'FAIL'}  "
              f"fifo(d8/d6 x c1/c3): {res}  chunks {'OK' if not cerrs else 'FAIL'}"
              + ('' if good else '   <<< FAIL'))
    return ok


def run_phase4():
    print("== phase4 (butterfly, packed) - ALL gates must pass ==")
    ok = True
    for n in SIZES:
        minP = max(1, math.ceil((n // 2) / NCCL_STEPS)) if n > 2 else 1
        e = static_checks(n, Butterfly, postFreq=minP)
        res = [butterfly_sim(n, P) for P in {minP, minP * 2, max(n // 2, 1)}]
        r3 = butterfly_sim(n, minP, nchunks=3)
        below = butterfly_sim(n, minP // 2) if minP > 1 else 'n/a'
        good = not e and all(x == 'OK' for x in res) and r3 == 'OK'
        ok &= good
        print(f"n={n:4d}: static {'OK' if not e else 'FAIL'}  minP={minP:3d}  "
              f"sim: {res} 3ch:{r3}  belowMinP(expect DEADLOCK): {below}"
              + ('' if good else '   <<< FAIL'))
    return ok


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    ok = True
    if what in ('baseline', 'all'):
        ok &= run_baseline()
    if what in ('phase2', 'all'):
        ok &= run_phase2()
    if what in ('phase4', 'all'):
        ok &= run_phase4()
    print('\nRESULT:', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)

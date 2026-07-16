"""Phase 7 caller-consistency gate: brute-force the NEW device caller (all_gather.h)
vs the NEW host caller (proxy.cc PatDown) arithmetic across a parameter grid and
assert the PatAGAlgorithm inputs are equivalent (host bytes == device elements * szT).

Device:
  gate    = stripe && count*szT > xover
  striped: off=0, end=count, chunk = min(count, (buff//8)//szT)   [elements]
  legacy : off=chOff, end=chOff+chCnt, chunk=chunkCount           [elements]
Host:
  sizePerRank = count*szT (exact, from info->count)
  gate    = stripe && sizePerRank > xover
  striped: off=0, end=blockBytes, chunk = min(blockBytes, ((buff//8)//szT)*szT) [bytes]
  legacy : off=0, end=size(=nbytes/n), chunk=op->chunkSize                      [bytes]

Also re-run the ctor's mode decision on both parameterizations to prove the SAME mode
(butterfly/relay) and postFreq come out (host T=char / device T=szT equivalence).
"""
import itertools, sys

NCCL_STEPS = 8


def ctor_mode(slotBytes, xover, off, end, count, chunk, szT, nranks):
    """Mirror of PatAGAlgorithm mode/postFreq selection (collectives.h)."""
    chunkBytes = chunk * szT
    sliceBytes = (end - off) * szT if (end - off) < chunk else chunkBytes
    postFreq = slotBytes // sliceBytes if sliceBytes > 0 else 1
    postFreq = max(postFreq, 1)
    postFreq = min(postFreq, nranks // 2)
    minPost = (nranks // 2 + NCCL_STEPS - 1) // NCCL_STEPS
    perChan = (end - off) * szT
    useB = (perChan <= xover) and (postFreq >= minPost)
    return useB, postFreq


fails = 0
checked = 0
for n, szT, buff, xover, stripe_env, C in itertools.product(
        [8, 32, 128, 256],
        [1, 2, 4, 8, 16],
        [4 << 20, 1 << 20, 1000000, 3670016],          # incl. slot NOT divisible by szT
        [0, 48 * 1024, 2_000_000_000],
        [0, 1],
        [1, 2, 4, 8, 16]):
    slot = buff // NCCL_STEPS
    slotE = slot // szT                                 # device
    slotAligned = (slot // szT) * szT                   # host
    for countBytesTarget in [4096, 48*1024, 49*1024, 256*1024, 1 << 20, 32 << 20, 256 << 20]:
        count = countBytesTarget // szT                 # per-rank elements
        if count == 0 or C > n // 2:
            continue
        blockBytes = count * szT
        gate_d = stripe_env and blockBytes > xover
        gate_h = stripe_env and blockBytes > xover      # sizePerRank == count*szT exactly
        checked += 1
        if gate_d != gate_h:
            print(f"GATE MISMATCH n={n} szT={szT} buff={buff} xover={xover} count={count}")
            fails += 1
            continue
        if not gate_d:
            continue                                    # legacy path: untouched code
        # device (elements)
        end_d, chunk_d = count, min(count, slotE)
        # host (bytes)
        end_h, chunk_h = blockBytes, min(blockBytes, slotAligned)
        for c in range(C):
            if (end_h, chunk_h) != (end_d * szT, chunk_d * szT):
                print(f"RANGE/CHUNK MISMATCH n={n} szT={szT} buff={buff} count={count}: "
                      f"host(end={end_h},chunk={chunk_h}) dev*szT(end={end_d*szT},chunk={chunk_d*szT})")
                fails += 1
                break
        # mode decision equivalence: device runs ctor with (szT, elements),
        # host with (szT=1, bytes)
        mode_d = ctor_mode(slot, xover, 0, end_d, count, chunk_d, szT, n)
        mode_h = ctor_mode(slot, xover, 0, end_h, blockBytes, chunk_h, 1, n)
        if mode_d != mode_h:
            print(f"MODE MISMATCH n={n} szT={szT} buff={buff} xover={xover} count={count}: "
                  f"dev={mode_d} host={mode_h}")
            fails += 1
        elif mode_d[0]:
            print(f"UNEXPECTED BUTTERFLY in striped mode n={n} szT={szT} xover={xover} count={count}")
            fails += 1

print(f"checked {checked} combos")
print("PASS" if fails == 0 else f"{fails} FAILURES")
sys.exit(1 if fails else 0)

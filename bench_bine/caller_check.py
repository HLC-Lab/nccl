"""Phase 7 caller-consistency gate: brute-force the device caller (all_gather.h)
vs the host caller (proxy.cc PatDown) arithmetic across a parameter grid and
assert the PatAGAlgorithm inputs are equivalent (host bytes == device elements * szT).

Gate (both sides, must be identical):
  engage = stripe && C <= n && (blockBytes > xover || C > 1)
Striped: off=0, end=count, chunk = min(count, slotElems)  [device elements]
         off=0, end=blockBytes, chunk = min(blockBytes, slotAligned)  [host bytes]
Legacy : untouched pre-Phase-7 path.

Also re-runs the ctor's mode decision on both parameterizations to prove the SAME
mode (butterfly/relay) and postFreq come out (host T=char / device T=szT
equivalence), and that the striped-mode expectations hold:
  blockBytes >  xover              -> relay
  blockBytes <= xover, pf >= minPost -> striped BUTTERFLY
  blockBytes <= xover, pf <  minPost -> striped relay (safety floor)
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
stats = {"striped-bfly": 0, "striped-relay": 0, "legacy": 0, "cgtn-fallback": 0}
for n, szT, buff, xover, stripe_env, C in itertools.product(
        [8, 16, 32, 128, 256],
        [1, 2, 4, 8, 16],
        [4 << 20, 1 << 20, 1000000, 3670016],          # incl. slot NOT divisible by szT
        [0, 48 * 1024, 2_000_000_000],
        [0, 1],
        [1, 2, 4, 8, 16]):
    slot = buff // NCCL_STEPS
    slotE = slot // szT                                 # device
    slotAligned = (slot // szT) * szT                   # host
    for countBytesTarget in [4096, 8192, 32768, 48 * 1024, 49 * 1024, 64 * 1024,
                             256 * 1024, 1 << 20, 32 << 20, 256 << 20]:
        count = countBytesTarget // szT                 # per-rank elements
        if count == 0:
            continue
        blockBytes = count * szT
        # device gate
        gate_d = bool(stripe_env) and C <= n and (blockBytes > xover or C > 1)
        # host gate (sizePerRank == count*szT exactly; stripeC == C from devWork)
        gate_h = bool(stripe_env) and C <= n and (blockBytes > xover or C > 1)
        checked += 1
        if gate_d != gate_h:
            print(f"GATE MISMATCH n={n} szT={szT} buff={buff} xover={xover} C={C} count={count}")
            fails += 1
            continue
        if not gate_d:
            stats["cgtn-fallback" if (stripe_env and C > n) else "legacy"] += 1
            continue                                    # legacy path: untouched code
        # device (elements)
        end_d, chunk_d = count, min(count, slotE)
        # host (bytes)
        end_h, chunk_h = blockBytes, min(blockBytes, slotAligned)
        if (end_h, chunk_h) != (end_d * szT, chunk_d * szT):
            print(f"RANGE/CHUNK MISMATCH n={n} szT={szT} buff={buff} count={count}: "
                  f"host(end={end_h},chunk={chunk_h}) dev*szT(end={end_d*szT},chunk={chunk_d*szT})")
            fails += 1
            continue
        # mode decision equivalence: device runs ctor with (szT, elements),
        # host with (szT=1, bytes)
        mode_d = ctor_mode(slot, xover, 0, end_d, count, chunk_d, szT, n)
        mode_h = ctor_mode(slot, xover, 0, end_h, blockBytes, chunk_h, 1, n)
        if mode_d != mode_h:
            print(f"MODE MISMATCH n={n} szT={szT} buff={buff} xover={xover} count={count}: "
                  f"dev={mode_d} host={mode_h}")
            fails += 1
            continue
        useB, pf = mode_d
        minPost = (n // 2 + NCCL_STEPS - 1) // NCCL_STEPS
        # striped-mode expectations
        if blockBytes > xover and useB:
            print(f"UNEXPECTED BUTTERFLY above xover n={n} szT={szT} xover={xover} count={count}")
            fails += 1
        elif blockBytes <= xover and blockBytes <= slotAligned and useB != (pf >= minPost):
            print(f"BUTTERFLY GATE WRONG n={n} szT={szT} xover={xover} count={count}: "
                  f"useB={useB} pf={pf} minPost={minPost}")
            fails += 1
        if useB and pf * min(blockBytes, slotAligned) > slot:
            print(f"PACK OVERFLOW n={n} szT={szT} buff={buff} count={count}: pf={pf}")
            fails += 1
        stats["striped-bfly" if useB else "striped-relay"] += 1

print(f"checked {checked} combos: {stats}")
print("PASS" if fails == 0 else f"{fails} FAILURES")
sys.exit(1 if fails else 0)

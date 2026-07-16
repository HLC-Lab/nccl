"""Phase 7 multi-chunk liveness gate: striped relay FIFO sim with nchunks=3
(the chunk-loop replay the real kernel does for blocks > slot), depths 8 and 6.
Previous stripe_study battery ran nchunks=1 only.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stripe_study as ss

ss.STRIPE = ss.STRIPE_FNS['blk']

bad = 0
runs = 0
for n in [8, 16, 32, 64, 128, 256]:
    for C in [2, 4, 8, 16]:
        if C > n // 2:
            continue
        for depth in [8, 6]:
            for c in range(C):
                r = ss.fifo_sim_striped(n, C, c, depth=depth, nchunks=3)
                runs += 1
                if r != 'OK':
                    print(f"FAIL n={n} C={C} ch={c} depth={depth} nchunks=3: {r}")
                    bad += 1
print(f"{runs} striped-relay multi-chunk sims")
print("PASS" if bad == 0 else f"{bad} FAILURES")
sys.exit(1 if bad else 0)

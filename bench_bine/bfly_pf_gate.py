"""Striped-BUTTERFLY liveness gate over the SHIPPED postFreq range.

The mode gate only admits the butterfly when pf >= minPost = divUp(n/2, NCCL_STEPS)
and pf is clamped to [1, n/2], so the shipped striped butterfly always runs with
pf in [minPost, n/2]. Sim that whole range (po2 points + minPost+1 odd sample),
every channel, C up to min(16, n) -- including C == n (one block per channel).
Multi-chunk butterfly (block > slot, possible under a huge NCCL_BINE_XOVER at
n <= 16 where minPost == 1) is covered with nchunks=3 at pf = minPost.
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stripe_study as ss

ss.STRIPE = ss.STRIPE_FNS['blk']
NCCL_STEPS = 8

bad = 0
runs = 0
for n in (8, 16, 32, 64, 128, 256):
    minPost = (n // 2 + NCCL_STEPS - 1) // NCCL_STEPS
    for C in (2, 4, 8, 16):
        if C > n:
            continue
        pfs = sorted({p for p in (minPost, minPost + 1, 2 * minPost, n // 4, n // 2)
                      if minPost <= p <= max(1, n // 2)})
        for pf in pfs:
            for c in range(C):
                r = ss.bfly_sim_striped(n, C, c, pf)
                runs += 1
                if r != 'OK':
                    print(f"FAIL n={n} C={C} ch={c} pf={pf}: {r}")
                    bad += 1
        # multi-chunk edge (chunk-loop replay of the packed schedule)
        for c in range(C):
            r = ss.bfly_sim_striped(n, C, c, minPost, nchunks=3)
            runs += 1
            if r != 'OK':
                print(f"FAIL n={n} C={C} ch={c} pf={minPost} nchunks=3: {r}")
                bad += 1

print(f"{runs} striped-butterfly sims over the shipped pf range")
print("PASS" if bad == 0 else f"{bad} FAILURES")
sys.exit(1 if bad else 0)

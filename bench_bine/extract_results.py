#!/usr/bin/env python3
"""Tabulate busbw (GB/s) from a run_compare.slurm output file.

Splits the file on the '###### LABEL (NCCL_ALGO=...) ######' markers emitted by
run_compare.slurm and prints out-of-place busbw per message size for each label,
plus the Bine/PAT speedup. Assumes nccl-tests was run with -c 1 (so each data
row ends with 8 fields: oop[time,algbw,busbw,#wrong] ip[time,algbw,busbw,#wrong]).

Usage: python3 extract_results.py ag_compare_<jobid>.out
"""
import sys, re
from collections import OrderedDict


def humansize(b):
    b = int(b)
    for u, s in (("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10)):
        if b >= s:
            return f"{b / s:g}{u}"
    return str(b)


def parse(path):
    blocks = OrderedDict()  # label -> {size_bytes: busbw_oop}
    cur = None
    for line in open(path):
        m = re.search(r"#+\s*([A-Za-z0-9_]+)\s*\(NCCL_ALGO", line)
        if m:
            cur = m.group(1)
            blocks[cur] = {}
            continue
        f = line.split()
        if cur and len(f) >= 8 and re.fullmatch(r"\d+", f[0]):
            try:
                blocks[cur][int(f[0])] = float(f[-6])  # out-of-place busbw
            except ValueError:
                pass
    return blocks


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: extract_results.py <slurm-out>")
    b = parse(sys.argv[1])
    if not b:
        sys.exit("no labelled blocks found — was this a run_compare.slurm output?")
    labels = list(b.keys())
    sizes = sorted(set().union(*[set(d) for d in b.values()]))
    has_ratio = "BINE" in b and "PAT" in b

    print("\nbusbw GB/s (out-of-place); higher is better\n")
    hdr = f"{'size':>8} " + " ".join(f"{l:>10}" for l in labels)
    if has_ratio:
        hdr += f"{'Bine/PAT':>10}"
    print(hdr)
    print("-" * len(hdr))
    for sz in sizes:
        row = f"{humansize(sz):>8} " + " ".join(
            f"{b[l].get(sz, float('nan')):>10.2f}" for l in labels
        )
        if has_ratio:
            bn, pt = b["BINE"].get(sz), b["PAT"].get(sz)
            row += f"{bn / pt:>10.2f}" if bn and pt else f"{'-':>10}"
        print(row)
    if has_ratio:
        print("\nBine/PAT > 1.0  => Bine faster at that size.")


if __name__ == "__main__":
    main()

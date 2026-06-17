# Bine AllGather — Leonardo benchmark

Compares the **Bine** AllGather schedule against the **original PAT** and **Ring**
on Leonardo Booster (A100, sm_80). Bine replaced PAT's AG schedule *in place*, so
`NCCL_ALGO=PAT` on this checkout runs Bine; the original PAT comes from a clean
baseline checkout built side-by-side.

## 0. Get the code onto Leonardo

The Bine edits may be uncommitted locally. Either:

- **git (recommended):** commit them to a branch and push, e.g.
  ```bash
  git checkout -b bine-allgather
  git add -A && git commit -m "Bine AllGather schedule"
  git push -u origin bine-allgather
  ```
  then on Leonardo `git clone ... && git checkout bine-allgather`.
- **rsync:** `rsync -a --exclude build /path/nccl/ leonardo:work/nccl/`

Either way, commit `5067397` (the pre-Bine HEAD) must exist in history — it's the
baseline. Override with `BASELINE_REF=<hash>` if your baseline differs.

## 1. Build both NCCL variants + nccl-tests

```bash
bash bench_bine/build_leonardo.sh
```
Edit the `module load` lines first (`module avail cuda gcc openmpi`). Produces
`nccl/build` (Bine), `nccl-baseline/build` (PAT), `nccl-tests/build`.

## 2. Run the comparison

```bash
sbatch bench_bine/run_compare.slurm     # edit -A account, ROOT, modules first
```

## Must respect
- **Node count = power of two** (`-N 2/4/8/16…`). Bine falls back to Ring otherwise.
- **1 GPU/node** (`--ntasks-per-node=1`) so `nNodes==nRanks` — required for PAT/Bine.
- **Verify selection:** set `NCCL_DEBUG=INFO` once and confirm the log shows PAT
  for AllGather and an IB net device (not Socket).
- Debug QOS caps at 2 nodes / 30 min; use the normal QOS for larger scaling runs.

Read the `busbw` (GB/s) column to compare bandwidth; `Out of bounds values: 0 OK`
is the correctness signal.

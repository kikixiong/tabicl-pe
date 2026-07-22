# Noether TabICL operational status

**As of:** 2026-07-22 17:13:33 BST (`Europe/London`)

This is a time-stamped snapshot, not a live dashboard. Rerun the refresh
commands at the end of this document before changing jobs or drawing current
conclusions.

## Summary

- Host: `noether.cs.ox.ac.uk`
- Phase: final pre-submission hardening; no formal three-arm cohort submitted
- Pilot source HEAD: `69e6d3d961e16a9b33bc2c9f9107c382df469d2f`;
  later documentation-only publication commits do not change pilot semantics
- Branch: `codex/h100-rope-none-full`
- Public research repository: `https://github.com/kikixiong/tabicl-pe`
- GitHub SSH authentication: verified as `kikixiong`; no `gh` CLI required
- Public branches: `main` at upstream baseline `46b9196`; sanitized experiment
  branch `codex/h100-rope-none-full` at `8513d8a`
- Publication scope: sanitized source and tests only; Noether context documents,
  six historical root-level `tabicl-*.bundle` files, and artifacts are excluded
- Shared filesystem: 3.4T total, 3.2T used, approximately 24G available,
  reported as 100% used
- Repository `artifacts/`: approximately 2.7G
- Formal namespace `position-identity-v1-seed42`: not present
- Access: Noether-native Slurm queries succeeded at this timestamp. The
  external SSH path was separately reported to be timing out during banner
  exchange; do not infer one access path's health from the other.
- Current disposition: leave the healthy pilots running as infrastructure
  evidence until the fixed commit and H100 gates pass and the user explicitly
  authorizes a cutover.

## Slurm snapshot

| Job | Role | State | Node/reason | Elapsed at snapshot |
|---|---|---|---|---:|
| 249092 | No-PE Stage 1 pilot | RUNNING | hopper | 11:20:16 |
| 249093 | No-PE Stage 2 pilot | PENDING | Dependency | 00:00:00 |
| 249094 | No-PE Stage 3 pilot | PENDING | Dependency | 00:00:00 |
| 249095 | RoPE Stage 1 pilot | RUNNING | hopper | 08:49:20 |
| 249096 | RoPE Stage 2 pilot | PENDING | Dependency | 00:00:00 |
| 249097 | RoPE Stage 3 pilot | PENDING | Dependency | 00:00:00 |

Historical Slurm accounting was also reconfirmed:

- `248677`, `248678`: `FAILED` wrappers after completing their intended
  100-step preflight work; corrected active-window GPU gates pass.
- `248699`: `FAILED` wrapper after the FA3 wheel installed successfully.
- `248702`: `COMPLETED` FA3 H100 forward/backward smoke in 13 seconds.
- `248679`: `CANCELLED`, never started.

## Pilot health

Both Stage 1 logs were advancing at the snapshot:

- No-PE `249092`: progress display `20191/494000` after resuming at step 6000,
  corresponding to approximately global step 26191.
- Stable RoPE `249095`: progress display `14531/500000` from scratch.

No OOM, `ENOSPC`, traceback, fatal error, non-finite loss, or NaN signature was
found in the four current stdout/stderr files during the audit.

Active-window GPU summaries measured shortly before this snapshot:

| Job | Samples | Mean | Median | p10 | Gate |
|---|---:|---:|---:|---:|---|
| 249092 | 1345 | 83.8% | 93.0% | 37.0% | PASS |
| 249095 | 1043 | 84.8% | 93.0% | 42.0% | PASS |

The locked gate is mean utilization at least 80% per GPU. These are interim
pilot summaries; rerun the summarizer on completion for final pilot health.

## Checkpoints at snapshot

| Arm | Retained files | Size each | Latest verified |
|---|---|---:|---|
| No-PE | `step-25000.ckpt`, `step-26000.ckpt` | 220,697,053 bytes | step 26000 |
| Stable RoPE | `step-13000.ckpt`, `step-14000.ckpt` | 220,697,438 bytes | step 14000 |

The latest checkpoint in each arm passed `unzip -tq`, confirming no compressed
record CRC errors. The files remain pilot artifacts, not formal evidence.

## Reported upstream/local hardening state

The user reports that the upstream/local patch set now has the three identity
modes and inference cache, isolated checkpointable Temporary RNG, 14 passing
identity RNG tests, exact two-rank Gloo identity-stream recovery, atomic
checkpoint writes, and tested raw evaluation/ingestion/paired statistics. It
also fixes the Stage 2/3 500k-step inheritance bug, removes arbitrary child
checkpoint resume, and restricts formal launches to `RUN_POLICY=fresh`.

None of those claims are represented by the current Noether HEAD. They remain
reported upstream/local state until a fixed commit is transferred, reviewed,
and tested here.

Current upstream work is hardening the mode-aware checkpoint validator, atomic
formal submission/rollback and capacity checks, and commit/environment-bound
six-case H100 maximum-sequence evidence.

## Current blockers

1. The shared filesystem has only about 24G free and reports 100% usage. Do not
   submit the formal cohort or generate substantial artifacts until the disk
   reserve gate passes. The reported estimate suggests seed 42 alone may fit,
   but all three seeds are not safely budgeted.
2. External SSH was reported to have a banner-exchange timeout, although the
   native Noether task could query Slurm at this snapshot.
3. The complete, fixed upstream commit/overlay is not present. The old sync
   archive is stale.
4. The current Noether Slurm wrapper and submitter support only `rope|none`;
   they do not create a matched temporary arm.
5. Formal identity/prior reproducibility, provenance, checkpoint-parent,
   maximum-sequence, non-finite, and disk-reserve gates have not yet been
   deployed and run on this HEAD.
6. Full prior/DataLoader resume equivalence to uninterrupted training has not
   been proved; only the identity stream is reported exactly recoverable.
7. CUDA identity-RNG, NCCL, tiny uninterrupted-versus-resume, and all six H100
   maximum-sequence gates remain outstanding.
8. No-PE W&B resumed from a run whose remote step was ahead of the restored
   checkpoint; steps 6001--6823 were ignored as non-monotonic. This is an
   additional pilot-only provenance limitation.
9. The locked arms directly support the narrower ordinal Row-RoPE comparison.
   Broader temporary-identity conclusions need `cls_only`, non-ordinal
   temporary phase, and row-wise resampling controls.

## Recent actions

- At 2026-07-22 17:28 BST, atomically published the full official baseline as
  public `main` and a sanitized experiment snapshot as public
  `codex/h100-rope-none-full` (`8513d8a`). The publication passed shell tests,
  focused Python tests (9 passed), syntax/whitespace checks, secret/internal-
  metadata scans, and a full Git object-integrity check.
- Recorded the provenance mapping from the pilots' raw source `69e6d3d` and
  local documentation descendant `3d70a17` to public sanitized `8513d8a`.
  These commits are not interchangeable: the public snapshot is squashed,
  omits operational documents, and parameterizes three Slurm paths.
- Refreshed the native Noether job, log, GPU, checkpoint, and disk snapshot at
  2026-07-22 17:13 BST; no pilot job was modified.
- Verified the existing Noether SSH key authenticates to GitHub as `kikixiong`.
  The user explicitly selected the public `kikixiong/tabicl-pe` repository for
  ongoing source and evidence publication without `gh`.
- Performed a read-only repository, Slurm, disk, log, checkpoint, and GPU CSV
  audit on 2026-07-22.
- Did not cancel, submit, requeue, or modify any Slurm job.
- Did not deploy any overlay or alter checkpoints/artifacts.
- Added the repository context-management documents requested by the user:
  `AGENTS.md`, `docs/NOETHER_HANDOFF.md`, and `docs/NOETHER_STATUS.md`.
- Reconciled the user's later external-SSH handoff with a newer Noether-native
  snapshot. No pilot was cancelled or resubmitted.

## Refresh commands

Run from the repository root. These commands are read-only:

```bash
hostname
pwd
git rev-parse --show-toplevel
git rev-parse HEAD
git branch --show-current
git status --short
squeue -u jiaxio
sacct -j 249092,249093,249094,249095,249096,249097 \
  --format=JobIDRaw,JobName,State,ExitCode,Elapsed,Start,End,NodeList
df -h /slurm-storage/jiaxio/ws/TabFM/train/tabicl-v2-pretrain
tail -c 65536 artifacts/logs/tabicl-identity-full-249092.err | tr '\r' '\n' | tail
tail -c 65536 artifacts/logs/tabicl-identity-full-249095.err | tr '\r' '\n' | tail
python3 scripts/summarize_gpu_usage.py \
  artifacts/gpu-monitor/none-stage1-249092.csv \
  --expected-gpus 1 --start-after-active --end-after-active
python3 scripts/summarize_gpu_usage.py \
  artifacts/gpu-monitor/rope-stage1-249095.csv \
  --expected-gpus 1 --start-after-active --end-after-active
find artifacts/tabiclv2-clf-identity -maxdepth 5 -type f -name '*.ckpt' -print
```

After refreshing, replace the volatile facts and timestamp above. Do not append
new live-state claims without evidence.

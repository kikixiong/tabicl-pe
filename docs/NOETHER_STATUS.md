# Noether TabICL operational status

**As of:** 2026-08-05 10:48:14 BST (`Europe/London`)

This is a time-stamped snapshot, not a live dashboard. Rerun the refresh
commands at the end of this document before changing jobs or drawing current
conclusions.

## Summary

- Host: `noether.cs.ox.ac.uk`
- Phase: final pre-submission hardening; no formal three-arm cohort submitted
- Current raw checkout HEAD: `953339565c090f3fc67f4127f19a775962624b86`
  on `codex/h100-rope-none-full`
- Branch: `codex/h100-rope-none-full`
- Public research repository: `https://github.com/kikixiong/tabicl-pe`
- GitHub SSH authentication: verified as `kikixiong`; no `gh` CLI required
- Public branches: `main` at upstream baseline `46b9196`; sanitized experiment
  branch `codex/h100-rope-none-full` at `8513d8a`
- Publication scope: sanitized source and tests only; Noether context documents,
  six historical root-level `tabicl-*.bundle` files, and artifacts are excluded
- Shared filesystem: 3.4T total, 3.1T used, approximately 184G available,
  reported as 95% used
- Formal namespace `position-identity-v1-seed42`: not present
- Access: Noether-native Slurm queries succeeded at this timestamp. The
  external SSH path was separately reported to be timing out during banner
  exchange; do not infer one access path's health from the other.
- Current disposition: the resumed RoPE/No-RoPE chains are running as pilot
  infrastructure only. They are not formal evidence because they resume old
  checkpoints without exact prior/DataLoader state or immutable provenance.

## Slurm snapshot

| Job | Role | State | Node/reason | Elapsed at snapshot |
|---|---|---|---|---:|
| 253081 | No-PE Stage 1 resumed pilot | RUNNING | hopper | 00:02:40 |
| 253082 | No-PE Stage 2 pilot | PENDING | Dependency | 00:00:00 |
| 253083 | No-PE Stage 3 pilot | PENDING | Dependency | 00:00:00 |
| 253084 | RoPE Stage 1 resumed pilot | RUNNING | hopper | 00:02:40 |
| 253085 | RoPE Stage 2 pilot | PENDING | Dependency | 00:00:00 |
| 253086 | RoPE Stage 3 pilot | PENDING | Dependency | 00:00:00 |

Historical Slurm accounting was also reconfirmed:

- `248677`, `248678`: `FAILED` wrappers after completing their intended
  100-step preflight work; corrected active-window GPU gates pass.
- `248699`: `FAILED` wrapper after the FA3 wheel installed successfully.
- `248702`: `COMPLETED` FA3 H100 forward/backward smoke in 13 seconds.
- `248679`: `CANCELLED`, never started.

## Pilot health

Both newly resumed Stage 1 logs were advancing at the snapshot:

- No-PE `253081`: loaded `step-149000.ckpt` with full model, optimizer,
  scheduler, and step state, then advanced to approximately global step 149030.
- Stable RoPE `253084`: loaded `step-129000.ckpt` with full model, optimizer,
  scheduler, and step state, then advanced to approximately global step 129026.

The most recent GPU-monitor samples were 100% utilization on both allocated
H100s. This is an initial-start check, not yet a complete active-window mean.
No OOM, `ENOSPC`, traceback, fatal error, non-finite loss, or NaN signature was
found in the current logs.

Jobs `252749` and `252752` had previously remained Slurm-RUNNING while both
training processes were stalled from 2026-08-04 06:17 BST: GPU utilization was
continuously 0%, logs/checkpoints stopped, and 65--67 GiB remained allocated.
With explicit user authorization, both stale Stage 1 jobs and their four old
dependencies (`252749`--`252754`) were cancelled at 2026-08-05 10:45 BST and
replaced by the current chains.

## Checkpoints at snapshot

| Arm | Retained files | Size each | Latest verified |
|---|---|---:|---|
| No-PE | `step-148000.ckpt`, `step-149000.ckpt` | 220,697,839 bytes | step 149000 |
| Stable RoPE | `step-128000.ckpt`, `step-129000.ckpt` | 220,698,225 bytes | step 129000 |

The latest checkpoint in each arm passed ZIP CRC, full `torch.load`, mode/step,
optimizer/scheduler, and finite-tensor checks. The files remain pilot artifacts,
not formal evidence. The resume discarded the unsaved ranges 149001--149483
and 129001--129426; the reused W&B runs will ignore those duplicate step logs
until the new training passes the old remote step.

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

1. The shared filesystem has approximately 184G free and reports 95% usage.
   This clears the old immediate 20G reserve failure but does not replace a
   full worst-case capacity calculation before formal submission.
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

- At 2026-08-05 10:45 BST, confirmed that `252749` and `252752` had been
  stalled for about 28 hours with persistent 0% GPU utilization. After
  validating `none/step-149000.ckpt` and `rope/step-129000.ckpt`, cancelled the
  exact old chains `252749`--`252754` with user authorization.
- Submitted replacement pilot chains `253081`--`253086`. Both Stage 1 jobs
  started immediately on hopper, loaded the intended complete training states,
  advanced multiple steps, and produced non-zero/100% GPU-utilization samples.
- Did not delete or alter any checkpoint, log, W&B state, bundle, or source
  code. These resumed jobs remain pilot-only and are not formal evidence.
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
sacct -j 253081,253082,253083,253084,253085,253086 \
  --format=JobIDRaw,JobName,State,ExitCode,Elapsed,Start,End,NodeList
df -h /slurm-storage/jiaxio/ws/TabFM/train/tabicl-v2-pretrain
tail -c 65536 artifacts/logs/tabicl-identity-full-253081.err | tr '\r' '\n' | tail
tail -c 65536 artifacts/logs/tabicl-identity-full-253084.err | tr '\r' '\n' | tail
python3 scripts/summarize_gpu_usage.py \
  artifacts/gpu-monitor/none-stage1-253081.csv \
  --expected-gpus 1 --start-after-active --end-after-active
python3 scripts/summarize_gpu_usage.py \
  artifacts/gpu-monitor/rope-stage1-253084.csv \
  --expected-gpus 1 --start-after-active --end-after-active
find artifacts/tabiclv2-clf-identity -maxdepth 5 -type f -name '*.ckpt' -print
```

After refreshing, replace the volatile facts and timestamp above. Do not append
new live-state claims without evidence.

# Noether TabICL operational status

**As of:** 2026-08-06 08:34:42 BST (`Europe/London`)

This is a time-stamped snapshot, not a live dashboard. Rerun the refresh
commands at the end of this document before changing jobs or drawing current
conclusions.

## Summary

- Host: `noether.cs.ox.ac.uk`
- Phase: final pre-submission hardening; no formal three-arm cohort submitted
- Current raw checkout HEAD: `80714179ee792fbc39cb62591073b41f8da96042`
  on `codex/h100-rope-none-full`
- Branch: `codex/h100-rope-none-full`
- Public research repository: `https://github.com/kikixiong/tabicl-pe`
- GitHub SSH authentication: verified as `kikixiong`; no `gh` CLI required
- Public branches: `main` at upstream baseline `46b9196`; sanitized experiment
  branch `codex/h100-rope-none-full` at `8513d8a`
- Publication scope: sanitized source and tests only; Noether context documents,
  six historical root-level `tabicl-*.bundle` files, and artifacts are excluded
- Shared filesystem: 3.4T total, 3.2T used, approximately 106G available,
  reported as 97% used. It reached zero free bytes on 2026-08-05 and later
  recovered; capacity remains a hard live gate.
- Formal namespace `position-identity-v1-seed42`: not present
- Access: Noether-native Slurm queries succeeded at this timestamp. The
  external SSH path was separately reported to be timing out during banner
  exchange; do not infer one access path's health from the other.
- Current disposition: the replacement RoPE/No-RoPE chains are running as
  pilot infrastructure only. They are not formal evidence because they resume
  old checkpoints without exact prior/DataLoader state or immutable
  provenance.

## Slurm snapshot

| Job | Role | State | Node/reason | Elapsed at snapshot |
|---|---|---|---|---:|
| 253196 | No-PE Stage 1 replacement pilot | RUNNING | hopper | 00:04:54 |
| 253197 | No-PE Stage 2 pilot | PENDING | Dependency | 00:00:00 |
| 253198 | No-PE Stage 3 pilot | PENDING | Dependency | 00:00:00 |
| 253199 | RoPE Stage 1 replacement pilot | RUNNING | hopper | 00:04:54 |
| 253200 | RoPE Stage 2 pilot | PENDING | Dependency | 00:00:00 |
| 253201 | RoPE Stage 3 pilot | PENDING | Dependency | 00:00:00 |

Historical Slurm accounting was also reconfirmed:

- `248677`, `248678`: `FAILED` wrappers after completing their intended
  100-step preflight work; corrected active-window GPU gates pass.
- `248699`: `FAILED` wrapper after the FA3 wheel installed successfully.
- `248702`: `COMPLETED` FA3 H100 forward/backward smoke in 13 seconds.
- `248679`: `CANCELLED`, never started.

## Pilot health

Both replacement Stage 1 jobs started on hopper at 2026-08-06 08:29:48 BST,
loaded the intended checkpoints, and were advancing at the snapshot:

- No-PE `253196`: loaded `step-164000.ckpt` and advanced to approximately
  global step 164097.
- Stable RoPE `253199`: loaded `step-143000.ckpt` and advanced to approximately
  global step 143094.

Recent GPU-monitor samples were active, reaching 100% utilization on both
allocated H100s. This is an initial-start check, not yet a complete
active-window mean. No OOM, `ENOSPC`, traceback, fatal CUDA/NCCL error,
non-finite loss, or NaN signature was found in the replacement logs. The W&B
non-monotonic-step warnings are expected pilot provenance noise: each reused
remote run is ahead of the restored checkpoint.

On 2026-08-05 the shared filesystem fell from tens of GiB free to zero. Jobs
`253081` and `253084` then stopped advancing at approximately global steps
164373 and 143809: logs stopped, GPU utilization became persistently 0%, and
allocated memory remained resident. Free space later recovered, but the
workers did not. With explicit user authorization, both stalled Stage 1 jobs
and their four dependencies (`253081`--`253086`) were cancelled at
2026-08-06 08:28:47 BST and replaced by the current chains. No artifact was
deleted.

## Checkpoints at snapshot

| Arm | Retained files | Size each | Latest verified |
|---|---|---:|---|
| No-PE | `step-163000.ckpt`, `step-164000.ckpt` | 220,697,839 bytes | step 164000 |
| Stable RoPE | `step-142000.ckpt`, `step-143000.ckpt` | 220,698,225 bytes | step 143000 |

The latest checkpoint in each arm was previously validated for full model,
optimizer, scheduler, step state, finite tensors, and expected mode; ZIP CRC
was reconfirmed at this snapshot. The files remain pilot artifacts, not formal
evidence. The resume repeats the unsaved ranges 164001--164373 and
143001--143809; reused W&B runs will ignore duplicate step logs until the new
training passes the old remote step.

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

1. The shared filesystem has approximately 106G free and reports 97% usage.
   It reached zero free bytes on 2026-08-05 and stalled both predecessor
   workers. The recovered space does not replace continuous monitoring or a
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
8. Both replacement W&B runs resumed from remote steps ahead of their restored
   checkpoints. Duplicate steps are ignored as non-monotonic until the workers
   pass approximately 164373 and 143809. This is an additional pilot-only
   provenance limitation.
9. The locked arms directly support the narrower ordinal Row-RoPE comparison.
   Broader temporary-identity conclusions need `cls_only`, non-ordinal
   temporary phase, and row-wise resampling controls.

## Recent actions

- On 2026-08-05, detected that shared storage reached zero free bytes and that
  Stage 1 jobs `253081` and `253084` became stuck with 0% GPU utilization,
  frozen logs, and resident GPU memory. They did not recover when space
  returned.
- On 2026-08-06, after explicit user authorization, reconfirmed ZIP integrity
  of `none/step-164000.ckpt` and `rope/step-143000.ckpt`, then cancelled the
  exact stalled chains `253081`--`253086` at 08:28:47 BST.
- Submitted replacement pilot chains `253196`--`253201`. Stage 1 jobs `253196`
  and `253199` started at 08:29:48 BST, loaded the intended checkpoints,
  advanced multiple steps, and produced active GPU-utilization samples.
- Did not delete or alter any checkpoint, log, W&B state, bundle, or source
  code. These replacement jobs remain pilot-only and are not formal evidence.

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
sacct -j 253196,253197,253198,253199,253200,253201 \
  --format=JobIDRaw,JobName,State,ExitCode,Elapsed,Start,End,NodeList
df -h /slurm-storage/jiaxio/ws/TabFM/train/tabicl-v2-pretrain
tail -c 65536 artifacts/logs/tabicl-identity-full-253196.err | tr '\r' '\n' | tail
tail -c 65536 artifacts/logs/tabicl-identity-full-253199.err | tr '\r' '\n' | tail
python3 scripts/summarize_gpu_usage.py \
  artifacts/gpu-monitor/none-stage1-253196.csv \
  --expected-gpus 1 --start-after-active --end-after-active
python3 scripts/summarize_gpu_usage.py \
  artifacts/gpu-monitor/rope-stage1-253199.csv \
  --expected-gpus 1 --start-after-active --end-after-active
find artifacts/tabiclv2-clf-identity -maxdepth 5 -type f -name '*.ckpt' -print
```

After refreshing, replace the volatile facts and timestamp above. Do not append
new live-state claims without evidence.

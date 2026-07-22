# Noether TabICL position-identity handoff

This document is the stable handoff for the TabICLv2 position-identity study.
For live job, disk, log, and checkpoint state, read `NOETHER_STATUS.md` and
refresh its evidence before acting.

Current phase: final pre-submission hardening. No new formal three-arm training
cohort has been submitted.

## Objective

Complete a training/inference-aligned and reproducible three-arm TabICLv2
classifier study:

1. Stable RoPE (`rope`)
2. Temporary Table-wise Identity (`temporary`)
3. No positional encoding (`none`)

The locked confirmatory seeds are `{42, 43, 44}`. Each arm uses the same prior,
architecture, optimizer configuration, training budget, and seed-specific data
stream. The official training schedule is FP32 `500000 + 40000 + 10000` steps.
The actual launcher precision semantics must be recorded explicitly; at the
current pilot commit it passes `--dtype float32 --amp True`, and the training
code constructs a float32 autocast context.

Required analyses include ID/OOD performance, column-permutation robustness,
Temporary-K, mechanism analysis, paired statistics, official ensemble and
single-view results, raw predictions, checkpoints, and complete provenance.

The narrow claim directly supported by the locked arms is **stable ordinal
Row-RoPE versus table-wise random ordinal Row-RoPE versus no Row-RoPE**. A
stronger generic "temporary identity" claim should add `cls_only`, a
non-ordinal temporary phase, and row-wise resampling controls. Keep conclusions
within the actual controls and evidence.

## Repository identity

- Host: `noether.cs.ox.ac.uk`
- Slurm path: `/slurm-storage/jiaxio/ws/TabFM/train/tabicl-v2-pretrain`
- Real path: `/mnt/data/slurm-storage/jiaxio/ws/TabFM/train/tabicl-v2-pretrain`
- Working branch: `codex/h100-rope-none-full`
- Baseline handoff commit: `69e6d3d961e16a9b33bc2c9f9107c382df469d2f`
- Local operational-doc commit: `3d70a17f42efaea65e7800cdce21fd2e2e7028da`
- Public research repository: `https://github.com/kikixiong/tabicl-pe`
- Research push URL: `git@github.com:kikixiong/tabicl-pe.git`
- Public branch: `codex/h100-rope-none-full`
- Public sanitized commit: `8513d8a19afd8b301bc08ab05dbec9bd34e09cc6`
- Dedicated public checkout: `/mnt/data/slurm-storage/jiaxio/ws/TabFM/train/tabicl-pe-public`

The public research repository is an explicit user choice. Publish source,
tests, small protocol documents, and machine-readable provenance only. Never
publish checkpoints, raw predictions, W&B state, logs, GPU CSVs, credentials,
transfer bundles, or the Noether context documents in this checkout. Keep the
official `soda-inria/tabicl` remote available as the upstream reference.

The first public publication is a sanitized squash of the functional state at
raw commit `69e6d3d`: public commit `8513d8a`. It excludes `AGENTS.md` and the
two Noether context documents, changes the author to the GitHub noreply
identity, and replaces three hard-coded Slurm storage paths with
`TABICL_ROOT`/`SLURM_SUBMIT_DIR`-based paths. Consequently these commits are
scientifically related but not byte-identical. The running pilots remain bound
to raw commit `69e6d3d`; do not relabel them as runs of public commit
`8513d8a`.

Do not add the public GitHub repository as a push remote in this raw checkout.
All future public commits and pushes must be prepared and reviewed in the
dedicated sanitized checkout named above.

Always recheck the branch, HEAD, and worktree. The six root-level
`tabicl-*.bundle` files are known historical untracked inputs; their presence
does not authorize applying or deleting them.

## Evidence classification

The existing `249092`--`249097` chains are infrastructure pilots only:

| Arm | Stage 1 | Stage 2 | Stage 3 |
|---|---:|---:|---:|
| No-PE | 249092 | 249093 | 249094 |
| Stable RoPE | 249095 | 249096 | 249097 |

The No-PE pilot resumed from `step-6000.ckpt` without exact RNG/DataLoader
state. Its W&B run had already advanced to step 6824, so duplicate log steps
6001--6823 were ignored after resume. These limitations prevent causal pairing
even if all three stages complete.

The first formal cohort is the immutable namespace
`position-identity-v1-seed42`. It requires three fresh one-H100 chains (nine
jobs including dependent stages). It was not present at the baseline handoff.
Do not reuse a pilot namespace or checkpoint for formal evidence.

## Important paths

- Slurm logs: `artifacts/logs/`
- GPU monitor CSVs: `artifacts/gpu-monitor/`
- Pilot checkpoints: `artifacts/tabiclv2-clf-identity/`
- Existing experiment description: `experiments/TEMPORARY_IDENTITY.md`
- Stable handoff: `docs/NOETHER_HANDOFF.md`
- Volatile state: `docs/NOETHER_STATUS.md`

Known pilot files include:

- `artifacts/logs/tabicl-identity-full-249092.{out,err}`
- `artifacts/logs/tabicl-identity-full-249095.{out,err}`
- `artifacts/gpu-monitor/none-stage1-249092.csv`
- `artifacts/gpu-monitor/rope-stage1-249095.csv`

## Overlay and implementation state

The baseline HEAD does not implement the complete formal submission path. In
particular, the Slurm wrapper accepts only `rope|none`, and the current submit
script creates only those two arms. Although the Stage 1 training entrypoint
recognizes `temporary`, this is not sufficient for a formal three-arm cohort.

The consolidated overlay described by the external handoff was still being
updated and was not available in this repository. The following work was later
reported complete in the upstream/local patch set, but was **not synced to or
verified on Noether at the current baseline HEAD**:

- Stable RoPE, Temporary, and No-PE training/inference cache semantics;
- a separately checkpointed Temporary identity RNG that does not consume the
  global CUDA RNG;
- identity RNG tests (14 passing locally, with one CUDA-only test outstanding)
  and exact two-rank Gloo interrupted/resumed identity sequences;
- atomic checkpoint writes, including three passing failure-path tests;
- raw permutation evaluation, result ingestion, and paired-statistics tests;
- correction of a launcher bug that previously allowed Stage 2/3 to inherit
  500k steps instead of enforcing `500k / 40k / 10k`;
- removal of implicit Stage 2/3 resume from arbitrary unvalidated child
  checkpoints; and
- a formal wrapper contract restricted to `RUN_POLICY=fresh`.

The upstream/local patch set was still being hardened in these areas:

- one fixed commit containing identity RNG, atomic checkpoints, and launcher
  hardening;
- checkpoint validation that requires sampler state for `temporary`, rejects
  it for `rope`/`none`, and locks architecture, prior, seed, source commit, and
  parent lineage;
- atomic formal submission with rollback of only newly submitted jobs, invalid
  dependency cleanup, and a capacity budget;
- six exact commit/environment-bound H100 maximum-sequence smoke proofs; and
- CUDA identity RNG, NCCL, tiny uninterrupted-versus-resume, and maximum-
  sequence validation after Noether access is available.

Exact prior/DataLoader resume equivalence remains unproven. Exact identity
stream recovery, including two-rank Gloo recovery, must not be presented as a
proof of the complete training trajectory.

Do not deploy the previously generated `tabicl-position-identity-sync.tgz`:
its recorded digest was
`f372ffcdfea67f14cbd136917461a67ddf71ff5e0f8b46d5d64823ec666f2ac3`,
but the archive was already stale when the handoff was written.

## Ordered continuation procedure

1. Run a read-only audit: HEAD/worktree, `squeue`, `sacct`, `df -h`, pilot
   logs, current checkpoints, GPU CSVs, and W&B state.
2. Let the current pilots continue as throughput/infrastructure evidence while
   they are healthy. A recommendation to replace them later is not immediate
   authorization to cancel them.
3. Obtain the newly fixed source commit or a newly generated complete overlay
   after upstream writes stop.
   Verify its digest and manifest before applying it.
4. Apply only the audited overlay. Review all diffs and run identity/cache,
   prior reproducibility, launcher, safety-gate, compilation, and whitespace
   checks.
5. Commit the frozen training protocol and launcher independently from the
   evaluation implementation. Record both SHAs.
6. On H100, run the outstanding CUDA identity-RNG test, NCCL resume test, tiny
   uninterrupted-versus-resume test, and the six commit/environment-bound
   maximum-sequence smokes. Require no OOM or non-finite output and mean
   active-window utilization at least 80% per GPU.
7. Re-audit storage and calculate checkpoint/log/raw-prediction capacity for
   the entire requested cohort, not only seed 42.
8. Only after every gate passes and the user authorizes the cutover, audit and
   cancel the old pilots/dependents, then atomically submit a fresh immutable
   seed-42 three-arm cohort with `RUN_POLICY=fresh`.
9. Repeat the unchanged protocol for seeds 43 and 44, then run the locked
   evaluation and paired statistical analysis.

## Definition of complete evidence

A result is paper-usable only when it can be traced to the immutable study
namespace, seed, arm, stage, Git SHA, launcher arguments, prior/data-stream
identity, parent checkpoint, environment, raw predictions, and evaluation SHA.
Aggregate metrics without those artifacts are diagnostic only.

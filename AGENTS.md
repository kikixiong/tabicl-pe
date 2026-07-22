# Agent instructions for the Noether TabICL study

These instructions apply to this repository and all of its subdirectories.

## Start here

Before changing code, jobs, or artifacts:

1. Confirm the repository with `pwd` and `git rev-parse --show-toplevel`.
2. Inspect `git status --short` and preserve all pre-existing work, including
   untracked files.
3. Read `docs/NOETHER_HANDOFF.md` for the stable research contract.
4. Read `docs/NOETHER_STATUS.md` for the latest time-stamped operational
   snapshot. Treat it as historical until its commands are rerun.
5. If Slurm jobs are in scope, refresh `squeue`, `sacct`, `df -h`, the relevant
   logs, checkpoints, and GPU-monitor CSVs before taking action.
6. Keep reported upstream/local patch state separate from code actually present
   and verified on Noether. Never describe an unsynced change as deployed.

## Safety and ownership

- This is a shared Noether workspace. Changes not made by the current agent
  belong to the user or another process; do not discard or overwrite them.
- Do not cancel, requeue, resubmit, or alter an existing Slurm job unless the
  current user request explicitly authorizes that action.
- Do not delete or prune checkpoints, logs, W&B state, bundles, or other
  artifacts without explicit authorization and an exact target audit.
- Do not deploy an archive merely because it exists. Verify that it is the
  current, complete overlay and check its digest and manifest first. The old
  `tabicl-position-identity-sync.tgz` described in prior handoffs is stale.
- Never store credentials, tokens, `.netrc` contents, or private keys in this
  repository or its context documents.
- The research repository `kikixiong/tabicl-pe` is public by explicit user
  choice. Before every push, inspect the exact staged paths and reject secrets,
  checkpoints, raw predictions, logs, GPU CSVs, W&B state, and transfer bundles.
- Never push this raw operational checkout to the public repository: its local
  history and context documents intentionally contain Noether-only metadata.
  Publish from the sanitized checkout at `../tabicl-pe-public`, where `origin`
  is `git@github.com:kikixiong/tabicl-pe.git` and `upstream` is
  `https://github.com/soda-inria/tabicl.git`.
- Maintain an explicit raw-to-public commit mapping in
  `docs/NOETHER_HANDOFF.md`. A sanitized or squashed public commit is not the
  exact source of a running job unless its tree and provenance say so.
- Disk reserve is a hard gate. The shared filesystem has previously reached
  100% usage and produced `ENOSPC`; inspect it before tests, checkpoint writes,
  submissions, or artifact generation.

## Locked research contract

The confirmatory study compares exactly three arms:

- Stable RoPE (`rope`)
- Temporary Table-wise Identity (`temporary`)
- No positional encoding (`none`)

Use matched seeds `{42, 43, 44}` and the official three-stage training budget
`500000 + 40000 + 10000` steps. Prior, architecture, optimizer configuration,
training budget, evaluation views, and seeds must be matched across arms. Do
not silently change the protocol to solve scheduling, memory, speed, or disk
problems. Record and obtain approval for any proposed deviation.

Training and inference identity semantics must be tested as one contract. The
formal evaluation includes ID/OOD, column-permutation robustness, Temporary-K,
mechanism analysis, paired statistics, raw predictions, and provenance.

Use precise scientific language. The currently locked three arms directly test
stable ordinal Row-RoPE versus table-wise random ordinal Row-RoPE versus no
Row-RoPE. A broader claim about generic "temporary identity" requires explicit
controls such as `cls_only`, a non-ordinal temporary phase, and row-wise
resampling. Do not imply that those controls were run unless their artifacts
and provenance exist.

Jobs `249092` through `249097` are infrastructure pilots, not confirmatory
evidence. Do not relabel their outputs as formal results. The first formal
namespace is `position-identity-v1-seed42` and must be created only after all
acceptance gates pass.

## Implementation and verification rules

- Keep the frozen training protocol/launcher commit separate from the later
  evaluation implementation commit. Record both Git SHAs in provenance.
- Before formal submission, run identity/cache tests, prior reproducibility
  tests, launcher and safety-gate tests, Python compilation, and
  `git diff --check`.
- Require mode-aware checkpoint validation: `temporary` checkpoints must carry
  the identity sampler state, while `rope` and `none` checkpoints must reject
  it. Lock architecture, prior, seed, source commit, and parent lineage.
- Formal launches must use `RUN_POLICY=fresh`. Do not resume a formal child
  from an arbitrary or unvalidated checkpoint.
- Bind H100 smoke evidence to the exact source commit and environment. Require
  the complete six-case maximum-sequence matrix plus CUDA identity-RNG, NCCL,
  and tiny uninterrupted-versus-resume checks before submission.
- Formal multi-job submission must be atomic from the user's perspective:
  validate capacity first, detect invalid dependencies, and cancel only the
  jobs created by a failed submission attempt. Never touch pre-existing jobs
  during rollback.
- Each allocated H100 must pass the locked active-window utilization gate:
  mean utilization at least 80% per card. OOM, skipped OOM, non-finite values,
  missing parent checkpoints, provenance mismatch, or insufficient disk fails
  the gate.
- Identity-stream reproducibility alone is not full resume reproducibility.
  The prior/DataLoader trajectory must also match uninterrupted execution
  before exact-resume claims are allowed.
- Prefer immutable, seed-specific artifact namespaces. Preserve raw
  predictions and machine-readable provenance alongside aggregate statistics.
- Do not commit generated checkpoints, W&B data, GPU CSVs, or large bundles.

## Context-document discipline

- `docs/NOETHER_HANDOFF.md` contains stable goals, invariants, paths, known
  limitations, and the ordered handoff procedure. Update it only when those
  facts change.
- `docs/NOETHER_STATUS.md` contains volatile state. Every update must include
  an explicit timezone-aware timestamp and the commands or evidence used.
- Replace superseded status facts instead of appending an unbounded diary.
  Keep a short "recent actions" section for decisions that matter to the next
  agent.
- When finishing material work, update both documents as appropriate and make
  clear which actions were performed, which were only inspected, and what
  remains blocked.

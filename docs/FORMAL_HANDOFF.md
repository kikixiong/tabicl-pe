# Formal Hardware Handoff

This handoff begins only after the candidate commit is final. Hardware evidence
must be generated from a clean detached checkout of that exact commit; changing
code or documentation afterward creates a different candidate.

## 1. Prepare the exact candidate

Use an isolated interpreter and explicitly disable user-site imports. The only
accepted source is `https://github.com/kikixiong/tabicl-pe.git` at
`refs/heads/codex/position-identity-v1`. Before creating an artifact namespace
or calling `sbatch`, execute the digest-bound absolute Git binary with the
bounded, 30-second `ls-remote --refs` query and require that exact ref to return
the intended commit as its only stdout row and no stderr. Verify the checkout
is detached and clean, Git object integrity passes, and the tracked tree
matches the expected tree. Put pytest caches, bytecode, temporary directories,
logs, checkpoints, and evidence outside the checkout.

Every compute case independently repeats the commit/tree/source/import
attestation. A successful attestation from another case cannot be reused.

## 2. Run the canonical twelve-case H100 matrix

The case set is exact:

1. Three Stage 1 one-step functional cases: `rope`, `temporary`, and `none`,
   using the formal GraphSCM prior and its 16-worker/4-per-group CPU data path.
2. Three Stage 2 cases at an observed fixed sequence length of 10,240.
3. Three Stage 3 cases at an observed fixed sequence length of 60,000 with
   activation recomputation enabled.
4. One Temporary CUDA RNG isolation-and-resume case.
5. One two-rank NCCL initialization/all-reduce/barrier case. It does not claim
   distributed state restore.
6. One real GraphSCM prior/DataLoader tiny uninterrupted-versus-resume case with
   worker and prefetch state exercised.

The three one-step cases establish functionality and the real Stage 1
GraphSCM/DataLoader integration; they make no utilization claim. The six Stage 2/3
max-sequence cases establish utilization: use one-second unfiltered sampling,
an explicitly signaled active training window, at least ten active samples on
every allocated GPU, and a per-GPU arithmetic mean of at least 80%. If a case
is too short, repeat iterations at the same fixed shape; do not change the
formal sequence length. All nine Trainer cases use one H100 and save a full
checkpoint bounded by the precommitted `C`. The NCCL case is the only two-GPU
case.

Submit this matrix with `scripts/submit_h100_identity_validation.sh`, using a
canonical external overlay and `RUN_POLICY=fresh`. The controller hard-codes
the `h100` partition, `short` QOS, and three-hour limit. Eleven cases request
one H100, 32 CPUs, and 128 GiB; only `nccl_2gpu` uses its separate static
wrapper and requests two H100s and 64 CPUs. Both wrappers independently reject
the wrong case/GPU count and inspect exactly the CUDA-visible allocation
(ordinal, GPU UUID, or MIG token) before cloning the candidate. Node-global
GPUs outside that allocation are never included in runtime or utilization
evidence.

The overlay precommits separate one- and two-GPU environment digests. They
represent the same Python/Torch/CUDA/cuDNN environment; the final assembler
requires their runtime envelopes to differ only in
`visible_cuda_device_count` (one versus two). Every case checks its applicable
digest immediately after compute. The controller derives a unique artifact
identity from the validation ID, case, exact commit/tree/source manifest,
applicable environment digest, and `C`; callers do not supply arbitrary case
identities.

Before creating the fresh namespace or calling `sbatch`, the controller takes
a stable `statvfs` snapshot of the artifact parent. Let `A` be that
filesystem's positive `f_frsize`, and let `ceil_A(x)` round a positive file
ceiling up to a multiple of `A`. The receipt reports the logical file total
below, but admission is based on the fragment-rounded physical budget:

```text
W_logical = 9*C
  + 12*RUN_LOG_CEILING
  + 37*ATTESTATION_CEILING     # 36 case files plus final smoke attestation
  + 12*(1 MiB)                 # case-result publications
  + 12*GPU_MONITOR_CEILING     # CSV + JSONL for six max-sequence cases
  + 6*RUN_LOG_CEILING          # six GPU summaries
  + 4*RECEIPT_CEILING          # held, success, recovery, and durable journal
  + 24*SCHEDULER_LOG_CEILING   # stdout/stderr for twelve Slurm cases

F_physical = 9*ceil_A(C)
  + 12*ceil_A(RUN_LOG_CEILING)
  + 37*ceil_A(ATTESTATION_CEILING)
  + 12*ceil_A(1 MiB)
  + 12*ceil_A(GPU_MONITOR_CEILING)
  + 6*ceil_A(RUN_LOG_CEILING)
  + 4*ceil_A(RECEIPT_CEILING)
  + 24*ceil_A(SCHEDULER_LOG_CEILING)

D_physical = (28 + 157)*A
required_free_bytes = ceil_A(20 GiB) + F_physical + D_physical
```

The 28 directory slots are the complete gate directory topology. The 157
directory-entry slots are one artifact-root entry, 27 child-directory entries,
116 durable file entries, and thirteen extra temporary entries because all
twelve cases may overlap an atomic publication while the controller publishes
the final receipt. This budget therefore does not depend on current GPU
concurrency or scheduler serialization.

The physical `case_work_root` must be on a different `st_dev` from the artifact
parent. This no-follow descriptor check happens before namespace creation,
repeats against the created artifact root during the held capacity recheck,
and runs again on the compute node before `mktemp` or `git clone`. Device drift
at either controller check rolls the held transaction back.

The parent inode/metadata must be unchanged across the snapshot and available
bytes must be greater than or equal to the requirement. Every term, the
observed available blocks/bytes, logical total, fragment-rounded file and
directory components, and `required_free_bytes` are stored in both the held
plan and successful receipt. The controller repeats the snapshot while all
twelve jobs are still held and binds that fresher result; failure rolls the
batch back before release. One byte below either boundary is a hard block.

The same GitHub query is repeated while all twelve jobs remain held. The
canonical URL, fixed ref, candidate SHA, Git executable SHA-256, repository
identity SHA-256, and query/output SHA-256 are stored in the held plan and
released receipt. Every compute wrapper receives those exact values, re-runs
the query before `mktemp` or clone, and its runtime allocation is bound to the
held-plan digest. Ref movement or an unadvertised commit rolls the held batch
back before release.

All twelve jobs are submitted held with collision-resistant transaction-bound
job names and external `--chdir`, `--output`, and `--error` paths. The
controller fsyncs a hash-chained transaction journal after every scheduler
transition, publishes a durable held plan, releases and reconciles every job,
and only then publishes
`submission-receipt.json`. Any submission or release failure cancels only this
batch in reverse order and reconciles terminal scheduler state. Release and
cancellation reconciliation use three bounded, no-sleep observations so one
short scheduler-propagation lag does not create a false recovery. SIGINT,
SIGTERM, or Python interruption follows the same rollback path. A failed
cancellation or an untrusted successful `sbatch` result produces
`rollback-recovery.json`. Scheduler executables are absolute paths whose bytes
are precommitted in the overlay and receipt; cluster suffixes returned by
`sbatch --parsable` are preserved on every later scheduler operation. If an
`sbatch` response is lost, the controller searches `squeue` and then `sacct`
for the exact full-transaction job name; exactly one match is absorbed into
the rollback, while zero or multiple matches remain a fail-closed recovery.
That response-loss discovery is scoped to the local single cluster used by
this study; it must not be presented as federation-safe reconciliation.
The artifact namespace is
fresh-only and has a dedicated `cases/` child, so the final assembly input
contains exactly the twelve case directories.

The canonical overlay has these exact fields (serialize it with sorted keys,
compact separators, and one trailing newline):

```json
{
  "artifact_root": "/external/fresh-gate-root",
  "environment_sha256_by_world_size": {"1": "ONE_GPU_SHA256", "2": "TWO_GPU_SHA256"},
  "kind": "h100_identity_gate_submit",
  "limits": {
    "attestation_ceiling_bytes": 1000000,
    "checkpoint_ceiling_bytes": 300000000,
    "gpu_monitor_ceiling_bytes": 10000000,
    "receipt_ceiling_bytes": 1000000,
    "run_log_ceiling_bytes": 10000000,
    "scheduler_log_ceiling_bytes": 1000000
  },
  "run_policy": "fresh",
  "runtime": {
    "case_work_root": "/node-local/existing-work-root",
    "git": "/absolute/path/to/git",
    "git_sha256": "GIT_SHA256",
    "nvidia_smi": "/absolute/path/to/nvidia-smi",
    "python": "/absolute/path/to/python"
  },
  "schema_version": 1,
  "scheduler_commands": {
    "sacct": {"path": "/absolute/path/to/sacct", "sha256": "SACCT_SHA256"},
    "sbatch": {"path": "/absolute/path/to/sbatch", "sha256": "SBATCH_SHA256"},
    "scancel": {"path": "/absolute/path/to/scancel", "sha256": "SCANCEL_SHA256"},
    "scontrol": {"path": "/absolute/path/to/scontrol", "sha256": "SCONTROL_SHA256"},
    "squeue": {"path": "/absolute/path/to/squeue", "sha256": "SQUEUE_SHA256"}
  },
  "source": {
    "candidate_repository": "https://github.com/kikixiong/tabicl-pe.git",
    "candidate_ref": "refs/heads/codex/position-identity-v1",
    "commit_sha": "COMMIT",
    "manifest_path": "/external/source-manifest.json",
    "manifest_sha256": "SOURCE_SHA256",
    "tree_sha": "TREE"
  },
  "validation_id": "candidate-gate-v1"
}
```

Final assembly must bind that released receipt:

```text
python -I -B scripts/run_h100_identity_validation.py \
  --assemble-cases-root /external/gate/cases \
  --submission-receipt /external/gate/submission-receipt.json \
  --submission-receipt-max-bytes RECEIPT_CEILING \
  --sacct /absolute/path/to/sacct \
  --expected-commit-sha COMMIT --expected-tree-sha TREE \
  --expected-source-manifest-sha256 SOURCE_SHA256 \
  --expected-checkpoint-ceiling-bytes C \
  --max-input-bytes INPUT_CEILING --max-output-bytes OUTPUT_CEILING \
  --output /external/gate/h100-smoke.json
```

Assemble the final smoke attestation only after all twelve case directories,
runtime evidence, completion receipts, checkpoint hashes/sizes, and (for the
six utilization cases) raw-window summaries validate as one exact set. Each
runtime/action/result chain must match the receipt's Slurm job ID, cluster, and
requested-resource digest; a successful case from another job cannot be
substituted. Before reading the spool, assembly executes the absolute
receipt-digest-bound `sacct` through a stable no-follow descriptor. Every one
of the twelve allocation rows (never a step row) must be unique and exactly
`COMPLETED` with both `ExitCode=0:0` and `DerivedExitCode=0:0`; the canonical
queries, stdout/stderr digests, job/cluster bindings, and command digest are
embedded in the final attestation together with the submission-receipt SHA.
Assembly also requires exactly one `.out` and `.err` scheduler log
for each case in the sibling `scheduler-logs/` directory. Every log must be a
physical regular file within the receipt's per-file ceiling; its exact size and
SHA-256 digest are embedded in the final smoke attestation. Fatal signatures
are rejected, and all 24 files are hashed a second time immediately before
publication so a post-query append fails closed. The final output must stay on
the gate artifact filesystem and fit the additional attestation slot charged
in `F_physical`.

## 3. Build and validate the fresh formal overlay

The overlay fixes three arms by three stages, all as held one-H100 jobs with
budgets 500,000/40,000/10,000. It binds the exact candidate, environment,
protocol digests, prior, architecture, optimizer, scientific configuration,
the twelve-case smoke attestation, `C`, and durable-output ceilings.

The H100 matrix's fixed `03:00:00` limit is only a smoke-gate contract; it is
not a production-training walltime. Production submission remains blocked
until the formal overlay precommits an exact time limit for every stage and
the submit receipt, exported request, compute wrapper, and observed `scontrol`
allocation all bind the same values. Stage 2/3 limits must be derived from the
six maximum-sequence smokes, while Stage 1 needs measured full-prior throughput
and an explicitly approved bound. A `long` QOS selection alone is not a
reproducible walltime contract.

Before the first scheduler submission, the controller must validate the smoke
attestation and enforce the byte-exact capacity rule documented in
`FORMAL_STATUS.md`. Insufficient space is a hard block, not a warning.

The physical production `job_work_root` must likewise use a different
filesystem from the durable artifact namespace. The controller checks this
before namespace creation and rechecks both device identity and capacity while
all nine allocations remain held. The exact-root bootstrap binding is exported
to every job, whose compute wrapper invokes the same exact-T no-follow helper
before creating a temporary checkout.

Submission is one transaction: submit all nine jobs held, atomically publish
the protocol ledger and held-job receipt, then release them. Stage 2 and Stage
3 use same-arm `afterok` dependencies with invalid-dependency termination. If
any of the nine submissions or any release fails, cancel only the jobs accepted
by that transaction in reverse order. If cancellation is incomplete, publish a
truthful recovery record.

The production overlay must also precommit an absolute `sacct` executable and
its SHA-256 digest. The successful receipt records that path/digest, the
65,536-byte ceiling for each of the nine runtime completion envelopes, the
131,072-byte terminal-log attestation ceiling, and the per-spool log ceiling.
It also records the overlay's finalized-manifest ceiling; each runtime and the
terminal finalizer must match that receipt value, and finalized manifests are
read with this narrower ceiling rather than the general protocol-metadata
allowance. All ten evidence ceilings are charged to the protocol-metadata
allowance.

## 4. Monitor without mutating training

`scripts/monitor_formal_identity.py` is the standalone read-only monitor. Call
it with absolute manifest, state-directory, and event-ledger-directory paths.
It defaults to a bounded 30-minute run with 30-second polling, supports
`--once`, and uses a nonblocking per-state-directory lock. Healthy polls are
silent; only anomalies, scheduler/stage transitions, and checkpoint issues are
emitted and appended to the bounded event ledger.

The monitor warns below 22 GiB available and emits a submit-block condition
below 20 GiB. It queries only scheduler/accounting status, reads bounded log
tails, and rechecks checkpoint ZIP/hash integrity only when checkpoint path
identity or modification metadata changes. It never invokes a submit, release,
or cancellation command.

Treat the manifest's normalized absolute `disk_path` as the fixed artifact
namespace. Every configured log, checkpoint path/directory, GPU sample,
validation report, formal artifact root, and transaction ledger must be
lexically below that namespace and its nearest existing physical ancestor must
have the same `st_dev` as `disk_path`. Existing components are traversed with
no-follow descriptors, so a symlink or nested different-device mount fails
closed. A pending path may be absent, but only below a same-device physical
ancestor in that fixed namespace. Place both writable monitor directories on a
filesystem different from this proven artifact device; startup validates the
complete artifact/disk/writable topology before acquiring the monitor lock or
writing state.

After all nine allocations have left the queue, finalize the Slurm-owned logs
from outside every batch allocation. This is a separate write-once operation;
an in-job completion record is deliberately marked non-terminal and cannot be
used as a substitute:

```text
python -I -B scripts/finalize_formal_scheduler_logs.py \
  --submission-receipt /external/formal/submission-receipt.json \
  --transaction-ledger /external/formal/transaction-ledger.json \
  --artifact-root /external/formal \
  --max-metadata-bytes PROTOCOL_METADATA_ALLOWANCE
```

The finalizer executes the receipt-bound `sacct` bytes through their already
opened descriptor, requires one exact allocation row per receipt job with
`COMPLETED` plus `ExitCode=0:0` and `DerivedExitCode=0:0`, validates all nine
write-once runtime completions, and
then reads each stdout/stderr spool twice without following symlinks. Both
reads must have identical inode metadata, size, and SHA-256. OOM, non-finite,
storage-exhaustion, or traceback signatures reject finalization. Only then is
`terminal-scheduler-logs.json` published and bound to the submission receipt
SHA-256.

## 5. Stop conditions

Do not start the nine formal jobs if any exact-source/import check, one of the
twelve hardware cases, checkpoint ceiling, nonfinite/OOM scan, runtime
attestation, capacity gate, or atomic-publication test fails. Preserve the
failed evidence outside the public repository, fix the candidate, and rerun the
entire hardware matrix because a code change creates a new exact commit.

Do not treat a completed training cohort as accepted until the independent
terminal scheduler-log attestation has also published successfully.

Evaluation code and datasets remain a separate follow-on workstream and must
not be added to this training candidate.

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
represent the same Python binary/cache tag/SOABI, Torch/CUDA/cuDNN/NCCL
environment, effective formal dependency builds, and a path-free digest of
the complete visible distribution inventory;
the final assembler requires their runtime envelopes to differ only in
`visible_cuda_device_count` (one versus two). Every case checks its applicable
digest before creating its work tree and captures the same strict manifest
again immediately after compute. Generate the pair only with
`scripts/generate_formal_environment.py` from clean detached exact T under
isolated Python. Its private, write-once inventory artifact records the full
path-free digest preimage and binds both public environment digests to the
exact source commit/tree. The one-GPU manifest, two-GPU manifest, and private
inventory are staged in one physical directory and become an authoritative set
only when the generator publishes their write-once completion marker last. An
interrupted marker-less set is incomplete and may only be recovered by rerunning
the same digest-bound transaction; all four outputs must be outside exact T.
Before consuming the set, run
`scripts/verify_formal_environment_transaction.py` with externally held
completion, transaction, one-/two-GPU environment, inventory, source, and Git
digests. The verifier opens the marker and all three named siblings with
bounded no-follow reads, reconstructs the transaction descriptor, validates
all manifest self-hashes and the private fingerprint preimage, and emits a
path-free canonical verification summary. Final files without their marker are
never accepted; once a marker exists, a missing or changed sibling is
corruption rather than recoverable partial publication.
The H100 submit overlay carries that marker path and all externally held
digests and ceilings. Before creating the gate namespace or contacting Slurm,
the exact-T controller runs the verifier and embeds its path-free summary in
both the held plan and the released receipt. Final assembly validates that
summary against the receipt's source, Git, and one-/two-GPU environment
bindings.

Each effective formal dependency binds every actual regular file that has a
hash in the installed wheel `RECORD`, not only the imported entry point. Reads
are bounded and use no-follow component traversal; missing files, symlinks,
installation-prefix escapes, and non-bytecode hash mismatches are rejected.
Relocatable Python bytecode may differ from its wheel declaration, so both its
declared and actual size/hash plus the mismatch bit are committed. The verifier
recomputes that exact actual-byte commitment, making any later bytecode drift a
hard failure. It also rejects a preloaded `tabicl` namespace and proves
`_provenance.__file__` is the exact-T source file. The controller derives a unique artifact
identity from the validation ID, case, exact commit/tree/source manifest,
applicable environment digest, and `C`; callers do not supply arbitrary case
identities.

The final smoke attestation derives one exact H100 model and NVIDIA driver
version from all twelve allocations. Formal production exports those values
from the validated smoke and rejects any allocation whose model or driver is
different before creating a work tree. Production repeats the strict software
environment verification after each stage before publishing successful
completion evidence.

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
Each compute wrapper also replaces inherited account and desktop paths with a
private mode-0700 `HOME` beneath that node-local temporary root, together with
matching XDG cache/config/data roots and fixed non-personal `USER`/`LOGNAME`
values. This is required on compute images where the Slurm UID has no passwd
entry and prevents libraries from falling back to shared or user-specific
locations. The complete runtime home is removed with the temporary checkout.

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
short scheduler-propagation lag does not create a false recovery. SIGHUP,
SIGINT, SIGTERM, or Python interruption follows the same rollback path. A failed
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
  "environment_transaction": {
    "completion_max_bytes": 1048576,
    "completion_path": "/external/environment/environment-complete.json",
    "completion_sha256": "COMPLETION_FILE_SHA256",
    "inventory_max_bytes": 134217728,
    "inventory_sha256": "INVENTORY_MANIFEST_SHA256",
    "manifest_max_bytes": 1048576,
    "transaction_sha256": "ENVIRONMENT_TRANSACTION_SHA256"
  },
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
    "nvidia_smi_sha256": "NVIDIA_SMI_SHA256",
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

First publish one immutable `formal_campaign` manifest. It binds exact T, the
validated H100 attestation and its exact GPU model/driver, `C`, all three stage
time limits, and all stage-static protocol digests. Each seed overlay carries
the campaign path, external campaign digest, acceptance-registry path, and the
canonical campaign binding returned by `formal_campaign_registry.py`
`authorize-seed`. Seed 42 requires an empty acceptance registry; seeds 43 and 44
are authorized only when that directory is the exact write-once predecessor
acceptance prefix. The campaign files and all other external evidence must be
outside exact T, the fresh artifact namespace, and the node-local work root.

Run the publisher from the clean detached exact-T checkout. The bounded
canonical spec is not a bare payload object. It is a newline-terminated,
self-hashed manifest with `schema_version: 1`, kind `formal_campaign_spec`, the
fields below inside `payload`, and `sha256` equal to SHA-256 of the canonical
JSON bytes of `{schema_version, kind, payload}`. Duplicate keys, a noncanonical
encoding, a missing newline, or a digest mismatch are rejected. Its payload
names that exact root plus the external source manifest, H100
attestation, H100 held-submission receipt, digest-bound Git executable,
environment completion-marker path, its fixed completion/manifest/inventory
read ceilings, the other evidence ceilings, checkpoint ceiling, expected GPU
model, training source, and all three frozen stage protocols. The marker path
is runtime evidence in the external campaign artifact; it is not a private
path literal in the public source tree or the path-free H100 receipt:

The corresponding required spec fields are `environment_completion_path`,
`environment_completion_max_bytes`, `environment_manifest_max_bytes`, and
`environment_inventory_max_bytes`; their ceilings are exactly 1 MiB, 1 MiB,
and 128 MiB respectively.

Create a fresh campaign directory beneath an already-validated physical
parent. Do not use `-p`, reuse an old directory, or add any other entries:

```bash
umask 077
mkdir /external/campaign
mkdir /external/campaign/acceptances /external/campaign/evaluations
```

The publisher requires those two subdirectories to exist and be empty, with no
other entry present before the write-once `campaign.json` publication.

```text
python -I -B /exact/T/scripts/formal_campaign_registry.py publish-campaign \
  --spec /external/formal-campaign-spec.json \
  --spec-max-bytes SPEC_CEILING \
  --output /external/campaign/campaign.json

python -I -B /exact/T/scripts/formal_campaign_registry.py authorize-seed \
  --campaign /external/campaign/campaign.json \
  --campaign-sha256 CAMPAIGN_SHA256 \
  --acceptance-registry /external/campaign/acceptances \
  --seed 42
```

`publish-campaign` independently validates the advertised exact Git ref,
source archive/manifest, full twelve-case H100 attestation, its actual held-job
receipt and environment transaction before write-once publication. A generic
or hand-built draft is not authorizable. Campaign source validation requires
both `scripts` and `src/tabicl` as manifest code roots, so the archive scan
covers transitive shell, Python, and native-code files rather than only a
short validator allowlist. `authorize-seed` reopens those same
external trust anchors on every call, including the raw completion marker,
one-/two-GPU manifests, and private inventory through bounded no-follow reads.
`accept-seed` performs the same raw revalidation before consuming seed
evidence. Either command fails if the marker or any named sibling is missing,
replaced, changed, or exceeds the frozen ceiling. `authorize-seed` returns the
canonical binding that must be embedded unchanged in the formal overlay.
The same exact-T pass also executes a fresh bounded `ls-remote` through the
digest-bound Git executable and requires its complete repository binding to
equal the H100 receipt. Publication, post-publication verification, every
authorization, and every acceptance fail if the fixed public ref is missing,
moved, or produces a different query binding.

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
Production uses the same private node-local runtime-home contract as the H100
matrix; it never relies on a passwd entry or an inherited `HOME`/XDG path.

Submission is one transaction: submit all nine jobs held, atomically publish
the protocol ledger and held-job receipt, then release them. Stage 2 and Stage
3 use same-arm `afterok` dependencies with invalid-dependency termination. If
any of the nine submissions or any release fails, cancel only the jobs accepted
by that transaction in reverse order. If cancellation is incomplete, publish a
truthful recovery record.

The immutable ledger, held receipt, and post-release commit each carry the
same full H100-gate and campaign bindings. Every receipt job also carries the
SHA-256 of its exact `sbatch` argument vector, and the hash-chained transaction
journal records that digest before invocation. Runtime jobs require their
exported gate/campaign digests to match the ledger and receipt before training;
the independent finalizer re-derives the campaign stage protocols and writes
the same binding into terminal evidence consumed by campaign acceptance.
The precommitted protocol-metadata allowance `M` is likewise copied
unchanged into the ledger, held receipt, release commit, every runtime
completion, and the terminal attestation. `M` must be positive and at most
128 MiB. Overlay preflight, runtime, finalization, and campaign acceptance
reject a value outside that range or an exported/caller-supplied value that is
not exactly the bound `M`. This `M` is one charged partition of the separate
aggregate durable-output allowance `L` used by the capacity formula.

All five submission-side Slurm executables are opened without following
symlinks, hashed once, retained, and executed through `/proc/self/fd` with the
same descriptors for submission, reconciliation, release, and rollback. Git
is likewise executed through the already hashed descriptor in both H100 and
production compute wrappers, so replacing a pathname after validation cannot
change the executed bytes.
The one- and two-GPU H100 spool wrappers, and the production spool wrapper,
are also opened without following symlinks, checked against exact-T committed
bytes, retained, and supplied to `sbatch` through `/proc/self/fd`; Slurm never
spools a pathname that can be replaced after source attestation.
`nvidia-smi` is likewise opened with no-follow traversal, bounded and hashed.
Every H100 and production GPU query executes through a descriptor for that
verified inode. When a durable logger or `torchrun` closes inherited file
descriptors, the child reacquires the inode only through the still-live owner's
`/proc/<pid>/fd/<n>` entry, repeats the stable bounded hash check, and never
reopens the mutable pathname. Its SHA-256 is repeated through runtime, ledger,
receipt, H100, and campaign bindings.

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
SHA-256. `PROTOCOL_METADATA_ALLOWANCE` is not a free read limit: it must equal
the `protocol_metadata_allowance_bytes` value frozen in the ledger, receipt,
release commit, and all nine completions.

After the separately versioned minimum evaluation receipt has been published,
accept the seed from the same clean detached exact-T controller used to publish
the campaign:

```text
python -I -B /exact/T/scripts/formal_campaign_registry.py accept-seed \
  --campaign /external/campaign/campaign.json \
  --campaign-sha256 CAMPAIGN_SHA256 \
  --acceptance-registry /external/campaign/acceptances \
  --seed 42 \
  --formal-artifact-root /external/formal \
  --terminal-attestation /external/formal/terminal-scheduler-logs.json \
  --submission-receipt /external/formal/submission-receipt.json \
  --transaction-ledger /external/formal/transaction-ledger.json \
  --terminal-metadata-max-bytes PROTOCOL_METADATA_ALLOWANCE \
  --evaluation-receipt /external/campaign/evaluations/seed-42.json
```

These terminal paths are mandatory and must form that one canonical artifact
namespace. Acceptance does not trust the terminal JSON by itself: it reads the
existing terminal before and after independently rebuilding it from the held
receipt, ledger, transaction commit, nine completions, finalized checkpoints,
stable scheduler logs, and a fresh receipt-bound `sacct` query. It publishes no
replacement terminal file. Every later `authorize-seed` or `accept-seed` call
repeats the same raw-evidence validation for all predecessor acceptances, so
replacing a checkpoint, log, receipt, ledger, terminal file, or accounting
executable after acceptance fails closed. The CLI metadata value must again be
the exact receipt-bound `M`; a merely larger caller-selected ceiling is
rejected even when it remains below the registry's defensive fixed maximum.

## 5. Stop conditions

Do not start the nine formal jobs if any exact-source/import check, one of the
twelve hardware cases, checkpoint ceiling, nonfinite/OOM scan, runtime
attestation, capacity gate, or atomic-publication test fails. Preserve the
failed evidence outside the public repository, fix the candidate, and rerun the
entire hardware matrix because a code change creates a new exact commit.

Do not treat a completed training cohort as accepted until the independent
terminal scheduler-log attestation has published successfully, its raw formal
evidence has been re-derived by `accept-seed`, and the matched minimum
evaluation receipt has passed.

Evaluation code and datasets remain a separate follow-on workstream and must
not be added to this training candidate.

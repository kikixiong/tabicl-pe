# Repository Guidance

These instructions apply to the whole repository.

The formal position-versus-identity study is a matched three-arm pre-training
experiment: `rope`, `temporary`, and `none`. Keep the canonical stage budgets
at 500,000, 40,000, and 10,000 steps. The only treatment difference permitted
within a stage cohort is `row_identity_mode` and its derived treatment digest.
The supported experiment seeds are exactly 42, 43, and 44. Each seed owns a
new fresh namespace, while all three seeds use the same immutable training
commit and frozen protocol.

Formal execution is fresh-only and must use a clean detached checkout of the
exact candidate commit advertised by `https://github.com/kikixiong/tabicl-pe.git`
at `refs/heads/codex/position-identity-v1`. Do not weaken the bounded,
digest-bound exact-ref query, source/import attestation, checkpoint and
parent validation, byte-exact capacity checks, atomic publication, retention,
or the held nine-job submission transaction. Submit the three Stage 1 jobs
first, followed by the three Stage 2 and three Stage 3 jobs with same-arm
`afterok` dependencies. Every production job is one H100 on the `long` QoS
with 64 CPUs and 128 GiB of memory.

Formal seeds are campaign-ordered: the write-once campaign manifest binds T,
the twelve-case H100 gate, exact GPU model/driver, checkpoint ceiling,
walltimes, and static protocols. Ledger, receipt, commit, runtime, and terminal
evidence must preserve that same campaign/H100 binding. Seed 43 requires the
accepted seed-42 record; seed 44 requires the exact accepted 42/43 prefix.
Execute digest-bound Git and Slurm tools through their already verified open
descriptors, never by re-resolving a mutable pathname.
Execute `nvidia-smi` queries through a no-follow, digest-bound descriptor for
the same verified inode in H100 and production jobs. Across `close_fds`
boundaries, reacquire that inode only through the live owner's `/proc` descriptor
and repeat stable bounded hashing before any query; never reopen the pathname.

Node-local checkout work roots must be physical directories on a different
filesystem from durable artifact namespaces. Controllers check this before
creating a namespace and again while jobs are held; compute wrappers repeat it
before `mktemp` or `git clone`.

Never hand-author or mutate formal environment digests. Generate the paired
one-/two-GPU manifests and private inventory only with
`scripts/generate_formal_environment.py` from clean detached exact T under the
project interpreter with `-I -B` and `PYTHONNOUSERSITE=1`. Treat the three
artifacts as valid only with their matching marker-last transaction completion
artifact and a successful externally digest-bound
`scripts/verify_formal_environment_transaction.py` summary; marker-less partial
output is recovery state, not evidence. H100 cases must pass
the preflight and post-compute environment checks. Production must match the
exact GPU model and driver derived from the successful twelve-case gate and
must repeat environment verification after every stage.

Do not call a production cohort complete from in-job receipts alone. Require
the independent receipt-bound accounting check and write-once terminal
scheduler-log attestation after all nine allocations are successfully terminal.

Evaluation and benchmark implementation are intentionally outside the formal
training candidate. Do not add evaluation code to a training-protocol change.

Never commit cluster-specific paths, scheduler job identifiers, logs,
checkpoints, raw predictions, telemetry exports, credentials, or personal
information. Keep generated caches and test artifacts outside the source tree.

Read `docs/FORMAL_STATUS.md` before changing the formal protocol and
`docs/FORMAL_HANDOFF.md` before attempting hardware validation or submission.
